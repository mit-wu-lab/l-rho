import os
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_mean

from flexible_jss_util import set_seed, save_model, load_model, load_dataset, RunningNormalization
from flexible_jss_data_common import FlexibleJSSDataset, get_dataloader
from flexible_jss_test_rollout_configs import get_default_configs, get_simple_configs

from flexible_jss_main import rolling_horizon, flexible_jss
from flexible_jss_main_machine_breakdown import rolling_horizon as rolling_horizon_machine_breakdown
from flexible_jss_main_machine_breakdown import generate_machine_breakdown
     

############################ HELPER FUNCTIONS ############################
def get_params_and_fn(machine_breakdown=False):
    if machine_breakdown:
        INPUT_TASK_DIM = 15 + 3   # is_breakdown, breakdown_operation, recover_operation
        INPUT_MACHINE_DIM = 11 + 2   # is_breakdown, breakdown_machine
    else:
        INPUT_TASK_DIM = 15
        INPUT_MACHINE_DIM = 11

    TASK_FEAT_IDX = 8  # previously assigned machine index
    IN_PREV_IDX = 7  # is it an overlapping operation
    OUTPUT_DIM = 1  

    return INPUT_TASK_DIM, INPUT_MACHINE_DIM, TASK_FEAT_IDX, IN_PREV_IDX, \
            OUTPUT_DIM, FlexibleJSSDataset, get_dataloader


def tasks_select_assignment(data):
    task_m = data.x_tasks[:, TASK_FEAT_IDX].clone()
    num_machines_per_graph = torch.bincount(data.x_machines_batch)
    cumulative_machines = torch.cumsum(num_machines_per_graph, dim=0) - num_machines_per_graph
    task_m += cumulative_machines[data.x_tasks_batch]
    task_m[data.x_tasks[:, TASK_FEAT_IDX]==-1] = data.x_machines.shape[0]
    return task_m.long()
    

def get_separate_tasks(data):
    in_prev = data.x_tasks[:, IN_PREV_IDX]
    data.prev_tasks = in_prev == 1
   
    if  data.x_tasks.shape[1] > 15: # machine breakdown
        BREAKDOWN_IDX, RECOVER_IDX = 16, 17
        #  or recovered tasks or breakdown tasks
        is_breakdown = data.x_tasks[:, BREAKDOWN_IDX] == 1
        is_recovered = data.x_tasks[:, RECOVER_IDX] == 1
        data.new_tasks = ~data.prev_tasks | is_breakdown | is_recovered
    else:
        data.new_tasks = ~data.prev_tasks

    # print(data.v_label_idx, data.prev_vehicles)
    prev_tasks_idx = data.prev_tasks.nonzero().reshape(-1)
    prev_tasks_dict = {t_label.item(): i for i, t_label in enumerate(prev_tasks_idx)}
    data.prev_tasks_label_idx = torch.tensor([prev_tasks_dict[t_label.item()] 
                                              for t_label in data.task_label_idx 
                                              if data.prev_tasks[t_label]]).to(in_prev.device)


############################ Joint network ############################
class FlexibleJSSNet(nn.Module):
    overlap_machine_task_edge_dim = 2
    other_machine_task_edge_dim = 2
    task_precedence_edge_dim = 1
    task_solution_edge_dim = 1
    
    def __init__(self, hidden_dim=64, output_dim=1, 
                 x_tasks_norm=None, x_machines_norm=None):  
        super(FlexibleJSSNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.input_task_dim = INPUT_TASK_DIM
        self.input_machine_dim = INPUT_MACHINE_DIM

        # normalize
        self.x_tasks_norm = x_tasks_norm
        self.x_machines_norm = x_machines_norm

        ### operations nodes         
        self.tasks_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_task_dim),   # torch.nn.BatchNorm1d(self.input_task_dim), 
            torch.nn.Linear(self.input_task_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        ### machines nodes
        self.machines_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_machine_dim),   # torch.nn.BatchNorm1d(self.input_machine_dim),
            torch.nn.Linear(self.input_machine_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        ### concate and aggregate
        self.aggr_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.local_global_aggr_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def normalize_data(self, data):
        if self.x_tasks_norm is not None:
            data.x_tasks = self.x_tasks_norm.normalize(data.x_tasks)
        if self.x_machines_norm is not None:
            data.x_machines = self.x_machines_norm.normalize(data.x_machines)
        return data
    

    def forward(self, data):
        device = data.x_tasks.device
        get_separate_tasks(data)
        task_m = tasks_select_assignment(data)
        data = self.normalize_data(data)

        # initial embedding
        task_hidden_embed = self.tasks_embedding(data.x_tasks)   # (#task * hidden)
        machine_hidden_embed = self.machines_embedding(data.x_machines)   # (#machine * hidden)

        ########################### local / global info aggregation ###########################
        aggr_hidden_cat = torch.cat([scatter_mean(task_hidden_embed, data.x_tasks_batch, dim=0),
                                     scatter_mean(machine_hidden_embed, data.x_machines_batch, dim=0)], dim=1)
        aggr_embed = self.aggr_layers(aggr_hidden_cat)   # hidden * 1
        aggr_embed_expanded = aggr_embed[data.x_tasks_batch]  
        machine_hidden_cat = torch.cat([machine_hidden_embed, torch.zeros(1, self.hidden_dim).to(device)], dim=0)
        t_m_embed = machine_hidden_cat.index_select(0, task_m)
        task_hidden_embed = torch.cat([task_hidden_embed, aggr_embed_expanded, t_m_embed], dim=1)
        task_hidden_embed = self.local_global_aggr_layers(task_hidden_embed)

        out = self.output_layers(task_hidden_embed)[data.task_label_idx]  # (#veh * hidden)
            
        return out


############################ Train / Test Models ############################
def train_model(model, optimizer, train_loader, val_loader, num_epochs, writer, scheduler=None,
                eval_every=20, save_every=20, 
                model_dir='train_test_dir/model', model_name='model', pos_weight=1):
    model.train()
    num_grad_steps = 0
    for epoch in range(num_epochs):
        if epoch % eval_every == 0:
            print('================== Eval train ==================')
            evaluate_model(model, train_loader, writer, epoch, eval_dir=f'train_accuracy')
            print('================== Eval val ==================')
            evaluate_model(model, val_loader, writer, epoch, eval_dir=f'eval_accuracy')

        if epoch % save_every == 0:
            save_model(model, model_dir, model_name, epoch)

        running_loss = 0.0

        for data_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = F.binary_cross_entropy_with_logits(output, data.task_label.reshape(-1, 1), 
                                                        pos_weight=torch.tensor([pos_weight]).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if scheduler is not None:
                scheduler.step()

            if writer is not None:
                writer.add_scalar('train_loss/step', loss, num_grad_steps)
                if scheduler is not None:
                    writer.add_scalar('learning_rate/step', scheduler.get_last_lr()[0], num_grad_steps)
                num_grad_steps += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        if writer is not None:
            writer.add_scalar('train_loss/epoch', epoch_loss, epoch + 1)

    print('Training finished.')


def evaluate_model(model, data_loader, writer, epoch, conf_mat=True, 
                   eval_dir='eval_accuracy'):
    model.eval()
    # compute the TP TN FP FN at each index
    # different thresholds
    eval_str = eval_dir.split("_")[0]

    threshold_list = [0.5, 0.6]
    correct_predictions_list = [0 for _ in range(len(threshold_list))]
    total_samples = 0
    gt_num_ones = 0
    running_loss = 0.0
    if conf_mat:
        TP_list, TN_list = [0 for _ in range(len(threshold_list))], [0 for _ in range(len(threshold_list))]
        FP_list, FN_list = [0 for _ in range(len(threshold_list))], [0 for _ in range(len(threshold_list))]

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            ground_truth = data.task_label.cpu().reshape(-1).numpy()
            gt_num_ones += ground_truth.sum()
            total_samples += len(ground_truth)
            running_loss += F.binary_cross_entropy_with_logits(output, data.task_label.reshape(-1, 1), 
                                                                pos_weight=torch.tensor([pos_weight]).to(device))

            for i_th, threshold in enumerate(threshold_list):
                predictions = (F.sigmoid(output) >= threshold).cpu().reshape(-1).numpy()
                correct_predictions_list[i_th] += (predictions == ground_truth).sum()
            
                # Update confusion matrix counts
                if conf_mat:
                    TP_list[i_th] += ((predictions == 1) & (ground_truth == 1)).sum()
                    TN_list[i_th] += ((predictions == 0) & (ground_truth == 0)).sum()
                    FP_list[i_th] += ((predictions == 1) & (ground_truth == 0)).sum()
                    FN_list[i_th] += ((predictions == 0) & (ground_truth == 1)).sum()
    
    running_loss = running_loss / len(data_loader)
    if 'eval' in eval_str and writer is not None:
        writer.add_scalar(f'{eval_str}_loss/epoch', running_loss, epoch + 1)

    for i_th, threshold in enumerate(threshold_list):
        print(f"***************** Threshold: {threshold:.2f} *****************")
        correct_predictions = correct_predictions_list[i_th]
        accuracy = correct_predictions / total_samples
        TP, TN, FP, FN = TP_list[i_th], TN_list[i_th], FP_list[i_th], FN_list[i_th]
        print(f"Accuracy: {accuracy:.4f} | # Correct {correct_predictions}, # Total {total_samples}, "
                f"Ground truth: Number of selected {gt_num_ones}, frac {gt_num_ones / total_samples:.2f}")
    
        # Print Confusion Matrix
        if conf_mat:
            TP_rate = TP * 1.0 / (TP + FN)
            TN_rate = TN * 1.0 / (TN + FP)
            FP_rate = FP * 1.0 / (FP + TN)
            FN_rate = FN * 1.0 / (TP + FN)
            print(f"Confusion Matrix:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} "
                    f"(TP rate = {TP_rate:.2f} TN rate = {TN_rate:.2f} " 
                    f"FP rate = {FP_rate:.2f} FN rate = {FN_rate:.2f})")

        if writer is not None:
            writer.add_scalar(f'{eval_str}_accuracy_epoch/th{threshold}', accuracy, epoch + 1)
            writer.add_scalar(f'conf_mat/{eval_str}_TP_rate_epoch/th{threshold}', TP_rate, epoch + 1)
            writer.add_scalar(f'conf_mat/{eval_str}_TN_rate_epoch/th{threshold}', TN_rate, epoch + 1)
            writer.add_scalar(f'conf_mat/{eval_str}_FP_rate_epoch/th{threshold}', FP_rate, epoch + 1)
            writer.add_scalar(f'conf_mat/{eval_str}_FN_rate_epoch/th{threshold}', FN_rate, epoch + 1)



'''test rollout'''
def rollout_model(model, index_start, index_end, jss_data_dir, model_th=0.5, 
                  test_stats_dir='train_test_dir/test_stats', 
                  model_only=False,   # only rollout model
                  model_suffix="",
                  full_only=False,    # only run full fjsp
                  default_rho_only=False,    # only run default rho
                  do_simple_test=False,   # only run default rho and model
                  machine_breakdown=False, 
                  breakdown_data_dir=''):
    model.eval()

    oracle_params = (window, step, time_limit, stop_search_time)
        
    # load label + param config from file
    if model_only:
        configurations = [(f'model_only {model_suffix}', {'action_type': 'model', 'model': model, 'model_th': model_th}),
                          (f'model_only {model_suffix} (include model time)', {'time_limit': time_limit, 'action_type': 'model', 'model': model, 
                            'model_th': model_th, 'include_model_time': True, 'stop_search_time': stop_search_time}),]
    elif full_only:
        # full FJSP (no decomposition)
        configurations = [(f'full {time_limit}', {'action_type': 'full', 'time_limit': time_limit})]
    elif default_rho_only:
        configurations = [(f'default [{oracle_params}]', 
                           {'action_type': 'default', 'window': oracle_params[0], 
                            'step': oracle_params[1], 'time_limit': oracle_params[2], 
                            'stop_search_time': oracle_params[3]})]
    elif do_simple_test:
        # default and model: run for longer without stop_search_time
        configurations = get_simple_configs(oracle_params, model, model_th)
    else:
        print(f'Using param from args ({window}, {step}, {time_limit}, {stop_search_time})')
        # Define the configurations for each run
        configurations = get_default_configs(model, model_th, time_limit, stop_search_time)

        # full FJSP
        for full_t in [200, 500, 1800]:
            configurations += [(f'full {full_t}', {'action_type': 'full', 'time_limit': full_t})]

    # Run the dispatch for each configuration
    for vi in range(index_start, index_end):
        # load jss data
        data_loc = f'{jss_data_dir}/data_{vi}.pkl'
        assert os.path.exists(data_loc), "Data file does not exist!"

        print(f'Load data from {data_loc}')
        jobs_data, n_machines, n_jobs = pickle.load(open(data_loc, 'rb'))
        
        stats_file_name = os.path.join(test_stats_dir, f'stats_e{load_model_epoch}th{model_th}_{vi}.pkl')
        print(f'Stats (objective and time) save to {stats_file_name}') 

        if os.path.exists(stats_file_name):
            makespan_stats, solve_time_stats, average_solve_time_stats = pickle.load(open(stats_file_name, 'rb'))
        else:
            makespan_stats, solve_time_stats, average_solve_time_stats = {}, {}, {}

        if machine_breakdown:
            breakdown_data_loc = f'{breakdown_data_dir}/breakdown_data_{vi}.pkl'
            if not os.path.exists(breakdown_data_loc):
                print(f'Generate machine breakdown data to {breakdown_data_loc}')
                breakdown_times = generate_machine_breakdown(jobs_data, n_machines, 
                                                            num_machine_breakdown_p=num_machine_breakdown_p,
                                                            first_breakdown_buffer_lb=first_breakdown_buffer_lb,
                                                            first_breakdown_buffer_ub=first_breakdown_buffer_ub,
                                                            machine_breakdown_duration_lb=machine_breakdown_duration_lb,
                                                            machine_breakdown_duration_ub=machine_breakdown_duration_ub,
                                                            breakdown_buffer_lb=breakdown_buffer_lb,
                                                            breakdown_buffer_ub=breakdown_buffer_ub)
                pickle.dump(breakdown_times, open(breakdown_data_loc, 'wb'))

            breakdown_times = pickle.load(open(breakdown_data_loc, 'rb'))

        sel_configs = [(label, params) for label, params in configurations if label not in makespan_stats]  
        
        print(f'{len(sel_configs)} configurations to run: {[label for label, _ in sel_configs]} ...')
        for label, params_dict in sel_configs:
            print(f'*************** {label.upper()} ****************')

            if 'full' in label.lower():
                if "time_limit" in params_dict:
                    full_time_limit = params_dict["time_limit"]
                else:
                    full_time_limit = time_limit
                print(f'Solve full problem with time limit {full_time_limit}!')
                jobs_solution, machines_assignment, solve_time, make_span = flexible_jss(
                    jobs_data, n_machines, time_limit=full_time_limit, stop_search_time=full_time_limit,  # do not stop search earlier
                    machines_start_time=None, jobs_start_time=None)
                avg_solve_time = solve_time

            else:
                config_labels = ["window", "step", "time_limit", "stop_search_time"]
                config_params = [window, step, time_limit, stop_search_time]
                for config_i, config_label in enumerate(config_labels):
                    if config_label in params_dict:
                        config_params[config_i] = params_dict[config_label]
                        del params_dict[config_label]
                        
                config_window, config_step, config_time_limit, config_stop_search_time = config_params
                in_params = [n_machines, n_jobs, jobs_data, config_window, config_step, config_time_limit, config_stop_search_time]

                if machine_breakdown:
                    if 'breakdown_times' not in params_dict:
                        params_dict['breakdown_times'] = breakdown_times

                    _, _, solve_time, avg_solve_time, make_span = rolling_horizon_machine_breakdown(*in_params, **params_dict)
                else:
                    _, _, solve_time, avg_solve_time, make_span = rolling_horizon(*in_params, **params_dict)
            
            makespan_stats[label] = make_span
            solve_time_stats[label] = solve_time
            average_solve_time_stats[label] = avg_solve_time

            pickle.dump([makespan_stats, solve_time_stats, average_solve_time_stats], 
                        open(stats_file_name, 'wb'))

        print('\n==================stats==================')
        for k in makespan_stats:
            print(f'{k} makespan: {np.mean(makespan_stats[k]):.2f} +- {np.std(makespan_stats[k]):.2f} | ' 
                  f'solve time: {np.mean(solve_time_stats[k]):.2f} +- {np.std(solve_time_stats[k]):.2f} '
                  f'(average solve time: {np.mean(average_solve_time_stats[k]):.2f} +- {np.std(average_solve_time_stats[k]):.2f})')
        print('=========================================\n')


if __name__ == '__main__':
    # JSS parameters
    parser = argparse.ArgumentParser("Learning-guided RHO for FJSP")

    parser.add_argument("--jss_data_dir", default="train_data_dir/instance", type=str, help="save instance directory")  
    parser.add_argument("--train_data_dir", default="train_data_dir/train_data", type=str, help="save train data directory")  
    parser.add_argument('--log_dir', type=str, default='train_test_dir/train_log', help="tensorboard log dir")
    parser.add_argument('--test_stats_dir', type=str, default='train_test_dir/test_stats', help="directory to save test stats")
    parser.add_argument('--model_dir', type=str, default='train_test_dir/model', help="saved model location")

    parser.add_argument("--data_index_start", default=1, type=int, help="the start index of the collected data (train or test)")
    parser.add_argument("--data_index_end", default=2, type=int, help="the end index of the collected data (train or test)")
    parser.add_argument("--val_index_start", default=90, type=int, help="the start index of the collected data (val))")
    parser.add_argument("--val_index_end", default=100, type=int, help="the end index of the collected data (val)))")
    # save and load
    parser.add_argument('--model_name', type=str, default='model', help="saved model name")
    parser.add_argument("--load_model_epoch", default=120, type=int, help="the epoch that we load the model from")
    # learning
    parser.add_argument("--eval_every", default=20, type=int, help="frequency of validation")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lr_delay", default=1, type=float, help="learning rate decay")
    parser.add_argument("--train_steps", default=100000, type=int, help="number of training steps")  
    parser.add_argument("--num_epochs", default=150, type=int, help="number of training epochs")

    # architecture
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden dimension of the model")
    parser.add_argument("--pos_weight", default=1, type=float, help="weight of positive label for BCE loss")

    # test options
    parser.add_argument('--test', action="store_true", help="test or train?")
    parser.add_argument('--model_only', action="store_true", help="only test model?")
    parser.add_argument('--model_suffix', default='', type=str, help="model suffix")
    parser.add_argument('--full_only', action="store_true", help="only test full?")
    parser.add_argument('--default_rho_only', action="store_true", help="only test default and model?")
    parser.add_argument('--do_simple_test', action="store_true", help="only test default and model?")
    parser.add_argument('--model_th', default=0.5, type=float, help="model threshold")

    # roll out parameters
    parser.add_argument("--time_limit", default=30, type=int, help="cp-sat solver time limit") 
    parser.add_argument("--stop_search_time", default=5, type=int, help="cp-sat solver stop search wait time") 
    parser.add_argument("--window", default=80, type=int) 
    parser.add_argument("--step", default=30, type=int) 

    parser.add_argument("--oracle_time_limit", default=30, type=int, help="time limit to find ground truth label")
    parser.add_argument("--oracle_stop_search_time", default=5, type=int, help="time limit to stop search for finding ground truth label")
    parser.add_argument("--n_cpus", default=1, type=int, help="cpus to find best") 
    parser.add_argument("--num_oracle_trials", default=3, type=int, help="number of oracle trials to find the most overlap")
    
    ### machine breakdown parameters
    parser.add_argument("--machine_breakdown", action='store_true', help="machine breakdown")
    parser.add_argument("--breakdown_data_dir", default="train_data_dir/breakdown_data/j20-m10-t30_mix", type=str, help="breakdown data directory")
    parser.add_argument("--num_machine_breakdown_p", default=0.2, type=float, help="number of machines breakdown probability")
    parser.add_argument("--first_breakdown_buffer_lb", default=50, type=int, help="first breakdown buffer lower bound")
    parser.add_argument("--first_breakdown_buffer_ub", default=150, type=int, help="first breakdown buffer upper bound")
    parser.add_argument("--machine_breakdown_duration_lb", default=100, type=int, help="machine breakdown duration lower bound")
    parser.add_argument("--machine_breakdown_duration_ub", default=100, type=int, help="machine breakdown duration upper bound")
    parser.add_argument("--breakdown_buffer_lb", default=400, type=int, help="breakdown buffer lower bound")
    parser.add_argument("--breakdown_buffer_ub", default=600, type=int, help="breakdown buffer upper bound")
    args = parser.parse_args()

    seed = 123
    set_seed(seed)

    #### global variables
    window = args.window 
    step = args.step  
    time_limit = args.time_limit
    stop_search_time = args.stop_search_time
    oracle_time_limit = args.oracle_time_limit
    oracle_stop_search_time = args.oracle_stop_search_time

    n_cpus = args.n_cpus
    num_oracle_trials = args.num_oracle_trials

    ### machine breakdown
    machine_breakdown = args.machine_breakdown
    breakdown_data_dir = args.breakdown_data_dir
    num_machine_breakdown_p = args.num_machine_breakdown_p
    first_breakdown_buffer_lb = args.first_breakdown_buffer_lb
    first_breakdown_buffer_ub = args.first_breakdown_buffer_ub
    machine_breakdown_duration_lb = args.machine_breakdown_duration_lb
    machine_breakdown_duration_ub = args.machine_breakdown_duration_ub
    breakdown_buffer_lb = args.breakdown_buffer_lb
    breakdown_buffer_ub = args.breakdown_buffer_ub
        

    #### model parameters
    INPUT_TASK_DIM, INPUT_MACHINE_DIM, TASK_FEAT_IDX, IN_PREV_IDX, \
        OUTPUT_DIM, FlexibleJSSDataset, get_dataloader = get_params_and_fn(args.machine_breakdown)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_weight = args.pos_weight
    model_name = args.model_name
    model_dir = os.path.join(args.model_dir, model_name)
    log_dir = os.path.join(args.log_dir, model_name)
    test_stats_dir = os.path.join(args.test_stats_dir, model_name)

    for d in [log_dir, model_dir, test_stats_dir]:
        if not os.path.exists(d):
            time.sleep(0.1 + 10 * np.random.rand())  # sleep for a random number of seconds to avoid conflict between parallel runs
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
    writer = SummaryWriter(log_dir)
    if args.test:
        print(f'Load model from {model_dir}')
    else:
        print(f'Save model to {model_dir}')

    # load data
    if args.data_index_start == -1 and args.data_index_end == -1:
        training_data = load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=-1)
    else:
        training_data = []
        for idx in range(args.data_index_start, args.data_index_end):
            training_data += load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=idx)

    train_dataset = FlexibleJSSDataset(training_data)
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True, 
                                  follow_batch=['x_tasks', 'x_machines'], 
                                  num_workers=0, pin_memory=False)

    ### compute (or load) normalization statistics (mean, std) to normalize input features
    x_tasks_norm = RunningNormalization(device=device)
    x_machines_norm = RunningNormalization(device=device)

    if args.machine_breakdown:
        normalizer_dir = os.path.join(args.model_dir, f'input_normalizer_machine_breakdown')
    else:
        normalizer_dir = os.path.join(args.model_dir, f'input_normalizer')

    if not os.path.exists(normalizer_dir):
        os.makedirs(normalizer_dir, exist_ok=True)

    machine_normalizer_file = os.path.join(normalizer_dir, 'normalizer_machines.pkl')
    task_normalizer_file = os.path.join(normalizer_dir, 'normalizer_tasks.pkl')

    if os.path.exists(machine_normalizer_file) and os.path.exists(task_normalizer_file):
        print(f'Load normalization statistics from {machine_normalizer_file} and {task_normalizer_file}')
        x_tasks_norm.load(task_normalizer_file)
        x_machines_norm.load(machine_normalizer_file)
    else:
        print(f'Compute normalization statistics to {machine_normalizer_file} and {task_normalizer_file}')
        for data_idx, data in enumerate(train_loader):
            x_tasks_norm.update(data.x_tasks)
            x_machines_norm.update(data.x_machines)
        
        x_tasks_norm.save(task_normalizer_file)
        x_machines_norm.save(machine_normalizer_file)
    ###

    ### construct model, optimizer, criterion
    model = FlexibleJSSNet(hidden_dim=args.hidden_dim, 
                           output_dim=OUTPUT_DIM, 
                           x_tasks_norm=x_tasks_norm, 
                           x_machines_norm=x_machines_norm,
                           ).to(device)

    load_model_epoch = args.load_model_epoch
    
    if args.test:
        ### rollout pretrained model
        load_model(model, load_dir=model_dir, load_name=model_name, model_epoch=load_model_epoch) 
        rollout_model(model, args.val_index_start, args.val_index_end, args.jss_data_dir, 
                      model_th=args.model_th, 
                      test_stats_dir=test_stats_dir, 
                      model_only=args.model_only,
                      model_suffix=args.model_suffix,
                      full_only=args.full_only,
                      default_rho_only=args.default_rho_only,
                      do_simple_test=args.do_simple_test,
                      machine_breakdown=args.machine_breakdown,
                      breakdown_data_dir=args.breakdown_data_dir,
                      )
    else:
        ### train and validate
        if args.val_index_start == -1 and args.val_index_end == -1:
            val_data = load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=-1)
        else:
            val_data = []
            for idx in range(args.val_index_start, args.val_index_end):
                val_data += load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=idx)
        
        val_dataset = FlexibleJSSDataset(val_data)
        val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=True, follow_batch=['x_tasks', 'x_machines'], num_workers=0,
                                    pin_memory=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_steps)
        train_model(model, optimizer, train_loader, val_loader, args.num_epochs, writer, scheduler=scheduler, eval_every=args.eval_every,
                    model_dir=model_dir, model_name=model_name, pos_weight=pos_weight)
        


