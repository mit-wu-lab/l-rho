import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from flexible_jss_util import set_seed, save_model, load_model, load_dataset, RunningNormalization

from makespan.flexible_jss_data_common import FlexibleJSSDataset, get_dataloader
from flexible_jss_instance import get_jss_argparser, get_flexible_jss_data_from_args
from flexible_jss_main import rolling_horizon, flexible_jss
from flexible_jss_test_rollout_configs import get_oracle_configs, get_best_configs, get_default_configs, get_simple_configs
import time

############################ HELPER FUNCTIONS ############################
def get_params_and_fn(optim_option='start_delay', perturb_data=False):
    if optim_option == 'start_delay':  # start delay
        INPUT_TASK_DIM = 16
        INPUT_MACHINE_DIM = 6

        TASK_FEAT_IDX = 10
        IN_PREV_IDX = 8
    elif not perturb_data:  # start and delay
        INPUT_TASK_DIM = 19
        INPUT_MACHINE_DIM = 6

        TASK_FEAT_IDX = 11
        IN_PREV_IDX = 9
    else:   # start end delay + observation noise
        INPUT_TASK_DIM = 19 + 3
        INPUT_MACHINE_DIM = 6 + 6

        TASK_FEAT_IDX = 11
        IN_PREV_IDX = 9
    
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
    data.new_tasks = ~data.prev_tasks
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
                 x_tasks_norm=None, x_machines_norm=None, do_local_global_aggr=True, dropout=0.1): 
        super(FlexibleJSSNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.input_task_dim = INPUT_TASK_DIM
        self.input_machine_dim = INPUT_MACHINE_DIM

        # normalize
        self.x_tasks_norm = x_tasks_norm
        self.x_machines_norm = x_machines_norm

        ### operation nodes         
        self.tasks_embedding = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.input_task_dim), # torch.nn.LayerNorm(self.input_task_dim),
            torch.nn.Linear(self.input_task_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )

        ### machines nodes
        self.machines_embedding = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.input_machine_dim),  # torch.nn.LayerNorm(self.input_machine_dim),
            torch.nn.Linear(self.input_machine_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )

        ### concate and aggregate
        self.do_local_global_aggr = do_local_global_aggr
        if do_local_global_aggr:
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
        # print(task_m.max(), task_m.min(), machine_hidden_cat.shape, data.x_machines.shape, data.x_tasks.shape)
        t_m_embed = machine_hidden_cat.index_select(0, task_m)
        task_hidden_embed = torch.cat([task_hidden_embed, aggr_embed_expanded, t_m_embed], dim=1)
        task_hidden_embed = self.local_global_aggr_layers(task_hidden_embed)
        
        out = self.output_layers(task_hidden_embed)[data.prev_tasks_label_idx]
        return out



############################ Train / Test Models ############################
def train_model(model, optimizer, train_loader, val_loader, num_epochs, writer, scheduler=None,
                eval_every=20, save_every=20, model_dir='train_test_dir/model', model_name='model', pos_weight=1):
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
                writer.add_scalar('learning_rate/step', scheduler.get_last_lr()[0], num_grad_steps)
                num_grad_steps += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        if writer is not None:
            writer.add_scalar('train_loss/epoch', epoch_loss, epoch + 1)

    print('Training finished.')


def evaluate_model(model, data_loader, writer, epoch, conf_mat=True, eval_dir='eval_accuracy'):
    model.eval()

    eval_str = eval_dir.split("_")[0]
    threshold_list = [0.5]  # can consider other thresholds to trade-off FP and FN rates
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
    if 'eval' in eval_str:
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

        writer.add_scalar(f'{eval_str}_accuracy_epoch/th{threshold}', accuracy, epoch + 1)
        writer.add_scalar(f'conf_mat/{eval_str}_TP_rate_epoch/th{threshold}', TP_rate, epoch + 1)
        writer.add_scalar(f'conf_mat/{eval_str}_TN_rate_epoch/th{threshold}', TN_rate, epoch + 1)
        writer.add_scalar(f'conf_mat/{eval_str}_FP_rate_epoch/th{threshold}', FP_rate, epoch + 1)
        writer.add_scalar(f'conf_mat/{eval_str}_FN_rate_epoch/th{threshold}', FN_rate, epoch + 1)


'''test rollout'''
def rollout_model(model, index_start, index_end, jss_data_dir, model_th=0.5, optim_option='start_delay',
                  test_stats_dir='train_test_dir/test_stats', 
                  do_simple_test=False):
    model.eval()
    if not os.path.exists(args.best_params_file):
        oracle_params = (window, step, time_limit, stop_search_time)
    else:
        best_params = pickle.load(open(args.best_params_file, 'rb'))
        oracle_params = best_params['oracle']
    
    if do_simple_test:   # default and L-RHO only
        configurations = get_simple_configs(oracle_params, model, model_th)
    elif args.load_best_params and os.path.exists(args.best_params_file):
        print(f'Loading best params from {args.best_params_file}')
        configurations = get_oracle_configs(oracle_params, model, model_th, num_oracle_trials, oracle_time_limit, 
                                            oracle_stop_search_time)
        configurations = configurations + get_best_configs(best_params)
        # run full FJSP for different time limit based on the problem size
        if args.n_jobs == 35:
            for full_t in [500, 1500]:
                configurations += [(f'full {full_t}', {'action_type': 'full', 'time_limit': full_t})]
        elif args.n_jobs == 40:
            for full_t in [1000, 2500]:
                configurations += [(f'full {full_t}', {'action_type': 'full', 'time_limit': full_t})]
        else:
            for full_t in [200, 500]:
                configurations += [(f'full {full_t}', {'action_type': 'full', 'time_limit': full_t})]
    else:
        print(f'Using param from args ({window}, {step}, {time_limit}, {stop_search_time})')
        configurations = get_default_configs(window, model, model_th, time_limit, stop_search_time, 
                       num_oracle_trials, oracle_time_limit, oracle_stop_search_time)

    for vi in range(index_start, index_end):
        data_loc = f'{jss_data_dir}/data_{vi}.pkl'
        if os.path.exists(data_loc):
            print(f'Load data from {data_loc}')
            if 'end' in optim_option:  # start and end delay
                jobs_data, n_machines, n_jobs, scheduled_starts, scheduled_ends = pickle.load(open(data_loc, 'rb'))
            else:  # start delay
                jobs_data, n_machines, n_jobs, scheduled_starts = pickle.load(open(data_loc, 'rb'))
                scheduled_ends = {}
        else:
            print(f'Generate data, {data_loc} does not exist!')
            if 'end' in optim_option:
                jobs_data, n_machines, scheduled_starts, scheduled_ends = get_flexible_jss_data_from_args(args)
                n_jobs = len(jobs_data)
                pickle.dump([jobs_data, n_machines, n_jobs, scheduled_starts, scheduled_ends], open(data_loc, 'wb'))
            else:
                jobs_data, n_machines, scheduled_starts = get_flexible_jss_data_from_args(args)
                n_jobs = len(jobs_data)
                pickle.dump([jobs_data, n_machines, n_jobs, scheduled_starts], open(data_loc, 'wb'))
                scheduled_ends = {}
        
        stats_file_name = os.path.join(test_stats_dir, f'stats_e{load_model_epoch}th{model_th}_{vi}.pkl')
        print(f'Stats (objective and time) save to {stats_file_name}') 
        
        if os.path.exists(stats_file_name):
            total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats = pickle.load(open(stats_file_name, 'rb'))
        else:
            total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats = {}, {}, {}, {}

        sel_configs = [(label, params) for label, params in configurations if label not in total_delay_stats]  
        
        print(f'{len(sel_configs)} configurations to run: {[label for label, _ in sel_configs]} ...')
        for label, params_dict in sel_configs:
            print(f'*************** {label.upper()} ****************')

            if 'full' in label.lower():
                if "time_limit" in params_dict:
                    full_time_limit = params_dict["time_limit"]
                else:
                    full_time_limit = time_limit
                print(f'Solve full problem with time limit {full_time_limit}!')
                jobs_solution, machines_assignment, solve_time, objective = flexible_jss(
                    jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                    time_limit=full_time_limit, stop_search_time=full_time_limit,  # do not stop search earlier
                    machines_start_time=None, jobs_start_time=None, optim_option=optim_option)
                average_solve_time = solve_time
                total_delay = sum([jobs_solution[(job_id, task_id)][4] for job_id, job in jobs_data.items()
                                for task_id, task in job.items()])
                average_delay = total_delay / len(jobs_solution)
            else:
                config_labels = ["window", "step", "time_limit", "stop_search_time"]
                config_params = [window, step, time_limit, stop_search_time]
                for config_i, config_label in enumerate(config_labels):
                    if config_label in params_dict:
                        config_params[config_i] = params_dict[config_label]
                        del params_dict[config_label]
                        
                config_window, config_step, config_time_limit, config_stop_search_time = config_params
                in_params = [n_machines, n_jobs, jobs_data, scheduled_starts, scheduled_ends, 
                            config_window, config_step, config_time_limit, config_stop_search_time,
                            optim_option]

                if 'perturb_data' not in params_dict and args.perturb_data:
                    params_dict['perturb_data'] = args.perturb_data
                    params_dict['perturb_p'] = args.perturb_p

                total_delay, average_delay, solve_time, average_solve_time = rolling_horizon(*in_params, **params_dict)
            
            total_delay_stats[label] = total_delay
            average_delay_stats[label] = average_delay
            solve_time_stats[label] = solve_time
            average_solve_time_stats[label] = average_solve_time

            pickle.dump([total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats], 
                        open(stats_file_name, 'wb'))

        print('\n==================stats==================')
        for k in total_delay_stats:
            print(f'{k} delay: {np.mean(total_delay_stats[k]):.2f} +- {np.std(total_delay_stats[k]):.2f} ' 
                  f'(average delay: {np.mean(average_delay_stats[k]):.2f} +- {np.std(average_delay_stats[k]):.2f}) | ' 
                  f'solve time: {np.mean(solve_time_stats[k]):.2f} +- {np.std(solve_time_stats[k]):.2f} '
                  f'(average solve time: {np.mean(average_solve_time_stats[k]):.2f} +- {np.std(average_solve_time_stats[k]):.2f})')
        print('=========================================\n')


if __name__ == '__main__':
    # JSS parameters
    parser = get_jss_argparser()

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
    parser.add_argument("--load_model_epoch", default=100, type=int, help="the epoch that we load the model from")
    # learning
    parser.add_argument("--eval_every", default=20, type=int, help="frequency of validation")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--lr_delay", default=1, type=float, help="learning rate decay")
    parser.add_argument("--train_steps", default=100000, type=int, help="number of training steps") 
    parser.add_argument("--num_epochs", default=600, type=int, help="number of training epochs")
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden dimension of the model")
    parser.add_argument("--pos_weight", default=1, type=float, help="weight of positive label for BCE loss")

    # test statistics
    parser.add_argument('--test', action="store_true", help="test or train?")
    parser.add_argument('--do_simple_test', action="store_true", help="only test default and model?")
    parser.add_argument('--model_th', default=0.5, type=float, help="model threshold")
    parser.add_argument('--load_best_params', action="store_true", 
                        help="load the best window-step-time_limit-stop_search_time param for each test mehot")
    parser.add_argument('--best_params_file', default='train_test_dir/param_search/params/best_params.pkl', 
                        help="where to load the best params")

    # roll out parameters
    parser.add_argument("--time_limit", default=30, type=int, help="cp-sat solver time limit") 
    parser.add_argument("--stop_search_time", default=10, type=int, help="cp-sat solver stop search wait time") 
    parser.add_argument("--window", default=50, type=int) 
    parser.add_argument("--step", default=10, type=int) 
    parser.add_argument("--perturb_data", action='store_true')
    parser.add_argument("--perturb_p", default=0.1, type=float)

    parser.add_argument("--oracle_time_limit", default=30, type=int, help="time limit to find ground truth label")
    parser.add_argument("--oracle_stop_search_time", default=3, type=int, help="time limit to stop search for finding ground truth label")
    parser.add_argument("--n_cpus", default=1, type=int, help="cpus to find best") 
    parser.add_argument("--num_oracle_trials", default=5, type=int, help="number of oracle trials to find the most overlap")
    parser.add_argument("--num_best_randoms", default=5, type=int, help="number of random sample to find the best random") 
    parser.add_argument("--random_p", default=0.8, type=float, help="best random label") 
    
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

    #### FJSP instance parameters
    n_machines = args.n_machines
    n_jobs = args.n_jobs

    #### model parameters
    INPUT_TASK_DIM, INPUT_MACHINE_DIM, TASK_FEAT_IDX, IN_PREV_IDX, \
            N_TRANS_HEADS, TRANS_DROPOUT, OUTPUT_DIM, FlexibleJSSDataset, get_dataloader = get_params_and_fn(
        args.optim_option, args.perturb_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_weight = args.pos_weight
    model_name = args.model_name
    model_dir = os.path.join(args.model_dir, model_name)
    log_dir = os.path.join(args.log_dir, model_name)
    test_stats_dir = os.path.join(args.test_stats_dir, model_name)

    for d in [log_dir, model_dir, test_stats_dir]:
        if not os.path.exists(d):
            time.sleep(0.1 + 10 * np.random.rand())  # sleep for a random number of seconds
            if not os.path.exists(d):
                os.makedirs(d)
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

    ### compute normalization statistics (TODO: save and load this instead of recompute everytime)
    x_tasks_norm = RunningNormalization(device=device)
    x_machines_norm = RunningNormalization(device=device)
    normalizer_dir = os.path.join(args.model_dir, f'input_normalizer')
    if not os.path.exists(normalizer_dir):
        os.makedirs(normalizer_dir)
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

    ### construct model, optimizer, criterion
    model = FlexibleJSSNet(hidden_dim=args.hidden_dim, 
                           output_dim=OUTPUT_DIM, 
                           x_tasks_norm=x_tasks_norm, 
                           x_machines_norm=x_machines_norm).to(device)

    if args.test:
        load_model_epoch = args.load_model_epoch
        load_model(model, load_dir=model_dir, load_name=model_name, model_epoch=load_model_epoch) 
        rollout_model(model, args.val_index_start, args.val_index_end, args.jss_data_dir, 
                      model_th=args.model_th, optim_option=args.optim_option,
                      test_stats_dir=test_stats_dir, 
                      do_simple_test=args.do_simple_test)
    else:
        ### load val data
        if args.val_index_start == -1 and args.val_index_end == -1:
            val_data = load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=-1)
        else:
            val_data = []
            for idx in range(args.val_index_start, args.val_index_end):
                val_data += load_dataset(data_dir=args.train_data_dir, data_name=f'train', data_index=idx)
        
        val_dataset = FlexibleJSSDataset(val_data)
        val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=True, follow_batch=['x_tasks', 'x_machines'], num_workers=0,
                                    pin_memory=False)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_steps)

        train_model(model, optimizer, train_loader, val_loader, args.num_epochs, writer, 
                    scheduler=scheduler, eval_every=args.eval_every,
                    model_dir=model_dir, model_name=model_name, pos_weight=pos_weight)
        


