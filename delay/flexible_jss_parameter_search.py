'''
hyperparameter search to find the best window, step, time_limit, and early_stop_time (stop_search_time) 
for each setting on a small hold out dataset, save them to a file
'''
import os
import sys
import pickle
import numpy as np
from flexible_jss_instance import get_jss_argparser, get_flexible_jss_data_from_args
from flexible_jss_main import rolling_horizon
from flexible_jss_test_rollout_configs import get_param_search_configs

def rollout_model(index_start, index_end, jss_data_dir, window, step, time_limit, stop_search_time,
                  optim_option='start_delay', test_stats_dir='train_test_dir/param_search/stats',
                  perturb_data=False, perturb_option=0, perturb_p=0.1):    

    configurations = get_param_search_configs(num_oracle_trials, oracle_time_limit, oracle_stop_search_time)

    # Run the dispatch for each configuration
    for vi in range(index_start, index_end):
        # load jss data
        data_loc = f'{jss_data_dir}/data_{vi}.pkl'
        if os.path.exists(data_loc):
            print(f'Load data from {data_loc}')
            if 'end' in optim_option:
                jobs_data, n_machines, n_jobs, scheduled_starts, scheduled_ends = pickle.load(open(data_loc, 'rb'))
            else:
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
        
        test_stats_dir_i = os.path.join(test_stats_dir, f'stats-w{window}-s{step}-t{time_limit}-st{stop_search_time}')
        if not os.path.exists(test_stats_dir_i):
            os.makedirs(test_stats_dir_i)

        stats_file_name = os.path.join(test_stats_dir_i, f'{vi}.pkl')     
        print(f'Stats (objective and time) save to {stats_file_name}') 
        if os.path.exists(stats_file_name):
            total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats = pickle.load(open(stats_file_name, 'rb'))
        else:
            total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats = {}, {}, {}, {}

        sel_configs = [(label, params) for label, params in configurations if label not in total_delay_stats]
        print(f'{len(sel_configs)} configurations to run ...')
        print(f'Parameter search window = {window}, step = {step}, time_limit = {time_limit}, stop_search_time = {stop_search_time} ...')
        stats = [{}, {}, {}, {}]
        for label, params_dict in sel_configs:
            print(f'*************** {label.upper()} ****************')
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

            if 'perturb_data' not in params_dict and perturb_data:
                params_dict['perturb_data'] = perturb_data
                params_dict['perturb_option'] = perturb_option
                params_dict['perturb_p'] = perturb_p
                
            if 'exec_choice' not in params_dict:
                if args.exec_choice == 'diff':
                    if 'model' in label or 'oracle' in label:
                        params_dict['exec_choice'] = 'do_extra'
                    else:
                        params_dict['exec_choice'] = 'default'
                else:
                    params_dict['exec_choice'] = args.exec_choice

            total_delay, average_delay, solve_time, average_solve_time = rolling_horizon(*in_params, **params_dict)
            total_delay_stats[label] = total_delay
            average_delay_stats[label] = average_delay
            solve_time_stats[label] = solve_time
            average_solve_time_stats[label] = average_solve_time

            # dump after each parameter run
            stats = [total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats]
            pickle.dump(stats, open(stats_file_name, 'wb'))

        
        print('\n==================stats==================')
        for k in total_delay_stats:
            print(f'{k} delay: {np.mean(total_delay_stats[k]):.2f} +- {np.std(total_delay_stats[k]):.2f} ' 
                  f'(average delay: {np.mean(average_delay_stats[k]):.2f} +- {np.std(average_delay_stats[k]):.2f}) | ' 
                  f'solve time: {np.mean(solve_time_stats[k]):.2f} +- {np.std(solve_time_stats[k]):.2f} '
                  f'(average solve time: {np.mean(average_solve_time_stats[k]):.2f} +- {np.std(average_solve_time_stats[k]):.2f})')
        with open('search_param_stats.txt', 'w') as f:
            for k in total_delay_stats:
                f.write(f'{k} delay: {np.mean(total_delay_stats[k]):.2f} +- {np.std(total_delay_stats[k]):.2f} ' 
                    f'(average delay: {np.mean(average_delay_stats[k]):.2f} +- {np.std(average_delay_stats[k]):.2f}) | ' 
                    f'solve time: {np.mean(solve_time_stats[k]):.2f} +- {np.std(solve_time_stats[k]):.2f} '
                    f'(average solve time: {np.mean(average_solve_time_stats[k]):.2f} +- {np.std(average_solve_time_stats[k]):.2f})')
        print('=========================================\n')
    return stats
      

def gen_param_data(n_jobs, param_dir, window_step_grid=None, time_limit_stop_time_grid=None):
    if window_step_grid is None:
        window_step_grid = [(80, 20), (80, 25), (80, 30), (80, 35), (80, 40), 
                            (100, 20), (100, 30), (100, 40), (100, 50), 
                            (50,15), (50, 20), (50, 25), (50, 30)] 

    if time_limit_stop_time_grid is None:
        time_limit_stop_time_grid = [(15, 2), (30, 3), (60, 3)]  

    param_file = os.path.join(param_dir, f'params.pkl')

    params = [(window, step, time_limit, stop_search_time) 
              for (window, step) in window_step_grid 
              for (time_limit, stop_search_time) in time_limit_stop_time_grid]
   
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    pickle.dump(params, open(param_file, 'wb'))
    param_text = os.path.join(param_dir, f'params_text.txt')
    with open(param_text, 'w') as f:
        for i, (window, step, time_limit, stop_search_time) in enumerate(params):
            f.write(f'Param {i}: window = {window}, step = {step}, time_limit = {time_limit}, stop_search_time = {stop_search_time}\n')


def get_param(param_dir, param_idx, window_step_grid=None, time_limit_stop_time_grid=None):
    # load hyperparam from file
    param_file = os.path.join(param_dir, f'params.pkl')
    if not os.path.exists(param_file):
        gen_param_data(
            args.n_jobs, param_dir, 
            window_step_grid=window_step_grid, time_limit_stop_time_grid=time_limit_stop_time_grid)
        
    param_list = pickle.load(open(param_file, 'rb'))
    if len(param_list) >= param_idx + 1:
        window, step, time_limit, stop_search_time = param_list[param_idx]
    else:
        print(f'{param_idx} out of range ({len(param_list)-1} max)! Exiting ...')
        sys.exit()

    return window, step, time_limit, stop_search_time


if __name__ == '__main__':
    parser = get_jss_argparser()
    # Parameter search
    parser.add_argument("--param_idx", default=1, type=int, help="number of data generated")
    parser.add_argument("--param_dir", default="train_test_dir/param_search/params", type=str, help="save instance directory")
    parser.add_argument("--param_stats_dir", default="train_test_dir/param_search/stats", type=str, help="stats directory")
    # JSS data
    parser.add_argument("--jss_data_dir", default="train_data_dir/instance", type=str, help="save instance directory")
    parser.add_argument("--data_index_start", default=0, type=int, help="the start index of the collected data")
    parser.add_argument("--data_index_end", default=30, type=int, help="the end index of the collected data")     
    # oracle
    parser.add_argument("--num_oracle_trials", default=5, type=int,
                        help="number of oracle trials to find the most overlap")  
    parser.add_argument("--oracle_time_limit", default=30, type=int, help="time limit to find ground truth label")
    parser.add_argument("--oracle_stop_search_time", default=3, type=int, help="time limit to stop search for finding ground truth label")             
    
    parser.add_argument("--perturb_data", action='store_true')
    parser.add_argument("--perturb_option", default=0, type=int, choices=[0, 1])
    parser.add_argument("--perturb_p", default=0.1, type=float)
    parser.add_argument("--exec_choice", default='default', type=str, choices=['default', 'do_extra', 'by_schedule', 'diff'])
    
    args = parser.parse_args()
     
    num_oracle_trials = args.num_oracle_trials
    oracle_time_limit = args.oracle_time_limit
    oracle_stop_search_time = args.oracle_stop_search_time
    data_index_start = args.data_index_start
    data_index_end = args.data_index_end
    jss_data_dir = args.jss_data_dir
    param_dir = args.param_dir
    param_idx = args.param_idx
    param_stats_dir = args.param_stats_dir

    window, step, time_limit, stop_search_time = get_param(param_dir, param_idx)
    optim_option = args.optim_option

    rollout_model(data_index_start, data_index_end, jss_data_dir, window, step, time_limit, stop_search_time,
                  optim_option=optim_option, test_stats_dir=param_stats_dir,
                  perturb_data=args.perturb_data, perturb_option=args.perturb_option, perturb_p=args.perturb_p)