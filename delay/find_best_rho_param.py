import os
import pickle
import numpy as np
from collections import defaultdict
from flexible_jss_instance import get_jss_argparser, get_flexible_jss_data_from_args

def parse_param_search_stats(param_dir, test_stats_dir, index_start, index_end, verbose=True):
    total_delay_stats = defaultdict(dict)
    average_delay_stats = defaultdict(dict)
    solve_time_stats = defaultdict(dict) 
    average_solve_time_stats = defaultdict(dict)

    params_file = os.path.join(param_dir, f'params.pkl')
    if not os.path.exists(params_file):
        print(f'Params filter {params_file} does not exist!')
        return total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats
    
    params = pickle.load(open(params_file, 'rb'))

    for pi, (window, step, time_limit, stop_search_time) in enumerate(params):
        for vi in range(index_start, index_end):
            file_name = os.path.join(test_stats_dir, f'stats-w{window}-s{step}-t{time_limit}-st{stop_search_time}/{vi}.pkl')
            if not os.path.exists(file_name):
                continue

            total_delay_stats_i, average_delay_stats_i, solve_time_stats_i, average_solve_time_stats_i = pickle.load(open(file_name, 'rb'))
            for k in total_delay_stats_i:
                if pi not in total_delay_stats[k]:
                    total_delay_stats[k][pi] = []
                    average_delay_stats[k][pi] = []
                    solve_time_stats[k][pi] = []
                    average_solve_time_stats[k][pi] = []
                
                total_delay_stats[k][pi].append(total_delay_stats_i[k])
                average_delay_stats[k][pi].append(average_delay_stats_i[k])
                solve_time_stats[k][pi].append(solve_time_stats_i[k])
                average_solve_time_stats[k][pi].append(average_solve_time_stats_i[k])

    total_delay_stats_res  = defaultdict(dict)
    average_delay_stats_res = defaultdict(dict)
    solve_time_stats_res = defaultdict(dict)
    average_solve_time_stats_res = defaultdict(dict)
    for k in total_delay_stats:
        for pi in total_delay_stats[k]:
            if len(total_delay_stats[k][pi]) < index_end - index_start: continue

            total_delay_stats_res[k][pi] = np.mean(total_delay_stats[k][pi]) 
            average_delay_stats_res[k][pi] = np.mean(average_delay_stats[k][pi])
            solve_time_stats_res[k][pi] = np.mean(solve_time_stats[k][pi])
            average_solve_time_stats_res[k][pi] = np.mean(average_solve_time_stats[k][pi])
    
    if verbose:
        print('\n==================stats==================')
        for k in total_delay_stats_res:
            for pi in total_delay_stats_res[k]:
                print(f'{k} param {params[pi]} ({len(total_delay_stats[k][pi])}) '
                      f'delay: {np.mean(total_delay_stats[k][pi]):.2f} +- {np.std(total_delay_stats[k][pi]):.2f} ' 
                      f'(average delay: {np.mean(average_delay_stats[k][pi]):.2f} +- {np.std(average_delay_stats[k][pi]):.2f}) | ' 
                      f'solve time: {np.mean(solve_time_stats[k][pi]):.2f} +- {np.std(solve_time_stats[k][pi]):.2f} '
                      f'(average solve time: {np.mean(average_solve_time_stats[k][pi]):.2f} +- {np.std(average_solve_time_stats[k][pi]):.2f})')
                
    return total_delay_stats_res, average_delay_stats_res, solve_time_stats_res, average_solve_time_stats_res


def get_best_param_search(param_dir, params_stats_dir, index_start, index_end):
    params_file = os.path.join(param_dir, f'params.pkl')
    if not os.path.exists(params_file):
        print(f'Params filter {params_file} does not exist!')
        return

    total_delay_stats, average_delay_stats, solve_time_stats, average_solve_time_stats = parse_param_search_stats(
        param_dir, params_stats_dir, index_start, index_end, verbose=True)
    
    all_delays = [total_delay_stats[k][pi] for k in total_delay_stats for pi in total_delay_stats[k]]
    if len(all_delays) == 0:
        print('No param search stats found!')
        return
    
    best_delay = min(all_delays) 
    print(f'best delay {best_delay:.2f}')
    delay_thresholds = [best_delay * mult for mult in [1.05, 1.1, 1.15, 1.2, 1.25, 
                                                       1.3, 1.35, 1.4, 1.45, 1.5, 
                                                       1.6, 1.7, 1.8, 1.9, 
                                                       2, 3, 5, 10, 20, 30, 40, 50, 100]]

    params = pickle.load(open(params_file, 'rb'))
    best_params_file = os.path.join(param_dir, f'best_params.pkl')
    print('Best RHO Parameter saved to', best_params_file)

    best_params = {}
    default_best_params = (80, 25, 60, 3)  # in case we can't find any good parameter

    total_delay_stats_filter = {}
    average_delay_stats_filter = {}
    solve_time_stats_filter = {}
    average_solve_time_stats_filter = {}

    oracle_pi = -1

    for k, p_list in total_delay_stats.items():
        min_solve_time = float('inf')
        best_pi = -1
        for delay_threshold in delay_thresholds:
            for pi in p_list:
                if total_delay_stats[k][pi] < delay_threshold and solve_time_stats[k][pi] < min_solve_time:
                    min_solve_time = solve_time_stats[k][pi]
                    best_pi = pi
            if best_pi != -1:
                break
        
        if k == 'oracle':
            oracle_pi = best_pi
        
        if best_pi == -1:            
            print(f'No best pi found for {k}')
            best_params[k] = default_best_params
            continue
        else:
            best_p = params[best_pi]
            print(f'{k} best pi {best_p} '
                  f'total delay {total_delay_stats[k][best_pi]:.2f}, solve time {solve_time_stats[k][best_pi]:.2f}, '
                  f'average delay {average_delay_stats[k][best_pi]:.2f}, average solve time {average_solve_time_stats[k][best_pi]:.2f}')
            total_delay_stats_filter[k] = {best_pi: total_delay_stats[k][best_pi]}
            average_delay_stats_filter[k] = {best_pi: average_delay_stats[k][best_pi]}
            solve_time_stats_filter[k] = {best_pi: solve_time_stats[k][best_pi]}
            average_solve_time_stats_filter[k] = {best_pi: average_solve_time_stats[k][best_pi]}
            best_params[k] = params[best_pi]

    print('******************** oracle param search ********************')
    for k, p_list in total_delay_stats.items():
        oracle_p = params[oracle_pi]
        if oracle_pi not in total_delay_stats[k]:
            continue
        print(f'{k} best pi {oracle_p} '
              f'total delay {total_delay_stats[k][oracle_pi]:.2f}, solve time {solve_time_stats[k][oracle_pi]:.2f}, '
              f'average delay {average_delay_stats[k][oracle_pi]:.2f}, average solve time {average_solve_time_stats[k][oracle_pi]:.2f}')

    pickle.dump(best_params, open(best_params_file, 'wb'))



if __name__ == '__main__':
    parser = get_jss_argparser()
    parser.add_argument("--data_index_start", default=0, type=int, help="the start index of the collected data")
    parser.add_argument("--data_index_end", default=30, type=int, help="the end index of the collected data")   
    parser.add_argument("--jss_data_dir", default="train_data_dir/instance", type=str, help="save instance directory")
    parser.add_argument("--param_dir", default="train_test_dir/param_search/params", type=str, help="save instance directory")
    parser.add_argument("--param_stats_dir", default="train_test_dir/param_search/stats", type=str, help="stats directory")
    args = parser.parse_args()

    index_start = args.data_index_start
    index_end = args.data_index_end
    param_dir = args.param_dir
    params_stats_dir = args.param_stats_dir
    jss_data_name = get_flexible_jss_data_from_args(args)

    get_best_param_search(param_dir=param_dir, params_stats_dir=params_stats_dir, 
                          index_start=index_start, index_end=index_end)
