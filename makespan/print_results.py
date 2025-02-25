import pickle
import os
import argparse
from collections import defaultdict
import numpy as np

def print_sol(make_span_stats, solve_time_stats, average_solve_time_stats, index_start, index_end,
             skipped_indices=[]):
    default_key = None
    for k in make_span_stats:
        if 'default' in k:
            default_key = k
            break
    default_rho_make_span = np.mean(make_span_stats[default_key])
    default_rho_time = np.mean(solve_time_stats[default_key])

    for k in make_span_stats:
        if len(make_span_stats[k]) < index_end - index_start - len(skipped_indices): 
            print(f'Skip {k} because only {len(make_span_stats[k])} samples are tested...')
            continue
        
        # time improvement from default rho
        k_make_span = np.mean(make_span_stats[k])
        k_time = np.mean(solve_time_stats[k])
        k_make_span_improv = (default_rho_make_span - k_make_span) / default_rho_make_span * 100
        k_time_improv = (default_rho_time - k_time) / default_rho_time * 100

        print(f'{k} ({len(make_span_stats[k])}) solve time: {np.mean(solve_time_stats[k]):.1f} +- {np.std(solve_time_stats[k]):.1f} ({k_time_improv:.1f}%) | ' 
            f'makespan: {np.mean(make_span_stats[k]):.1f} +- {np.std(make_span_stats[k]):.1f} ({k_make_span_improv:.1f}%) | '
            f'(average solve time: {np.mean(average_solve_time_stats[k]):.1f} +- {np.std(average_solve_time_stats[k]):.1f})')
    

def get_sol_data(index_start, index_end, stats_dir, data_name, 
                 verbose=True, skipped_indices=[]):
    make_span_stats = defaultdict(list)
    solve_time_stats = defaultdict(list)
    average_solve_time_stats = defaultdict(list)

    for vi in range(index_start, index_end):
        if vi in skipped_indices: continue
        file_name = f'{stats_dir}/{data_name}_{vi}.pkl'
        if not os.path.exists(file_name):
            print(f'{file_name} does not exist')
            continue
        
        make_span_stats_i, solve_time_stats_i, average_solve_time_stats_i = pickle.load(open(file_name, 'rb'))
        for k in make_span_stats_i:
            make_span_stats[k].append(make_span_stats_i[k])
            solve_time_stats[k].append(solve_time_stats_i[k])
            average_solve_time_stats[k].append(average_solve_time_stats_i[k])

    if verbose:
        print('\n==================stats==================')
        print_sol(make_span_stats, solve_time_stats, average_solve_time_stats, index_start, index_end, skipped_indices=skipped_indices)

    return make_span_stats, solve_time_stats, average_solve_time_stats


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--index_start', type=int, default=0)
    parse.add_argument('--index_end', type=int, default=1)
    parse.add_argument('--stats_dir', type=str, default="")
    parse.add_argument('--data_name', type=str, default="data")
    args = parse.parse_args()

    index_start = args.index_start
    index_end = args.index_end
    stats_dir = args.stats_dir
    data_name = args.data_name

    get_sol_data(index_start, index_end, stats_dir, data_name)
