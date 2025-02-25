'''test time analysis'''
import itertools
import random
import pickle
import os
import argparse
from collections import defaultdict
import numpy as np


def get_sol_data(index_start, index_end, stats_dir, data_name, verbose=True):
    total_delay_stats = defaultdict(list)
    average_delay_stats = defaultdict(list)
    solve_time_stats = defaultdict(list)
    average_solve_time_stats = defaultdict(list)


    for vi in range(index_start, index_end):
        file_name = os.path.join(stats_dir, f'{data_name}_{vi}.pkl')
        if not os.path.exists(file_name):
            continue
        
        total_delay_stats_i, average_delay_stats_i, solve_time_stats_i, average_solve_time_stats_i = pickle.load(open(file_name, 'rb'))
        for k in total_delay_stats_i:
            total_delay_stats[k].append(total_delay_stats_i[k])
            average_delay_stats[k].append(average_delay_stats_i[k])
            solve_time_stats[k].append(solve_time_stats_i[k])
            average_solve_time_stats[k].append(average_solve_time_stats_i[k])

    if verbose:
        print('\n==================stats==================')
        for k in total_delay_stats:
            print(f'{k} ({len(total_delay_stats[k])}) delay: {np.mean(total_delay_stats[k]):.2f} +- {np.std(total_delay_stats[k]):.2f} ' 
                f'(average delay: {np.mean(average_delay_stats[k]):.2f} +- {np.std(average_delay_stats[k]):.2f}) | ' 
                f'solve time: {np.mean(solve_time_stats[k]):.2f} +- {np.std(solve_time_stats[k]):.2f} '
                f'(average solve time: {np.mean(average_solve_time_stats[k]):.2f} +- {np.std(average_solve_time_stats[k]):.2f})')
    
    ret_dict = {'Label': [], 
                'Total Delay Mean': [], 
                'Total Delay Std': [], 
                'Average Delay Mean': [], 
                'Average Delay Std': [], 
                'Total Solve Time Mean': [], 
                'Total Solve Time Std': [], 
                'Average Solve Time Mean': [], 
                'Average Solve Time Std': []}
                
    for k in total_delay_stats:
        if len(total_delay_stats[k]) < index_end - index_start: continue
        ret_dict['Label'].append(k)
        ret_dict['Total Delay Mean'].append(np.mean(total_delay_stats[k]))
        ret_dict['Total Delay Std'].append(np.std(total_delay_stats[k]))
        ret_dict['Average Delay Mean'].append(np.mean(average_delay_stats[k]))
        ret_dict['Average Delay Std'].append(np.std(average_delay_stats[k]))
        ret_dict['Total Solve Time Mean'].append(np.mean(solve_time_stats[k]))
        ret_dict['Total Solve Time Std'].append(np.std(solve_time_stats[k]))
        ret_dict['Average Solve Time Mean'].append(np.mean(average_solve_time_stats[k]))
        ret_dict['Average Solve Time Std'].append(np.std(average_solve_time_stats[k]))
    return ret_dict


if __name__ == '__main__':

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
    
