'''test time analysis'''
import itertools
import random
import pickle
import os
import argparse
from collections import defaultdict
import numpy as np
import pdb

def print_sol(make_span_stats, solve_time_stats, average_solve_time_stats, index_start, index_end,
             skipped_indices=[]):
    default_key = None
    for k in make_span_stats:
        if 'default [' in k:
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
    

def get_sol_data(index_start, index_end, test_dir, load_model_epoch, model_th, 
                verbose=True, skipped_indices=[]):
    make_span_stats = defaultdict(list)
    solve_time_stats = defaultdict(list)
    average_solve_time_stats = defaultdict(list)

    for vi in range(index_start, index_end):
        if vi in skipped_indices: continue
        # if vi == 474: continue
        file_name = f'{test_dir}/stats_e{load_model_epoch}th{model_th}_{vi}.pkl'
        # print(file_name)
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


def get_lns_data(index_start, index_end, test_dir, load_model_epoch, model_th, verbose=True):
    make_span_stats = defaultdict(list)
    solve_time_stats = defaultdict(list)
    average_solve_time_stats = defaultdict(list)

    for vi in range(index_start, index_end):
        # if vi == 474: continue
        file_name = f'{test_dir}/stats_{vi}.pkl'

        # print(file_name)
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
        print_sol(make_span_stats, solve_time_stats, average_solve_time_stats, index_start, index_end)
    
    return make_span_stats, solve_time_stats, average_solve_time_stats

if __name__ == '__main__':
    index_start = 500
    index_end = 530

    def get_default():

        test_data_dict = {
            '30': 'j20-m10-t30_mix-w80-s30-t60-st3',
            '40': 'j20-m10-t40_mix-w80-s30-t60-st3',
            '60': 'j20-m10-t60_mix-w80-s30-t60-st3',
            '100_transfer_from_40': 'j20-m10-t100_mix-w80-s30-t60-st3', 
            '100_transfer_from_60': 'j20-m10-t100_mix-w80-s30-t60-st3_transfer-from-60',
        }

        test_data_name = test_data_dict['100_transfer_from_60']
        model_name = 'model_mlp_full_mha_pw0.5'
        test_dir = f'train_test_dir/test_stats/{test_data_name}/{model_name}'
        model_th = 0.5
        load_model_epoch = 120
        get_sol_data(index_start, index_end, test_dir, load_model_epoch, model_th)

    def get_machine_breakdown():
        test_data_dict = {
            'low': 'j20-m10-t40_mix-w80-s30-t60-st3-low',
            'mid': 'j20-m10-t40_mix-w80-s30-t60-st3-mid',
            'high': 'j20-m10-t40_mix-w80-s30-t60-st3-high',
        }

        test_data_name = test_data_dict['high']
        model_name = 'model_mlp_full_mha_pw0.5'
        test_dir = f'train_test_dir/machine_breakdown/test_stats/{test_data_name}/{model_name}'
        model_th = 0.5
        load_model_epoch = 120
        get_sol_data(index_start, index_end, test_dir, load_model_epoch, model_th)

    get_default()
    # get_machine_breakdown()