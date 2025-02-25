import sys
import os
import pdb
import random
import copy
import collections
import numpy as np
from collections import defaultdict
import argparse
import pickle
import time

from flexible_jss_main_common import flexible_jss, validate_jss_sol, sort_by_task_order, find_overlapping_jobs_solution_oracle 
from flexible_jss_util import store_dataset
from flexible_jss_data_common import get_rollout_prediction
from flexible_jss_data import get_data, get_rollout_data

EPS = 1e-5

def select_jss_data(sel_task_indices, jobs_data, jobs_solution={}):
    jobs_data_sel = {}

    num_fix = 0
    num_not_fix = 0

    for job_id, task_id in sel_task_indices:
        if job_id not in jobs_data_sel:
            jobs_data_sel[job_id] = {}

        if (job_id, task_id) in jobs_solution:
            alt_id = jobs_solution[(job_id, task_id)][0]
            jobs_data_alt = copy.deepcopy(jobs_data[job_id][task_id][alt_id])
            jobs_data_sel[job_id][task_id] = {alt_id: jobs_data_alt}
            num_fix += 1
        else:
            jobs_data_sel[job_id][task_id] = copy.deepcopy(jobs_data[job_id][task_id])
            num_not_fix += 1

    if len(jobs_solution) > 0:    
        print(f'RHO subproblem: {num_fix} fix, {num_not_fix} not fix')

    for job_id in jobs_data_sel:
        # sort by task id to maintain precedence order
        jobs_data_sel[job_id] = dict(sorted(jobs_data_sel[job_id].items(), key=lambda x: x[0]))
        
    return jobs_data_sel


def get_overlapping_jobs_solution(action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                                machines_start_time, jobs_start_time, jobs_solution_sel, overlapping_task_indices, 
                                sel_task_indices, n_cpus, first_frac,
                                random_p, model, model_decode_strategy, model_th, model_topk_th, model_time, 
                                training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time):
    
    overlapping_jobs_solution_all = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices}
    overlapping_machines_assignment_all = {machine: [task for task in machines_assignment_sel[machine]
                                                        if (task.job, task.index) in overlapping_task_indices]
                                            for machine in machines_assignment_sel}
    if action_type == 'fix_all':
        overlapping_jobs_solution = overlapping_jobs_solution_all
    elif action_type == 'oracle':
        overlapping_jobs_solution, objective_reference = find_overlapping_jobs_solution_oracle(
            jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
            oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
            overlapping_jobs_solution_all, num_oracle_trials=num_oracle_trials, n_cpus=n_cpus)
    elif action_type == 'first':
        num_first = int(first_frac * len(overlapping_task_indices))
        overlapping_jobs_solution = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices[:num_first]}
    elif action_type == 'random':
        # for each task in overlapping_task_indices randomly decide whether to fix the task or not
        overlapping_jobs_solution = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices
                                        if random_p > 0 and np.random.rand() < random_p}
        print('-----------------------------------')
    elif action_type == 'collect_data':
        overlapping_jobs_solution, objective_reference = find_overlapping_jobs_solution_oracle(
            jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
            oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
            overlapping_jobs_solution_all, num_oracle_trials=num_oracle_trials, n_cpus=n_cpus)
       
        data = get_data(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
                        overlapping_task_indices, overlapping_jobs_solution_all,
                        overlapping_machines_assignment_all,
                        overlapping_jobs_solution)
        training_data.append(data)
    elif action_type == 'model':
        model_start_time = time.time()
        data = get_rollout_data(jobs_data_sel, n_machines, machines_start_time,
                                jobs_start_time, overlapping_task_indices, overlapping_jobs_solution_all,
                                overlapping_machines_assignment_all)
        tasks_pred_fix = get_rollout_prediction(model, data, decode_strategy=model_decode_strategy, 
                                                threshold=model_th, topk_th=model_topk_th)
        model_time += time.time() - model_start_time
        print(f'{len(tasks_pred_fix)} tasks predicted to be fixed out of {len(overlapping_task_indices)}')
        overlapping_jobs_solution = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices
                                        if (job_id, task_id) in tasks_pred_fix}
    else:
        print(f'Action type {action_type} not implemented!')
        raise NotImplementedError
    print(f'Number of overlapping solutions {len(overlapping_jobs_solution)} out of {len(overlapping_task_indices)}')
    return overlapping_jobs_solution, jobs_data_sel, model_time



#######################################################################################
def select_window(start_loc, num_tasks, sorted_task_indices, jobs_solution, window):
    '''
    return (job_id, task_id) and the location of the task in the sorted_task_indices
    '''
    # skip all tasks that have been executed before with a later start time
    sel_task_indices = []
    sel_i_task_loc = []
    i_task = start_loc   # find from the beginning
    while len(sel_task_indices) < window and i_task < num_tasks:
        task_idx = sorted_task_indices[i_task]
        if task_idx not in jobs_solution:
            sel_task_indices.append(task_idx)
            sel_i_task_loc.append(i_task)
        i_task += 1

    print('RHO window indices (overlapping + new)', sel_i_task_loc)
    if len(sel_i_task_loc) == 0:
        print('There is a bug here!', len(jobs_solution), num_tasks)
        pdb.set_trace()

    return sel_task_indices, sel_i_task_loc



def exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
              jobs_solution, jobs_solution_sel,  machines_assignment, machines_assignment_sel, 
              machines_start_time, jobs_start_time, n_machines, step, debug=True):
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    if len(jobs_solution) + len(sel_task_indices) >= num_tasks:
        step_task_indices = sel_task_indices
        start_loc = num_tasks
    else:
        # execute step number of tasks
        step_task_indices = sel_task_indices[:step]
        print('Do execution by actual start time')
        # execute the first len(step_task_indices) tasks whose actual start time is the earliest
        sel_task_indices_with_start = [(jobs_solution_sel[(job_id, task_id)][2], job_id, task_id) 
                                        for job_id, task_id in sel_task_indices]
        sel_task_indices_with_start.sort()
        print('step_task_indices before', step_task_indices)
        step_task_indices = [(job_id, task_id) for _, job_id, task_id in sel_task_indices_with_start[:step]]
        print('step_task_indices after', step_task_indices)
       
        # the following shouldn't matter
        start_loc = min(sel_i_task_loc)
        while start_loc < num_tasks and sorted_task_indices[start_loc] in step_task_indices:
            start_loc += 1
        print('next start loc', start_loc)

    num_exec_sel, num_exec_all = 0, 0
    
    for job_id, task_id in step_task_indices:
        # execute the task
        selected_alt, machine, start_value, duration = jobs_solution_sel[(job_id, task_id)]
        jobs_solution[(job_id, task_id)] = copy.deepcopy(jobs_solution_sel[(job_id, task_id)])
        machines_assignment[machine].append(
            assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration)
        )
        machines_start_time[machine] = max(machines_start_time[machine], start_value + duration)
        jobs_start_time[job_id] = max(jobs_start_time[job_id], start_value + duration)
        if (job_id, task_id) in step_task_indices: num_exec_sel += 1
        num_exec_all += 1
    print(f'Execute {num_exec_all} tasks, num sel {num_exec_sel} (out of {len(step_task_indices)})')

            
    return start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time


def rolling_horizon(n_machines, n_jobs, jobs_data, window, step, time_limit, stop_search_time, 
                    oracle_time_limit=30, oracle_stop_search_time=3, 
                    action_type='default', do_warm_start=False, best_criterion='solve_time',
                    first_frac=0.5, random_p=0.8, num_oracle_trials=1, 
                    train_data_dir='train_data_dir/train_data', data_idx=1,
                    model=None, model_decode_strategy='argmax', model_th=0.5, model_topk_th=0.5, 
                    include_model_time=False, n_cpus=1, 
                    print_str=''):

    oracle_time_limit, oracle_stop_search_time = time_limit, stop_search_time
    if action_type == 'default':
        do_warm_start = False
    if 'warm_start' in action_type:
        do_warm_start = True
    
    sorted_task_indices = sort_by_task_order(jobs_data)

    #### tracking solutions
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    jobs_solution_sel = {}  # overlapping subproblem
    machines_assignment_sel = {}  # overlapping subproblem

    #### tracking stats
    solve_time, model_time, num_solves, num_tasks = 0, 0, 0, len(sorted_task_indices)
    
    # information required by the solving process
    machines_start_time = [0 for _ in range(n_machines)]
    jobs_start_time = [0 for _ in range(n_jobs)]
    prev_sel_task_indices = []  # task indices in the previous solution
    start_loc = 0  # location of the start of the next window in sorted_task_indices

    training_data = []

    # analysis_stats 
    new_tasks_in_overlap_list = []
    new_tasks_list = []
    prev_tasks_in_overlap_list = []
    prev_tasks_list = []
    print(f'*************** window {window} step {step} ***************')

    # maximum duration of the job data
    max_duration = max([jobs_data[job_id][task_id][alt_id][0] for job_id in jobs_data 
                        for task_id in jobs_data[job_id] for alt_id in jobs_data[job_id][task_id]])
    print(f'Max duration {max_duration}')


    while len(jobs_solution) < num_tasks:
        sel_task_indices, sel_i_task_loc = select_window(start_loc, num_tasks, sorted_task_indices, jobs_solution, window)
        jobs_data_sel = select_jss_data(sel_task_indices, jobs_data)
        
        if len(jobs_solution) > 0 and action_type != 'default':
            overlapping_task_indices = [task for task in prev_sel_task_indices if task in sel_task_indices]
            print(f'# Overlapping tasks = {len(overlapping_task_indices)}, out of {len(sel_task_indices)}')
            
            get_overlapping_jobs_solution_input = [
                action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                machines_start_time, jobs_start_time, jobs_solution_sel, overlapping_task_indices, sel_task_indices, 
                n_cpus, first_frac, random_p, model, model_decode_strategy, model_th, model_topk_th, model_time, 
                training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time
            ]

            overlapping_jobs_solution, jobs_data_sel, model_time = get_overlapping_jobs_solution(*get_overlapping_jobs_solution_input)
            sel_task_loc_dict = dict(zip(sel_task_indices, sel_i_task_loc))
            overlapping_task_loc = [sel_task_loc_dict[task] for task in overlapping_task_indices]
            fixed_tasks = [task for task in overlapping_task_indices if task in overlapping_jobs_solution]
            print(f'{action_type.upper()} overlapping operations indices (select {len(fixed_tasks)} out of {len(overlapping_task_loc)} ops) ['
                  f'{min(overlapping_task_loc) if len(overlapping_task_loc) > 0 else 0}, '
                  f'{max(overlapping_task_loc) if len(overlapping_task_loc) > 0 else 0}]:')
            print(f'All overlapping indices: {overlapping_task_indices}')
            print(f'Fixed indices: {fixed_tasks}')
            if not do_warm_start:  # hard filtering of the assignments (solution)
                jobs_data_sel = select_jss_data(sel_task_indices, jobs_data_sel, jobs_solution=overlapping_jobs_solution)

        else:  # default or first iteration
            overlapping_jobs_solution = {}

        ############################################################################################
        jobs_solution_sel, machines_assignment_sel, solve_time_sel, objective_sel = flexible_jss(
            jobs_data_sel, n_machines, time_limit=time_limit, stop_search_time=stop_search_time,
            machines_start_time=machines_start_time, jobs_start_time=jobs_start_time,
            do_warm_start=do_warm_start, jobs_solution_warm_start=overlapping_jobs_solution)

        if objective_sel == float('inf'):
            print('Infeasible!!')

        start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time = \
            exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
                      jobs_solution, jobs_solution_sel, machines_assignment, machines_assignment_sel,
                      machines_start_time, jobs_start_time, n_machines, step, jobs_data)
        
        prev_sel_task_indices = copy.deepcopy(sel_task_indices)   # for next iter
        solve_time += solve_time_sel
        num_solves += 1

        print(f'*************** exec_tasks {len(jobs_solution)} total num tasks {num_tasks} '
              f'window range [{sel_i_task_loc[0]}, {sel_i_task_loc[-1]+1}] ***************\n')
  
    if include_model_time:
        solve_time += model_time

    if action_type == 'collect_data':
        store_dataset(training_data, data_dir=train_data_dir, data_name=f'train', data_index=data_idx)

    average_solve_time = solve_time / num_solves
    make_span = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] for job_id, task_id in sorted_task_indices])
       
    # check if final solution is valid
    if not validate_jss_sol(jobs_data, n_machines, jobs_solution, machines_assignment, make_span, 
                     machines_start_time=None, jobs_start_time=None, 
                     check_final=False):
        print('Final solution is invalid!')
    else:
        print('Final solution is valid!')

    p_str = print_str if print_str != '' else f'{action_type.upper()}{" Warm Start " if do_warm_start else " "}'
    if len(new_tasks_in_overlap_list) > 0:
        print('.....................Simple Analysis Aggregated Results.....................')
        print(f'Average number of new tasks in overlap {np.mean(new_tasks_in_overlap_list)} '
              f'(std {np.std(new_tasks_in_overlap_list)}) out of {np.mean(new_tasks_list)}')
        print(f'Average number of prev tasks in overlap {np.mean(prev_tasks_in_overlap_list)} '
              f'(std {np.std(prev_tasks_in_overlap_list)}) out of {np.mean(prev_tasks_list)}')
        print('............................................................................')
    
    print(f'\n********************************* Rolling Horizon {p_str} (objective {make_span}) *********************************')
    print(f'total solve time {solve_time:.2f} average solve time {average_solve_time:.2f}, number of solves {num_solves}, makespan {make_span}')
    
    if 'model' in action_type: 
        print(f'model time {model_time:.2f} ({"Include" if include_model_time else "Exclude"})')
    print('\n')

    return jobs_solution, machines_assignment, solve_time, average_solve_time, make_span



def run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, time_limit, stop_search_time):
    data_loc = f'{jss_data_dir}/data_{data_idx}.pkl'
    assert os.path.exists(data_loc), f'{data_loc} does not exist!'
    print(f'Load data from {data_loc}')

    jobs_data, n_machines, n_jobs = pickle.load(open(data_loc, 'rb'))

    if os.path.exists(f'{stats_dir}/stats_{data_idx}.pkl'):
        print(f'Load stats from {stats_dir}/stats_{data_idx}.pkl')
        makespan_stats, solve_time_stats, average_solve_time_stats = pickle.load(open(f'{stats_dir}/stats_{data_idx}.pkl', 'rb'))
    else:
        makespan_stats = defaultdict(list)
        solve_time_stats = defaultdict(list)
        average_solve_time_stats = defaultdict(list)
    
    sel_configs = [(label, params_dict) for label, params_dict in configurations if label not in makespan_stats]
    print(f'Selected configurations: {len(sel_configs)} out of {len(configurations)}')
    for label, params_dict in sel_configs:
        print(f'*************** {label.upper()} ****************')
        if 'full' in label.lower():
            print(f'Solve full problem!')
            if "time_limit" in params_dict:
                full_time_limit = params_dict["time_limit"]
            _, _, solve_time, make_span = flexible_jss(
                jobs_data, n_machines, time_limit=full_time_limit, stop_search_time=full_time_limit,  # do not stop search earlier
                machines_start_time=None, jobs_start_time=None)
            avg_solve_time = solve_time
        else:  # rolling horizon
            config_labels = ["window", "step", "time_limit", "stop_search_time"]
            config_params = [window, step, time_limit, stop_search_time]
            for config_i, config_label in enumerate(config_labels):
                if config_label in params_dict:
                    config_params[config_i] = params_dict[config_label]
                    del params_dict[config_label]
                    
            config_window, config_step, config_time_limit, config_stop_search_time = config_params
            in_params = [n_machines, n_jobs, jobs_data, config_window, config_step, config_time_limit, config_stop_search_time]
            
            params_dict['print_str'] = label.upper()
            params_dict['data_idx'] = data_idx
            _, _, solve_time, avg_solve_time, make_span = rolling_horizon(*in_params, **params_dict)

        makespan_stats[label] = make_span
        solve_time_stats[label] = solve_time
        average_solve_time_stats[label] = avg_solve_time

    if len(sel_configs) > 0:
        print('\n==================stats==================')
        for i, (label, params_dict) in enumerate(configurations):
            print(f'{label}: makespan {makespan_stats[label]}, solve time {solve_time_stats[label]:.2f} (avg solve time {average_solve_time_stats[label]:.2f})')

        pickle.dump([makespan_stats, solve_time_stats, average_solve_time_stats], open(f'{stats_dir}/stats_{data_idx}.pkl', 'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flexible Job Shop Scheduling')
    # data directories
    parser.add_argument("--train_data_dir", default="debug_dir/train_data", type=str,
                        help="save train data directory")
    parser.add_argument("--jss_data_dir", default="train_data_dir/instance/j20-m10-t30_mix", 
                        type=str, help="save instance directory")
    parser.add_argument("--stats_dir", default="debug_dir/stats", type=str, help="stats directory")
    
    parser.add_argument("--data_idx", default=1, type=int, help="save instance index")

    parser.add_argument("--time_limit", default=60, type=int, help="cp-sat solver time limit")
    parser.add_argument("--stop_search_time", default=3, type=int, help="cp-sat solver stop search wait time")
    parser.add_argument("--oracle_time_limit", default=60, type=int, help="time limit to find ground truth label")
    parser.add_argument("--oracle_stop_search_time", default=3, type=int, help="time limit to stop search for finding ground truth label")
    
    parser.add_argument("--n_cpus", default=1, type=int, help="cpus to find best")
    parser.add_argument("--num_oracle_trials", default=5, type=int,
                        help="number of oracle trials to find the most overlap")
        
    # rolling horizon parameters
    parser.add_argument("--window", default=80, type=int)
    parser.add_argument("--step", default=30, type=int)

    parser.add_argument("--script_action", default='collect_data', 
                    choices=['collect_data', 'debug'], help="script action")
    args = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    stop_search_time = args.stop_search_time
    time_limit = args.time_limit
    oracle_time_limit = args.oracle_time_limit
    oracle_stop_search_time = args.oracle_stop_search_time
    num_oracle_trials = args.num_oracle_trials
    n_cpus = args.n_cpus
    
    #### Flexible jss instance parameters
    window = args.window 
    step = args.step  

    jss_data_dir = args.jss_data_dir
    train_data_dir = args.train_data_dir
    data_idx = args.data_idx
    stats_dir = args.stats_dir

    for d in [jss_data_dir, train_data_dir, stats_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    collect_data_configs = [
        (f'collect_data',
          {'action_type': 'collect_data', 'num_oracle_trials': num_oracle_trials,
           'train_data_dir': train_data_dir, 'data_idx': data_idx,
           'oracle_time_limit': oracle_time_limit, 
           'oracle_stop_search_time': oracle_stop_search_time})
    ]

    debug_configs = [
        ('default', {'action_type': 'default'}),
        ('oracle', {'action_type': 'oracle', 
                    'num_oracle_trials': num_oracle_trials,
                    'oracle_time_limit': oracle_time_limit,
                    'oracle_stop_search_time': oracle_stop_search_time}),
    ]


    if args.script_action == 'collect_data':
        configurations = collect_data_configs
    else:
        configurations = debug_configs
        
    run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, time_limit, stop_search_time)
    

    