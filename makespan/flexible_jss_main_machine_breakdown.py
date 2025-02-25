import pdb
import sys
import os
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
from flexible_jss_data_machine_breakdown import get_data as get_data_machine_breakdown
from flexible_jss_data_machine_breakdown import get_rollout_data as get_rollout_data_machine_breakdown

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
        print(f'select_jss_data: {num_fix} fix, {num_not_fix} not fix')

    for job_id in jobs_data_sel:
        # sort by task id to maintain precedence order
        jobs_data_sel[job_id] = dict(sorted(jobs_data_sel[job_id].items(), key=lambda x: x[0]))
        
    return jobs_data_sel


def select_jss_data_machine_breakdown(sel_task_indices, jobs_data, 
                                      breakdown_times=[], 
                                      breakdown_state=(False, 0)):
    jobs_data_sel = {}
    ignored_tasks = []
    reduced_tasks = []

    all_sel_jobs = set([job_id for job_id, task_id in sel_task_indices])

    for job_id in all_sel_jobs:
        all_sel_tasks = sorted([t_id for j_id, t_id in sel_task_indices if j_id == job_id])

        for task_id in all_sel_tasks:
            jobs_data_id_filter = copy.deepcopy(jobs_data[job_id][task_id])
            is_breakdown, breakdown_idx = breakdown_state

            if is_breakdown and len(breakdown_times) > breakdown_idx:
                cur_breakdown_machines = breakdown_times[breakdown_idx][-1]
                # remove these machines from the jobs_data
                jobs_data_id_filter = {alt_id: alt for alt_id, alt in jobs_data_id_filter.items() if alt[1] not in cur_breakdown_machines}

                if len(jobs_data_id_filter) == 0:
                    tasks_after = [(job_id, t_id) for t_id in all_sel_tasks if t_id >= task_id]
                    print(f'Job {job_id} task {task_id} has no alternative after breakdown. Ignore all tasks after ({len(tasks_after)} tasks: {tasks_after}). Wait for it to recover.')
                    ignored_tasks.extend(tasks_after)
                    break

                if len(jobs_data_id_filter) < len(jobs_data[job_id][task_id]):
                    reduced_tasks.append((job_id, task_id))

            if job_id not in jobs_data_sel:
                jobs_data_sel[job_id] = {}

            jobs_data_sel[job_id][task_id] = jobs_data_id_filter

            
    print(f'select_jss_data: {len(sel_task_indices) - len(ignored_tasks)} tasks selected')

    for job_id in jobs_data_sel:
        # sort by task id to maintain precedence order
        jobs_data_sel[job_id] = dict(sorted(jobs_data_sel[job_id].items(), key=lambda x: x[0]))
    
    return jobs_data_sel, ignored_tasks, reduced_tasks


def get_overlapping_jobs_solution(action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                                machines_start_time, jobs_start_time, jobs_solution_sel, overlapping_task_indices, 
                                n_cpus, first_frac,
                                random_p, model, model_decode_strategy, model_th, model_topk_th, model_time, 
                                training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time,
                                is_breakdown=False, breakdown_machines=[], 
                                breakdown_operations=[], recovered_operations=[]):
    
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
       
        data = get_data_machine_breakdown(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
                                            overlapping_task_indices, overlapping_jobs_solution_all,
                                            overlapping_machines_assignment_all,
                                            overlapping_jobs_solution, 
                                            is_breakdown, breakdown_machines, 
                                            breakdown_operations, recovered_operations)
        training_data.append(data)
    elif action_type == 'model':
        model_start_time = time.time()
        data = get_rollout_data_machine_breakdown(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
                                                    overlapping_task_indices, overlapping_jobs_solution_all,
                                                    overlapping_machines_assignment_all,
                                                    is_breakdown, breakdown_machines, 
                                                    breakdown_operations, recovered_operations)
        tasks_pred_fix = get_rollout_prediction(model, data, decode_strategy=model_decode_strategy, 
                                                threshold=model_th, topk_th=model_topk_th)
        model_time += time.time() - model_start_time
        overlapping_jobs_solution = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices
                                        if (job_id, task_id) in tasks_pred_fix}
        if is_breakdown:
            # remove tasks that are fixed by the model but are breakdown operations
            overlapping_jobs_solution = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_jobs_solution
                                        if (job_id, task_id) not in breakdown_operations}
        
        print(f'{len(overlapping_jobs_solution)} tasks predicted to be fixed out of {len(overlapping_task_indices)}')

    else:
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
    i_task = 0   # find from the beginning
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




def exec_step_machine_breakdown(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
              jobs_solution, jobs_solution_sel,  machines_assignment, machines_assignment_sel, 
              machines_start_time, jobs_start_time, n_machines, step, 
              breakdown_times=[], breakdown_state=(False, 0), debug=True):

    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    sel_task_indices_with_start = [(jobs_solution_sel[(job_id, task_id)][2], 
                                    jobs_solution_sel[(job_id, task_id)][2] + jobs_solution_sel[(job_id, task_id)][3],
                                    job_id, task_id) 
                                    for job_id, task_id in sel_task_indices]
    sel_task_indices_with_start.sort(key=lambda x: (x[0], x[1]))

    if len(jobs_solution) + len(sel_task_indices) >= num_tasks:
        chosen_task_indices = [(job_id, task_id) for _, _, job_id, task_id in sel_task_indices_with_start]
    else:        
        chosen_task_indices = [(job_id, task_id) for _, _, job_id, task_id in sel_task_indices_with_start[:step]]
        print('step_task_indices after', chosen_task_indices)
    

    num_exec_sel, num_exec_all = 0, 0
    step_task_indices = []

    curr_is_breakdown = breakdown_state[0]
    curr_breakdown_idx = breakdown_state[1]
    first_start_value = float('inf') if len(chosen_task_indices) == 0 else jobs_solution_sel[chosen_task_indices[0]][2]
    
    print(f'************ 1. [Curr Breakdown? {curr_is_breakdown}, idx {curr_breakdown_idx}] ************')

    if curr_is_breakdown:
        if curr_breakdown_idx == len(breakdown_times) or breakdown_times[curr_breakdown_idx][0][1] <= first_start_value:
            curr_is_breakdown = False

    print(f'************ 2. Reset If Not Breakdown [Curr Breakdown? {curr_is_breakdown}, idx {curr_breakdown_idx}] ************')
    while curr_breakdown_idx < len(breakdown_times) and breakdown_times[curr_breakdown_idx][0][1] <= first_start_value:
        curr_breakdown_idx += 1

    for chosen_task_idx, (job_id, task_id) in enumerate(chosen_task_indices):
        # execute the task
        selected_alt, machine, start_value, duration = jobs_solution_sel[(job_id, task_id)]
        curr_interval = (start_value, start_value + duration)
        if not curr_is_breakdown:
            if curr_breakdown_idx < len(breakdown_times) and breakdown_times[curr_breakdown_idx][0][0] < curr_interval[1]:
                # breakdown time overlaps with the task, stop execution
                curr_is_breakdown = True
                print(f'Machine breakdown at chosen index {chosen_task_idx} | curr_interval {curr_interval} | breakdown time {breakdown_times[curr_breakdown_idx][0]}')
                # set all breakdown machine start time to end of the breakdown time
                breakdown_end_time = breakdown_times[curr_breakdown_idx][0][1]
                breakdown_machines = breakdown_times[curr_breakdown_idx][-1]
                for breakdown_machine in breakdown_machines:
                    machines_start_time[breakdown_machine] = breakdown_end_time
                break   
        else:
            if curr_breakdown_idx == len(breakdown_times) or breakdown_times[curr_breakdown_idx][0][1] <= curr_interval[0]:
                curr_is_breakdown = False
                print("Machines are available again after breakdown")
                if curr_breakdown_idx != len(breakdown_times):
                    curr_breakdown_idx += 1
                break

        jobs_solution[(job_id, task_id)] = copy.deepcopy(jobs_solution_sel[(job_id, task_id)])
        machines_assignment[machine].append(
            assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration)
        )
        machines_start_time[machine] = max(machines_start_time[machine], start_value + duration)
        jobs_start_time[job_id] = max(jobs_start_time[job_id], start_value + duration)
        num_exec_all += 1
        step_task_indices.append((job_id, task_id))

    print(f'************ 3. Final [Curr Breakdown? {curr_is_breakdown}, idx {curr_breakdown_idx}] ************')
    
    breakdown_state_change = curr_is_breakdown != breakdown_state[0]
    if breakdown_state_change:
        print(f'************ 4. Notice! Breakdown state change: {breakdown_state[0]} -> {curr_is_breakdown} ************')

    breakdown_state = (curr_is_breakdown, curr_breakdown_idx)

    # the following shouldn't matter
    start_loc = min(sel_i_task_loc)
    while start_loc < num_tasks and sorted_task_indices[start_loc] in step_task_indices:
        start_loc += 1
    print(f'Execute {num_exec_all} tasks (out of {len(chosen_task_indices)})')
    print('next start loc', start_loc)

    return start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time, breakdown_state, breakdown_state_change


def rolling_horizon(n_machines, n_jobs, jobs_data, window, step, time_limit, stop_search_time, 
                    oracle_time_limit=30, oracle_stop_search_time=3, 
                    action_type='default', do_warm_start=False, best_criterion='solve_time',
                    first_frac=0.5, random_p=0.8, num_oracle_trials=1, 
                    train_data_dir='train_data_dir/train_data', data_idx=1,
                    model=None, model_decode_strategy='argmax', model_th=0.5, model_topk_th=0.5, 
                    include_model_time=False, n_cpus=1, print_str='',
                    breakdown_times=None, num_machine_breakdown_p=0.2,  
                    first_breakdown_buffer_lb=50, first_breakdown_buffer_ub=150,
                    machine_breakdown_duration_lb=100, machine_breakdown_duration_ub=100,
                    breakdown_buffer_lb=400, breakdown_buffer_ub=600):

    oracle_time_limit, oracle_stop_search_time = time_limit, stop_search_time
    if action_type == 'default':
        do_warm_start = False
    
    sorted_task_indices = sort_by_task_order(jobs_data)

    #### tracking solutions
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    jobs_solution_sel = {}  
    machines_assignment_sel = {} 

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

    # generate a set of times where a subset of machine breakdown: start time, end time, machine id
    if breakdown_times is None:
        breakdown_times = generate_machine_breakdown(jobs_data, n_machines, 
                                                    num_machine_breakdown_p=num_machine_breakdown_p,
                                                    first_breakdown_buffer_lb=first_breakdown_buffer_lb,
                                                    first_breakdown_buffer_ub=first_breakdown_buffer_ub,
                                                    machine_breakdown_duration_lb=machine_breakdown_duration_lb,
                                                    machine_breakdown_duration_ub=machine_breakdown_duration_ub,
                                                    breakdown_buffer_lb=breakdown_buffer_lb,
                                                    breakdown_buffer_ub=breakdown_buffer_ub)
    breakdown_state = (False, 0)  # whether breakdown, next index to check

    i_iter = 0

    reduced_tasks = reduced_tasks_prev = []
    breakdown_state_change = False
    while len(jobs_solution) < num_tasks:
        sel_task_indices, sel_i_task_loc = select_window(start_loc, num_tasks, sorted_task_indices, jobs_solution, window)
        
        reduced_tasks_prev = reduced_tasks
        jobs_data_sel, ignored_tasks, reduced_tasks = select_jss_data_machine_breakdown(
            sel_task_indices, jobs_data, breakdown_times, breakdown_state)

        sel_task_indices_filter = []
        sel_i_task_loc_filter = []

        for i, task_idx in enumerate(sel_task_indices):
            if task_idx not in ignored_tasks:
                sel_task_indices_filter.append(task_idx)
                sel_i_task_loc_filter.append(sel_i_task_loc[i])

        # if empty move to the machine available time after the breakdown and recompute
        if len(sel_task_indices_filter) == 0:
            print('No tasks to schedule after breakdown. Move to the next breakdown time.')
            # for all job move to the end of the breakdown time
            breakdown_end_time = breakdown_times[breakdown_state[1]][0][1]
            for machine in range(n_machines):
                machines_start_time[machine] = max(breakdown_end_time, machines_start_time[machine])
            for job_id in range(n_jobs):
                jobs_start_time[job_id] = max(breakdown_end_time, jobs_start_time[job_id])

            breakdown_state = (False, breakdown_state[1] + 1)

            jobs_data_sel, ignored_tasks, reduced_tasks = select_jss_data_machine_breakdown(
                sel_task_indices, jobs_data, breakdown_times, breakdown_state)
            
            sel_task_indices_filter = []
            sel_i_task_loc_filter = []

            for i, task_idx in enumerate(sel_task_indices):
                if task_idx not in ignored_tasks:
                    sel_task_indices_filter.append(task_idx)
                    sel_i_task_loc_filter.append(sel_i_task_loc[i])

        sel_task_indices = sel_task_indices_filter
        sel_i_task_loc = sel_i_task_loc_filter
        
        if len(jobs_solution) > 0 and action_type != 'default':
            overlapping_task_indices = [task for task in prev_sel_task_indices if task in sel_task_indices]
            
            is_breakdown = False
            breakdown_machines = []
            breakdown_operations = []
            recovered_operations = []
            is_breakdown, breakdown_idx = breakdown_state[0], breakdown_state[1]
            if is_breakdown:
                # remove all overlapping task indices if previous solution is scheduled on a breakdown machine
                overlapping_task_indices = [task for task in overlapping_task_indices
                                            if jobs_solution_sel[task][1] not in breakdown_times[breakdown_idx][1]]
                breakdown_machines = breakdown_times[breakdown_idx][-1]
                breakdown_operations = [task in overlapping_task_indices and 
                                        jobs_solution_sel[task][1] in breakdown_times[breakdown_idx][1]
                                        for task in sel_task_indices]               
            elif breakdown_state_change and breakdown_idx > 0:  # just recovered operations
                recovered_operations = [task in reduced_tasks_prev for task in sel_task_indices]
    
            print(f'# Overlapping tasks = {len(overlapping_task_indices)}, out of {len(sel_task_indices)}')
            if len(overlapping_task_indices) == 0:
                overlapping_jobs_solution = {}
            else:
                get_overlapping_jobs_solution_input = [
                    action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                    machines_start_time, jobs_start_time, jobs_solution_sel, overlapping_task_indices, 
                    n_cpus, first_frac, random_p, model, model_decode_strategy, model_th, model_topk_th, model_time, 
                    training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time,
                    is_breakdown, breakdown_machines, breakdown_operations, recovered_operations
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

        start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time, breakdown_state, breakdown_state_change = \
            exec_step_machine_breakdown(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
                    jobs_solution, jobs_solution_sel,  machines_assignment, machines_assignment_sel, 
                    machines_start_time, jobs_start_time, n_machines, step, 
                    breakdown_times=breakdown_times, breakdown_state=breakdown_state)

        prev_sel_task_indices = copy.deepcopy(sel_task_indices)   # for next iter
        solve_time += solve_time_sel
        num_solves += 1

        cur_total_makespan = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] for job_id, task_id in jobs_solution])
        if not validate_jss_sol(jobs_data, n_machines, jobs_solution, machines_assignment, cur_total_makespan, 
                                machine_breakdown=True):
            print('Step: Solution is invalid!')
            pdb.set_trace()
        else:
            print('Step: Solution is valid!')

        next_machine_breakdown_time = breakdown_times[breakdown_state[1]][0] if breakdown_state[1] < len(breakdown_times) else None
        machine_breakdown_str = ''
        if breakdown_state[0]:
            machine_breakdown_str = f'current breakdown time {next_machine_breakdown_time}'
        elif next_machine_breakdown_time is not None:
            machine_breakdown_str = f'next breakdown time {next_machine_breakdown_time}'
        else:
            machine_breakdown_str = 'no breakdown time left'

        curr_makespan = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] for job_id, task_id in jobs_solution])

        print(f'*************** [{i_iter}] exec_tasks {len(jobs_solution)} total num tasks {num_tasks} '
              f'window range [{sel_i_task_loc[0]}, {sel_i_task_loc[-1]+1}] | current makespan {curr_makespan} | '
              f'{machine_breakdown_str} ***************\n')

        i_iter += 1
  
    if include_model_time:
        solve_time += model_time

    if action_type == 'collect_data':
        store_dataset(training_data, data_dir=train_data_dir, data_name=f'train', data_index=data_idx)

    average_solve_time = solve_time / num_solves
    make_span = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] for job_id, task_id in sorted_task_indices])
      
    # check if final solution is valid
    if not validate_jss_sol(jobs_data, n_machines, jobs_solution, machines_assignment, make_span, 
                            check_final=True, machine_breakdown=True):
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


def generate_machine_breakdown(jobs_data, n_machines, num_machine_breakdown_p=0.2,
                               first_breakdown_buffer_lb=50, first_breakdown_buffer_ub=150,
                               machine_breakdown_duration_lb=100, machine_breakdown_duration_ub=100,
                               breakdown_buffer_lb=400, breakdown_buffer_ub=600):
    # horizon for the entire problem
    horizon = sum([max([jobs_data[job_id][task_id][alt_id][0] for alt_id in jobs_data[job_id][task_id]])
                    for job_id in jobs_data for task_id in jobs_data[job_id]])
    print('Horizon', horizon)

    # generate machine breakdown times 
    breakdown_end = 0
    breakdown_times = []
    while breakdown_end < horizon:
        if len(breakdown_times) == 0:
            breakdown_buffer = random.randint(first_breakdown_buffer_lb, first_breakdown_buffer_ub)
        else:
            breakdown_buffer = random.randint(breakdown_buffer_lb, breakdown_buffer_ub)
        breakdown_start = breakdown_end + breakdown_buffer
        breakdown_end = breakdown_start + random.randint(machine_breakdown_duration_lb, machine_breakdown_duration_ub)
        breakdown_machines = random.sample(range(n_machines), int(num_machine_breakdown_p * n_machines))
        breakdown_interval = (breakdown_start, breakdown_end)
        breakdown_times.append((breakdown_interval, breakdown_machines))
        print(f'[{len(breakdown_times)}] Breakdown interval', breakdown_interval, 'Breakdown machines', breakdown_machines)

    return breakdown_times


def run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, time_limit, stop_search_time,
            breakdown_data_dir=None, num_machine_breakdown_p=0.2,
            first_breakdown_buffer_lb=50, first_breakdown_buffer_ub=150,
            machine_breakdown_duration_lb=100, machine_breakdown_duration_ub=100,
            breakdown_buffer_lb=400, breakdown_buffer_ub=600):

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
    if breakdown_data_dir is None:
        breakdown_data_dir = jss_data_dir

    breakdown_data_loc = f'{breakdown_data_dir}/breakdown_{data_idx}.pkl'
    # assert os.path.exists(breakdown_data_loc), f'{breakdown_data_loc} does not exist!'
    if not os.path.exists(breakdown_data_loc):
        # generate and save
        breakdown_times = generate_machine_breakdown(jobs_data, n_machines, num_machine_breakdown_p=num_machine_breakdown_p,
                                                    first_breakdown_buffer_lb=first_breakdown_buffer_lb,
                                                    first_breakdown_buffer_ub=first_breakdown_buffer_ub,
                                                    machine_breakdown_duration_lb=machine_breakdown_duration_lb,
                                                    machine_breakdown_duration_ub=machine_breakdown_duration_ub,
                                                    breakdown_buffer_lb=breakdown_buffer_lb,
                                                    breakdown_buffer_ub=breakdown_buffer_ub)
        os.makedirs(breakdown_data_dir, exist_ok=True)
        pickle.dump(breakdown_times, open(breakdown_data_loc, 'wb'))
    
    breakdown_times = pickle.load(open(breakdown_data_loc, 'rb'))

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

            if not 'breakdown_times' in params_dict:
                params_dict['breakdown_times'] = breakdown_times

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
    # parser = get_jss_argparser()
    parser = argparse.ArgumentParser(description='Flexible Job Shop Scheduling')
    # data directories
    parser.add_argument("--train_data_dir", default="debug_dir/train_data", type=str,
                        help="save train data directory")
    parser.add_argument("--jss_data_dir", default="train_data_dir/instance/j20-m10-t30_mix", 
                        type=str, help="save instance directory")
    
    parser.add_argument("--stats_dir", default="debug_dir/stats", type=str, help="stats directory")
    # parser.add_argument("--plots_dir", default="debug_dir/plots", type=str, help="stats directory")
    
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

    parser.add_argument("--breakdown_data_dir", default="train_data_dir/breakdown_data/j20-m10-t30_mix", type=str, help="breakdown data directory")
    parser.add_argument("--num_machine_breakdown_p", default=0.2, type=float, help="number of machines breakdown probability")
    parser.add_argument("--first_breakdown_buffer_lb", default=50, type=int, help="first breakdown buffer lower bound")
    parser.add_argument("--first_breakdown_buffer_ub", default=150, type=int, help="first breakdown buffer upper bound")
    parser.add_argument("--machine_breakdown_duration_lb", default=100, type=int, help="machine breakdown duration lower bound")
    parser.add_argument("--machine_breakdown_duration_ub", default=100, type=int, help="machine breakdown duration upper bound")
    parser.add_argument("--breakdown_buffer_lb", default=400, type=int, help="breakdown buffer lower bound")
    parser.add_argument("--breakdown_buffer_ub", default=600, type=int, help="breakdown buffer upper bound")

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
    
    ### RHO prameters
    window = args.window  
    step = args.step  

    jss_data_dir = args.jss_data_dir
    train_data_dir = args.train_data_dir
    data_idx = args.data_idx
    stats_dir = args.stats_dir

    for d in [jss_data_dir, train_data_dir, stats_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    ### collect training data for L-RHO
    collect_data_configs = [
        (f'collect_data',
          {'action_type': 'collect_data', 'num_oracle_trials': num_oracle_trials,
           'train_data_dir': train_data_dir, 'data_idx': data_idx,
           'oracle_time_limit': oracle_time_limit, 
           'oracle_stop_search_time': oracle_stop_search_time})
    ]

    ### default RHO and oracle RHO
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

    print('Start running...')
    run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, time_limit, stop_search_time,
            breakdown_data_dir=args.breakdown_data_dir, 
            num_machine_breakdown_p=args.num_machine_breakdown_p,
            first_breakdown_buffer_lb=args.first_breakdown_buffer_lb, first_breakdown_buffer_ub=args.first_breakdown_buffer_ub,
            machine_breakdown_duration_lb=args.machine_breakdown_duration_lb, machine_breakdown_duration_ub=args.machine_breakdown_duration_ub,
            breakdown_buffer_lb=args.breakdown_buffer_lb, breakdown_buffer_ub=args.breakdown_buffer_ub)

    

    