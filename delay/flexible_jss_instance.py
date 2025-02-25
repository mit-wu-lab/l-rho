import itertools
import random
import pickle
import os
import argparse
from collections import defaultdict
import numpy as np
import argparse

# simple FJSP instance: show it here for illustration of the format of the instance
def get_simple_flexible_jobs_data():
    jobs_data = {0:  # task = (processing_time, machine_id)
        {  # Job 0
            0: {0: (3, 0), 1: (1, 1), 2: (5, 2)},  # task 0 with 3 alternatives
            1: {0: (2, 0), 1: (4, 1), 2: (6, 2)},  # task 1 with 3 alternatives
            2: {0: (2, 0), 1: (3, 1), 2: (1, 2)},  # task 2 with 3 alternatives
        },
        1: {  # Job 1
            0: {0: (2, 0), 1: (3, 1), 2: (4, 2)},
            1: {0: (1, 0), 1: (5, 1), 2: (4, 2)},
            2: {0: (2, 0), 1: (1, 1), 2: (4, 2)},
        },
        2: {0: {0: (2, 0), 1: (1, 1), 2: (4, 2)},  # Job 2
            1: {0: (2, 0), 1: (3, 1), 2: (4, 2)},
            2: {0: (3, 0), 1: (1, 1), 2: (5, 2)},
            },
    }

    n_machines = 3

    scheduled_starts = {0: {0: 0, 1: 2, 2: 4},
                        1: {0: 0, 1: 2, 2: 3},
                        2: {0: 0, 1: 3, 2: 4}}

    return jobs_data, n_machines, scheduled_starts

#### duration
def get_l_low_high_machine(n_machines, n_jobs, n_tasks_per_job, p_deploy, l_low, l_high, 
                           machine_dist, change_freq, verbose=False):
    def l_low_high_update_function(option=0):
        if option == 0:
            choice_idx = np.random.choice(range(4))
            l_low_options = [l_low-2, l_low, l_low+2, l_low+4]
            l_low_options = [np.clip(v, 2, 15) for v in l_low_options]
            cur_l_low = l_low_options[choice_idx]
            l_offset = l_high - l_low
            l_offset_list = [l_offset-6, l_offset-3, l_offset, l_offset+3, l_offset+6]
            l_offset_list = [np.clip(v, 5, 30) for v in l_offset_list]
            cur_l_high = cur_l_low + np.random.choice(l_offset_list)
        return cur_l_low, cur_l_high

    l_low_machine, l_high_machine = [], []
    l_low_machine_per_task, l_high_machine_per_task = [], []
    if machine_dist == 'uniform':
        l_low_machine = [l_low for _ in range(n_machines)]
        l_high_machine = [l_high for _ in range(n_machines)]
    elif machine_dist == 'adaptive':
        l_low_machine, l_high_machine = list(zip(*[l_low_high_update_function() for _ in range(n_machines)]))

        if verbose:
            for i_machine in range(n_machines):
                print(f'machine {i_machine}: l_low {l_low_machine[i_machine]}, l_high {l_high_machine[i_machine]}')
    elif machine_dist == 'adaptive_task':
        l_low_machine_per_task = [[l_low for _ in range(n_machines)] for _ in range(n_tasks_per_job)]
        l_high_machine_per_task = [[l_high for _ in range(n_machines)] for _ in range(n_tasks_per_job)]
        change_num_center = n_tasks_per_job // change_freq
        change_points = sorted(np.random.choice(range(3, n_tasks_per_job), change_num_center, replace=False))
        if verbose:
            print('change_points', change_points)
        for i_machine in range(n_machines):
            for i_task in range(1, n_tasks_per_job):
                prev_l_low = l_low_machine_per_task[i_task-1][i_machine]
                prev_l_high = l_high_machine_per_task[i_task-1][i_machine]
                if i_task == 0 or i_task in change_points:
                    l_low_machine_i, l_high_machine_i = l_low_high_update_function()
                else:
                    l_low_machine_i, l_high_machine_i = prev_l_low, prev_l_high
                l_low_machine_per_task[i_task][i_machine] = l_low_machine_i
                l_high_machine_per_task[i_task][i_machine] = l_high_machine_i
    elif machine_dist == 'adaptive_task_machine':
        l_low_machine_per_task = [[l_low for _ in range(n_machines)] for _ in range(n_tasks_per_job)]
        l_high_machine_per_task = [[l_high for _ in range(n_machines)] for _ in range(n_tasks_per_job)]
        for i_machine in range(n_machines):
            change_num_center = n_tasks_per_job // change_freq
            change_points_i = sorted(np.random.choice(range(3, n_tasks_per_job), change_num_center, replace=False))
            if verbose:
                print(f'change_points low high {i_machine}', change_points_i)
            for i_task in range(1, n_tasks_per_job):
                prev_l_low = l_low_machine_per_task[i_task-1][i_machine]
                prev_l_high = l_high_machine_per_task[i_task-1][i_machine]
                if i_task == 0 or i_task in change_points_i:
                    l_low_machine_i, l_high_machine_i = l_low_high_update_function()
                else:
                    l_low_machine_i, l_high_machine_i = prev_l_low, prev_l_high

                l_low_machine_per_task[i_task][i_machine] = l_low_machine_i
                l_high_machine_per_task[i_task][i_machine] = l_high_machine_i
    else:
        print(f"{machine_dist} not implemented")
        raise NotImplementedError

    return l_low_machine, l_high_machine, l_low_machine_per_task, l_high_machine_per_task


#### start time
def get_interval_center_list(n_machines, n_jobs, n_tasks_per_job, p_deploy, l_low, l_high, 
                             task_interval_high, data_dist, change_freq,
                             change_low, change_high, verbose=False):
    def interval_update_function(prev_interval_center, option=0):
        if option == 0:
            task_interval_high_list = [task_interval_high-10, task_interval_high-5, 
                                       task_interval_high, task_interval_high+5, 
                                       task_interval_high+10]
            # 15, 25
            task_interval_high_list = [np.clip(v, 5, 35) for v in task_interval_high_list]
            p_high = [0.1, 0.15, 0.15, 0.3, 0.3]
            p_low = [0.3, 0.3, 0.15, 0.15, 0.1]

            if prev_interval_center < task_interval_high:
                cur_interval_center = np.random.choice(task_interval_high_list, p=p_high)
            else:
                cur_interval_center = np.random.choice(task_interval_high_list, p=p_low) 
        elif option == 1:
            cur_interval_center = np.random.choice([prev_interval_center + 5 * i for i in range(-1, 2)])
        else:
            change_low_i = max(change_low, -prev_interval_center)
            cur_interval_center = prev_interval_center + np.random.choice([change_low_i, change_high])
            cur_interval_center = np.clip(cur_interval_center, 5, 35)
        return cur_interval_center
    
    interval_center_list, interval_center_list_per_job = [], []
    if data_dist == 'uniform':
        interval_center_list = [task_interval_high for _ in range(n_tasks_per_job)]
    elif data_dist == 'adaptive':
        interval_center_list = [task_interval_high]
        change_num_center = n_tasks_per_job // change_freq
        change_points = sorted(np.random.choice(range(3, n_tasks_per_job), change_num_center, replace=False))
        if verbose:
            print('change_points', change_points)
        for i_task in range(1, n_tasks_per_job):
            if i_task in change_points:
                task_interval_high_i = interval_update_function(interval_center_list[-1])
            else:
                task_interval_high_i = interval_center_list[-1]
            interval_center_list.append(task_interval_high_i)
        if verbose:
            print(interval_center_list)
    elif data_dist == 'adaptive_job':
        interval_center_list_per_job = [[] for _ in range(n_jobs)]
        change_num_center = n_tasks_per_job // change_freq
        change_points = sorted(np.random.choice(range(3, n_tasks_per_job), 
                                                      change_num_center, replace=False))
        if verbose:
            print(f'change_points', change_points)
        for i_job in range(n_jobs):
            interval_center_list_i = []
            interval_center_list_i.append(task_interval_high)
            for i_task in range(n_tasks_per_job)[1:]:
                if i_task in change_points:
                    task_interval_high_i = interval_update_function(interval_center_list_i[-1])
                else:
                    task_interval_high_i = interval_center_list_i[-1]
                interval_center_list_i.append(task_interval_high_i)
            interval_center_list_per_job[i_job] = interval_center_list_i
    elif data_dist == 'adaptive_job_task':
        interval_center_list_per_job = [[] for _ in range(n_jobs)]
        change_num_center = n_tasks_per_job // change_freq
        for i_job in range(n_jobs):
            change_points_i = sorted(np.random.choice(range(3, n_tasks_per_job), 
                                                      change_num_center, replace=False))
            if verbose:
                print(f'change_points interval {i_job}', change_points_i)
            interval_center_list_i = []
            interval_center_list_i.append(task_interval_high)
            for i_task in range(n_tasks_per_job)[1:]:
                if i_task in change_points_i:
                    task_interval_high_i = interval_update_function(interval_center_list_i[-1])
                else:
                    task_interval_high_i = interval_center_list_i[-1]
                interval_center_list_i.append(task_interval_high_i)
            interval_center_list_per_job[i_job] = interval_center_list_i

    else:
        print(f"{data_dist} not implemented")
        raise NotImplementedError
    return interval_center_list, interval_center_list_per_job


def get_flexible_jss_data(n_machines, n_jobs, n_tasks_per_job, p_deploy, l_low, l_high, 
                          machine_dist='uniform', start_time_high=50, task_interval_high=20, 
                          data_dist='uniform', change_freq=5, 
                          change_low=-5, change_high=5, post_process=False, cutoff_percent=0.8,
                          optim_option='start_delay',
                          l_low_end=2, l_high_end=5):
    
    if cutoff_percent < 1-1e-5:
        n_tasks_per_job = int(np.ceil(n_tasks_per_job / cutoff_percent))

    jobs_data = {}
    
    l_low_machine, l_high_machine, l_low_machine_per_task, l_high_machine_per_task = get_l_low_high_machine(
        n_machines, n_jobs, n_tasks_per_job, p_deploy, l_low, l_high, machine_dist, change_freq)
    
    for i_job in range(n_jobs):
        job_data = {}
        for i_task in range(n_tasks_per_job):
            task_data = {}
            sel_machines = [i_machine for i_machine in range(n_machines) if np.random.rand() < p_deploy]
            
            if len(sel_machines) == 0:
                sel_machines = [np.random.choice(n_machines)]

            for alt_id, i_machine in enumerate(sel_machines):
                if 'adaptive_task' in machine_dist:
                    l_low_machine = l_low_machine_per_task[i_task]
                    l_high_machine = l_high_machine_per_task[i_task]

                l_low_i, l_high_i = l_low_machine[i_machine], l_high_machine[i_machine]
                task_data[alt_id] = (np.random.randint(l_low_i, l_high_i), i_machine)

            job_data[i_task] = task_data

        jobs_data[i_job] = dict(sorted(job_data.items(), key=lambda x: x[0]))

    scheduled_starts = {}
    scheduled_ends = {}
    interval_center_list, interval_center_list_per_job = get_interval_center_list(
        n_machines, n_jobs, n_tasks_per_job, p_deploy, l_low, l_high, 
        task_interval_high, data_dist, change_freq, change_low, change_high)
    
    for job_id, job in jobs_data.items():
        if len(job) == 0: continue
        start = int(np.random.rand() * start_time_high)
        job_start = {}
        job_end = {}
        for task_id in job:
            if 'adaptive_job' in data_dist:
                interval_center_list = interval_center_list_per_job[job_id]
            interval_center = interval_center_list[task_id]
            # start = start + np.random.randint(interval_center-interval_width, interval_center+interval_width+1)
            start = start + np.random.randint(0, interval_center+1)
            job_start[task_id] = start

            if l_low_end < l_high_end:
                job_end[task_id] = start + max(np.random.randint(l_low_end, l_high_end), 0)
            else:
                job_end[task_id] = start
        scheduled_starts[job_id] = job_start
        scheduled_ends[job_id] = job_end
        
    ''' post process data: remove tasks that are scheduled too late'''
    if post_process:
        all_start_times = [scheduled_starts[job_id][task_id] for job_id in scheduled_starts
                                                             for task_id in scheduled_starts[job_id]]
        cutoff_start_time =  np.quantile(all_start_times, cutoff_percent)
        jobs_data = {job_id: {task_id: task for task_id, task in job.items() 
                              if scheduled_starts[job_id][task_id] < cutoff_start_time}
                     for job_id, job in jobs_data.items()}
        scheduled_starts = {job_id: {task_id: start_time for task_id, start_time in job.items() 
                                     if task_id in jobs_data[job_id]}
                            for job_id, job in scheduled_starts.items()}
        
        scheduled_ends = {job_id: {task_id: end_time for task_id, end_time in job.items() 
                                   if task_id in jobs_data[job_id]}
                            for job_id, job in scheduled_ends.items()}
        
        
    if 'end' in optim_option:
        return jobs_data, n_machines, scheduled_starts, scheduled_ends
    else:
        return jobs_data, n_machines, scheduled_starts


def get_flexible_jss_data_from_args(args):
    return get_flexible_jss_data(n_machines=args.n_machines, n_jobs=args.n_jobs, n_tasks_per_job=args.n_tasks_per_job, 
                                 p_deploy=args.p_deploy, l_low=args.l_low, l_high=args.l_high, 
                                 machine_dist=args.machine_dist, start_time_high=args.start_time_high, 
                                 task_interval_high=args.task_interval_high, data_dist=args.data_dist, 
                                 change_freq=args.change_freq, change_low=args.change_low, change_high=args.change_high, 
                                 post_process=not args.not_post_process, cutoff_percent=args.cutoff_percent,
                                 optim_option=args.optim_option,
                                 l_low_end=args.l_low_end, l_high_end=args.l_high_end)
    
def get_jss_name(n_machines=10, n_jobs=10, n_tasks_per_job=30, p_deploy=1, l_low=5, l_high=20, 
                 start_time_high=0, task_interval_high=20, machine_dist='uniform', data_dist='uniform', 
                 change_freq=5, change_low=-5, change_high=5, post_process=False, cutoff_percent=0.8,
                 optim_option='start_delay', l_low_end=2, l_high_end=5):
    post_process_str = f'p-ct{cutoff_percent}' if post_process else 'np'

    jss_data_name = f'm{n_machines}-j{n_jobs}-t{n_tasks_per_job}-p{p_deploy}-lo{l_low}-hi{l_high}-s{start_time_high}-h{task_interval_high}-{post_process_str}'
    
    if machine_dist == 'adaptive':
        jss_data_name += f'-madapt'
    elif machine_dist == 'adaptive_task':
        jss_data_name += f'-madapt-task-cf{change_freq}'
    elif machine_dist == 'adaptive_task_machine':
        jss_data_name += f'-madapt-task-machine-cf{change_freq}'

    if data_dist == 'adaptive':
        jss_data_name += f'-dadapt-cf{change_freq}-cl{change_low}-ch{change_high}'
    elif data_dist == 'adaptive_job':
        jss_data_name += f'-dadapt-job-cf{change_freq}-cl{change_low}-ch{change_high}'
    elif data_dist == 'adaptive_job_task':
        jss_data_name += f'-dadapt-job-task-cf{change_freq}-cl{change_low}-ch{change_high}'
    
    if optim_option != 'start_delay':
        jss_data_name += f'-optimtype_{optim_option}'
    if 'end' in optim_option:
        jss_data_name += f'-tlo{l_low_end}-thi{l_high_end}'
        
    return jss_data_name    


def get_jss_name_from_args(args):
    return get_jss_name(n_machines=args.n_machines, n_jobs=args.n_jobs, n_tasks_per_job=args.n_tasks_per_job, 
                        p_deploy=args.p_deploy, l_low=args.l_low, l_high=args.l_high, 
                        start_time_high=args.start_time_high, task_interval_high=args.task_interval_high, 
                        machine_dist=args.machine_dist, data_dist=args.data_dist, 
                        change_freq=args.change_freq, change_low=args.change_low, change_high=args.change_high, 
                        post_process=not args.not_post_process, cutoff_percent=args.cutoff_percent,
                        optim_option=args.optim_option, 
                        l_low_end=args.l_low_end, l_high_end=args.l_high_end)

def get_data_dir_name(jss_data_dir_name, args):
    dir_name = f'{jss_data_dir_name}-w{args.window}-s{args.step}-t{args.time_limit}-st{args.stop_search_time}'
    if args.perturb_data:
        dir_name += f'-perturbo{args.perturb_option}-perturbp{args.perturb_p}'

    if args.exec_choice != 'default':
        dir_name += f'-exec{args.exec_choice}'
    return dir_name


def get_data_dir_name_param_search(jss_data_dir_name, args):
    dir_name = jss_data_dir_name
    if args.perturb_data:
        dir_name += f'-perturbo{args.perturb_option}-perturbp{args.perturb_p}'

    if args.exec_choice != 'default':
        dir_name += f'-exec{args.exec_choice}'
    return dir_name


def print_jss_name(args):
    print(f'n_machines={args.n_machines}, n_jobs={args.n_jobs}, n_tasks_per_job={args.n_tasks_per_job}, p_deploy={args.p_deploy}, '
          f'l_low={args.l_low}, l_high={args.l_high}, start_time_high={args.start_time_high}, task_interval_high={args.task_interval_high}, '
          f'machine_dist={args.machine_dist}, data_dist={args.data_dist}, change_freq={args.change_freq}, change_low={args.change_low}, '
          f'change_high={args.change_high}, post_process={not args.not_post_process}, cutoff_percent={args.cutoff_percent}, '
          f'optim_option={args.optim_option}, '
          f'l_low_end={args.l_low_end}, l_high_end={args.l_high_end}')

def get_jss_argparser():
    parser = argparse.ArgumentParser()
    # JSS instance parameters
    parser.add_argument("--n_machines", default=30, type=int, help="number of machines")
    parser.add_argument("--n_jobs", default=30, type=int, help="number of jobs")
    parser.add_argument("--n_tasks_per_job", default=30, type=int, help="number of operations per job")
    parser.add_argument("--optim_option", default='start_delay', choices=['start_delay', 'start_and_end_delay'], help="FJSP objective")  
    parser.add_argument("--p_deploy", default=1, type=float)
    parser.add_argument("--l_low", default=5, type=int, help="related to the processing duration of operations, see code")
    parser.add_argument("--l_high", default=20, type=int, help="related to the processing duration of operations, see code")
    parser.add_argument("--start_time_high", default=0, type=int)
    parser.add_argument("--task_interval_high", default=15, type=int, 
                        help="related to the gaps in release time between two consecutive operations from the same job, see code")
    parser.add_argument("--machine_dist", default='adaptive_task_machine', type=str, 
                        choices=['uniform', 'adaptive', 'adaptive_task', 'adaptive_task_machine'])
    parser.add_argument("--data_dist", default='uniform', type=str, 
                        choices=['uniform', 'adaptive', 'adaptive_job', 'adaptive_job_task'])
    parser.add_argument("--change_freq", default=3, type=int)
    parser.add_argument("--change_low", default=-5, type=int)
    parser.add_argument("--change_high", default=5, type=int)
    parser.add_argument("--not_post_process", action='store_true')
    parser.add_argument("--cutoff_percent", default=0.8, type=float)
    parser.add_argument("--l_low_end", default=0, type=int)
    parser.add_argument("--l_high_end", default=30, type=int)

    return parser

if __name__ == '__main__':
    parser = get_jss_argparser()
    parser.add_argument("--data_loc", default='train_data_dir/instance', type=str)
    parser.add_argument("--data_index_start", default=0, type=int, help="the start index of the collected data")
    parser.add_argument("--data_index_end", default=30, type=int, help="the end index of the collected data")     
   
    args = parser.parse_args()

    jss_data_name = get_jss_name_from_args(args)
    save_dir = os.path.join(args.data_loc, jss_data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data_idx in range(args.data_index_start, args.data_index_end):
        n_jobs = args.n_jobs
        if args.optim_option == 'start_delay':
            jobs_data, n_machines, scheduled_starts = get_flexible_jss_data_from_args(args)
            pickle.dump([jobs_data, n_machines, n_jobs, scheduled_starts], 
                        open(os.path.join(save_dir, f'data_{data_idx}.pkl'), 'wb'))
        else:
            jobs_data, n_machines, scheduled_starts, scheduled_ends = get_flexible_jss_data_from_args(args)
            pickle.dump([jobs_data, n_machines, n_jobs, scheduled_starts, scheduled_ends],
                        open(os.path.join(save_dir, f'data_{data_idx}.pkl'), 'wb'))
        
