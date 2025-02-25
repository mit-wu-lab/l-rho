import os
import random
import copy
import numpy as np
import collections
import numpy as np
from ortools.sat.python import cp_model
from flexible_jss_util import store_dataset
from collections import defaultdict
import pickle
import threading
import time
from flexible_jss_util import multiprocess

from makespan.flexible_jss_data_common import get_rollout_prediction
from flexible_jss_data_start import get_data, get_rollout_data
from flexible_jss_data_start_end import get_data as get_start_end_data
from flexible_jss_data_start_end import get_rollout_data as get_start_end_rollout_data
from flexible_jss_data_start_end_perturb import get_data as get_start_end_perturb_data
from flexible_jss_data_start_end_perturb import get_rollout_data as get_start_end_perturb_rollout_data

from flexible_jss_instance import get_jss_argparser, get_flexible_jss_data_from_args, print_jss_name
EPS = 1e-5


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__solution_time = []
        self.__solution_obj = []

        self.__last_improvement_time = float('inf')
        self.__best_objective = float('inf')

    def on_solution_callback(self):
        """Called at each new solution."""
        current_time = self.WallTime()
        current_obj = self.ObjectiveValue()
        self.__solution_count += 1
        self.__solution_time.append(current_time)
        self.__solution_obj.append(current_obj)

        # Check for improvement
        if current_obj < self.__best_objective:
            # Update best objective and last improvement time
            self.__best_objective = current_obj
            self.__last_improvement_time = time.time()

    @property
    def last_improvement_time(self):
        return self.__last_improvement_time
    
    @property
    def best_objective(self):
        return self.__best_objective


def validate_jss_sol(jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                     jobs_solution, machines_assignment, objective, 
                     machines_start_time=None, jobs_start_time=None, check_final=False,
                     optim_option='start_delay'):
    '''Check if the FJSP solution is valid'''
    # check if the size of the jobs solution equals the number of tasks in jobs_data
    valid = True
    num_sol_tasks = len(jobs_solution)
    num_data_tasks = sum([len(job) for job in jobs_data.values()])
    if check_final and num_sol_tasks != num_data_tasks:
        valid = False
        print(f'Number of tasks in jobs solution {num_sol_tasks} != number of tasks in jobs data {num_data_tasks}')
        # check which job's task is missing
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                if (job_id, task_id) not in jobs_solution:
                    print(f'Job {job_id} task {task_id} is missing in jobs solution')


    previous_end = None
    intervals_per_resources = defaultdict(list)
    starts = defaultdict(int)
    ends = defaultdict(int)
    for job_id, job in jobs_data.items():
        previous_end = None
        for task_id, task in job.items():
            if not check_final and (job_id, task_id) not in jobs_solution: break  # skip this job 
            # check if start time is correct
            start_value = jobs_solution[(job_id, task_id)][2]
            scheduled_start = scheduled_starts[job_id][task_id]
            if start_value < scheduled_start:
                valid = False
                print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < scheduled starts {scheduled_start}')
            
            if previous_end is not None and start_value < previous_end: 
                valid = False
                print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < previous_end {previous_end}')

            if jobs_start_time is not None and job_id in jobs_start_time:
                if start_value < jobs_start_time[job_id]:
                    valid = False
                    print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < job start time {jobs_start_time[job_id]}')

            # check if duration is correct, given the machine assignment
            select_alt = jobs_solution[(job_id, task_id)][0]
            machine = jobs_solution[(job_id, task_id)][1]
            duration = jobs_solution[(job_id, task_id)][3]
            machine_duration = task[select_alt][0]   # (duration, machine)
            if duration != machine_duration:
                valid = False
                print(f'Job {job_id} task {task_id} duration is not correct')


            if machines_start_time is not None and machine in machines_start_time:
                if start_value < machines_start_time[machine]:
                    valid = False
                    print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < machine start time {machines_start_time[machine]}')

            end_value = start_value + duration
            previous_end = end_value
            if scheduled_ends is not None and (job_id, task_id) in scheduled_ends:
                scheduled_end = scheduled_ends[job_id][task_id]
                if end_value > scheduled_end:
                    valid = False
                    print(f'Job {job_id} task {task_id} end time is not correct: end {end_value} > scheduled end {scheduled_end}')

            intervals_per_resources[machine].append([start_value, end_value])
            starts[(job_id, task_id)] = start_value
            ends[(job_id, task_id)] = end_value

    def interval_overlap(interval1, interval2):
        return interval1[1] > interval2[0] and interval2[1] > interval1[0]
    
    # Create machines constraints.
    for machine in intervals_per_resources:
        intervals = intervals_per_resources[machine]
        if len(intervals) > 1:
            # for each pair of intervals, check if they overlap
            for i in range(len(intervals)):
                for j in range(i+1, len(intervals)):
                    if interval_overlap(intervals[i], intervals[j]) or interval_overlap(intervals[j], intervals[i]):
                        valid = False
                        print(f'Machine {machine} intervals {intervals[i]} and {intervals[j]} overlap')

    # Minimize the schedule time
    if optim_option == 'start_delay':
        start_delay = {}
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                if not check_final and (job_id, task_id) not in jobs_solution: break  # skip this job
                start_delay[(job_id, task_id)] = max(starts[(job_id, task_id)] - scheduled_starts[job_id][task_id], 0)

                # check if the delay for the task is correct
                delay = start_delay[(job_id, task_id)]
                if abs(delay - jobs_solution[(job_id, task_id)][4]) > EPS:
                    valid = False
                    print(f'Job {job_id} task {task_id} delay is not correct: '
                          f'delay {delay} != solution delay {jobs_solution[(job_id, task_id)][4]}')
    elif optim_option == 'start_and_end_delay':
        start_delay = {}
        end_delay = {}
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                if not check_final and (job_id, task_id) not in jobs_solution: break  # skip this job
                end_delay_var = max(ends[(job_id, task_id)] - scheduled_ends[job_id][task_id], 0)
                end_delay[(job_id, task_id)] = end_delay_var
                start_delay[(job_id, task_id)] = max(starts[(job_id, task_id)] - scheduled_starts[job_id][task_id], 0)

                # check if the delay for the task is correct
                delay = end_delay[(job_id, task_id)] + start_delay[(job_id, task_id)] 
                if abs(delay - jobs_solution[(job_id, task_id)][4]) > EPS:
                    valid = False
                    print(f'Job {job_id} task {task_id} delay is not correct: '
                          f'delay {delay} != solution delay {jobs_solution[(job_id, task_id)][4]}')
    else:
        raise NotImplementedError

    print(f'Is solution valid? {valid}')       

    return valid


def flexible_jss(jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                 time_limit=-1, stop_search_time=5,
                 machines_start_time=None, jobs_start_time=None,
                 do_warm_start=False, jobs_solution_warm_start={},
                 optim_option='start_delay'):
    # each task has a scheduled start time; minimize the delay for all jobs
    """Solve a small flexible jobshop problem."""
    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job_id, job in jobs_data.items():
        for task_id, task in job.items():
            max_task_duration = 0
            for alt_id, alt in task.items():
                max_task_duration = max(max_task_duration, alt[0])
            horizon += max_task_duration
            horizon += scheduled_starts[job_id][task_id]

    offset = 0 if jobs_start_time is None else max(jobs_start_time)
    offset = offset if machines_start_time is None else max(offset, max(machines_start_time))
    horizon = horizon + offset  # horizon = 2000

    print("Horizon = %i" % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    ends = {}
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id, job in jobs_data.items():
        previous_end = None
        for task_id, task in job.items():
            min_duration = float('inf')
            max_duration = -float('inf')

            for alt_id, alt in task.items():
                min_duration = min(min_duration, alt[0])
                max_duration = max(max_duration, alt[0])

            # Create main interval for the task.
            suffix_name = "_j%i_t%i" % (job_id, task_id)

            st_task = jobs_start_time[job_id] if jobs_start_time is not None else 0
            st_task = max(st_task, scheduled_starts[job_id][task_id])
            
            start = model.NewIntVar(st_task, st_task + horizon, "start" + suffix_name)
            duration = model.NewIntVar(
                min_duration, max_duration, "duration" + suffix_name
            )
            end = model.NewIntVar(st_task, st_task + horizon, "end" + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start
            ends[(job_id, task_id)] = end

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            l_presences = []
            for alt_id, alt in task.items():
                alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                l_presence = model.NewBoolVar("presence" + alt_suffix)

                l_machine = alt[1]
                l_st_task = 0 if machines_start_time is None else machines_start_time[l_machine]
                l_st_task = max(l_st_task, scheduled_starts[job_id][task_id])
                
                l_start = model.NewIntVar(l_st_task, l_st_task + horizon, "start" + alt_suffix)
                l_duration = alt[0]

                l_end = model.NewIntVar(l_st_task, l_st_task + horizon, "end" + alt_suffix)

                l_interval = model.NewOptionalIntervalVar(
                    l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                )

                l_presences.append(l_presence)

                # Link the primary/global variables with the local ones.
                model.Add(start == l_start).OnlyEnforceIf(l_presence)
                model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                model.Add(end == l_end).OnlyEnforceIf(l_presence)

                # Add the local interval to the right machine.
                intervals_per_resources[l_machine].append(l_interval)

                # Store the presences for the solution.
                presences[(job_id, task_id, alt_id)] = l_presence

                if do_warm_start and (job_id, task_id) in jobs_solution_warm_start:
                    # add hint
                    if jobs_solution_warm_start[(job_id, task_id)][0] == alt_id:
                        model.AddHint(l_presence, 1)
                    else:
                        model.AddHint(l_presence, 0)

            # Select exactly one presence variable.
            model.AddExactlyOne(l_presences)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in range(n_machines):
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Minimize the schedule time
    if optim_option == 'start_delay':
        print('.................Objective is total start delay.................')
        model.Minimize(sum([starts[(job_id, task_id)] - scheduled_starts[job_id][task_id]
                            for job_id, job in jobs_data.items() for task_id, task in job.items()]))
    elif optim_option == 'start_and_end_delay':
        print('.................Objective is total start and end delay.................')
        start_delay = {}
        end_delay = {}
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                end_var = model.NewIntVar(0, horizon, f"end_{job_id}_{task_id}")
                scheduled_end = scheduled_ends[job_id][task_id] 
                model.Add(end_var >= ends[(job_id, task_id)] - scheduled_end)
                end_delay[(job_id, task_id)] = end_var
                start_delay[(job_id, task_id)] = starts[(job_id, task_id)] - scheduled_starts[job_id][task_id]
        model.Minimize(sum(list(start_delay.values())) + sum(list(end_delay.values())))
    else:
        raise NotImplementedError
    
    solver = cp_model.CpSolver()
    if time_limit > 0:
        solver.parameters.max_time_in_seconds = time_limit

    if stop_search_time >= time_limit:
        status = solver.Solve(model)
        model.status = status
    else:
        def solve_model(model, solver, callback):
            # Solve model.
            status = solver.Solve(model, callback)
            model.status = status

        solution_callback = SolutionCallback()
        solver_thread = threading.Thread(target=solve_model, args=(model, solver, solution_callback))
        solver_thread.start()

        # Monitor in main thread
        while solver_thread.is_alive():
            time.sleep(0.1)  # Short sleep to prevent excessive CPU usage
            if time.time() - solution_callback.last_improvement_time > stop_search_time:
                solution_callback.StopSearch()
                print(f"No improvement in {stop_search_time} seconds, stopping search.")

    
    status = model.status
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration delay")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        start_delay = 0
        end_delay = 0
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                start_value = solver.Value(starts[(job_id, task_id)])
                end_value = solver.Value(ends[(job_id, task_id)])
                machine, duration, selected_alt, delay = -1, -1, -1, float('inf')
                for alt_id, alt in task.items():
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = alt[0]
                        machine = alt[1]
                        selected_alt = alt_id
                        if optim_option == 'start_delay':  # start delay
                            delay = start_value - scheduled_starts[job_id][task_id]
                            start_delay += delay
                        else:  # start and end delay
                            start_delay_val = start_value - scheduled_starts[job_id][task_id]
                            end_delay_val = end_value - scheduled_ends[job_id][task_id]
                            end_delay_val = max(end_delay_val, 0)
                            start_delay += start_delay_val
                            end_delay += end_delay_val
                            delay = start_delay_val + end_delay_val
                            
                jobs_solution[(job_id, task_id)] = (selected_alt, machine, start_value, duration, delay)
                machines_assignment[machine].append(
                    assigned_task_type(
                        start=start_value,
                        job=job_id,
                        index=task_id,
                        duration=duration,
                        delay=delay
                    )
                )

        for machine in machines_assignment:
            machines_assignment[machine].sort()
        objective = solver.ObjectiveValue()
    else:
        print("No solution found.")
        objective = float('inf')

    print("Solve status: %s" % solver.StatusName(status))
    if status == cp_model.OPTIMAL:
        print("Optimal objective value: %i" % objective)
    elif status == cp_model.FEASIBLE:
        print("Feasible objective value: %i" % objective)

    solve_time = solver.WallTime()
    print("Statistics")
    print("  - conflicts : %i" % solver.NumConflicts())
    print("  - branches  : %i" % solver.NumBranches())
    print("  - wall time : %f s" % solve_time)

    # check validity of solution
    valid = validate_jss_sol(jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                             jobs_solution, machines_assignment, objective, 
                             machines_start_time=machines_start_time, 
                             jobs_start_time=jobs_start_time,
                             optim_option=optim_option)
    if not valid:
        print("Inside Solve: Solution is not valid.")
    return jobs_solution, machines_assignment, solve_time, objective


def sort_scheduled_tasks(jobs_data, scheduled_starts):
    task_indices = [(job_id, task_id) for job_id, job in jobs_data.items() for task_id, task in job.items()]
    sorted_task_indices = sorted(task_indices, key=lambda x: scheduled_starts[x[0]][x[1]])

    return sorted_task_indices


def select_jss_data(sel_task_indices, jobs_data, scheduled_starts, jobs_solution={}):
    jobs_data_sel = {}
    scheduled_starts_sel = {}

    num_fix = 0
    num_not_fix = 0
    for job_id, task_id in sel_task_indices:
        if job_id not in jobs_data_sel:
            jobs_data_sel[job_id] = {}

        if job_id not in scheduled_starts_sel:
            scheduled_starts_sel[job_id] = {}

        if (job_id, task_id) in jobs_solution:
            alt_id = jobs_solution[(job_id, task_id)][0]
            jobs_data_alt = copy.deepcopy(jobs_data[job_id][task_id][alt_id])
            jobs_data_sel[job_id][task_id] = {alt_id: jobs_data_alt}
            num_fix += 1
        else:
            jobs_data_sel[job_id][task_id] = copy.deepcopy(jobs_data[job_id][task_id])
            num_not_fix += 1
        scheduled_starts_sel[job_id][task_id] = copy.deepcopy(scheduled_starts[job_id][task_id])
    
    for job_id in jobs_data_sel:
        # sort by task id to maintain precedence order
        jobs_data_sel[job_id] = dict(sorted(jobs_data_sel[job_id].items(), key=lambda x: x[0]))
        
    return jobs_data_sel, scheduled_starts_sel


def same_machine_assignment(job_solution1, job_solution2):
    # selected_alt, machine, start_value, duration, delay
    return job_solution1[1] == job_solution2[1]


def do_oracle_trial_i(args):
    jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, machines_start_time, jobs_start_time, \
        oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices, \
            overlapping_jobs_solution_all, optim_option, oracle_i = args

    jobs_solution_sel_oracle, machines_assignment_sel_oracle, solve_time_sel_i, objective_sel_i = \
        flexible_jss(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, 
                     time_limit=oracle_time_limit, stop_search_time=oracle_stop_search_time,
                     machines_start_time=machines_start_time, jobs_start_time=jobs_start_time,
                     do_warm_start=True, jobs_solution_warm_start=overlapping_jobs_solution_all,
                     optim_option=optim_option)
    
    overlapping_jobs_solution_i = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                              for job_id, task_id in overlapping_task_indices
                                              if same_machine_assignment(jobs_solution_sel[(job_id, task_id)],
                                                        jobs_solution_sel_oracle[(job_id, task_id)])}
    return overlapping_jobs_solution_i, objective_sel_i


def find_overlapping_jobs_solution_oracle(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, 
                                          machines_start_time, jobs_start_time,
                                          oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
                                          overlapping_jobs_solution_all, optim_option='start_delay', 
                                          num_oracle_trials=1, n_cpus=1):
    print('Solve JSS warm start (all_fix)')
    overlapping_jobs_solution = {}
    print('-----------------------------------')
    tasks = [(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, machines_start_time, jobs_start_time,
              oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
              overlapping_jobs_solution_all, optim_option, oracle_i)
             for oracle_i in range(num_oracle_trials)]
    if n_cpus > 1:
        results = multiprocess(do_oracle_trial_i, tasks, cpus=min(num_oracle_trials, n_cpus))
    else:
        results = [do_oracle_trial_i(task) for task in tasks]

    overlapping_jobs_solutions, objectives = zip(*results)
    most_overlap_idx = np.argmax([len(overlapping_jobs_solution_i) for overlapping_jobs_solution_i in overlapping_jobs_solutions])
    overlapping_jobs_solution = overlapping_jobs_solutions[most_overlap_idx]
    print('-----------------------------------')
    return overlapping_jobs_solution, objectives[most_overlap_idx]


def get_overlapping_jobs_solution(action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                sel_task_indices, scheduled_starts_sel, scheduled_ends_sel, machines_start_time,
                jobs_start_time, jobs_solution_sel, overlapping_task_indices, optim_option, perturb_data, 
                time_limit, stop_search_time, n_cpus, first_frac, 
                random_p, num_best_random, best_criterion, 
                model, model_decode_strategy, model_th, model_topk_th, model_time, 
                training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time):
    
    overlapping_jobs_solution_all = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                        for job_id, task_id in overlapping_task_indices}
    overlapping_machines_assignment_all = {machine: [task for task in machines_assignment_sel[machine]
                                                        if (task.job, task.index) in overlapping_task_indices]
                                            for machine in machines_assignment_sel}
    if action_type == 'fix_all':
        overlapping_jobs_solution = overlapping_jobs_solution_all
    elif action_type == 'oracle':
        overlapping_jobs_solution, _ = find_overlapping_jobs_solution_oracle(
            jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, 
            machines_start_time, jobs_start_time,
            oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
            overlapping_jobs_solution_all, optim_option=optim_option, 
            num_oracle_trials=num_oracle_trials, n_cpus=n_cpus)
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
        overlapping_jobs_solution, _ = find_overlapping_jobs_solution_oracle(
            jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, machines_start_time, jobs_start_time,
            oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
            overlapping_jobs_solution_all, optim_option=optim_option, 
            num_oracle_trials=num_oracle_trials, n_cpus=n_cpus)
        if 'end' in optim_option:  # total start and end delay
            get_data_fn = get_start_end_perturb_data if perturb_data else get_start_end_data
            data = get_data_fn(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel,
                                machines_start_time, jobs_start_time,
                                overlapping_task_indices, overlapping_jobs_solution_all,
                                overlapping_machines_assignment_all,
                                overlapping_jobs_solution)
        else:  # total start delay
            data = get_data(jobs_data_sel, n_machines, scheduled_starts_sel, machines_start_time, jobs_start_time,
                            overlapping_task_indices, overlapping_jobs_solution_all,
                            overlapping_machines_assignment_all,
                            overlapping_jobs_solution)
        training_data.append(data)
    elif action_type == 'model':
        model_start_time = time.time()
        if 'end' in optim_option:
            get_rollout_data_fn = get_start_end_perturb_rollout_data if perturb_data else get_start_end_rollout_data
            data = get_rollout_data_fn(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel,
                            machines_start_time, jobs_start_time,
                            overlapping_task_indices, overlapping_jobs_solution_all,
                            overlapping_machines_assignment_all)
            tasks_pred_fix = get_rollout_prediction(model, data, decode_strategy=model_decode_strategy, 
                                                threshold=model_th, topk_th=model_topk_th)
        else:
            data = get_rollout_data(jobs_data_sel, n_machines, scheduled_starts_sel, machines_start_time,
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
        raise NotImplementedError
    print(f'Number of overlapping solutions {len(overlapping_jobs_solution)} out of {len(overlapping_task_indices)}')
    return overlapping_jobs_solution, jobs_data_sel, scheduled_starts_sel, model_time

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

    return sel_task_indices, sel_i_task_loc


def select_jss_data_perturb(sel_task_indices, jobs_data, scheduled_starts, scheduled_ends,  
                            window, step, step_buffer=0, clean_frac=0.0, perturb_p=0.1, 
                            jobs_solution={}, perturb_jobs_data=None, 
                            perturb_scheduled_starts=None,
                            perturb_scheduled_ends=None,
                            perturb_start=False,
                            perturb_end=False):
    ''' information in the window may change from previous iteration'''
    print('Do perturb data ...')
    jobs_data_sel, scheduled_starts_sel = select_jss_data(sel_task_indices, jobs_data, scheduled_starts)
    perturb_jobs_data_sel = copy.deepcopy(jobs_data_sel)
    perturb_scheduled_starts_sel = copy.deepcopy(scheduled_starts_sel)
    perturb_scheduled_ends_sel = copy.deepcopy(scheduled_ends)  # full scheduled_ends, don't subsample

    # perturb later part of the tasks (only see noisy future after a certain point)
    perturb_task_indices = [task_idx for idx, task_idx in enumerate(sel_task_indices) 
                            if idx >= max(int(len(sel_task_indices) * clean_frac), step + step_buffer)]
    num_perturb_tasks = 0

    for job_id, task_id in perturb_task_indices:
        # perturb_duration
        if np.random.rand() < perturb_p:  # whether to perturb this task
            for alt_id, _ in perturb_jobs_data_sel[job_id][task_id].items():
                duration, machine = jobs_data[job_id][task_id][alt_id]                    
                perturbed_duration = np.clip(duration + np.random.randint(-5, 5), 3, 30) # 30
                perturb_jobs_data_sel[job_id][task_id][alt_id] = (perturbed_duration, machine)
        if perturb_start and np.random.rand() < perturb_p:
            start_min = scheduled_starts[job_id][task_id - 1] if task_id > 0 else 0
            start_max = scheduled_starts[job_id][task_id + 1] if task_id < len(scheduled_starts[job_id]) - 1 else float('inf')
            perturbed_start = scheduled_starts[job_id][task_id] + np.random.randint(-5, 5)
            perturb_scheduled_starts_sel[job_id][task_id] = np.clip(perturbed_start, start_min, start_max)
                
        if  perturb_end and job_id in scheduled_ends and task_id in scheduled_ends[job_id] and np.random.rand() < perturb_p:
            end_min, end_max = perturb_scheduled_starts_sel[job_id][task_id], 50
            perturbed_end = scheduled_ends[job_id][task_id] + np.random.randint(-15, 15)
            perturb_scheduled_ends_sel[job_id][task_id] = np.clip(perturbed_end, end_min, end_max)                        
        num_perturb_tasks += 1

    print(f'Perturb {num_perturb_tasks} tasks out of {len(sel_task_indices)} tasks')

    for job_id in perturb_jobs_data_sel:
        # sort by task id for each job to maintain precedence order; should be in the same order as jobs_data_sel
        perturb_jobs_data_sel[job_id] = dict(sorted(perturb_jobs_data_sel[job_id].items(), key=lambda x: x[0]))

    return perturb_jobs_data_sel, perturb_scheduled_starts_sel, perturb_scheduled_ends_sel



def exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, scheduled_starts,
              jobs_solution, jobs_solution_sel,  machines_assignment, machines_assignment_sel, 
              machines_start_time, jobs_start_time, n_machines, step, 
              perturb_data, jobs_data, optim_option, scheduled_ends, debug=True):

    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration delay")

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
        step_task_indices = [(job_id, task_id) for _, job_id, task_id in sel_task_indices_with_start[:step]]
       
        # the following shouldn't matter
        start_loc = min(sel_i_task_loc)
        while start_loc < num_tasks and sorted_task_indices[start_loc] in step_task_indices:
            start_loc += 1
        print('next start loc', start_loc)

    num_exec_sel, num_exec_all = 0, 0
    if perturb_data:     # different for perturbed instances
        print('Execute step for perturbed data...')
        org_delay = 0
        new_delay = 0
        step_task_indices_sorted_with_start = sorted([(jobs_solution_sel[(job_id, task_id)][2], job_id, task_id) 
                                                        for job_id, task_id in step_task_indices])
        step_task_indices_sorted = [(job_id, task_id) for _, job_id, task_id in step_task_indices_sorted_with_start]
        max_end_value_per_job = {job_id: -float('inf') for job_id in set([job_id for job_id, _ in sel_task_indices])}
        max_end_value_per_machine = {machine: -float('inf') for machine in range(n_machines)}
        for job_id, task_id in step_task_indices_sorted:
            # execute the task
            selected_alt, machine, start_value, duration_value, _ = jobs_solution_sel[(job_id, task_id)]
            # the task should be after all tasks that have been executed
            earliest_possible_time = max(max_end_value_per_job[job_id], max_end_value_per_machine[machine],
                                         scheduled_starts[job_id][task_id])
            start_value = max(start_value, earliest_possible_time)
            # clean duration, only available at execution time
            duration = jobs_data[job_id][task_id][selected_alt][0]

            end_value = start_value + duration
            max_end_value_per_job[job_id] = max(max_end_value_per_job[job_id], end_value)
            max_end_value_per_machine[machine] = max(max_end_value_per_machine[machine], end_value)

            if 'end' in optim_option:  # start and end delay
                delay = max(start_value - scheduled_starts[job_id][task_id], 0) + max(end_value - scheduled_ends[job_id][task_id], 0)
            else:  # start delay
                delay = max(start_value - scheduled_starts[job_id][task_id], 0)
            
            org_delay += jobs_solution_sel[(job_id, task_id)][4]
            new_delay += delay
            jobs_solution[(job_id, task_id)] = (selected_alt, machine, start_value, duration, delay)
                
            machines_assignment[machine].append(
                assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration, delay=delay)
            )

            machines_start_time[machine] = max(machines_start_time[machine], start_value + duration)
            jobs_start_time[job_id] = max(jobs_start_time[job_id], start_value + duration)
            if (job_id, task_id) in step_task_indices: num_exec_sel += 1
            num_exec_all += 1
    else:
        for job_id, task_id in step_task_indices:
            # execute the task
            selected_alt, machine, start_value, duration, delay = jobs_solution_sel[(job_id, task_id)]
            jobs_solution[(job_id, task_id)] = copy.deepcopy(jobs_solution_sel[(job_id, task_id)])
            machines_assignment[machine].append(
                assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration, delay=delay)
            )
            machines_start_time[machine] = max(machines_start_time[machine], start_value + duration)
            jobs_start_time[job_id] = max(jobs_start_time[job_id], start_value + duration)
            if (job_id, task_id) in step_task_indices: num_exec_sel += 1
            num_exec_all += 1

            
    return start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time


def rolling_horizon(n_machines, n_jobs, jobs_data, scheduled_starts, scheduled_ends, 
                    window, step, time_limit, stop_search_time, optim_option, 
                    action_type='default', do_warm_start=False, best_criterion='solve_time',
                    first_frac=0.5, random_p=0.8, num_best_random=10, 
                    num_oracle_trials=1, oracle_time_limit=30, oracle_stop_search_time=3,
                    train_data_dir='../train_data_dir', data_idx=1,
                    model=None, model_decode_strategy='argmax', model_th=0.5, 
                    model_topk_th=0.5, include_model_time=False, 
                    n_cpus=1, perturb_data=False, clean_frac=0.0, perturb_p=0.1):

    oracle_time_limit, oracle_stop_search_time = time_limit, stop_search_time
    if action_type in ['oracle_and_warm_start', 'model_and_warm_start']:
        do_warm_start = True
    elif action_type == 'default':
        do_warm_start = False
    if 'model' in action_type:
        model_time = 0

    sorted_task_indices = sort_scheduled_tasks(jobs_data, scheduled_starts)

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

    # perturb the entire dataset
    if perturb_data:
        print('Do perturb data!')
        perturb_jobs_data, perturb_scheduled_starts, perturb_scheduled_ends = select_jss_data_perturb(sorted_task_indices, 
                                                                            jobs_data, scheduled_starts, 
                                                                            scheduled_ends, window, 0, 
                                                                            clean_frac=0.0, perturb_p=perturb_p)

    print(f'*************** window {window} step {step} ***************')
    while len(jobs_solution) < num_tasks:
        sel_task_indices, sel_i_task_loc = select_window(start_loc, num_tasks, sorted_task_indices, jobs_solution, window)
        if perturb_data and len(jobs_solution) + len(sel_task_indices) < num_tasks:
            # apply perturbation on the data; keep a copy of the old data
            jobs_data_sel, scheduled_starts_sel, scheduled_ends_sel = select_jss_data_perturb(
                                                                        sel_task_indices, jobs_data, 
                                                                        scheduled_starts, scheduled_ends,
                                                                        window, step, 
                                                                        clean_frac=clean_frac, perturb_p=perturb_p, 
                                                                        perturb_jobs_data=perturb_jobs_data, 
                                                                        perturb_scheduled_starts=perturb_scheduled_starts,
                                                                        perturb_scheduled_ends=perturb_scheduled_ends)
        else:
            jobs_data_sel, scheduled_starts_sel = select_jss_data(sel_task_indices, jobs_data, scheduled_starts)
            scheduled_ends_sel = scheduled_ends
        
        '''current job, update jobs_data and scheduled_starts'''
        if len(jobs_solution) > 0 and action_type != 'default':
            overlapping_task_indices = [task for task in prev_sel_task_indices if task in sel_task_indices]
            print(f'# Overlapping tasks = {len(overlapping_task_indices)}, out of {len(sel_task_indices)}')
            get_overlapping_jobs_solution_input = [
                action_type, n_machines, jobs_data_sel, machines_assignment_sel, 
                sel_task_indices, scheduled_starts_sel, scheduled_ends_sel, 
                machines_start_time, jobs_start_time, jobs_solution_sel, overlapping_task_indices, 
                optim_option, perturb_data, 
                time_limit, stop_search_time, n_cpus, first_frac, 
                random_p, num_best_random, best_criterion, 
                model, model_decode_strategy, model_th, model_topk_th, model_time, 
                training_data, num_oracle_trials, oracle_time_limit, oracle_stop_search_time
            ]
            overlapping_jobs_solution, jobs_data_sel, scheduled_starts_sel, model_time = get_overlapping_jobs_solution(*get_overlapping_jobs_solution_input)
            sel_task_loc_dict = dict(zip(sel_task_indices, sel_i_task_loc))
            overlapping_task_loc = [sel_task_loc_dict[task] for task in overlapping_task_indices]
            print(f'{action_type.upper()} selected operation indices (within ['
                  f'{min(overlapping_task_loc) if len(overlapping_task_loc) > 0 else 0}, '
                  f'{max(overlapping_task_loc) if len(overlapping_task_loc) > 0 else 0}])', 
                    [sel_task_loc_dict[task] for task in overlapping_jobs_solution if task in sel_task_loc_dict])
                
            if not do_warm_start:  # hard filtering of the assignments (solution)
                jobs_data_sel, scheduled_starts_sel = select_jss_data(sel_task_indices, jobs_data_sel, scheduled_starts_sel,
                                                                      jobs_solution=overlapping_jobs_solution)
        else:  # default or first iteration
            overlapping_jobs_solution = {}
        
        jobs_solution_sel, machines_assignment_sel, solve_time_sel, objective_sel = flexible_jss(
            jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends_sel, 
            time_limit=time_limit, stop_search_time=stop_search_time,
            machines_start_time=machines_start_time, jobs_start_time=jobs_start_time,
            do_warm_start=do_warm_start, jobs_solution_warm_start=overlapping_jobs_solution, 
            optim_option=optim_option)        
        
        
        if objective_sel == float('inf'):
            print('Infeasible!!')

        start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time = \
            exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, scheduled_starts, 
                      jobs_solution, jobs_solution_sel, machines_assignment, machines_assignment_sel,
                      machines_start_time, jobs_start_time, n_machines, step, 
                      perturb_data, jobs_data, optim_option, scheduled_ends)
        prev_sel_task_indices = copy.deepcopy(sel_task_indices)   # for next iter
        solve_time += solve_time_sel
        num_solves += 1

        cur_total_delay = sum([jobs_solution[(job_id, task_id)][4] for job_id, task_id in sorted_task_indices 
                               if (job_id, task_id) in jobs_solution])
        if not validate_jss_sol(jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                     jobs_solution, machines_assignment, cur_total_delay, 
                     machines_start_time=None, jobs_start_time=None, optim_option=optim_option):
            print('Step: Solution is invalid!')
            # display all jobs solution
            print(f'************** My Saved Solution **************')
            for job_id, job in jobs_data.items():
                print(f'========= Job {job_id} =========')
                for task_id, task in job.items():
                    if (job_id, task_id) not in jobs_solution: break  # skip this job
                    _, _, start, duration, delay = jobs_solution[(job_id, task_id)]
                    end = start + duration
                    scheduled_start = scheduled_starts[job_id][task_id]
                    if job_id in scheduled_ends and task_id in scheduled_ends[job_id]:
                        scheduled_end = scheduled_ends[job_id][task_id] 
                    else: 
                        scheduled_end = None
                    print(f'Job {job_id} task {task_id} start {start} end {end} '
                        f'scheduled start {scheduled_start} {"scheduled end " + str(scheduled_end) if scheduled_end is not None else ""}')
        else:
            print('Step: Solution is valid!')

        print(f'*************** # exec operations {len(jobs_solution)} | total # operations {num_tasks}  ***************\n')

        
    if include_model_time:
        solve_time += model_time

    if action_type == 'collect_data':
        store_dataset(training_data, data_dir=train_data_dir, data_name=f'train', data_index=data_idx)

    average_solve_time = solve_time / num_solves
    total_delay = sum([jobs_solution[(job_id, task_id)][4] for job_id, task_id in sorted_task_indices])
    make_span = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] for job_id, task_id in sorted_task_indices])
    max_schedule_start = max([scheduled_starts[job_id][task_id] for job_id, task_id in sorted_task_indices])
    average_delay = total_delay / num_tasks

    # check if final solution is valid
    if not validate_jss_sol(jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                     jobs_solution, machines_assignment, total_delay, 
                     machines_start_time=None, jobs_start_time=None, 
                     check_final=False, optim_option=optim_option):
        print('Final solution is invalid!')
    else:
        print('Final solution is valid!')

    p_str = f'{action_type.upper()}{" Warm Start " if do_warm_start else " "}'
    
    print(f'\n********************************* Rolling Horizon {p_str} (objective {optim_option}) *********************************')
    print(f'total solve time {solve_time:.2f} average solve time {average_solve_time:.2f}, number of solves {num_solves}')
    print(f'total delay {total_delay:.2f} average delay {average_delay:.2f}, make span {make_span:.2f}, max schedule start {max_schedule_start:.2f}')
    print('All delays', [jobs_solution[(job_id, task_id)][4] for job_id, task_id in sorted_task_indices])
    
    if 'model' in action_type: 
        print(f'model time {model_time:.2f} ({"Include" if include_model_time else "Exclude"})')
    print('\n')

    return total_delay, average_delay, solve_time, average_solve_time


def run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, args, 
            optim_option='start_delay', perturb_data=False, perturb_p=0.1):
    data_loc = f'{jss_data_dir}/data_{data_idx}.pkl'
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

    if os.path.exists(f'{stats_dir}/stats_{data_idx}.pkl'):
        print(f'Load stats from {stats_dir}/stats_{data_idx}.pkl')
        total_delay_stats, average_delay_stats, solve_time_stats = pickle.load(open(f'{stats_dir}/stats_{data_idx}.pkl', 'rb'))

    else:
        total_delay_stats = defaultdict(list)
        average_delay_stats = defaultdict(list)
        solve_time_stats = defaultdict(list)
        average_solve_time_stats = defaultdict(list)
    
    sel_configs = [(label, params_dict) for label, params_dict in configurations if label not in total_delay_stats]
    print(f'Selected configurations: {len(sel_configs)} out of {len(configurations)}')
    for label, params_dict in sel_configs:
        print(f'*************** {label.upper()} ****************')
        if 'full' in label.lower():
            print(f'Solve full problem!')
            if "time_limit" in params_dict:
                full_time_limit = params_dict["time_limit"]
            jobs_solution, machines_assignment, solve_time, objective = flexible_jss(
                jobs_data, n_machines, scheduled_starts, scheduled_ends, 
                time_limit=full_time_limit, stop_search_time=full_time_limit,  # do not stop search earlier
                machines_start_time=None, jobs_start_time=None, optim_option=optim_option)
            avg_solve_time = solve_time
            total_delay = sum([jobs_solution[(job_id, task_id)][4] for job_id, job in jobs_data.items()
                               for task_id, task in job.items()])
            avg_delay = total_delay / len(jobs_solution)
        else:  # rolling horizon
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
                params_dict['perturb_p'] = perturb_p

            total_delay, avg_delay, solve_time, avg_solve_time = rolling_horizon(
                *in_params, **params_dict)

        total_delay_stats[label] = total_delay
        average_delay_stats[label] = avg_delay
        solve_time_stats[label] = solve_time
        average_solve_time_stats[label] = avg_solve_time

    if len(sel_configs) > 0:
        print_jss_name(args)
        print('\n==================stats==================')
        for i, (label, params_dict) in enumerate(configurations):
            print(f'{label}: total delay {total_delay_stats[label]} (avg delay {average_delay_stats[label]:.2f}), '
                f'solve time {solve_time_stats[label]:.2f} (avg solve time {average_solve_time_stats[label]:.2f})')

        pickle.dump([total_delay_stats, average_delay_stats, solve_time_stats],
                    open(f'{stats_dir}/stats_{data_idx}.pkl', 'wb'))


if __name__ == '__main__':
    parser = get_jss_argparser()
    parser.add_argument("--num_data_gen", default=100, type=int, help="number of data generated")
    parser.add_argument("--load_data", action="store_true", help="whether to load data")
    # data directories
    parser.add_argument("--train_data_dir", default="train_data_dir/train_data", type=str,
                        help="save train data directory")
    parser.add_argument("--jss_data_dir", default="train_data_dir/instance/m30-j30-t30-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3", 
                        type=str, help="save instance directory")
    parser.add_argument("--stats_dir", default="train_data_dir/stats", type=str, help="stats directory")
    
    parser.add_argument("--data_idx", default=1, type=int, help="save instance index")
    # parameters for optimization solver
    parser.add_argument("--time_limit", default=60, type=int, help="cp-sat solver time limit")
    parser.add_argument("--stop_search_time", default=3, type=int, help="cp-sat solver stop search wait time")
    parser.add_argument("--oracle_time_limit", default=60, type=int, help="time limit to find ground truth label")
    parser.add_argument("--oracle_stop_search_time", default=3, type=int, help="time limit to stop search for finding ground truth label")
    parser.add_argument("--n_cpus", default=1, type=int, help="cpus to do multiple aolves at the same time")   
    parser.add_argument("--num_oracle_trials", default=5, type=int,
                        help="number of oracle trials to find the most overlap")
    parser.add_argument("--num_best_randoms", default=5, type=int,
                        help="number of random sample to find the best random")
    parser.add_argument("--random_p", default=0.8, type=float, help="best random label")
    # rolling horizon parameters
    parser.add_argument("--window", default=50, type=int)
    parser.add_argument("--step", default=10, type=int)
    parser.add_argument("--perturb_data", action='store_true')
    parser.add_argument("--perturb_p", default=0.2, type=float)
    
    # the purpose of running this script
    parser.add_argument("--script_action", default='collect_data', 
                        choices=['collect_data', 'debug'])
    args = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    stop_search_time = args.stop_search_time
    time_limit = args.time_limit
    oracle_time_limit = args.oracle_time_limit
    oracle_stop_search_time = args.oracle_stop_search_time
    n_cpus = args.n_cpus
    #### Flexible jss instance parameters
    n_machines = args.n_machines
    n_jobs = args.n_jobs
    n_tasks_per_job = args.n_tasks_per_job
    l_low = args.l_low
    l_high = args.l_high
    start_time_high = args.start_time_high
    task_interval_high = args.task_interval_high
    # RHO loop
    window = args.window  
    step = args.step  
    optim_option = args.optim_option  # optimization objective
    perturb_data = args.perturb_data
    perturb_p = args.perturb_p

    random_p = args.random_p
    num_oracle_trials = args.num_oracle_trials
    num_best_randoms = args.num_best_randoms

    jss_data_dir = args.jss_data_dir
    train_data_dir = args.train_data_dir
    data_idx = args.data_idx
    stats_dir = args.stats_dir
    for d in [jss_data_dir, train_data_dir, stats_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    print_jss_name(args)

    
    collect_data_configs = [
        (f'collect_data',
          {'action_type': 'collect_data', 'num_oracle_trials': num_oracle_trials,
           'train_data_dir': train_data_dir, 'data_idx': data_idx,
           'oracle_time_limit': oracle_time_limit, 
           'oracle_stop_search_time': oracle_stop_search_time})
    ]

   
    debug_configs = [
        (f'oracle', {'action_type': 'oracle', 'num_oracle_trials': num_oracle_trials,
        'oracle_time_limit': oracle_time_limit, 
        'oracle_stop_search_time': oracle_stop_search_time}),
        (f'default', {'action_type': 'default'}), 
    ]

    if args.script_action == 'collect_data':
        configurations = collect_data_configs
    else:
        configurations = debug_configs

    run_all(configurations, stats_dir, data_idx, jss_data_dir, window, step, args,
            optim_option=optim_option, perturb_data=perturb_data, perturb_p=perturb_p)
    
