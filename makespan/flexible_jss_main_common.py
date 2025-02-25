import random
import time
import copy
import threading
import collections
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model
from flexible_jss_util import multiprocess

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
        # print(
        #     "Solution %i, time = %f s, objective = %i"
        #     % (self.__solution_count, current_time, current_obj)
        # )

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


def validate_jss_sol(jobs_data, n_machines, jobs_solution, machines_assignment, objective, 
                     machines_start_time=None, jobs_start_time=None, check_final=False, machine_breakdown=False):

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
            select_alt, machine, start_value, duration = jobs_solution[(job_id, task_id)]
            start_value = jobs_solution[(job_id, task_id)][2]
            if previous_end is not None and start_value < previous_end: 
                valid = False
                print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < previous_end {previous_end}')

            if jobs_start_time is not None and job_id in jobs_start_time:
                if start_value < jobs_start_time[job_id]:
                    valid = False
                    print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < job start time {jobs_start_time[job_id]}')

            # check if duration is correct, given the machine assignment
            try:
                machine_duration = task[select_alt][0]   # (duration, machine)
            except:
                print(f'Job {job_id} task {task_id} select_alt {select_alt} not in task {task}')
                return False

            if duration != machine_duration:
                valid = False
                print(f'Job {job_id} task {task_id} duration is not correct')


            if machines_start_time is not None and machine in machines_start_time:
                if start_value < machines_start_time[machine]:
                    valid = False
                    print(f'Job {job_id} task {task_id} start time is not correct: start {start_value} < machine start time {machines_start_time[machine]}')

            end_value = start_value + duration
            previous_end = end_value
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

    # Makespan objective
    jobs_data_job_task_pairs = [(job_id, task_id) for job_id, job in jobs_data.items() for task_id, task in job.items()]
    # check if this pair is the same as jobs_solution pair: for machine breakdown only check if its final solution
    if (not machine_breakdown or check_final) and not set(jobs_solution.keys()) == set(jobs_data_job_task_pairs):
        valid = False
        print('Jobs solution keys are not the same as jobs data keys')
        different_keys = set(jobs_solution.keys()) ^ set(jobs_data_job_task_pairs)
        print(f'Different keys: {different_keys}')

    my_objective = max([ends[(job_id, task_id)] 
                       for job_id, job in jobs_data.items() for task_id, task in job.items() 
                       if (job_id, task_id) in jobs_solution])
        
    if abs(my_objective - objective) > EPS:
        print(f'Objective value may not correct: my_objective {my_objective} != objective {objective}')

    print(f'Is solution valid? {valid}')

    return valid


def flexible_jss(jobs_data, n_machines, time_limit=-1, stop_search_time=5,
                 machines_start_time=None, jobs_start_time=None,
                 do_warm_start=False, jobs_solution_warm_start={}):
    # minimize the makespan for all jobs
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
        if len(job) == 0:
            continue
        for task_id, task in job.items():
            min_duration = float('inf')
            max_duration = -float('inf')

            for alt_id, alt in task.items():
                min_duration = min(min_duration, alt[0])
                max_duration = max(max_duration, alt[0])

            # Create main interval for the task.
            suffix_name = "_j%i_t%i" % (job_id, task_id)

            st_task = jobs_start_time[job_id] if jobs_start_time is not None else 0
            
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
            # if num_alternatives > 1:
            l_presences = []
            for alt_id, alt in task.items():
                alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                l_presence = model.NewBoolVar("presence" + alt_suffix)

                l_machine = alt[1]
                l_st_task = 0 if machines_start_time is None else machines_start_time[l_machine]
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

            model.AddExactlyOne(l_presences)
        
        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in range(n_machines):
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    print('.................Objective is makespan.................')
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)
    
    solver = cp_model.CpSolver()
    if time_limit > 0:
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 4

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
            # print(time.time(), solution_callback.last_improvement_time)
            if time.time() - solution_callback.last_improvement_time > stop_search_time:
                solution_callback.StopSearch()
                print(f"No improvement in {stop_search_time} seconds, stopping search.")
                solver_thread.join()  # Clean up the solver thread


    status = model.status
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        for job_id, job in jobs_data.items():
            # print("Job %i:" % job_id)
            for task_id, task in job.items():
                start_value = solver.Value(starts[(job_id, task_id)])
                end_value = solver.Value(ends[(job_id, task_id)])
                machine, duration, selected_alt = -1, -1, -1
                for alt_id, alt in task.items():
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = alt[0]
                        machine = alt[1]
                        selected_alt = alt_id
                
                jobs_solution[(job_id, task_id)] = (selected_alt, machine, start_value, duration)
                machines_assignment[machine].append(
                    assigned_task_type(
                        start=start_value,
                        job=job_id,
                        index=task_id,
                        duration=duration
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

    # print("All machine assignments", [x[1] for x in jobs_solution.values()])
    # print("All start", [x[2] for x in jobs_solution.values()])
    # print("All durations", [x[3] for x in jobs_solution.values()])

    solve_time = solver.WallTime()
    print("Statistics")
    print("  - conflicts : %i" % solver.NumConflicts())
    print("  - branches  : %i" % solver.NumBranches())
    print("  - wall time : %f s" % solve_time)

    return jobs_solution, machines_assignment, solve_time, objective


def sort_by_task_order(jobs_data):
    task_indices = [(job_id, task_id) for job_id, job in jobs_data.items() for task_id, task in job.items()]
    task_indices_sort_keys = {(job_id, task_id): (job_id, task_id, task_id / len(job)) for job_id, job in jobs_data.items() for task_id, task in job.items()}
    sorted_task_indices = sorted(task_indices, key=lambda x: (task_indices_sort_keys[x][2], task_indices_sort_keys[x][1], random.random()))  # proportion of task id, actual task id, random tie breaker

    return sorted_task_indices


def same_machine_assignment(job_solution1, job_solution2):
    return job_solution1[1] == job_solution2[1]


def do_oracle_trial_i(args):
    jobs_data_sel, n_machines, machines_start_time, jobs_start_time, \
        oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices, \
            overlapping_jobs_solution_all, oracle_i = args

    jobs_solution_sel_oracle, machines_assignment_sel_oracle, solve_time_sel_i, objective_sel_i = \
        flexible_jss(jobs_data_sel, n_machines, time_limit=oracle_time_limit, stop_search_time=oracle_stop_search_time,
                     machines_start_time=machines_start_time, jobs_start_time=jobs_start_time,
                     do_warm_start=True, jobs_solution_warm_start=overlapping_jobs_solution_all) 


    overlapping_jobs_solution_i = {(job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                                              for job_id, task_id in overlapping_task_indices
                                              if same_machine_assignment(jobs_solution_sel[(job_id, task_id)],
                                                        jobs_solution_sel_oracle[(job_id, task_id)])}
    
    print(f'[Trial {oracle_i}] Number of overlapping assignments {len(overlapping_jobs_solution_i)} '
          f'out of {len(overlapping_task_indices)}')
    
    # print('+++++++++++++++++++++++DEBUG++++++++++++++++++++++++')
    # print(list(overlapping_jobs_solution_i.values()))
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return overlapping_jobs_solution_i, objective_sel_i


def find_overlapping_jobs_solution_oracle(jobs_data_sel, n_machines, machines_start_time, jobs_start_time, oracle_time_limit, 
                                          oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
                                          overlapping_jobs_solution_all, 
                                          num_oracle_trials=1, n_cpus=1):
    overlapping_jobs_solution = {}
    print('-----------------------------------')
    tasks = [(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
              oracle_time_limit, oracle_stop_search_time, jobs_solution_sel, overlapping_task_indices,
              overlapping_jobs_solution_all, oracle_i)
              for oracle_i in range(num_oracle_trials)]
    if n_cpus > 1:
        results = multiprocess(do_oracle_trial_i, tasks, cpus=min(num_oracle_trials, n_cpus))
    else:
        results = [do_oracle_trial_i(task) for task in tasks]

    overlapping_jobs_solutions, objectives = zip(*results)
    # most_overlap_idx = np.argmax([len(overlapping_jobs_solution_i) for overlapping_jobs_solution_i in overlapping_jobs_solutions])
    best_objective_idx = np.argmin(objectives)
    overlapping_jobs_solution = overlapping_jobs_solutions[best_objective_idx]
    print('-----------------------------------')
    return overlapping_jobs_solution, objectives[best_objective_idx]