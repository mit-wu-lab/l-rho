import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from flexible_jss_data_common import FlexibleJSSData, get_task_index, get_edges, get_dummy_label, get_label


def get_data(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends, 
             machines_start_time, jobs_start_time,
             overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all,
             overlapping_jobs_solution_oracle):
    '''
    jobs_data_sel: {job_id: {task_id: {alt_id: (duration, machine), ... }}}
    scheduled_starts_sel: {job_id: {task_id: start_time}}
    machines_start_time: [start_time for _ in range(n_machines)]
    jobs_start_time: [start_time for _ in range(n_jobs)]
    overlapping_task_indices: [(job_id, task_id)]
    overlapping_jobs_solution_all: {(job_id, task_id): (selected_alt, machine, start_value, duration, delay)} (prev iter, overlap)
    machines_assignment_all: {machine: [assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration, delay=delay)]} (prev iter, overlap)
    '''
    # TODO: parse machines_assignment_all to compute the wait time between end and next start for each task in overlapping_task_indices

    x_tasks, tasks_tuple, tasks_dict, x_machines, \
        overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val = get_state(jobs_data_sel, n_machines,
                                                                   scheduled_starts_sel, scheduled_ends, 
                                                                   machines_start_time, jobs_start_time,
                                                                   overlapping_task_indices,
                                                                   overlapping_jobs_solution_all,
                                                                   overlapping_machines_assignment_all)

    task_label_idx, task_label = get_label(tasks_dict, overlapping_task_indices, overlapping_jobs_solution_oracle)

    data = FlexibleJSSData(x_tasks, x_machines,
                           overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
                           other_machine_task_edge_idx, other_machine_task_edge_val, \
                           task_precedence_edge_idx, task_precedence_edge_val,
                           task_solution_edge_idx, task_solution_edge_val,
                           task_label_idx, task_label)
    return {'data': data, 'tasks_tuple': tasks_tuple, 'tasks_dict': tasks_dict}


def get_rollout_data(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends,
                     machines_start_time, jobs_start_time,
                     overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all):
    x_tasks, tasks_tuple, tasks_dict, x_machines, \
        overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val = get_state(jobs_data_sel, n_machines,
                                                                   scheduled_starts_sel, scheduled_ends, 
                                                                   machines_start_time, jobs_start_time,
                                                                   overlapping_task_indices,
                                                                   overlapping_jobs_solution_all,
                                                                   overlapping_machines_assignment_all)

    task_label_idx, task_label = get_dummy_label(tasks_dict, overlapping_task_indices)

    data = FlexibleJSSData(x_tasks, x_machines,
                           overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
                           other_machine_task_edge_idx, other_machine_task_edge_val, \
                           task_precedence_edge_idx, task_precedence_edge_val,
                           task_solution_edge_idx, task_solution_edge_val,
                           task_label_idx, task_label)
    return {'data': data, 'tasks_tuple': tasks_tuple, 'tasks_dict': tasks_dict}


def get_state(jobs_data_sel, n_machines, scheduled_starts_sel, scheduled_ends, 
              machines_start_time, jobs_start_time,
              overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all):
    task_tuple, tasks_dict = get_task_index(jobs_data_sel)

    x_tasks = get_tasks_features(tasks_dict, jobs_data_sel, scheduled_starts_sel, scheduled_ends, 
                                 jobs_start_time, overlapping_task_indices, overlapping_jobs_solution_all)
    x_machines = get_machines_features(n_machines, machines_start_time, overlapping_machines_assignment_all)

    overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val = get_edges(tasks_dict, jobs_data_sel, n_machines,
                                                                   overlapping_task_indices,
                                                                   overlapping_jobs_solution_all,
                                                                   overlapping_machines_assignment_all)

    return x_tasks, task_tuple, tasks_dict, x_machines, \
        overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val


def get_tasks_features(tasks_dict, jobs_data_sel, scheduled_starts_sel, scheduled_ends,
                       jobs_start_time, overlapping_task_indices, overlapping_jobs_solution_all):
    x_tasks = [None for _ in range(len(tasks_dict))]

    for job_id, job in jobs_data_sel.items():
        for task_id, task in job.items():
            job_start_time = jobs_start_time[job_id]
            scheduled_task_start_time = scheduled_starts_sel[job_id][task_id]
            # add feature
            scheduled_task_end_time = scheduled_ends[job_id][task_id]

            task = jobs_data_sel[job_id][task_id]
            durations = [task[alt][0] for alt in task]
            avg_dur, std_dur, min_dur, max_dur = np.mean(durations), np.std(durations), np.min(durations), np.max(
                durations)

            is_overlap = (job_id, task_id) in overlapping_task_indices
            is_overlap_feat = 1 if is_overlap else -1

            alt_delay = overlapping_jobs_solution_all[(job_id, task_id)][4] if is_overlap else -1
            alt_machine = overlapping_jobs_solution_all[(job_id, task_id)][1] if is_overlap else -1
            # add feature
            alt_start = overlapping_jobs_solution_all[(job_id, task_id)][2] if is_overlap else -1
            # add feature
            alt_end = alt_start + overlapping_jobs_solution_all[(job_id, task_id)][3] if is_overlap else -1
            alt_duration = overlapping_jobs_solution_all[(job_id, task_id)][3] if is_overlap else -1
            # alternative average, std, min, max duration on other machines
            alt_avg_dur = np.mean([task[alt][0] for alt in task if task[alt][1] != alt_machine]) if is_overlap else -1
            alt_std_dur = np.std([task[alt][0] for alt in task if task[alt][1] != alt_machine]) if is_overlap else -1
            alt_min_dur = np.min([task[alt][0] for alt in task if task[alt][1] != alt_machine]) if is_overlap else -1
            alt_max_dur = np.max([task[alt][0] for alt in task if task[alt][1] != alt_machine]) if is_overlap else -1

            task_input_feature = [job_start_time, scheduled_task_start_time,
                                  scheduled_task_end_time, 
                                  avg_dur, std_dur, min_dur, max_dur,
                                  job_id, task_id, is_overlap_feat,
                                  alt_delay, alt_machine, alt_start, alt_end, alt_duration,
                                  alt_avg_dur, alt_std_dur, alt_min_dur, alt_max_dur]
            x_tasks[tasks_dict[(job_id, task_id)]] = task_input_feature

    return x_tasks


def get_machines_features(n_machines, machines_start_time, overlapping_machines_assignment_all):
    x_machines = []

    for machine in range(n_machines):
        machine_start_time = machines_start_time[machine]

        num_overlap = len(overlapping_machines_assignment_all[machine]) if machine in overlapping_machines_assignment_all else 0
        if num_overlap > 0:
            # avg / std / min / max delay of assigned tasks on this machine
            avg_delay = np.mean([task.delay for task in overlapping_machines_assignment_all[machine]]) 
            std_delay = np.std([task.delay for task in overlapping_machines_assignment_all[machine]])
            min_delay = np.min([task.delay for task in overlapping_machines_assignment_all[machine]])
            max_delay = np.max([task.delay for task in overlapping_machines_assignment_all[machine]])
        else:
            num_overlap, avg_delay, std_delay, min_delay, max_delay = 0, -1, -1, -1, -1

        machine_input_feature = [machine_start_time, num_overlap,
                                 avg_delay, std_delay, min_delay, max_delay]
        x_machines.append(machine_input_feature)
    return x_machines