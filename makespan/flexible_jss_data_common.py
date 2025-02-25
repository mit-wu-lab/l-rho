import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


#### data for model training
class FlexibleJSSData(Data):
    def __init__(self, x_tasks, x_machines,
                 overlap_machine_task_edge_idx, overlap_machine_task_edge_val,
                 other_machine_task_edge_idx, other_machine_task_edge_val,
                 task_precedence_edge_idx, task_precedence_edge_val,
                 task_solution_edge_idx, task_solution_edge_val,
                 task_label_idx, task_label):
        super().__init__()
        self.x_tasks = x_tasks
        self.x_machines = x_machines
        self.overlap_machine_task_edge_idx = overlap_machine_task_edge_idx
        self.overlap_machine_task_edge_val = overlap_machine_task_edge_val
        self.other_machine_task_edge_idx = other_machine_task_edge_idx
        self.other_machine_task_edge_val = other_machine_task_edge_val
        self.task_precedence_edge_idx = task_precedence_edge_idx
        self.task_precedence_edge_val = task_precedence_edge_val
        self.task_solution_edge_idx = task_solution_edge_idx
        self.task_solution_edge_val = task_solution_edge_val
        self.task_label_idx = task_label_idx
        self.task_label = task_label

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['x_tasks', 'x_machines', 'overlap_machine_task_edge_val', 'other_machine_task_edge_val',
                   'task_precedence_edge_val', 'task_solution_edge_val', 'task_label']:
            inc = 0
        elif key in ['task_precedence_edge_idx', 'task_solution_edge_idx']:
            inc = torch.tensor([[self.x_tasks.size(0)],
                                [self.x_tasks.size(0)]])
        elif key == 'task_label_idx':
            inc = self.x_tasks.size(0)
        elif key in ['overlap_machine_task_edge_idx', 'other_machine_task_edge_idx']:
            inc = torch.tensor([[self.x_machines.size(0)],
                                [self.x_tasks.size(0)]])
        else:
            print('Resorting to default', key)
            inc = super().__inc__(key, value, *args, **kwargs)

        return inc

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['x_tasks', 'x_machines', 'overlap_machine_task_edge_val',
                   'other_machine_task_edge_val', 'task_precedence_edge_val',
                   'task_solution_edge_val', 'task_label_idx', 'task_label']:
            cat_dim = 0
        elif key in ['task_precedence_edge_idx', 'task_solution_edge_idx',
                     'overlap_machine_task_edge_idx', 'other_machine_task_edge_idx']:
            cat_dim = 1
        else:
            print('Resorting to default', key)
            cat_dim = super().__inc__(key, value, *args, **kwargs)

        return cat_dim


class FlexibleJSSDataset(Dataset):
    def __init__(self, dataset):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, index):
        data = self.dataset[index]['data']
        return FlexibleJSSData(torch.tensor(data.x_tasks).to(torch.float32),
                               torch.tensor(data.x_machines).to(torch.float32),
                               torch.tensor(data.overlap_machine_task_edge_idx).to(torch.int64),
                               torch.tensor(data.overlap_machine_task_edge_val).to(torch.float32),
                               torch.tensor(data.other_machine_task_edge_idx).to(torch.int64),
                               torch.tensor(data.other_machine_task_edge_val).to(torch.float32),
                               torch.tensor(data.task_precedence_edge_idx).to(torch.int64),
                               torch.tensor(data.task_precedence_edge_val).to(torch.float32),
                               torch.tensor(data.task_solution_edge_idx).to(torch.int64),
                               torch.tensor(data.task_solution_edge_val).to(torch.float32),
                               torch.tensor(data.task_label_idx).to(torch.int64),
                               torch.tensor(data.task_label).to(torch.float32))



def get_rollout_prediction(model, data, decode_strategy='argmax', threshold=0.0, topk_th=0.5):
    dataset = FlexibleJSSDataset([data])

    loader = get_dataloader(dataset, batch_size=1, shuffle=False,
                            follow_batch=['x_tasks', 'x_machines'],
                            num_workers=0, pin_memory=False)
    loader_data = next(iter(loader))
    with torch.no_grad():
        output = model(loader_data.to(next(model.parameters()).device))

    if decode_strategy == 'argmax':
        predictions = (F.sigmoid(output) >= threshold).cpu().reshape(-1).numpy()  # torch.sign(output).cpu().reshape(-1).numpy()
    elif decode_strategy == 'topk':
        output_topk_th = torch.topk(output, int(topk_th * len(output)))
        predictions = (output >= output_topk_th).cpu().reshape(-1).numpy()
    else:  # sampling
        predictions = torch.bernoulli(output).cpu().reshape(-1).numpy()

    ''' select vehicles that have positive prediction '''
    return set([data['tasks_tuple'][task_idx] for pred, task_idx in zip(predictions,
                                                                        loader_data.task_label_idx) if pred > 0])


def get_task_index(jobs_data_sel):
    tasks_dict = {}
    task_tuple = []
    for job_id, job in jobs_data_sel.items():
        for task_id in job:
            task_tuple.append((job_id, task_id))
            tasks_dict[(job_id, task_id)] = len(tasks_dict)
    return task_tuple, tasks_dict


def get_edges(tasks_dict, jobs_data_sel, n_machines, overlapping_task_indices,
              overlapping_jobs_solution_all, overlapping_machines_assignment_all):
    overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, = get_machine_task_edges(
        tasks_dict, jobs_data_sel, n_machines, overlapping_task_indices, overlapping_jobs_solution_all,
        overlapping_machines_assignment_all)
    job_precedence_edge_idx, job_precedence_edge_val = get_task_precedence_edges(tasks_dict, jobs_data_sel,
                                                                                 overlapping_task_indices)
    solution_edge_idx, solution_edge_val = get_task_solution_edges(tasks_dict, n_machines, overlapping_task_indices,
                                                                   overlapping_machines_assignment_all)
    return overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        job_precedence_edge_idx, job_precedence_edge_val, \
        solution_edge_idx, solution_edge_val


def get_machine_task_edges(tasks_dict, jobs_data_sel, n_machines,
                           overlapping_task_indices, overlapping_jobs_solution_all,
                           overlapping_machines_assignment_all):
    overlap_machine_task_edge_idx = [[], []]
    overlap_machine_task_edge_val = []

    for machine in range(n_machines):
        if machine in overlapping_machines_assignment_all:
            for task in overlapping_machines_assignment_all[machine]:
                overlap_machine_task_edge_idx[0].append(machine)
                overlap_machine_task_edge_idx[1].append(tasks_dict[(task.job, task.index)])
                overlap_machine_task_edge_val.append([task.duration, 1])

    other_machine_task_edge_idx = [[], []]
    other_machine_task_edge_val = []
    for job_id, job in jobs_data_sel.items():
        for task_id, task in job.items():
            for alt_id, (l_duration, l_machine) in task.items():  # (duration, machine)
                if (job_id, task_id) in overlapping_task_indices and \
                        l_machine == overlapping_jobs_solution_all[(job_id, task_id)][1]:
                    continue

                other_machine_task_edge_idx[0].append(l_machine)
                other_machine_task_edge_idx[1].append(tasks_dict[(job_id, task_id)])
                other_machine_task_edge_val.append(
                    [l_duration, -1 if (job_id, task_id) in overlapping_task_indices else 0])

    return overlap_machine_task_edge_idx, overlap_machine_task_edge_val, other_machine_task_edge_idx, other_machine_task_edge_val


def get_task_precedence_edges(tasks_dict, jobs_data_sel, overlapping_task_indices):
    job_precedence_edge_idx = [[], []]
    job_precedence_edge_val = []
    for job_id, job in jobs_data_sel.items():
        if len(job) >= 2:
            all_tasks = list(job.keys())
            for task_id, next_task_id in zip(all_tasks[:-1], all_tasks[1:]):
                job_precedence_edge_idx[0].append(tasks_dict[(job_id, task_id)])
                job_precedence_edge_idx[1].append(tasks_dict[(job_id, next_task_id)])
                both_overlap = (job_id, task_id) in overlapping_task_indices and (
                job_id, next_task_id) in overlapping_task_indices
                job_precedence_edge_val.append(1 if both_overlap else 0)
    return job_precedence_edge_idx, job_precedence_edge_val


def get_task_solution_edges(tasks_dict, n_machines, overlapping_task_indices,
                            overlapping_machines_assignment_all):
    solution_edge_idx = [[], []]
    solution_edge_val = []

    for machine in range(n_machines):
        if machine in overlapping_machines_assignment_all:
            prev_machine_assignments = [(task.job, task.index) for task in overlapping_machines_assignment_all[machine]
                                        if (task.job, task.index) in overlapping_task_indices]
            if len(prev_machine_assignments) >= 2:
                for task1, task2 in zip(prev_machine_assignments[:-1], prev_machine_assignments[1:]):
                    solution_edge_idx[0].append(tasks_dict[task1])
                    solution_edge_idx[1].append(tasks_dict[task2])
                    solution_edge_val.append(1)

    return solution_edge_idx, solution_edge_val


def get_dummy_label(tasks_index, overlapping_task_indices):
    ''' not fix anything '''
    task_label_idx = [tasks_index[task] for task in overlapping_task_indices]
    task_label = [0 for _ in range(len(task_label_idx))]
    return task_label_idx, task_label


def get_label(tasks_index, overlapping_task_indices, overlapping_jobs_solution_oracle):
    task_label_idx = [tasks_index[task] for task in overlapping_task_indices]
    task_label = [1 if task in overlapping_jobs_solution_oracle else 0 for task in overlapping_task_indices]
    return task_label_idx, task_label


def get_dataloader(dataset, batch_size=64, shuffle=True, follow_batch=['x_tasks', 'x_machines'], num_workers=0,
                   pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch,
                      num_workers=num_workers, pin_memory=pin_memory)
