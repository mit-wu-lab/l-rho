import numpy as np
from flexible_jss_data_common import FlexibleJSSData, get_task_index, get_edges, get_label, get_dummy_label


def get_data(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
             overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all,
             overlapping_jobs_solution_oracle, is_breakdown, breakdown_machines, 
             breakdown_operations, recovered_operations):
    '''
    jobs_data_sel: {job_id: {task_id: {alt_id: (duration, machine), ... }}}
    machines_start_time: [start_time for _ in range(n_machines)]
    jobs_start_time: [start_time for _ in range(n_jobs)]
    overlapping_task_indices: [(job_id, task_id)]
    overlapping_jobs_solution_all: {(job_id, task_id): (selected_alt, machine, start_value, duration)} (prev iter, overlap)
    machines_assignment_all: {machine: [assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration)]} (prev iter, overlap)
    '''
    x_tasks, tasks_tuple, tasks_dict, x_machines, \
        overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val = get_state(jobs_data_sel, n_machines,
                                                                   machines_start_time, jobs_start_time,
                                                                   overlapping_task_indices,
                                                                   overlapping_jobs_solution_all,
                                                                   overlapping_machines_assignment_all,
                                                                   is_breakdown, 
                                                                   breakdown_machines, 
                                                                   breakdown_operations,
                                                                   recovered_operations)

    task_label_idx, task_label = get_label(tasks_dict, overlapping_task_indices, overlapping_jobs_solution_oracle)

    data = FlexibleJSSData(x_tasks, x_machines,
                           overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
                           other_machine_task_edge_idx, other_machine_task_edge_val, \
                           task_precedence_edge_idx, task_precedence_edge_val,
                           task_solution_edge_idx, task_solution_edge_val,
                           task_label_idx, task_label)
    return {'data': data, 'tasks_tuple': tasks_tuple, 'tasks_dict': tasks_dict}


def get_rollout_data(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
                     overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all,
                     is_breakdown, breakdown_machines, breakdown_operations, recovered_operations):
    x_tasks, tasks_tuple, tasks_dict, x_machines, \
        overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
        other_machine_task_edge_idx, other_machine_task_edge_val, \
        task_precedence_edge_idx, task_precedence_edge_val, \
        task_solution_edge_idx, task_solution_edge_val = get_state(jobs_data_sel, n_machines,
                                                                   machines_start_time, jobs_start_time,
                                                                   overlapping_task_indices,
                                                                   overlapping_jobs_solution_all,
                                                                   overlapping_machines_assignment_all,
                                                                   is_breakdown,
                                                                   breakdown_machines,
                                                                   breakdown_operations,
                                                                   recovered_operations)

    task_label_idx, task_label = get_dummy_label(tasks_dict, overlapping_task_indices)

    data = FlexibleJSSData(x_tasks, x_machines,
                           overlap_machine_task_edge_idx, overlap_machine_task_edge_val, \
                           other_machine_task_edge_idx, other_machine_task_edge_val, \
                           task_precedence_edge_idx, task_precedence_edge_val,
                           task_solution_edge_idx, task_solution_edge_val,
                           task_label_idx, task_label)
    return {'data': data, 'tasks_tuple': tasks_tuple, 'tasks_dict': tasks_dict}


def get_state(jobs_data_sel, n_machines, machines_start_time, jobs_start_time,
              overlapping_task_indices, overlapping_jobs_solution_all, overlapping_machines_assignment_all,
              is_breakdown, breakdown_machines, breakdown_operations, recovered_operations):
    task_tuple, tasks_dict = get_task_index(jobs_data_sel)

    x_tasks = get_tasks_features(tasks_dict, jobs_data_sel, jobs_start_time, overlapping_task_indices, overlapping_jobs_solution_all,
                                 is_breakdown, breakdown_operations, recovered_operations)
    x_machines = get_machines_features(n_machines, machines_start_time, overlapping_machines_assignment_all,
                                       is_breakdown, breakdown_machines)

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


def get_tasks_features(tasks_dict, jobs_data_sel, jobs_start_time, overlapping_task_indices, overlapping_jobs_solution_all,
                       is_breakdown, breakdown_operations, recovered_operations):
    # overlapping_tasks = self.subp_pair.overlapping_tasks
    x_tasks = [None for _ in range(len(tasks_dict))]

    for job_id, job in jobs_data_sel.items():
        for task_id, task in job.items():
            job_start_time = jobs_start_time[job_id]

            task = jobs_data_sel[job_id][task_id]
            durations = [task[alt][0] for alt in task]
            avg_dur, std_dur, min_dur, max_dur = np.mean(durations), np.std(durations), np.min(durations), np.max(
                durations)

            is_overlap = (job_id, task_id) in overlapping_task_indices
            is_overlap_feat = 1 if is_overlap else -1
            alt_machine = overlapping_jobs_solution_all[(job_id, task_id)][1] if is_overlap else -1
            alt_duration = overlapping_jobs_solution_all[(job_id, task_id)][3] if is_overlap else -1
            # alternative average, std, min, max duration on other machines
            alt_durs = [task[alt][0] for alt in task if task[alt][1] != alt_machine]
            alt_avg_dur = np.mean(alt_durs) if is_overlap and len(alt_durs) > 0 else -1
            alt_std_dur = np.std(alt_durs) if is_overlap and len(alt_durs) > 0 else -1
            alt_min_dur = np.min(alt_durs) if is_overlap and len(alt_durs) > 0 else -1
            alt_max_dur = np.max(alt_durs) if is_overlap and len(alt_durs) > 0 else -1
            alt_end_time = job_start_time + alt_duration if is_overlap else -1

            task_input_feature = [job_start_time, avg_dur, std_dur, min_dur, max_dur,
                                  job_id, task_id, is_overlap_feat,
                                  alt_machine, alt_duration,
                                  alt_avg_dur, alt_std_dur, alt_min_dur, alt_max_dur,
                                  alt_end_time]   # 15

            task_input_feature += [1 if is_breakdown else -1,
                                   1 if (job_id, task_id) in breakdown_operations else -1,
                                   1 if (job_id, task_id) in recovered_operations else -1]
                                        
            x_tasks[tasks_dict[(job_id, task_id)]] = task_input_feature

    return x_tasks


def get_machines_features(n_machines, machines_start_time, overlapping_machines_assignment_all,
                          is_breakdown, breakdown_machines):
    x_machines = []

    for machine in range(n_machines):
        machine_start_time = machines_start_time[machine]

        num_overlap = len(overlapping_machines_assignment_all[machine]) if machine in overlapping_machines_assignment_all else 0
        if num_overlap > 0:
            # avg / std / min / max makespan of assigned tasks on this machine
            avg_end_time = np.mean([task.start + task.duration for task in overlapping_machines_assignment_all[machine]]) 
            std_end_time = np.std([task.start + task.duration for task in overlapping_machines_assignment_all[machine]])
            min_end_time = np.min([task.start + task.duration for task in overlapping_machines_assignment_all[machine]])
            max_end_time = np.max([task.start + task.duration for task in overlapping_machines_assignment_all[machine]])
            avg_duration = np.mean([task.duration for task in overlapping_machines_assignment_all[machine]])
            std_duration = np.std([task.duration for task in overlapping_machines_assignment_all[machine]])
            min_duration = np.min([task.duration for task in overlapping_machines_assignment_all[machine]])
            max_duration = np.max([task.duration for task in overlapping_machines_assignment_all[machine]])

            # end time of the machine
            machine_end_time = max([task.start + task.duration for task in
                                    overlapping_machines_assignment_all[machine]])
        else:
            num_overlap, avg_end_time, std_end_time, min_end_time, max_end_time, machine_end_time = 0, -1, -1, -1, -1, -1
            avg_duration, std_duration, min_duration, max_duration = -1, -1, -1, -1
        machine_input_feature = [machine_start_time, num_overlap, avg_end_time, std_end_time, min_end_time, max_end_time, 
                                 avg_duration, std_duration, min_duration, max_duration, machine_end_time]   # 11
        machine_input_feature += [1 if is_breakdown else -1, 
                                  1 if machine in breakdown_machines else -1]
        x_machines.append(machine_input_feature)
    return x_machines

