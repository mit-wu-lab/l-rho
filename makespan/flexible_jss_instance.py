import os
import sys
import random
import pickle
import argparse
import numpy as np


EPS = 1e-5


class Config:
    def __init__(self, n_j, n_m, op_per_job=0, 
                low=1, high=99, data_suffix='mix', op_per_mch_min=1, op_per_mch_max=5):
        self.n_j = n_j
        self.n_m = n_m
        self.op_per_job = op_per_job  # if 0: number equals m
        self.low = low
        self.high = high
        self.data_suffix = data_suffix
        self.op_per_mch_min = op_per_mch_min
        self.op_per_mch_max = op_per_mch_max if op_per_mch_max is not None else n_m

def SD2_instance_generator(config):
    """
    :param config: a package of parameters
    :return: a fjsp instance generated by SD2, with
        job_length : the number of operations in each job (shape [J])
        op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
        op_per_mch : the average number of compatible machines of each operation
    """
    n_j = config.n_j
    n_m = config.n_m
    if config.op_per_job == 0:
        op_per_job = n_m
    else:
        op_per_job = config.op_per_job

    low = config.low
    high = config.high
    data_suffix = config.data_suffix

    op_per_mch_min = 1
    if data_suffix == "nf":
        op_per_mch_max = 1
    elif data_suffix == "mix":
        op_per_mch_max = n_m
    else:
        op_per_mch_min = config.op_per_mch_min
        op_per_mch_max = config.op_per_mch_max
    if op_per_mch_min < 1 or op_per_mch_max > n_m:
        print(f'Error from Instance Generation: [{op_per_mch_min},{op_per_mch_max}] '
              f'with num_mch : {n_m}')
        sys.exit()

    n_op = int(n_j * op_per_job)
    job_length = np.full(shape=(n_j,), fill_value=op_per_job, dtype=int)
    op_use_mch = np.random.randint(low=op_per_mch_min, high=op_per_mch_max + 1,
                                   size=n_op)

    op_per_mch = np.mean(op_use_mch)
    op_pt = np.random.randint(low=low, high=high + 1, size=(n_op, n_m))

    for row in range(op_pt.shape[0]):
        mch_num = int(op_use_mch[row])
        if mch_num < n_m:
            inf_pos = np.random.choice(np.arange(0, n_m), n_m - mch_num, replace=False)
            op_pt[row][inf_pos] = 0

    return job_length, op_pt, op_per_mch


def matrix_to_text(job_length, op_pt, op_per_mch):
    """
        Convert matrix form of the data into test form
    :param job_length: the number of operations in each job (shape [J])
    :param op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    :param op_per_mch: the average number of compatible machines of each operation
    :return: the standard text form of the instance
    """
    n_j = job_length.shape[0]
    n_op, n_m = op_pt.shape
    text = [f'{n_j}\t{n_m}\t{op_per_mch}']

    op_idx = 0
    for j in range(n_j):
        line = f'{job_length[j]}'
        for _ in range(job_length[j]):
            use_mch = np.where(op_pt[op_idx] != 0)[0]
            line = line + ' ' + str(use_mch.shape[0])
            for k in use_mch:
                line = line + ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])
            op_idx += 1

        text.append(line)

    return text


def generate_instances(data_dir='train_data_dir', n_j=40, n_m=10, op_per_job=20, op_per_mch_min=1, op_per_mch_max=10, n_data=600):
    config = Config(n_j, n_m, op_per_job=op_per_job, low=1, high=99, 
                    data_suffix='mix', op_per_mch_min=op_per_mch_min, op_per_mch_max=op_per_mch_max)

    instance_name = f'j{n_j}-m{n_m}-t{op_per_job}'
    if op_per_mch_min != 1 or op_per_mch_max != n_m:
        instance_name += f'-m{op_per_mch_min}-{op_per_mch_max}'
    instance_name += f'_{config.data_suffix}'

    ### typical FJSP txt format
    dir_path_txt = f'{data_dir}/instance_txt/{instance_name}'
    ### L-RHO format
    dir_path = f'{data_dir}/instance/{instance_name}'

    os.makedirs(dir_path_txt, exist_ok=True)
    os.makedirs(dir_path, exist_ok=True)

    for idx in range(n_data):
        if idx % 50 == 0:
            print(f'Generating data {idx} / {n_data}')

        file_path_txt = dir_path_txt + '/' + 'data_{}.fjs'.format(str.zfill(str(idx + 1), 3))
        file_path = os.path.join(dir_path, f'data_{idx}.pkl')

        if os.path.exists(file_path_txt) and os.path.exists(file_path): continue

        job_length, op_pt, op_per_mch = SD2_instance_generator(config)

        lines_doc = matrix_to_text(job_length, op_pt, op_per_mch)

        doc = open(file_path_txt, 'w')
        for i in range(len(lines_doc)):
            print(lines_doc[i], file=doc)
        doc.close()

        jobs_data = {}

        op_id = 0
        for job_id in range(len(job_length)):
            num_task = job_length[job_id]
            jobs_data[job_id] = {}
            for task_id in range(num_task):
                tasks_data = {}
                for i_machine, duration in enumerate(op_pt[op_id]):
                    if duration > 0:
                        tasks_data[i_machine] = (duration, i_machine)

                jobs_data[job_id][task_id] = tasks_data
                op_id += 1

        n_machines = len(op_pt[0])
        n_jobs = len(job_length)
        n_ops = sum(job_length)

        pickle.dump([jobs_data, n_machines, n_jobs], open(file_path, 'wb'))
############################################################################################

### Breakdown events generation
def generate_machine_breakdown(jobs_data, n_machines, num_machine_breakdown_p=0.2,
                               first_breakdown_buffer_lb=50, first_breakdown_buffer_ub=150,
                               machine_breakdown_duration_lb=100, machine_breakdown_duration_ub=100,
                               breakdown_buffer_lb=400, breakdown_buffer_ub=600):
    # estimated horizon upper bound for the entire problem
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


class BreakdownConfig:
    def __init__(self, num_machine_breakdown_p=0.2,
                 first_breakdown_buffer_lb=50, first_breakdown_buffer_ub=150,
                 machine_breakdown_duration_lb=100, machine_breakdown_duration_ub=100,
                 breakdown_buffer_lb=400, breakdown_buffer_ub=600):
        self.num_machine_breakdown_p = num_machine_breakdown_p
        self.first_breakdown_buffer_lb = first_breakdown_buffer_lb
        self.first_breakdown_buffer_ub = first_breakdown_buffer_ub
        self.machine_breakdown_duration_lb = machine_breakdown_duration_lb
        self.machine_breakdown_duration_ub = machine_breakdown_duration_ub
        self.breakdown_buffer_lb = breakdown_buffer_lb
        self.breakdown_buffer_ub = breakdown_buffer_ub
    
    def get_data_name(self):
        name_list = ['num_p', 'first_lb', 'first_ub', 'duration_lb', 'duration_ub', 'buffer_lb', 'buffer_ub']
        attr_list = [self.num_machine_breakdown_p, self.first_breakdown_buffer_lb, self.first_breakdown_buffer_ub,
                     self.machine_breakdown_duration_lb, self.machine_breakdown_duration_ub,
                     self.breakdown_buffer_lb, self.breakdown_buffer_ub]
        
        default_list = [0.2, 50, 150, 100, 100, 400, 600]
        for name, attr, default in zip(name_list, attr_list, default_list):
            if attr != default:
                name += f'_{attr}'
        return '_'.join(name_list)


def main_gen_breakdowns(breakdown_configs, 
                        n_j=40, n_m=10, op_per_job=20, op_per_mch_min=1, op_per_mch_max=10, 
                        data_dir='train_data_dir', n_data=600, breakdown_suffix='low'):
    config = Config(n_j, n_m, op_per_job=op_per_job, low=1, high=99, 
                    data_suffix='mix', op_per_mch_min=op_per_mch_min, op_per_mch_max=op_per_mch_max)

    instance_name = f'j{n_j}-m{n_m}-t{op_per_job}'
    if op_per_mch_min != 1 or op_per_mch_max != n_m:
        instance_name += f'-m{op_per_mch_min}-{op_per_mch_max}'
    instance_name += f'_{config.data_suffix}'

    dir_path = os.path.join(data_dir, f'instance/{instance_name}')

    breakdown_name = breakdown_configs.get_data_name()
    breakdown_full_name = '_'.join([instance_name, breakdown_name])
    if breakdown_suffix:
        breakdown_full_name += f'-{breakdown_suffix}'
        
    machine_breakdown_path = os.path.join(data_dir, f'machine_breakdown/breakdown_data/{breakdown_full_name}')

    for idx in range(n_data):
        file_path = os.path.join(dir_path, f'data_{idx}.pkl')

        machine_breakdown_path = os.path.join(machine_breakdown_path, f'breakdown_{idx}.pkl')

        if os.path.exists(machine_breakdown_path): continue

        jobs_data, n_machines, n_jobs = pickle.load(open(file_path, 'rb'))

        breakdown_times = generate_machine_breakdown(jobs_data, n_machines,
                                                     num_machine_breakdown_p=num_machine_breakdown_p,
                                                     first_breakdown_buffer_lb=first_breakdown_buffer_lb,
                                                     first_breakdown_buffer_ub=first_breakdown_buffer_ub,
                                                     machine_breakdown_duration_lb=machine_breakdown_duration_lb,
                                                     machine_breakdown_duration_ub=machine_breakdown_duration_ub,
                                                     breakdown_buffer_lb=breakdown_buffer_lb,
                                                     breakdown_buffer_ub=breakdown_buffer_ub)

        pickle.dump(breakdown_times, open(machine_breakdown_path, 'wb'))
###


if __name__ == '__main__':


    # write an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instance_type', type=str, default='standard', choices=['standard', 'breakdown'])
    parser.add_argument('--n_j', type=int, default=20, help='number of jobs')
    parser.add_argument('--n_m', type=int, default=10, help='number of machines')
    parser.add_argument('--op_per_job', type=int, default=30, help='number of operations per job')
    parser.add_argument('--data_dir', type=str, default='train_data_dir', help="directory to store the generated data")
    parser.add_argument('--n_data', type=str, default=600, help='number of FJSP instances to generate')
    
    ### machine breakdown events arguments
    parser.add_argument("--num_machine_breakdown_p", default=0.2, type=float, help="number of machines breakdown probability")
    parser.add_argument("--first_breakdown_buffer_lb", default=50, type=int, help="first breakdown buffer lower bound")
    parser.add_argument("--first_breakdown_buffer_ub", default=150, type=int, help="first breakdown buffer upper bound")
    parser.add_argument("--machine_breakdown_duration_lb", default=100, type=int, help="machine breakdown duration lower bound")
    parser.add_argument("--machine_breakdown_duration_ub", default=100, type=int, help="machine breakdown duration upper bound")
    parser.add_argument("--breakdown_buffer_lb", default=400, type=int, help="breakdown buffer lower bound")
    parser.add_argument("--breakdown_buffer_ub", default=600, type=int, help="breakdown buffer upper bound")
    parser.add_argument("--breakdown_suffix", default='low', type=str, help="breakdown data suffix")
    args = parser.parse_args

    seed = args.seed
    instance_type = args.instance_type
    n_j = args.n_j
    n_m = args.n_m
    op_per_job = args.op_per_job
    data_dir = args.data_dir
    n_data = args.n_data

    op_per_mch_min = 1
    op_per_mch_max = n_m

    random.seed(seed)
    np.random.seed(seed)

    
    data_dir = args.data_dir
    generate_instances(data_dir=data_dir, 
                        n_j=n_j, n_m=n_m, op_per_job=op_per_job, op_per_mch_min=op_per_mch_min, op_per_mch_max=op_per_mch_max,
                        n_data=n_data)

    if instance_type == 'breakdown':
        ### additionally generate breakdown events
        num_machine_breakdown_p = args.num_machine_breakdown_p
        first_breakdown_buffer_lb = args.first_breakdown_buffer_lb
        first_breakdown_buffer_ub = args.first_breakdown_buffer_ub
        machine_breakdown_duration_lb = args.machine_breakdown_duration_lb
        machine_breakdown_duration_ub = args.machine_breakdown_duration_ub
        breakdown_buffer_lb = args.breakdown_buffer_lb
        breakdown_buffer_ub = args.breakdown_buffer_ub
        breakdown_suffix = args.breakdown_suffix

        breakdown_configs = BreakdownConfig(num_machine_breakdown_p=num_machine_breakdown_p,
                                            first_breakdown_buffer_lb=first_breakdown_buffer_lb,
                                            first_breakdown_buffer_ub=first_breakdown_buffer_ub,
                                            machine_breakdown_duration_lb=machine_breakdown_duration_lb,
                                            machine_breakdown_duration_ub=machine_breakdown_duration_ub,
                                            breakdown_buffer_lb=breakdown_buffer_lb,
                                            breakdown_buffer_ub=breakdown_buffer_ub)

        main_gen_breakdowns(breakdown_configs, 
                            n_j=n_j, n_m=n_m, op_per_job=op_per_job, op_per_mch_min=op_per_mch_min, 
                            op_per_mch_max=op_per_mch_max, 
                            data_dir=data_dir, n_data=n_data,
                            breakdown_suffix=breakdown_suffix)
    else:
        print('Invalid instance type')
        sys.exit()