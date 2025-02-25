import random
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


### General utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.imap(func, tasks))

def multithread(func, tasks, cpus=None, show_bar=True):
    bar = lambda x: tqdm(x, total=len(tasks)) if show_bar else x
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in bar(tasks)]
    with ThreadPool(cpus or os.cpu_count()) as pool:
        return list(bar(pool.imap(func, tasks)))


class RunningNormalization:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.n_samples = torch.tensor(0.).to(device)
        self.running_mean = torch.tensor(0.).to(device)
        self.running_variance = torch.tensor(0.).to(device)

    def load(self, load_file):
        self.n_samples, self.running_mean, self.running_variance = torch.load(load_file, map_location=self.device)

    def save(self, save_file, save_device=torch.device('cpu')):
        torch.save([self.n_samples.to(save_device), self.running_mean.to(save_device), 
                    self.running_variance.to(save_device)],  save_file)

    def update(self, batch_data):
        batch_data = batch_data.to(self.device)

        batch_mean = torch.mean(batch_data, dim=0)
        batch_variance = torch.var(batch_data, dim=0, unbiased=False)
        batch_size = batch_data.size(0)

        total_size = self.n_samples + batch_size
        total_size_tensor = torch.tensor(total_size, dtype=torch.float32, device=self.device)
        
        new_mean = (self.running_mean * self.n_samples + batch_mean * batch_size) / total_size_tensor
        new_variance = ((self.n_samples * self.running_variance + batch_size * batch_variance) +
                        ((batch_mean - self.running_mean) ** 2) * self.n_samples * batch_size / total_size_tensor) / total_size_tensor

        self.n_samples = total_size
        self.running_mean = new_mean
        self.running_variance = new_variance

    def normalize(self, batch_data):
        batch_data = batch_data.to(self.device)
        return (batch_data - self.running_mean) / torch.sqrt(self.running_variance + 1e-8)


class RunningStats:
    '''
    Tracks first and second moments (mean and variance) of a streaming time series
    https://github.com/joschu/modular_rl
    http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, stats={}):
        self.n = self.mean = self._nstd = 0
        self.__dict__.update(stats)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean
            self.mean = old_mean + (x - old_mean) / self.n
            self._nstd = self._nstd + (x - old_mean) * (x - self.mean)
    @property
    def var(self):
        return self._nstd / (self.n - 1) if self.n > 1 else np.square(self.mean)
    @property
    def std(self):
        return np.sqrt(self.var)


#### IO
def store_dataset(dataset, data_dir='data', data_name='train', data_index=-1):
    if data_index >= 0:
        data_name = f'{data_name}_{data_index}.pkl'
    else:
        data_name = f'{data_name}.pkl'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pickle.dump(dataset, open(os.path.join(data_dir, data_name), 'wb'))

def load_dataset(data_dir='data', data_name='train', data_index=-1):
    if data_index >= 0:
        data_name = f'{data_name}_{data_index}.pkl'
    else:
        data_name = f'{data_name}.pkl'

    if not os.path.exists(os.path.join(data_dir, data_name)):
        print(f'{os.path.join(data_dir, data_name)} does not exist')
        dataset = []
    else:
        dataset = pickle.load(open(os.path.join(data_dir, data_name), 'rb')) 
        # at least have some previous tasks and some new tasks
        dataset = [data for data in dataset if (torch.tensor(data['data'].x_tasks)[:, 7] == 1).sum() > 0
                                            and (torch.tensor(data['data'].x_tasks)[:, 7] == 1).sum() < len(data['data'].x_tasks)]
                                               
    return dataset  # [:10]


def save_model(model, save_dir='model', save_name='model', model_epoch=0):
    save_name = f'{save_name}_{model_epoch}.pth'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))


def load_model(model, load_dir='model', load_name='model', model_epoch=0, device=torch.device('cpu')):
    load_name = f'{load_name}_{model_epoch}.pth'
    print('Try to load model from', os.path.join(load_dir, load_name))
    if os.path.exists(os.path.join(load_dir, load_name)):
        print(f'Find model! Load model from {load_name}!')
        try:
            model.load_state_dict(torch.load(os.path.join(load_dir, load_name), map_location=device), strict=False)
        except:
            print('Failed to load model!')
            return
        print('Model loaded successfully!')
        
    else:
        print('Failed to load model!')
    

### Learning Components
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """
    def __init__(self, edge_dim, emb_size=64):
        super().__init__('add')
        self.edge_dim = edge_dim

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_size),
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_size)
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):

        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output

