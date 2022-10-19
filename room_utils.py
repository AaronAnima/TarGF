from weakref import ref
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import deque
import cv2
import sys
import os
import time
from ipdb import set_trace
import torch
from torch.utils.data import Subset
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from PIL import Image
import pickle
from torchvision.utils import make_grid
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardNormalizer(object):
    def __init__(self, is_norm, writer, update_freq=10, name='default'):
        self.reward_mean = 0
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0
        self.is_norm = is_norm
        ''' to log running mu,std '''
        self.writer = writer
        self.update_freq = update_freq
        self.name = name
    
    def update_mean_std(self, reward):
        self.num_steps += 1
        if self.num_steps == 1:
            # the first step, no need to normalize
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            # running mean, running std
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt/self.num_steps
            self.vk = self.vk + delt * (reward-self.reward_mean)
            self.reward_std = np.sqrt(self.vk/(self.num_steps - 1))
    
    def get_normalized_reward(self, rewards):
        rewards_norm = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        return rewards_norm
    
    def update_writer(self):
        self.writer.add_scalar(f'Episode_rewards/RunningMean_{self.name}', np.mean(self.reward_mean), self.num_steps)
        self.writer.add_scalar(f'Episode_rewards/RunningStd_{self.name}', np.mean(self.reward_std), self.num_steps)

    def update(self, reward, is_eval=False):
        if not is_eval and self.is_norm:
            if type(reward) is np.ndarray:
                for item in reward:
                    self.update_mean_std(item)
            else:
                self.update_mean_std(reward)
            reward = self.get_normalized_reward(reward)
            ''' log the running mean/std '''
            if self.num_steps % self.update_freq == 0:
                self.update_writer()
        return reward


def chamfer_dist(x, y, metric='l2'):
    x = x.reshape(-1, 2)
    y = y.reshape(-1, 2)
    # x: [nx, dim], y: [ny, dim]
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    return dist

def my_dist(state1, state2):
    # state1: [num_obj, 7]
    # state2: [num_obj, 7]
    return np.sum((state1[:, 0:2] - state2[:, 0:2])**2)

def calc_coverage(room_name_to_gt_states, room_name_to_states):
    res = []
    room_names = list(room_name_to_gt_states.keys())
    for room_name in room_names:
        gt_states = room_name_to_gt_states[room_name]
        states = room_name_to_states[room_name]
        min_dists = []
        for gt_state in gt_states:
            min_dist = np.min(np.array([my_dist(gt_state, state) for state in states]))
            min_dists.append(min_dist)
        res.append(np.min(np.array(min_dists))) 
    res = np.array(res)
    return np.mean(res), np.std(res)


def snapshot(env, file_name):
    img = env.render(256)
    cv2.imwrite(file_name, img)


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def string2bool(str_bool):
    return str_bool == 'True'


def save_video(env, states, save_path, simulation=False, fps = 50, render_size = 256, suffix='avi'):
    # states: [state, ....]
    # state: (60, )
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        env.set_state(state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))


def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for item in images:
        out.write(item)
    out.release()


def split_dataset(dataset, seed, test_ratio, full_train='False'):
    random.seed(seed)

    # get train and test indices
    # get test split according to mode
    if dataset.mode == 'multi':
        items_dict = dataset.items_dict
        test_num = int(len(items_dict.keys()) * test_ratio)
        test_keys = random.sample(list(items_dict.keys()), test_num)
        test_indics = []
        for key in test_keys:
            test_indics += items_dict[key]
    else:
        test_num = int(len(dataset) * test_ratio)
        test_indics = random.sample(range(len(dataset)), test_num)

    # get train according to test
    train_indics = list(set(range(len(dataset))) - set(test_indics))

    # assertion of indices
    assert len(train_indics) + len(test_indics) == len(dataset)
    assert len(set(train_indics) & set(test_indics)) == 0

    # split dataset according to indices
    test_dataset = Subset(dataset, test_indics)
    train_dataset = dataset if full_train == 'True' else Subset(dataset, train_indics)

    # log infos
    infos_dict = {
        'test_indices': test_indics,
        'train_indices': train_indics,
        'room_num': len(dataset.items_dict.keys()),
    }
    return train_dataset, test_dataset, infos_dict

class GraphDataset4RL:
    def __init__(self, data_name, base_noise_scale=0.01):
        self.data_root = f'../ExpertDatasets/{data_name}/content'
        self.folders_path = os.listdir(self.data_root)
        self.items = []
        self.items_dict = {}
        ptr = 0
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            assert len(files_list) % 2 == 0
            if files not in self.items_dict.keys():
                self.items_dict[files] = []
            for idx in range(len(files_list)//2):
                item = {
                    'wall_path': cur_folder_path+f'{idx+1}_wall.pickle',
                    'obj_path': cur_folder_path+f'{idx+1}_obj.pickle',
                    'room_name': files,
                }
                self.items.append(item)
                self.items_dict[files].append(ptr)
                ptr += 1
        self.state_dim = 4
        self.size_dim = 2
        self.scale = base_noise_scale
        self.histogram_path = f'../ExpertDatasets/{data_name}/histogram.png'

        self.draw_histogram()

        self.mode = 'multi' if len(self.items_dict.keys()) > 1 else 'single'

    def draw_histogram(self):
        plt.figure(figsize=(10,10))
        histogram = []
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            histogram.append(len(files_list) // 2)
        histogram = np.array(histogram)
        plt.hist(histogram, bins=4)
        plt.title(f'Total room num: {len(self.folders_path)}')
        plt.savefig(self.histogram_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_path = self.items[item]
        with open(item_path['wall_path'], 'rb') as f:
            wall_feat = pickle.load(f)
        with open(item_path['obj_path'], 'rb') as f:
            obj_batch = pickle.load(f)
        return wall_feat, obj_batch, item_path['room_name']

class GraphDataset:
    def __init__(self, data_name, base_noise_scale=0.01, data_ratio=1):
        self.data_root = f'../ExpertDatasets/{data_name}/content'
        self.folders_path = os.listdir(self.data_root)
        self.items = []
        self.items_dict = {}
        ptr = 0
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            assert len(files_list) % 2 == 0
            if files not in self.items_dict.keys():
                self.items_dict[files] = []
            for idx in range(int((len(files_list) * data_ratio) // 2)):
                item = {
                    'wall_path': cur_folder_path + f'{idx + 1}_wall.pickle',
                    'obj_path': cur_folder_path + f'{idx + 1}_obj.pickle',
                    'room_name': files,
                }
                self.items.append(item)
                self.items_dict[files].append(ptr)
                ptr += 1

        self.state_dim = 4
        self.size_dim = 2
        self.scale = base_noise_scale
        self.histogram_path = f'../ExpertDatasets/{data_name}/histogram.png'

        self.draw_histogram()

        self.mode = 'multi' if len(self.items_dict.keys()) > 1 else 'single'

    def draw_histogram(self):
        plt.figure(figsize=(10, 10))
        histogram = []
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            histogram.append(len(files_list) // 2)
        histogram = np.array(histogram)
        plt.hist(histogram, bins=4)
        plt.title(f'Total room num: {len(self.folders_path)}')
        plt.savefig(self.histogram_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_path = self.items[item]
        with open(item_path['wall_path'], 'rb') as f:
            wall_feat = pickle.load(f)
        with open(item_path['obj_path'], 'rb') as f:
            obj_batch = pickle.load(f)
        wall_feat = torch.tensor(wall_feat).float()
        obj_batch = torch.tensor(obj_batch)

        edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0] - 1)  # fully connected
        data_obj = Data(x=obj_batch[:, 0:self.state_dim].float(),
                        geo=obj_batch[:, self.state_dim:self.state_dim + self.size_dim].float(),
                        category=obj_batch[:, -1:].long(),
                        edge_index=edge_obj)
        # augment the data with slight perturbation
        scale = self.scale

        data_obj.x += torch.cat([torch.randn_like(data_obj.x[:, 0:2]), torch.zeros_like(data_obj.x[:, 2:4])],
                                dim=1) * scale
        return wall_feat, data_obj, item_path['room_name']

def prepro_dynamic_graph(state, state_dim=4, cuda=False):
    wall_feat, obj_batch = state
    wall_feat = torch.tensor(wall_feat).float()
    obj_batch = torch.tensor(obj_batch)
    edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0] - 1)
    data_obj = Data(x=obj_batch[:, 0:state_dim].float(),
                    geo=obj_batch[:, state_dim:state_dim+2].float(),
                    category=obj_batch[:, -1:].long(),
                    edge_index=edge_obj)
    if cuda:
        wall_feat = wall_feat.to(device)
        data_obj = data_obj.to(device)
    return wall_feat, data_obj

def prepro_state(state, state_dim=4, cuda=False):
    wall_feat, obj_batch = state
    wall_feat = torch.tensor(wall_feat).float()
    obj_batch = torch.tensor(obj_batch)
    edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0] - 1)
    data_obj = Data(x=obj_batch[:, 0:state_dim].float(),
                    geo=obj_batch[:, state_dim:state_dim+2].float(),
                    category=obj_batch[:, -1:].long(),
                    edge_index=edge_obj)
    if cuda:
        wall_feat = wall_feat.to(device)
        data_obj = data_obj.to(device)
    return wall_feat, data_obj

def pre_pro_dynamic_vec(vec, dim=1, cuda=False):
    vec = torch.tensor(vec).view(-1, dim).float()
    data = Data(x=vec)
    return data

def prepro_graph_batch(states):
    if not isinstance(states[0][-1], Data):
        states = [prepro_dynamic_graph(state) for state in states]
    
    wall_batch = torch.tensor([state[0] for state in states]).unsqueeze(1).to(device)
    samples_batch = []
    ptr = [0]
    x = []
    geo = []
    category = []
    cur_ptr = 0
    for idx, state in enumerate(states):
        cur_num_nodes = state[-1].x.shape[0]
        for _ in range(cur_num_nodes):
            samples_batch.append(idx)
        x.append(state[-1].x)
        geo.append(state[-1].geo)
        category.append(state[-1].category)
        cur_ptr += cur_num_nodes
        ptr.append(cur_ptr)
    
    samples_batch = torch.tensor(samples_batch, dtype=torch.int64).to(device)
    ptr = torch.tensor(ptr, dtype=torch.int64).to(device)
    x = torch.cat(x, dim=0).to(device)
    geo = torch.cat(geo, dim=0).to(device)
    category = torch.cat(category, dim=0).to(device)
    edge_index = knn_graph(x, k=10, batch=samples_batch)
    obj_batch = Data(x=x, edge_index=edge_index, batch=samples_batch, ptr=ptr, geo=geo, category=category)
    return wall_batch, obj_batch

def batch_to_data_list(my_graph_batch, ref_batch):
    wall_feats = [item[0].numpy() for item in ref_batch[0].cpu()]
    obj_batch_large = torch.cat([my_graph_batch[0].cpu(), ref_batch[1].geo.cpu(), ref_batch[1].category.cpu()], dim=-1).cpu().numpy()
    ptr = ref_batch[1].ptr
    obj_batchs = [obj_batch_large[ptr[i]:ptr[i+1], :] for i in range(ref_batch[0].shape[0])]
    return [prepro_state((wall_feat, obj_batch)) for wall_feat, obj_batch in zip(wall_feats, obj_batchs)]

class Timer:
    def __init__(self, writer):
        self.writer = writer
        self.counter = 0
        self.t_s = deque(maxlen=1000)
    
    def set(self):
        t_s = time.time()
        self.t_s.append(t_s)
    
    def log(self, name):
        self.counter += 1
        self.writer.add_scalar(f'Timer/{name}', time.time() - self.t_s.pop(), self.counter)

