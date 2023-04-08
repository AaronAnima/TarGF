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


class GraphDataset:
    def __init__(self, data_name, is_numpy=False, base_noise_scale=0.01, data_ratio=1):
        self.data_root = f'./expert_datasets/{data_name}/content'
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
        self.histogram_path = f'./expert_datasets/{data_name}/histogram.png'

        self.draw_histogram()

        self.mode = 'multi' if len(self.items_dict.keys()) > 1 else 'single'
        self.is_numpy = is_numpy # if False, then return numpy objects

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
        if self.is_numpy:
            return wall_feat, obj_batch, item_path['room_name']
        
        ''' else, we prepro numpy data into tensors/graphs ''' 
        wall_feat = torch.tensor(wall_feat).float()
        obj_batch = torch.tensor(obj_batch)

        edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0] - 1)  # fully connected
        data_obj = Data(x=obj_batch[:, 0:self.state_dim].float(),
                        geo=obj_batch[:, self.state_dim:self.state_dim + self.size_dim].float(),
                        category=obj_batch[:, -1:].long(),
                        edge_index=edge_obj,
                        wall_feat=wall_feat)
        # augment the data with slight perturbation
        scale = self.scale

        data_obj.x += torch.cat([torch.randn_like(data_obj.x[:, 0:2]), torch.zeros_like(data_obj.x[:, 2:4])],
                                dim=1) * scale
        return data_obj
