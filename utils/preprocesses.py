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
