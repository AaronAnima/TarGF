import os
import sys
import pickle
import numpy as np
from tqdm import trange
import functools
from ipdb import set_trace

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch_geometric.loader import DataLoader

from envs.Room.RoomArrangement import SceneSampler 

# for room rearrangement
def visualize_room_states(vis_states, ref_batch, writer, nrow, epoch, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    sampler = SceneSampler(gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    if not isinstance(vis_states, torch.Tensor):
        vis_states = vis_states[1].x # obj_batch.x
    wall_batch, obj_batch, names = ref_batch
    ptr = obj_batch.ptr
    imgs = []
    for idx in range(nrow**2):
        cur_state = vis_states[ptr[idx]: ptr[idx+1]]
        sim = sampler[names[idx]]
        sim.normalize_room()
        
        # states_to_eval
        cur_state = torch.cat(
            [cur_state,
             obj_batch.geo[ptr[idx]: ptr[idx+1]],
             obj_batch.category[ptr[idx]: ptr[idx+1]]], dim=-1)
        sim.set_state(cur_state.cpu().numpy(), wall_batch[idx].cpu().numpy())
        img = sim.take_snapshot(512, height=10.0)
        imgs.append(img)

        # close sim
        sim.disconnect()

    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    # each row: [synthesised, perturbed, GT]
    grid = make_grid(ts_imgs.float(), padding=2, nrow=2, normalize=True)
    writer.add_image(f'Images/{suffix}', grid, epoch)

# for ball rearrangement:
def visualize_ball_states(vis_states, ref_batch, writer, nrow, epoch, suffix, env=None, configs=None):
    # ref_batch: place holder
    # states -> images
    # if scores, then try to visualize gradient
    if not isinstance(vis_states, torch.Tensor):
        # visualise real_ata
        vis_states = torch.cat([vis_states.x, vis_states.c.unsqueeze(-1)], dim=-1)
        vis_states = vis_states.view(-1, configs.num_objs*3)
        vis_states = vis_states
    imgs = []
    for obj_state in vis_states:
        obj_state = obj_state.detach().cpu().numpy()
        obj_state = env.unflatten_states([obj_state])[0]
        env.set_state(obj_state)
        img = env.render(img_size=256)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    writer.add_image(f'Images/{suffix}', grid, epoch)

