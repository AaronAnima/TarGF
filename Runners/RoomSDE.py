import pybullet as p
import pybullet_data
from ipdb import set_trace
import os
import sys
import time
import math
import copy
import pickle
import cv2
import numpy as np
from tqdm import tqdm, trange
from collections import deque
import argparse
import functools
import random
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.data import Subset
 
 
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, EdgeConv
from torch_scatter import scatter_max, scatter_mean

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from room_utils import exists_or_mkdir, string2bool, images_to_video, save_video, split_dataset, GraphDataset
from Algorithms.RoomSDE import marginal_prob_std, diffusion_coeff, loss_fn_cond, cond_ode_vel_sampler
from Algorithms.RoomSDENet import DualScore, CondScoreModelGNN
from Envs.RoomArrangement import SceneSampler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_states(eval_states, room_type, ref_batch, logger, nrow, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    sampler = SceneSampler(room_type, gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    # sampler = SceneSampler(room_type, gui='DIRECT')
    wall_batch, obj_batch, names = ref_batch
    ptr = obj_batch.ptr
    # set_trace()
    imgs = []
    for idx in range(nrow**2):
        cur_state = eval_states[ptr[idx]: ptr[idx+1]]
        # set_trace()
        sim = sampler[names[idx]]
        # img = sim.take_snapshot(512, height=10.0)
        # imgs.append(img)
        sim.normalize_room()

        # generated
        cur_state = torch.cat(
            [cur_state,
             obj_batch.geo[ptr[idx]: ptr[idx+1]],
             obj_batch.category[ptr[idx]: ptr[idx+1]]], dim=-1)
        sim.set_state(cur_state.cpu().numpy(), wall_batch[idx].cpu().numpy())
        img = sim.take_snapshot(512, height=10.0)
        imgs.append(img)

        # perturbed room
        cur_state = torch.cat(
            [obj_batch.x[ptr[idx]: ptr[idx + 1]],
             obj_batch.geo[ptr[idx]: ptr[idx + 1]],
             obj_batch.category[ptr[idx]: ptr[idx + 1]]], dim=-1)
        sim.set_state(cur_state.cpu().numpy(), wall_batch[idx].cpu().numpy())
        img = sim.take_snapshot(512, height=10.0)
        imgs.append(img)

        # GT room
        sim.reset_checkpoint()
        img = sim.take_snapshot(512, height=10.0)
        imgs.append(img)

        # close sim
        sim.disconnect()

    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    # each row: [synthesised, perturbed, GT]
    grid = make_grid(ts_imgs.float(), padding=2, nrow=3, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--test_decay', type=str, default='False')
    parser.add_argument('--room_type', type=str, default='bedroom')
    parser.add_argument('--score_mode', type=str, default='target')
    parser.add_argument('--n_epoches', type=int, default=10000)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--t0', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--full_train', type=str)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--base_noise_scale', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=25)
    # load args
    args = parser.parse_args()

    # create log path
    exists_or_mkdir('../logs')
    ckpt_path = f'../logs/{args.log_dir}/'
    exists_or_mkdir(ckpt_path)
    eval_path = f'../logs/{args.log_dir}/test_batch/'
    exists_or_mkdir(eval_path)
    tb_path = f'../logs/{args.log_dir}/tb'
    exists_or_mkdir(tb_path)

    # init writer
    writer = SummaryWriter(tb_path)

    # create dataset and dataloader
    # split train and test
    test_col = 2
    
    dataset = GraphDataset(f'{args.data_name}', args.base_noise_scale)
    
    train_dataset, test_dataset, infos= split_dataset(dataset,
                                                       args.seed,
                                                       args.test_ratio,
                                                       full_train=args.full_train)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloader_test = DataLoader(test_dataset, batch_size=test_col**2, shuffle=True, num_workers=args.workers)

    # init SDE-related params
    sigma = args.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # create models, optimizers, and loss
    score = CondScoreModelGNN(
        marginal_prob_std_fn,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        wall_dim=1, mode=args.score_mode)

    score.to(device)

    optimizer = optim.Adam(score.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    print("Starting Training Loop...")
    for epoch in trange(args.n_epoches):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader_train):
            # augment data first
            real_data_wall, real_data_obj, _ = real_data
            real_data_wall = real_data_wall.to(device)
            real_data_obj = real_data_obj.to(device)

            # calc score-matching loss
            loss = loss_fn_cond(score, (real_data_wall, real_data_obj), args.batch_size, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader_train))

        # start eval
        # if (epoch+1) % args.eval_freq == 0:
        if epoch % args.eval_freq == 0:
            test_batch = next(iter(dataloader_test))
            with torch.no_grad():
                t0 = args.t0
                # use different suffix to identify each sampler's results
                # in_process_sample_ode, res_ode = cond_ode_sampler(score,
                #                                           marginal_prob_std_fn,
                #                                           diffusion_coeff_fn,
                #                                           test_batch,
                #                                           t0=t0,
                #                                           num_steps=2000,
                #                                           batch_size=test_col**2)
                dual_score = DualScore(tar_score=score if args.score_mode == 'target' else None,
                                       sup_score=score if args.score_mode == 'support' else None)
                in_process_sample_ode_vel, res_ode_vel = cond_ode_vel_sampler(dual_score,
                                                                              marginal_prob_std_fn,
                                                                              diffusion_coeff_fn,
                                                                              test_batch,
                                                                              t0=t0,
                                                                              num_steps=500,
                                                                              max_pos_vel=0.6,
                                                                              max_ori_vel=0.6,
                                                                              scale=0.04,
                                                                              is_decay=args.test_decay == 'True',
                                                                              batch_size=test_col**2)

                # visualize_states(res_ode,
                #                  'bedroom',
                #                  test_batch,
                #                  writer,
                #                  test_col,
                #                  suffix='Neural_ODE_sampler|Final Sample')
                # res_ode_vel: [1, node_num, state_dim]

                # visualisation is irrelevant to the env, just related to sampler and igibson sim
                visualize_states(res_ode_vel.to(device)[0],
                                 args.room_type,
                                 test_batch,
                                 writer,
                                 test_col,
                                 suffix='Neural_ODE_vel_sampler|Final Sample')
                # save model
                with open(ckpt_path + f'score.pt', 'wb') as f:
                    pickle.dump(score, f)
                # torch.save(score.cpu().state_dict(), ckpt_path + f'score.pt')
                score.to(device)

            # if (epoch + 1) % 20 == 0 and epoch > 50:
            #     # visualize the process
            #     save_video(in_process_sample.cpu().numpy(), test_env, save_path=eval_path + f'{epoch+1}')
            #
            #     # save model
            #     torch.save(score.cpu().state_dict(), ckpt_path + f'score.pt')
            #     score.to(device)


