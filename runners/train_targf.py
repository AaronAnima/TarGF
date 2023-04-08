import os
import time
import pickle
from tqdm import trange
import functools
from ipdb import set_trace
import gym
import ebor

import torch
import torch.optim as optim
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


from utils.datasets import split_dataset, GraphDataset, collect_ball_dataset
from utils.visualisations import visualize_room_states, visualize_ball_states
from algorithms.sgm import marginal_prob_std, diffusion_coeff
from algorithms.sgm import loss_fn_cond, loss_fn_uncond
from algorithms.sgm import cond_ode_vel_sampler, ode_sampler
from algorithms.sgm_nets import CondScoreModelGNN, ScoreModelGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(configs, env):
    if configs.env_type == 'Room':
        # create dataset and dataloader
        # split train and test, we only train GF on seen rooms (train-dataset)
        dataset = GraphDataset(f'{configs.data_name}', base_noise_scale=configs.base_noise_scale)
        train_dataset, _, _ = split_dataset(
            dataset,
            configs.seed,
            configs.test_ratio,
            full_train=configs.full_train
        )
        dataloader_train = DataLoader(train_dataset, batch_size=configs.batch_size_gf, shuffle=True, num_workers=configs.workers)
        dataloader_vis = DataLoader(train_dataset, batch_size=configs.vis_col**2, shuffle=True, num_workers=configs.workers)
    elif configs.env_type == 'Ball':
        dataset_path = f'./expert_datasets/{configs.data_name}.pth'
        if os.path.exists(dataset_path):
            print('### found existing dataset ###')
            with open(dataset_path, 'rb') as f:
                data_samples = pickle.load(f)
            dataset = torch.tensor(data_samples)
            dataset = dataset[:configs.n_samples]
        else:
            print('### not found existding dataset, start collecting data ###')
            ts = time.time()
            data_samples = collect_ball_dataset(configs, env)
            with open(dataset_path, 'wb') as f:
                pickle.dump(data_samples, f)
            dataset = torch.tensor(data_samples)
            print('### data collection done! takes {:.2f} to collect {} samples ###'.format(time.time() - ts, configs.n_samples))
        
        ''' Prepare dataloader '''
        print(f'Dataset Size: {len(dataset)}')
        print(f'Dataset Shape: {dataset.shape}')
        dataset = dataset.reshape(-1, dataset.shape[-1])

        # prepare graph-based dataset
        k = configs.num_objs - 1 # fully connected graph
        edge = knn_graph(dataset[0].reshape(configs.num_objs, 2+1)[:, :2], k, loop=False)
        dataset = list(map(lambda x: Data(x=x[:, :2].float(),  edge_index=edge, c=x[:, -1].long()), dataset.reshape(dataset.shape[0], configs.num_objs, 2+1)))
        # convert dataset to dataloader
        dataloader_train = DataLoader(dataset, batch_size=configs.batch_size_gf, shuffle=True, num_workers=configs.workers)
        dataloader_vis = DataLoader(dataset, batch_size=configs.vis_col**2, shuffle=True, num_workers=configs.workers)
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return dataloader_train, dataloader_vis


def get_score_network(configs, marginal_prob_std_fn):
    if configs.env_type == 'Room':     
        score_net = CondScoreModelGNN(
            marginal_prob_std_fn,
            hidden_dim=configs.hidden_dim,
            embed_dim=configs.embed_dim,
        )
    elif configs.env_type == 'Ball':
        score_net = ScoreModelGNN(
            marginal_prob_std_fn, 
            num_classes=configs.num_classes, 
            hidden_dim=configs.hidden_dim,
            embed_dim=configs.embed_dim,
            device=device,
        )
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return score_net


def get_functions(configs, env):
    if configs.env_type == 'Room': 
        sampler_fn = functools.partial(cond_ode_vel_sampler, num_steps=configs.sampling_steps)
        loss_fn = functools.partial(loss_fn_cond, batch_size=configs.batch_size_gf)
        vis_fn = functools.partial(visualize_room_states, batch_size=configs.batch_size_gf)
    elif configs.env_type == 'Ball':
        sampler_fn = functools.partial(ode_sampler, num_objs=configs.num_objs, num_steps=configs.sampling_steps)
        loss_fn = functools.partial(loss_fn_uncond, num_objs=configs.num_objs)
        vis_fn = functools.partial(visualize_ball_states, env=env, configs=configs)
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return loss_fn, sampler_fn, vis_fn

def get_env(configs):
    if configs.env_type == 'Room': 
        env = None
    elif configs.env_type == 'Ball':
        env_name = '{}-{}Ball{}Class-v0'.format(configs.pattern, configs.num_objs, configs.num_classes)
        env = gym.make(env_name)
        env.seed(configs.seed)
        env.reset()
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return env


def gf_trainer(configs, log_dir, writer):
    # get an env for visualisation, it can be None (if not used latter)
    env = get_env(configs)

    # get dataloaders
    dataloader_train, dataloader_vis = get_dataloaders(configs, env)

    # init SDE-related params
    sigma = configs.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # create models, optimizers, and loss
    score = get_score_network(configs, marginal_prob_std_fn)
    score.to(device)
    optimizer = optim.Adam(score.parameters(), lr=configs.lr, betas=(configs.beta1, 0.999))

    # determine loss, sde-sampler and visualisation functions
    loss_fn, sde_sampler_fn, vis_fn = get_functions(configs, env)

    print("Starting Training Loop...")
    for epoch in trange(configs.n_epoches):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader_train):
            # calc score-matching loss
            loss = loss_fn(score, real_data, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader_train))

        # start eval
        if (epoch+1) % configs.vis_freq_gf == 0:
            ref_batch = next(iter(dataloader_vis))
            with torch.no_grad():
                _, res_ode_vel = sde_sampler_fn(
                    score,
                    marginal_prob_std_fn,
                    diffusion_coeff_fn,
                    ref_batch,
                    t0=configs.t0,
                    batch_size=configs.vis_col**2,
                )
                
                # visualise generated state
                vis_fn(
                    res_ode_vel.to(device),
                    ref_batch=ref_batch,
                    writer=writer,
                    nrow=configs.vis_col,
                    epoch=epoch,
                    suffix='generated_samples',
                )
                # visualise the ground truth states
                vis_fn(
                    ref_batch,
                    ref_batch=ref_batch,
                    writer=writer,
                    nrow=configs.vis_col,
                    epoch=epoch,
                    suffix='real_data',
                )
                
                # save model
                ckpt_path = os.path.join('./logs', log_dir, 'score.pt')
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(score, f)
                score.to(device)


