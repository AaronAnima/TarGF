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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.datasets import split_dataset, GraphDataset
from algorithms.sgm import marginal_prob_std, diffusion_coeff, loss_fn_cond, cond_ode_vel_sampler
from algorithms.sgm_nets import CondScoreModelGNN
from envs.Room.RoomArrangement import SceneSampler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    grid = make_grid(ts_imgs.float(), padding=2, nrow=3, normalize=True)
    writer.add_image(f'Images/{suffix}', grid, epoch)

def gf_trainer(configs, log_dir, writer):
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

    # init SDE-related params
    sigma = configs.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    # create models, optimizers, and loss
    score = CondScoreModelGNN(
        marginal_prob_std_fn,
        hidden_dim=configs.hidden_dim,
        embed_dim=configs.embed_dim,
    )

    score.to(device)

    optimizer = optim.Adam(score.parameters(), lr=configs.lr, betas=(configs.beta1, 0.999))

    print("Starting Training Loop...")
    for epoch in trange(configs.n_epoches):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader_train):
            # calc score-matching loss
            loss = loss_fn_cond(score, real_data, marginal_prob_std_fn, batch_size=configs.batch_size_gf)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader_train))

        # start eval
        if (epoch+1) % configs.eval_freq_gf == 0:
            test_batch = next(iter(dataloader_vis))
            with torch.no_grad():
                t0 = configs.t0
                _, res_ode_vel = cond_ode_vel_sampler(
                    score,
                    marginal_prob_std_fn,
                    diffusion_coeff_fn,
                    test_batch,
                    t0=t0,
                    batch_size=configs.vis_col**2,
                )
                # visualise generated state
                visualize_room_states(
                    res_ode_vel.to(device),
                    ref_batch=test_batch,
                    writer=writer,
                    nrow=configs.vis_col,
                    epoch=epoch,
                    suffix='generated_samples',
                )
                # visualise the ground truth states
                visualize_room_states(
                    test_batch,
                    ref_batch=test_batch,
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


