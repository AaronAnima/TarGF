import os
import sys
import time
import pickle
import cv2
import numpy as np
from tqdm import tqdm, trange
import argparse
import functools
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ball_utils import exists_or_mkdir, save_video
from Envs.SortingBall import RLSorting
from Envs.PlacingBall import RLPlacing
from Envs.HybridBall import RLHybrid
ENV_DICT = {'sorting': RLSorting, 'placing': RLPlacing, 'hybrid': RLHybrid}
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, loss_fn_state, loss_fn_img, ode_sampler, pc_sampler_img, pc_sampler_state
from Networks.BallSDENet import ScoreModelGNN, ScoreNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def snapshot(env, file_name):
    img = env.render(256)
    cv2.imwrite(file_name, img)

def collect_data(env, env_name, num_samples, num_boxes, is_state=True, obs_size=64, is_random=False, is_gui=False, suffix=''):
    exists_or_mkdir('../ExpertDatasets/')
    debug_path = f'../ExpertDatasets/{env_name}_{suffix}/'
    exists_or_mkdir(debug_path)
    samples = []
    debug_freq = 100
    num_boxes
    
    with tqdm(total=num_samples) as pbar:
        while len(samples) < num_samples:
            cur_state = env.reset(is_random=is_random)
            if is_state:
                samples.append(cur_state)
            else:
                cur_obs = env.render(obs_size)
                samples.append(cur_obs)
            if len(samples) % debug_freq == 0:
                snapshot(env, f'{debug_path}debug_{len(samples)//debug_freq}.png')
            pbar.update(1)
    samples = np.stack(samples, axis=0)
    return samples


def visualize_states(eval_states, env, logger, nrow, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    for box_state in eval_states:
        env.set_state(box_state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


class MyImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        # [0, 1] -> [-1, 1]
        image = image*2 - 1
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--inp_mode', type=str, default='state')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--sampler', type=str, default='ode')
    parser.add_argument('--sigma', type=float, default=25.)
    parser.add_argument('--n_box', type=int, default=7)
    parser.add_argument('--n_epoches', type=int, default=10000)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--visualize_freq', type=int, default=10)
    parser.add_argument('--video_freq', type=int, default=100)
    parser.add_argument('--test_num', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--t0', type=float, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--render_size', type=int, default=256)
    parser.add_argument('--obs_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    # load args
    args = parser.parse_args()
    n_box = args.n_box
    n_samples = args.n_samples
    batch_size = args.batch_size
    workers = args.workers
    lr = args.lr
    beta1 = args.beta1
    render_size = args.render_size
    dataset_path = f'../ExpertDatasets/{args.data_name}.pth'
    inp_mode = args.inp_mode
    is_state = (inp_mode == 'state')
    if not is_state:
        assert inp_mode == 'image'
    obs_size = args.obs_size
    
    # create log path
    exists_or_mkdir('../logs')
    ckpt_path = f'../logs/{args.log_dir}/'
    exists_or_mkdir(ckpt_path)
    eval_path = f'../logs/{args.log_dir}/test_batch/'
    exists_or_mkdir(eval_path)
    tb_path = f'../logs/{args.log_dir}/tb'
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)

    ''' init my env '''
    exp_data = None
    PB_FREQ = 4
    dt = 0.02
    MAX_VEL = 0.3
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes': args.n_box,
        'exp_data': exp_data,
        'time_freq': time_freq * PB_FREQ,
        'is_gui': False,
        'max_action': MAX_VEL,
    }
    env_class = ENV_DICT[args.env]
    env = env_class(**env_kwargs)
    env.seed(args.seed)
    env.reset()

    ''' Prepare training data '''
    if os.path.exists(dataset_path):
        print('### found existing dataset ###')
        with open(dataset_path, 'rb') as f:
            data_samples = pickle.load(f)
        dataset = torch.tensor(data_samples)
        dataset = dataset[: n_samples]
    else:
        print('### not found existing dataset, start collecting data ###')
        ts = time.time()
        data_samples = collect_data(env, args.env, n_samples, n_box, is_random=False, suffix=args.data_name, is_state=is_state, obs_size=obs_size)
        with open(dataset_path, 'wb') as f:
            pickle.dump(data_samples, f)
        dataset = torch.tensor(data_samples)
        print('### data collection done! takes {:.2f} to collect {} samples ###'.format(time.time() - ts, n_samples))
    
    ''' Prepare dataloader '''
    print(f'Dataset Size: {len(dataset)}')
    print(f'Dataset Shape: {dataset.shape}')
    if is_state:
        dataset = dataset.reshape(-1, dataset.shape[-1])
        # convert dataset
        k = 3*n_box - 1
        edge = knn_graph(dataset[0].reshape(n_box*3, 2), k, loop=False)
        dataset = list(map(lambda x: Data(x=x, edge_index=edge), dataset.reshape(dataset.shape[0], n_box*3, 2)))
        # convert samples to dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    else:
        dataset = MyImageDataset(dataset.numpy(), transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ''' Init Model '''
    # init SDE-related params
    sigma = args.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    if is_state:
        # create models, optimizers, and loss
        mode = 'target' if args.env in ['sorting', 'hybrid', 'npa', 'sorting6'] else 'support'
        score = ScoreModelGNN(marginal_prob_std_fn, n_box=n_box, mode=mode, device=device)
    else:
        score = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    score.to(device)

    optimizer = optim.Adam(score.parameters(), lr=lr, betas=(beta1, 0.999))

    num_epochs = args.n_epoches
    print("Starting Training Loop...")
    for epoch in trange(num_epochs):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader):
            # augment data first
            real_data = real_data.to(device)
            # calc score-matching loss
            if is_state:
                loss = loss_fn_state(score, real_data, marginal_prob_std_fn, n_box=n_box)
            else:
                loss = loss_fn_img(score, real_data, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader))
        # start eval
        with torch.no_grad():
            if (epoch + 1) % args.visualize_freq == 0:
                test_num = args.test_num
                if is_state:
                    # use different suffix to identify each sampler's results
                    t0 = args.t0
                    if args.sampler == 'ode':
                        in_process_sample, res = ode_sampler(score, marginal_prob_std_fn, diffusion_coeff_fn,
                                                            t0=t0, batch_size=test_num, n_box=n_box, num_steps=2000)
                    else: 
                        assert args.sampler == 'pc'
                        in_process_sample, res = pc_sampler_state(score, marginal_prob_std_fn, diffusion_coeff_fn,
                                                            t0=t0, batch_size=test_num, n_box=n_box, num_steps=2000)
                                                        
                    visualize_states(res.view(test_num, -1), env, writer, 4, suffix='Final Sample')
                    assert args.batch_size >= test_num
                    visualize_states(real_data.x.view(-1, n_box*3*2)[0:test_num], env, writer, 4,  suffix='Training Data')

                else:
                    samples = pc_sampler_img(score, marginal_prob_std_fn, diffusion_coeff_fn, im_size=args.obs_size, batch_size=test_num, device=device)
                    samples = samples.clamp(-1.0, 1.0)
                    grid = make_grid(samples.float(), padding=2, nrow=4, normalize=True)
                    suffix = 'pc'
                    writer.add_image(f'Images/dynamic_{suffix}', grid, epoch)
                    grid_real = make_grid(real_data[0:test_num].float(), padding=2, nrow=4, normalize=True)
                    writer.add_image(f'Images/real_data', grid_real, epoch)
                
                # save model
                with open(ckpt_path + f'score.pt', 'wb') as f:
                    pickle.dump(score, f)
            
            if (epoch + 1) % args.video_freq == 0 and is_state:
                # visualize the process
                in_process_sample = in_process_sample.view(100, -1, in_process_sample.shape[-1])[:, 0, :]
                save_video(env, in_process_sample.cpu().numpy(), save_path=eval_path + f'{epoch+1}', suffix='mp4')

    env.close()

