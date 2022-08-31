from tkinter import Image
import numpy as np

import torch
from torchvision.transforms import ToTensor
from torchvision import transforms
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import adaptive_avg_pool2d
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image
import cv2
from scipy import linalg
import sys
import os
from ipdb import set_trace
import pickle
import imageio


# class RewardNormalizer(object):
#     def __init__(self, is_norm, writer, update_freq=10, name='default'):
#         self.reward_mean = 0
#         self.reward_std = 1
#         self.num_steps = 0
#         self.vk = 0
#         self.is_norm = is_norm
#         self.writer = writer
#         self.update_freq = update_freq
#         self.name = name

#     def update(self, reward, is_eval=False):
#         if not is_eval and self.is_norm:
#             self.num_steps += 1
#             if self.num_steps == 1:
#                 # the first step, no need to normalize
#                 self.reward_mean = reward
#                 self.vk = 0
#                 self.reward_std = 1
#             else:
#                 # running mean, running std
#                 delt = reward - self.reward_mean
#                 self.reward_mean = self.reward_mean + delt/self.num_steps
#                 self.vk = self.vk + delt * (reward-self.reward_mean)
#                 self.reward_std = np.sqrt(self.vk/(self.num_steps - 1))
#             reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
#             ''' log the running mean/std '''
#             if self.num_steps % self.update_freq == 0:
#                 self.writer.add_scalar(f'Episode_rewards/RunningMean_{self.name}', np.mean(self.reward_mean), self.num_steps)
#                 self.writer.add_scalar(f'Episode_rewards/RunningStd_{self.name}', np.mean(self.reward_std), self.num_steps)
#         return reward

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
        # if self.name == 'sim':
        #     print(rewards_norm.min())
        #     print(rewards_norm.max())
        return rewards_norm
    
    def update_writer(self):
        self.writer.add_scalar(f'Episode_rewards/RunningMean_{self.name}', np.mean(self.reward_mean), self.num_steps)
        self.writer.add_scalar(f'Episode_rewards/RunningStd_{self.name}', np.mean(self.reward_std), self.num_steps)

    def update(self, reward, is_eval=False):
        # if self.name == 'sim':
        #     set_trace()
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


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms=None):
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img = self.img_list[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class PIDController:
    def __init__(self, Kp=0, Ki=0, Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.err_last = 0
        self.target_val = None
    
    def update(self, target_val):
        self.target_val = target_val
    
    def step(self, actual_val):
        assert self.target_val is not None
        if actual_val is None:
            return self.target_val
        err = self.target_val - actual_val
        self.integral += err
        output_val = self.Kp * err + self.Ki * self.integral + self.Kd * (err - self.err_last)
        self.err_last = err
        return output_val


def pdf_color(center, balls, scale=0.05):
    z = (balls.reshape(-1, 2) - center)/scale
    densities = np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return densities.prod()

def pdf(state, num_boxes, is_normed=True):
    bases = np.array(list(range(3))) * (2 * np.pi) / 3
    coins = [0, 1]
    delta = [-2*np.pi/3, 2*np.pi/3]
    radius = 0.18
    scale = 0.05
    bound = 0.3
    r = 0.025
    if is_normed:
        state_ = state*(bound - r)
    else:
        state_ = state
    res = 0
    red_balls, green_balls, blue_balls = state_[:2*num_boxes], state_[2*num_boxes:4*num_boxes], state_[4*num_boxes:6*num_boxes]
    for base in bases:
        for coin in coins:
            theta_red = base
            theta_green = theta_red + delta[coin]
            theta_blue = theta_red + delta[1-coin]
            red_center = radius*np.array([np.cos(theta_red), np.sin(theta_red)])
            green_center = radius*np.array([np.cos(theta_green), np.sin(theta_green)])
            blue_center = radius*np.array([np.cos(theta_blue), np.sin(theta_blue)])
            res += (pdf_color(red_center, red_balls, scale=scale) *
                    pdf_color(green_center, green_balls, scale=scale) *
                    pdf_color(blue_center, blue_balls, scale=scale)) / (len(bases) * len(coins))
    return res

def pdf_sorting(state, num_boxes, is_normed=True):
    bases = np.array(list(range(1))) * (2 * np.pi) / 3 
    coins = [0, 1]
    delta = [-2*np.pi/3, 2*np.pi/3]
    radius = 0.18
    scale = 0.05
    bound = 0.3
    r = 0.025
    if is_normed:
        state_ = state*(bound - r)
    else:
        state_ = state
    res = 0
    red_balls, green_balls, blue_balls = state_[:2*num_boxes], state_[2*num_boxes:4*num_boxes], state_[4*num_boxes:6*num_boxes]
    for base in bases:
        for coin in coins:
            theta_red = base
            theta_green = theta_red + delta[coin]
            theta_blue = theta_red + delta[1-coin]
            red_center = radius*np.array([np.cos(theta_red), np.sin(theta_red)])
            green_center = radius*np.array([np.cos(theta_green), np.sin(theta_green)])
            blue_center = radius*np.array([np.cos(theta_blue), np.sin(theta_blue)])
            res += (pdf_color(red_center, red_balls, scale=scale) *
                    pdf_color(green_center, green_balls, scale=scale) *
                    pdf_color(blue_center, blue_balls, scale=scale)) / (len(bases) * len(coins))
    return res

def pdf_sorting6(state, num_boxes, is_normed=True):
    bases = np.array(list(range(3))) * (2 * np.pi) / 3 
    coins = [0, 1]
    delta = [-2*np.pi/3, 2*np.pi/3]
    radius = 0.18
    scale = 0.05
    bound = 0.3
    r = 0.025
    if is_normed:
        state_ = state*(bound - r)
    else:
        state_ = state
    res = 0
    red_balls, green_balls, blue_balls = state_[:2*num_boxes], state_[2*num_boxes:4*num_boxes], state_[4*num_boxes:6*num_boxes]
    for base in bases:
        for coin in coins:
            theta_red = base
            theta_green = theta_red + delta[coin]
            theta_blue = theta_red + delta[1-coin]
            red_center = radius*np.array([np.cos(theta_red), np.sin(theta_red)])
            green_center = radius*np.array([np.cos(theta_green), np.sin(theta_green)])
            blue_center = radius*np.array([np.cos(theta_blue), np.sin(theta_blue)])
            res += (pdf_color(red_center, red_balls, scale=scale) *
                    pdf_color(green_center, green_balls, scale=scale) *
                    pdf_color(blue_center, blue_balls, scale=scale)) / (len(bases) * len(coins))
    return res

def get_delta_thetas_std(thetas):
    thetas_sorted = np.sort(thetas)
    delta_thetas = thetas_sorted - np.concatenate(
        [np.array([thetas_sorted[-1] - 2 * np.pi]), thetas_sorted[0:-1]])  # deltas = [a_1+2pi - a_n, a_2 - a_1, ... a_n - a_n-1]
    theta_std = np.std(delta_thetas)
    return theta_std

def get_thetas_std(thetas):
    positions = np.concatenate([np.cos(thetas).reshape(-1, 1), np.sin(thetas).reshape(-1, 1)], axis=-1)
    return get_positions_std(positions)

def get_positions_std(positions):
    mean_pos = np.mean(positions, axis=0)
    dists = np.sqrt(np.sum((positions - mean_pos)**2, axis=-1))
    return np.std(dists)


def pdf_circlerect(state, num_boxes=5, is_normed=True):
    # comp circle likelihood
    pdf_circle = pdf_placing(state, num_boxes=5, is_normed=True)
    pdf_rect = 0
    return pdf_circle + pdf_rect

def pdf_placing(state, num_boxes=5, is_normed=True):
    positions = state.reshape(-1, 2)
    positions_centered = positions - np.mean(positions, axis=0)
    positions_centered /= np.max(np.abs(positions_centered))
    radiuses = np.sqrt(np.sum(positions_centered**2, axis=-1))
    # radiuses_normed = radiuses/np.mean(radiuses)
    radius_std = np.std(radiuses)
    thetas = np.arctan2(positions_centered[:, 1], positions_centered[:, 0]) # theta = atan2(y, x)
    theta_std = get_delta_thetas_std(thetas)
    return np.exp(-(theta_std+radius_std))

def pdf_hybrid(state, num_boxes=5, is_normed=True):
    positions = state.reshape(-1, 2)
    positions_centered = positions - np.mean(positions, axis=0) # [num_balls, 2]
    positions_centered /= np.max(np.abs(positions_centered))
    n_per_class = positions.shape[0]//3
    radius_std = np.std(np.sqrt(np.sum(positions_centered**2, axis=-1)))

    thetas = np.arctan2(positions_centered[:, 1], positions_centered[:, 0]) # theta = atan2(y, x)
    theta_std = get_delta_thetas_std(thetas)

    theta_std_r = get_thetas_std(thetas[0:n_per_class])
    theta_std_g = get_thetas_std(thetas[n_per_class:2*n_per_class])
    theta_std_b = get_thetas_std(thetas[2*n_per_class:3*n_per_class])

    center_r = np.mean(positions_centered[0:n_per_class], axis=0)
    center_g = np.mean(positions_centered[n_per_class:2*n_per_class], axis=0)
    center_b = np.mean(positions_centered[2*n_per_class:3*n_per_class], axis=0)
    center_std = np.std(np.array([center_r, center_g, center_b]))

    return np.exp(-((theta_std+radius_std) + (theta_std_r+theta_std_g+theta_std_b) - center_std)) # !!! 血坑！最后center std应该越大越好


def chamfer_dist(x, y, metric='l1'):
    x = x.reshape(-1, 2)
    y = y.reshape(-1, 2)
    # x: [nx, dim], y: [ny, dim]
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    return dist


def coverage_score(gt_states, states, who_cover='gen', is_category=True):
    min_dists = []
    # GT_num == Gen_num//2 
    # 200: 1.688
    # 20: 1.868
    # 50: 1.685
    # 100: 1.645
    # orca_knn1_score_network_001_eval_100_gtnum10_placing
    # who_cover == 'gt': each gen find nearest gt， who_cover == ‘gen’: vice versa
    cover_states = gt_states if who_cover == 'gt' else states
    covered_states = states if who_cover == 'gt' else gt_states
    for covered_state in covered_states:
        min_dist = np.min(np.array([my_dist(covered_state, cover_state, is_category=is_category) for cover_state in cover_states]))
        # print(min_dist)
        min_dists.append(min_dist)
    return np.mean(np.array(min_dists)), np.std(np.array(min_dists))


def my_dist(state1, state2, is_category=True):
    n_balls = np.prod(state1.shape)//6
    assert len(state1.shape) == 1
    assert len(state2.shape) == 1
    dist = 0
    if is_category:
        # 分part
        dist += chamfer_dist(state1[0:n_balls*2].reshape(-1, 2), state2[0:n_balls*2].reshape(-1, 2))
        dist += chamfer_dist(state1[n_balls*2:n_balls*4].reshape(-1, 2), state2[n_balls*2:n_balls*4].reshape(-1, 2))
        dist += chamfer_dist(state1[n_balls*4:n_balls*6].reshape(-1, 2), state2[n_balls*4:n_balls*6].reshape(-1, 2))
        # # 整体
        # dist += chamfer_dist(state1.reshape(-1, 2), state2.reshape(-1, 2))
    else:
        dist += chamfer_dist(state1.reshape(-1, 2), state2.reshape(-1, 2))
    return dist


def diversity_score(last_states, is_category=True):
    dists = []
    ns = len(last_states)
    for idx1 in range(ns-1):
        for idx2 in range(idx1+1, ns):
            dists.append(my_dist(last_states[idx1], last_states[idx2], is_category=is_category))
    score = np.mean(np.array(dists))
    return score


# def get_act(imgs, batch_size=64, num_workers=1, dims=2048, device='cuda'):
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#     model = InceptionV3([block_idx]).to(device)
#     model.eval()

#     dataset = ImageDataset(imgs, transforms=ToTensor())
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              drop_last=False,
#                                              num_workers=1)
    
#     pred_arr = np.empty((len(imgs), dims))
#     start_idx = 0
    
#     for batch in tqdm(dataloader):
#         batch = batch.to(device)
#         with torch.no_grad():
#             pred = model(batch)[0]
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()
#         pred_arr[start_idx:start_idx + pred.shape[0]] = pred
#         start_idx = start_idx + pred.shape[0]
#     return pred_arr

def get_act(imgs, batch_size=64, img_size=64, num_workers=1, dims=2048, device='cuda'):
    # model_path = './Models/vae.pt'
    model_path = './Models/vae_random_sorting.pt'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    imgs = [Image.fromarray(img) for img in imgs]
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(), # load to [0, 1]
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # norm to [-1, 1]
    ])
    dataset = ImageDataset(imgs, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=1)
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            features.append(mu)
        features = torch.cat(features, dim=0).detach().cpu().numpy()
    return features


def fid_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, return_full=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    mean_dist = diff.dot(diff)
    cov_dist = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    if return_full:
        return mean_dist + cov_dist, mean_dist, cov_dist
    else:
        return mean_dist + cov_dist 


def calc_fid_from_act(gt_act, gen_act, return_full=False):
    m1, s1 = fid_statistics(gt_act)
    m2, s2 = fid_statistics(gen_act)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2, return_full=return_full)
    return fid_value


def calc_fid_from_imgs(gt_imgs, gen_imgs, dims=2048, device='cuda', return_full=False):
    gt_act = get_act(gt_imgs, dims, device)
    gen_act = get_act(gen_imgs, dims, device)
    fid_value = calc_fid_from_act(gt_act, gen_act, return_full=return_full)
    return fid_value


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
        # set_trace()
        env.set_state(state)
        img = env.render(render_size)
        imgs.append(img[:, :, ::-1])
    if suffix == 'gif':
        from PIL import Image
        images_to_gif(save_path+f'.{suffix}', [Image.fromarray(img[:, :, ::-1], mode='RGB') for img in imgs], fps=len(imgs)//5)
    else:
        batch_imgs = np.stack(imgs, axis=0)
        images_to_video(save_path+f'.{suffix}', batch_imgs, fps, (render_size, render_size))

def images_to_gif(path, images, fps):
    images[0].save(path, save_all=True, append_images=images[1:], fps=fps, loop=0)

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item)
    out.release()