from tkinter import Image
import numpy as np

import torch
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image
import cv2
from scipy import linalg
import os
from ipdb import set_trace
import pickle

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

def my_dist(state1, state2, num_objs, is_category=True):
    # remove labels
    state1 = state1.reshape((num_objs, -1))[:, :2].reshape(-1)
    state2 = state2.reshape((num_objs, -1))[:, :2].reshape(-1)

    # !! overfit to the paper's setting, where there are 3 classes
    num_per_class = num_objs // 3

    dist = 0
    if is_category:
        dist += chamfer_dist(state1[0:num_per_class*2].reshape(-1, 2), state2[0:num_per_class*2].reshape(-1, 2))
        dist += chamfer_dist(state1[num_per_class*2:num_per_class*4].reshape(-1, 2), state2[num_per_class*2:num_per_class*4].reshape(-1, 2))
        dist += chamfer_dist(state1[num_per_class*4:num_per_class*6].reshape(-1, 2), state2[num_per_class*4:num_per_class*6].reshape(-1, 2))
    else:
        dist += chamfer_dist(state1.reshape(-1, 2), state2.reshape(-1, 2))
    return dist

def coverage_score(gt_states, states, num_objs, who_cover='gen', is_category=True):
    min_dists = []
    # who_cover == 'gt': each gen find nearest gt， who_cover == ‘gen’: vice versa
    cover_states = gt_states if who_cover == 'gt' else states
    covered_states = states if who_cover == 'gt' else gt_states
    for covered_state in covered_states:
        min_dist = np.min(np.array([my_dist(covered_state, cover_state, num_objs, is_category=is_category) for cover_state in cover_states]))
        min_dists.append(min_dist)
    return np.mean(np.array(min_dists)), np.std(np.array(min_dists))


def snapshot(env, file_name):
    img = env.render(img_size=256)
    cv2.imwrite(file_name, img)


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def string2bool(str_bool):
    return str_bool == 'True'


def save_video(env, states, save_path, fps = 50, render_size = 256, suffix='avi'):
    # states: [state1, state2, ....]
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        state = env.unflatten_states([state])[0]
        env.set_state(state)
        img = env.render(img_size=render_size)
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