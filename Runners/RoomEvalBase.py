import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import pickle
from math import *
import cv2

import numpy as np

import torch
from tqdm import tqdm, trange

from Envs.RoomArrangement import RLEnvDynamic, SceneSampler
from room_utils import calc_coverage, exists_or_mkdir, images_to_video

from ipdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_max_vel():
    MAX_VEL = 10
    return MAX_VEL

def load_test_env(horizon, is_single_room):
    MAX_VEL = get_max_vel()
    tar_data = 'UnshuffledRoomsMeta'
    exp_kwargs = {
        'max_vel': MAX_VEL,
        'pos_rate': 1,
        'ori_rate': 1,
        'max_horizon': horizon,
    }
    env = RLEnvDynamic(
        tar_data,
        exp_kwargs,
        meta_name='ShuffledRoomsMeta',
        is_gui=False,
        fix_num=None,
        is_single_room=is_single_room,
        split='test',
    )
    return env


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def collect_trajectories(eval_env, eval_policy, eval_episodes, is_random_init=True):
    vis_states = []
    traj_infos = []
    gt_states = []
    room_names = []
    
    for _ in trange(eval_episodes):
        _, _ = eval_env.reset(is_random_sample=False)
        room_name = eval_env.sim.name
        for flip_num in range(2):
            for rotate_num in range(4):
                flip_rotate_params = {'flip': flip_num, 'rotate': rotate_num}
                gt_state = eval_env.reset(room_name=room_name, flip=True, rotate=True, brownian=False, is_random_sample=False, flip_rotate_params=flip_rotate_params)
                state, done = eval_env.reset(room_name=room_name, flip=True, rotate=True, brownian=is_random_init, is_random_sample=False, flip_rotate_params=flip_rotate_params), False
                cur_vis_states = []
                cur_traj_infos = []
                while not done:
                    action = eval_policy.select_action(state, sample=False)
                    # action = eval_env.sample_action()
                    next_state, _, done, infos = eval_env.step(action)
                    state = next_state
                    cur_vis_states.append(state)
                    cur_traj_infos.append(infos)

                vis_states.append(cur_vis_states)
                traj_infos.append(cur_traj_infos)
                gt_states.append(gt_state)
                room_names.append(room_name)
    return vis_states, traj_infos, gt_states, room_names


def eval_trajs(
        traj_states,
        traj_infos,
        gt_states,
        room_names,
        exp_name_,
        seed, 
        is_average_collision=True,
        is_coverage=True,
    ):
    # metrics: average_collision, diversity_score, coverage, likelihood_curve
    average_collision = None
    coverage = None
    res = f'--- Seed: {seed} ---\n'

    if is_average_collision:
        ''' average_collision '''
        collisions = np.array([sum([np.sum(item['collision_num']) for item in traj])/len(traj) for traj in traj_infos])
        average_collision_mu, average_collision_std = np.mean(collisions), np.std(collisions)
        res += f'-----average_collision: {average_collision_mu:.3f} +- {average_collision_std:.3f}-----\n' 
        average_collision = {'mu': average_collision_mu, 'std': average_collision_std}
    
    if is_coverage:
        # gt_num = len(trajs)//2
        # gt_num = len(traj_states)*1
        # gt_num = len(trajs)//10
        # gt_num = len(trajs)*2
        room_name_to_gt_states = {}
        for room_name, gt_state in zip(room_names, gt_states):
            if room_name not in room_name_to_gt_states:
                    room_name_to_gt_states[room_name] = []
            state = gt_state
            room_name_to_gt_states[room_name].append(gt_state[1])
        
        room_name_to_states = {}
        for room_name, traj in zip(room_names, traj_states):
            if room_name not in room_name_to_states:
                room_name_to_states[room_name] = []
            state = traj[-1]
            room_name_to_states[room_name].append(state[1])
        coverage_mu, coverage_std = calc_coverage(room_name_to_gt_states, room_name_to_states)
        res += f'-----coverage_score: mu: {coverage_mu:.3f} std: {coverage_std:.3f}-----\n'
    
    # print('Start collecting end imgs!')
    # sampler = SceneSampler('bedroom', gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    # init_imgs = []
    # end_imgs = []
    # gt_imgs = []
    # for i, room_name in enumerate(room_names):
    #     print(f'[{i}/{len(room_names)}]')
    #     init_state = traj_states[i][0]
    #     end_state = traj_states[i][-1]
    #     gt_state = gt_states[i]

    #     sim = sampler[room_name]
    #     sim.normalize_room()
    #     sim.set_state(init_state[1], init_state[0])
    #     img = sim.take_snapshot(512, height=10.0)
    #     init_imgs.append(img)
    #     sim.set_state(end_state[1], end_state[0])
    #     img = sim.take_snapshot(512, height=10.0)
    #     end_imgs.append(img)
    #     sim.set_state(gt_state[1], gt_state[0])
    #     img = sim.take_snapshot(512, height=10.0)
    #     gt_imgs.append(img)

    #     # close env after rendering
    #     sim.disconnect()

    # for i, (traj_state, room_name) in enumerate(zip(traj_states, room_names)):
    #     print(f'[{i}/{len(room_names)}]')
    #     state_item = traj_state[-1]
    #     sim = sampler[room_name]
    #     sim.normalize_room()

    #     # vis GT state
    #     sim.set_state(state_item[1], state_item[0])
    #     img = sim.take_snapshot(512, height=10.0)
    #     end_imgs.append(img)

    #     # close env after rendering
    #     sim.disconnect()
    #     # eval_env.reset(room_name=room_name, goal=state_item, flip=False, rotate=False, brownian=False)
    #     # end_imgs.append(np.transpose(cv2.cvtColor(eval_env.sim.sim.take_snapshot(512, height=10.0), cv2.COLOR_BGR2RGB), (2, 0, 1)))
    #     # end_imgs.append(cv2.cvtColor(eval_env.sim.sim.take_snapshot(512, height=10.0), cv2.COLOR_BGR2RGB))

    print(res)
    print(f'EXP_NAME: {exp_name_}')
    metrics = {
        'average_collision': average_collision,
        'coverage_mu': coverage_mu,
        'coverage_std': coverage_std,
        # 'init_imgs': init_imgs,
        # 'end_imgs': end_imgs,
        # 'gt_imgs': gt_imgs,
    }
    return metrics


def merge_metrics_dicts(metrics_dicts, exp_name_):
    merged_dict = {}

    res = f'---- Summary Across 5 Seeds ----\n'
    # merge ACN
    ACNs = [metrics_dict['average_collision']['mu'] for metrics_dict in metrics_dicts]
    merged_dict['average_collision'] = {
        'mu': np.mean(ACNs),
        'std': np.std(ACNs),
    }
    res += f'-----average_collision: {np.mean(ACNs):.3f} +- {np.std(ACNs):.3f}-----\n' 

    # merge coverage score
    CSs = [metrics_dict['coverage_mu'] for metrics_dict in metrics_dicts]
    merged_dict['coverage_mu'] = np.mean(CSs)
    merged_dict['coverage_std'] = np.std(CSs)
    res += f'-----coverage_score: {np.mean(CSs):.3f} +- {np.std(CSs):.3f}-----\n'

    print(res)
    print(f'EXP_NAME: {exp_name_}')
    return merged_dict


def eval_policy(eval_env, policy, eval_num, exp_name, exp_path, save_video, recover=True, is_random_init=True, 
# seeds=[0, 5, 10, 15, 20],
seeds=[0],
):
    exists_or_mkdir(exp_path)
    metrics_path = os.path.join(exp_path, f'{len(seeds)}seeds_metrics_{eval_num}.pkl')
    trajs_path = os.path.join(exp_path, f'{len(seeds)}seeds_trajs_{eval_num}.pkl')
    eval_env.reset()

    if os.path.exists(trajs_path) and recover:
        with open(trajs_path, 'rb') as f:
            trajs_res = pickle.load(f)
        terminal_states, traj_infos, gt_states, room_names = trajs_res['terminal_states'], trajs_res['traj_infos'], trajs_res['gt_states'], trajs_res['room_names']
    else:
        terminal_states, traj_infos, gt_states, room_names = [], [], [], []
        for seed in seeds:
            eval_env.seed(seed)
            terminal_state, traj_info, gt_state, room_name = collect_trajectories(eval_env, policy, eval_num, is_random_init)
            terminal_states.append(terminal_state)
            traj_infos.append(traj_info)
            gt_states.append(gt_state)
            room_names.append(room_name)
        
        trajs_res = {'terminal_states': terminal_states, 'traj_infos': traj_infos, 'gt_states': gt_states, 'room_names': room_names}
        with open(trajs_path, 'wb') as f:
            pickle.dump(trajs_res, f)

    eval_env.close()
    metrics_dicts = []
    for terminal_state, traj_info, gt_state, room_name, seed in zip(terminal_states, traj_infos, gt_states, room_names, seeds):
        metrics_dict = eval_trajs(terminal_state, traj_info, gt_state, room_name, exp_name, seed)
        metrics_dicts.append(metrics_dict)
    
    metrics = merge_metrics_dicts(metrics_dicts, exp_name)

    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    if save_video == 'True':
        take_videos(terminal_states, room_names, exp_name, exp_path)
    print(f'EXP_NAME: {exp_name}')


def take_videos(states, room_names, exp_name, exp_path, render_freq=2, render_size=256):
    ''' render and vis terminal states '''
    sampler = SceneSampler('bedroom', gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    eval_path = f'{exp_path}eval_{exp_name}'
    exists_or_mkdir(eval_path)
    for video_idx, (state, room_name) in tqdm(enumerate(zip(states, room_names))):
        imgs = []
        sim = sampler[room_name]
        sim.normalize_room()

        # # vis GT state
        # img = sim.take_snapshot(render_size, height=10.0)
        # imgs.append(img)

        # vis init/terminal state
        for idx, state_item in enumerate(state):
            sim.set_state(state_item[1], state_item[0])
            img = sim.take_snapshot(render_size, height=10.0)
            if idx % render_freq == 0:
                imgs.append(img[:, :, ::-1])

        # close env after rendering
        sim.disconnect()
        batch_imgs = np.stack(imgs, axis=0)
        # save imgs
        images_to_video(f'{eval_path}/{video_idx}.mp4', batch_imgs, len(imgs)//5, (render_size, render_size))


