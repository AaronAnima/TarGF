import pickle
import os
import numpy as np
import sys
import random
import torch
import gym
import functools
from ipdb import set_trace
from tqdm import trange
from tqdm import tqdm

from envs.envs import get_env
from envs.Room.RoomArrangement import SceneSampler
from planners.planners import get_planners
from utils.misc import exists_or_mkdir
from utils.visualisations import images_to_video, images_to_gif
from utils.evaluations import coverage_score_room, coverage_score_ball

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_policy(configs, log_dir):

    ''' init my env '''
    eval_env, max_action = get_env(configs)

    ''' set seeds '''
    torch.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    ''' init policy '''
    policy = get_planners(configs, eval_env, max_action)

    ''' setup eval functions '''
    rollout_func_dict = {'Ball': collect_trajectories_ball, 'Room': collect_trajectories_room}
    collect_trajectories = rollout_func_dict[configs.env_type]
    metrics_func_dist = {
        'Ball': functools.partial(calc_metrics_ball, configs=configs, eval_env=eval_env),
        'Room': calc_metrics_room
        }
    calc_metrics = metrics_func_dist[configs.env_type]
    visualise_func_dist = {'Ball': save_videos_ball, 'Room': save_videos_room}
    save_videos = visualise_func_dist[configs.env_type]


    ''' Collect trajectories '''
    assert len(configs.test_seeds) > 1 # make sure the results have confidence interval

    trajs_save_path = os.path.join('./logs', log_dir, f'{len(configs.test_seeds)}seeds_trajs_{configs.test_num}.pkl')
    metrics_save_path = os.path.join('./logs', log_dir, f'{len(configs.test_seeds)}seeds_metrics_{configs.test_num}.pkl')
    # activate env
    eval_env.reset()
    if os.path.exists(trajs_save_path) and configs.recover:
        with open(trajs_save_path, 'rb') as f:
            trajs_results_list = pickle.load(f)
    else:
        trajs_results_list = []
        for seed in configs.test_seeds:
            eval_env.seed(seed)
            trajs_result = collect_trajectories(eval_env, policy, configs.test_num)
            trajs_results_list.append(trajs_result)
        # save trajs
        with open(trajs_save_path, 'wb') as f:
            pickle.dump(trajs_results_list, f)


    ''' calc_metric '''
    metrics_dicts = []
    for trajs_result, test_seed in zip(trajs_results_list, configs.test_seeds):
        metrics_dict = calc_metrics(trajs_result, log_dir, test_seed)
        metrics_dicts.append(metrics_dict)
    final_metrics = merge_metrics_dicts(metrics_dicts, log_dir)
    # update metrics
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(final_metrics, f)

    ''' take videos '''
    if configs.save_videos:
        # we only visualse the configs.seeds[0]'s results
        video_save_path = os.path.join('./logs', log_dir, f'seed{configs.test_seeds[0]}_{configs.test_num}videos')
        exists_or_mkdir(video_save_path)
        save_videos(trajs_results_list[0], video_save_path, eval_env) # only save for seed 0

#####################################
############ Collect Trajs ##########
#####################################

def collect_trajectories_room(eval_env, eval_policy, eval_episodes):
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
                state, done = eval_env.reset(room_name=room_name, flip=True, rotate=True, brownian=True, is_random_sample=False, flip_rotate_params=flip_rotate_params), False
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
    return {'vis_states': vis_states, 'traj_infos': traj_infos, 'gt_states': gt_states, 'room_names': room_names}


def collect_trajectories_ball(eval_env, eval_policy, eval_episodes):
    
    cur_trajs = []
    for _ in trange(eval_episodes):
        if hasattr(eval_policy, 'reset_policy'):
            eval_policy.reset_policy()
        state, done = eval_env.reset(is_random=True), False
        state = eval_env.flatten_states([state])[0]
        traj = []
        while not done:
            action = eval_policy.select_action(np.array(state), sample=False)
            new_state, _, done, infos = eval_env.step(action)
            new_state = eval_env.flatten_states([new_state])[0]
            state = new_state
            cur_infos = {'state': state, 'collision_num': infos['collision_num'].sum()}
            traj.append(cur_infos)
        cur_trajs.append(traj)
    return cur_trajs


#####################################
############ Calc Metrics ###########
#####################################

def calc_metrics_room(
        trajs_result,
        exp_name_,
        seed, 
    ):

    vis_states, traj_infos, gt_states, room_names = trajs_result['vis_states'], trajs_result['traj_infos'], trajs_result['gt_states'], trajs_result['room_names']
    info_str = f'--- Seed: {seed} ---\n'

    ''' average_collision_number '''
    collisions = np.array([sum([np.sum(item['collision_num']) for item in traj])/len(traj) for traj in traj_infos])
    average_collision_number = np.mean(collisions)
    info_str += f'-----average_collision_number: {average_collision_number:.3f} -----\n' 
    
    ''' coverage_score '''
    room_name_to_gt_states = {}
    for room_name, gt_state in zip(room_names, gt_states):
        if room_name not in room_name_to_gt_states:
                room_name_to_gt_states[room_name] = []
        state = gt_state
        room_name_to_gt_states[room_name].append(gt_state[1])
    
    room_name_to_states = {}
    for room_name, traj in zip(room_names, vis_states):
        if room_name not in room_name_to_states:
            room_name_to_states[room_name] = []
        state = traj[-1]
        room_name_to_states[room_name].append(state[1])
    coverage_score = coverage_score_room(room_name_to_gt_states, room_name_to_states)
    info_str += f'-----coverage_score: {coverage_score:.3f} -----\n'

    info_str += f'EXP_NAME: {exp_name_}\n'
    print(info_str)

    metrics = {
        'average_collision_number': average_collision_number,
        'coverage_score': coverage_score,
    }
    return metrics


def calc_metrics_ball(
    trajs_result,
    exp_name_,
    seed, 
    configs,
    eval_env,
):

    info_str =  f'--- Seed: {seed} ---\n'
    ''' average_collision_number '''
    collisions = np.array([sum([item['collision_num'] for item in traj])/len(traj) for traj in trajs_result])
    average_collision_number = np.mean(collisions)
    info_str += f'-----average_collision_number: {average_collision_number:.3f} -----\n' 

    ''' coverage_score '''
    gt_num = len(trajs_result)//2 if configs.pattern == 'Cluster' else len(trajs_result)//5
    eval_env.seed(seed+10)
    # get a batch of GF-states
    gt_states = []
    for _ in range(gt_num):
        gt_example_state = eval_env.reset(is_random=False)
        gt_example_state = eval_env.flatten_states([gt_example_state])[0]
        gt_states.append(gt_example_state.reshape(-1))
    
    end_states = [traj[-1]['state'] for traj in trajs_result]
    is_category = 'Cluster' in configs.pattern
    coverage_score = coverage_score_ball(gt_states, end_states, num_objs=configs.num_objs, is_category=is_category, who_cover='gen')
    info_str += f'-----coverage_score: {coverage_score:.3f} -----\n'

    ''' pseudo_likelihood_curve '''
    pdf_func = eval_env.pseudo_likelihoods
    trajs_likelihoods = []
    for traj in trajs_result:
        traj_states = [item['state'] for item in traj]
        traj_likelihoods = np.stack(pdf_func(traj_states))
        trajs_likelihoods.append(traj_likelihoods)
    trajs_likelihoods = np.stack(trajs_likelihoods)
    pseudo_likelihood_curve = np.mean(trajs_likelihoods, axis=0)
    info_str += f'-----pseudo_likelihood_curve finished -----\n'
    

    info_str += f'EXP_NAME: {exp_name_}\n'
    print(info_str)

    metrics = {
        'average_collision_number': average_collision_number,
        'coverage_score': coverage_score,
        'pseudo_likelihood_curve': pseudo_likelihood_curve,
    }
    return metrics


#####################################
############ Merge Dicts  ###########
#####################################


def merge_metrics_dicts(metrics_dicts, exp_name_):
    merged_dict = {}

    res = f'---- Summary Across {len(metrics_dicts)} Seeds ----\n'

    # ACN
    ACNs = [metrics_dict['average_collision_number'] for metrics_dict in metrics_dicts]
    merged_dict['average_collision_number'] = {
        'mu': np.mean(ACNs),
        'std': np.std(ACNs),
    }
    res += f'-----average_collision: {np.mean(ACNs):.3f} +- {np.std(ACNs):.3f}-----\n' 

    # CS
    CSs = [metrics_dict['coverage_score'] for metrics_dict in metrics_dicts]
    merged_dict['coverage_score'] = {
        'mu': np.mean(CSs),
        'std': np.std(CSs),
    }
    res += f'-----coverage_score: {np.mean(CSs):.3f} +- {np.std(CSs):.3f}-----\n'

    # PL
    if 'pseudo_likelihood_curve' in metrics_dicts[0].keys():
        pseudo_likelihood_curves = np.stack([metrics_dict['pseudo_likelihood_curve'] for metrics_dict in metrics_dicts])
        mu = np.mean(pseudo_likelihood_curves, axis=0)
        std = np.std(pseudo_likelihood_curves, axis=0)
        merged_dict['pseudo_likelihood_curve'] = {
            'upper': mu+std, 
            'mu': mu, 
            'lower': mu-std,
        }

    print(res)
    print(f'EXP_NAME: {exp_name_}')

    return merged_dict


#####################################
############ Save Videos  ###########
#####################################

def save_videos_room(trajs_result, save_path, eval_env, render_freq=2, render_size=256, suffix='mp4'):
    eval_env.close()

    traj_states, room_names = trajs_result['vis_states'], trajs_result['room_names']
    ''' render and vis terminal states '''
    sampler = SceneSampler(gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    for video_idx, (state, room_name) in tqdm(enumerate(zip(traj_states, room_names))):
        imgs = []
        sim = sampler[room_name]
        sim.normalize_room()

        # vis init/terminal state
        for idx, state_item in enumerate(state):
            if idx % render_freq == 0:
                sim.set_state(state_item[1], state_item[0])
                img = sim.take_snapshot(render_size, height=10.0)
                imgs.append(img[:, :, ::-1])

        # close env after rendering
        sim.disconnect()
        batch_imgs = np.stack(imgs, axis=0)

        # save imgs
        fps = len(imgs)//5
        if suffix == 'gif':
            from PIL import Image
            images_to_gif(os.path.join(save_path, f'video_{video_idx}.{suffix}'), [Image.fromarray(img[:, :, ::-1], mode='RGB') for img in imgs], fps=fps)
        else:
            batch_imgs = np.stack(imgs, axis=0)
            images_to_video(os.path.join(save_path, f'video_{video_idx}.{suffix}'), batch_imgs, fps, (render_size, render_size))


def save_videos_ball(trajs_result, save_path, eval_env, render_freq=2, render_size=256, suffix='mp4'):
    for video_idx, traj in enumerate(trajs_result):
        imgs = []
        for idx, state_info in tqdm(enumerate(traj), desc='Saving video'):
            if idx % render_freq == 0:
                state = state_info['state']
                state = eval_env.unflatten_states([state])[0]
                eval_env.set_state(state)
                img = eval_env.render(img_size=render_size)
                imgs.append(img[:, :, ::-1])
        
        # save imgs
        fps = len(imgs)//5
        if suffix == 'gif':
            from PIL import Image
            images_to_gif(os.path.join(save_path, f'video_{video_idx}.{suffix}'), [Image.fromarray(img[:, :, ::-1], mode='RGB') for img in imgs], fps=fps)
        else:
            batch_imgs = np.stack(imgs, axis=0)
            images_to_video(os.path.join(save_path, f'video_{video_idx}.{suffix}'), batch_imgs, fps, (render_size, render_size))



