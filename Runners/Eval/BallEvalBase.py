import os
import sys

import pickle
from math import *
from ipdb import set_trace
import numpy as np
from tqdm import trange

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ball_utils import save_video, coverage_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_simulation_constants():
    MAX_VEL = 0.3
    dt = 0.02
    PB_FREQ = 4
    RADIUS = 0.025
    WALL_BOUND = 0.3
    return MAX_VEL, dt, PB_FREQ, RADIUS, WALL_BOUND

def eval_trajs(pattern, pdf_func, trajs, num_objs, eval_env, exp_name_, seed=0):

    ''' delta_likelihood '''
    end_states = [traj[-1]['state'] for traj in trajs]
    start_states = [traj[0]['state'] for traj in trajs]
    end_likelihoods = np.stack(pdf_func(end_states))
    start_likelihoods = np.stack(pdf_func(start_states))
    delta_likelihood_mu, delta_likelihood_std = np.mean(end_likelihoods - start_likelihoods), \
                                                np.std(end_likelihoods - start_likelihoods)
    print(f'----- seed {seed}: delta_likelihood: {delta_likelihood_mu:.3f} +- {delta_likelihood_std:.3f} -----')
    delta_likelihood = {'mu': delta_likelihood_mu, 'std': delta_likelihood_std}

    ''' coverage scores (coverage_score-gen is used in paper)'''
    gt_num = len(trajs)//2 if pattern == 'Cluster' else len(trajs)//5
    eval_env.seed(seed+10)
    # get a batch of GF-states
    gt_states = []
    for _ in range(gt_num):
        goal_state = eval_env.reset(is_random=False)
        goal_state = eval_env.flatten_states([goal_state])[0]
        gt_states.append(goal_state.reshape(-1))
    
    end_states = [traj[-1]['state'] for traj in trajs]
    is_category = 'Cluster' in pattern
    coverage_gt_cover_mu, coverage_gt_cover_std = coverage_score(gt_states, end_states, num_objs=num_objs, is_category=is_category, who_cover='gt')
    print(f'----- seed {seed}: coverage_score-gt cover mu: {coverage_gt_cover_mu:.3f} std: {coverage_gt_cover_std:.3f}-----')
    coverage_gen_cover_mu, coverage_gen_cover_std = coverage_score(gt_states, end_states, num_objs=num_objs, is_category=is_category, who_cover='gen')
    print(f'----- seed {seed}: coverage_score-gen cover mu: {coverage_gen_cover_mu:.3f} std: {coverage_gen_cover_std:.3f}-----')


    ''' average_collision '''
    collisions = np.array([sum([item['collision_num'] for item in traj])/len(traj) for traj in trajs])
    average_collision_mu, average_collision_std = np.mean(collisions), np.std(collisions)
    print(f'----- seed {seed}: average_collision: {average_collision_mu:.3f} +- {average_collision_std:.3f}-----')
    average_collision = {'mu': average_collision_mu, 'std': average_collision_std}

    ''' likelihood curve '''
    trajs_likelihoods = []
    for traj in trajs:
        traj_states = [item['state'] for item in traj]
        traj_likelihoods = np.stack(pdf_func(traj_states))
        trajs_likelihoods.append(traj_likelihoods)
    trajs_likelihoods = np.stack(trajs_likelihoods)
    mu = np.mean(trajs_likelihoods, axis=0)
    std = np.std(trajs_likelihoods, axis=0)
    likelihood_curve = {'mu': mu, 'upper': mu+std, 'lower': mu-std}
    

    print(f'EXP_NAME: {exp_name_}')
    metrics = {
        'delta_likelihood': delta_likelihood,
        'average_collision': average_collision,
        'coverage_gen_cover': coverage_gen_cover_mu,
        'coverage_gen_cover_std': coverage_gen_cover_std,
        'coverage_gt_cover': coverage_gt_cover_mu,
        'coverage_gt_cover_std': coverage_gt_cover_std,
        'likelihood_curve': likelihood_curve,
    }
    return metrics

def merge_metrics_dicts(metrics_dicts):
    metrics = {}

    # delta likelihood 
    delta_likelihoods = np.array([metrics_dict['delta_likelihood']['mu'] for metrics_dict in metrics_dicts])
    metrics['delta_likelihood'] = {
        'mu': np.mean(delta_likelihoods),
        'std': np.std(delta_likelihoods),
    }
    print(f'----- Total: delta_likelihood: {np.mean(delta_likelihoods):.2f} +- {np.std(delta_likelihoods):.2f} -----')

    # average collision 
    average_collisions = np.array([metrics_dict['average_collision']['mu'] for metrics_dict in metrics_dicts])
    metrics['average_collision'] = {
        'mu': np.mean(average_collisions),
        'std': np.std(average_collisions),
    }
    print(f'----- Total: average_collision: {np.mean(average_collisions):.2f} +- {np.std(average_collisions):.2f} -----')

    # coverage gen
    coverage_gen_covers = np.array([metrics_dict['coverage_gen_cover'] for metrics_dict in metrics_dicts])
    metrics['coverage_gen_cover'] = np.mean(coverage_gen_covers)
    metrics['coverage_gen_cover_std'] = np.std(coverage_gen_covers)
    print(f'----- Total: coverage_gen_cover: {np.mean(coverage_gen_covers):.2f} +- {np.std(coverage_gen_covers):.2f} -----')

    # coverage gt
    coverage_gt_covers = np.array([metrics_dict['coverage_gt_cover'] for metrics_dict in metrics_dicts])
    metrics['coverage_gt_cover'] = np.mean(coverage_gt_covers)
    metrics['coverage_gt_cover_std'] = np.std(coverage_gt_covers)
    print(f'----- Total: coverage_gt_cover: {np.mean(coverage_gt_covers):.2f} +- {np.std(coverage_gt_covers):.2f} -----')

    # likelihood curve
    likelihood_curves = np.stack([metrics_dict['likelihood_curve']['mu'] for metrics_dict in metrics_dicts])
    mu = np.mean(likelihood_curves, axis=0)
    std = np.std(likelihood_curves, axis=0)
    metrics['likelihood_curve'] = {
        'upper': mu+std, 
        'mu': mu, 
        'lower': mu-std,
    }
    return metrics


def full_metric(pattern, env, pdf_func, exp_path, policy, num_objs, exp_name, eval_num, recover=False, seeds=[0, 5, 10, 15, 20]):
    ''' If there exists trajs, then skip '''
    trajs = None
    trajs_path = exp_path+f'{len(seeds)}seeds_trajs_{num_objs}_{eval_num}_{env.max_episode_len}.pickle'
    metrics_path = exp_path+f'{len(seeds)}seeds_metrics_{num_objs}_{eval_num}_{env.max_episode_len}.pickle'

    if os.path.exists(trajs_path) and recover:
        print('----- Find Existing Trajs!! -----')
        with open(trajs_path, 'rb') as f:
            trajs = pickle.load(f)
    
    if trajs is None:
        print('----- Start Collecting Trajs -----')
        trajs = []
        for seed in seeds:
            env.seed(seed)
            print(f'Seed {seed}: Starting collecting trajs! num: {eval_num}')
            cur_trajs = []
            for _ in trange(eval_num):
                if hasattr(policy, 'reset_policy'):
                    policy.reset_policy()
                state, done = env.reset(is_random=True), False
                state = env.flatten_states([state])[0]
                traj = []
                while not done:
                    action = policy.select_action(np.array(state), sample=False)
                    new_state, _, done, infos = env.step(action)
                    new_state = env.flatten_states([new_state])[0]
                    state = new_state
                    cur_infos = {'state': state, 'collision_num': infos['collision_num'].sum()}
                    traj.append(cur_infos)
                cur_trajs.append(traj)
            trajs.append(cur_trajs)

        # save trajs
        with open(trajs_path, 'wb') as f:
            pickle.dump(trajs, f)

    # align with env's horizon
    horizon = env.max_episode_len
    trajs = [[traj[0:horizon] for traj in cur_trajs] for cur_trajs in trajs]

    print('----- Start Eval Trajs -----')
    metrics_dicts = []
    for seed, cur_trajs in zip(seeds, trajs):
        # By default, seeds = [0, 5, 10, ... 5*k, ... ]
        metrics_dicts.append(eval_trajs(pattern, pdf_func, cur_trajs, num_objs, env, exp_name, seed=5*seed))
    metrics = merge_metrics_dicts(metrics_dicts)

    # update metrics
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)


def analysis(eval_env, pdf_func, policy, save_path=None, eval_episodes=100):
    save_freq = eval_env.max_episode_len // 50
    for idx in trange(eval_episodes):
        if hasattr(policy, 'reset_policy'):
            policy.reset_policy()
        state, done = eval_env.reset(is_random=True), False
        state = eval_env.flatten_states([state])[0]
        curve_gt = [np.log(pdf_func([state])[0])]
        video_states = [state.reshape(-1)]
        time_step = 0
        while not done:
            time_step += 1
            action = policy.select_action(np.array(state), sample=False)
            new_state, _, done, _ = eval_env.step(action)
            new_state = eval_env.flatten_states([new_state])[0]

            curve_gt.append(np.log(pdf_func([state])[0]))

            state = new_state
            if time_step % save_freq == 0:
                video_states.append(state.reshape(-1))

        # save video
        video_duration = 5
        save_video(eval_env, video_states, save_path=f'{save_path}video_{idx}', fps=len(video_states) // video_duration, suffix='mp4') # save as gifs


