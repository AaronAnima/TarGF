import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pickle
import argparse
from math import *

import cv2
import numpy as np
from tqdm import trange

import torch

from Envs.SortingBall import RLSorting
from Envs.PlacingBall import RLPlacing
from Envs.HybridBall import RLHybrid
from Runners.BallSAC import GTTargetScore
from utils import save_video, diversity_score, coverage_score, get_act, calc_fid_from_act

from ipdb import set_trace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_max_vel():
    MAX_VEL = 0.3
    return MAX_VEL

def load_env(env, max_vel, n_boxes, horizon, seed, is_onebyone=False, action_type='vel'):

    ENV_DICT = {'sorting': RLSorting, 'placing': RLPlacing, 'hybrid': RLHybrid}
    exp_data = None  # load expert examples for LfD algorithms
    PB_FREQ = 8
    dt_dict = {
        'vel': 0.1 if is_onebyone else 0.02, 
        'force': 0.05,
    }
    max_vel_dict = {
        'vel': 0.3,
        'force': 0.3,
    }
    MAX_VEL = max_vel_dict[action_type]
    dt = dt_dict[action_type]
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes': n_boxes,
        'exp_data': exp_data,
        'time_freq': time_freq * PB_FREQ,
        'is_gui': False,
        'max_action': MAX_VEL,
        'max_episode_len': horizon,
        'action_type': action_type,
    }
    env_class = ENV_DICT[env]
    env = env_class(**env_kwargs)
    env.seed(seed)
    env.reset(is_random=True)

    return env


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def eval_trajs(env_type, trajs, num_objs, eval_env, exp_name_, render_size=256,
               is_delta_likelihood=True,
               is_average_collision=True,
               is_diversity_score=True,
               is_coverage=True,
               is_likelihood_curve=True,
               seed=0,
               is_fid=True,
               is_act=True,
               is_img=True,
               old_metrics=None,
    ):
    # metrics: delta_likelihood, average_collision, diversity_score, coverage, likelihood_curve
    likelihood_estimator = GTTargetScore(num_objs, env_type=env_type)
    delta_likelihood = None
    average_collision = None
    diversity_score_ = None
    coverage = None
    # fid = None
    # act = None
    # imgs = None
    likelihood_curve = None

    if is_delta_likelihood:
        ''' delta_likelihood '''
        end_states = np.stack([traj[-1]['state'] for traj in trajs])
        start_states = np.stack([traj[0]['state'] for traj in trajs])
        end_likelihoods = likelihood_estimator.total_likelihood(end_states)
        start_likelihoods = likelihood_estimator.total_likelihood(start_states)
        delta_likelihood_mu, delta_likelihood_std = np.mean(end_likelihoods - start_likelihoods), \
                                                    np.std(end_likelihoods - start_likelihoods)
        print(f'----- seed {seed}: delta_likelihood: {delta_likelihood_mu:.3f} +- {delta_likelihood_std:.3f} -----')
        delta_likelihood = {'mu': delta_likelihood_mu, 'std': delta_likelihood_std}

    if is_diversity_score:
        ''' diversity score '''
        is_category = env_type in ['sorting', 'hybrid']
        diversity_score_ = diversity_score([traj[-1]['state'] for traj in trajs], is_category=is_category)
        print(f'----- seed {seed}: diversity_score: {diversity_score_:.3f}-----')

    if is_coverage:
        # gt_num = len(trajs)//2
        # gt_num = len(trajs)*1
        # gt_num = len(trajs)//10
        # gt_num = len(trajs)//5 # for placing and hybrid? or for all? 
        gt_num = len(trajs)//2 if env_type == 'sorting' else len(trajs)//5
        # gt_num = 20
        eval_env.seed(seed+10)
        # get a batch of GF-states
        gt_states = []
        for _ in range(gt_num):
            goal_state = eval_env.reset(is_random=False)
            goal_state = goal_state.reshape(num_objs*3, -1)
            if goal_state.shape[1] == 4:
                goal_state = goal_state[:, 2:4]
            gt_states.append(goal_state.reshape(-1))
        end_states = [traj[-1]['state'] for traj in trajs]
        is_category = env_type in ['sorting', 'hybrid']
        coverage_gt_cover_mu, coverage_gt_cover_std = coverage_score(gt_states, end_states, is_category=is_category, who_cover='gt')
        print(f'----- seed {seed}: coverage_score-gt cover mu: {coverage_gt_cover_mu:.3f} std: {coverage_gt_cover_std:.3f}-----')
        coverage_gen_cover_mu, coverage_gen_cover_std = coverage_score(gt_states, end_states, is_category=is_category, who_cover='gen')
        print(f'----- seed {seed}: coverage_score-gen cover mu: {coverage_gen_cover_mu:.3f} std: {coverage_gen_cover_std:.3f}-----')

    if is_average_collision:
        ''' average_collision '''
        collisions = np.array([sum([item['collision_num'] for item in traj])/len(traj) for traj in trajs])
        average_collision_mu, average_collision_std = np.mean(collisions), np.std(collisions)
        print(f'----- seed {seed}: average_collision: {average_collision_mu:.3f} +- {average_collision_std:.3f}-----')
        average_collision = {'mu': average_collision_mu, 'std': average_collision_std}

    
    
    # # Inception dims: 64, 192, 768, 2048
    # # set_dims = 192
    # set_dims = 2048
    # render_size = 256
    # if is_img or is_fid or is_act:
    #     end_imgs = []
    #     for traj in trajs:
    #         eval_env.set_state(traj[-1]['state'])
    #         end_imgs.append(cv2.cvtColor(eval_env.render(render_size), cv2.COLOR_BGR2RGB))
    #     imgs = end_imgs

    # if is_fid or is_act:
    #     # gt_num = len(trajs)//2
    #     gt_num = len(trajs)*1
    #     # gt_num = len(trajs)//10
    #     # gt_num = len(trajs)*2
    #     eval_env.seed(0)
    #     gt_imgs = []
    #     for _ in range(gt_num):
    #         eval_env.reset(is_random=False)
    #         gt_imgs.append(cv2.cvtColor(eval_env.render(render_size), cv2.COLOR_BGR2RGB))
    #     gt_act = get_act(gt_imgs, dims=set_dims)

    #     end_act = get_act(end_imgs, dims=set_dims)
    #     act = end_act

    #     if is_fid:
    #         fid = calc_fid_from_act(gt_act, end_act)
    #         print(f'-----fid_score: {fid:.3f}-----')

    if is_likelihood_curve:
        trajs_likelihoods = []
        for traj in trajs:
            traj_states = np.stack([item['state'] for item in traj])
            traj_likelihoods = likelihood_estimator.total_likelihood(traj_states)
            trajs_likelihoods.append(traj_likelihoods)
        trajs_likelihoods = np.stack(trajs_likelihoods)
        mu = np.mean(trajs_likelihoods, axis=0)
        std = np.std(trajs_likelihoods, axis=0)
        likelihood_curve = {'mu': mu, 'upper': mu+std, 'lower': mu-std}
    
    ''' Absolute State Change '''
    ASC = []
    horizon = len(trajs[0])
    states_ = []
    for idx in range(horizon):
        time_aligned_states = np.stack([traj[idx]['state'] for traj in trajs])
        states_.append(time_aligned_states)
        if idx == 0:
            continue
        ASC.append(np.sum(np.abs(states_[-1] - states_[-2]), axis=-1))
    ASC = np.stack(ASC, axis=-1)
    ASC = np.sum(ASC, axis=-1)
    ASC_mu = np.mean(ASC)
    ASC_std = np.std(ASC)
    print(f'----- seed {seed}: abosolute_state_change: {ASC_mu:.3f} +- {ASC_std:.3f}-----')
    

    print(f'EXP_NAME: {exp_name_}')
    metrics = {
        'delta_likelihood': delta_likelihood,
        'average_collision': average_collision,
        'diversity_score': diversity_score_,
        'coverage_gen_cover': coverage_gen_cover_mu,
        'coverage_gen_cover_std': coverage_gen_cover_std,
        'coverage_gt_cover': coverage_gt_cover_mu,
        'coverage_gt_cover_std': coverage_gt_cover_std,
        # 'fid': fid,
        # 'act': act,
        # 'imgs': imgs,
        'likelihood_curve': likelihood_curve,
        'ASC_mu': ASC_mu,
        'ASC_std': ASC_std,
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

    # diversity score 
    diversity_scores = np.array([metrics_dict['diversity_score'] for metrics_dict in metrics_dicts])
    metrics['diversity_score'] = {
        'mu': np.mean(diversity_scores),
        'std': np.std(diversity_scores),
    }

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

    # ASC
    ASCs = np.array([metrics_dict['ASC_mu'] for metrics_dict in metrics_dicts])
    metrics['ASC_mu'] = np.mean(ASCs)
    metrics['ASC_std'] = np.std(ASCs)
    print(f'----- Total: ASC: {np.mean(ASCs):.2f} +- {np.std(ASCs):.2f} -----')

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



# def full_metric(env, env_type, exp_path, policy, n_balls, exp_name, eval_num, recover=False, seeds=[0, 5, 10, 15, 20]):
def full_metric(env, env_type, exp_path, policy, n_balls, exp_name, eval_num, recover=False, seeds=[0, 5]):
    ''' If there exists trajs, then skip '''
    trajs = None
    # trajs_path = exp_path+f'{len(seeds)}seeds_trajs_{n_balls}_{eval_num}.pickle'
    # metrics_path = exp_path+f'{len(seeds)}seeds_metrics_{n_balls}_{eval_num}.pickle'
    trajs_path = exp_path+f'{len(seeds)}seeds_trajs_{n_balls}_{eval_num}_{env.max_episode_len}.pickle'
    metrics_path = exp_path+f'{len(seeds)}seeds_metrics_{n_balls}_{eval_num}_{env.max_episode_len}.pickle'
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
                state, done = env.reset(is_random=True), False
                traj = []
                while not done:
                    action = policy.select_action(np.array(state), sample=False)
                    new_state, _, done, infos = env.step(action, centralized=False)
                    state = new_state
                    cur_infos = {'state': state.reshape(n_balls*3, -1)[:, 0:2].reshape(-1), 'collision_num': infos['collision_num'].sum()}
                    traj.append(cur_infos)
                cur_trajs.append(traj)
            trajs.append(cur_trajs)

        # save trajs
        with open(trajs_path, 'wb') as f:
            pickle.dump(trajs, f)


    # align with env's horizon
    horizon = env.max_episode_len
    trajs = [[traj[0:horizon] for traj in cur_trajs] for cur_trajs in trajs]
    # set_trace()

    print('----- Start Eval Trajs -----')
    metrics_dicts = []
    for seed, cur_trajs in zip(seeds, trajs):
        # By default, seeds = [0, 5, 10, ... 5*k, ... ]
        metrics_dicts.append(eval_trajs(env_type, cur_trajs, n_balls, env, exp_name, seed=5*seed))
    metrics = merge_metrics_dicts(metrics_dicts)

    # update metrics
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)


def analysis(eval_env, pdf, policy, n_box, score, t0, save_path=None, eval_episodes=100, is_state=True, obs_size=64):
    import matplotlib.pyplot as plt
    # is_score_curve = (score is not None)
    # is_score_curve = False
    # curves_gt = []
    # curves_ours = []
    save_freq = eval_env.max_episode_len // 50
    for idx in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        cur_obs = eval_env.render(obs_size)
        state_ = state.reshape(3*n_box, -1)[:, 0:2]
        curve_gt = [np.log(pdf(state_, n_box))]
        # if is_score_curve:
        #     curve_estimated = [np.log(pdf(state_, n_box))]
        video_states = [state_.reshape(-1)]
        time_step = 0
        while not done:
            time_step += 1
            action = policy.select_action(np.array(state), sample=False)
            # action = eval_env.sample_action()
            new_state, _, done, infos = eval_env.step(action, centralized=False)
            new_obs = eval_env.render(obs_size)

            state_ = state.reshape(3 * n_box, -1)[:, 0:2]
            new_state_ = new_state.reshape(3 * n_box, -1)[:, 0:2]
            curve_gt.append(np.log(pdf(state_, n_box)))
            # if is_score_curve:
            #     if is_state:
            #         cur_tar_vels = score.get_score(state_, t0=t0, is_norm=False)
            #         curve_estimated.append(curve_estimated[-1]+np.sum(cur_tar_vels*(new_state_-state_)))
            #     else:
            #         cur_obs = 2*cur_obs/255. - 1
            #         new_obs = 2*new_obs/255. - 1
            #         cur_obs = np.transpose(cur_obs, (2, 0, 1)) # [c, h, w]
            #         new_obs = np.transpose(new_obs, (2, 0, 1))
            #         cur_score = score.get_score(cur_obs, t0=t0, is_norm=False).reshape(-1)
            #         delta_obs = (new_obs - cur_obs).reshape(-1)
            #         delta_likelihood = np.sum(delta_obs * cur_score)
            #         curve_estimated.append(curve_estimated[-1]+delta_likelihood)

            state = new_state
            cur_obs = new_obs
            if time_step % save_freq == 0:
                video_states.append(state_.reshape(-1))
        # curves_gt.append(curve_gt)
        # curves_ours.append(curve_estimated)
        # plot GT curve
        # plt.clf()
        # plt.plot(np.array(curve_gt), label='GT')
        # if is_score_curve:
        #     plt.plot(np.array(curve_estimated), label='Estimated')
        # plt.legend(loc=4)
        # plt.show()
        # plt.savefig(f'{save_path}curves_{idx}.png')

        # save video
        video_duration = 5
        video_states_np = np.stack(video_states)
        # save_video(eval_env, video_states_np, save_path=f'{save_path}video_{idx}', fps=len(video_states_np) // video_duration, suffix='mp4')
        save_video(eval_env, video_states_np, save_path=f'{save_path}video_{idx}', fps=len(video_states_np) // video_duration, suffix='gif') # save as gifs
    
    # # trajs_path = exp_path+f'trajs_{n_balls}_{eval_num}.pickle'
    # # if os.path.exists(trajs_path):
    # #     print('----- Find Existing Trajs!! -----')
    # #     with open(trajs_path, 'rb') as f:
    # #         trajs = pickle.load(f)
    
    # # for traj in trajs:
    # #     curve_estimated = []
    # #     curve_gt = []
    # #     for idx, info in enumerate(traj):
    # #         state_ = info['state'] 
    # #         curve_gt.append(np.log(pdf(state_, n_box)))
    # #         new_state_ = new_state.reshape(3 * n_box, -1)[:, 0:2]
    # #         cur_tar_vels = score.get_score(state_, t0=t0, is_norm=False)
    # #         curve_estimated.append(curve_estimated[-1]+np.sum(cur_tar_vels*(new_state_-state_)))

    
    # curves_gt = np.stack(curves_gt)

    # # base = np.min(curves_gt)
    # # scale = np.max(curves_gt) - np.min(curves_gt)
    # # curves_gt -= base
    # # curves_gt /= scale

    # curves_ours = np.stack(curves_ours)
    
    # # base = np.min(curves_estimated)
    # # scale = np.max(curves_estimated) - np.min(curves_estimated)
    # # curves_estimated -= base
    # # curves_estimated /= scale

    # curves_gt_mu = np.mean(curves_gt, axis=0)
    # curves_gt_std = np.std(curves_gt, axis=0)
    # curves_estimated_mu = np.mean(curves_ours, axis=0)
    # curves_estimated_std = np.std(curves_ours, axis=0)
    # comparative_curve = {
    #     'gt_mu': curves_gt_mu,
    #     'gt_std': curves_gt_std,
    #     'estimated_mu': curves_estimated_mu,
    #     'estimated_std': curves_estimated_std,
    # }
    # suffix = 'bad' if test_bad_policy else ''
    # with open(f'{save_path}comparative_likelihood_curve_{suffix}.pickle', 'wb') as f:
    #     pickle.dump(comparative_curve, f)

    # plt.plot(curves_gt_mu, label='gt_likelihood')
    # plt.fill_between(np.array(range(curves_gt_mu.shape[0])), curves_gt_mu+curves_gt_std, curves_gt_mu-curves_gt_std, alpha=0.1)
    # plt.plot(curves_estimated_mu, label='estimated_likelihood')
    # plt.fill_between(np.array(range(curves_estimated_mu.shape[0])), curves_estimated_mu+curves_estimated_std, curves_estimated_mu-curves_estimated_std, alpha=0.1)
    # plt.legend(loc=4)
    # plt.show()
    # plt.savefig(f'{save_path}comparative_likelihood_curve_{suffix}.png')



def take_video(eval_env, policy, eval_episodes=4):
    videos = []
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        video_states = []
        while not done:
            # 就算是goal_env，cur_state 也在前面半截
            video_states.append(state[:, 0:2].reshape(-1))
            action = policy.select_action(np.array(state), sample=False)
            next_state, _, done, infos = eval_env.step(action, centralized=False)
            state = next_state
        videos.append(video_states)
    return videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", nargs='+')  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--labels", nargs='+')  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--curve_name", type=str, default="")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--eval_num", type=int, default=100)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--alpha", type=float, default=0.3)  # Policy name (MATD3, DDPG or OurDDPG)
    args = parser.parse_args()

    exp_names = args.exp_names
    labels = args.labels
    curves = []
    for exp_name in exp_names:
        metric_path = f'../logs/{exp_name}/metrics_{args.eval_num}.pickle'
        with open(metric_path, 'rb') as f:
            metrics = pickle.load(f)
        curve = metrics['likelihood_curve']
        curves.append(curve)
    import matplotlib.pyplot as plt
    plt.clf()
    for curve, label in zip(curves, labels):
        # set_trace()
        plt.plot(curve['mu'], label=label)
        plt.fill_between(np.array(range(curve['mu'].shape[0])), curve['upper'], curve['lower'], alpha=args.alpha)
    plt.legend(loc=4)
    plt.show()
    plt.savefig(f'../logs/curve_{args.curve_name}_{args.eval_num}.png')

