import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
from scipy.spatial import distance_matrix
from math import *
from tqdm import tqdm, trange
import functools
import torch

import argparse
import pickle
from collections import deque

from Algorithms.ORCA.pyorca import Agent, orca
from Algorithms.ORCA.RVO import compute_V_des
from Algorithms.ORCA.RVO import compute_V_des
from utils import save_video
from Runners.BallSAC import load_target_score
from Runners.BallEvalBase import eval_trajs, merge_metrics_dicts


from utils import pdf_sorting, pdf_placing, pdf_hybrid, pdf_sorting6, pdf_circlerect


from ipdb import set_trace

from Envs.SortingBall import RLSorting
from Envs.PlacingBall import RLPlacing
from Envs.HybridBall import RLHybrid
ENV_DICT = {'sorting': RLSorting, 'placing': RLPlacing, 'hybrid': RLHybrid}
PDF_DICT = {'sorting': pdf_sorting, 'placing': pdf_placing, 'hybrid': pdf_hybrid}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalise_vels(vels, max_vel):
    # [-inf, +inf] -> [-max_vel, max_vel]**n
    vels = np.array(vels)
    max_vel_norm = np.max(np.abs(vels))
    scale_factor = max_vel / (max_vel_norm+1e-7)
    scale_factor = np.min([scale_factor, 1])
    vels_updated = scale_factor * vels
    return vels_updated

class PIDController:
    def __init__(self):
        self.Kp = 0.5
        self.Ki = 0.0
        self.Kd = 0.0
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



class PlanningAgent:
    def __init__(self, 
        target_score, 
        num_objs, 
        orca_mode, 
        env_, 
        is_orca=True,
        is_pid=False,
        goal_mode='Score', 
        radius=0.025,
        max_vel=0.3, 
        dt=0.02, 
        tau=0.1, 
        t0=0.01, 
        knn=1, 
        horizon=250, 
        is_decay=False, 
        one_by_one=False 
    ):
        self.target_score = target_score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.agents = []
        self.t0 = t0
        self.knn = knn
        self.dt = dt
        self.tau = tau
        self.mode = orca_mode
        self.goal_mode = goal_mode
        self.goal_buffer = deque(maxlen=128)
        self.env = env_
        self.is_orca = is_orca
        self.one_by_one = one_by_one
        self.last_idx = None
        self.last_duration = 0
        self.one_by_one_duration = 20
        self.pid = PIDController()

        for _ in range(num_objs*3):
            self.agents.append(Agent(np.zeros(2), (0., 0.), radius, max_vel, np.zeros(2)))
        self.goal_state = None

        self.horizon = horizon
        self.cur_time_step = 0
        self.is_decay = is_decay
    
    def fill_buffer(self):
        if self.goal_mode == 'GT':
            goal = self.env.reset(is_random=False)
            self.goal_buffer.append(goal)
        elif self.goal_mode == 'Score':
            assert self.target_score is not None
            goals = self.target_score.sample_goals(128, is_batch=True)
            for goal in goals:
                self.goal_buffer.append(goal)
        else:
            raise NotImplementedError

    def reset_policy(self):
        if self.mode == 'goal':
            if len(self.goal_buffer) <= 0:
                self.fill_buffer()
            self.goal_state = self.goal_buffer.pop() # [30, ]
            self.goal_state = self.goal_state.reshape((-1, 2)) # [15, 2]
        self.cur_time_step = 0

    def assign_tar_vels(self, vels):
        for agent, vel in zip(self.agents, vels):
            agent.pref_velocity = np.array(vel)

    def assign_pos(self, positions):
        for agent, position in zip(self.agents, positions):
            agent.position = np.array(position)

    def get_tar_vels(self, inp_state):
        if self.mode == 'score':
            if self.is_decay:
                t0 = self.t0*(self.horizon - self.cur_time_step + 1e-3) / self.horizon
            else:
                t0 = self.t0
            tar_vels = self.target_score.get_score(inp_state, t0, is_numpy=True, is_norm=True, empty=False)
            tar_vels = tar_vels.reshape((-1, 2))
        elif self.mode == 'goal':
            tar_vels = compute_V_des(inp_state.reshape((-1, 2)).tolist(),
                                     goal=self.goal_state.tolist(), V_max=[self.max_vel] * (3 * self.num_objs))
        else:
            print('orca mode error! should be score/goal !!')
            raise NotImplementedError
        
        if self.one_by_one:
            # only take the largest grad
            if self.mode == 'score':
                norms = np.sqrt(np.sum(tar_vels**2, axis=-1))
            else:
                assert self.mode == 'goal'
                # largest dist to goal is better
                norms = np.sqrt(np.sum((inp_state.reshape(-1, 2) - self.goal_state.reshape(-1, 2))**2, axis=-1))
            max_norm = np.max(norms)
            max_idx = norms >= max_norm
            cur_max_idx = max_idx.reshape(-1, 1)
            if self.last_idx is None:
                self.last_idx = cur_max_idx
                self.last_duration = 0
            else:
                if self.last_duration >= self.one_by_one_duration:
                    self.last_idx = cur_max_idx
                    self.last_duration = 0
                else:
                    self.last_duration += 1
            tar_vels *= self.last_idx
        return tar_vels

    def select_action(self, inp_state, infos=None, sample=True):
        self.cur_time_step += 1
        # get and assign tar vels
        tar_vels = self.get_tar_vels(inp_state)
        self.assign_tar_vels(tar_vels)

        # assign positions
        positions = inp_state.reshape((-1, 2))
        self.assign_pos(positions)
        new_vels = [None] * len(self.agents)
        if self.is_orca:
            
            # ORCA: compute the optimal vel to avoid collision
            for i, agent in enumerate(self.agents):
                cur_vel = tar_vels[i]
                if np.sum(np.array(cur_vel)**2) < 1e-10 and self.one_by_one:
                    new_vels[i] = np.array([0, 0])
                    continue
                # 意思就是刚好抹掉 i-th agent，考虑其他所有agent的碰撞
                candidates = self.agents[:i] + self.agents[i + 1:]

                # 再筛选一遍candidates，candidates就是k nearest
                def my_comp(x, y):
                    x_dist = np.sum((x.position - agent.position) ** 2)
                    y_dist = np.sum((y.position - agent.position) ** 2)
                    return x_dist - y_dist

                k_nearest_candidates = sorted(candidates, key=functools.cmp_to_key(my_comp))[0:self.knn]
                new_vels[i], _ = orca(agent, k_nearest_candidates, self.tau, self.dt)
        else:
            new_vels = tar_vels
        # print(np.sum((np.array(new_vels) - np.array(tar_vels))**2))
        new_vels = normalise_vels(new_vels, self.max_vel)

        for i, agent in enumerate(self.agents):
            agent.velocity = new_vels[i]
        return np.array(new_vels).reshape(-1)

def get_delta_thetas(cur_state):
    positions = cur_state.reshape(-1, 2)
    positions_centered = positions - np.mean(positions, axis=0)
    thetas = np.arctan2(positions_centered[:, 1], positions_centered[:, 0]) # theta = atan2(y, x)
    thetas_sorted = np.sort(thetas)
    thetas_1 = thetas_sorted
    thetas_2 = np.concatenate([np.array([thetas_sorted[-1] - 2 * np.pi]), thetas_sorted[0:-1]])  # deltas = [a_1+2pi - a_n, a_2 - a_1, ... a_n - a_n-1]
    delta_thetas = thetas_1 - thetas_2
    print(thetas_1)
    print(thetas_2)
    print(delta_thetas)
    print(np.std(delta_thetas))
    return delta_thetas


def debug_orca(eval_num):
    # save 50 images at most
    save_freq = args.horizon // 50
    for episode_idx in tqdm(range(eval_num)):
        collision_num = 0
        vel_errs = []
        vel_errs_mean = []
        policy.reset_policy()  # if goal, then this func will init the goal state
        state = env.reset(is_random=True)
        states_np = [state]
        HORIZON = args.horizon
        pdf_func = PDF_DICT[args.env]
        # print(get_delta_thetas(state))
        for idx in range(HORIZON):
            action = policy.select_action(state)
            state, _, _, infos = env.step(action, step_size=PB_FREQ)
            vel_errs.append(infos['vel_err'])
            vel_errs_mean.append(infos['vel_err_mean'])
            if idx % save_freq == 0:
                states_np.append(state)
            collision_num += infos['collision_num']
        # print(get_delta_thetas(state))
        print(
            f'### Average vel err: {sum(vel_errs) / len(vel_errs)} || Mean vel err: {sum(vel_errs_mean) / len(vel_errs_mean)}###')
        print(f'### Mean Collision Num: {collision_num / HORIZON}, Total Collision Num: {collision_num} ###')
        print(f'### likelihood: {np.log(pdf_func(state, args.num_objs))} ###')
        # save_video(env, states_np, save_path=f'{exp_path}test_video_{episode_idx}', fps=len(states_np) // 5, suffix='mp4')
        save_video(env, states_np, save_path=f'{exp_path}test_video_{episode_idx}', fps=len(states_np) // 5, suffix='gif')


# def eval_orca(seeds=[0, 5, 10, 15, 20]):
# def eval_orca(seeds=[0, 5]):
def eval_orca(seeds=[0]):
    print('----- Start Collecting Trajs -----')
    trajs = []
    trajs_path = exp_path+f'trajs_{args.num_objs}_{args.eval_num}.pickle'
    metrics_path = exp_path+f'metrics_{args.num_objs}_{args.eval_num}.pickle'
    # trajs_path = exp_path+f'{len(seeds)}seeds_trajs_{args.num_objs}_{args.eval_num}.pickle'
    # metrics_path = exp_path+f'{len(seeds)}seeds_metrics_{args.num_objs}_{args.eval_num}.pickle'
    # trajs_path = exp_path+f'{len(seeds)}seeds_trajs_{args.num_objs}_{args.eval_num}_{env.max_episode_len}.pickle'
    # metrics_path = exp_path+f'{len(seeds)}seeds_metrics_{args.num_objs}_{args.eval_num}_{env.max_episode_len}.pickle'
    
    if os.path.exists(trajs_path) and (args.recover == 'True'):
        print('----- Find Existing Trajs!! -----')
        with open(trajs_path, 'rb') as f:
            trajs = pickle.load(f)
    else:
        for seed in seeds:
            env.seed(seed)
            print(f'Seed {seed}: Starting collecting trajs! num: {args.eval_num}')
            cur_trajs = []
            for _ in trange(args.eval_num):
                policy.reset_policy()  # if orca_mode == 'goal', then this func will init the goal state
                state = env.reset(is_random=True)
                traj = []
                infos = None
                for _ in range(args.horizon):
                    action = policy.select_action(state, infos)
                    state, _, _, infos = env.step(action, step_size=PB_FREQ)
                    cur_infos = {'state': state, 'collision_num': infos['collision_num']}
                    traj.append(cur_infos)
                cur_trajs.append(traj)
            trajs.append(cur_trajs)

        # save trajs
        with open(trajs_path, 'wb') as f:
            pickle.dump(trajs, f)
    
    print('----- Start Eval Trajs -----')
    metrics_dicts = []
    for seed, cur_trajs in zip(seeds, trajs):
        # By default, seeds = [0, 5, 10, ... 5*k, ... ]
        metrics_dicts.append(eval_trajs(args.env, cur_trajs, args.num_objs, env, args.exp_name, seed=5*seed))
    # metrics = merge_metrics_dicts(metrics_dicts)
    metrics = metrics_dicts[0]

    # save metrics
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="debug")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--env", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--action_type", type=str, default="vel")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--goal_mode", type=str, default="Score")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--recover", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_onebyone", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_pid", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--score_exp", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_orca", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_decay", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--orca_mode", type=str, default="score")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--horizon", type=int, default=250)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--num_objs", type=int, default=5)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--knn_orca", type=int, default=1)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--target_t0", type=float, default=0.01)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--seed", type=int, default=0)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--eval_num", type=int, default=100)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--mode", type=str, default='debug')  # Policy name (MATD3, DDPG or OurDDPG)
    args = parser.parse_args()

    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    exp_path = f"../logs/{args.exp_name}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    is_onebyone = (args.is_onebyone == 'True')
    is_pid = (args.is_pid == 'True')

    ''' init my env '''
    assert args.env in ['sorting', 'placing', 'hybrid', 'sorting6', 'circlerect']
    exp_data = None  # load expert examples for LfD algorithms
    # PB_FREQ = 4
    PB_FREQ = 8
    MARGIN = 0.10
    BOUND = 0.3
    RADIUS = 0.025 * (1 + MARGIN) / (BOUND - 0.025)
    action_type = args.action_type
    dt_dict = {
        'vel': 0.1 if is_onebyone else 0.02, 
        'force': 0.02,
    }
    max_vel_dict = {
        'vel': 0.3,
        'force': 0.3,
    }
    MAX_VEL = max_vel_dict[action_type]
    dt = dt_dict[action_type]
    tau = 10 * dt
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes': args.num_objs,
        'exp_data': exp_data,
        'time_freq': time_freq * PB_FREQ,
        'is_gui': False,
        'max_action': MAX_VEL,
        'max_episode_len': args.horizon,
        'action_type': action_type,
    }
    env_class = ENV_DICT[args.env]
    env = env_class(**env_kwargs)
    env.seed(args.seed)
    env.reset(is_random=True)

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Load Target Score '''
    network_mode = 'target' if args.env in ['sorting', 'hybrid'] else 'support'
    target_score, diffusion_coeff_fn = load_target_score(args.score_exp, network_mode, args.num_objs, MAX_VEL)

    policy = PlanningAgent(
        target_score, 
        args.num_objs, 
        orca_mode=args.orca_mode, 
        goal_mode=args.goal_mode, 
        env_=env, 
        radius=RADIUS, 
        max_vel=MAX_VEL,
        dt=dt, 
        tau=tau, 
        t0=args.target_t0, 
        knn=args.knn_orca, 
        is_orca=(args.is_orca == 'True'),
        horizon=args.horizon,
        is_decay=(args.is_decay == 'True'), 
        one_by_one=(args.is_onebyone == 'True'),
        is_pid=is_pid,
        )

    if args.mode == 'debug':
        ''' If u gonna debug '''
        debug_orca(args.eval_num)
    elif args.mode == 'eval':
        ''' Eval Episodes '''
        # full_metric(env, args.env, exp_path, policy, args.num_objs, args.exp_name, args.eval_num, recover=(args.recover == 'True'))
        eval_orca()
    else:
        print('ORCA Mode Error!!!')
        raise NotImplementedError



