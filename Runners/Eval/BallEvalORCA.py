import os
import sys

import numpy as np
from math import *
import functools
import torch
import gym
import ebor

import argparse
from collections import deque


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Algorithms.ORCA.pyorca import Agent, orca
from Algorithms.ORCA.RVO import compute_V_des
from Algorithms.ORCA.RVO import compute_V_des
from Runners.Train.BallSAC import load_target_score
from Runners.Eval.BallEvalBase import get_simulation_constants, full_metric, analysis
from ball_utils import pdf_sorting, pdf_placing, pdf_hybrid
PDF_DICT = {'Clustering-v0': pdf_sorting, 'Circling-v0': pdf_placing, 'CirclingClustering-v0': pdf_hybrid}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalise_vels(vels, max_vel):
    # [-inf, +inf] -> [-max_vel, max_vel]**n
    vels = np.array(vels).copy()
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
        knn=2, 
        horizon=100, 
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
            self.goal_state = self.goal_buffer.pop() 
            self.goal_state = self.goal_state.reshape((-1, 2)) 
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
                candidates = self.agents[:i] + self.agents[i + 1:]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="debug")  
    parser.add_argument("--env", type=str, default="sorting")  
    parser.add_argument("--action_type", type=str, default="vel")  
    parser.add_argument("--goal_mode", type=str, default="Score")  
    parser.add_argument("--recover", type=str, default="False")  
    parser.add_argument("--is_onebyone", type=str, default="False")  
    parser.add_argument("--is_pid", type=str, default="False")  
    parser.add_argument("--score_exp", type=str)  
    parser.add_argument("--is_orca", type=str, default="True")  
    parser.add_argument("--is_decay", type=str, default="False")  
    parser.add_argument("--orca_mode", type=str, default="score")  
    parser.add_argument("--horizon", type=int, default=100)  
    parser.add_argument("--num_objs", type=int, default=7)  
    parser.add_argument("--knn_orca", type=int, default=2)  
    parser.add_argument("--residual_t0", type=float, default=0.01)  
    parser.add_argument("--seed", type=int, default=0)  
    parser.add_argument("--eval_num", type=int, default=100)  
    parser.add_argument("--eval_mode", type=str, default='full_metric')  
    args = parser.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    exp_path = f"./logs/{args.log_dir}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    is_onebyone = (args.is_onebyone == 'True')
    is_pid = (args.is_pid == 'True')

    ''' init env and set some physical parameters '''
    MAX_VEL, dt, PB_FREQ, RADIUS, WALL_BOUND = get_simulation_constants()
    MARGIN = 0.1
    RADIUS = RADIUS * (1 + MARGIN) / (WALL_BOUND - RADIUS)
    tau = dt * 10 # orca hyper parameter
    env = gym.make(args.env, n_boxes=args.num_objs)
    env.seed(args.seed)
    env.reset(is_random=True)

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Load Target Score '''
    network_mode = 'target' if 'Clustering' in args.env else 'support'
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
        t0=args.residual_t0, 
        knn=args.knn_orca, 
        is_orca=(args.is_orca == 'True'),
        horizon=args.horizon,
        is_decay=(args.is_decay == 'True'), 
        one_by_one=(args.is_onebyone == 'True'),
        is_pid=is_pid,
        )

    if args.eval_mode == 'analysis':
        ''' If u gonna debug '''
        eval_path = f"./logs/{args.log_dir}/analysis_{args.log_dir}/"
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        analysis_score = policy.target_score
        analysis(env, PDF_DICT[args.env], policy, args.num_objs, eval_episodes=args.eval_num, save_path=f'{eval_path}')
    elif args.eval_mode == 'full_metric':
        ''' Eval Episodes '''
        full_metric(env, args.env, exp_path, policy, args.num_objs, args.log_dir, args.eval_num, recover=(args.recover == 'True'))
    else:
        print('ORCA Mode Error!!!')
        raise NotImplementedError



