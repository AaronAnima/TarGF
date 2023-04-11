import pickle
import os
import numpy as np
import sys
import argparse
import torch
import gym
import functools
from ipdb import set_trace


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Runners.Eval.BallEvalBase import get_simulation_constants, analysis, full_metric
from Runners.Train.BallSAC import TargetScore # to unpickle the checkpoints
from Runners.Train.BallSAC import load_target_score
from Algorithms.ORCA.pyorca import Agent, orca
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    parser = argparse.ArgumentParser()
    # file 
    parser.add_argument("--log_dir", type=str)  
    parser.add_argument("--score_exp", type=str) 
    # env
    parser.add_argument("--pattern", type=str, default="CircleCluster") 
    parser.add_argument("--num_per_class", default=7, type=int)  
    parser.add_argument("--num_classes", default=3, type=int)  
    parser.add_argument("--action_type", type=str, default="vel")  
    parser.add_argument("--seed", default=0, type=int)  
    # eval
    parser.add_argument("--eval_mode", type=str, default="full_metric") 
    parser.add_argument("--eval_num", type=int, default=100) 
    parser.add_argument("--horizon", default=100, type=int)  
    parser.add_argument("--recover", type=str, default="False")
    parser.add_argument("--is_best", type=str, default="False")  
    # policy
    parser.add_argument("--policy_mode", type=str) # ['SAC', 'ORCA']  
    parser.add_argument("--is_decay", type=str, default='False') # 'True' in paper
    parser.add_argument("--orca_t0", type=float, default=0.01) # key hyper parameter of ORCA, 0.1 in paper
    parser.add_argument("--knn_orca", type=int, default=2) # key hyper parameter of ORCA

    args = parser.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    exp_path = f"./logs/{args.log_dir}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    num_per_class = args.num_per_class
    num_classes = args.num_classes
    num_objs = num_per_class*num_classes
    eval_num = args.eval_num

    ''' init my env '''
    env_name = '{}-{}Ball{}Class-v0'.format(args.pattern, num_objs, num_classes)
    env = gym.make(env_name)
    env.seed(args.seed)
    env.reset()
    max_action = env.action_space['obj1']['linear_vel'].high[0]
    horizon = env.max_episode_len

    # get pseudo likelihood function
    pdf_func = env.pseudo_likelihoods

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' init policy '''
    if args.policy_mode == 'SAC':
        with open(f'{exp_path}policy.pickle', 'rb') as f:
            policy = pickle.load(f)
    elif args.policy_mode == 'ORCA':
        # get constants for ORCA planner
        MAX_VEL, dt, PB_FREQ, RADIUS, WALL_BOUND = get_simulation_constants() # !! these parameters should be aligned with the Ebor
        MARGIN = 0.1
        RADIUS = RADIUS * (1 + MARGIN) / (WALL_BOUND - RADIUS)
        tau = dt * 10 # orca's hyper-parameter
        target_score, diffusion_coeff_fn = load_target_score(args.score_exp, num_objs, max_action)
        policy = PlanningAgent(
            target_score, 
            num_objs, 
            env_=env, 
            radius=RADIUS, 
            max_vel=MAX_VEL,
            is_decay=args.is_decay == 'True', 
            dt=dt, 
            tau=tau, 
            t0=args.orca_t0, 
            knn=args.knn_orca, 
            horizon=horizon,
        )
    else:
        raise NotImplementedError

    ''' Start Eval and Save Videos '''
    if args.eval_mode == 'full_metric':
        # set seeds for evaluation
        seeds = [args.seed + i*5 for i in range(2)] 

        full_metric(args.pattern, env, pdf_func, exp_path, policy, num_objs, args.log_dir, eval_num, recover=(args.recover == 'True'), seeds=seeds)
    elif args.eval_mode == 'analysis':
        # create folder for analysis
        eval_path = f"./logs/{args.log_dir}/analysis_{args.log_dir}/"
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        
        analysis(env, pdf_func, policy, eval_episodes=eval_num, save_path=f'{eval_path}')
    else:
        raise NotImplementedError

