import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

import numpy as np

import torch

from Algorithms import BallSAC
from Runners.BallSAC import TargetScore, load_target_score
from Runners.BallEvalBase import get_max_vel, load_env, set_seed, analysis, full_metric


from utils import pdf_sorting, pdf_placing, pdf_hybrid

from ipdb import set_trace

PDF_DICT = {'sorting': pdf_sorting, 'placing': pdf_placing, 'hybrid': pdf_hybrid}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="debug")  
parser.add_argument("--env", type=str, default="sorting")  
parser.add_argument("--action_type", type=str, default="vel")  
parser.add_argument("--is_onebyone", type=str, default="False")  
parser.add_argument("--n_boxes", default=10, type=int)  
parser.add_argument("--seed", default=0, type=int)  

parser.add_argument("--eval_mode", type=str, default="analysis") 
parser.add_argument("--eval_num", type=int, default=40) 
parser.add_argument("--horizon", default=250, type=int)  
parser.add_argument("--recover", type=str, default="False")
parser.add_argument("--is_best", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)

parser.add_argument("--is_residual", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--score_exp", type=str, default="")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--inp_mode", type=str, default="state")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--model_type", type=str, default="tanh")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--hidden_dim", type=int, default=128)  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--embed_dim", type=int, default=64)  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--knn_actor", default=10, type=int)  # How often (time steps) we evaluate
parser.add_argument("--knn_critic", default=15, type=int)  # How often (time steps) we evaluate
parser.add_argument("--reward_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--residual_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float) 
parser.add_argument("--policy_freq", default=1, type=int)  

args = parser.parse_args()

if not os.path.exists("../logs"):
    os.makedirs("../logs")

exp_path = f"../logs/{args.exp_name}/"
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

tb_path = f"../logs/{args.exp_name}/tb"
if not os.path.exists(tb_path):
    os.makedirs(tb_path)

num_per_class = args.n_boxes
inp_mode = args.inp_mode
is_state = (inp_mode == 'state')
is_onebyone = (args.is_onebyone == 'True')

''' init my env '''
max_vel = get_max_vel()
env = load_env(args.env, max_vel, args.n_boxes, args.horizon, args.seed, is_onebyone=is_onebyone, action_type=args.action_type)

''' set seeds '''
set_seed(args.seed)

with open(f'{exp_path}/policy.pickle', 'rb') as f:
    policy = pickle.load(f)

''' Start Eval and Save Videos '''
EVAL_NUM = args.eval_num
if args.eval_mode == 'fullmetric':
    seeds = [args.seed + i*5 for i in range(5)]
    full_metric(env, args.env, exp_path, policy, args.n_boxes, args.exp_name, args.eval_num, recover=(args.recover == 'True'), seeds=seeds)
elif args.eval_mode == 'analysis':
    eval_path = f"../logs/{args.exp_name}/analysis_{args.exp_name}/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    analysis_score = policy.target_score
    analysis(env, PDF_DICT[args.env], policy, args.n_boxes, t0=args.reward_t0, score=analysis_score, eval_episodes=EVAL_NUM, save_path=f'{eval_path}', is_state=is_state)
    # analysis(env, args.env, policy, args.n_boxes, t0=args.reward_t0, score=None, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
else:
    print('--- Eval Mode Error! ---')
    raise NotImplementedError