import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

import torch
import pickle

from Runners.RoomSAC import load_target_score
from Runners.RoomEvalBase import get_max_vel, load_test_env, set_seed, eval_policy
from Runners.RoomSAC import TargetScore

from ipdb import set_trace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)  
    parser.add_argument("--is_single_room", type=str, default="False")  
    parser.add_argument("--horizon", default=250, type=int)  
    
    parser.add_argument("--score_exp", type=str)  
    parser.add_argument("--sigma", type=float, default=25.)  
    parser.add_argument("--hidden_dim", type=int, default=128)  
    parser.add_argument("--embed_dim", type=int, default=64)  
    parser.add_argument("--is_residual", type=str, default="True")  
    parser.add_argument("--residual_t0", default=0.01, type=float)  
    
    parser.add_argument("--discount", default=0.95, type=float) 
    parser.add_argument("--tau", default=0.005, type=float) 
    parser.add_argument("--policy_freq", default=1, type=int)

    parser.add_argument("--eval_num", type=int, default=100)  
    parser.add_argument("--save_video", type=str, default="True")
    args = parser.parse_args()

    exp_path = f"../logs/{args.exp_name}/"
    
    ''' init my env '''
    max_vel = get_max_vel()
    env = load_test_env(args.horizon, (args.is_single_room == 'True'))

    ''' Init policy '''
    with open(f'{exp_path}/policy.pickle', 'rb') as f:
        policy = pickle.load(f)

    eval_policy(env, policy, args.eval_num, args.exp_name, exp_path, args.save_video)

