import os
import sys
import argparse
import torch
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Runners.Eval.RoomEvalBase import get_max_vel, load_test_env, eval_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file
    parser.add_argument("--log_dir", type=str)  
    parser.add_argument("--score_exp", type=str)  
    # env
    parser.add_argument("--is_single_room", type=str, default="False")  
    parser.add_argument("--horizon", default=250, type=int)  
    # eval
    parser.add_argument("--eval_num", type=int, default=100)  
    parser.add_argument("--save_video", type=str, default="True")
    args = parser.parse_args()

    exp_path = f"./logs/{args.log_dir}/"
    
    ''' init my env '''
    max_vel = get_max_vel()
    env = load_test_env(args.horizon, (args.is_single_room == 'True'))

    ''' Init policy '''
    with open(f'{exp_path}/policy.pickle', 'rb') as f:
        policy = pickle.load(f)

    eval_policy(env, policy, args.eval_num, args.log_dir, exp_path, args.save_video)

