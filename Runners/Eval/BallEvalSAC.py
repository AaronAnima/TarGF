import pickle
import os
import sys
import argparse
import torch
import gym


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Runners.Eval.BallEvalBase import get_simulation_constants, set_seed, analysis, full_metric
from Runners.Train.BallSAC import TargetScore # to unpickle the checkpoints
from ball_utils import pdf_sorting, pdf_placing, pdf_hybrid
PDF_DICT = {'Clustering-v0': pdf_sorting, 'Circling-v0': pdf_placing, 'CirclingClustering-v0': pdf_hybrid}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)  
parser.add_argument("--env", type=str)  
parser.add_argument("--action_type", type=str, default="vel")  
parser.add_argument("--is_onebyone", type=str, default="False")  
parser.add_argument("--num_objs", default=7, type=int)  
parser.add_argument("--seed", default=0, type=int)  

parser.add_argument("--eval_mode", type=str, default="full_metric") 
parser.add_argument("--eval_num", type=int, default=100) 
parser.add_argument("--horizon", default=100, type=int)  
parser.add_argument("--recover", type=str, default="False")
parser.add_argument("--is_best", type=str, default="False")  

parser.add_argument("--is_residual", type=str, default="True")  
parser.add_argument("--score_exp", type=str)  
parser.add_argument("--inp_mode", type=str, default="state")  
parser.add_argument("--model_type", type=str, default="tanh")  
parser.add_argument("--hidden_dim", type=int, default=128)  
parser.add_argument("--embed_dim", type=int, default=64)  
parser.add_argument("--knn_actor", default=20, type=int)  
parser.add_argument("--knn_critic", default=20, type=int)  
parser.add_argument("--reward_t0", default=0.01, type=float)  
parser.add_argument("--residual_t0", default=0.01, type=float)  
parser.add_argument("--discount", default=0.95, type=float)
parser.add_argument("--tau", default=0.005, type=float) 
parser.add_argument("--policy_freq", default=1, type=int)  

args = parser.parse_args()

if not os.path.exists("./logs"):
    os.makedirs("./logs")

exp_path = f"./logs/{args.log_dir}/"
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

tb_path = f"./logs/{args.log_dir}/tb"
if not os.path.exists(tb_path):
    os.makedirs(tb_path)

num_per_class = args.num_objs
inp_mode = args.inp_mode
is_state = (inp_mode == 'state')
is_onebyone = (args.is_onebyone == 'True')

''' init my env '''
MAX_VEL, dt, PB_FREQ, RADIUS, _ = get_simulation_constants()
env = gym.make(args.env, n_boxes=args.num_objs)
env.seed(args.seed)
env.reset()

''' set seeds '''
set_seed(args.seed)

with open(f'{exp_path}policy.pickle', 'rb') as f:
    policy = pickle.load(f)

''' Start Eval and Save Videos '''
EVAL_NUM = args.eval_num
if args.eval_mode == 'full_metric':
    seeds = [args.seed + i*5 for i in range(5)]
    full_metric(env, args.env, exp_path, policy, args.num_objs, args.log_dir, args.eval_num, recover=(args.recover == 'True'), seeds=seeds)
elif args.eval_mode == 'analysis':
    eval_path = f"./logs/{args.log_dir}/analysis_{args.log_dir}/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    analysis_score = policy.target_score
    analysis(env, PDF_DICT[args.env], policy, args.num_objs, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
else:
    print('--- Eval Mode Error! ---')
    raise NotImplementedError