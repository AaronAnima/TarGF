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
parser.add_argument("--exp_name", type=str, default="debug")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--env", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--action_type", type=str, default="vel")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--is_onebyone", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--n_boxes", default=10, type=int)  # How often (time steps) we evaluate
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds

parser.add_argument("--eval_mode", type=str, default="analysis")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--eval_num", type=int, default=40)  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--horizon", default=250, type=int)  # How often (time steps) we evaluate
parser.add_argument("--recover", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--is_best", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)

parser.add_argument("--is_residual", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--score_mode", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--inp_mode", type=str, default="state")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--model_type", type=str, default="tanh")  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--hidden_dim", type=int, default=128)  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--embed_dim", type=int, default=64)  # Policy name (MATD3, DDPG or OurDDPG)
parser.add_argument("--knn_actor", default=10, type=int)  # How often (time steps) we evaluate
parser.add_argument("--knn_critic", default=15, type=int)  # How often (time steps) we evaluate
parser.add_argument("--reward_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--residual_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument("--policy_freq", default=1, type=int)  # Frequency of delayed policy updates

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

''' Load Target Score '''
network_mode = 'target' if args.env in ['sorting', 'hybrid', 'sorting6'] else 'support'
target_score, diffusion_coeff_fn = load_target_score(args.score_mode, network_mode, num_per_class, max_vel, is_state=True)
if not is_state:
    target_score_img, _ = load_target_score(args.score_mode + 'Image', network_mode, num_per_class, max_vel, is_state=False)

''' Init policy '''
kwargs = {
    "num_boxes": args.n_boxes,
    "max_action": max_vel,
    "discount": args.discount,
    "tau": args.tau,
    "policy_freq": args.policy_freq,
    "writer": None,
    "knn_actor": args.knn_actor,
    "knn_critic": args.knn_critic,
    "target_score": target_score,
    "is_residual": args.is_residual == 'True',
    "model_type": args.model_type,
    "hidden_dim": args.hidden_dim,
    "embed_dim": args.embed_dim,
    "residual_t0": args.residual_t0,
}
policy = MASAC.MASAC(**kwargs)
if args.is_best == 'True':
    policy.load(f"{exp_path}"+'best')
else:
    policy.load(f"{exp_path}")

''' Start Eval and Save Videos '''
EVAL_NUM = args.eval_num
if args.eval_mode == 'fullmetric':
    seeds = [args.seed + i*5 for i in range(5)]
    full_metric(env, args.env, exp_path, policy, args.n_boxes, args.exp_name, args.eval_num, recover=(args.recover == 'True'), seeds=seeds)
elif args.eval_mode == 'analysis':
    eval_path = f"../logs/{args.exp_name}/analysis_{args.exp_name}/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    analysis_score = target_score if is_state else target_score_img
    analysis(env, PDF_DICT[args.env], policy, args.n_boxes, t0=args.reward_t0, score=analysis_score, eval_episodes=EVAL_NUM, save_path=f'{eval_path}', is_state=is_state)
    # analysis(env, args.env, policy, args.n_boxes, t0=args.reward_t0, score=None, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
else:
    print('--- Eval Mode Error! ---')
    raise NotImplementedError


# if args.eval_mode == 'quickmetric':
#     eval_path = f"../logs/{args.exp_name}/metric_{args.exp_name}/"
#     if not os.path.exists(eval_path):
#         os.makedirs(eval_path)
#     metric(env, args.env, policy, args.n_boxes, eval_episodes=EVAL_NUM)
# elif args.eval_mode == 'fullmetric':
#     full_metric()
# elif args.eval_mode == 'analysis':
#     eval_path = f"../logs/{args.exp_name}/analysis_{args.exp_name}/"
#     if not os.path.exists(eval_path):
#         os.makedirs(eval_path)
#     analysis(env, args.env, policy, args.n_boxes, t0=args.reward_t0, score=target_score, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
# else:
#     print('--- Eval Mode Error! ---')
#     raise NotImplementedError


# import numpy as np
# import torch
# from torch import nn
# import argparse
# import os
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid, save_image
# from ipdb import set_trace
# import sys
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import knn_graph, radius_graph
# import functools
# from functools import partial
# from tqdm import trange
# import pickle



# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from Algorithms.SDE import ScoreModelGNN, marginal_prob_std, diffusion_coeff, ode_likelihood
# from Algorithms.SDEOld import ScoreModelGNNOld
# from Algorithms import MASAC
# from Runners.MASACSorting import load_target_score, RewardSampler
# from Runners.EvalBase import eval_trajs, analysis, full_metric

# from Envs.SortingBall import RLSorting
# from Envs.PlacingBall import RLPlacing
# from Envs.HybridBall import RLHybrid
# from utils import pdf_sorting, pdf_placing, pdf_hybrid, diversity_score, images_to_video, save_video
# ENV_DICT = {'sorting': RLSorting, 'placing': RLPlacing, 'hybrid': RLHybrid}
# PDF_DICT = {'sorting': pdf_sorting, 'placing': pdf_placing, 'hybrid': pdf_hybrid}


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# def debug():
#     videos = take_video(env, policy, eval_episodes=EVAL_NUM)
#     # save video
#     video_duration = 5
#     video_states_np = np.stack(videos)
#     save_video(env, video_states_np, save_path=f'{save_path}video_{idx}', fps=len(video_states_np) // video_duration, suffix='mp4')


# parser = argparse.ArgumentParser()
# parser.add_argument("--policy", default="MASAC")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--exp_name", type=str, default="debug")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--recover", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--env", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--reward_mode", type=str, default="full")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--is_residual", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--load_model", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--eval_mode", type=str, default="analysis")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--score_mode", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--hidden_dim", type=int, default=128)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--embed_dim", type=int, default=64)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--reward_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
# parser.add_argument("--residual_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
# parser.add_argument("--is_ema", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--mar", type=float, default=0.9)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--queue_len", type=int, default=10)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--eval_num", type=int, default=40)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--eval_col", type=int, default=6)  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--model_type", type=str, default="tanh")  # Policy name (MATD3, DDPG or OurDDPG)
# parser.add_argument("--lambda_sim", default=1.0, type=float)  # Std of Gaussian exploration noise
# parser.add_argument("--horizon", default=250, type=int)  # How often (time steps) we evaluate
# parser.add_argument("--reward_freq", default=1, type=int)  # How often (time steps) we evaluate
# parser.add_argument("--lambda_col", default=1, type=float)  # How often (time steps) we evaluate
# parser.add_argument("--n_boxes", default=10, type=int)  # How often (time steps) we evaluate
# parser.add_argument("--knn_actor", default=10, type=int)  # How often (time steps) we evaluate
# parser.add_argument("--knn_critic", default=15, type=int)  # How often (time steps) we evaluate
# parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
# parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
# parser.add_argument("--eval_freq", default=10, type=int)  # How often (episodes!) we evaluate
# parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
# parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
# parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
# parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
# parser.add_argument("--policy_freq", default=1, type=int)  # Frequency of delayed policy updates
# args = parser.parse_args()

# if not os.path.exists("../logs"):
#     os.makedirs("../logs")

# exp_path = f"../logs/{args.exp_name}/"
# if not os.path.exists(exp_path):
#     os.makedirs(exp_path)

# tb_path = f"../logs/{args.exp_name}/tb"
# if not os.path.exists(tb_path):
#     os.makedirs(tb_path)
# writer = SummaryWriter(tb_path)

# num_per_class = args.n_boxes

# ''' init my env '''
# assert args.env in ['sorting', 'placing', 'hybrid']
# exp_data = None  # load expert examples for LfD algorithms
# PB_FREQ = 4
# dt = 0.02
# MAX_VEL = 0.3
# time_freq = int(1 / dt)
# env_kwargs = {
#     'n_boxes': args.n_boxes,
#     'exp_data': exp_data,
#     'time_freq': time_freq * PB_FREQ,
#     'is_gui': False,
#     'max_action': MAX_VEL,
#     'max_episode_len': args.horizon,
# }
# env_class = ENV_DICT[args.env]
# env = env_class(**env_kwargs)
# env.seed(args.seed)
# env.reset(is_random=True)

# ''' set seeds '''
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# ''' Load Target Score '''
# network_mode = 'target' if args.env in ['sorting', 'hybrid'] else 'support'
# target_score, diffusion_coeff_fn = load_target_score(args.score_mode, network_mode, num_per_class, MAX_VEL)

# ''' Init policy '''
# kwargs = {
#     "num_boxes": args.n_boxes,
#     "max_action": MAX_VEL,
#     "discount": args.discount,
#     "tau": args.tau,
#     "policy_freq": args.policy_freq,
#     "writer": writer,
#     "knn_actor": args.knn_actor,
#     "knn_critic": args.knn_critic,
#     "target_score": target_score,
#     "is_residual": args.is_residual == 'True',
#     "model_type": args.model_type,
#     "hidden_dim": args.hidden_dim,
#     "embed_dim": args.embed_dim,
#     "residual_t0": args.residual_t0,
# }
# policy = MASAC.MASAC(**kwargs)
# if args.load_model == 'True':
#     policy.load(f"{exp_path}")

# ''' Start Eval and Save Videos '''
# EVAL_NUM = args.eval_num
# if args.eval_mode == 'fullmetric':
#     full_metric(env, args.env, exp_path, policy, args.n_boxes, args.exp_name, args.eval_num, recover=(args.recover == 'True'))
# elif args.eval_mode == 'analysis':
#     eval_path = f"../logs/{args.exp_name}/analysis_{args.exp_name}/"
#     if not os.path.exists(eval_path):
#         os.makedirs(eval_path)
#     target_score = None if args.env in ['hybrid', 'placing'] else target_score
#     analysis(env, PDF_DICT[args.env], policy, args.n_boxes, t0=args.reward_t0, score=target_score, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
#     # analysis(env, args.env, policy, args.n_boxes, t0=args.reward_t0, score=None, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
# else:
#     print('--- Eval Mode Error! ---')
#     raise NotImplementedError


# # if args.eval_mode == 'quickmetric':
# #     eval_path = f"../logs/{args.exp_name}/metric_{args.exp_name}/"
# #     if not os.path.exists(eval_path):
# #         os.makedirs(eval_path)
# #     metric(env, args.env, policy, args.n_boxes, eval_episodes=EVAL_NUM)
# # elif args.eval_mode == 'fullmetric':
# #     full_metric()
# # elif args.eval_mode == 'analysis':
# #     eval_path = f"../logs/{args.exp_name}/analysis_{args.exp_name}/"
# #     if not os.path.exists(eval_path):
# #         os.makedirs(eval_path)
# #     analysis(env, args.env, policy, args.n_boxes, t0=args.reward_t0, score=target_score, eval_episodes=EVAL_NUM, save_path=f'{eval_path}')
# # else:
# #     print('--- Eval Mode Error! ---')
# #     raise NotImplementedError
