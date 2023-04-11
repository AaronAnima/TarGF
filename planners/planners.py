import torch
import os
import pickle

from planners.gf_wrapper.targf import load_targf
from planners.sac.targf_sac import TarGFSACPlanner # for pickle.load()
from planners.orca.targf_orca import TarGFORCAPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_planners(configs, eval_env, max_action):
    if configs.policy_type == 'targf_orca':
        targf = load_targf(configs, max_action)
        
        # get simulation parameters
        horizon = eval_env.max_episode_len
        dt = (1 / eval_env.time_freq) * eval_env.sim_steps_each_time
        tau = 10 * dt
        radius = eval_env.r
        bound = eval_env.bound
        max_vel = eval_env.max_action

        radius_orca = (1 + 0.1) * radius / (bound - radius)
        policy = TarGFORCAPlanner(
            targf,
            configs,
            radius=radius_orca,
            max_vel=max_vel, 
            dt=dt, 
            tau=tau, 
            horizon=horizon, 
        )
    elif configs.policy_type == 'targf_sac':
        with open(os.path.join('./logs', configs.policy_exp, 'policy.pickle'), 'rb') as f:
            policy = pickle.load(f)
        policy.to(device)
    else:
        raise ValueError(f"Mode {configs.policy_type} not recognized.")
    return policy
