import numpy as np
import torch
import random
import os
from tqdm import trange
import pickle
from ipdb import set_trace
import functools
from collections import OrderedDict

from planners.sac.sac import MASAC, ReplayBuffer
from planners.sac.targf_sac import TarGFSACPlanner
from planners.gf_wrapper.targf import load_targf
from utils.misc import Timer, RewardNormalizer
from utils.evaluations import training_time_eval_room, training_time_eval_ball
from envs.envs import get_env
from networks.actor_critics import BallActor, RoomActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardSampler:
    def __init__(
        self,
        normalizer_sim,
        normalizer_col,
        targf,
        configs,
    ):
        self.normalizer_sim = normalizer_sim
        self.normalizer_col = normalizer_col
        self.targf = targf
        self.reward_mode = configs.reward_mode
        self.reward_freq = configs.reward_freq
        self.lambda_sim = configs.lambda_sim
        self.lambda_col = configs.lambda_col
        self.t0 = configs.reward_t0
        self.configs = configs

    def get_reward(self, info, cur_state, new_state, is_eval=False):
        collision_reward = self.get_collision_reward(info)
        similarity_reward = self.get_similarity_reward(info, cur_state, new_state)
        
        similarity_reward = self.normalizer_sim.update(similarity_reward, is_eval=is_eval) 
        collision_reward = self.normalizer_col.update(collision_reward, is_eval=is_eval)

        total_reward = self.lambda_sim*similarity_reward + self.lambda_col*collision_reward

        return total_reward, similarity_reward, collision_reward

    def get_collision_reward(self, info):
        collision_num = info['collision_num']
        collisions = np.array([np.sum(collision_num[:, i]) + np.sum(collision_num[i, :]) for i in range(collision_num.shape[0])])
        collision_reward = -np.array(collisions, 'float32')
        return collision_reward
    
    def get_state_change(self, cur_state, new_state):
        if self.configs.env_type == 'Room':
            state_change = new_state[-1][:, 0:4] - cur_state[-1][:, 0:4] # delta_state: [num_nodes, 2]
        elif self.configs.env_type == 'Ball':
            new_state = new_state.reshape((-1, 3))[:, :2]
            cur_state = cur_state.reshape((-1, 3))[:, :2]
            state_change = new_state - cur_state
        else:
            raise ValueError(f"Mode {self.configs.env_type} not recognized.")
        return state_change

    def get_similarity_reward(self, info, cur_state, new_state):
        if self.reward_mode == 'densityIncre':
            cur_score = self.targf.inference(cur_state, t0=self.t0, grad_2_act=False, norm_type='None') # cur_score: [num_nodes, 2]
            state_change = self.get_state_change(cur_state, new_state)
            similarity_reward = np.sum(state_change * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        else:
            print('Unknown Reward Type!!')
            raise NotImplementedError
        return similarity_reward



def get_functions(configs, env, reward_func):
    if configs.env_type == 'Room': 
        eval_fn = functools.partial(training_time_eval_room, reward_func=reward_func)
    elif configs.env_type == 'Ball':
        eval_fn = functools.partial(training_time_eval_ball, nrow=configs.eval_col, pdf_func=env.pseudo_likelihoods)
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return eval_fn

def get_targf_sac_policy(configs, targf, max_action):
    # init actor_net
    ACTOR_DICT = {'Ball': BallActor, 'Room': RoomActor}
    actor_class = ACTOR_DICT[configs.env_type]
    actor_net = actor_class(
        configs,
        targf=targf,
        max_action=max_action,
        log_std_bounds=(-5, 2),
    ).to(device)
    # init policy_net
    policy = TarGFSACPlanner(
        configs,
        actor_net,
        targf,
        max_action,
    )
    return policy

def rl_trainer(configs, log_dir, writer):

    ''' init my env '''
    env, max_action = get_env(configs)

    ''' set seeds '''
    torch.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    ''' Load Target Score '''
    targf = load_targf(configs, max_action)
 
    ''' Set Timer '''
    # monitoring the time cost for each step
    timer = Timer(writer=writer)

    ''' Init Buffer '''
    replay_buffer = ReplayBuffer(configs, timer=timer)

    reward_normalizer_sim = RewardNormalizer(configs.normalize_reward, writer, name='sim')
    reward_normalizer_col = RewardNormalizer(configs.normalize_reward, writer, name='col')
    reward_func = RewardSampler(
        reward_normalizer_sim,
        reward_normalizer_col,
        targf=targf,
        configs=configs
    )

    ''' setup eval functions '''
    eval_func = get_functions(configs, env, reward_func)

    ''' Init RL Trainer and targf(sac) policy '''
    policy = get_targf_sac_policy(configs, targf, max_action)
    kwargs = {
        "max_action": max_action,
        "actor": policy, 
        "writer": writer,
        "targf": targf,
        "timer": timer,
        "configs": configs,
    }
    sac_trainer = MASAC(**kwargs)

    ''' Start Training Episodes '''
    state, done = env.reset(), False
    if isinstance(state, OrderedDict):
        state = env.flatten_states([state])[0]
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in trange(int(configs.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < configs.start_timesteps:
            action = env.sample_action()
            if isinstance(action, OrderedDict):
                action = env.flatten_actions([action])[0]
        else:
            timer.set()
            action = policy.select_action(state, sample=True)
            timer.log('action')

        # Perform action
        timer.set()
        next_state, _, done, infos = env.step(action)
        if isinstance(next_state, OrderedDict):
            next_state = env.flatten_states([next_state])[0]
        reward, reward_similarity, reward_collision = reward_func.get_reward(info=infos, cur_state=state, new_state=next_state)
        timer.log('step_reward')

        done_bool = float(done) if episode_timesteps < env.max_episode_len else 0

        timer.set()
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        timer.log('buffer')

        state = next_state
        episode_reward += reward.sum().item()
        episode_similarity_reward += reward_similarity.sum().item()
        episode_collision_reward += reward_collision.sum().item()

        # Train agent after collecting sufficient data
        if t >= configs.start_timesteps:
            timer.set()
            sac_trainer.train(replay_buffer, configs.batch_size_rl)
            timer.log('train_step')

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                f"Total: {episode_reward:.3f} "
                f"Collision: {episode_collision_reward:.3f} "
                f"Similarity: {episode_similarity_reward*env.num_objs:.3f}")
            writer.add_scalars('Episode_rewards/Compare',
                               {'total': episode_reward,
                                'collision': episode_collision_reward,
                                'similarity': episode_similarity_reward*env.num_objs},
                               episode_num + 1)
            writer.add_scalar('Episode_rewards/Total Reward', episode_reward, episode_num + 1)

            # Evaluate episode, save model before eval
            if (episode_num + 1) % configs.eval_freq_rl == 0:
                # save models
                print('------Now Save Models!------')
                ckpt_path = os.path.join('./logs', log_dir, 'policy.pickle')
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(policy, f)
                
                # eval cur policy
                print('------Now Start Eval!------')
                eval_num = configs.eval_num     
                eval_func(env, policy, writer, eval_episodes=eval_num, eval_idx=episode_num+1)
                print(f'log_dir: {log_dir}')
            
            # Reset environment
            state, done = env.reset(), False
            if isinstance(state, OrderedDict):
                state = env.flatten_states([state])[0]
            episode_reward = 0
            episode_collision_reward = 0
            episode_similarity_reward = 0
            episode_timesteps = 0
            episode_num += 1

