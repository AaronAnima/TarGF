import numpy as np
import torch
import random
import os
from torchvision.utils import make_grid
from tqdm import trange
import pickle

from algorithms.sac import MASAC, ReplayBuffer
from planners.targf_base import load_target_score
from utils.misc import Timer, RewardNormalizer
from envs.Room.RoomArrangement import RLEnvDynamic, SceneSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardSampler:
    def __init__(
        self,
        normalizer_sim,
        normalizer_col,
        target_score,
        configs,
    ):
        self.normalizer_sim = normalizer_sim
        self.normalizer_col = normalizer_col
        self.target_score = target_score
        self.reward_mode = configs.reward_mode
        self.reward_freq = configs.reward_freq
        self.lambda_sim = configs.lambda_sim
        self.lambda_col = configs.lambda_col
        self.t0 = configs.t0

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
            new_state = new_state.reshape((-1, 3))[:, :2].reshape(-1)
            cur_state = cur_state.reshape((-1, 3))[:, :2].reshape(-1)
            state_change = new_state - cur_state
        else:
            raise ValueError(f"Mode {self.configs.env_type} not recognized.")
        return state_change

    def get_similarity_reward(self, info, cur_state, new_state):
        if self.reward_mode == 'densityIncre':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False) # cur_score: [num_nodes, 2]
            state_change = self.get_state_change(cur_state, new_state)
            similarity_reward = np.sum(state_change * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        else:
            print('Unknown Reward Type!!')
            raise NotImplementedError
        return similarity_reward


def eval_policy(eval_env, eval_policy, reward_func, writer, eval_episodes, eval_idx, visualise=True):
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    # collect meta-datas for follow-up visualisations
    vis_states = []
    room_names = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        cur_vis_states = []
        cur_vis_states.append(state)
        while not done:
            action = eval_policy.select_action(state, sample=False)
            next_state, _, done, infos = eval_env.step(action)
            reward, reward_similarity, reward_collision = reward_func.get_reward(info=infos, cur_state=state, new_state=next_state, is_eval=True)
            state = next_state 
            episode_reward += reward.sum().item()
            episode_similarity_reward += reward_similarity.sum().item()
            episode_collision_reward += reward_collision.sum().item()
        cur_vis_states.append(state)
        room_names.append(eval_env.sim.name)
        vis_states.append(cur_vis_states)
    
    episode_reward /= eval_episodes
    episode_similarity_reward /= eval_episodes
    episode_collision_reward /= eval_episodes

    ''' log eval metrics '''
    writer.add_scalars('Eval/Compare',
                       {'total': episode_reward,
                        'collision': episode_collision_reward,
                        'similarity': episode_similarity_reward},
                        eval_idx)
    writer.add_scalar('Eval/Total', episode_reward, eval_idx)


    ''' visualise the terminal states '''
    if visualise:
        # the real-sense image can only be rendered by scene-sampler (instead of proxy simulator for RL)
        eval_env.close()
        sampler = SceneSampler('bedroom', gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
        imgs = []
        for state, room_name in zip(vis_states, room_names):
            # order: GT -> init -> terminal
            sim = sampler[room_name]
            sim.normalize_room()

            # vis GT state
            img = sim.take_snapshot(512, height=10.0)
            imgs.append(img)

            # vis init/terminal state
            for state_item in state:
                sim.set_state(state_item[1], state_item[0])
                img = sim.take_snapshot(512, height=10.0)
                imgs.append(img)

            # close env after rendering
            sim.disconnect()
        batch_imgs = np.stack(imgs, axis=0)
        ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
        grid = make_grid(ts_imgs.float(), padding=2, nrow=3, normalize=True)
        writer.add_image(f'Images/igibson_terminal_states', grid, eval_idx)


def sac_trainer(configs, log_dir, writer):

    ''' init my env '''
    MAX_VEL = configs.max_vel
    tar_data = 'UnshuffledRoomsMeta'
    exp_kwconfigs = {
        'max_vel': MAX_VEL,
        'pos_rate': 1,
        'ori_rate': 1,
        'max_horizon': configs.horizon,
    }
    env = RLEnvDynamic(
        tar_data,
        exp_kwconfigs,
        meta_name='ShuffledRoomsMeta',
        is_gui=False,
        fix_num=None, 
        is_single_room=(configs.is_single_room == 'True'),
        split='train',
    )

    ''' set seeds '''
    torch.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    ''' Load Target Score '''
    target_score = load_target_score(configs)
 
    ''' Set Timer '''
    # monitoring the time cost for each step
    timer = Timer(writer=writer)

    ''' Init Buffer '''
    replay_buffer = ReplayBuffer(max_size=configs.buffer_size, timer=timer)

    reward_normalizer_sim = RewardNormalizer(configs.normalize_reward == 'True', writer, name='sim')
    reward_normalizer_col = RewardNormalizer(configs.normalize_reward == 'True', writer, name='col')
    reward_func = RewardSampler(
        reward_normalizer_sim,
        reward_normalizer_col,
        target_score=target_score,
        reward_mode=configs.reward_mode,
        reward_freq=configs.reward_freq,
        lambda_sim=configs.lambda_sim,
        lambda_col=configs.lambda_col,
        t0=configs.reward_t0,
    )

    ''' Init policy '''
    kwconfigs = {
        "max_action": MAX_VEL,
        "discount": configs.discount,
        "tau": configs.tau,
        "policy_freq": configs.policy_freq,
        "writer": writer,
        "target_score": target_score,
        "is_residual": configs.is_residual == 'True',
        "hidden_dim": configs.hidden_dim,
        "embed_dim": configs.embed_dim,
        "residual_t0": configs.residual_t0,
        "timer": timer,
    }
    policy = MASAC(**kwconfigs)

    ''' Start Training Episodes '''
    state, done = env.reset(), False
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
        else:
            timer.set()
            action = policy.select_action(state, sample=True)
            timer.log('action')

        # Perform action
        timer.set()
        next_state, _, done, infos = env.step(action)
        reward, reward_similarity, reward_collision = reward_func.get_reward(info=infos, cur_state=state, new_state=next_state)
        timer.log('step_reward')


        done_bool = float(done) if episode_timesteps < configs.horizon else 0

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
            policy.train(replay_buffer, configs.batch_size)
            timer.log('train_step')

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                f"Total: {episode_reward:.3f} "
                f"Collision: {episode_collision_reward:.3f} "
                f"Similarity: {episode_similarity_reward:.3f}")
            writer.add_scalars('Episode_rewards/Compare',
                               {'total': episode_reward,
                                'collision': episode_collision_reward,
                                'similarity': episode_similarity_reward},
                               episode_num + 1)
            writer.add_scalar('Episode_rewards/Total Reward', episode_reward, episode_num + 1)

            # Evaluate episode, save model before eval
            if (episode_num+1) % configs.eval_freq == 0:
                print('------Now Save Models!------')
                ckpt_path = os.path.join('./logs', log_dir, 'policy.pickle')
                with open(ckpt_path, 'wb') as f:
                    policy.writer = None
                    policy.timer = None
                    pickle.dump(policy, f)
                policy.writer = writer
                policy.timer = timer
                print('------Now Start Eval!------')
                eval_num = configs.eval_num     
                eval_policy(env, policy, reward_func, writer, eval_episodes=eval_num, eval_idx=episode_num+1)
                print(f'log_dir: {log_dir}')
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_collision_reward = 0
            episode_similarity_reward = 0
            episode_timesteps = 0
            episode_num += 1

