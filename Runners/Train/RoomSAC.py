from re import T
import numpy as np
import torch
import argparse
import random
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import sys
import functools
from tqdm import trange
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from room_utils import RewardNormalizer
from Algorithms.RoomSDE import marginal_prob_std, diffusion_coeff, score_to_action
from Algorithms.RoomSAC import MASAC, ReplayBuffer
from room_utils import prepro_graph_batch, Timer
from Envs.RoomArrangement import RLEnvDynamic, SceneSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TargetScore:
    def __init__(self, score, max_vel, sigma=25):
        self.score = score
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False, is_action=False):
        if not empty:
            if not torch.is_tensor(state_inp[0]):
                state_inp = prepro_graph_batch([state_inp])
            bs = state_inp[0].shape[0] # state_inp[0]: wall_feat
            t = torch.tensor([t0] * bs)[state_inp[1].batch].unsqueeze(1).to(device) # [num_nodes, 1]
            out_score = self.score(state_inp, t) 
            out_score = out_score.detach()
            if is_action:
                out_score = score_to_action(out_score, state_inp[-1].x)
            if is_norm:
                out_score = out_score * torch.min(
                    torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
            else:
                out_score = out_score
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score

class RewardSampler:
    def __init__(
            self,
            normalizer_sim,
            normalizer_col,
            target_score=None,
            reward_mode='cos',
            reward_freq=1,
            lambda_sim=1.0,
            lambda_col=1.0,
            t0=0.01,
    ):
        self.normalizer_sim = normalizer_sim
        self.normalizer_col = normalizer_col
        self.target_score = target_score
        self.reward_mode = reward_mode
        self.reward_freq = reward_freq
        self.lambda_sim = lambda_sim
        self.lambda_col = lambda_col
        self.t0 = t0
        self.is_centralized = False

    def get_reward(self, actions, info, cur_state, new_state, is_eval=False):
        collision_reward = self.get_collision_reward(info)
        similarity_reward = self.get_similarity_reward(actions, info, cur_state, new_state)
        
        # ''' update sim, instead of total reward '''
        similarity_reward = self.normalizer_sim.update(similarity_reward, is_eval=is_eval) 
        collision_reward = self.normalizer_col.update(collision_reward, is_eval=is_eval)

        total_reward = self.lambda_sim*similarity_reward + self.lambda_col*collision_reward

        return total_reward, similarity_reward, collision_reward

    def get_collision_reward(self, info):
        collision_num = info['collision_num']
        collisions = np.array([np.sum(collision_num[:, i]) + np.sum(collision_num[i, :]) for i in range(collision_num.shape[0])])
        collision_reward = -np.sum(collisions).item() if self.is_centralized else -np.array(collisions, 'float32')
        return collision_reward

    def get_similarity_reward(self, actions, info, cur_state, new_state):
        if self.reward_mode == 'densityIncre':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False) # cur_score: [num_nodes, 2]
            delta_state = new_state[-1][:, 0:4] - cur_state[-1][:, 0:4] # delta_state: [num_nodes, 2]
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        else:
            print('Unknown Reward Type!!')
            raise NotImplementedError
        return similarity_reward


def load_target_score(score_exp, sigma, max_action, hidden_dim, embed_dim):
    tar_path = f'../logs/{score_exp}/score.pt'

    with open(tar_path, 'rb') as f:
        score = pickle.load(f)

    return TargetScore(score, max_action)


def eval_metric(eval_env, eval_policy, reward_func, writer, eval_episodes, eval_idx):
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    vis_states = []
    room_names = []
    imgs = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        cur_vis_states = []
        cur_vis_states.append(state)
        while not done:
            action = eval_policy.select_action(state, sample=False)
            next_state, _, done, infos = eval_env.step(action)
            reward, reward_similarity, reward_collision = reward_func.get_reward(actions=action, info=infos, cur_state=state, new_state=next_state, is_eval=True)
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
    return vis_states, room_names


def visualize_states(states, room_names, eval_idx, writer):
    ''' render and vis terminal states '''
    sampler = SceneSampler('bedroom', gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
    imgs = []
    for state, room_name in zip(states, room_names):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="MASAC")  
    parser.add_argument("--log_dir", type=str)  
    parser.add_argument("--reward_mode", type=str, default="densityIncre")  
    parser.add_argument("--is_residual", type=str, default="True")  
    parser.add_argument("--is_single_room", type=str, default="False")  
    parser.add_argument("--normalize_reward", type=str, default="True")  
    parser.add_argument("--score_exp", type=str)  
    parser.add_argument("--eval_num", type=int, default=4)  
    parser.add_argument("--sigma", type=float, default=25.)  
    parser.add_argument("--max_vel", type=float, default=4.)  
    parser.add_argument("--hidden_dim", type=int, default=128)  
    parser.add_argument("--embed_dim", type=int, default=64)  
    parser.add_argument("--buffer_size", type=int, default=1e6)  
    parser.add_argument("--reward_t0", default=0.01, type=float)  
    parser.add_argument("--residual_t0", default=0.01, type=float)  
    parser.add_argument("--horizon", default=250, type=int)  
    parser.add_argument("--reward_freq", default=1, type=int)  
    parser.add_argument("--lambda_col", default=1.0, type=float) 
    parser.add_argument("--lambda_sim", default=5.0, type=float)  
    parser.add_argument("--seed", default=0, type=int) 
    parser.add_argument("--start_timesteps", default=25e3, type=int)  
    parser.add_argument("--eval_freq", default=50, type=int) 
    parser.add_argument("--max_timesteps", default=1e6, type=int)  
    parser.add_argument("--batch_size", default=256, type=int) 
    parser.add_argument("--discount", default=0.95, type=float) 
    parser.add_argument("--tau", default=0.005, type=float) 
    parser.add_argument("--policy_freq", default=1, type=int) 
    args = parser.parse_args()

    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    exp_path = f"../logs/{args.log_dir}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    tb_path = f"../logs/{args.log_dir}/tb"
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

    ''' init my env '''
    MAX_VEL = args.max_vel
    tar_data = 'UnshuffledRoomsMeta'
    exp_kwargs = {
        'max_vel': MAX_VEL,
        'pos_rate': 1,
        'ori_rate': 1,
        'max_horizon': args.horizon,
    }
    env = RLEnvDynamic(
        tar_data,
        exp_kwargs,
        meta_name='ShuffledRoomsMeta',
        is_gui=False,
        fix_num=None, 
        is_single_room=(args.is_single_room == 'True'),
        split='train',
    )

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ''' Load Target Score '''
    target_score = load_target_score(args.score_exp, args.sigma, MAX_VEL, args.hidden_dim, args.embed_dim)
 
    ''' Set Timer '''
    timer = Timer(writer=writer)

    ''' Init Buffer '''
    replay_buffer = ReplayBuffer(max_size=args.buffer_size, timer=timer)

    reward_normalizer_sim = RewardNormalizer(args.normalize_reward == 'True', writer, name='sim')
    reward_normalizer_col = RewardNormalizer(args.normalize_reward == 'True', writer, name='col')
    reward_func = RewardSampler(
        reward_normalizer_sim,
        reward_normalizer_col,
        target_score=target_score,
        reward_mode=args.reward_mode,
        reward_freq=args.reward_freq,
        lambda_sim=args.lambda_sim,
        lambda_col=args.lambda_col,
        t0=args.reward_t0,
    )

    ''' Init policy '''
    kwargs = {
        "max_action": MAX_VEL,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "writer": writer,
        "target_score": target_score,
        "is_residual": args.is_residual == 'True',
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "residual_t0": args.residual_t0,
        "timer": timer,
    }
    policy = MASAC(**kwargs)

    ''' Start Training Episodes '''
    state, done = env.reset(), False
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in trange(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            timer.set()
            action = policy.select_action(state, sample=True)
            timer.log('action')

        # Perform action
        timer.set()
        next_state, _, done, infos = env.step(action)
        reward, reward_similarity, reward_collision = reward_func.get_reward(actions=action, info=infos, cur_state=state, new_state=next_state)
        timer.log('step_reward')


        done_bool = float(done) if episode_timesteps < args.horizon else 0

        timer.set()
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        timer.log('buffer')

        state = next_state
        episode_reward += reward.sum().item()
        episode_similarity_reward += reward_similarity.sum().item()
        episode_collision_reward += reward_collision.sum().item()

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            timer.set()
            policy.train(replay_buffer, args.batch_size)
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
            if (episode_num+1) % args.eval_freq == 0:
                print('------Now Save Models!------')
                with open(f'{exp_path}/policy.pickle', 'wb') as f:
                    policy.writer = None
                    policy.timer = None
                    pickle.dump(policy, f)
                policy.writer = writer
                policy.timer = timer
                # policy.save(f"{exp_path}")
                print('------Now Start Eval!------')
                eval_num = args.eval_num     
                terminal_states, room_names = eval_metric(env, policy, reward_func, writer, eval_episodes=eval_num, eval_idx=episode_num+1)
                env.close()
                visualize_states(terminal_states, room_names, eval_idx=episode_num+1, writer=writer)
                print(f'log_dir: {args.log_dir}')
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_collision_reward = 0
            episode_similarity_reward = 0
            episode_timesteps = 0
            episode_num += 1

