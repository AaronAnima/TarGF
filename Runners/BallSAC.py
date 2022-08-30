import numpy as np
import torch
from torch import nn
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from ipdb import set_trace
import sys
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, radius_graph
import functools
from functools import partial
from tqdm import trange
import pickle
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import pdf_sorting, pdf_placing, pdf_hybrid, diversity_score, RewardNormalizer
from Algorithms.BallSDE import ScoreModelGNN, marginal_prob_std, diffusion_coeff, ode_likelihood, pc_sampler_state, ode_sampler, Euler_Maruyama_sampler, ScoreNet
from Algorithms import BallSAC

from Envs.SortingBall import RLSorting
from Envs.PlacingBall import RLPlacing
from Envs.HybridBall import RLHybrid
ENV_DICT = {'sorting': RLSorting, 'placing': RLPlacing, 'hybrid': RLHybrid}
PDF_DICT = {'sorting': pdf_sorting, 'placing': pdf_placing, 'hybrid': pdf_hybrid}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_states(eval_states, env, logger, epoch, nrow, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    render_size = 256
    for box_state in eval_states:
        env.set_state(box_state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


class TargetScore:
    def __init__(self, score, num_objs, max_vel, sigma=25, sampler='ode', is_state=True):
        self.score = score
        self.is_state = is_state
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        self.sampler_dict = {
            'pc': pc_sampler_state,
            'ode': ode_sampler, 
            'em': Euler_Maruyama_sampler,
        }
        self.sampler = self.sampler_dict[sampler]

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if self.is_state:
                # 不管是numpy，或者tensor，都可以 -> tensor-device
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                edge_index = knn_graph(state_inp.view(-1, 2), k=self.num_objs*3-1, batch=samples_batch)
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                inp_data = Data(x=state_inp.view(-1, 2), edge_index=edge_index)
                out_score = self.score(inp_data, t, self.num_objs) # 这里传每类就好了
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                # state_ts = 2*state_ts/255. - 1 # [0., 255] -> [-1, 1]
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score

    def sample_goals(self, num_samples, is_batch=False):
        # _, final_states = self.sampler(
        #     self.score, 
        #     self.marginal_prob_std_fn, 
        #     self.diffusion_coeff_fn, 
        #     n_box=self.num_objs, 
        #     batch_size=num_samples,
        #     t0=1.0,
        #     n=3*self.num_objs,
        # )
        final_states = self.graphvae.sample(num_samples, device)
        # final_states = final_states.detach().cpu().numpy()
        if is_batch:
            final_states = final_states.reshape(-1, self.num_objs*6)
        else:
            final_states = final_states.flatten()
        return final_states


class RewardSampler:
    def __init__(
            self,
            num_objs,
            normalizer_sim,
            normalizer_col,
            env_type,
            target_score=None,
            reward_mode='cos',
            reward_freq=1,
            is_centralized=True,
            lambda_sim=1.0,
            lambda_col=1.0,
            t0=0.01,
            is_state=True,
    ):
        self.num_objs = num_objs
        self.normalizer_sim = normalizer_sim
        self.normalizer_col = normalizer_col
        self.target_score = target_score
        self.reward_mode = reward_mode
        self.reward_freq = reward_freq
        self.is_centralized = is_centralized
        self.lambda_sim = lambda_sim
        self.lambda_col = lambda_col
        self.t0 = t0
        self.prev_sim_reward = None
        self.env_type = env_type
        self.is_state = is_state
    
    def reset_reward(self):
        self.prev_sim_reward = None

    def get_reward(self, actions, info, cur_state, new_state, is_eval=False):
        collision_reward = self.get_collision_reward(info)
        similarity_reward = self.get_similarity_reward(actions, info, cur_state, new_state)

        # ''' update sim, instead of total reward '''
        similarity_reward = self.normalizer_sim.update(similarity_reward, is_eval=is_eval)
        collision_reward = self.normalizer_col.update(collision_reward, is_eval=is_eval)
        
        total_reward = self.lambda_sim*similarity_reward + self.lambda_col*collision_reward
        # ''' update total, instead of sim reward '''
        # total_reward = self.normalizer.update(total_reward, is_eval=is_eval)

        return total_reward, similarity_reward, collision_reward

    def get_collision_reward(self, info):
        collision_num = info['collision_num']
        collisions = np.array([np.sum(collision_num[:, i]) + np.sum(collision_num[i, :]) for i in range(self.num_objs*3)])
        collision_reward = -np.sum(collisions).item() if self.is_centralized else -np.array(collisions, 'float32')
        return collision_reward

    def get_similarity_reward(self, actions, info, cur_state, new_state):
        pdf = PDF_DICT[self.env_type]
        if self.reward_mode == 'collision_only':
            return 0
        elif self.reward_mode == 'pdf':
            similarity_reward = np.log((new_state, self.num_objs))
            similarity_reward = np.array([similarity_reward])
            similarity_reward = (similarity_reward+200)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
            # similarity_reward *= cur_infos['progress'] >= 0.96 # last 10 steps
        elif self.reward_mode == 'pdfIncre':
            prev_sim = np.log(pdf(cur_state, cur_state.shape[0]//6))
            cur_sim = np.log(pdf(new_state, new_state.shape[0]//6))
            similarity_reward = cur_sim - prev_sim
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'pdfIncreDecay':
            prev_sim = np.log(pdf(cur_state, cur_state.shape[0]//6))
            cur_sim = np.log(pdf(new_state, new_state.shape[0]//6))
            similarity_reward = cur_sim - prev_sim
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'pdfIncreFull':
            prev_sim = np.log(pdf(cur_state, cur_state.shape[0]//6))
            cur_sim = np.log(pdf(new_state, new_state.shape[0]//6))
            init_state = info['init_state']
            init_sim = np.log(pdf(init_state, self.num_objs))
            similarity_reward = cur_sim - prev_sim + (200 + init_sim)/info['max_episode_len']
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        # elif self.reward_mode == 'density':
        #     similarity_reward = np.zeros((self.num_objs*3))
        #     if info['is_done']:
        #         _, nll = ode_likelihood(cur_state, score_target, marginal_prob_std_fn, diffusion_coeff_fn)
        #         similarity_reward = -nll.cpu().repeat(self.num_objs*3).numpy()  # [centralized_density] * num_balls
        elif self.reward_mode == 'densityIncre':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            delta_state = new_state - cur_state
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'densityIncreImg':
            # state --actually--> obs
            cur_state = 2*cur_state/255. - 1
            new_state = 2*new_state/255. - 1
            cur_state = np.transpose(cur_state, (2, 0, 1)) # [c, h, w]
            new_state = np.transpose(new_state, (2, 0, 1))
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            delta_state = (new_state - cur_state).reshape(-1)
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'densityCumulative':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            delta_state = new_state - cur_state
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
            if self.prev_sim_reward is None:
                self.prev_sim_reward = similarity_reward
            else:
                similarity_reward += self.prev_sim_reward
        elif self.reward_mode == 'densityIncreDecay':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            delta_state = new_state - cur_state
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= 1 - info['progress'] # 所以越晚越拿不到sim reward
            similarity_reward *= (info['cur_steps'] % self.reward_freq == 0)
        else:
            print('Unknown Reward Type!!')
            raise NotImplementedError
        return similarity_reward


def eval_metric(eval_env, env_name, policy, n_box, reward_func, writer, eval_episodes=100):
    horizon = eval_env.max_episode_len
    episode_delta_likelihoods = []
    episode_avg_collisions = []
    last_states = []
    avg_reward = 0.
    avg_reward_similarity = 0
    avg_reward_collision = 0
    pdf = PDF_DICT[env_name]
    for idx in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        avg_collision = 0
        delta_likelihoods = -np.log(pdf(state, n_box))
        while not done:
            action = policy.select_action(np.array(state), sample=False)
            # action = env.sample_action()
            state, _, done, infos = eval_env.step(action, centralized=False)
            collisions = infos['collision_num']
            avg_collision += np.sum(collisions)
            # reward, reward_similarity, reward_collision = reward_func.get_reward(actions=action, info=infos, cur_state=state, new_state=next_state, is_eval=True)
            # avg_reward_similarity += reward_similarity.sum()
            # avg_reward_collision += reward_collision.sum()
            # avg_reward += reward.sum()
        last_states.append(state)
        avg_collision /= horizon
        delta_likelihoods += np.log(pdf(state, n_box))
        episode_delta_likelihoods.append(delta_likelihoods)
        episode_avg_collisions.append(avg_collision)
        # if (idx+1) % 10 == 0:
        #     print('----Delta Likelihood: {:.2f} +- {:.2f}'.format(np.mean(episode_delta_likelihoods), np.std(episode_delta_likelihoods)))
        #     print('----Avg Collisions: {:.2f} +- {:.2f}'.format(np.mean(episode_avg_collisions), np.std(episode_avg_collisions)))
        #     print('----Diversity Score: {:.2f}'.format(diversity_score(last_states)))
    # avg_reward, avg_reward_collision, avg_reward_similarity = \
    #     avg_reward/eval_episodes, avg_reward_collision/eval_episodes, avg_reward_similarity/eval_episodes
    # writer.add_scalars('Eval/Compare',
    #                    {'total': avg_reward,
    #                     'collision': avg_reward_collision,
    #                     'similarity': avg_reward_similarity*n_box*3},
    #                    episode_num + 1)
    # writer.add_scalar('Eval/Total',avg_reward, episode_num + 1)

    # Declaration
    mu_dl, std_dl = np.mean(episode_delta_likelihoods), np.std(episode_delta_likelihoods)
    mu_ac, std_ac = np.mean(episode_avg_collisions), np.std(episode_avg_collisions)
    ds = diversity_score(last_states)
    print('----Delta Likelihood: {:.2f} +- {:.2f}'.format(mu_dl, std_dl))
    print('----Avg Collisions: {:.2f} +- {:.2f}'.format(mu_ac, std_ac))
    print('----Diversity Score: {:.2f}'.format(ds))
    return {'DL': {'mu': mu_dl, 'std': std_dl}, 'AC': {'mu': mu_ac, 'std': std_ac}, 'DS': ds, 'Results': last_states}


def load_target_score(score_exp, network_mode, num_objs, max_action, is_state=True):
    diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = f'../logs/{score_exp}/score.pt'
    # marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25)
    # if is_state:
    #     score_target = ScoreModelGNN(marginal_prob_std_fn, n_box=num_objs, mode=network_mode, device=device, hidden_dim=64, embed_dim=32)
    # else:
    #     score_target = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    # score_target.load_state_dict(torch.load(tar_path))
    with open(tar_path, 'rb') as f:
        score_target = pickle.load(f)
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state), diffusion_coeff_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="BallSAC")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--exp_name", type=str, default="debug")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--env", type=str, default="sorting")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--reward_mode", type=str, default="full")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_residual", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--normalize_reward", type=str, default="True")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--score_exp", type=str, default="")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--action_type", type=str, default="vel")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--inp_mode", type=str, default="state")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--obs_size", type=int, default=64)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--is_ema", type=str, default="False")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--mar", type=float, default=0.9)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--queue_len", type=int, default=10)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--eval_num", type=int, default=40)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--hidden_dim", type=int, default=128)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--embed_dim", type=int, default=64)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--reward_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--residual_t0", default=0.01, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--eval_col", type=int, default=6)  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--model_type", type=str, default="tanh")  # Policy name (MATD3, DDPG or OurDDPG)
    parser.add_argument("--lambda_sim", default=1.0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--horizon", default=250, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--reward_freq", default=1, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--lambda_col", default=1, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--n_boxes", default=10, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--knn_actor", default=10, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--knn_critic", default=15, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10, type=int)  # How often (episodes!) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_freq", default=1, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--load_exp", type=str, default="None")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    exp_path = f"../logs/{args.exp_name}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    best_ckpt_path = exp_path + 'best'
    last_ckpt_path = exp_path # to be consistent with prev experiments

    tb_path = f"../logs/{args.exp_name}/tb"
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

    num_per_class = args.n_boxes

    inp_mode = args.inp_mode
    is_state = (inp_mode == 'state')
    if not is_state:
        assert inp_mode == 'image'
    obs_size = args.obs_size

    ''' init my env '''
    exp_data = None  # load expert examples for LfD algorithms
    PB_FREQ = 4
    action_type = args.action_type
    dt_dict = {
        'vel': 0.02, 
        'force': 0.05,
    }
    max_vel_dict = {
        'vel': 0.3,
        'force': 100.0,
    }
    MAX_VEL = max_vel_dict[action_type]
    dt = dt_dict[action_type]
    time_freq = int(1 / dt)
    env_kwargs = {
        'n_boxes': args.n_boxes,
        'exp_data': exp_data,
        'time_freq': time_freq * PB_FREQ,
        'is_gui': False,
        'max_action': MAX_VEL,
        'max_episode_len': args.horizon,
        'action_type': action_type,
    }
    env_class = ENV_DICT[args.env]
    env = env_class(**env_kwargs)
    env.seed(args.seed)
    env.reset(is_random=True)

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Load Target Score '''
    network_mode = 'target' if args.env in ['sorting', 'hybrid', 'sorting6'] else 'support'
    target_score, diffusion_coeff_fn = load_target_score(args.score_exp, network_mode, num_per_class, MAX_VEL, is_state=is_state)
    target_score_state, _ = load_target_score(args.score_exp, network_mode, num_per_class, MAX_VEL, is_state=True)

    ''' Init policy '''
    kwargs = {
        "num_boxes": args.n_boxes,
        "max_action": MAX_VEL,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "writer": writer,
        "knn_actor": args.knn_actor,
        "knn_critic": args.knn_critic,
        "target_score": target_score_state,
        "is_residual": args.is_residual == 'True',
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "residual_t0": args.residual_t0,
    }
    policy = BallSAC.MASAC(**kwargs)
    if not args.load_exp == 'None':
        policy.load(f"../logs/{args.load_exp}/")

    ''' Init Buffer '''
    state_dim = 3*2*args.n_boxes
    action_dim = 3*2*args.n_boxes
    replay_buffer = BallSAC.ReplayBuffer(state_dim, action_dim,
                                       n_nodes=args.n_boxes*3,
                                       centralized=False,
                                       is_ema=args.is_ema=='True', mar=args.mar, queue_len=args.queue_len)

    reward_normalizer_sim = RewardNormalizer(args.normalize_reward == 'True', writer, name='sim')
    reward_normalizer_col = RewardNormalizer(args.normalize_reward == 'True', writer, name='col')
    reward_func = RewardSampler(
        num_per_class,
        reward_normalizer_sim,
        reward_normalizer_col,
        env_type=args.env,
        target_score=target_score,
        reward_mode=args.reward_mode,
        reward_freq=args.reward_freq,
        is_centralized=False,
        lambda_sim=args.lambda_sim,
        lambda_col=args.lambda_col,
        t0=args.reward_t0,
    )

    state, done = env.reset(is_random=True), False
    reward_func.reset_reward()
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_delta_likelihood = 0
    cur_obs = env.render(obs_size)

    for t in trange(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            action = policy.select_action(np.array(state), sample=True)

        # Perform action
        next_state, _, done, infos = env.step(action, centralized=False)
        new_obs = env.render(obs_size)
        if is_state:
            reward, reward_similarity, reward_collision =  reward_func.get_reward(actions=action, info=infos, cur_state=state, new_state=next_state)
        else:
            reward, reward_similarity, reward_collision =  reward_func.get_reward(actions=action, info=infos, cur_state=cur_obs, new_state=new_obs)

        done_bool = float(done) if episode_timesteps < env.max_episode_len else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward.sum().item()
        episode_similarity_reward += reward_similarity.sum().item()
        episode_collision_reward += reward_collision.sum().item()

        cur_obs = new_obs

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                f"Total: {episode_reward:.3f} "
                f"Collision: {episode_collision_reward:.3f} "
                f"Similarity: {episode_similarity_reward*3*args.n_boxes:.3f}")
            writer.add_scalars('Episode_rewards/Compare',
                               {'total': episode_reward,
                                'collision': episode_collision_reward,
                                'similarity': episode_similarity_reward*num_per_class*3},
                               episode_num + 1)
            writer.add_scalar('Episode_rewards/Total Reward', episode_reward, episode_num + 1)

            # Reset environment
            state, done = env.reset(is_random=True), False
            cur_obs = env.render(obs_size)
            reward_func.reset_reward()
            episode_reward = 0
            episode_collision_reward = 0
            episode_similarity_reward = 0
            episode_timesteps = 0
            episode_num += 1
            replay_buffer.clear_queue()

            # Evaluate episode, save model before eval
            if episode_num % args.eval_freq == 0:
                print('------Now Save Models!------')
                # with open(f"{exp_path}policy.pickle", 'wb') as f:
                #     pickle.dump(policy, f)
                policy.save(f"{last_ckpt_path}")
                print('------Now Start Eval!------')
                file_name = f"{args.policy}_{args.env}_{args.seed}"
                eval_num = args.eval_num
                eval_col = args.eval_col
                assert eval_col**2 <= eval_num
                infos = eval_metric(env, args.env, policy, num_per_class, reward_func, writer, eval_episodes=eval_num)
                writer.add_scalars('Eval/Delta_Likelihood', {'upper': infos['DL']['mu']+infos['DL']['std'],
                                                             'mean': infos['DL']['mu'],
                                                             'lower': infos['DL']['mu']-infos['DL']['std']}, episode_num)
                writer.add_scalars('Eval/Average_Collision', {'upper': infos['AC']['mu']+infos['AC']['std'],
                                                             'mean': infos['AC']['mu'],
                                                             'lower': infos['AC']['mu']-infos['AC']['std']}, episode_num)
                writer.add_scalar('Eval/Diversity_Score', infos['DS'], episode_num)
                # save best ckpt according to eval
                if infos['DL']['mu'] > best_delta_likelihood:
                    best_delta_likelihood = infos['DL']['mu']
                    policy.save(f"{best_ckpt_path}")

                visualize_states(np.stack(infos['Results'][0:eval_col**2]), env, writer, episode_num, eval_col, suffix='episode last state')
                # eval_policy(policy, args.seed, writer, episode_num, reward_func=reward_func, reward_mode=args.reward_mode)
                print(f'EXP_NAME: {args.exp_name}')

            # as train and eval share the same env
            state, done = env.reset(is_random=True), False


class GTTargetScore(nn.Module):
    def __init__(self, num_objs, env_type='sorting'):
        super(GTTargetScore, self).__init__()
        self.scale = 0.05
        self.bound = 0.3
        self.r = 0.025
        self.num_objs = num_objs
        self.env_type = env_type
        self.likelihood_funcs = {
            'sorting': self.likelihood_sorting,
            'placing': self.likelihood_placing,
            'hybrid': self.likelihood_hybrid,
        }
        ''' For Sorting '''
        self.centers = []
        bases = np.array(list(range(3))) * (2 * np.pi) / 3
        coins = [0, 1]
        delta = [-2 * np.pi / 3, 2 * np.pi / 3]
        radius = 0.18
        for base in bases:
            for coin in coins:
                theta_red = base
                theta_green = theta_red + delta[coin]
                theta_blue = theta_red + delta[1 - coin]
                red_center = radius * np.array([np.cos(theta_red), np.sin(theta_red)])
                green_center = radius * np.array([np.cos(theta_green), np.sin(theta_green)])
                blue_center = radius * np.array([np.cos(theta_blue), np.sin(theta_blue)])
                cur_center = np.array([red_center]*num_objs + [green_center]*num_objs + [blue_center]*num_objs)
                self.centers.append(torch.tensor(cur_center).view(-1).to(device))

    def pdf_sorting(self, data, center):
        # data: [bs, dim], center:[num_objs*3, 2]
        # set_trace()
        bs = data.shape[0]
        z = (data - center.repeat(bs, 1))/self.scale
        densities = torch.exp(-z**2/2)/torch.sqrt(torch.tensor(2*np.pi, device=device))
        # 一定要先log 再加，不能先加再log！！ 精度限制！！
        # log_densities = torch.log(densities)
        return torch.prod(densities, dim=-1)

    def likelihood_sorting(self, inp_states):
        # # inp_states: [bs, dim]
        # states_ts = torch.tensor(inp_states).to(device)
        # states_ts *= self.bound - self.r
        # total_likelihood = 0
        # for center in self.centers:
        #     total_likelihood += self.pdf_sorting(states_ts, center)
        # total_likelihood /= len(self.centers)
        # return np.log(total_likelihood.cpu().numpy())
        likelihoods = []
        for inp_state in inp_states:
            likelihoods.append(np.log(pdf_sorting(inp_state, self.num_objs)+1e-300)) # 放爆nan
        return np.array(likelihoods)

    @staticmethod
    def likelihood_placing(inp_states):
        # bs = np.prod(inp_states.shape)//(self.num_objs*3*2)
        # positions = inp_states.reshape(bs, -1, 2)
        # radius_std = np.std(np.sqrt(np.sum(positions**2, axis=-1)), axis=-1)
        # theta_std = np.std(np.arctan2(positions[:, :, 1], positions[:, :, 0]), axis=-1) # theta = atan2(y, x)
        # return -theta_std*radius_std
        likelihoods = []
        for inp_state in inp_states:
            likelihoods.append(np.log(pdf_placing(inp_state)))
        return np.array(likelihoods)
        # return np.log(pdf_sorting(np.stack(inp_states), self.num_objs))

    @staticmethod
    def likelihood_hybrid(inp_states):
        # bs = np.prod(inp_states.shape)//(self.num_objs*3*2)
        # positions = inp_states.reshape(bs, -1, 2)
        # radius_std = np.std(np.sqrt(np.sum(positions**2, axis=-1)), axis=-1)
        # theta_std = np.std(np.arctan2(positions[:, :, 1], positions[:, :, 0]), axis=-1) # theta = atan2(y, x)
        # return -theta_std*radius_std
        likelihoods = []
        for inp_state in inp_states:
            likelihoods.append(np.log(pdf_hybrid(inp_state)))
        return np.array(likelihoods)

    def total_likelihood(self, inp_states):
        likelihood_func = self.likelihood_funcs[self.env_type]
        return likelihood_func(inp_states)

    def forward(self, inp_data, _, __):
        # inp_data:
        source_ts = inp_data.x.detach().clone().cpu().numpy()
        inp_var = torch.tensor(source_ts, requires_grad=True, device=device)
        h = inp_var * (self.bound - self.r)
        h = h.view(-1, self.num_objs*3, 2)
        # inp_var: [bs, n, 2]
        # centers: [bs, n, 2] * 6
        total_likelihood = 0
        for center in self.centers:
            total_likelihood += self.calc_likelihood(h, center)
        total_likelihood /= len(self.centers)
        total_likelihood.backward()
        score = inp_var.grad
        return score.detach()