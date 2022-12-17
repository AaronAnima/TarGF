import numpy as np
import torch
import argparse
import os
import gym
import ebor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import sys
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import functools
from tqdm import trange
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ball_utils import pdf_sorting, pdf_placing, pdf_hybrid, diversity_score, RewardNormalizer
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, pc_sampler_state, ode_sampler, Euler_Maruyama_sampler
from Algorithms.BallSAC import MASAC, ReplayBuffer
from Runners.Eval.BallEvalBase import get_simulation_constants
PDF_DICT = {'Clustering-v0': pdf_sorting, 'Circling-v0': pdf_placing, 'CirclingClustering-v0': pdf_hybrid}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_states(eval_states, env, logger, epoch, nrow, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    render_size = 256
    for box_state in eval_states:
        env.set_state(box_state)
        img = env.render(img_size=render_size)
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
                if not torch.is_tensor(state_inp):
                    state_inp = torch.tensor(state_inp)
                state_inp = state_inp.view(-1, self.num_objs * 3, 2).to(device)
                bs = state_inp.shape[0]
                samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs*3)], dtype=torch.int64).to(device)
                edge_index = knn_graph(state_inp.view(-1, 2), k=self.num_objs*3-1, batch=samples_batch)
                t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
                inp_data = Data(x=state_inp.view(-1, 2), edge_index=edge_index)
                out_score = self.score(inp_data, t, self.num_objs) 
                out_score = out_score.detach()
                if is_norm:
                    out_score = out_score * torch.min(
                        torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
                else:
                    out_score = out_score
            else:
                state_ts = torch.FloatTensor(state_inp).to(device).unsqueeze(0)
                assert -1 <= torch.min(state_ts) <= torch.max(state_ts) <= 1
                t = torch.FloatTensor([t0]).to(device)
                out_score = self.score(state_ts, t)
                out_score = out_score.detach()
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score


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
            is_centralized=False,
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

        similarity_reward = self.normalizer_sim.update(similarity_reward, is_eval=is_eval)
        collision_reward = self.normalizer_col.update(collision_reward, is_eval=is_eval)
        
        total_reward = self.lambda_sim*similarity_reward + self.lambda_col*collision_reward

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
            similarity_reward *= 1 - info['progress'] 
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
    pdf = PDF_DICT[env_name]
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        avg_collision = 0
        delta_likelihoods = -np.log(pdf(state, n_box))
        while not done:
            action = policy.select_action(np.array(state), sample=False)
            state, _, done, infos = eval_env.step(action)
            collisions = infos['collision_num']
            avg_collision += np.sum(collisions)
        last_states.append(state)
        avg_collision /= horizon
        delta_likelihoods += np.log(pdf(state, n_box))
        episode_delta_likelihoods.append(delta_likelihoods)
        episode_avg_collisions.append(avg_collision)

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
    tar_path = f'./logs/{score_exp}/score.pt'
    with open(tar_path, 'rb') as f:
        score_target = pickle.load(f)
    return TargetScore(score_target.to(device), num_objs, max_action, is_state=is_state), diffusion_coeff_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="BallSAC") 
    parser.add_argument("--log_dir", type=str) 
    parser.add_argument("--env", type=str, default="sorting") 
    parser.add_argument("--reward_mode", type=str, default="densityIncre") 
    parser.add_argument("--is_residual", type=str, default="True") 
    parser.add_argument("--normalize_reward", type=str, default="True") 
    parser.add_argument("--score_exp", type=str) 
    parser.add_argument("--action_type", type=str, default="vel") 
    parser.add_argument("--inp_mode", type=str, default="state") 
    parser.add_argument("--obs_size", type=int, default=64) 
    parser.add_argument("--is_ema", type=str, default="False") 
    parser.add_argument("--mar", type=float, default=0.9) 
    parser.add_argument("--queue_len", type=int, default=10) 
    parser.add_argument("--eval_num", type=int, default=25)  
    parser.add_argument("--eval_col", type=int, default=5) 
    parser.add_argument("--hidden_dim", type=int, default=128) 
    parser.add_argument("--embed_dim", type=int, default=64) 
    parser.add_argument("--reward_t0", default=0.01, type=float)  
    parser.add_argument("--residual_t0", default=0.01, type=float) 
    parser.add_argument("--model_type", type=str, default="tanh") 
    parser.add_argument("--lambda_sim", default=1.0, type=float)  
    parser.add_argument("--horizon", default=100, type=int)  
    parser.add_argument("--reward_freq", default=1, type=int)  
    parser.add_argument("--lambda_col", default=1, type=float)  
    parser.add_argument("--n_boxes", default=7, type=int)  
    parser.add_argument("--knn_actor", default=20, type=int)  
    parser.add_argument("--knn_critic", default=20, type=int)  
    parser.add_argument("--seed", default=0, type=int)  
    parser.add_argument("--start_timesteps", default=25e2, type=int)  
    parser.add_argument("--eval_freq", default=100, type=int) 
    parser.add_argument("--max_timesteps", default=5e5, type=int)  
    parser.add_argument("--batch_size", default=256, type=int)  
    parser.add_argument("--discount", default=0.95, type=float)  
    parser.add_argument("--tau", default=0.005, type=float) 
    parser.add_argument("--policy_freq", default=1, type=int) 
    parser.add_argument("--load_exp", type=str, default="None")  
    args = parser.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    exp_path = f"./logs/{args.log_dir}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    best_ckpt_path = exp_path + 'best'
    last_ckpt_path = exp_path # to be consistent with prev experiments

    tb_path = f"./logs/{args.log_dir}/tb"
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
    MAX_VEL, dt, PB_FREQ, RADIUS, _ = get_simulation_constants()
    env = gym.make(args.env, n_boxes=args.n_boxes)
    env.seed(args.seed)
    env.reset()

    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Load Target Score '''
    network_mode = 'target' if 'Clustering' in args.env else 'support'
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
    policy = MASAC(**kwargs)
    if not args.load_exp == 'None':
        policy.load(f"./logs/{args.load_exp}/")

    ''' Init Buffer '''
    state_dim = 3*2*args.n_boxes
    action_dim = 3*2*args.n_boxes
    replay_buffer = ReplayBuffer(state_dim, action_dim,
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
    cur_obs = env.render(img_size=obs_size)

    for t in trange(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            action = policy.select_action(np.array(state), sample=True)

        # Perform action
        next_state, _, done, infos = env.step(action)
        new_obs = env.render(img_size=obs_size)

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
            cur_obs = env.render(img_size=obs_size)
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
                with open(f"{exp_path}policy.pickle", 'wb') as f:
                    policy.writer = None
                    pickle.dump(policy, f)
                    policy.writer = writer
    

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
                print(f'log_dir: {args.log_dir}')

            # as train and eval share the same env
            state, done = env.reset(is_random=True), False

