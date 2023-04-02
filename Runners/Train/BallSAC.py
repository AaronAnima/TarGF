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
from ipdb import set_trace

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ball_utils import  RewardNormalizer
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff
from Algorithms.BallSAC import MASAC, ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_target_score(score_exp, num_objs, max_action):
    diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = f'./logs/{score_exp}/score.pt'
    with open(tar_path, 'rb') as f:
        score_target = pickle.load(f)
    return TargetScore(score_target.to(device), num_objs, max_action), diffusion_coeff_func


def visualize_states(eval_states, env, logger, nrow, epoch, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    for obj_state in eval_states:
        if not isinstance(obj_state, np.ndarray):
            obj_state = obj_state.detach().cpu().numpy()
        obj_state = env.unflatten_states([obj_state])[0]
        env.set_state(obj_state)
        img = env.render(img_size=256)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


class TargetScore:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        """
        state_inp: [bs, 3*num_objs]
        """
        if not empty:
            # construct graph-input for score network
            if not torch.is_tensor(state_inp):
                state_inp = torch.tensor(state_inp)
            positions = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, :2]
            categories = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, -1:]
            bs = positions.shape[0]
            positions = positions.view(-1, 2).float()
            categories = categories.view(-1).long()
            samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
            edge_index = knn_graph(positions, k=self.num_objs-1, batch=samples_batch)
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            inp_data = Data(x=positions, edge_index=edge_index, c=categories)

            out_score = self.score(inp_data, t, self.num_objs)
            out_score = out_score.detach()
            # normalise the gradient
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
            num_objs,
            normalizer_sim,
            normalizer_col,
            pdf_func,
            target_score=None,
            reward_mode='cos',
            reward_freq=1,
            lambda_sim=1.0,
            lambda_col=1.0,
            t0=0.01,
    ):
        self.num_objs = num_objs
        self.normalizer_sim = normalizer_sim
        self.normalizer_col = normalizer_col
        self.target_score = target_score
        self.reward_mode = reward_mode
        self.reward_freq = reward_freq
        self.lambda_sim = lambda_sim
        self.lambda_col = lambda_col
        self.t0 = t0
        self.prev_sim_reward = None
        self.pdf_func = pdf_func
    
    def reset_reward(self):
        self.prev_sim_reward = None

    def get_reward(self, infos, cur_state, new_state, is_eval=False):
        collision_reward = self.get_collision_reward(infos)
        similarity_reward = self.get_similarity_reward(infos, cur_state, new_state)

        similarity_reward = self.normalizer_sim.update(similarity_reward, is_eval=is_eval)
        collision_reward = self.normalizer_col.update(collision_reward, is_eval=is_eval)
        
        total_reward = self.lambda_sim*similarity_reward + self.lambda_col*collision_reward

        return total_reward, similarity_reward, collision_reward

    def get_collision_reward(self, infos):
        collision_num = infos['collision_num']
        collisions = np.array([np.sum(collision_num[:, i]) + np.sum(collision_num[i, :]) for i in range(self.num_objs)])
        collision_reward = -np.array(collisions, 'float32')
        return collision_reward

    def get_similarity_reward(self, infos, cur_state, new_state):
        pdf = self.pdf_func
        if self.reward_mode == 'collision_only':
            return 0
        elif self.reward_mode == 'pdfIncre':
            prev_sim = np.log(pdf([cur_state]))
            cur_sim = np.log(pdf([new_state]))
            similarity_reward = cur_sim - prev_sim
            similarity_reward *= (infos['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'densityIncre':
            '''!! the reward mode used in paper !!'''
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            new_state = new_state.reshape((-1, 3))[:, :2].reshape(-1)
            cur_state = cur_state.reshape((-1, 3))[:, :2].reshape(-1)
            delta_state = new_state - cur_state
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (infos['cur_steps'] % self.reward_freq == 0)
        elif self.reward_mode == 'densityCumulative':
            cur_score = self.target_score.get_score(cur_state, t0=self.t0, is_norm=False).reshape(-1)
            new_state = new_state.reshape((-1, 3))[:, :2].reshape(-1)
            cur_state = cur_state.reshape((-1, 3))[:, :2].reshape(-1)
            delta_state = new_state - cur_state
            similarity_reward = np.sum(delta_state * cur_score)
            similarity_reward *= (infos['cur_steps'] % self.reward_freq == 0)
            if self.prev_sim_reward is None:
                self.prev_sim_reward = similarity_reward
            else:
                similarity_reward += self.prev_sim_reward
        else:
            print('Unknown Reward Type!!')
            raise NotImplementedError
        return similarity_reward


def eval_metric(eval_env, policy, pdf_func, eval_episodes=100):
    horizon = eval_env.max_episode_len
    episode_delta_likelihoods = []
    episode_avg_collisions = []
    last_states = []
    pdf = pdf_func
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        state = env.flatten_states([state])[0] # flatten state!
        avg_collision = 0
        delta_likelihoods = -np.log(pdf([state]))
        while not done:
            action = policy.select_action(np.array(state), sample=False)
            state, _, done, infos = eval_env.step(action)
            state = env.flatten_states([state])[0] # flatten state!
            collisions = infos['collision_num']
            avg_collision += np.sum(collisions)
        last_states.append(state)
        avg_collision /= horizon
        delta_likelihoods += np.log(pdf([state]))
        episode_delta_likelihoods.append(delta_likelihoods)
        episode_avg_collisions.append(avg_collision)

    # Declaration
    mu_dl, std_dl = np.mean(episode_delta_likelihoods), np.std(episode_delta_likelihoods)
    mu_ac, std_ac = np.mean(episode_avg_collisions), np.std(episode_avg_collisions)
    print('----Delta Likelihood: {:.2f} +- {:.2f}'.format(mu_dl, std_dl))
    print('----Avg Collisions: {:.2f} +- {:.2f}'.format(mu_ac, std_ac))
    return {'DL': {'mu': mu_dl, 'std': std_dl}, 'AC': {'mu': mu_ac, 'std': std_ac}, 'Results': last_states}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file
    parser.add_argument("--policy", default="BallSAC") 
    parser.add_argument("--log_dir", type=str) 
    parser.add_argument("--score_exp", type=str) 
    # env
    parser.add_argument("--pattern", type=str, default="CircleCluster") 
    parser.add_argument("--num_per_class", default=7, type=int)  
    parser.add_argument("--num_classes", default=3, type=int)  
    parser.add_argument("--action_type", type=str, default="vel") 
    parser.add_argument("--seed", default=0, type=int)  
    # model
    parser.add_argument("--knn_actor", default=20, type=int)  
    parser.add_argument("--knn_critic", default=20, type=int)  
    parser.add_argument("--hidden_dim", type=int, default=128) 
    parser.add_argument("--embed_dim", type=int, default=64) 
    parser.add_argument("--is_residual", type=str, default="True") 
    parser.add_argument("--reward_t0", default=0.01, type=float)  
    parser.add_argument("--residual_t0", default=0.01, type=float) 
    # train
    parser.add_argument("--discount", default=0.95, type=float)  
    parser.add_argument("--start_timesteps", default=25e2, type=int) 
    parser.add_argument("--max_timesteps", default=5e5, type=int)  
    parser.add_argument("--batch_size", default=256, type=int)  
    parser.add_argument("--tau", default=0.005, type=float) 
    parser.add_argument("--policy_freq", default=1, type=int) 
    # eval
    parser.add_argument("--eval_num", type=int, default=25)  
    parser.add_argument("--eval_col", type=int, default=5) 
    parser.add_argument("--eval_freq", default=100, type=int) 
    # reward
    parser.add_argument("--reward_mode", type=str, default="densityIncre") 
    parser.add_argument("--normalize_reward", type=str, default="True") 
    parser.add_argument("--lambda_col", default=1, type=float) 
    parser.add_argument("--lambda_sim", default=1.0, type=float)   
    parser.add_argument("--reward_freq", default=1, type=int) 

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

    num_per_class = args.num_per_class
    num_classes = args.num_classes
    num_objs = num_per_class*num_classes

    ''' init my env '''
    env_name = '{}-{}Ball{}Class-v0'.format(args.pattern, num_objs, num_classes)
    env = gym.make(env_name)
    env.seed(args.seed)
    env.reset()
    max_action = env.action_space['obj1']['linear_vel'].high[0]

    # get pseudo likelihood function
    pdf_func = env.pseudo_likelihoods


    ''' set seeds '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Load Target Score '''
    target_score, diffusion_coeff_fn = load_target_score(args.score_exp, num_objs, max_action)

    ''' Init policy '''
    kwargs = {
        "num_objs": num_objs,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "writer": writer,
        "knn_actor": args.knn_actor,
        "knn_critic": args.knn_critic,
        "target_score": target_score,
        "is_residual": args.is_residual == 'True',
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "residual_t0": args.residual_t0,
    }
    policy = MASAC(**kwargs)

    ''' Init Buffer '''
    state_dim = 3*num_objs
    action_dim = 2*num_objs
    replay_buffer = ReplayBuffer(state_dim, action_dim, num_objs=num_objs, centralized=False)

    reward_normalizer_sim = RewardNormalizer(args.normalize_reward == 'True', writer, name='sim')
    reward_normalizer_col = RewardNormalizer(args.normalize_reward == 'True', writer, name='col')
    reward_func = RewardSampler(
        num_objs,
        reward_normalizer_sim,
        reward_normalizer_col,
        pdf_func=pdf_func,
        target_score=target_score,
        reward_mode=args.reward_mode,
        reward_freq=args.reward_freq,
        lambda_sim=args.lambda_sim,
        lambda_col=args.lambda_col,
        t0=args.reward_t0,
    )

    ''' Start SAC Training! '''
    state, done = env.reset(is_random=True), False
    state = env.flatten_states([state])[0] # flatten state!

    reward_func.reset_reward()
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_delta_likelihood = 0

    for t in trange(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample_action()
            action = env.flatten_actions([action])[0] # flatten action !!
        else:
            action = policy.select_action(np.array(state), sample=True)

        # Perform action
        next_state, _, done, infos = env.step(action)
        next_state = env.flatten_states([next_state])[0] # flatten state!

        reward, reward_similarity, reward_collision =  reward_func.get_reward(infos=infos, cur_state=state, new_state=next_state)

        done_bool = float(done) if episode_timesteps < env.max_episode_len else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state # update state !!
        episode_reward += reward.sum().item()
        episode_similarity_reward += reward_similarity.sum().item()
        episode_collision_reward += reward_collision.sum().item()

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                f"Total: {episode_reward:.3f} "
                f"Collision: {episode_collision_reward:.3f} "
                f"Similarity: {episode_similarity_reward*num_objs:.3f}")
            writer.add_scalars('Episode_rewards/Compare',
                               {'total': episode_reward,
                                'collision': episode_collision_reward,
                                'similarity': episode_similarity_reward*num_per_class*3},
                               episode_num + 1)
            writer.add_scalar('Episode_rewards/Total Reward', episode_reward, episode_num + 1)

            # Reset environment
            state, done = env.reset(is_random=True), False
            state = env.flatten_states([state])[0] # flatten state!
            
            reward_func.reset_reward()
            episode_reward = 0
            episode_collision_reward = 0
            episode_similarity_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Evaluate episode, save model before eval
            if episode_num % args.eval_freq == 0:
                print('------Now Save Models!------')
                with open(f"{exp_path}policy.pickle", 'wb') as f:
                    policy.writer = None
                    pickle.dump(policy, f)
                    policy.writer = writer
    

                print('------Now Start Eval!------')
                eval_num = args.eval_num
                eval_col = args.eval_col
                assert eval_col**2 <= eval_num
                eval_infos = eval_metric(env, policy, pdf_func, eval_episodes=eval_num)
                writer.add_scalars('Eval/Delta_Likelihood', {'upper': eval_infos['DL']['mu']+eval_infos['DL']['std'],
                                                             'mean': eval_infos['DL']['mu'],
                                                             'lower': eval_infos['DL']['mu']-eval_infos['DL']['std']}, episode_num)
                writer.add_scalars('Eval/Average_Collision', {'upper': eval_infos['AC']['mu']+eval_infos['AC']['std'],
                                                             'mean': eval_infos['AC']['mu'],
                                                             'lower': eval_infos['AC']['mu']-eval_infos['AC']['std']}, episode_num)
                # save best ckpt according to eval
                if eval_infos['DL']['mu'] > best_delta_likelihood:
                    best_delta_likelihood = eval_infos['DL']['mu']
                    policy.save(f"{best_ckpt_path}")
                visualize_states(np.stack(eval_infos['Results'][0:eval_col**2]), env, writer, epoch=episode_num, nrow=eval_col, suffix='episode last state')
                print(f'log_dir: {args.log_dir}')

            # reset again, as train and eval share the same env
            state, done = env.reset(is_random=True), False
            state = env.flatten_states([state])[0] # flatten state!

