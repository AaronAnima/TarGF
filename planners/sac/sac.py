import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from ipdb import set_trace

from networks.actor_critics import BallCritic, RoomCritic
from utils.preprocesses import prepro_state, prepro_graph_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITIC_DICT = {'Ball': BallCritic, 'Room': RoomCritic}


class MASAC(object):
    def __init__(
            self,
            max_action,
            actor,
            writer,
            targf,
            configs,
            timer=None,
            learnable_temperature=True,
            init_temperature=0.1,
    ):
        Critic = CRITIC_DICT[configs.env_type]
        self.actor = actor
        # for param in self.actor.parameters():
        #     print(param)
        # set_trace()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, betas=(0.9, 0.999))
        self.critic = Critic(
            configs,
            targf=targf,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, betas=(0.9, 0.999))

        # alpha
        self.learnable_temperature = learnable_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4, betas=(0.9, 0.999))

        self.max_action = max_action
        self.discount = configs.discount
        self.tau = configs.tau
        self.policy_freq = configs.policy_freq
        self.critic_target_update_frequency = 2

        self.total_it = 0
        self.writer = writer
        self.timer = timer
        self.env_type = configs.env_type
        self.configs = configs

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        # state: ([bs, 1], Batch), action: [num_nodes, 3], reward: [num_nodes, 1], not_done: [bs, 1]
        self.timer.set()
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        if self.env_type == 'Room':
            not_done = not_done[state[-1].batch] # expand tensor
        self.timer.log('train_sample_batch')
        

        self.timer.set()
        ''' update critic '''
        next_dist = self.actor(next_state)
        # Select action according to policy and add clipped noise
        next_action = next_dist.rsample()
        log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action.detach())
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob.detach()
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.writer.add_scalar('Losses/critic_loss', critic_loss, self.total_it)
        self.timer.log('train_inference_critic')

        self.timer.set()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.timer.log('train_update_critic')

        ''' update actor '''
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            self.timer.set()
            dist = self.actor(state)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.critic(state, action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
            self.writer.add_scalar('Losses/actor_loss', actor_loss, self.total_it)
            self.timer.log('train_inference_actor')

            self.timer.set()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.timer.log('train_update_actor')

            if self.learnable_temperature:
                self.timer.set()
                self.log_alpha_optimizer.zero_grad()
                if self.env_type == 'Room':
                    target_entropy = -torch.tensor([(state[1].ptr[i+1] - state[1].ptr[i])*3 for i in range(batch_size)]).to(device)
                elif self.env_type == 'Ball':
                    target_entropy = - 2 * self.configs.num_objs
                else:
                    raise ValueError(f"Mode {self.env_type} not recognized.")

                # log_prob: [bs, 1], target_entropy: [bs, 1]
                alpha_loss = (self.alpha * (-log_prob - target_entropy).detach()).mean()
                self.writer.add_scalar('Losses/alpha_loss', alpha_loss, self.total_it)
                self.writer.add_scalar('Losses/alpha', self.alpha, self.total_it)
                self.timer.log('train_inference_alpha')

                self.timer.set()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                self.timer.log('train_update_alpha')
        if self.total_it % self.critic_target_update_frequency == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.log_alpha, filename + "_log_alpha")
        torch.save(self.log_alpha_optimizer.state_dict(), filename + "_log_alpha_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha.dara = torch.load(filename + "_log_alpha").data
        self.log_alpha_optimizer.load_state_dict(torch.load(filename + "_log_alpha_optimizer"))


class ReplayBuffer(object):
    def __init__(self, configs, timer=None):
        self.max_size = int(configs.buffer_size)
        self.env_type = configs.env_type
        self.timer = timer
        
        if self.env_type == 'Room':
            self.state = deque(maxlen=self.max_size)
            self.action = deque(maxlen=self.max_size)
            self.next_state = deque(maxlen=self.max_size)
            self.reward = deque(maxlen=self.max_size)
            self.not_done = deque(maxlen=self.max_size)
        elif self.env_type == 'Ball':
            state_dim = configs.num_objs * 3
            action_dim = configs.num_objs * 2
            self.state = np.zeros((self.max_size, state_dim))
            self.action = np.zeros((self.max_size, action_dim))
            self.next_state = np.zeros((self.max_size, state_dim))
            self.reward = np.zeros((self.max_size, configs.num_objs))
            self.not_done = np.zeros((self.max_size, 1))
        else:
            raise ValueError(f"Mode {self.env_type} not recognized.")
    
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def prepro_tensor_batch(rewards):
        reward_ts_cat = torch.cat([torch.FloatTensor(reward) for reward in rewards], dim=0).to(device)
        if len(reward_ts_cat.shape) == 1:
            reward_ts_cat = reward_ts_cat.view(-1, 1)
        return reward_ts_cat

    def add(self, state, action, next_state, reward, done):
        if self.env_type == 'Room':
            self.state.append(prepro_state(state))
            self.action.append(action)
            self.next_state.append(prepro_state(next_state))
            self.reward.append(reward)
            self.not_done.append([1. - done])
        elif self.env_type == 'Ball':
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.next_state[self.ptr] = next_state
            self.reward[self.ptr] = reward
            self.not_done[self.ptr] = 1. - done
        else:
            raise ValueError(f"Mode {self.env_type} not recognized.")
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        self.timer.set()
        ind = np.random.randint(0, self.size, size=batch_size)
        if self.env_type == 'Room':
            states = [self.state[i] for i in ind]
            actions = [self.action[i].reshape(-1, 3) for i in ind]
            next_states = [self.next_state[i] for i in ind]
            rewards = [self.reward[i] for i in ind]
            not_dones = [self.not_done[i] for i in ind]
            batch = (
                prepro_graph_batch(states),
                self.prepro_tensor_batch(actions),
                prepro_graph_batch(next_states),
                self.prepro_tensor_batch(rewards),
                torch.FloatTensor(not_dones).to(self.device)
            )
        elif self.env_type == 'Ball':
            batch = (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
        else:
            raise ValueError(f"Mode {self.env_type} not recognized.")
        self.timer.log('sample_batch_prepro_batch')

        return batch
