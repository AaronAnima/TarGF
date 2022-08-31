import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace
from collections import deque
import os
import sys

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, EdgeConv
from torch_scatter import scatter_max, scatter_mean
from torch import distributions as pyd

from Algorithms.RoomSACNet import Actor, Critic


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from room_utils import prepro_dynamic_graph, prepro_state, pre_pro_dynamic_vec, prepro_graph_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MASAC(object):
    def __init__(
            self,
            max_action,
            writer,
            target_score,
            discount=0.99,
            tau=0.005,
            policy_freq=1,
            hidden_dim=128,
            embed_dim=64,
            residual_t0=0.01,
            init_temperature=0.1,
            is_residual=True,
            learnable_temperature=True,
            timer=None,
    ):
        self.actor = Actor(
            log_std_bounds=(-5, 2),
            max_action=max_action,
            target_score=target_score,
            is_residual=is_residual,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            t0=residual_t0,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, betas=(0.9, 0.999))

        self.critic = Critic(
            target_score=target_score,
            is_residual=is_residual,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            t0=residual_t0,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, betas=(0.9, 0.999))

        # alpha
        self.learnable_temperature = learnable_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4, betas=(0.9, 0.999))

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.critic_target_update_frequency = 2

        self.total_it = 0
        self.writer = writer
        self.timer = timer

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action(self, state, sample=False):
        if not torch.is_tensor(state[0]):
            state_inp = prepro_state(state, cuda=True)
            obj_batch = Batch.from_data_list([state_inp[1]])
            wall_batch = state_inp[0].view(-1, 1)
            state_inp = (wall_batch, obj_batch)
        else:
            state_inp = state
        dist = self.actor(state_inp)
        action = dist.sample() if sample else dist.mean
        return action.clamp(-self.max_action, self.max_action).detach().cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        # state: ([bs, 1], Batch), action: [num_nodes, 3], reward: [num_nodes, 1], not_done: [bs, 1]
        self.timer.set()
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        not_done = not_done[state[-1].batch] # [bs, 1] -> [num_nodes, 1]
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
                target_entropy = -torch.tensor([(state[1].ptr[i+1] - state[1].ptr[i])*3 for i in range(batch_size)]).to(device)
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
    def __init__(self, max_size=int(1e6), timer=None):
        self.max_size = int(max_size)
        
        self.state = deque(maxlen=self.max_size)
        self.action = deque(maxlen=self.max_size)
        self.next_state = deque(maxlen=self.max_size)
        self.reward = deque(maxlen=self.max_size)
        self.not_done = deque(maxlen=self.max_size)
        self.timer = timer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def prepro_tensor_batch(rewards):
        reward_ts_cat = torch.cat([torch.FloatTensor(reward) for reward in rewards], dim=0).to(device)
        if len(reward_ts_cat.shape) == 1:
            reward_ts_cat = reward_ts_cat.view(-1, 1)
        return reward_ts_cat

    @property
    def size(self):
        return len(self.state)


    def add(self, state, action, next_state, reward, done):
        self.state.append(prepro_state(state))
        self.action.append(action)
        self.next_state.append(prepro_state(next_state))
        self.reward.append(reward)
        self.not_done.append([1. - done])

    def sample(self, batch_size):
        self.timer.set()
        ind = np.random.randint(0, self.size, size=batch_size)
        self.timer.log('sample_batch_idx')

        self.timer.set()
        states = [self.state[i] for i in ind]
        actions = [self.action[i].reshape(-1, 3) for i in ind]
        next_states = [self.next_state[i] for i in ind]
        rewards = [self.reward[i] for i in ind]
        not_dones = [self.not_done[i] for i in ind]
        self.timer.log('sample_batch_fetch_list')

        self.timer.set()
        batch = (
            prepro_graph_batch(states),
            self.prepro_tensor_batch(actions),
            prepro_graph_batch(next_states),
            self.prepro_tensor_batch(rewards),
            torch.FloatTensor(not_dones).to(self.device)
        )
        self.timer.log('sample_batch_prepro_batch')

        return batch
