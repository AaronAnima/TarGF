import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Networks.BallSACNet import ActorTanh, CriticTanh


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
            num_objs=21,
            knn_actor=10,
            knn_critic=15,
            hidden_dim=128,
            embed_dim=64,
            residual_t0=0.01,
            init_temperature=0.1,
            is_residual=True,
            learnable_temperature=True,
    ):
        self.actor = ActorTanh(
            log_std_bounds=(-5, 2),
            max_action=max_action,
            target_score=target_score,
            num_objs=num_objs,
            knn=knn_actor,
            is_residual=is_residual,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            t0=residual_t0,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, betas=(0.9, 0.999))

        self.critic = CriticTanh(
            target_score=target_score,
            num_objs=num_objs,
            knn=knn_critic,
            is_residual=is_residual,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            t0=residual_t0,
        ).to(device)
        self.target_score = target_score
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, betas=(0.9, 0.999))

        # alpha
        self.learnable_temperature = learnable_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -num_objs*2 # action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4, betas=(0.9, 0.999))

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.critic_target_update_frequency = 2

        self.total_it = 0
        self.writer = writer

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action(self, state, sample=False):
        # return [action_dim, ]
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.clamp(-self.max_action, self.max_action).detach().cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
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

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ''' update actor '''
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            dist = self.actor(state)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.critic(state, action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
            self.writer.add_scalar('Losses/actor_loss', actor_loss, self.total_it)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.learnable_temperature:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                self.writer.add_scalar('Losses/alpha_loss', alpha_loss, self.total_it)
                self.writer.add_scalar('Losses/alpha', self.alpha, self.total_it)

                alpha_loss.backward()
                self.log_alpha_optimizer.step()
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
    def __init__(self, state_dim, action_dim, num_objs, max_size=int(1e6), centralized=True):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1 if centralized else num_objs))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

