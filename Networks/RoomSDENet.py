import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv
from Algorithms.RoomSDE import marginal_prob_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CondScoreModelGNN(nn.Module):
    def __init__(self, marginal_prob_std_func, hidden_dim, embed_dim, wall_dim=2, class_num=10, state_dim=4, size_dim=2,
                 mode='target'):
        super(CondScoreModelGNN, self).__init__()
        self.marginal_prob_std = marginal_prob_std_func
        hidden_dim = hidden_dim
        embed_dim = embed_dim
        self.mode = mode

        self.init_enc = nn.Sequential(
            nn.Linear(state_dim + size_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # wall geo feature is more complicated
        self.wall_enc = nn.Sequential(
            nn.Linear(wall_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.embed_sigma = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                         nn.Linear(embed_dim, embed_dim))
        self.embed_category = nn.Embedding(class_num, embed_dim)

        # cond_dim = hidden_dim*2 + embed_dim*2 # consider wall
        if self.mode == 'target':
            cond_dim = embed_dim * 3  # remove wall
        else:
            cond_dim = embed_dim * 2  # no category feature, only bbox

        ''' wall geometry '''
        self.mlp1_wall = nn.Sequential(
            nn.Linear(wall_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_wall = EdgeConv(self.mlp1_wall)
        self.mlp2_wall = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2_wall = EdgeConv(self.mlp2_wall)

        ''' main backbone '''
        # conv1
        self.mlp1_main = nn.Sequential(
            nn.Linear((hidden_dim + cond_dim) * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_main = EdgeConv(self.mlp1_main)
        # conv2
        self.mlp2_main = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cond_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, state_dim),
        )
        self.conv2_main = EdgeConv(self.mlp2_main)

    def forward(self, batches, t):
        wall_batch, obj_batch = batches
        ''' get cond feat'''
        # class_feat: -> [num_nodes, embed_dim]
        class_feat = F.relu(self.embed_category(obj_batch.category.squeeze(-1)))
        # sigma_feat: [num_nodes, embed_dim]
        sigma_feat = F.relu(self.embed_sigma(t.squeeze(-1)))
        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*2]
        wall_feat = self.wall_enc(wall_batch)[obj_batch.batch]
        if self.mode == 'target':
            total_cond_feat = torch.cat([class_feat, sigma_feat, wall_feat], dim=-1)  # consider wall
        else:
            total_cond_feat = torch.cat([sigma_feat, wall_feat], dim=-1)  # no category feature

        ''' get init x feat '''
        # obj_init_feat: -> [num_nodes, hidden_dim]
        obj_init_feat = self.init_enc(torch.cat([obj_batch.x, obj_batch.geo], dim=-1))

        ''' main backbone of x '''
        # start massage passing from init-feature
        x = torch.cat([obj_init_feat, total_cond_feat], dim=-1)
        x = F.relu(self.conv1_main(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = self.conv2_main(x, obj_batch.edge_index)

        # normalize the output
        x = x / (self.marginal_prob_std(t) + 1e-7)
        return x

class ScoreWrapper:
    def __init__(self, tar_score=None, sup_score=None, sup_t0=0.001):
        self.tar = tar_score
        self.sup = sup_score
        self.sup_t0 = sup_t0

    def load_scores(self, my_device, params, exp_tar='None', exp_sup='None'):
        # set device
        self.tar = self.load_score(exp_tar, my_device, 'target', params)
        self.sup = self.load_score(exp_sup, my_device, 'support', params)

    @staticmethod
    def load_score(exp_name, my_device, mode, params):
        if exp_name != 'None':
            sigma = 25.0  # @param {'type':'number'}
            marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
            score = CondScoreModelGNN(
                marginal_prob_std_fn,
                params[f'{mode}_hidden_dim'],
                params[f'{mode}_embed_dim'],
                wall_dim=1,
                mode=mode
            )
            ckpt_path = f'./logs/{exp_name}/' + f'score.pt'
            score.load_state_dict(torch.load(ckpt_path))
            score.eval()
            score.to(my_device)
        else:
            score = None
        return score

    def mode(self):
        if self.tar is None and self.sup is None:
            raise NotImplementedError
        if self.tar is None and self.sup is not None:
            return 'sup'
        if self.tar is not None and self.sup is None:
            return 'tar'
        if self.tar is not None and self.sup is not None:
            return 'dual'

    def inference(self, sample, time_step, sup_rate=1.0, is_langevin=False):
        assert not (self.tar is None and self.sup is None)
        grad = 0
        grad_tar = None
        grad_sup = None
        if self.tar is not None:
            grad_tar = self.tar(sample, time_step)
            grad += grad_tar

        if self.sup is not None:
            time_sup = self.sup_t0 * torch.ones_like(time_step).to(time_step.device)
            grad_sup = self.sup(sample, time_sup)
            grad += grad_sup * sup_rate
        if is_langevin:
            return grad_tar, grad_sup
        else:
            return grad

