import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, EdgeConv, GINConv, GATConv
from torch_scatter import scatter_max, scatter_mean
from torch import distributions as pyd


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


class Actor(nn.Module):
    def __init__(self, log_std_bounds, target_score, t0=0.01, hidden_dim=128, embed_dim=64, max_action=None, is_residual=True, class_num=10):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.target_score = target_score
        self.log_std_bounds = log_std_bounds
        self.t0 = t0
        self.max_action = max_action
        self.is_residual = is_residual

        self.init_enc = nn.Sequential(
            nn.Linear(4 + 3, hidden_dim), # [state, geo, tar_scores]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # wall geo feature is more complicated
        self.wall_enc = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_category = nn.Sequential(
            nn.Embedding(class_num, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geo_enc = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        cond_dim = embed_dim * 3  # remove wall

        ''' main backbone '''
        # conv1
        self.mlp1_main = nn.Sequential(
            nn.Linear((hidden_dim + cond_dim) * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_main = EdgeConv(self.mlp1_main)
        # conv2
        self.mlp2_main = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cond_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # [mu, std]
        )
        self.conv2_main = EdgeConv(self.mlp2_main)

        ''' tail '''
        self.actor_tail = nn.Sequential(
            nn.Linear(hidden_dim+cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3*2),
        )

    def forward(self, batches):
        wall_batch, obj_batch = batches
        ''' get tar scores'''
        # tar_scores: -> [num_nodes, 4]
        tar_scores = self.target_score.get_score(
            batches, t0=self.t0, is_numpy=False, is_norm=False, empty=not self.is_residual, is_action=True
        )
        ''' get cond feat'''
        # class_feat: -> [num_nodes, embed_dim]
        class_feat = torch.tanh(self.embed_category(obj_batch.category.squeeze(-1)))
        # wall_feat: [num_nodes, embed_dim]
        wall_feat = torch.tanh(self.wall_enc(wall_batch)[obj_batch.batch])
        # geo_feat: [num_nodes, embed_dim]
        geo_feat = torch.tanh(self.geo_enc(obj_batch.geo))
        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*3]
        total_cond_feat = torch.cat([class_feat, wall_feat, geo_feat], dim=-1)

        ''' get init x feat '''
        # obj_init_feat: -> [num_nodes, hidden_dim]
        obj_init_feat = torch.tanh(self.init_enc(torch.cat([obj_batch.x, tar_scores], dim=-1)))

        ''' main backbone of x '''
        # start massage passing from init-feature
        x = torch.cat([obj_init_feat, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv1_main(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv2_main(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = self.actor_tail(x)

        ''' get mu, sigma '''
        mu, log_std = x.chunk(2, dim=-1)
        # std: [num_nodes, 4]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()  # [bs*30, 2] -> [bs, 60]

        # mu: [num_nodes, 4]
        mu = self.max_action * torch.tanh(mu)
        if self.is_residual:
            mu += self.max_action * torch.tanh(tar_scores)

        dist = SquashedNormal(mu, std)
        # set_trace()
        return dist


class Critic(nn.Module):
    def __init__(self, target_score, t0=0.01, hidden_dim=128, embed_dim=64, is_residual=True, class_num=10):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.target_score = target_score
        self.t0 = t0
        self.is_residual = is_residual
        init_dim = 4+3+3 # [state, action, tar_scores]
        cond_dim = embed_dim * 3  # wall feat, geo feat, cate feat

        ''' Q1 Networks '''
        self.init_enc_1 = nn.Sequential(
            nn.Linear(init_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.wall_enc_1 = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geo_enc_1 = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_category_1 = nn.Sequential(
            nn.Embedding(class_num, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # main backbone
        self.mlp1_main_1 = nn.Sequential(
            nn.Linear((hidden_dim + cond_dim) * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_main_1 = EdgeConv(self.mlp1_main_1)
        # conv2
        self.mlp2_main_1 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cond_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # [mu, std]
        )
        self.conv2_main_1 = EdgeConv(self.mlp2_main_1)
        # tail
        self.critic_tail_1 = nn.Sequential(
            nn.Linear(hidden_dim+cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        ''' Q2 Networks '''
        self.init_enc_2 = nn.Sequential(
            nn.Linear(init_dim, hidden_dim), # [state, action, geo, tar_scores]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.wall_enc_2 = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geo_enc_2 = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_category_2 = nn.Sequential(
            nn.Embedding(class_num, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        # main backbone
        self.mlp1_main_2 = nn.Sequential(
            nn.Linear((hidden_dim + cond_dim) * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_main_2 = EdgeConv(self.mlp1_main_2)
        # conv2
        self.mlp2_main_2 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cond_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # [mu, std]
        )
        self.conv2_main_2 = EdgeConv(self.mlp2_main_2)
        # tail
        self.critic_tail_2 = nn.Sequential(
            nn.Linear(hidden_dim+cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions):
        return self.Q1(states, actions), self.Q2(states, actions)

    def Q1(self, states, actions):
        wall_batch, obj_batch = states
        ''' get tar scores'''
        # tar_scores: -> [num_nodes, 4]
        tar_scores = self.target_score.get_score(
            states, t0=self.t0, is_numpy=False, is_norm=False, empty=not self.is_residual, is_action=True
        )
        ''' get cond feat'''
        # class_feat: -> [num_nodes, embed_dim]
        class_feat = torch.tanh(self.embed_category_1(obj_batch.category.squeeze(-1)))
        # wall_feat: [num_nodes, embed_dim]
        wall_feat = torch.tanh(self.wall_enc_1(wall_batch)[obj_batch.batch])
        # geo_feat: [num_nodes, embed_dim]
        geo_feat = torch.tanh(self.geo_enc_1(obj_batch.geo))
        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*3]
        total_cond_feat = torch.cat([class_feat, wall_feat, geo_feat], dim=-1)

        ''' get init x feat '''
        # obj_init_feat: -> [num_nodes, hidden_dim]
        obj_init_feat = torch.tanh(self.init_enc_1(torch.cat([obj_batch.x, actions, tar_scores], dim=-1)))

        ''' main backbone of x '''
        # start massage passing from init-feature
        x = torch.cat([obj_init_feat, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv1_main_1(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv2_main_1(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        q1 = self.critic_tail_1(x)

        return q1

    def Q2(self, states, actions):
        wall_batch, obj_batch = states
        ''' get tar scores'''
        # tar_scores: -> [num_nodes, 4]
        tar_scores = self.target_score.get_score(
            states, t0=self.t0, is_numpy=False, is_norm=False, empty=not self.is_residual, is_action=True
        )
        ''' get cond feat'''
        # class_feat: -> [num_nodes, embed_dim]
        class_feat = torch.tanh(self.embed_category_2(obj_batch.category.squeeze(-1)))
        # wall_feat: [num_nodes, embed_dim]
        wall_feat = torch.tanh(self.wall_enc_2(wall_batch)[obj_batch.batch])
        # geo_feat: [num_nodes, embed_dim]
        geo_feat = torch.tanh(self.geo_enc_2(obj_batch.geo))

        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*3]
        total_cond_feat = torch.cat([class_feat, wall_feat, geo_feat], dim=-1)

        ''' get init x feat '''
        # obj_init_feat: -> [num_nodes, hidden_dim]
        obj_init_feat = torch.tanh(self.init_enc_2(torch.cat([obj_batch.x, actions, tar_scores], dim=-1)))

        ''' main backbone of x '''
        # start massage passing from init-feature
        x = torch.cat([obj_init_feat, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv1_main_2(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = torch.tanh(self.conv2_main_2(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        q2 = self.critic_tail_2(x)
        return q2



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

