import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph
from torch_geometric.nn import EdgeConv
from torch import distributions as pyd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorTanh(nn.Module):
    def __init__(self, log_std_bounds, target_score, t0=0.01, hidden_dim=128, embed_dim=64, max_action=None, num_boxes=10, knn=10, is_residual=True):
        super(ActorTanh, self).__init__()
        self.knn = knn
        self.num_boxes = num_boxes*3
        self.max_action = max_action
        self.t0 = t0
        self.is_residual = is_residual
        self.target_score = target_score
        self.init_lin = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_category = nn.Sequential(
            nn.Embedding(3, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        # conv 1
        self.mlp1 = nn.Sequential(
            nn.Linear((hidden_dim+embed_dim)*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = EdgeConv(self.mlp1)

        self.shared_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2*2)
        )
        self.log_std_bounds = log_std_bounds

    def forward(self, state_inp):
        # convert normal batch to graph
        k = self.knn
        bs = state_inp.shape[0]
        # prepare target scores
        tar_scores = self.target_score.get_score(state_inp, t0=self.t0, is_numpy=False, is_norm=False, empty=(not self.is_residual)) # [num_nodes, 2]

        state_inp = state_inp.view(-1, 2)
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_boxes)], dtype=torch.int64).to(device)
        x, edge_index = state_inp, knn_graph(state_inp, k=k, batch=samples_batch)

        # get category_feature
        categories = torch.cat([torch.ones(self.num_boxes//3, device=device) * c for c in [0, 1, 2]], dim=0).repeat(x.shape[0]//self.num_boxes)
        categories = categories.long().to(device)
        class_feature = self.embed_category(categories)

        # get overall feature
        # note that it is better to normalise the tar_scores to [-1, 1]
        init_feature = torch.tanh(torch.cat([self.init_lin(torch.cat([x, torch.tanh(tar_scores)], -1)), class_feature], dim=-1))

        # get sigma feature
        # init_feature: [bs, num_nodes, dim]
        ''' GNN '''
        x = torch.tanh(self.conv1(init_feature, edge_index))
        mu, log_std = self.shared_actor(x).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std+1)
        std = log_std.exp().view(-1, 2*self.num_boxes)

        # mix with tar vel func
        mu = self.max_action * torch.tanh(mu.contiguous().view(-1, 2*self.num_boxes))
        if self.is_residual:
            mu += self.max_action * torch.tanh(tar_scores.view(bs, -1))
        dist = SquashedNormal(mu, std) 
        return dist


class CriticTanh(nn.Module):
    def __init__(self, target_score, t0=0.01, hidden_dim=128, embed_dim=64, is_residual=False, num_boxes=10, knn=15-1):
        super(CriticTanh, self).__init__()
        self.num_boxes = num_boxes*3
        self.knn = knn
        self.t0 = t0

        self.target_score = target_score
        self.is_residual = is_residual

        # Q1 architecture
        self.init_lin_1 = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_category_1 = nn.Sequential(
            nn.Embedding(3, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp1_1 = nn.Sequential(
            nn.Linear((hidden_dim+embed_dim)*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_1 = EdgeConv(self.mlp1_1)

        self.shared_critic_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.init_lin_2 = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_category_2 = nn.Sequential(
            nn.Embedding(3, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp1_2 = nn.Sequential(
            nn.Linear((hidden_dim+embed_dim)*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_2 = EdgeConv(self.mlp1_2)
        self.shared_critic_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

    def Q1(self, state, action):
        tar_scores = self.target_score.get_score(state, t0=self.t0, is_numpy=False, is_norm=False, empty=(not self.is_residual)) # [num_nodes, 2]
        state_inp = torch.cat([state.view(-1, self.num_boxes, 2), action.view(-1, self.num_boxes, 2)], -1)

        # convert normal batch to graph
        k = self.knn
        bs = state_inp.shape[0]
        state_inp = state_inp.view(-1, 4) 
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_boxes)], dtype=torch.int64).to(device)
        x, edge_index = state_inp, knn_graph(state_inp, k=k, batch=samples_batch)

        # get category_feature
        categories = torch.cat([torch.ones(self.num_boxes//3, device=device) * c for c in [0, 1, 2]], dim=0).repeat(x.shape[0]//self.num_boxes)
        categories = categories.long().to(device)
        class_feature = self.embed_category_1(categories)

        # get overall feature
        init_feature = self.init_lin_1(torch.cat([torch.tanh(tar_scores), x], dim=-1))
        h = torch.cat([init_feature, class_feature], dim=-1)
        h = torch.tanh(h)

        # get sigma feature
        x = torch.tanh(self.conv1_1(h, edge_index))
        out = self.shared_critic_1(x)
        return out.view(bs, -1)

    def Q2(self, state, action):
        # state: [bs, n_obj*2], action: [bs, n_obj*2]
        tar_scores = self.target_score.get_score(state, t0=self.t0, is_numpy=False, is_norm=False, empty=(not self.is_residual)) # [num_nodes, 2]
        state_inp = torch.cat([state, action], 1)
        # convert normal batch to graph
        k = self.knn
        bs = state_inp.shape[0]
        state_inp = state_inp.view(-1, 4)
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_boxes)], dtype=torch.int64).to(device)
        x, edge_index = state_inp, knn_graph(state_inp, k=k, batch=samples_batch)

        # get category_feature
        categories = torch.cat([torch.ones(self.num_boxes//3, device=device) * c for c in [0, 1, 2]], dim=0).repeat(x.shape[0]//self.num_boxes)
        categories = categories.long().to(device)
        class_feature = self.embed_category_2(categories)

        # get overall feature
        init_feature = self.init_lin_2(torch.cat([torch.tanh(tar_scores), x], dim=-1))
        h = torch.cat([init_feature, class_feature], dim=-1)
        h = torch.tanh(h)


        # 1. message passing
        # init_feature: [bs*n_agents, hidden_dim]
        x = torch.tanh(self.conv1_2(h, edge_index)) # [bs*n_agents, hidden_dim] -> [bs*n_agents, hidden_dim]

        # 2. pass through shared-critic(2*fc)
        out = self.shared_critic_2(x) # [bs*n_agents, hidden_dim] -> [bs*n_agents, 1]

        return out.view(bs, -1)


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

