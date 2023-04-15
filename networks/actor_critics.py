import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn_graph
from torch import distributions as pyd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# --- Shared -----------
# ----------------------

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

# ----------------------
# ----- Room -----------
# ----------------------


class RoomActor(nn.Module):
    def __init__(self, configs, targf, max_action=None, class_num=10, log_std_bounds=(-5, 2)):
        super(RoomActor, self).__init__()
        hidden_dim = configs.hidden_dim
        embed_dim = configs.embed_dim
        self.targf = targf
        self.log_std_bounds = log_std_bounds
        self.t0 = configs.residual_t0
        self.max_action = max_action

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
        tar_scores = self.targf.inference(
            batches, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False,
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
        std = log_std.exp()  

        # mu: [num_nodes, 4]
        mu = self.max_action * torch.tanh(mu)
        return mu, std


class RoomCritic(nn.Module):
    def __init__(self, configs, targf, class_num=10):
        super(RoomCritic, self).__init__()
        hidden_dim = configs.hidden_dim
        embed_dim = configs.embed_dim
        self.targf = targf
        self.t0 = configs.residual_t0
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
        tar_scores = self.targf.inference(
            states, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False,
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
        tar_scores = self.targf.inference(
            states, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False,
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


# ----------------------
# ----- Ball -----------
# ----------------------


class BallActor(nn.Module):
    def __init__(self, configs, targf, max_action=None, log_std_bounds=(-5, 2)):
        super(BallActor, self).__init__()
        hidden_dim = configs.hidden_dim
        embed_dim = configs.embed_dim
        self.knn = configs.knn_actor
        self.num_objs = configs.num_objs
        self.max_action = max_action
        self.t0 = configs.residual_t0
        self.targf = targf
        self.spatial_lin = nn.Sequential(
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
        """
        state_inp: [bs, num_objs * 3]
        """
        # convert normal batch to graph
        k = self.knn
        bs = state_inp.shape[0]
        # prepare target scores
        tar_scores = self.targf.inference(state_inp, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False) 

        # pre-pro data for message passing
        positions = state_inp.view(bs, self.num_objs, 3)[:, :, :2].view(-1, 2)
        categories = state_inp.view(bs, self.num_objs, 3)[:, :, -1:].view(-1).long()
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
        x, edge_index = positions, knn_graph(positions, k=k, batch=samples_batch)

        # encode init feature
        class_embed = self.embed_category(categories)
        # note that it is better to normalise the tar_scores to [-1, 1]
        spatial_inp = torch.cat([x, torch.tanh(tar_scores)], -1)
        spatial_embed = self.spatial_lin(spatial_inp)
        init_feature = torch.tanh(torch.cat([spatial_embed, class_embed], dim=-1))

        ''' GNN '''
        x = torch.tanh(self.conv1(init_feature, edge_index))
        mu, log_std = self.shared_actor(x).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std+1)
        std = log_std.exp().view(-1, 2*self.num_objs)

        # mix with tar vel func
        mu = self.max_action * torch.tanh(mu.contiguous().view(-1, 2*self.num_objs))
        return mu, std


class BallCritic(nn.Module):
    def __init__(self, configs, targf):
        super(BallCritic, self).__init__()
        hidden_dim = configs.hidden_dim
        embed_dim = configs.embed_dim
        num_classes = configs.num_classes
        
        self.num_objs = configs.num_objs
        self.knn = configs.knn_critic
        self.t0 = configs.residual_t0

        self.targf = targf

        # Q1 architecture
        self.spatial_lin_1 = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_category_1 = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp1_1 = nn.Sequential(
            nn.Linear((hidden_dim+embed_dim)*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_1 = EdgeConv(self.mlp1_1)

        self.tail_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.spatial_lin_2 = nn.Sequential(
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
        self.tail_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

    def Q1(self, state, action):
        k = self.knn
        bs = state.shape[0]
        positions = state.view(bs, self.num_objs, 3)[:, :, :2].view(-1, 2)
        categories = state.view(bs, self.num_objs, 3)[:, :, -1:].view(-1).long()
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
        x, edge_index = positions, knn_graph(positions, k=k, batch=samples_batch)

        tar_scores = self.targf.inference(state, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False)
        spatial_inp = torch.cat([positions, action.view(-1, 2), torch.tanh(tar_scores)], -1)

        # get init feature
        spatial_embed = self.spatial_lin_1(spatial_inp)
        class_embed = self.embed_category_1(categories)
        h = torch.tanh(torch.cat([spatial_embed, class_embed], dim=-1))

        # get sigma feature
        x = torch.tanh(self.conv1_1(h, edge_index))
        out = self.tail_1(x)
        return out.view(bs, -1)

    def Q2(self, state, action):
        k = self.knn
        bs = state.shape[0]
        positions = state.view(bs, self.num_objs, 3)[:, :, :2].view(-1, 2)
        categories = state.view(bs, self.num_objs, 3)[:, :, -1:].view(-1).long()
        samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
        x, edge_index = positions, knn_graph(positions, k=k, batch=samples_batch)

        tar_scores = self.targf.inference(state, t0=self.t0, is_numpy=False, grad_2_act=True, norm_type='None', empty=False) 
        spatial_inp = torch.cat([positions, action.view(-1, 2), torch.tanh(tar_scores)], -1).view(-1, 2+2+2) 

        # get init feature
        spatial_embed = self.spatial_lin_2(spatial_inp)
        class_embed = self.embed_category_2(categories)
        h = torch.tanh(torch.cat([spatial_embed, class_embed], dim=-1))

        # get sigma feature
        x = torch.tanh(self.conv1_2(h, edge_index))
        out = self.tail_2(x)
        return out.view(bs, -1)


