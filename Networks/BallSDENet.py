import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv
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


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreModelGNN(nn.Module):
    def __init__(self, marginal_prob_std_func, n_box, mode, device, hidden_dim=64, embed_dim=32):
        super(ScoreModelGNN, self).__init__()
        self.mode = mode        
        self.device = device
        self.n_box = n_box

        # original x
        self.init_lin = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # t-feature
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )

        # category-feature
        self.embed_category = nn.Sequential(
            nn.Embedding(3, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        if mode == 'target':
            init_dim = hidden_dim + embed_dim
        else:
            init_dim = hidden_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(init_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = EdgeConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim*2+embed_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = EdgeConv(self.mlp2)
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim*2+embed_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2),
        )
        self.conv3 = EdgeConv(self.mlp3)
        
        self.marginal_prob_std = marginal_prob_std_func

    def forward(self, state_inp, t, n_box):
        self.n_box = n_box
        # t.shape == [bs, 1]
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch
        # we can norm here as ncsn's code did
        if self.mode == 'target':
            categories = torch.cat([torch.ones(self.n_box) * c for c in [0, 1, 2]], dim=0).repeat(x.shape[0]//(3*self.n_box)).long().to(device)
            class_feature = self.embed_category(categories)
            init_feature = torch.cat([self.init_lin(x), class_feature], dim=-1)
        else:
            init_feature = self.init_lin(x)

        # get t feature
        bs = t.shape[0]
        x_sigma = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, 3*self.n_box, 1).view(bs*self.n_box*3, -1)

        # start massage passing from init-feature
        x = F.relu(self.conv1(init_feature, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.conv3(x, edge_index)

        # normalize the output
        x = x / (self.marginal_prob_std(t.repeat(1, 3*self.n_box).view(bs*3*self.n_box, -1))+1e-7)
        return x



class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, im_ch=3):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(im_ch, channels[0], 3, 1, 1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 4, 2, 1, bias=False)    
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 4, 2, 1, bias=False)    
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], im_ch, 3, 1, 1)
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
    
    def forward(self, x, t): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))    
        # Encoding path
        h1 = self.conv1(x)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None] # expand dims
        return h
