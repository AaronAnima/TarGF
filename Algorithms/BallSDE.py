import os
from re import I
import time
import math
import copy
import pickle
import argparse
import functools
from collections import deque

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy import integrate
from ipdb import set_trace
from tqdm import tqdm, trange


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, TransformerConv
from torch_scatter import scatter_max, scatter_mean, scatter_sum

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
            # categories = categories
            class_feature = self.embed_category(categories)
            # set_trace()
            init_feature = torch.cat([self.init_lin(x), class_feature], dim=-1)
        else:
            init_feature = self.init_lin(x)

        # get t feature
        # t -> [bs*30, embed_dim]
        bs = t.shape[0]
        # set_trace()
        x_sigma = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, 3*self.n_box, 1).view(bs*self.n_box*3, -1)

        # start massage passing from init-feature
        x = F.relu(self.conv1(init_feature, edge_index))
        # set_trace()
        x = torch.cat([x, x_sigma], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.conv3(x, edge_index)

        # normalize the output
        # 注意当t=0时，marginal_prob_std = 0，其实意思就是梯度无穷大
        x = x / (self.marginal_prob_std(t.repeat(1, 3*self.n_box).view(bs*3*self.n_box, -1))+1e-7)
        return x


# fc version for overfitting
# class ScoreModelGNN(nn.Module):
#     def __init__(self, marginal_prob_std_func, n_box, mode, device, hidden_dim=64, embed_dim=32):
#         super(ScoreModelGNN, self).__init__()
#         self.mode = mode        
#         self.device = device
#         self.n_box = n_box

#         # t-feature
#         self.embed = nn.Sequential(
#             GaussianFourierProjection(embed_dim=embed_dim),
#             nn.ReLU(True),
#             nn.Linear(embed_dim, embed_dim),
#         )

#         self.backbone = nn.Sequential(
#             nn.Linear(2*self.n_box*3, hidden_dim),
#             nn.ReLU(True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(True),
#         )
#         self.out_tail = nn.Linear(hidden_dim+embed_dim, 2*self.n_box*3)
        
#         self.marginal_prob_std = marginal_prob_std_func

#     def forward(self, state_inp, t, n_box):
#         self.n_box = n_box
#         # t.shape == [bs, 1]
#         x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch
#         x_sigma = F.relu(self.embed(t.squeeze(1)))
#         x = self.backbone(x.view(-1, 2*self.n_box*3))
#         x = torch.cat([x, x_sigma], dim=-1)
#         x = self.out_tail(x)
#         x = x.view(-1, 2)
#         bs = t.shape[0]

#         x = x / (self.marginal_prob_std(t.repeat(1, 3*self.n_box).view(bs*3*self.n_box, -1))+1e-7)
#         return x


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


def marginal_prob_std(t, sigma):
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t

# for unconditional ball-arrangement
def loss_fn_state(model, x, marginal_prob_std_func, n_box, eps=1e-5):
    random_t = torch.rand(x.x.shape[0]//(3*n_box), device=device) * (1. - eps) + eps
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    z = torch.randn_like(x.x)
    # -> [bs*30, 1]
    std = marginal_prob_std_func(random_t).repeat(1, 3*n_box).view(-1, 1)
    perturbed_x = copy.deepcopy(x)
    perturbed_x.x += z * std
    output = model(perturbed_x, random_t, n_box)
    bs = random_t.shape[0]
    loss_ = torch.mean(torch.sum(((output * std + z)**2).view(bs, -1)), dim=-1)
    return loss_


def loss_fn_img(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss


def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           n_box=5, 
                           num_steps=500,
                           device='cuda',
                           t0=0.1,
                           eps=1e-3,
                           n=15):
    # start from prior distribution, this 't' is only used to compute the sigma of prior dist
    t = torch.ones(batch_size, device=device).unsqueeze(-1)
    k = n - 1
    init_x = torch.rand(batch_size*n, 2, device=device) * marginal_prob_std(t).repeat(1, n).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
    init_x = Data(x=init_x, edge_index=knn_graph(init_x.cpu(), k=k, batch=init_x_batch),
                       batch=init_x_batch).to(device)
    # init_x: Data(x=[480, 2], edge_index=[2, 13920], batch=[480])
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    states = []
    with torch.no_grad():
        for time_step in time_steps:
            # [bs, 1]
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # [bs*30, 1]
            g = diffusion_coeff(batch_time_step).repeat(1, n).view(-1, 1)
            # [bs*30, 2]
            mean_x = x.x + (g ** 2) * score_model(x, batch_time_step, n_box=n_box) * step_size
            # [bs*30, 2]
            x.x = (mean_x + torch.sqrt(step_size) * g * torch.randn_like(x.x)).clamp(-1, 1)
            # Do not include any noise in the last sampling step.
            states.append(x.x.view(-1).unsqueeze(0))

    
    return torch.cat(states, dim=0), mean_x

#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

def pc_sampler_state(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               n_box,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3,
               n=21,
               t0=1,
               ):
    
    # t = torch.ones(batch_size, device=device).unsqueeze(-1)*t0
    t = torch.ones(batch_size, device=device).unsqueeze(-1)*0.1
    k = n - 1
    init_x = torch.randn(batch_size * n, 2, device=device) * marginal_prob_std(t).repeat(1, n).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
    init_x = Data(x=init_x, edge_index=knn_graph(init_x.cpu(), k=k, batch=init_x_batch, loop=False),
            batch=init_x_batch).to(device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    step_size *= 10
    noise_norm = np.sqrt(n)
    x = init_x
    states = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # # Corrector step (Langevin MCMC)
            # grad = score_model(x, batch_time_step, n_box=n_box)
            # grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            # langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            # x.x = x.x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x.x)      

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step).repeat(1, n).view(-1, 1)
            mean_x = x.x + (g ** 2) * score_model(x, batch_time_step, n_box=n_box) * step_size
            x.x = mean_x
            # # [bs*30, 2]
            # x.x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x.x)

            x.x = torch.clamp(x.x, min=-1.0, max=1.0)
            states.append(x.x.view(-1).unsqueeze(0))

    # The last step does not include any noise
    return torch.cat(states, dim=0), mean_x


def pc_sampler_img(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               im_size=64,
               batch_size=64, 
               num_steps=500, # from colab tutorial
               snr=0.16,      # from colab tutorial
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.
    
    Returns: 
        Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 3, im_size, im_size, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
        
        # The last step does not include any noise
        return x_mean


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                n_box,
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                eps=1e-3,
                t0=1,
                n=None,
                num_steps=None):
    t = torch.ones(batch_size, device=device).unsqueeze(-1) * t0
    # t = torch.ones(batch_size, device=device).unsqueeze(-1)

    n = 3*n_box
    k = n - 1
    # Create the latent code
    init_x = torch.randn(batch_size * n, 2, device=device) \
        * marginal_prob_std(t).repeat(1, n).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(n)], dtype=torch.int64)
    edge_index = knn_graph(init_x.cpu(), k=k, batch=init_x_batch)
    
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():    
            score = score_model(sample, time_steps, n_box)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        x = Data(x=torch.tensor(x.reshape(-1, 2), dtype=torch.float32), edge_index=edge_index, batch=init_x_batch).to(device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    t_eval = None
    if num_steps is not None:
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.clamp(torch.tensor(res.y[:n* 2], device=device).T, min=-1.0, max=1.0)
    x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
    return xs, x

def ode_likelihood(x,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=1,
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
      x: Input data.
      score_model: A PyTorch model representing the score-based model.
      marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
      batch_size: The batch size. Equals to the leading dimension of `x`.
      device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
      eps: A `float` number. The smallest time step for numerical stability.

    Returns:
      z: The latent code for `x`.
      bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    # set_trace()
    shape = x.shape # [-1]
    x = torch.tensor(x).view(-1, 2).to(device)
    epsilon = torch.randn_like(x).to(device)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(15)], dtype=torch.int64).to(device)
    edge_index = knn_graph(x, k=14, batch=init_x_batch)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            sample_data = Data(x=sample, edge_index=edge_index).to(device)
            score_e = torch.sum(score_model(sample_data, time_steps, num_cat=5) * epsilon)
            # set_trace()
            grad_score_e = torch.autograd.grad(score_e, sample)[0] # [15, 2]
        return torch.sum(grad_score_e * epsilon)


    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample.reshape((-1, 2)), device=device, dtype=torch.float32)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32)
        sample = Data(x=sample, edge_index=edge_index).to(device)
        # set_trace()
        with torch.no_grad():
            score = score_model(sample, time_steps, num_cat=5)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample.reshape((-1, 2)), device=device, dtype=torch.float32)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32)
            # Compute likelihood.
            # sample = Data(x=sample, edge_index=edge_index).to(device)
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((1, 1)) * t
        sample = x[:-batch_size]
        logp = x[-batch_size:]
        # set_trace()
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((batch_size,))], axis=0)
    # Black-box ODE solver
    # set_trace()
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [31]
    z = zp[:-batch_size].reshape(shape) # [30]
    delta_logp = zp[-batch_size:].reshape(batch_size)
    sigma_max = marginal_prob_std(torch.tensor([1.])).to(device)
    # set_trace()
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    # set_trace()
    return z, bpd
