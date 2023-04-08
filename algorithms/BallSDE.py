import copy

import numpy as np
from scipy import integrate
from ipdb import set_trace

import torch

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def marginal_prob_std(t, sigma):
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t

# for unconditional ball-arrangement
def loss_fn_state(model, x, marginal_prob_std_func, num_objs, eps=1e-5):
    random_t = torch.rand(x.x.shape[0]//num_objs, device=device) * (1. - eps) + eps
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    z = torch.randn_like(x.x)
    # -> [bs*num_objs, 1]
    std = marginal_prob_std_func(random_t).repeat(1, num_objs).view(-1, 1)
    perturbed_x = copy.deepcopy(x)
    perturbed_x.x += z * std
    output = model(perturbed_x, random_t, num_objs)
    bs = random_t.shape[0]
    loss_ = torch.mean(torch.sum(((output * std + z)**2).view(bs, -1), dim=-1))
    return loss_

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                num_objs,
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                eps=1e-3,
                t0=1,
                num_steps=None):
    # initial t, t0 can be specified with a scalar ranges in (0, 1]
    t = torch.ones(batch_size, device=device).unsqueeze(-1) * t0
    
    k = num_objs - 1
    # Create the latent code
    init_x = torch.randn(batch_size * num_objs, 2, device=device) * marginal_prob_std(t).repeat(1, num_objs).view(-1, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(num_objs)], dtype=torch.int64)
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


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                num_objs,
                ref_batch, # a reference batch, provide category feature
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                eps=1e-3,
                t0=None,
                num_steps=None):

    # initial t, t0 can be specified with a scalar ranges in (0, 1]
    t0 = 1 if t0 is None else t0
    init_t = torch.ones(batch_size, device=device).unsqueeze(-1) * t0
    knn = num_objs - 1

    # construct init x batch
    init_x = torch.randn(batch_size * num_objs, 2, device=device) * marginal_prob_std(init_t).repeat(1, num_objs).view(-1, 1)
    
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(num_objs)], dtype=torch.int64)
    ref_c = ref_batch.c[ref_batch.ptr[0]:ref_batch.ptr[1]]
    
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        score = score_model(sample, time_steps, num_objs)
        score = score.detach()
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        x_inp = torch.tensor(x.reshape(-1, 2), dtype=torch.float32)
        edge = knn_graph(x_inp, k=knn, batch=init_x_batch)
        x = Data(
            x=x_inp, 
            edge_index=edge, 
            batch=init_x_batch, 
            c=ref_c.repeat(batch_size),
        ).to(device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        ret = -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
        return ret
  
    # Run the black-box ODE solver.
    t_eval = None
    if num_steps is not None:
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.clamp(torch.tensor(res.y[:num_objs* 2], device=device).T, min=-1.0, max=1.0) # save only one video (idx-0)
    x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
    # concat sampled positions with labels
    xs = torch.cat([xs.contiguous().view(-1, 2), ref_c.repeat(num_steps).unsqueeze(-1)], dim=-1).view(num_steps, -1).float()
    x = torch.cat([x, ref_c.repeat(batch_size).unsqueeze(-1)], dim=-1).view(batch_size, -1).float()
    return xs, x
