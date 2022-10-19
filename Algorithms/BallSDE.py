import copy

import numpy as np
from scipy import integrate


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
