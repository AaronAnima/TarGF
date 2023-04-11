import copy
from ipdb import set_trace
import numpy as np
from scipy import integrate

import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# room sampler 

def cond_ode_vel_sampler(
        score_fn,
        marginal_prob_std,
        diffusion_coeff,
        ref_batch,
        t0=1.,
        eps=1e-3,
        num_steps=500,
        batch_size=1,
        max_pos_vel=0.6,
        max_ori_vel=0.6,
        scale=0.04,
        device='cuda',
):
    wall_batch, obj_batch, _ = ref_batch
    wall_batch = wall_batch.to(device)
    obj_batch = obj_batch.to(device)

    # Create the latent code
    init_x = torch.randn_like(obj_batch.x, device=device) * marginal_prob_std(t0)
    

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_fn(sample, time_steps)
        return score

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        cur_obj_batch = copy.deepcopy(obj_batch)
        cur_obj_batch.x = torch.tensor(x.reshape(-1, 4)).to(device).float()
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        time_steps = time_steps[obj_batch.batch]

        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return 0.5 * (g ** 2) * score_eval_wrapper((wall_batch, cur_obj_batch), time_steps)

    t_eval = np.linspace(t0, eps, num_steps)

    xs = []
    x = init_x
    x[:, 2:4] /= torch.sqrt(torch.sum(x[:, 2:4] ** 2, dim=-1, keepdim=True))

    # linear decay freq
    for _, t in enumerate(t_eval):
        velocity = ode_func(t, x)  # [bs, 4]
        pos_vel = velocity[:, 0:2]
        max_pos_vel_norm = torch.max(torch.abs(pos_vel))
        pos_vel /= max_pos_vel_norm
        pos_vel *= max_pos_vel  # RL action, pos_vel
        delta_pos = pos_vel * scale
        new_pos = x[:, 0:2] + delta_pos

        ori_vel = velocity[:, 2:4]
        cur_n = x[:, 2:4]  # [sin(x), cos(x)]
        assert torch.abs(torch.sum((torch.sum(cur_n ** 2, dim=-1) - 1))) < 1e7
        cur_n = torch.cat([-cur_n[:, 0:1], cur_n[:, 1:2]], dim=-1)  # [-sin(x), cos(x)]
        grad_theta = torch.sum(torch.cat([ori_vel[:, 1:2], ori_vel[:, 0:1]], dim=-1) * cur_n, dim=-1, keepdim=True)
        ori_vel = grad_theta * max_ori_vel / torch.max(torch.abs(grad_theta))  # RL action, ori vel
        delta_ori = ori_vel * scale
        cd = torch.cos(delta_ori)
        sd = torch.sin(delta_ori)
        cx = x[:, 3:4]
        sx = x[:, 2:3]
        new_ori = torch.cat([sx * cd + cx * sd, cx * cd - sx * sd], dim=-1)
        new_ori /= torch.sqrt(torch.sum(new_ori ** 2, dim=-1, keepdim=True))

        
        x = torch.cat([new_pos, new_ori], dim=-1)

        xs.append(x.cpu().unsqueeze(0).clone())
    return torch.cat(xs, dim=0), xs[-1][0]


    
# ball sampler
def ode_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    ref_batch, # a reference batch, provide category feature
    num_objs=21,
    batch_size=64, 
    atol=1e-5, 
    rtol=1e-5, 
    device='cuda', 
    eps=1e-3,
    t0=None,
    num_steps=None,
):
    ref_batch = ref_batch.to(device)
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

