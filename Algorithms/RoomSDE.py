import math
import copy

import numpy as np
from scipy import integrate

import torch
from torch_scatter import scatter_sum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def marginal_prob_std(t, sigma):
    t = torch.tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t


# for conditional room arrangement
# irrelevant to state, size dim
def loss_fn_cond(model, x, batchsize, marginal_prob_std_func, eps=1e-5):
    wall_batch, obj_batch = x
    random_t = torch.rand(batchsize, device=device) * (1. - eps) + eps
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    # [bs, 1] -> [num_nodes, 1]
    random_t = random_t[obj_batch.batch]
    # z: [num_nodes, 3]
    z = torch.randn_like(obj_batch.x)
    # std: [num_nodes, 1]
    std = marginal_prob_std_func(random_t)
    perturbed_obj_batch = copy.deepcopy(obj_batch)
    perturbed_obj_batch.x += z * std
    output = model((wall_batch, perturbed_obj_batch), random_t)
    # output: [num_nodes, 3]
    node_l2 = torch.sum((output * std + z) ** 2, dim=-1)
    batch_l2 = scatter_sum(node_l2, obj_batch.batch, dim=0)
    loss_ = torch.mean(batch_l2)
    return loss_


def cond_ode_sampler(score_model,
                     marginal_prob_std,
                     diffusion_coeff,
                     test_batch,
                     batch_size=64,
                     atol=1e-5,
                     rtol=1e-5,
                     device='cuda',
                     eps=1e-3,
                     t0=1,
                     num_steps=None):
    wall_batch, obj_batch, _ = test_batch
    wall_batch = wall_batch.to(device)
    obj_batch = obj_batch.to(device)
    # set_trace()
    t0_ = torch.ones(batch_size, device=device).unsqueeze(-1) * t0
    t0_ = t0_[obj_batch.batch]

    # Create the latent code
    init_x = torch.randn_like(obj_batch.x, device=device) * marginal_prob_std(t0_)
    shape = init_x.shape
    state_dim = shape[-1]

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,))

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        cur_obj_batch = copy.deepcopy(obj_batch)
        cur_obj_batch.x = torch.tensor(x.reshape(-1, state_dim)).to(device).float()
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        time_steps = time_steps[obj_batch.batch]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper((wall_batch, cur_obj_batch), time_steps)

    # Run the black-box ODE solver.
    t_eval = None
    if num_steps is not None:
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (t0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45', t_eval=t_eval)
    # process, xs: [total_nodes*3, samples_num]
    xs = torch.clamp(torch.tensor(res.y, device=device).T, min=-1.0, max=1.0)
    xs = xs.view(num_steps, obj_batch.x.shape[0], -1)
    # xs = torch.tensor(res.y, device=device).T
    # result x: [total_nodes, 3]
    x = torch.clamp(torch.tensor(res.y[:, -1], device=device).reshape(shape), min=-1.0, max=1.0)
    # x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return xs, x

def score_to_action(scores, states_x):
    pos_grad = scores[:, 0:2]
    ori_2d_grad = scores[:, 2:4]
    cur_n = states_x[:, 2:4]  # [sin(x), cos(x)]
    # set_trace()
    assert torch.abs(torch.sum((torch.sum(cur_n ** 2, dim=-1) - 1))) < 1e7
    cur_n = torch.cat([-cur_n[:, 0:1], cur_n[:, 1:2]], dim=-1)  # [-sin(x), cos(x)]
    ori_grad = torch.sum(torch.cat([ori_2d_grad[:, 1:2], ori_2d_grad[:, 0:1]], dim=-1) * cur_n, dim=-1, keepdim=True)
    return torch.cat([pos_grad, ori_grad], dim=-1)

def cond_ode_vel_sampler(
        score_fn,
        marginal_prob_std,
        diffusion_coeff,
        test_batch,
        t0=1.,
        eps=1e-3,
        num_steps=100,
        batch_size=1,
        max_pos_vel=0.4,
        max_ori_vel=0.2,
        scale=1e-4,
        is_decay=False,
        decay_rate=0.95,
        device='cuda',
        sup_rate=1,
):
    wall_batch, obj_batch, _ = test_batch
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
    decay_freq = 10
    for idx, t in enumerate(t_eval):
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

        if (idx + 1) % decay_freq == 0 and is_decay:
            scale *= decay_rate

    return torch.cat(xs, dim=0), xs[-1]

