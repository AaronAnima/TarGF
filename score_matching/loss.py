import copy
from ipdb import set_trace

import torch
from torch_scatter import scatter_sum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for conditional room arrangement
# irrelevant to state, size dim
def loss_fn_cond(model, x, marginal_prob_std_func, batch_size, eps=1e-5):
    wall_batch, obj_batch, _ = x
    wall_batch = wall_batch.to(device)
    obj_batch = obj_batch.to(device)
    random_t = torch.rand(batch_size, device=device) * (1. - eps) + eps
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

# for unconditional ball-arrangement
def loss_fn_uncond(model, x, marginal_prob_std_func, num_objs, eps=1e-5):
    x = x.to(device)
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

