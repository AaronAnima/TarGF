import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch_geometric.data import Batch
from ipdb import set_trace

from utils.preprocesses import prepro_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TarGFSACPlanner(nn.Module):
    def __init__(self, configs, actor_net, targf, max_action):
        super(TarGFSACPlanner, self).__init__()
        self.residual_actor = actor_net
        self.targf = targf
        self.max_action = max_action
        self.residual_t0 = configs.residual_t0
        self.env_type = configs.env_type

    def forward(self, state_inp):
        ''' get gradient_based_action ''' 
        grad_based_action = self.targf.inference(
            state_inp, t0=self.residual_t0, is_numpy=False, grad_2_act=True, norm_type='tanh', empty=False
        ) 
        if self.env_type == 'Ball':
            bs = state_inp.shape[0]
            grad_based_action = grad_based_action.view(bs, -1)
        

        ''' get residual action '''
        residual_mu, residual_std = self.residual_actor(state_inp)

        final_mu = grad_based_action + residual_mu

        dist = SquashedNormal(final_mu, residual_std) 
        return dist
    
    def select_action(self, state, sample=False):
        # return [action_dim, ], numpy
        if self.env_type == 'Room':
            if not torch.is_tensor(state[0]):
                state_inp = prepro_state(state, cuda=True)
                obj_batch = Batch.from_data_list([state_inp[1]])
                wall_batch = state_inp[0].view(-1, 1)
                state_inp = (wall_batch, obj_batch)
            else:
                state_inp = state
        elif self.env_type == 'Ball':
            state_inp = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            raise ValueError(f"Mode {self.env_type} not recognized.")
    
        dist = self.forward(state_inp)
        action = dist.sample() if sample else dist.mean
        return action.clamp(-self.max_action, self.max_action).detach().cpu().numpy().flatten()


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

