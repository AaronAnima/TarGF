from ipdb import set_trace
import numpy as np

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def marginal_prob_std(t, sigma):
    t = torch.tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t

