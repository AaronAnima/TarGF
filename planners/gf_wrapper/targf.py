import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import functools
import pickle

from score_matching.sde import marginal_prob_std, diffusion_coeff
from score_matching.sampler import grad_projection
from utils.preprocesses import prepro_graph_batch
from networks import * # for pickle load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_targf(configs, max_action):
    tar_path = f'./logs/{configs.score_exp}/score.pt'

    with open(tar_path, 'rb') as f:
        score = pickle.load(f)

    return TarGF(score.to(device), configs, max_action)


class TarGF:
    def __init__(self, score_net, configs, max_action):
        self.configs = configs

        self.score_net = score_net
        self.max_action = max_action
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=configs.sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=configs.sigma)

    def get_gradient(self, state_inp, t0, grad_2_act):
        if self.configs.env_type == 'Room':
            if not torch.is_tensor(state_inp[0]):
                state_inp = prepro_graph_batch([state_inp])
            bs = state_inp[0].shape[0] # state_inp[0]: wall_feat
            t = torch.tensor([t0] * bs)[state_inp[1].batch].unsqueeze(1).to(device) # [num_nodes, 1]
            out_score = self.score_net(state_inp, t) 
            out_score = out_score.detach()
            if grad_2_act:
                out_score = grad_projection(out_score, state_inp[-1].x)
        elif self.configs.env_type == 'Ball':
            # construct graph-input for score network
            if not torch.is_tensor(state_inp):
                state_inp = torch.tensor(state_inp)
            positions = state_inp.view(-1, self.configs.num_objs, 3).to(device)[:, :, :2]
            categories = state_inp.view(-1, self.configs.num_objs, 3).to(device)[:, :, -1:]
            bs = positions.shape[0]
            positions = positions.view(-1, 2).float()
            categories = categories.view(-1).long()
            samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.configs.num_objs)], dtype=torch.int64).to(device)
            edge_index = knn_graph(positions, k=self.configs.num_objs-1, batch=samples_batch)
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            inp_data = Data(x=positions, edge_index=edge_index, c=categories)

            out_score = self.score_net(inp_data, t, self.configs.num_objs)
            out_score = out_score.detach()
        else:
            raise ValueError(f"Mode {self.configs.env_type} not recognized.")
            
        return out_score

    def inference(self, state_inp, t0, is_numpy=True, grad_2_act=True, norm_type='None', empty=False):
        ''' set grad_2_act=True if you want a gradient-based action, norm_type in ['None', 'inf', 'tanh'] '''
        if not empty:
            with torch.no_grad():
                out_score = self.get_gradient(state_inp, t0, grad_2_act)
            if norm_type == 'inf':
                # works best in experiments
                out_score *= torch.min(
                    torch.tensor([1, self.max_action / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
            elif norm_type == 'L2':
                # introduced in TarGF paper
                out_score *= self.max_action / torch.norm(out_score, p=2)
            elif norm_type == 'tanh':
                out_score = self.max_action * torch.tanh(out_score)
            else:
                assert norm_type == 'None'
                out_score = out_score
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score.detach()
