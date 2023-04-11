import numpy as np
import torch
import functools
from ipdb import set_trace

from planners.orca.pyorca import Agent, orca
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalise_vels(vels, max_vel):
    # [-inf, +inf] -> [-max_vel, max_vel]**n
    vels = np.array(vels).copy()
    max_vel_norm = np.max(np.abs(vels))
    scale_factor = max_vel / (max_vel_norm+1e-7)
    scale_factor = np.min([scale_factor, 1])
    vels_updated = scale_factor * vels
    return vels_updated

# ORCA Agent
class TarGFORCAPlanner:
    def __init__(self, 
        targf, 
        configs,
        radius=0.025,
        max_vel=0.3, 
        dt=0.02, 
        tau=0.1, 
        horizon=100, 
    ):
        self.targf = targf
        self.num_objs = configs.num_objs
        self.max_vel = max_vel
        self.radius = radius
        self.is_decay = configs.is_decay_t0_orca
        self.agents = []
        self.t0 = configs.orca_t0
        self.knn = configs.knn_orca
        self.dt = dt
        self.tau = tau

        for _ in range(self.num_objs):
            self.agents.append(Agent(np.zeros(2), (0., 0.), self.radius, max_vel, np.zeros(2)))
        self.goal_state = None

        self.horizon = horizon
        self.cur_time_step = 0

    def reset_policy(self):
        self.cur_time_step = 0

    def assign_tar_vels(self, vels):
        for agent, vel in zip(self.agents, vels):
            agent.pref_velocity = np.array(vel)

    def assign_pos(self, positions):
        for agent, position in zip(self.agents, positions):
            agent.position = np.array(position)

    def get_tar_vels(self, inp_state):
        if self.is_decay:
            t0 = self.t0*(self.horizon - self.cur_time_step + 1e-3) / self.horizon
        else:
            t0 = self.t0
        tar_vels = self.targf.inference(inp_state, t0, is_numpy=True, is_norm=True, empty=False)
        tar_vels = tar_vels.reshape((-1, 2))
        return tar_vels

    def select_action(self, inp_state, infos=None, sample=True):
        """
        inp_state: nparr, (3*num_objs, )
        infos, sample: placeholders
        """
        self.cur_time_step += 1
        # get and assign tar vels
        tar_vels = self.get_tar_vels(inp_state)
        self.assign_tar_vels(tar_vels)

        # assign positions
        positions = inp_state.reshape((-1, 3))[:, :2]
        self.assign_pos(positions)
        new_vels = [None] * len(self.agents)

        # ORCA: compute the optimal vel to avoid collision
        for i, agent in enumerate(self.agents):
            candidates = self.agents[:i] + self.agents[i + 1:]

            def my_comp(x, y):
                x_dist = np.sum((x.position - agent.position) ** 2)
                y_dist = np.sum((y.position - agent.position) ** 2)
                return x_dist - y_dist

            k_nearest_candidates = sorted(candidates, key=functools.cmp_to_key(my_comp))[0:self.knn]
            new_vels[i], _ = orca(agent, k_nearest_candidates, self.tau, self.dt)
        
        # project the vels into the action space
        new_vels = normalise_vels(new_vels, self.max_vel)

        for i, agent in enumerate(self.agents):
            agent.velocity = new_vels[i]
        return np.array(new_vels).reshape(-1)
    
