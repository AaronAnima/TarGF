import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import trange
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

from envs.Room.RoomArrangement import SceneSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# eval & visualisation func for room rearrangement
def training_time_eval_room(eval_env, eval_policy, writer, eval_episodes, eval_idx, reward_func=None, visualise=True):
    episode_reward = 0
    episode_similarity_reward = 0
    episode_collision_reward = 0
    # collect meta-datas for follow-up visualisations
    vis_states = []
    room_names = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        cur_vis_states = []
        cur_vis_states.append(state)
        while not done:
            action = eval_policy.select_action(state, sample=False)
            next_state, _, done, infos = eval_env.step(action)
            reward, reward_similarity, reward_collision = reward_func.get_reward(info=infos, cur_state=state, new_state=next_state, is_eval=True)
            state = next_state 
            episode_reward += reward.sum().item()
            episode_similarity_reward += reward_similarity.sum().item()
            episode_collision_reward += reward_collision.sum().item()
        cur_vis_states.append(state)
        room_names.append(eval_env.sim.name)
        vis_states.append(cur_vis_states)
    
    episode_reward /= eval_episodes
    episode_similarity_reward /= eval_episodes
    episode_collision_reward /= eval_episodes

    ''' log eval metrics '''
    writer.add_scalars('Eval/Compare',
                       {'total': episode_reward,
                        'collision': episode_collision_reward,
                        'similarity': episode_similarity_reward},
                        eval_idx)
    writer.add_scalar('Eval/Total', episode_reward, eval_idx)


    ''' visualise the terminal states '''
    if visualise:
        # the real-sense image can only be rendered by scene-sampler (instead of proxy simulator for RL)
        eval_env.close()
        sampler = SceneSampler(gui='DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
        imgs = []
        for state, room_name in zip(vis_states, room_names):
            # order: GT -> init -> terminal
            sim = sampler[room_name]
            sim.normalize_room()

            # vis GT state
            img = sim.take_snapshot(512, height=10.0)
            imgs.append(img)

            # vis init/terminal state
            for state_item in state:
                sim.set_state(state_item[1], state_item[0])
                img = sim.take_snapshot(512, height=10.0)
                imgs.append(img)

            # close env after rendering
            sim.disconnect()
        batch_imgs = np.stack(imgs, axis=0)
        ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
        grid = make_grid(ts_imgs.float(), padding=2, nrow=3, normalize=True)
        writer.add_image(f'Images/igibson_gt_terminal_init', grid, eval_idx)

# eval & visualisation func for ball rearrangement
def training_time_eval_ball(eval_env, eval_policy, writer, eval_episodes, eval_idx, pdf_func=None, nrow=4):
    horizon = eval_env.max_episode_len
    episode_delta_likelihoods = []
    episode_avg_collisions = []
    last_states = []
    pdf = pdf_func
    for _ in trange(eval_episodes):
        state, done = eval_env.reset(is_random=True), False
        if isinstance(state, OrderedDict):
            state = eval_env.flatten_states([state])[0]
        avg_collision = 0
        delta_likelihoods = -np.log(pdf([state]))
        while not done:
            action = eval_policy.select_action(np.array(state), sample=False)
            state, _, done, infos = eval_env.step(action)
            if isinstance(state, OrderedDict):
                state = eval_env.flatten_states([state])[0]
            collisions = infos['collision_num']
            avg_collision += np.sum(collisions)
        last_states.append(state)
        avg_collision /= horizon
        delta_likelihoods += np.log(pdf([state]))
        episode_delta_likelihoods.append(delta_likelihoods)
        episode_avg_collisions.append(avg_collision)

    # Declaration
    mu_dl, std_dl = np.mean(episode_delta_likelihoods), np.std(episode_delta_likelihoods)
    mu_ac, std_ac = np.mean(episode_avg_collisions), np.std(episode_avg_collisions)
    print('----Delta Likelihood: {:.2f} +- {:.2f}'.format(mu_dl, std_dl))
    print('----Avg Collisions: {:.2f} +- {:.2f}'.format(mu_ac, std_ac))

    writer.add_scalars('Eval/Delta_Likelihood', {
        'upper': mu_dl + std_dl,
        'mean': mu_dl,
        'lower': mu_dl - std_dl,
    }, 
    eval_idx)
    writer.add_scalars('Eval/Average_Collision', {
        'upper': mu_ac + std_ac,
        'mean': mu_ac,
        'lower': mu_ac - std_ac,
    }, eval_idx)

    eval_states = last_states[:nrow**2]
    imgs = []
    for obj_state in eval_states:
        if not isinstance(obj_state, np.ndarray):
            obj_state = obj_state.detach().cpu().numpy()
        if not isinstance(obj_state, OrderedDict):
            obj_state = eval_env.unflatten_states([obj_state])[0]
        eval_env.set_state(obj_state)
        img = eval_env.render(img_size=256)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    writer.add_image(f'Images/eval_terminal_states', grid, eval_idx)



# ball rearrangement, coverage score
def chamfer_dist(x, y, metric='l1'):
    x = x.reshape(-1, 2)
    y = y.reshape(-1, 2)
    # x: [nx, dim], y: [ny, dim]
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    return dist

def my_dist_ball(state1, state2, num_objs, is_category=True):
    # remove labels
    state1 = state1.reshape((num_objs, -1))[:, :2].reshape(-1)
    state2 = state2.reshape((num_objs, -1))[:, :2].reshape(-1)

    # !! overfit to the paper's setting, where there are 3 classes
    num_per_class = num_objs // 3

    dist = 0
    if is_category:
        dist += chamfer_dist(state1[0:num_per_class*2].reshape(-1, 2), state2[0:num_per_class*2].reshape(-1, 2))
        dist += chamfer_dist(state1[num_per_class*2:num_per_class*4].reshape(-1, 2), state2[num_per_class*2:num_per_class*4].reshape(-1, 2))
        dist += chamfer_dist(state1[num_per_class*4:num_per_class*6].reshape(-1, 2), state2[num_per_class*4:num_per_class*6].reshape(-1, 2))
    else:
        dist += chamfer_dist(state1.reshape(-1, 2), state2.reshape(-1, 2))
    return dist

def coverage_score_ball(gt_states, states, num_objs, who_cover='gen', is_category=True):
    min_dists = []
    # who_cover == 'gt': each gen find nearest gt， who_cover == ‘gen’: vice versa
    cover_states = gt_states if who_cover == 'gt' else states
    covered_states = states if who_cover == 'gt' else gt_states
    for covered_state in covered_states:
        min_dist = np.min(np.array([my_dist_ball(covered_state, cover_state, num_objs, is_category=is_category) for cover_state in cover_states]))
        min_dists.append(min_dist)
    return np.mean(np.array(min_dists))

# room rearrangement, coverage score
def my_dist_room(state1, state2):
    # we assume:
    # state1: [num_obj, 7]
    # state2: [num_obj, 7]
    return np.sum((state1[:, 0:2] - state2[:, 0:2])**2)

def coverage_score_room(room_name_to_gt_states, room_name_to_states):
    res = []
    room_names = list(room_name_to_gt_states.keys())
    for room_name in room_names:
        gt_states = room_name_to_gt_states[room_name]
        states = room_name_to_states[room_name]
        min_dists = []
        for gt_state in gt_states:
            min_dist = np.min(np.array([my_dist_room(gt_state, state) for state in states]))
            min_dists.append(min_dist)
        res.append(np.min(np.array(min_dists))) 
    res = np.array(res)
    return np.mean(res)

