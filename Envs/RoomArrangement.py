import argparse
import os
import random
import time
from numpy.lib.ufunclike import fix
import pybullet as p
import igibson
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipdb import set_trace
from tqdm import tqdm
import pickle
from gym import spaces
from igibson.scene_loader import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

import os
import sys
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
# sys.path.insert(0, pjoin(BASEPATH, '..', '..'))

from Envs.RoomCONSTANTS import bedroom_type_mapping, bedroom_typeidx_mapping, livingroom_type_mapping, livingroom_typeidx_mapping
from room_utils import GraphDataset4RL, split_dataset


class MySimulator:
    def __init__(self, ig_sim, idx, name, room_type='bedroom'):
        if isinstance(ig_sim, dict):
            self.sim = None
            self.cid = None
            self.scene = None
            # build a "fake" objects_by_name
            self.objects_by_name = {}
            metas = ig_sim['meta']
            for index, name in enumerate(metas.keys()):
                self.objects_by_name[name] = DummyObject([index])
            self.wall_obj_lists = None
            self.wall_obj_dicts = None

        else:
            self.sim = ig_sim
            self.cid = self.sim.cid
            self.scene = self.sim.scene
            self.objects_by_name = self.sim.scene.objects_by_name
            self.wall_obj_lists = self.scene.wall_obj_lists
            self.wall_obj_dicts = self.scene.wall_obj_dicts

        self.idx = idx
        self.name = name
        self.categories = ['_'.join(name.split('_')[0:-1]) for name, obj in self.objects_by_name.items()]
        self.bound_x = 0
        self.bound_y = 0
        self.bound_max = 0
        self.center = None
        self.room_type = room_type
        print([name for name, obj in self.objects_by_name.items()])
        print([len(obj.body_ids) for name, obj in self.objects_by_name.items()])
        # save initial condition
        self.inital_checkpoint = {}
        self.proxy_checkpoint = None
        self.type_mapping = bedroom_type_mapping if room_type == 'bedroom' else livingroom_type_mapping
        self.typeidx_mapping = bedroom_typeidx_mapping if room_type == 'bedroom' else livingroom_typeidx_mapping
        self.margin = 0
        self.proxy_height = None

    def change_dynamics(self):
        for name, obj in self.objects_by_name.items():
            for body_id in obj.body_ids:
                p.changeDynamics(body_id, -1, lateralFriction=0., spinningFriction=0., rollingFriction=0.,
                                 restitution=0.98)

    def check_bad(self, obj_num_range, wall_num_range, black_set, white_set,
                  collision_thres=0.01,
                  total_ratio_bound=1.0,
                  single_ratio_bound=1.0,
                  single_side_bound=1.0):
        # black_set: set(['cabinet'])
        # white_set: set(['bed', 'top_cabinet'])
        wall_num = len(self.wall_obj_dicts)
        obj_num = len(self.objects_by_name.keys()) - 2
        if not ((obj_num_range[0] <= obj_num <= obj_num_range[1]) \
                and (wall_num_range[0] <= wall_num <= wall_num_range[1])):
            return False

        # check category
        categories_set = set(['_'.join(key.split('_')[0:-1]) for key in self.objects_by_name.keys()]) - set([''])
        # should not contain obj in black set
        if len(categories_set & black_set) > 0:
            return False
        # must contain obj in white set
        if not white_set <= categories_set:
            return False

        # check size
        total_area = 0
        max_area = 0
        max_side = 0
        for name in self.inital_checkpoint.keys():
            if name in ['walls', 'floors']:
                continue
            bbox = self.inital_checkpoint[name]['size']
            max_side = np.max([max_side, np.max(bbox)])
            cur_area = bbox[0] * bbox[1]
            max_area = np.max([cur_area, max_area])
            total_area += cur_area
        room_area = 4 * (self.bound_x * self.bound_y)
        total_ratio = total_area / room_area
        single_side = max_side / np.max([self.bound_x * 2, self.bound_y * 2])
        single_ratio = max_area / room_area

        if total_ratio > total_ratio_bound or single_ratio > single_ratio_bound or single_side > single_side_bound:
            return False

        if not self.check_stable(collision_thres):
            return False

        return True

    def check_stable(self, collision_thres=0.001, tolerance=20, examination=40):
        # init tolarence
        for _ in range(tolerance):
            p.stepSimulation(physicsClientId=self.cid)

        # get state-prev
        prev = []
        for _, obj in self.objects_by_name.items():
            obj_id = obj.body_ids[0]
            prev.append(p.getBasePositionAndOrientation(obj_id, physicsClientId=self.cid)[0])
        prev = np.array(prev)

        # after simulation
        for _ in range(examination):
            p.stepSimulation(physicsClientId=self.cid)

        # get state-cur
        cur = []
        for _, obj in self.objects_by_name.items():
            obj_id = obj.body_ids[0]
            cur.append(p.getBasePositionAndOrientation(obj_id, physicsClientId=self.cid)[0])
        cur = np.array(cur)

        # check whether changes
        max_delta = np.linalg.norm(prev - cur, axis=-1).max()
        return max_delta <= collision_thres

    def set_state(self, state, wall_feat=None):
        assert state.shape[0] == (len(self.objects_by_name) - 2)
        if wall_feat is not None:
            if wall_feat.shape[0] == 1:
                if np.argmax((np.array([np.tanh(self.bound_x / self.bound_y),
                                        np.tanh(self.bound_y / self.bound_x)]) - wall_feat) ** 2) == 0:
                    # then rotate the room
                    self.augment_room()
            else:
                if np.argmax((self.bound_x - wall_feat) ** 2) == 0:
                    # then rotate the room
                    self.augment_room()
        # state: [num_obj, 2(pos)+2(ori)+2(size)+1(label)]
        state_ = state[:, 0:4]
        label = state[:, -1:]
        # ptr is not aligned with idx, typically, idx = ptr + 2
        ptr = 0
        for idx, (name, obj) in enumerate(self.objects_by_name.items()):
            if name in ['walls', 'floors']:
                continue
            category = '_'.join(name.split('_')[0:-1])
            category_ = self.typeidx_mapping[int(label[ptr][0].item())]
            if category_ != category:
                print(category_)
                print(category)
                set_trace()
            assert category_ == category
            assert self.bound_x > 0 and self.bound_y > 0
            for cur_id in obj.body_ids:
                # cur_id = obj.body_ids[0]
                # clip range
                pos_ = np.clip(state_[ptr][0:2], -1, 1)
                pos = np.array([pos_[0] * self.bound_max, pos_[1] * self.bound_max])
                ori = state_[ptr][2:4]
                # normalize to a normal vector
                ori /= np.linalg.norm(ori)
                ori = np.arctan2(ori[0:1], ori[1:2])
                # import ipdb
                # ipdb.set_trace()

                cur_ori = p.getQuaternionFromEuler([0, 0, ori[0]])
                if self.proxy_checkpoint is None:
                    height = self.inital_checkpoint[name]['pos'][-1]
                else:
                    # height = self.proxy_checkpoint[name][cur_id]['pos'][-1]
                    height = self.proxy_height / 2
                p.resetBasePositionAndOrientation(cur_id, [pos[0], pos[1], height], cur_ori, physicsClientId=self.cid)
                p.resetBaseVelocity(cur_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
            ptr += 1

    def get_state(self):
        # for concerned objects: [pos(2) | ori(2) | size(2) | label(1)]
        # edge_index: fully connected
        obj_feats = []
        for name, obj in self.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            category = '_'.join(name.split('_')[0:-1])
            # get label
            label = self.type_mapping[category]
            label = np.array([label], dtype=np.int64)
            cur_id = obj.body_ids[0]
            # get pos, ori
            cur_pos, cur_ori = p.getBasePositionAndOrientation(cur_id, physicsClientId=self.cid)
            # !! normalize the position
            assert self.bound_x > 0 and self.bound_y > 0
            pos = np.array([cur_pos[0] / self.bound_max, cur_pos[1] / self.bound_max], dtype=np.float32)
            ori = p.getEulerFromQuaternion(cur_ori)
            ori = np.array(ori[-1:], dtype=np.float32)
            # normalize
            ori = np.array([np.sin(ori)[0], np.cos(ori)[0]])
            # there is no API for exact bbox,
            # we can only get axis-aligned bbox here
            # here we normalize the object's ori first and get   the bbox,
            # then turn it back
            axis_align_ori = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(cur_id, cur_pos, axis_align_ori, physicsClientId=self.cid)
            bbox_min, bbox_max = p.getAABB(cur_id, physicsClientId=self.cid)
            p.resetBasePositionAndOrientation(cur_id, cur_pos, cur_ori, physicsClientId=self.cid)
            size = (np.array(bbox_max) - np.array(bbox_min)) / (1 + self.margin)
            size = np.array([size[0] * 2 / self.bound_max - 1, size[1] * 2 / self.bound_max - 1])
            obj_feat = np.concatenate([pos, ori, size, label])
            obj_feats.append(obj_feat)
        # # for walls: [from(2) | to(2) | normal(2)]
        # # edge_index: fully connected, but as wall_dicts is well orderd, we can design a more fine-grained edge_index
        # wall_feats = []
        # for wall in self.wall_obj_dicts:
        #     assert self.center is not None
        #     wall_feat = np.concatenate([(wall['from']-self.center[0:2]) / np.array([self.bound_x, self.bound_y]),
        #                                 (wall['to']-self.center[0:2]) / np.array([self.bound_x, self.bound_y]),
        #                                 wall['normal']])
        #     wall_feats.append(wall_feat)

        # wall_batch = np.stack(wall_feats) # [w, 6]
        # for walls, now we only consider 4-wall, so we only need to know the x/y ratio
        # wall_feat = np.array([self.bound_x, self.bound_y]) # [2]
        wall_feat = np.array([np.tanh(self.bound_x / self.bound_y)])  # [1]
        obj_batch = np.stack(obj_feats)  # [o, 7]

        # return: wall feature, objects batch
        return wall_feat, obj_batch

    def set_constraints(self):
        floor_id = self.objects_by_name['floors'].body_ids[0]
        floor_center, _ = p.getBasePositionAndOrientation(floor_id)
        for name, obj in self.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            cur_id = obj.body_ids[0]
            cur_center, _ = p.getBasePositionAndOrientation(cur_id)
            for jointAxis in [[1, 0, 0], [0, 1, 0]]:
                joint = p.createConstraint(
                    parentBodyUniqueId=floor_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=cur_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_PRISMATIC,
                    jointAxis=jointAxis,
                    parentFramePosition=-np.array(floor_center),
                    childFramePosition=-np.array(cur_center),
                )
                p.changeConstraint(joint, maxForce=0)

    def get_collision_num(self, centralized=True):
        # collision detection
        items = list(self.objects_by_name.items())
        num_objects = len(items)-2
        assert num_objects > 0
        processed_items = []
        for item in items:
            if item[0] in ['walls', 'floors']:
                continue
            processed_items.append(item)
        assert len(processed_items) == num_objects
        collisions = np.zeros((num_objects, num_objects))
        for idx1, (name1, obj1) in enumerate(processed_items[:-1]):
            for idx2, (name2, obj2) in enumerate(processed_items[idx1 + 1:]):
                id_1 = obj1.body_ids[0]
                id_2 = obj2.body_ids[0]
                points = p.getContactPoints(id_1, id_2, physicsClientId=self.cid)
                # cnt += len(points) > 0
                collisions[idx1][idx2] = (len(points) > 0)
                # for debug
                # print(f'{name1} {name2} {len(points)}')
        return np.sum(collisions).item() if centralized else collisions

    def check_valid(self, thres_pos=0.05, thres_ori=0.1, debug=False, is_init=True):
        for name, obj in self.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            height = self.inital_checkpoint[name]['size'][-1]
            for body_id in obj.body_ids:
                cur_pos, cur_ori = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
                tar_height = self.proxy_height / 2 if self.proxy_height is not None else (height + self.margin) / 2
                delta_pos = np.abs(tar_height - cur_pos[-1])
                if delta_pos > thres_pos:
                    if debug:
                        print(f'Pos out of thres! {delta_pos}')
                    return False
                delta_ori = np.abs(cur_ori[0]) + np.abs(cur_ori[1])
                if delta_ori > thres_ori:
                    if debug:
                        print(f'Ori out of thres! roll: {cur_ori[0]}, pitch: {cur_ori[1]}, delta: {delta_ori}')
                    return False
        if is_init:
            collision_num = self.get_collision_num()
            if collision_num > 0:
                if debug:
                    print(f'collision: {collision_num}, failed!')
                return False
        return True

    def regularize_state(self):
        for name, obj in self.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            for body_id in obj.body_ids:
                cur_pos, cur_ori = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
                cur_ori = p.getEulerFromQuaternion(cur_ori)
                regularize_pos = [cur_pos[0], cur_pos[1], self.proxy_height / 2]
                regularize_ori = [0, 0, cur_ori[-1]]
                regularize_ori = p.getQuaternionFromEuler(regularize_ori)
                p.resetBasePositionAndOrientation(body_id, regularize_pos, regularize_ori, physicsClientId=self.cid)

    def set_brownian_velocity(self, scale=0.2, scale_ang=0.1, step_num=100, fixed=[]):
        for _ in range(step_num):
            for name, obj in self.objects_by_name.items():
                if name in ['walls', 'floors']:
                    # for body_id in obj.body_ids:
                    #     p.resetBaseVelocity(body_id, linearVelocity=[0, 0, 0])
                    continue
                if obj.category in fixed:
                    p.resetBaseVelocity(obj.body_ids[0], linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0],
                                        physicsClientId=self.cid)
                    continue
                horizon_v = np.random.normal(size=2) * scale
                angular_v = np.random.normal(size=1) * scale_ang
                vel = [horizon_v[0].item(), horizon_v[1].item(), 0]
                ang_vel = [0, 0, angular_v[0].item()]
                # ang_vel = [0, 0, 0]
                p.resetBaseVelocity(obj.body_ids[0], linearVelocity=vel, angularVelocity=ang_vel,
                                    physicsClientId=self.cid)

            p.stepSimulation(physicsClientId=self.cid)
            self.regularize_state()

    def set_brownian_from_proxy(self, scale=5, scale_ang=0.5, step_num=10000, fixed=['bed']):
        wall_feat, obj_batch = self.get_state()
        proxy = ProxySimulator(self.sim, self.idx, self.name, self.room_type, is_gui=True)
        proxy.set_state(obj_batch, wall_feat)
        proxy.set_brownian_velocity(scale, scale_ang, step_num, fixed)
        wall_feat, obj_batch = proxy.get_state()
        proxy.disconnect()

        # Disconnected
        scene = InteractiveIndoorScene(
            self.name,
            texture_randomization=self.scene.texture_randomization,
            object_randomization=self.scene.object_randomization,
            scene_source=self.scene.scene_source,
            scene_data_type=self.room_type,
            floating_to_planar=self.scene.floating_to_planar,
            category_to_resize_ratio=self.scene.category_to_resize_ratio_by_category,
        )
        self.sim.load()
        self.sim.import_ig_scene(scene)
        p.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=0., cameraPitch=-89.,
                                     cameraTargetPosition=[0, 0, 0])
        self.normalize_room()
        self.set_state(obj_batch, wall_feat)

    def take_snapshot(self, img_size=512, height=20.0):
        viewmatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0.0, height],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
            physicsClientId=self.cid,
        )
        projectionmatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.05,
            farVal=50,
            physicsClientId=self.cid,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            img_size,
            img_size,
            viewMatrix=viewmatrix,
            projectionMatrix=projectionmatrix,
            physicsClientId=self.cid 
        )
        rgb = rgba[:, :, 0:3]
        # print('###### take snapshot success! ######')
        return rgb

    def augment_room(self):
        # !! must augment room after normalizing the room
        # !! and the augmentation is only suitable when wall_num == 4
        # augmentation: rotate all objects 90-degree, clockwise
        # pos: (x, y) -> (y, -x), ori: atan2
        # yaw: z| x->y, so yaw -= pi/2
        for name, obj in self.objects_by_name.items():
            for cur_id in obj.body_ids:
                cur_pos, cur_ori = p.getBasePositionAndOrientation(cur_id, physicsClientId=self.cid)
                aug_pos = np.array([cur_pos[1], -cur_pos[0], cur_pos[2]])
                ori = p.getEulerFromQuaternion(cur_ori)
                ori = np.array(ori)
                ori[-1] += -np.pi / 2
                aug_ori = p.getQuaternionFromEuler(ori)
                p.resetBasePositionAndOrientation(cur_id, aug_pos, aug_ori, physicsClientId=self.cid)
        # bound also change
        t = self.bound_x
        self.bound_x = self.bound_y
        self.bound_y = t

    def flip_room(self):
        # !! must augment room after normalizing the room
        # !! and the augmentation is only suitable when wall_num == 4
        # augmentation: flip the room by y-axis
        # pos: (x, y) -> (y, -x), ori: atan2
        # yaw: z| x->y, so yaw -= pi/2
        for name, obj in self.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            for cur_id in obj.body_ids:
                cur_pos, cur_ori = p.getBasePositionAndOrientation(cur_id, physicsClientId=self.cid)
                # as wall and floor is centered well, but the pos is not (0, 0, 0)
                # aug_pos = cur_pos if name in ['walls', 'floors'] else np.array([cur_pos[0], -cur_pos[1], cur_pos[2]])
                aug_pos = np.array([cur_pos[0], -cur_pos[1], cur_pos[2]])
                aug_ori = p.getEulerFromQuaternion(cur_ori)
                aug_ori = np.array([aug_ori[0], aug_ori[1], np.pi - aug_ori[2]])
                aug_ori = p.getQuaternionFromEuler(aug_ori)
                # ori = p.getEulerFromQuaternion(cur_ori)
                # ori = np.array(ori)
                # ori[-1] = -ori[-1]
                # aug_ori = p.getQuaternionFromEuler(ori)
                p.resetBasePositionAndOrientation(cur_id, aug_pos, aug_ori, physicsClientId=self.cid)
                # print(p.resetBasePositionAndOrientation(cur_id, cur_pos, cur_ori))

    def reset(self, is_initial=True):
        self.reset_checkpoint(is_initial)

    def step(self):
        p.stepSimulation()

    def save_checkpoint(self):
        self.saved_checkpoint = {}
        for name, obj in self.objects_by_name.items():
            body_id = obj.body_ids[0]
            cur_pos, cur_ori = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
            self.saved_checkpoint[name] = {'pos': cur_pos, 'ori': cur_ori}

    def reset_checkpoint(self, is_initial=True):
        if is_initial:
            checkpoint = self.inital_checkpoint
        else:
            checkpoint = self.saved_checkpoint
        for name, obj in self.objects_by_name.items():
            for cur_id in obj.body_ids:
                p.resetBasePositionAndOrientation(cur_id,
                                                  checkpoint[name]['pos'],
                                                  checkpoint[name]['ori'],
                                                  physicsClientId=self.cid)

    def normalize_room(self):
        positions = []
        for wall in self.wall_obj_dicts:
            positions.append(np.array([wall['from'][0], wall['from'][1], 0]))
            positions.append(np.array([wall['to'][0], wall['to'][1], 0]))

        if len(positions) == 0:
            return False
        self.center = sum(positions) / len(positions)
        all_poses = np.stack(positions) - np.expand_dims(self.center, axis=0)
        self.bound_x = np.max(np.abs(all_poses[:, 0:1]))
        self.bound_y = np.max(np.abs(all_poses[:, 1:2]))
        self.bound_max = np.max([self.bound_x, self.bound_y])
        for name, obj in self.objects_by_name.items():
            # remember to operate on each body
            for idx, body_id in enumerate(obj.body_ids):
                # set_trace()
                cur_pos, cur_ori = p.getBasePositionAndOrientation(body_id,
                                                                   physicsClientId=self.cid)
                                                                   
                demean_pos = np.array(cur_pos) - self.center
                p.resetBasePositionAndOrientation(body_id,
                                                  demean_pos,
                                                  cur_ori,
                                                  physicsClientId=self.cid)
                if idx == 0:
                    # get bbox
                    axis_align_ori = p.getQuaternionFromEuler([0, 0, 0])
                    p.resetBasePositionAndOrientation(body_id, cur_pos, axis_align_ori, physicsClientId=self.cid)
                    bbox_min, bbox_max = p.getAABB(body_id, physicsClientId=self.cid)
                    # remember to set back the pos/ori
                    p.resetBasePositionAndOrientation(body_id, demean_pos, cur_ori, physicsClientId=self.cid)
                    size = (np.array(bbox_max) - np.array(bbox_min))
                    self.inital_checkpoint[name] = {'pos': demean_pos, 'ori': cur_ori, 'size': size}

        return True

    def disconnect(self, sleep=1):
        self.sim.disconnect()
        time.sleep(sleep)


class DummyObject:
    def __init__(self, body_ids):
        self.body_ids = body_ids
        self.category = 'dummy'


class ProxySimulator(MySimulator):
    def __init__(self, ig_sim, idx, name, room_type, proxy_dict, is_gui=False):
        super().__init__(ig_sim, idx, name, room_type)
        if isinstance(ig_sim, dict):
            self.inital_checkpoint = ig_sim['meta']
            self.bound_x, self.bound_y = ig_sim['bounds']
            # ''' check validness of ckpt '''
            # for name, obj in self.inital_checkpoint.items():
            #     if name in ['walls', 'floors']:
            #         continue
            #     half_extents = (obj['size'] + self.margin) / 2
            #     valid_flags = np.abs(obj['pos'][0:2]) <= np.array([self.bound_x+0.1, self.bound_y+0.1] - half_extents[0:2])
            #     if np.sum(valid_flags) < 2:
            #         print('invalid pos/size!')
            #         set_trace()
            #     # assert obj['pos'][0] - half_extents[0] >= -(self.bound_x+0.01)
            #     # assert obj['pos'][0] + half_extents[0] <= self.bound_x+0.01
            #     # assert obj['pos'][1] - half_extents[1] >= -(self.bound_y+0.01)
            #     # assert obj['pos'][1] + half_extents[1] <= self.bound_y+0.01

        else:
            super().normalize_room()
            self.disconnect()

        padding = proxy_dict['padding']
        margin = proxy_dict['margin']
        proxy_height = proxy_dict['proxy_height']
        self.is_gui = is_gui
        self.padding = padding
        self.proxy_height = proxy_height

        obj_dicts = self.inital_checkpoint
        proxy_dict = {}
        if self.is_gui:
            self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=8.,
                                         cameraYaw=0.,
                                         cameraPitch=-89.,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)
        else:
            self.cid = p.connect(p.DIRECT)
        my_data_path = './Assets/'

        # plane: 200*200, plane_transparent: 30*30
        plane_base = p.loadURDF(my_data_path + "plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                                physicsClientId=self.cid)
        proxy_dict['floors'] = DummyObject([plane_base])

        # ori: pitch yaw roll
        ori1 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori2 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori3 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        ori4 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])

        # use padding to slightly relax the room
        bound_x = self.bound_x + self.padding
        bound_y = self.bound_y + self.padding
        pos1 = [bound_x, 0, 0]
        pos2 = [-bound_x, 0, 0]
        pos3 = [0, bound_y, 0]
        pos4 = [0, -bound_y, 0]
        plane_name = "plane_transparent.urdf"
        scale_x = bound_y / 2.5
        scale_y = bound_x / 2.5
        transplane1 = p.loadURDF(my_data_path + plane_name, pos1, ori1, globalScaling=scale_x, physicsClientId=self.cid)
        transplane2 = p.loadURDF(my_data_path + plane_name, pos2, ori2, globalScaling=scale_x, physicsClientId=self.cid)
        transplane3 = p.loadURDF(my_data_path + plane_name, pos3, ori3, globalScaling=scale_y, physicsClientId=self.cid)
        transplane4 = p.loadURDF(my_data_path + plane_name, pos4, ori4, globalScaling=scale_y, physicsClientId=self.cid)

        self.bound_x = bound_x
        self.bound_y = bound_y
        self.bound_max = np.max([self.bound_x, self.bound_y])
        self.margin = margin

        proxy_dict['walls'] = DummyObject([transplane1, transplane2, transplane3, transplane4])

        for name, obj in obj_dicts.items():
            if name in ['walls', 'floors']:
                continue
            half_extents = (obj['size'] + self.margin) / 2
            # [(l+margin)/2, (w+margin)/2, h/2]
            half_extents[-1] = self.proxy_height / 2
            collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.cid
            )
            # some bugs when set visual id
            # visual_id = p.createVisualShape(
            #     shapeType=p.GEOM_BOX,
            #     halfExtents=obj['size']/2,
            #     visualFramePosition=[0, 0, 0],
            #     visualFrameOrientation=[0, 0, 0, 0],
            #     physicsClientId=self.cid,
            #     rgbaColor=[0, 0, 1]
            # )
            body_id = p.createMultiBody(
                baseMass=10,
                baseCollisionShapeIndex=collision_id,
                # baseVisualShapeIndex=visual_id,
                basePosition=np.array([obj['pos'][0], obj['pos'][1], self.proxy_height / 2]),
                baseOrientation=obj['ori'],
                physicsClientId=self.cid
            )
            proxy_dict[name] = DummyObject([body_id])
        p.setGravity(0, 0, -10)
        p.setTimeStep(1. / 240)
        self.change_dynamics()
        self.objects_by_name = proxy_dict
        self.remove_collsion()
        # set_trace()
        # while True:
        #     p.stepSimulation()
        self.proxy_checkpoint = {}

    def save_checkpoint(self):
        # clear the prev checkpoint
        self.proxy_checkpoint = {}
        for name, obj in self.objects_by_name.items():
            self.proxy_checkpoint[name] = {}
            for body_id in obj.body_ids:
                cur_pos, cur_ori = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
                self.proxy_checkpoint[name][body_id] = {'pos': cur_pos, 'ori': cur_ori}

    def reset_checkpoint(self):
        for name, obj in self.objects_by_name.items():
            obj_dicts = self.proxy_checkpoint[name]
            for body_id in obj.body_ids:
                body_dict = obj_dicts[body_id]
                p.resetBasePositionAndOrientation(body_id, body_dict['pos'], body_dict['ori'], physicsClientId=self.cid)

    def remove_collsion(self, sim_steps=480):
        for step in range(sim_steps):
            p.stepSimulation(physicsClientId=self.cid)
            self.regularize_state()

    def reset(self, debug=False):
        # large object problem
        flag = False
        try_cnt = 0
        while not flag:
            # set_trace()
            for name, obj in self.objects_by_name.items():
                if name in ['walls', 'floors']:
                    continue
                ratio = 0.9
                rand_x = np.random.uniform(-self.bound_x * ratio, self.bound_x * ratio)
                rand_y = np.random.uniform(-self.bound_y * ratio, self.bound_y * ratio)
                rand_yall = np.random.uniform(-np.pi, np.pi)
                for body_id in obj.body_ids:
                    # height = self.inital_checkpoint[name]['size'][-1]/2
                    height = self.proxy_height / 2
                    cur_ori = p.getQuaternionFromEuler([0, 0, rand_yall])
                    p.resetBasePositionAndOrientation(body_id, [rand_x, rand_y, height], cur_ori,
                                                      physicsClientId=self.cid)
            self.remove_collsion()
            # set_trace()
            # self.set_brownian_velocity(step_num=50)
            # flag = self.check_valid()
            flag = self.check_stable()
            try_cnt += 1
        if debug:
            print(f'success! We try {try_cnt} times to get a valid room!')
        return {'try_cnt': try_cnt}

    def normalize_room(self):
        raise NotImplementedError


class RLEnv:
    def __init__(self, proxy_sim, exp_kwargs, name):
        self.sim = proxy_sim
        self.name = name
        self.pos_rate = exp_kwargs['pos_rate']
        self.ori_rate = exp_kwargs['ori_rate']
        self.max_vel = exp_kwargs['max_vel']
        self.max_horizon = exp_kwargs['max_horizon']
        self.obj_num = len(self.sim.objects_by_name) - 2
        self.cur_step_num = 0
        self.cur_steps = 0

    def step(self, actions, fixed=[]):
        self.cur_steps += 1
        # actions: [obj_num, 3] pos_vel(2)||ori_vel(1)
        # pos_vel: [-max_vel*pos_rate, max_vel*pos_rate] || ori_vel: [-max_vel*ori_rate, max_vel*ori_rate]
        # numpy array
        idx = 0
        actions = actions.reshape(self.obj_num, -1)
        pos_vel = actions[:, :2]
        max_pos_vel_norm = np.max(np.linalg.norm(pos_vel, ord=np.inf, axis=-1))
        scale_factor = self.max_vel * self.pos_rate / max_pos_vel_norm
        scale_factor = np.min([scale_factor, 1])
        actions[:, :2] *= scale_factor

        # set_trace()
        ang_vel = actions[:, 2:]
        max_ang_vel_norm = np.max(np.linalg.norm(ang_vel, ord=np.inf, axis=-1))
        scale_factor = self.max_vel * self.ori_rate / max_ang_vel_norm
        scale_factor = np.min([scale_factor, 1])
        actions[:, 2:] *= scale_factor

        for name, obj in self.sim.objects_by_name.items():
            if name in ['walls', 'floors']:
                continue
            if obj.category in fixed:
                p.resetBaseVelocity(obj.body_ids[0], linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0],
                                    physicsClientId=self.cid)
                continue
            action = actions[idx]
            horizon_v = action[0:2]
            angular_v = action[2:]
            if not (np.abs(horizon_v) <= self.max_vel * self.pos_rate+0.01).all():
                set_trace()
            assert angular_v[0] <= self.max_vel * self.ori_rate+0.01
            assert angular_v.shape[0] == 1
            pos_vel = [horizon_v[0].item(), horizon_v[1].item(), 0]
            ori_vel = [0, 0, angular_v[0].item()]
            p.resetBaseVelocity(obj.body_ids[0], linearVelocity=pos_vel, angularVelocity=ori_vel,
                                physicsClientId=self.sim.cid)
            idx += 1

        # 1/30 sec/step
        collision_num = 0
        for _ in range(8):
            p.stepSimulation(physicsClientId=self.sim.cid)
            collision_num += self.sim.get_collision_num(False)
            self.sim.regularize_state()
        collision_num /= 8.0

        self.cur_step_num += 1
        done = self.cur_step_num >= self.max_horizon
        return self.sim.get_state(), 0, done, {'collision_num': collision_num, 'cur_steps': self.cur_steps}

    def reset(self, flip=True, rotate=True, brownian=True, flip_rotate_params=None):
        if flip:
            if flip_rotate_params is not None:
                flip_num = flip_rotate_params['flip']
            else:
                flip_num = random.randint(0, 1)
            for _ in range(flip_num):
                self.sim.flip_room()
        if rotate:
            if flip_rotate_params is not None:
                rot_num = flip_rotate_params['rotate']
            else:
                rot_num = random.randint(0, 3)
            for _ in range(rot_num):
                self.sim.augment_room()
        if brownian:
            for _ in range(10):
                self.sim.set_brownian_velocity(
                    scale=5,
                    scale_ang=2,
                    step_num=100,
                    fixed=[]
                )
            self.cur_step_num = 0
        p.stepSimulation(physicsClientId=self.sim.cid)
        total_steps = 0
        while self.sim.get_collision_num() > 0:
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.sim.cid)
            total_steps += 20
            if total_steps > 10000:
                print('Warning! Reset takes too much trial!')
                break
        return self.sim.get_state()

    def close(self, sleep=1):
        p.disconnect(physicsClientId=self.sim.cid)
        time.sleep(sleep)
        # self.sim.disconnect()


class RLEnvDynamic:
    # wrap for baseline-rce, we padding the state space to [8, 6] + [1, 6](wall feat), and action space as well
    def __init__(
        self,
        tar_data_name='M12D25_Target_obj38_1_1_tr04_sr05_ss05_bs5_bsa1_bsn0_sbs0_preb0',
        exp_configs={'max_vel': 4, 'pos_rate': 1, 'ori_rate': 1, 'max_horizon': 250},
        meta_name='M12D25_Support_obj38_50_20_tr04_sr05_ss05_bs5_bsa2_bsn1000_sbs5_preb5_bedroom_proxy',
        is_gui=False,
        room_type='bedroom',
        fix_num=None,
        test_seed=0,
        test_ratio=0.1,
        is_single_room = False,
        split='train',
    ):

        self.exp_kwargs = exp_configs
        self.room_type = room_type
        self.is_gui = is_gui
        self.max_vel = exp_configs['max_vel']
        self.pos_rate = exp_configs['max_vel']
        self.ori_rate = exp_configs['ori_rate']
        self.fix_num = fix_num
        self.is_single_room = is_single_room

        ''' init state-action spec '''
        if fix_num is None:
            self.action_space = spaces.Box(-self.max_vel, self.max_vel, shape=(8 * 3,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=((8 + 1 + 1) * 7,), dtype=np.float32)
        else:
            assert 3 <= self.fix_num <= 8
            self.action_space = spaces.Box(-self.max_vel, self.max_vel, shape=(fix_num * 3,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=((fix_num + 1 + 1) * 7,), dtype=np.float32)

        self.tar_dataset = GraphDataset4RL(f'{tar_data_name}')

        ''' set expert dataset '''
        train_dataset, test_dataset, infos_dict = split_dataset(self.tar_dataset, seed=test_seed, test_ratio=test_ratio)
        # set_trace()
        # self.exp_data = []
        # for state in train_dataset:
        #     prepro_data = self.prepro_state(state)
        #     if self.fix_num is not None:
        #         if prepro_data.shape[0] == 7 * (self.fix_num + 1):  # +1 是因为有wall feat concat在里面！
        #             self.exp_data.append(prepro_data)
        #     else:
        #         self.exp_data.append(prepro_data)
        # self.exp_data = np.stack(self.exp_data)

        # len(self.room_metas_dict.keys()) == 903
        ''' set init states dataset '''
        with open(f'../ExpertDatasets/{meta_name}.pickle', 'rb') as f:
            self.room_metas_dict = pickle.load(f)

        # self.room_names = self.tar_dataset.folders_path
        # total: 839 rooms
        # test: 83 rooms
        # train: 756 rooms
        self.room_names = []
        dataset = train_dataset if split == 'train' else test_dataset
        for state in dataset:
            _, _, room_name = state
            self.room_names.append(room_name)
        self.room_names = list(set(self.room_names))
        self.room_names.sort()
        for room_name in self.room_names:
            assert room_name in self.room_metas_dict.keys()
        
        # # M5D4: test small room number
        # if not self.is_single_room:
        #     self.room_names = self.room_names[:5]

        self.proxy_dict = {
            'padding': 0.0,
            'margin': 0.0,
            'proxy_height': 0.5,
        }
        self.sim = None
        self.episode_cnt = 0
        self.scene_sampler = SceneSampler(self.room_type, 'DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
        self.name = None
        self.idx = 0
    
    def seed(self, seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    @property
    def obj_number(self):
        return self.sim.obj_num

    def reset(self, room_name=None, goal=None, flip=True, rotate=True, brownian=True, is_random_sample=True, flip_rotate_params=None):
        self.episode_cnt += 1
        if self.sim is not None:
            self.sim.close()
        self.sim = self.sample(room_name, is_random_sample=is_random_sample, margin=0.1)
        room_name = self.sim.name
        if goal is not None:
            self.sim.sim.set_state(goal[-1], wall_feat=goal[0])
        state = self.sim.reset(flip, rotate, brownian, flip_rotate_params)  # [obj_num, 7] -> [8+1, 7]
        # state = self.sim.reset(False, False, False)  # [obj_num, 7] -> [8+1, 7]
        # state = self.sim.reset(False, False, brownian)  # [obj_num, 7] -> [8+1, 7]
        self.sim.close()
        self.sim = self.sample(room_name, 0.0)
        self.sim.sim.set_state(state[1], state[0])

        print(f'No.{self.episode_cnt} Episode Success!')
        return state
    
    def close(self):
        self.sim.close()
        self.sim = None

    def step(self, actions):
        next_state, reward, is_done, infos = self.sim.step(actions)
        return next_state, reward, is_done, infos

    def sample(self, room_name=None, is_random_sample=True, margin=0.0):
        if room_name is None:
            if self.is_single_room:
                name = '3a19c7bb-bc35-44e3-8d48-4fb00b9789ff'
            else:
                if is_random_sample:
                    name = random.sample(self.room_names, 1)[0]
                else:
                    # iterate in order, test phase
                    name = self.room_names[self.idx]
                    self.idx = (self.idx + 1) % len(self.room_names)
        else:
            name = room_name
        room_meta = self.room_metas_dict[name]
        if self.fix_num is not None:
            flag = (len(room_meta['meta']) - 2 != self.fix_num)
            while flag:
                name = random.sample(self.room_names, 1)[0]
                room_meta = self.room_metas_dict[name]
                flag = (len(room_meta['meta']) - 2 != self.fix_num)
        # room_meta = self.scene_sampler[name]
        proxy_dict = self.proxy_dict
        proxy_dict['margin'] = margin
        sim = ProxySimulator(room_meta, 0, name, self.room_type, proxy_dict, is_gui=self.is_gui)
        self.name = name
        return RLEnv(sim, self.exp_kwargs, name=name)

    def sample_action(self):
        action_ = np.random.uniform(-1, 1, size=(self.obj_number, 3))
        action_[:, 0:2] = action_[:, 0:2] * self.max_vel * self.pos_rate
        action_[:, 2:3] = action_[:, 2:3] * self.max_vel * self.ori_rate
        return action_.flatten()


class RLEnvFull:
    # wrap for baseline-rce, we padding the state space to [8, 6] + [1, 6](wall feat), and action space as well
    def __init__(
            self,
            tar_data_name='UnshuffledRoomMeta',
            exp_configs={'max_vel': 4, 'pos_rate': 1, 'ori_rate': 1, 'max_horizon': 250},
            meta_name='ShuffledRoomMeta',
            is_gui=False,
            room_type='bedroom',
            fix_num=None,
            test_seed=0,
            test_ratio=0.2,
            is_single_room=False,
    ):
        self.max_vel = exp_configs['max_vel']
        self.pos_rate = exp_configs['max_vel']
        self.ori_rate = exp_configs['ori_rate']
        self.is_single_room = is_single_room
        # self.action_space = spaces.Box(-self.max_vel, self.max_vel, shape=(8, 3), dtype=np.float32)
        # self.observation_space = spaces.Box(-1, 1, shape=(9, 7), dtype=np.float32)
        self.fix_num = fix_num
        if fix_num is None:
            self.action_space = spaces.Box(-self.max_vel, self.max_vel, shape=(8 * 3,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=((8 + 1 + 1) * 7,), dtype=np.float32)
        else:
            assert 3 <= self.fix_num <= 8
            self.action_space = spaces.Box(-self.max_vel, self.max_vel, shape=(fix_num * 3,), dtype=np.float32)
            self.observation_space = spaces.Box(-1, 1, shape=((fix_num + 1 + 1) * 7,), dtype=np.float32)
        self.exp_kwargs = exp_configs
        self.room_type = room_type
        self.is_gui = is_gui
        self.tar_dataset = GraphDataset4RL(f'{tar_data_name}_{room_type}')

        train_dataset, test_dataset, infos_dict = split_dataset(self.tar_dataset, seed=test_seed, test_ratio=test_ratio)

        self.exp_data = []
        for state in train_dataset:
            prepro_data = self.prepro_state(state)
            if self.fix_num is not None:
                if prepro_data.shape[0] == 7 * (self.fix_num + 1):  
                    self.exp_data.append(prepro_data)
            else:
                self.exp_data.append(prepro_data)
        self.exp_data = np.stack(self.exp_data)

        self.room_names = []
        for state in test_dataset:
            _, _, room_name = state
            self.room_names.append(room_name)
        self.room_names.sort()

        # make sure all the room names are contained in metadatas
        with open(f'../ExpertDatasets/{meta_name}.pickle', 'rb') as f:
            self.room_metas_dict = pickle.load(f)
        for room_name in self.room_names:
            assert room_name in self.room_metas_dict.keys()

        # self.room_names = self.tar_dataset.folders_path
        self.proxy_dict = {
            'padding': 0.0,
            'margin': 0.0,
            'proxy_height': 0.5,
        }
        self.sim = None
        self.episode_cnt = 0

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def prepro_state(self, state):
        max_obj_num = 8 if self.fix_num is None else self.fix_num
        wall_feat = state[0]
        obj_feat = state[1]
        n_obj, feat_dim = obj_feat.shape
        state = obj_feat
        if self.fix_num is None:
            state = np.vstack([state, np.zeros((max_obj_num - n_obj, feat_dim))])
        state = np.vstack([state, np.tile(n_obj, (1, feat_dim))])
        state = np.vstack([state, np.tile(wall_feat, (1, feat_dim))])

        return state.flatten()

    def prepro_action(self, actions):
        return actions[0:self.sim.obj_num * 3].reshape(self.sim.obj_num, 3)

    def reset(self):
        self.episode_cnt += 1
        if self.sim is not None:
            self.sim.close()
        self.sim = self.sample()
        state = self.sim.reset()  # [obj_num, 7] -> [8+1, 7]
        print(f'No.{self.episode_cnt} Episode Success!')
        return self.prepro_state(state)

    def step(self, actions):
        actions_ = self.prepro_action(actions)  # [8, 3] -> [obj_num, 3]
        next_state, reward, is_done, infos = self.sim.step(actions_)
        return self.prepro_state(next_state), reward, is_done, infos

    def get_dataset(self, num_obs=256):
        # # sample a batch of random actions
        # action_vec = [self.sample_action() for _ in range(num_obs)]
        # # sample a batch of example states
        # ind = np.random.randint(0, self.exp_data.shape[0], size=(num_obs,))
        # obs_vec = self.exp_data[ind]
        # dataset = {
        #     'observations': np.array(obs_vec, dtype=np.float32),
        #     'actions': np.array(action_vec, dtype=np.float32),
        #     'rewards': np.zeros(num_obs, dtype=np.float32),
        # }
        num_obs_full = self.exp_data.shape[0]
        num_obs = num_obs_full
        # # sample a batch of random actions
        action_vec = [self.sample_action() for _ in range(num_obs)]
        # sample a batch of example states
        obs_vec = self.exp_data
        dataset = {
            'observations': np.array(obs_vec, dtype=np.float32),
            'actions': np.array(action_vec, dtype=np.float32),
            'rewards': np.zeros(num_obs, dtype=np.float32),
        }
        return dataset

    def sample(self, room_name=None):
        if room_name is None:
            if self.is_single_room:
                name = '3a19c7bb-bc35-44e3-8d48-4fb00b9789ff'
            else:
                name = random.sample(self.room_names, 1)[0]
        else:
            name = room_name
        room_meta = self.room_metas_dict[name]
        if self.fix_num is not None:
            flag = (len(room_meta['meta']) - 2 != self.fix_num)
            while flag:
                name = random.sample(self.room_names, 1)[0]
                room_meta = self.room_metas_dict[name]
                flag = (len(room_meta['meta']) - 2 != self.fix_num)
        # room_meta = self.scene_sampler[name]

        sim = ProxySimulator(room_meta, 0, name, self.room_type, self.proxy_dict, is_gui=self.is_gui)
        self.name = name
        return RLEnv(sim, self.exp_kwargs, name=name)

    # def sample(self):
    #     name = random.sample(self.room_names, 1)[0]
    #     room_meta = self.room_metas_dict[name]
    #     if self.fix_num is not None:
    #         flag = (len(room_meta['meta']) - 2 != self.fix_num)
    #         while flag:
    #             name = random.sample(self.room_names, 1)[0]
    #             room_meta = self.room_metas_dict[name]
    #             flag = (len(room_meta['meta']) - 2 != self.fix_num)

    #     sim = ProxySimulator(room_meta, 0, name, self.room_type, self.proxy_dict, is_gui=self.is_gui)
    #     return RLEnv(sim, self.exp_kwargs, name=name)

    def sample_action(self):
        action_ = np.random.normal(size=(8 if self.fix_num is None else self.fix_num, 3)) * self.max_vel
        action_[:, 0:2] = np.clip(action_[:, 0:2], -self.max_vel * self.pos_rate, self.max_vel * self.pos_rate)
        action_[:, 2:3] = np.clip(action_[:, 2:3], -self.max_vel * self.ori_rate, self.max_vel * self.ori_rate)
        return action_.flatten()


class SceneSampler:
    def __init__(self,
                 room_type,
                 gui='pbgui',
                 scene_source="THREEDFRONT",
                 floating_to_planar=False,
                 mode='normal',
                 proxy_dict={'padding': 0, 'margin': 0, 'proxy_height': 0.5},
                 resize_dict={'bed':0.8, 'shelf':0.8}):
        self.room_type = room_type
        self.gui = gui
        self.scene_source = scene_source
        data_root_path = igibson.threedfront_dataset_path
        data_type_path = os.path.join(data_root_path, self.room_type)
        self.name_list = os.listdir(data_type_path)
        
        self.name_dict = {name: idx for idx, name in enumerate(self.name_list)}
        self.floating_to_planar = floating_to_planar
        self.mode = mode
        self.resize_dict = {} if resize_dict is None else resize_dict
        self.proxy_dict = proxy_dict

    def sample(self):
        thres = 100
        cnt = 0
        while True:
            cnt += 1
            idx = random.randint(0, len(self.name_list))
            sim = self.get_scene(idx)
            if sim:
                return sim
            if cnt > thres:
                print(f'###### Scene Sampling Error! All the scenes are invalid after {thres} trails!')
                return None

    def get_name(self, idx):
        if 0 <= idx <= len(self.name_list):
            return self.name_list[idx]
        else:
            print('index error when getting name!')
            return None

    def get_scene(self, idx):
        # set_trace()
        scene_name = self.name_list[idx]
        # print('get scene name success')
        # s = None
        # try:
        scene = InteractiveIndoorScene(
                        scene_name,
                        texture_randomization=False,
                        object_randomization=False,
                        scene_source=self.scene_source,
                        scene_data_type=self.room_type,
                        floating_to_planar=self.floating_to_planar,
                        category_to_resize_ratio=self.resize_dict,
        )
        # print('create scene success')
        s = Simulator(mode=self.gui)
        # print('create Simulator success')

        s.import_ig_scene(scene)
        # print('import scene success')

        p.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=0., cameraPitch=-89., cameraTargetPosition=[0, 0, 0])
        # print('reset camera success')
        ts = time.time()
        sim = MySimulator(s, idx, self.name_list[idx], self.room_type) if self.mode == 'normal' \
            else ProxySimulator(s, idx, self.name_list[idx], self.room_type, proxy_dict=self.proxy_dict, is_gui=(self.gui=='pbgui'))
        print(time.time() - ts)
        return sim
        # except Exception as e:
        #     print(f'No.{idx}: {e}')
        #     if s:
        #         s.disconnect()
        #     return None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_scene(idx)
        elif isinstance(idx, str):
            return self.get_scene(self.name_dict[idx])
        else:
            print('index type error!')
            return None


if __name__ == '__main__':
    tar_data = 'M12D25_Target_obj38_1_1_tr04_sr05_ss05_bs5_bsa1_bsn0_sbs0_preb0'
    exp_kwargs = {'max_vel': 4, 'pos_rate': 1, 'ori_rate': 1, 'max_horizon': 250}
    # rlenv = RLEnvFull(
    #     tar_data,
    #     exp_kwargs,
    #     meta_name='M12D25_Support_obj38_50_20_tr04_sr05_ss05_bs5_bsa2_bsn1000_sbs5_preb5_bedroom_proxy',
    #     is_gui=False
    # )
    rlenv = RLEnvFull()

    with tqdm(total=len(rlenv.room_names)) as pbar:
        while True:
            cur_state = rlenv.reset()  # cur_state = (wall_feat: (1,), obj_feat: (6, 7))
            # img = /
            # set_trace()
            done = False
            while not done:
                action = rlenv.sample_action()
                _, _, done, _ = rlenv.step(action)
                # time.sleep(0.01)
            pbar.update(1)


# class RLEnvSampler:
#     def __init__(self, tar_data_name, exp_configs, is_gui=False, room_type='bedroom'):
#         self.exp_kwargs = exp_configs
#         tar_dataset = GraphDataset(f'{tar_data_name}_{room_type}')
#         self.room_names = tar_dataset.folders_path
#         proxy_dict = {
#             'padding': 0.0,
#             'margin': 0.0,
#             'proxy_height': 0.5,
#         }
#         self.sampler = SceneSampler(room_type, proxy_dict=proxy_dict, gui='pbgui' if is_gui else 'DIRECT', resize_dict={'bed': 0.8, 'shelf': 0.8})
#         # self.sampler.mode = 'proxy'
#
#     def sample(self):
#         sim = self.sampler[random.sample(self.room_names, 1)[0]]
#         return RLEnv(sim, self.exp_kwargs)
