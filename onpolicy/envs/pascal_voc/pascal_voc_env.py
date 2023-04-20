#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from onpolicy.envs.traffic_junction.traffic_helper import *
from inspect import getargspec
from copy import deepcopy

#Pascal Voc modules
from gluoncv import data, utils
from mxnet.gluon.data.vision import transforms
import mxnet as mx
# from matplotlib import pyplot as plt
import os

def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class PascalVocEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.__version__ = "0.0.1"
        self.name  = "PascalVoc"
        #print("init traffic junction", getargspec(self.reset).args)
        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        self.episode_over = False
        self.has_failed = 0

        self.resize = True #IMPORTANT IF YOU WANT TO RESIZE IMAGES
        self.ctx = mx.cpu(0)
        self.multi_agent_init(args)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.ncar):
            self.action_space.append(spaces.Discrete(self.naction))
            self.observation_space.append(self.get_obs_size())
            global_state_size = self.get_obs_size()
            # global_state_size.insert(0, self.ncar)
            global_state_size[0] *= self.ncar
            self.share_observation_space.append(global_state_size)
        if self.resize == True:
            # transformer = transforms.Resize(size=(32, 32))
            self.train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
            self.val_dataset = data.VOCDetection(splits=[(2007, 'test')])
        else:
            self.train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
            self.val_dataset = data.VOCDetection(splits=[(2007, 'test')])
        #print(os.getcwd())
        # self.train_idxs = np.loadtxt('./onpolicy/envs/pascal_voc/train_idxs.txt')
        self.train_idxs = np.loadtxt('../envs/pascal_voc/train_idxs.txt')
        self.agent_train_idxs = np.vstack((np.random.permutation(self.train_idxs), np.random.permutation(self.train_idxs)))
        self.curr_iteration_idx = 0
        self.curr_idx = self.agent_train_idxs[:, self.curr_iteration_idx]
        self.label_dict_reversed = {
            1: 0,
            6: 1,
            11: 2,
            14: 3,
            17: 4
        }
        self.label_dict = {
            0: [1, 6],
            1: [1, 11],
            2: [1, 14],
            3: [1, 17],
            4: [6, 11],
            5: [6, 14],
            6: [6, 17],
            7: [11, 14],
            8: [11, 17],
            9: [14, 17]
        }

    def multi_agent_init(self, args):
        #print("init tj")
        # General variables defining the environment : CONFIG
        params = ['vision', 'curr_start_epoch', 'curr_epochs', 'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))

        if self.difficulty == 'easy':
            self.dim = 6
            self.add_rate_min = 0.1
            self.add_rate_max = 0.3
            assert args.num_agents == 2

        elif self.difficulty == 'medium':
            self.dim = 14
            self.add_rate_min = 0.05
            self.add_rate_max = 0.2
            assert args.num_agents == 2

        elif self.difficulty == 'hard':
            self.dim = 18
            self.add_rate_min = 0.02
            self.add_rate_max = 0.05
            assert args.num_agents == 2

        else:
            raise RuntimeError("Difficulty key error")

        self.ncar = args.num_agents
        self.dims = dims = (375, 500) if self.resize == False else (32, 32)
        difficulty = args.difficulty
        vision = args.vision

        # if difficulty in ['medium','easy','longer_easy']:
        #     assert dims[0]%2 == 0, 'Only even dimension supported for now.'

        #     assert dims[0] >= 4 + vision, 'Min dim: 4 + vision'

        # if difficulty == 'hard':
        #     assert dims[0] >= 9, 'Min dim: 9'
        #     assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.naction = 10 if difficulty in ['medium','easy','longer_easy'] else 10
        self._action_space = spaces.Discrete(self.naction)

        # make no. of dims odd for easy case.
        # if difficulty == 'easy' or difficulty == 'longer_easy':
        #     self.dims = list(dims)
        #     for i in range(len(self.dims)):
        #         self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8,
                'longer_easy':6}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum,
                'longer_easy': dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            self._observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    # spaces.Discrete(self.npath),
                                    # spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size)),
                                    spaces.Box(0, 1, [375, 500, 3])))
            if self.resize == True:
                self._observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    # spaces.Discrete(self.npath),
                                    # spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size)),
                                    spaces.Box(0, 1, [32, 32, 3])))
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self._observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    # spaces.Discrete(self.npath),
                                    # spaces.MultiDiscrete(dims),
                                    # spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size)),
                                    spaces.Box(0, 1, [375, 500, 3])))
            if self.resize == True:
                self._observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    # spaces.Discrete(self.npath),
                                    # spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size)),
                                    spaces.Box(0, 1, [32, 32, 3])))
            # Actual observation will be of the shape 1 * ncar * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        # self._set_grid()

        # if difficulty == 'easy' or difficulty == 'longer_easy':
        #     self._set_paths_easy()
        # else:
        #     self._set_paths(difficulty)

        return

    def get_obs_size(self):
        """Returns the size of the observation."""
        if hasattr(self._observation_space, 'spaces'):
            if self.vocab_type == 'bool':
                total_obs_dim = [int(np.prod(self._observation_space.spaces[-1].shape))]
            else:
                total_obs_dim = [int(np.prod(self._observation_space.spaces[-1].shape))]
            return total_obs_dim
        else:
            return self._observation_space.shape


    def reset(self, epoch=None, success=False):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = self.curr_epochs
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        # print("reached first step", epoch, epoch_range, add_rate_range, self.epoch_last_update)
        if success and self.curr_start_epoch == -1:
            self.curr_start_epoch = epoch
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update and self.curr_start_epoch != -1:
            # print("running curriculum now")
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()
        available_actions = np.array([self.ncar * [True, True, True, True, True]*2]).reshape(self.ncar, 10)
        s_ob = self.get_state(obs)
        return obs, s_ob, available_actions

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (ncar x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        infos = [{} for i in range(self.ncar)]

        # Expected shape: either ncar or ncar x 1
        action = np.array(action).squeeze()

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            self._take_action(i, a)

        # self._add_cars()

        obs = self._get_obs()
        reward = self._get_reward()
        reward = np.expand_dims(reward, -1)

        debug = {'car_loc':self.car_loc,
                'alive_mask': np.copy(self.alive_mask),
                'wait': self.wait,
                'cars_in_sys': self.cars_in_sys,
                'is_completed': np.copy(self.is_completed)}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        # return obs, reward, self.episode_over, debug
        global_state = self.get_state(obs)
        local_obs = obs
        dones = np.zeros((self.ncar), dtype=bool)
        for i in range(self.ncar):
            infos[i] = self.stat.copy()
            dones[i] = self.has_failed == 1
        available_actions = np.array([self.ncar * [True, True, True, True, True]*2]).reshape(self.ncar, 10)
        # Next image
        self.curr_iteration_idx += 1
        if self.curr_iteration_idx == self.agent_train_idxs.shape[1]:
            self.curr_iteration_idx = 0
            self.agent_train_idxs = np.vstack((np.random.permutation(self.train_idxs), np.random.permutation(self.train_idxs)))
        self.curr_idx = self.agent_train_idxs[:, self.curr_iteration_idx]

        return local_obs, global_state, reward, dones, infos, available_actions

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed

    def _get_obs(self):
        h, w = self.dims


        obs = []
        for i, p in enumerate(self.curr_idx):
            # most recent action
            act = self.car_last_act[i] #/ (self.naction - 1)

            # provide current images
            train_image, train_label = self.train_dataset[int(p)]
            train_image = mx.ndarray.cast(train_image, dtype='float32')/255.0
            train_image = data.transforms.presets.segmentation.test_transform(train_image, self.ctx)
            train_image = train_image[0].reshape(375, 500, 3)
            if self.resize == True:
                train_image = data.transforms.image.imresize(src=train_image, w=32, h=32)
            train_image = train_image.asnumpy()

            if self.vocab_type == 'bool':
                o = tuple((#act,
                train_image))
                # if i == 0: print(i, act, r_i, v_sq)
            else:
                o = tuple((#act,
                train_image))
            obs.append(np.concatenate(o, axis=None))

        obs = np.array(obs)
        return obs

    def get_state(self, local_obs):
        n, s = local_obs.shape
        return np.tile(local_obs, [self.ncar, 1]).reshape(n, n * s)



    def _take_action(self, idx, act):
        # non-active car
        # if self.alive_mask[idx] == 0:
        #     return

        # add wait time for active cars
        # self.wait[idx] += 1

        # action BRAKE i.e STAY
        # if act == 1:
        #     self.car_last_act[idx] = 1
        #     return

        # GAS or move
        # if act==0:
        #     prev = self.car_route_loc[idx]
        #     self.car_route_loc[idx] += 1
        #     curr = self.car_route_loc[idx]

        #     # car/agent has reached end of its path
        #     if curr == len(self.chosen_path[idx]):
        #         self.cars_in_sys -= 1
        #         self.alive_mask[idx] = 0
        #         self.wait[idx] = 0

        #         # put it at dead loc
        #         self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
        #         self.is_completed[idx] = 1
        #         return

        #     elif curr > len(self.chosen_path[idx]):
        #         print(curr)
        #         raise RuntimeError("Out of boud car path")

        #     prev = self.chosen_path[idx][prev]
        #     curr = self.chosen_path[idx][curr]

        #     # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
        #     self.car_loc[idx] = curr

        #     # Change last act for color:
        #     self.car_last_act[idx] = 0
        self.car_last_act[idx] = act



    def _get_reward(self):
        # reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait
        reward = np.full(self.ncar, 0.0)
        for i, p in enumerate(self.curr_idx):
            # most recent action
            act = self.car_last_act[i] #/ (self.naction - 1)

            # retrieve current image labels
            train_image, train_label = self.train_dataset[int(p)]
            # if self.label_dict.get(act) in np.unique(train_label[:, 4:5]):
            #     reward[i] += 0.5
            # reward[i] += (np.intersect1d(self.label_dict.get(act), np.unique(train_label[:, 4:5])).size)*0.25
            reward[i] += (np.intersect1d(self.label_dict.get(act), np.unique(train_label[:, 4:5])).size)*0.75*0.5
        if reward[0] == 0.5 and reward[1] == 0.5:
            reward[0] = 1
            reward[1] = 1

        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())



    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_epochs)
        mod_val = int(step_size / step)

        # if self.curr_start <= epoch < self.curr_end and (epoch - self.curr_start) % mod_val == 0:
        if self.curr_start_epoch <= epoch < self.curr_start_epoch+self.curr_epochs and (epoch - self.curr_start_epoch) % mod_val == 0:
            self.exact_rate = self.exact_rate + step_size
            self.add_rate = self.exact_rate
            print("tj curriculum", self.add_rate)
            # self.add_rate = step_size * (self.exact_rate // step_size)
        else:
            print("not updating curriculum for tj for epoch", epoch)
