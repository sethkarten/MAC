import numpy as np
from onpolicy.envs.mpe.core import World, Agent, AgentState
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy import signal

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from typing import List

from scipy.stats import multivariate_normal

from onpolicy.envs.mpe.scenarios.GP_mixture import mix_GPs

import warnings
warnings.filterwarnings(action='ignore')

ENV_SIZE = 16

class AdaptiveSamplingAgentState(AgentState):
    def __init__(self):
        super(AdaptiveSamplingAgentState, self).__init__()
        # Current Agent Belief of World State
        self.env_size = ENV_SIZE
        self.A = np.zeros((self.env_size, self.env_size))
        # print('init agent state')

class AdaptiveSamplingWorld(World):
    def __init__(self):
        super(AdaptiveSamplingWorld, self).__init__()
        self.env_size = ENV_SIZE
        N = 7   # kernel size
        k1d = signal.gaussian(N, std=1).reshape(N, 1)
        kernel = np.outer(k1d, k1d)

        A1 = np.zeros((self.env_size, self.env_size))
        A1[5, 11] = 1    # random
        row, col = np.where(A1 == 1)
        A1[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        A2 = np.zeros((self.env_size, self.env_size))
        A2[3, 3] = 1    # random
        row, col = np.where(A2 == 1)
        A2[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        A3 = np.zeros((self.env_size, self.env_size))
        A3[11, 5] = 1    # random
        row, col = np.where(A3 == 1)
        A3[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        self.A = A1 + A2 + A3

class AdaptiveSamplingAgent(Agent):
    def __init__(self, world):
        super(AdaptiveSamplingAgent, self).__init__()
        self.state = AdaptiveSamplingAgentState()
        self.env_size = ENV_SIZE
        # Specify kernel with initial hyperparameter estimates
        def kernel_initial(
                σf_initial=1.0,         # covariance amplitude
                ell_initial=1.0,        # length scale
                σn_initial=0.1          # noise level
                ):
            return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)
        self.gp = GaussianProcessRegressor(kernel=kernel_initial(), n_restarts_optimizer=10)
        x_min = (0, 0)
        x_max = (self.env_size, self.env_size)
        n_train = 30
        x = np.random.choice(x_max[0], size=(n_train, 1))
        y = np.random.choice(x_max[1], size=(n_train, 1))
        self.X_train = np.hstack((x, y))
        self.y_train = world.A[self.X_train[:,0], self.X_train[:,1]]
        # self.y_train = None
        # print('init agent')
    
    def reset_agent(self, world):
        # Specify kernel with initial hyperparameter estimates
        def kernel_initial(
                σf_initial=1.0,         # covariance amplitude
                ell_initial=1.0,        # length scale
                σn_initial=0.1          # noise level
                ):
            return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)
        self.gp = GaussianProcessRegressor(kernel=kernel_initial(), n_restarts_optimizer=10)
        x_min = (0, 0)
        x_max = (self.env_size, self.env_size)
        n_train = 30
        x = np.random.choice(x_max[0], size=(n_train, 1))
        y = np.random.choice(x_max[1], size=(n_train, 1))
        self.X_train = np.hstack((x, y))
        self.y_train = world.A[self.X_train[:,0], self.X_train[:,1]]
        # print('reset agent')

class Scenario(BaseScenario):
    def make_world(self, args):
        # N = 7   # kernel size
        # k1d = signal.gaussian(N, std=1).reshape(N, 1)
        # kernel = np.outer(k1d, k1d)

        # A1 = np.zeros((16, 16))
        # A1[5, 11] = 1    # random
        # row, col = np.where(A1 == 1)
        # A1[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        # A2 = np.zeros((16, 16))
        # A2[3, 3] = 1    # random
        # row, col = np.where(A2 == 1)
        # A2[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        # A3 = np.zeros((16, 16))
        # A3[11, 5] = 1    # random
        # row, col = np.where(A3 == 1)
        # A3[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel

        # self.A = A1 + A2 + A3

        world = AdaptiveSamplingWorld()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        # num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [AdaptiveSamplingAgent(world) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.size = 0.15
        # # add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = False
        #     landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        self.use_GP = False
        # print('init world')
        return world
    
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.reset_agent(world)
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        # print('reset world')
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # Agents rewarded on how accurate agent reconstruction of world model is
        rew = -np.sum((agent.state.A - world.A)**2)
        if self.outside_boundary(agent):
            rew = rew - 10
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        # print(rew)
        return rew
    
    def observation(self, agent, world):
        loc = (agent.state.p_pos + 1)*((world.env_size-1)/2)
        # loc = loc.round().astype(int)
        loc = np.trunc(loc).astype(int)
        # loc = tuple(loc.round())
        if self.use_GP:
            if self.outside_boundary == False:
                agent.X_train = np.vstack((agent.X_train, loc))
            agent.y_train = world.A[agent.X_train[:,0], agent.X_train[:,1]]
            GPs = []
            for i, a in enumerate(world.agents):
                a.gp.fit(a.X_train, a.y_train)
                GPs.append(a.gp)
            GP_mixture_model = mix_GPs(GPs)
            X_test_x = np.arange(world.env_size)
            X_test_y = np.arange(world.env_size)
            X_test_xx, X_test_yy = np.meshgrid(X_test_x, X_test_y)
            X_test = np.vstack(np.dstack((X_test_xx, X_test_yy)))
            μ_test, σ_test = GP_mixture_model(X_test)
            μ_test_2D = μ_test.reshape((world.env_size, world.env_size))
            σ_test_2D = σ_test.reshape((world.env_size, world.env_size))
            agent.state.A = μ_test_2D

        loc = tuple(loc)
        sampled_loc = None
        if self.outside_boundary(agent):
            sampled_loc = -1
        else:
            sampled_loc = world.A[loc]
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [[sampled_loc]] + other_pos + comm)
        # return [world.A[loc]]
    
    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False