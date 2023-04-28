import numpy as np
from onpolicy.envs.mpe.core import World, Agent, AgentState, Landmark
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


class AdaptiveSamplingAgentState(AgentState):
    def __init__(self, env_size):
        super(AdaptiveSamplingAgentState, self).__init__()
        # Current Agent Belief of World State
        self.env_size = env_size
        self.A = np.zeros((self.env_size, self.env_size))
        # print('init agent state')



class AdaptiveSamplingWorld(World):
    def __init__(self, env_size):
        super(AdaptiveSamplingWorld, self).__init__()
        self.env_size = env_size
        self.reset_sampling()
        self.use_GP = False

    def reset_sampling(self):
        N = 7   # kernel size
        k1d = signal.gaussian(N, std=1).reshape(N, 1)
        kernel = np.outer(k1d, k1d)
        self.A = np.zeros((self.env_size, self.env_size))
        all_peaks = [1,1,1]
        for k in range(10):
            all_peaks.append(0.25)
        self.peaks = np.empty((len(all_peaks),2))
        self.landmarks = [Landmark() for i in range(len(all_peaks))]
        for i, peak in enumerate(all_peaks):
            A_new = np.zeros((self.env_size, self.env_size))
            x,y = np.random.randint(0,self.env_size), np.random.randint(0,self.env_size)
            self.landmarks[i].collide = False
            self.landmarks[i].movable = False
            loc = np.array([x, y]) # loc = (agent.state.p_pos + 1)*((world.env_size-1)/2)
            pos = ((2/(self.env_size-1))*loc) - 1
            self.landmarks[i].state.p_pos = pos
            self.landmarks[i].state.p_vel = np.zeros(self.dim_p)
            if peak == 1:
                self.landmarks[i].name = 'max landmark %d' % i
                self.landmarks[i].color = np.array([0.25, 0.25, 0.25])
            else:
                self.landmarks[i].name = 'min landmark %d' % i
                self.landmarks[i].color = np.array([0.85, 0.35, 0.35])
            self.peaks[i, 0] = x
            self.peaks[i, 1] = y
            A_new[x, y] = peak
            _kernel = kernel
            x_min = x-(N//2)
            x_max = x+(N//2)+1
            if x_min < 0:
                _kernel = _kernel[-x_min:]
                x_min = 0
            if x_max > self.env_size:
                _kernel = _kernel[:-(x_max-self.env_size)]
                x_max = self.env_size
            y_min = y-(N//2)
            y_max = y+(N//2)+1
            if y_min < 0:
                _kernel = _kernel[:,-y_min:]
                y_min = 0
            if y_max > self.env_size:
                _kernel = _kernel[:,:-(y_max-self.env_size)]
                y_max = self.env_size
            A_new[x_min:x_max, y_min:y_max] = peak*_kernel
    
            self.A = self.A + A_new

class AdaptiveSamplingAgent(Agent):
    def __init__(self, world):
        super(AdaptiveSamplingAgent, self).__init__()
        self.state = AdaptiveSamplingAgentState(world.env_size)
        self.env_size = world.env_size
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
        self.use_GP = args.use_GP
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

        world = AdaptiveSamplingWorld(args.env_size)
        world.world_length = args.episode_length
        world.use_GP = args.use_GP
        # set any world properties first
        world.dim_c = 2
        num_agents = args.num_agents
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
        self.use_sampling_reward = args.use_sampling_reward
        # print('init world')
        return world
    
    def reset_world(self, world):
        world.reset_sampling()
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
    
    def reward(self, _agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # Agents rewarded on how accurate agent reconstruction of world model is
        rew = 0
        if self.outside_boundary(_agent):
            rew += self.boundary_penalty(_agent, rew) 
            # return rew
        if self.use_sampling_reward:
            seen_peaks = []
            for i, agent in enumerate(world.agents):
                if self.outside_boundary(agent): continue
                
                # print(world.peaks, agent.state.p_pos)
                agentx, agenty = (agent.state.p_pos + 1)*((world.env_size-1)/2)
                dist_to_peaks = (world.peaks[:,0] - agentx)**2 +\
                                (world.peaks[:,1] - agenty)**2
                # for p in seen_peaks:
                #     dist_to_peaks[p] = np.inf
                closest_peak = np.argmin(dist_to_peaks)
                # print(closest_peak)
                # import sys
                # sys.exit()
                if closest_peak not in seen_peaks:
                    loc = (agent.state.p_pos + 1)*((world.env_size-1)/2)
                    loc = loc.round().astype(int)
                    loc = tuple(loc)
                    rew += world.A[loc]
                    seen_peaks.append(closest_peak)
        else:
            # reconstruction reward
            rew -= np.mean((_agent.state.A - world.A)**2)
        if _agent.collide:
            for a in world.agents:
                if a.name != _agent.name and self.is_collision(a, _agent, world):
                    rew -= 1
        return rew
    
    def observation(self, agent, world):
        loc = (agent.state.p_pos + 1)*((world.env_size-1)/2)
        loc = loc.round().astype(int)
        # loc = np.trunc(loc).astype(int)
        # loc = tuple(loc.round())
        # if self.use_GP:
        if self.outside_boundary == False:
            agent.X_train = np.vstack((agent.X_train, loc))
        agent.y_train = world.A[agent.X_train[:,0], agent.X_train[:,1]]
        if self.use_GP:
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
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([list(loc)] + [[sampled_loc]] + [agent.state.A.flatten()])# + np.array_split(agent.X_train.flatten(), agent.X_train.shape[0]) + [agent.y_train])
        # return [world.A[loc]]
    
    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False
    
    def boundary_penalty(self, agent, rew):
        # rew = 0
        if agent.state.p_pos[0] > 1:
            rew = rew - (agent.state.p_pos[0] - 1)
        if agent.state.p_pos[0] < -1:
            rew = rew - (-1 - agent.state.p_pos[0])
        if agent.state.p_pos[1] > 1:
            rew = rew - (agent.state.p_pos[1] - 1)
        if agent.state.p_pos[1] < -1:
            rew = rew - (-1 - agent.state.p_pos[1])
        return rew
    
    def is_collision(self, agent1, agent2, world):
        loc1 = (agent1.state.p_pos + 1)*((world.env_size-1)/2)
        loc1 = loc1.round().astype(int)
        
        loc2 = (agent2.state.p_pos + 1)*((world.env_size-1)/2)
        loc2 = loc2.round().astype(int)

        return True if np.allclose(loc1, loc2) else False

        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False