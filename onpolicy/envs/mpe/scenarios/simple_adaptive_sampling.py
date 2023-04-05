import numpy as np
from onpolicy.envs.mpe.core import World, Agent, AgentState
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy import signal

class AdaptiveSamplingAgentState(AgentState):
    def __init__(self):
        super(AdaptiveSamplingAgentState, self).__init__()
        # Current Agent Belief of World State
        self.A = np.zeros((16, 16))

class AdaptiveSamplingWorld(World):
    def __init__(self):
        super(AdaptiveSamplingWorld, self).__init__()
        self.env_size = 16
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
    def __init__(self):
        super(AdaptiveSamplingAgent, self).__init__()
        self.state = AdaptiveSamplingAgentState()

class Scenario(BaseScenario):
    def make_world(self):
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
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        # num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
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
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # Agents rewarded on how accurate agent reconstruction of world model is
        rew = np.sum((agent.state.A - world.A)**2)
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        return rew
    
    def observation(self, agent, world):
        loc = (agent.state.p_pos + 1)*(world.env_size//2)
        loc = tuple(loc.round())
        return world.A[loc]
