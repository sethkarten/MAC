import gym
from .multiagentenv import MultiAgentEnv

import torch
from torchvision import datasets, transforms
import numpy as np
from gym.spaces import Discrete


class MNISTMemorizationEnv(MultiAgentEnv):
    
    def __init__(self, args):
        self.name = "MNISTMemorization"
        self.episode_length = args.episode_length
        # define transforms
        transformer = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])

        # download and create datasets
        trainset = datasets.MNIST(root='mnist_data', 
                                    train=True, 
                                    transform=transformer,
                                    download=True)
        evens = list(range(0, len(trainset), 2))
        odds = list(range(1, len(trainset), 2))
        self.train_dataset = torch.utils.data.Subset(trainset, evens)
        self.train_dataset_questions = torch.utils.data.Subset(trainset, odds)

        self.valid_dataset = datasets.MNIST(root='mnist_data', 
                                    train=False, 
                                    transform=transformer)
        self._seed = args.seed
        self.n_agents = 1
        self.n_actions = 2
        torch.manual_seed(args.seed)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())

        self._reset()

    def step(self, actions):
        """Returns reward, terminated, info."""
        infos = [{} for i in range(self.n_agents)]
        # reward
        if actions[0]==1 and self.label_answers[self.timestep]:
            #or actions[0]==0 and self.label_questions[self.timestep] not in self.labels_seen[:self.timestep+1]:
            # correct guess
            reward = 1.
        elif actions[0]==0 and not self.label_answers[self.timestep]:
            reward = 0.1
        else:
            reward = -1
        rewards = np.array([[reward]]*self.n_agents)
        # terminated
        dones = np.zeros((self.n_agents), dtype=bool)
        if self.episode_length-1 == self.timestep:
            dones = np.ones((self.n_agents), dtype=bool)
        # obs
        local_obs = self.get_obs_agent(0)
        global_state = self.get_state()
        # avail actions
        available_actions = [self.get_avail_agent_actions(0)]
        # increment timestep
        self.timestep += 1
        return local_obs.reshape(1, -1), global_state.reshape(1, -1), rewards.reshape(1, -1), dones, infos, available_actions

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs = self.get_obs_agent(0)
        return np.array(obs)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        obs = self.train_dataset[self.data_indices[self.timestep]][0][0]
        if self.label_answers[self.timestep]:
            obs_question = self.train_dataset[self.label_questions[self.timestep]][0][0]
        else:
            obs_question = self.train_dataset_questions[self.label_questions[self.timestep]][0][0]
        obs = obs.reshape(-1).numpy()
        obs_question = obs_question.reshape(-1).numpy()
        return np.concatenate((obs, obs_question), axis=-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return [32*32 * 2]

    def get_state(self):
        """Returns the global state."""
        return self.get_obs()

    def get_state_size(self):
        """Returns the size of the global state."""
        return [32*32 * 2]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(0)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return np.ones(self.n_actions, dtype=bool)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._reset()
        # get data indices
        local_obs = self.get_obs()
        global_state = self.get_state()
        available_actions = self.get_avail_actions()
        return local_obs.reshape(1, -1), global_state.reshape(1, -1), available_actions
    
    def _reset(self):
        self.timestep = 0
        self.data_indices = np.random.randint(0, len(self.train_dataset), size=self.episode_length)
        self.labels_seen = np.zeros(self.episode_length)
        # self.label_questions = np.zeros(self.episode_length)
        self.label_questions = np.random.randint(0, len(self.train_dataset_questions), size=self.episode_length)
        self.label_answers = np.zeros(self.episode_length, dtype=bool)
        count = 0
        for i in range(self.episode_length):
            self.labels_seen[i] = self.train_dataset[self.data_indices[i]][1]
            if np.random.random() > 0.5 and count < self.episode_length//2:
                self.label_questions[i] = np.random.choice(self.data_indices[:max(i+1-10, 1)])
                self.label_answers[i] = True
                count += 1
            # else:
            #     self.label_questions[i] = np.random.randint(0, 9)
        

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "obs_alone_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_length}
        return env_info