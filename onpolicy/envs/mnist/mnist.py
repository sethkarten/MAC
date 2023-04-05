import gym
from .multiagentenv import MultiAgentEnv

import torch
from torchvision import datasets, transforms

class MNISTMemorizationEnv(MultiAgentEnv):
    
    def __init__(self, args):
        self.name = "MNISTMemorization"

        # define transforms
        transforms = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

        # download and create datasets
        self.train_dataset = datasets.MNIST(root='mnist_data', 
                                    train=True, 
                                    transform=transforms,
                                    download=True)

        self.valid_dataset = datasets.MNIST(root='mnist_data', 
                                    train=False, 
                                    transform=transforms)

        # define the data loaders
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)

        valid_loader = DataLoader(dataset=valid_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        
        torch.manual_seed(args.seed)

    def step(self, actions):
        """Returns reward, terminated, info."""

        return local_obs, global_state, reward, dones, infos, available_actions

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_state(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "obs_alone_shape": self.get_obs_alone_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info