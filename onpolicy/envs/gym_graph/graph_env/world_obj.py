import warnings
from abc import ABC, abstractmethod
from typing import Optional

from graph_env.utils import *
from graph_env.metrics import EventType


class WorldObj(ABC):
    def __init__(self, env):
        from graph_env.graph_env import GraphEnv
        self.env: GraphEnv = env

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def decode(code: str, env):
        if code[-1] == 'R':
            return Rubble(env)
        injury_index = InjuryIndex[code[-1]]
        if 'V' in code:
            return Victim(env, injury_index, 'S' in code)
        elif 'E' in code:
            return Evacuation(env, injury_index)
        else:
            raise RuntimeError()


class Victim(WorldObj):
    """
    Victim for Saturn
    """
    KNOWN_VICTIM_INDICES = (InjuryIndex.A, InjuryIndex.B, InjuryIndex.C)

    def __init__(self, env, injury_index: InjuryIndex, is_stabilized):
        super(Victim, self).__init__(env)

        self.injury_index = injury_index
        self.is_stabilized = is_stabilized

        self._is_valid_victim()

    @property
    def reward(self):
        self._is_valid_victim()
        return 50.0 if self.injury_index == InjuryIndex.C else 10.0

    @property
    def pick_up_reward(self):
        return self.reward / 10 if self.is_stabilized else 0  # reward picking-up treated victims

    def _is_valid_victim(self):
        if self.injury_index not in self.KNOWN_VICTIM_INDICES:
            raise RuntimeError(self.injury_index.name + 'should be used for obs only')

    def can_pickup(self):
        """
        Here, only allow regular victims to be picked up
        :return:
        """
        self._is_valid_victim()
        return self.is_stabilized

    def pick_up(self, agent_id) -> float:
        if not self.can_pickup():
            return 0.0

        self.env.agents[agent_id].carrying = self
        return self.pick_up_reward

    def drop(self, agent_id) -> float:
        # penalize drop
        self.env.agents[agent_id].carrying = None
        return -self.pick_up_reward

    def toggle(self, agent_id):
        self._is_valid_victim()

        agent: Agent = self.env.agents[agent_id]
        role = agent.role

        if role == Role.ENGINEER or role == Role.TRANSPORTER:
            return 0.0
        else:
            self.is_stabilized = True
            self.env.metrics_manager.num_victims[EventType.STABILIZED][self.injury_index] += 1
            return self.reward / 10.0

    def __str__(self):
        return ('S' if self.is_stabilized else '') + 'V' + self.injury_index.name


class Evacuation(WorldObj):
    def evacuate(self, agent_id) -> float:
        agent = self.env.agents[agent_id]

        if not isinstance(agent.carrying, Victim):
            raise RuntimeError()

        if not agent.carrying.is_stabilized:
            raise RuntimeError()

        if (not self.injury_index == agent.carrying.injury_index):
            raise RuntimeError()

        reward = agent.carrying.reward
        agent.carrying = None
        self.env.metrics_manager.num_victims[EventType.EVACUATED][self.injury_index] += 1
        self.env.metrics_manager.M1 += reward

        return reward

    def __init__(self, env, injury_index: InjuryIndex):
        super(Evacuation, self).__init__(env)
        self.injury_index = injury_index

    def __str__(self):
        return 'E' + self.injury_index.name


class Rubble(WorldObj):
    @property
    def reward(self):
        """
        It should be used to identify toggling success,
        as the return of toggle(...) is not visible to agents
        :return: CLEANING REWARD: DO NOT count toward game score
        """
        return 1

    def clean(self, agent_id):
        agent = self.env.agents[agent_id]
        if agent.role != Role.ENGINEER:
            warnings.warn('Agent' + str(agent_id) + 'tried to clean but is not an engineer!')
            return 0

        self.env.metrics_manager.num_rubble_cleaned += 1
        return self.reward

    def __str__(self):
        return 'R'


class Agent(WorldObj):
    def __init__(self, role, agent_id, env):
        super(Agent, self).__init__(env)
        self.carrying: Optional[Victim] = None
        self.role: Role = role
        self.current_node_id = None
        self.agent_id = agent_id

    def __str__(self):
        return (self.role.name + ', carrying: ' +
                ('None' if self.carrying is None
                 else self.carrying.injury_index.name))

    def get_injury_one_hot(self):
        if self.carrying is None:
            return np.zeros(InjuryIndex.one_hot_dim())
        else:
            return self.carrying.injury_index.to_one_hot()
