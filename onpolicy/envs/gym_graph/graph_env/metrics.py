import enum
import os
from typing import Dict

import numpy as np
from graph_env.utils import InjuryIndex
from ray.rllib import BaseEnv, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


# noinspection PyMethodOverriding
class SaturnGraphCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            env_index: int,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index]
        episode.custom_metrics.update(
                {'M1': env.metrics_manager.get_M1(),
                 'num_regular_evacuated': env.metrics_manager.get_num_victim(
                         EventType.EVACUATED, 'regular'),
                 'num_critical_evacuated': env.metrics_manager.get_num_victim(
                         EventType.EVACUATED, 'critical'),
                 'num_regular_stabilized': env.metrics_manager.get_num_victim(
                         EventType.STABILIZED, 'regular'),
                 'num_critical_stabilized': env.metrics_manager.get_num_victim(
                         EventType.STABILIZED, 'critical'),
                 'num_rubbles_cleaned': env.metrics_manager.get_num_rubble()}
        )
        # from pprint import pprint
        # pprint(vars(worker))
        # print('E')
        # pprint(vars(episode))
        # replay_dir = os.path.join(worker.io_context.log_dir, 'replays')
        # if not os.path.exists(replay_dir):
        #     os.makedirs(replay_dir)
        #
        # path = os.path.join(replay_dir, '%d.npz' % episode.episode_id)
        # np.savez_compressed(path, **env.replay)


class EventType(str, enum.Enum):
    STABILIZED = 'stabilized'
    EVACUATED = 'evacuated'


class MetricsManager:
    def __init__(self, env):
        from graph_env.world_obj import Victim
        self.num_victims = {
            EventType.EVACUATED: {injury_index: 0 for injury_index in Victim.KNOWN_VICTIM_INDICES},
            EventType.STABILIZED: {injury_index: 0 for injury_index in Victim.KNOWN_VICTIM_INDICES},
        }
        self.num_rubble_cleaned = 0
        self.M1 = 0
        from graph_env.graph_env import GraphEnv
        self.env: GraphEnv = env

    @staticmethod
    def parse_injury_index(injury_index):
        if injury_index == 'all':
            injury_index = (InjuryIndex.A, InjuryIndex.B, InjuryIndex.C)
        elif injury_index == 'regular':
            injury_index = (InjuryIndex.A, InjuryIndex.B)
        elif injury_index == 'critical':
            injury_index = (InjuryIndex.C,)
        elif isinstance(injury_index, int):
            injury_index = (injury_index,)
        else:
            raise ValueError(injury_index)
        return injury_index

    def get_num_victim(self, event_type: EventType, injury_index='all'):
        injury_indices = self.parse_injury_index(injury_index)
        cnt = 0
        for injury_index in injury_indices:
            cnt += self.num_victims[event_type][injury_index]
        return cnt

    def get_num_rubble(self):
        return self.num_rubble_cleaned

    def get_M1(self):
        return self.M1

    def get_time_in_seconds(self):
        return float(self.env.step_count / self.env.world.SECONDS_TO_STEPS)
