from typing import Dict, List

from graph_env.world_obj import *


class Node:
    NUM_MAX_ELEMENT = 4  # OF A CERTAIN TYPE

    entry_keys = [
        'SV' + InjuryIndex.A.name,
        'SV' + InjuryIndex.B.name,
        'SV' + InjuryIndex.C.name,
        'V' + InjuryIndex.A.name,
        'V' + InjuryIndex.B.name,
        'V' + InjuryIndex.C.name,
        'R',
        'E' + InjuryIndex.A.name,
        'E' + InjuryIndex.B.name,
        'E' + InjuryIndex.C.name,
    ]

    DIM = len(entry_keys) + len(Role)

    def __str__(self):
        list_of_strings = ['ID:' + str(self.node_id)]
        for key in self.entry_keys:
            list_of_strings.append(key + ': ' + str(len(self.objects[key])))
        for agent in self.agents.values():
            list_of_strings.append(str(agent.agent_id) + ', ' + agent.role.name)

        return '\n'.join(list_of_strings)

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return hash(self.node_id)

    def __init__(self, node_id: int, env):
        self.node_id = node_id
        from graph_env.graph_env import GraphEnv
        self.env: GraphEnv = env
        self.objects: Dict[str: List[WorldObj]] = {name: [] for name in self.entry_keys}
        self.agents: Dict[int:Agent] = {}

    def has_rubble(self):
        return len(self.objects['R']) > 0

    def encode(self, rubble_block_view=False):
        encoded = np.zeros(self.DIM)

        if self.has_rubble() and rubble_block_view:
            encoded[self.entry_keys.index('R')] = len(self.objects['R'])
        else:
            for index, key in enumerate(self.entry_keys):
                encoded[index] = len(self.objects[key])

        for agent in self.agents.values():
            encoded[len(self.entry_keys) + agent.role] += 1

        return encoded

    def add_agent(self, agent: Agent):
        self.agents[agent.agent_id] = agent
        agent.current_node_id = self.node_id

    def remove_agent(self, agent: Agent):
        return self.agents.pop(agent.agent_id)

    def add_object(self, world_obj: WorldObj):
        key = str(world_obj)
        assert key in self.entry_keys
        self.objects[key].append(world_obj)

    def interact(self, agent_id, action: Actions) -> float:
        """

        :param agent_id:
        :param action:
        :return: reward
        """

        assert isinstance(action, Actions)

        agent: Agent = self.agents[agent_id]
        if action is Actions.CLEAN:
            if agent.role == Role.ENGINEER and self.has_rubble():
                return self.objects['R'].pop().clean(agent_id)
            return 0.0
        elif self.has_rubble():
            return 0.0

        if action is Actions.STABILIZE:
            if agent.role == Role.MEDIC:
                def get_reward(key):
                    victim = self.objects[key].pop()
                    reward = victim.toggle(agent_id)
                    self.objects['S' + key].append(victim)
                    return reward

                key = 'V' + InjuryIndex.C.name
                if len(self.agents) >= 2 and len(self.objects[key]) > 0:
                    return get_reward(key)

                for injury_index in self.env.random_generator.permuted([InjuryIndex.A,
                                                                        InjuryIndex.B]):
                    injury_index = InjuryIndex(injury_index)
                    key = 'V' + injury_index.name
                    if len(self.objects[key]) != 0:
                        return get_reward(key)
            return 0.0

        elif action == Actions.PICK:
            if agent.carrying is None and (
                    self.env.world.ALL_ROLE_CAN_TRANSPORT or agent.role == Role.TRANSPORTER):
                injury_list = [InjuryIndex.C]
                injury_list.extend(self.env.random_generator.permuted([InjuryIndex.A,
                                                                       InjuryIndex.B]))
                for injury_index in injury_list:
                    injury_index = InjuryIndex(injury_index)
                    key = 'SV' + injury_index.name
                    if len(self.objects[key]) != 0:
                        victim: Victim = self.objects[key].pop()
                        reward = victim.pick_up(agent_id)
                        return reward
            return 0.0
        elif action == Actions.EVACUATE:
            if agent.carrying is not None:
                injury_index = agent.carrying.injury_index
                key = 'E' + injury_index.name

                if len(self.objects[key]) > 0:
                    return self.objects[key][0].evacuate(agent_id)
            return 0.0
        elif action is Actions.STILL:
            return 0.0
        else:
            raise RuntimeError('Not a valid action!')
