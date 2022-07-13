# noinspection PyUnresolvedReferences
from abc import ABC, abstractmethod
from typing import Tuple

import networkx
from graph_env.metrics import MetricsManager
from graph_env.node import *
from graph_env.window import *
# noinspection PyProtectedMember
from gym import ObservationWrapper, spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict


class GraphEnv(ABC, MultiAgentEnv):
    def __init__(self, total_time, agents,
                 adjacency_list: List[Tuple[int, ...]], world: World, seed=None,
                 base_image=None, nx_pos=None, graph_render_font=5, return_obs=True,
                 return_state=True, return_agent_id=False, return_action_mask=True,
                 return_merged_obs=False, return_edge_index=False,
                 pad_num_nodes_total=-1):
        self.world = world
        self.base_image = base_image
        assert world.VICTIM_STEPS == 1
        assert world.RUBBLE_STEPS == 1  # Must be atomic!!
        self.graph_render_font = graph_render_font

        self.max_steps = total_time * self.world.SECONDS_TO_STEPS
        self.step_count = None
        self.window = None

        self.agents: Optional[List[Agent]] = agents
        self._agent_ids = set(range(len(self.agents)))  # to be modified by rllib
        self.num_agents = len(self.agents)

        neighbors = dict()

        edge_index = []
        for i, j_list in enumerate(adjacency_list):
            if i not in neighbors:
                neighbors[i] = set()
            for j in j_list:
                if j not in neighbors:
                    neighbors[j] = set()

                edge_index.append((i, j))
                edge_index.append((j, i))
                neighbors[i].add(j)
                neighbors[j].add(i)

        self.neighbors: Dict[int:Tuple[int, ...]] = {i: tuple(sorted(v)) for i, v in
                                                     neighbors.items()}

        self.edge_index = np.asarray(edge_index).transpose()

        self.num_nodes = len(self.neighbors)  # Used for rendering & graph reset/randomization

        self._num_nodes_padded = max(self.num_nodes, pad_num_nodes_total)
        # Used for obs/action space

        self.num_actions = len(Actions) + self._num_nodes_padded

        self.dim_observations = (Node.DIM + 1) * self._num_nodes_padded \
                                + InjuryIndex.one_hot_dim() + Role.one_hot_dim()
        self.dim_state = (Node.DIM + self.num_agents) * self._num_nodes_padded + (
                InjuryIndex.one_hot_dim() + Role.one_hot_dim()) * self.num_agents
        self.dim_state_unflattened = {'graph': (self._num_nodes_padded, Node.DIM + self.num_agents),
                                      'agent': (self.num_agents,
                                                InjuryIndex.one_hot_dim() + Role.one_hot_dim())}
        self.dim_observations_unflattened = {'graph': (self._num_nodes_padded, Node.DIM + 1),
                                             'agent': (
                                                 InjuryIndex.one_hot_dim() + Role.one_hot_dim(),)}

        self.action_space = spaces.Discrete(self.num_actions)

        observation_space = {}
        if return_obs:
            observation_space['obs'] = spaces.Dict({
                k: spaces.Box(low=0, high=Node.NUM_MAX_ELEMENT, shape=v, dtype=float) for k, v
                in self.dim_observations_unflattened.items()})
        if return_state:
            observation_space['state'] = spaces.Dict({
                k: spaces.Box(low=0, high=Node.NUM_MAX_ELEMENT, shape=v, dtype=float) for k, v
                in self.dim_state_unflattened.items()})
        if return_agent_id:
            # One-hot encoding
            observation_space['agent_id'] = spaces.Box(0, 1, shape=(self.num_agents,), dtype=bool)
        if return_action_mask:
            observation_space['action_mask'] = spaces.Box(0, 1, shape=(self.num_actions,),
                                                          dtype=bool)
        if return_merged_obs:
            observation_space['merged_obs'] = spaces.Dict({
                k: spaces.Box(low=0, high=Node.NUM_MAX_ELEMENT, shape=v, dtype=float) for k, v
                in self.dim_observations_unflattened.items()})

        if return_edge_index:
            observation_space['edge_index'] = spaces.Box(0, self.num_nodes,
                                                         shape=self.edge_index.shape, dtype=int)

        self.observation_space = spaces.Dict(observation_space)

        self.random_generator: Optional[np.random.Generator] = None
        self.replay: Optional[Dict] = None
        self.window = None
        self.nodes: Optional[List[Node]] = None
        self.nx_graph = None
        self.nx_pos = nx_pos
        self.obs: Optional[List[Optional[Dict]]] = None
        self.state: Optional[Dict] = None
        self.metrics_manager: Optional[MetricsManager] = None
        self.max_M1 = None
        self.reset(seed=seed)

        super(GraphEnv, self).__init__()

    def get_real_agent_ids(self):
        """
        For some reason, rllib adds "__all__" to the _agent_ids set
        """
        return range(len(self.agents))

    def get_agent_id_one_hot(self, agent_id):
        res = np.zeros(self.num_agents, dtype=bool)
        res[agent_id] = True
        return res

    def get_configurations(self):
        return {
            "n_agents": self.num_agents,
            "n_actions": self.num_actions,
            "state_shape": self.dim_state,
            "obs_shape": self.dim_observations,
            "state_unflattened_shape": self.dim_state_unflattened,
            "obs_unflattened_shape": self.dim_observations_unflattened,
            "episode_limit": self.max_steps,
        }

    def get_avail_agent_actions(self, agent_id):
        """
        Boolean mask of actions
        """
        agent = self.agents[agent_id]

        node_id = agent.current_node_id
        action_mask = np.zeros(self.num_actions, dtype=bool)
        np.put(action_mask, self.neighbors[node_id], True)
        action_mask[Actions.STILL.to_RL(self._num_nodes_padded)] = True

        role = agent.role
        if role == Role.MEDIC:
            action_mask[Actions.STABILIZE.to_RL(self._num_nodes_padded)] = True
        elif role == Role.ENGINEER:
            action_mask[Actions.CLEAN.to_RL(self._num_nodes_padded)] = True

        if agent.carrying is None:
            action_mask[Actions.PICK.to_RL(self._num_nodes_padded)] = True
        else:
            action_mask[Actions.EVACUATE.to_RL(self._num_nodes_padded)] = True

        return action_mask

    def step_sc(self, actions: List):
        _, rewards, done, info = self.step({i: a for i, a in enumerate(actions)})
        return np.sum([r for r in rewards.values()]), done['__all__'], info

    def get_obs(self):
        return [np.concatenate([v.flatten() for v in agent_obs.values()]) for agent_obs in self.obs]

    def get_state(self):
        return np.concatenate([v.flatten() for v in self.state.values()])

    def get_obs_unflattened(self):
        return self.obs

    def get_state_unflattened(self):
        return self.state

    @abstractmethod
    def _reset_graph(self):
        """
        Responsible for initializing the nodes
        """
        pass

    def _reset_agents(self):
        for agent in self.agents:
            agent.carrying = None
            agent.current_node_id = None

    def save_replay(self, observations, states):
        self.replay['state_graph'][self.step_count] = states['graph']
        self.replay['state_agent'][self.step_count] = states['agent']
        for agent_id in self.get_real_agent_ids():
            self.replay['obs_graph'][self.step_count, agent_id] = observations[agent_id]['graph']
            self.replay['obs_agent'][self.step_count, agent_id] = observations[agent_id]['agent']

    def seed(self, seed=None):
        # Seed the random number generator
        self.random_generator = np.random.default_rng(seed)
        return [seed]

    def _remaining_victims(self):
        keys = Victim.KNOWN_VICTIM_INDICES
        contents = {k: 0 for k in keys}

        for node in self.nodes:
            for k in keys:
                contents[k] += len(node.objects['V' + k.name])
                contents[k] += len(node.objects['SV' + k.name])

        for agent in self.agents:
            if agent.carrying is not None:
                contents[agent.carrying.injury_index] += 1

        return contents

    def _total_victims(self):
        contents = self._remaining_victims()
        for k, v in self.metrics_manager.num_victims[EventType.EVACUATED].items():
            contents[k] += v
        return contents

    def get_node(self, node_id):
        return self.nodes[node_id]

    def _gen_obs(self):
        state_graph = np.zeros(self.dim_state_unflattened['graph'])
        state_agent = np.zeros(self.dim_state_unflattened['agent'])
        for node_id in range(self.num_nodes):
            state_graph[node_id, :Node.DIM] = self.get_node(node_id).encode(
                    rubble_block_view=self.world.RUBBLE_BLOCK_VIEW)
        for agent_id in self.get_real_agent_ids():
            agent = self.agents[agent_id]
            state_graph[agent.current_node_id, Node.DIM + agent_id] = 1
            state_agent[agent_id, :InjuryIndex.one_hot_dim()] = agent.get_injury_one_hot()
            state_agent[agent_id, InjuryIndex.one_hot_dim():] = agent.role.to_one_hot()

        observations: List[Optional[Dict]] = [None for _ in self.get_real_agent_ids()]
        states = {'graph': state_graph, 'agent': state_agent}

        for agent_id in self.get_real_agent_ids():
            obs_graph = np.zeros(self.dim_observations_unflattened['graph'])
            obs_agent = np.zeros(self.dim_observations_unflattened['agent'])
            agent = self.agents[agent_id]
            if self.world.FULLY_OBSERVABLE:
                obs_graph[:, :Node.DIM] = state_graph[:, :Node.DIM]
            else:
                obs_graph[agent.current_node_id, :Node.DIM] = \
                    state_graph[agent.current_node_id, :Node.DIM]

            obs_graph[agent.current_node_id, -1] = 1
            obs_agent[:] = state_agent[agent_id]

            observations[agent_id] = {'graph': obs_graph, 'agent': obs_agent}

        self.save_replay(observations=observations,
                         states=states)

        self.obs, self.state = observations, states

    def reset(self, **kwargs) -> MultiAgentDict:
        self.seed(kwargs.get('seed', None))

        self.step_count = 0
        self.metrics_manager = MetricsManager(self)

        self._reset_agents()  # call this prior to reset_graph
        self._reset_graph()

        self.max_M1 = 0
        for injury_index, num_victims in self._total_victims().items():
            self.max_M1 += Victim(None, injury_index, False).reward * num_victims

        # These fields should be defined by _gen_grid
        for agent in self.agents:
            assert agent.current_node_id is not None

        self.replay = {
            'obs_graph': np.zeros(
                    (self.max_steps + 1, len(self.agents)) + self.dim_observations_unflattened[
                        'graph'], dtype=np.float32),
            'obs_agent': np.zeros(
                    (self.max_steps + 1, len(self.agents)) + self.dim_observations_unflattened[
                        'agent'], dtype=np.float32),
            'state_graph': np.zeros((self.max_steps + 1,) + self.dim_state_unflattened['graph'],
                                    dtype=np.float32),
            'state_agent': np.zeros((self.max_steps + 1,) + self.dim_state_unflattened['agent'],
                                    dtype=np.float32),
            'actions': np.zeros((self.max_steps + 1, len(self.agents)), dtype=int),
            'rewards': np.zeros((self.max_steps + 1, len(self.agents)), dtype=int),
        }

        self._gen_obs()

        return self._return_obs()

    def step(self, actions: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Like a multi-agent rllib environment
        :param actions: dict of actions
        :return: {id:obs,},{id:reward,},{done},{info}
        """
        self.step_count += 1
        for agent_id in self.get_real_agent_ids():
            self.replay['actions'][self.step_count][agent_id] = actions[agent_id]

        order = self.random_generator.permutation(len(actions))

        rewards = {agent_id: 0.0 for agent_id in self.get_real_agent_ids()}
        done = {'__all__': False}  # do not return False for agent-specific dones
        info = {}

        for agent_id in order:
            action = actions[agent_id]

            if isinstance(action, Actions):
                rewards[agent_id] = self.get_node(self.agents[agent_id].current_node_id).interact(
                        agent_id, action)
            elif action < self._num_nodes_padded:
                if self.get_avail_agent_actions(agent_id)[action]:
                    self.get_node(self.agents[agent_id].current_node_id).remove_agent(
                            self.agents[agent_id])
                    self.get_node(action).add_agent(self.agents[agent_id])
            else:
                rewards[agent_id] = self.get_node(self.agents[agent_id].current_node_id).interact(
                        agent_id, Actions.from_RL(self._num_nodes_padded, action))

        if self.step_count >= self.max_steps:
            for agent_id in self.get_real_agent_ids():
                done[agent_id] = True
            done['__all__'] = True

        self._gen_obs()

        for agent_id in self.get_real_agent_ids():
            self.replay['rewards'][self.step_count][agent_id] = rewards[agent_id]

        return self._return_obs(), rewards, done, info

    def _return_obs(self) -> MultiAgentDict:
        res = {}
        keys = self.observation_space.spaces.keys()
        for agent_id in self.get_real_agent_ids():
            res_agent_id = {}
            if 'action_mask' in keys:
                res_agent_id['action_mask'] = self.get_avail_agent_actions(agent_id)

            if 'obs' in keys:
                res_agent_id['obs'] = self.obs[agent_id]
            if 'state' in keys:
                res_agent_id['state'] = self.state

            if 'agent_id' in keys:
                res_agent_id['agent_id'] = self.get_agent_id_one_hot(agent_id)

            if 'merged_obs' in keys:
                res_agent_id['merged_obs'] = {'graph': np.zeros_like(self.obs[agent_id]['graph']),
                                              'agent': self.obs[agent_id]['agent'], }
                for _agent_id in self.get_real_agent_ids():
                    _node_id = self.agents[_agent_id].current_node_id
                    res_agent_id['merged_obs']['graph'][_node_id, :] = self.obs[_agent_id]['graph'][
                                                                       _node_id, :]

            if 'edge_index' in keys:
                res_agent_id['edge_index'] = self.edge_index

            res[agent_id] = res_agent_id
        return res

    def render(self, mode='human') -> None:
        """
        mode: 'human', 'room_id'
        """
        if mode not in ('human', 'room_id'):
            return
        if self.window is None:
            self.window = Window(self.__class__.__name__)

        if self.nx_graph is None:
            # Init
            self.nx_graph = networkx.Graph()
            for current, adj in self.neighbors.items():
                for next_node in adj:
                    self.nx_graph.add_edge(current, next_node)

        pos = networkx.spring_layout(self.nx_graph, seed=0) if self.nx_pos is None else self.nx_pos

        seconds_remaining = (self.max_steps - self.step_count) // self.world.SECONDS_TO_STEPS
        timer_txt = f"{seconds_remaining // 60}:{seconds_remaining % 60}"
        text = timer_txt + ", M1: %d/%d" % (self.metrics_manager.M1, self.max_M1)
        labels = {node.node_id: (str(node) if mode == 'human' else str(node.node_id)) for node in
                  self.nodes}
        self.window.clear()
        networkx.draw_networkx(self.nx_graph, pos, with_labels=True, node_shape='s',
                               labels=labels,
                               font_size=self.graph_render_font * (1 if mode == 'human' else 2),
                               ax=self.window.ax, font_color='k',
                               edge_color='c',
                               node_size=0)
        # node_size=[len(v) * 30 for v in labels.values()])
        x_values, y_values = zip(*pos.values())
        self.window.set_lim(x_values, y_values)

        if self.base_image is not None:
            self.window.imshow(self.base_image, text)

        self.window.show()

    # For rllib

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        """Checks if the observation space contains the given key.
        Args:
            x: Observations to check.
        Returns:
            True if the observation space contains the given all observations
                in x.
        """
        for agent_id in x:
            if not self.observation_space.contains(x[agent_id]):
                return False
        return True

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        """Checks if the action space contains the given action.
        Args:
            x: Actions to check.
        Returns:
            True if the action space contains all actions in x.
        """
        for agent_id in x:
            if not self.action_space.contains(x[agent_id]):
                return False
        return True

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        """Returns a random action for each environment, and potentially each
            agent in that environment.
        Args:
            agent_ids: List of agent ids to sample actions for. If None or
                empty list, sample actions for all agents in the
                environment.
        Returns:
            A random action for each environment.
        """
        return {agent_id: self.action_space.sample() for agent_id in
                (agent_ids if agent_ids is not None else self.get_real_agent_ids())}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        """Returns a random observation from the observation space for each
        agent if agent_ids is None, otherwise returns a random observation for
        the agents in agent_ids.
        Args:
            agent_ids: List of agent ids to sample actions for. If None or
                empty list, sample actions for all agents in the
                environment.
        Returns:
            A random action for each environment.
        """

        return {agent_id: self.observation_space.sample() for agent_id in
                (agent_ids if agent_ids is not None else self.get_real_agent_ids())}


class FlattenObservation(ObservationWrapper):
    """
    Wrapper for rllib's multi-agent env
    """

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return {agent_id: spaces.flatten(self.env.observation_space, observation[agent_id]) for
                agent_id in observation}
