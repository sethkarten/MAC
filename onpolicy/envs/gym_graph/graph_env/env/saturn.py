import json
import os
import pathlib

import pandas

from graph_env.graph_env import *

resource_dir = os.path.join(pathlib.Path(__file__).parent, 'resources')


def read_adj_list(path_adj) -> List[Tuple[int, ...]]:
    with open(path_adj) as f:
        data = {}
        for line in f:
            if line[0] == '#':
                continue
            line_data = [int(item) for item in line.split(' ')]
            data[line_data[0]] = tuple(line_data[1:])
        res: List[Optional[Tuple[int, ...]]] = [None for _ in range(len(data))]
        for k, v in data.items():
            res[k] = v
        return res


def read_node_attribute(path_pos):
    if path_pos[-5:] == '.json':
        with open(path_pos) as f:
            data = json.load(f)
            res = {int(k): v for k, v in data.items()}
    elif path_pos[-4:] == '.tsv':
        df = pandas.read_csv(path_pos, delimiter='\t').fillna(0).astype(int)
        res = {v['id']: v.drop('id').to_dict() for _, v in df.iterrows()}
    else:
        raise NotImplementedError()
    return res


class SaturnGraph(GraphEnv):
    def __init__(self, data_name, world, total_time=100, seed=None, random=True,
                 transport_only=False, agents=None, content_extension='.json', **kwargs):
        print(self.__class__.__name__)

        if agents is None:
            agents = [Agent(Role.MEDIC, 0, self), Agent(Role.ENGINEER, 1, self),
                      Agent(Role.TRANSPORTER, 2, self)]
        base_image = np.load(os.path.join(resource_dir, data_name + '.npy'))
        adjacency_list = read_adj_list(
                os.path.join(resource_dir, data_name + '.adjlist'))
        nx_pos = read_node_attribute(
                os.path.join(resource_dir, data_name + '_Pos.json'))

        self.contents = read_node_attribute(
                os.path.join(resource_dir, data_name + '_Content' + content_extension))

        self.random_flag = random
        self.transport_only = transport_only

        super(SaturnGraph, self).__init__(
                agents=agents, adjacency_list=adjacency_list, world=world, base_image=base_image,
                nx_pos=nx_pos, total_time=total_time, seed=seed, **kwargs)

    def _reset_graph(self):
        self.nodes = [Node(node_id, self) for node_id in range(self.num_nodes)]
        sequence = range(self.num_nodes)
        if self.random_flag:
            sequence = self.random_generator.permuted(sequence)
        for node_id, content_id in enumerate(sequence):
            node_content = self.contents[content_id]
            for element_name in node_content:
                if self.transport_only and element_name in ("VA", "VB", "VC"):
                    loc = "S" + element_name
                else:
                    loc = element_name
                for num_element in range(node_content[element_name]):
                    self.nodes[node_id].add_object(WorldObj.decode(loc, self))

        for agent_id in self.get_real_agent_ids():
            node_id = self.random_generator.choice(range(self.num_nodes))
            self.nodes[node_id].add_agent(self.agents[agent_id])


class SaturnRight(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True)
        super(SaturnRight, self).__init__(world=world, data_name=self.__class__.__name__)


class SaturnRightTransporter(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False)
        super().__init__(world=world, data_name=SaturnRight.__name__)


class SaturnTopLeft(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True)
        super().__init__(world=world, data_name=self.__class__.__name__)


class SaturnTopLeftFixed(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False)


class SaturnTopLeftFixed600(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False,
                         total_time=600)


class SaturnTopLeftFixedTransporter(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False)


class SaturnFOTopLeftFixed(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False)


class SaturnFOTopLeft(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=True,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=True)


class SaturnFOTopLeftFixedTransporter(SaturnGraph):
    def __init__(self):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False)


class HomogeneousEnv(SaturnGraph, ABC):
    def __init__(self, data_name, world, total_time=100, seed=None, random=True):
        super(HomogeneousEnv, self).__init__(data_name, world, total_time, seed, random,
                                             transport_only=True,
                                             agents=[Agent(Role.TRANSPORTER, 0, self),
                                                     Agent(Role.TRANSPORTER, 1, self),
                                                     Agent(Role.TRANSPORTER, 2, self)])

    def _reset_graph(self):
        super(HomogeneousEnv, self)._reset_graph()
        for node in self.nodes:
            node.objects['R'] = []


class FOTopLeftFixed_Homogeneous(HomogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False, **kwargs)


class FOTopLeft_Homogeneous(HomogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=True, **kwargs)


class POTopLeftFixed_Homogeneous(HomogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False, **kwargs)


class POTopLeft_Homogeneous(HomogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=True, **kwargs)


class HeterogeneousEnv(SaturnGraph, ABC):
    def __init__(self, data_name, world, total_time=100, seed=None, random=True,
                 agents=None, **kwargs):
        super(HeterogeneousEnv, self).__init__(transport_only=False, data_name=data_name,
                                               world=world, total_time=total_time, seed=seed,
                                               random=random, agents=agents, **kwargs)


class POTopLeftFixed_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False, **kwargs)


class POTopLeft_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=True, **kwargs)


class FOTopLeftFixed_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=False, **kwargs)


class FOTopLeft_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnTopLeft.__name__, random=True, **kwargs)


class PORight_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnRight.__name__, random=True, **kwargs)


class PORight_Heterogeneous_Obs_Only(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=True, return_state=False, return_agent_id=False,
                         return_action_mask=True, **kwargs)


class PORight_Heterogeneous_State_Only(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, **kwargs)


class PORight_Heterogeneous_AgentID_Mask(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=True, return_state=True, return_agent_id=True,
                         return_action_mask=True, **kwargs)


class PORight_Heterogeneous_AgentID(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=True, return_state=True, return_agent_id=True,
                         return_action_mask=False, **kwargs)


class PORight_Heterogeneous_Merged_Mask_AgentID(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=True,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class PORight_Heterogeneous_Merged_AgentID(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=True,
                         return_action_mask=False, return_merged_obs=True, **kwargs)


class PORight_Heterogeneous_Merged(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class PORightFixed_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name=SaturnRight.__name__, random=False, **kwargs)


class FORight_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnRight.__name__, random=True, **kwargs)


class FORightFixed_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=True)
        super().__init__(world=world, data_name=SaturnRight.__name__, random=False, **kwargs)


class POFullFixed_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnFull', random=False,
                         content_extension='.tsv', **kwargs)


class POFull_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnFull', random=True,
                         content_extension='.tsv', total_time=500, **kwargs)


class POFull_Heterogeneous_Merged(POFull_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class POFull_Heterogeneous_Obs_Only(POFull_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=True, return_state=False, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=False, **kwargs)


class POFull_Heterogeneous_State_Only(POFull_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=False, **kwargs)


class POHalfR_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnHalfR', random=True, **kwargs)


class POHalfR_Heterogeneous_Merged(POHalfR_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class POMiddle_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnMiddle', random=True, **kwargs)


class POMiddle_Heterogeneous_Merged(POMiddle_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class POLeft_Heterogeneous(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnLeft', random=True, **kwargs)


class POLeft_Heterogeneous_Merged(POLeft_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class PORight_Heterogeneous_Padded29(PORight_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(pad_num_nodes_total=29, **kwargs)


class PORight_Heterogeneous_Merged_Padded29(PORight_Heterogeneous_Merged):
    def __init__(self, **kwargs):
        super().__init__(pad_num_nodes_total=29, **kwargs)


class POLeft_Heterogeneous_Padded29(POLeft_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(pad_num_nodes_total=29, **kwargs)


class POLeft_Heterogeneous_Merged_Padded29(POLeft_Heterogeneous_Merged):
    def __init__(self, **kwargs):
        super().__init__(pad_num_nodes_total=29, **kwargs)


class Heterogeneous_Shuffle(HeterogeneousEnv):
    """
    The graph structure is fixed,
     and node content will correspond to the same fixed/randomized flags.
    However, room ids will be randomized for each reset
    """

    def _reset_graph(self):
        super()._reset_graph()
        original_neighbors = self.neighbors
        order_permuted = self.random_generator.permutation(range(self.num_nodes))
        new_neighbors = dict()
        for room_id in original_neighbors:
            new_neighbors[order_permuted[room_id]] = tuple(sorted(order_permuted[x]
                                                                  for x in
                                                                  original_neighbors[room_id]))
        self.neighbors = new_neighbors

        self.nx_pos = {order_permuted[room_id]: self.nx_pos[room_id] for room_id in
                       self.nx_pos} if self.nx_pos is not None else None
        self.edge_index = np.choose(self.edge_index, order_permuted)


class POLeft_Heterogeneous_Shuffle(Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnLeft', random=True, **kwargs)


class POLeft_Heterogeneous_Shuffle_Merged(POLeft_Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class POMiddle_Heterogeneous_Shuffle(Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnMiddle', random=True, **kwargs)


class POMiddle_Heterogeneous_Shuffle_Merged(POMiddle_Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class PORight_Heterogeneous_Shuffle(Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnRight', random=True, **kwargs)


class PORight_Heterogeneous_Shuffle_Merged(PORight_Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        super().__init__(return_obs=False, return_state=True, return_agent_id=False,
                         return_action_mask=True, return_merged_obs=True, **kwargs)


class PORight_Heterogeneous_Shuffle_Return_Edge_Index(Heterogeneous_Shuffle):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnRight', random=True, return_edge_index=True,
                         **kwargs)


class PORight_Heterogeneous_Return_Edge_Index(HeterogeneousEnv):
    def __init__(self, **kwargs):
        world = World(VICTIM_STEPS=1, RUBBLE_STEPS=1, ALL_ROLE_CAN_TRANSPORT=False,
                      FULLY_OBSERVABLE=False)
        super().__init__(world=world, data_name='SaturnRight', random=True, return_edge_index=True,
                         **kwargs)


class POFull_Heterogeneous_Return_Edge_Index(POFull_Heterogeneous):
    def __init__(self, **kwargs):
        super().__init__(return_edge_index=True, **kwargs)


class POFull_Heterogeneous_Shuffle_Return_Edge_Index(POFull_Heterogeneous_Return_Edge_Index,
                                                     Heterogeneous_Shuffle):

    def __init__(self, **kwargs):
        POFull_Heterogeneous_Return_Edge_Index.__init__(self, **kwargs)

    def _reset_graph(self):
        return Heterogeneous_Shuffle._reset_graph(self)


if __name__ == '__main__':
    import argparse

    """
    Left: 0-27
    Middle: 0-28
    Right: 0-22
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='PORightFixed_Heterogeneous')
    parser.add_argument('--render', default='human', help='human | room_id | None')
    args = parser.parse_args()
    env = eval(args.env_name + '()')
    env.render(args.render)
    r, done, info = env.step_sc([Actions.STILL, Actions.STILL, Actions.STILL])
    obs = env.get_obs()
    state = env.get_state()
    obs_unflattened = env.get_obs_unflattened()
    state_unflattened = env.get_state_unflattened()
    env.render(args.render)

    t1 = 1

    obs_rllib, rewards_rllib, done_rllib, info_rllib = env.step(
            {agent_id: Actions.STILL for agent_id in env.get_real_agent_ids()})

    env.render(args.render)
    t2 = 2
