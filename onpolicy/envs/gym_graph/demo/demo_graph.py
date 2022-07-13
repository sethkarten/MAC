import argparse

from graph_env.graph_env import *


class DemoEnv(GraphEnv):

    def __init__(self):
        agents = [Agent(Role.MEDIC, 0, self), Agent(Role.ENGINEER, 1, self),
                  Agent(Role.TRANSPORTER, 2, self)]

        super(DemoEnv, self).__init__(total_time=10, agents=agents, base_image=None,
                                      adjacency_list=[(1, 2, 3), (0, 3), (0, 3), (0, 1, 2)],
                                      world=World(RUBBLE_STEPS=1, VICTIM_STEPS=1))

    def _reset_graph(self):
        self.nodes = [Node(node_id, self) for node_id in range(self.num_nodes)]

        self.nodes[0].add_agent(self.agents[0])
        self.nodes[1].add_agent(self.agents[1])
        self.nodes[2].add_agent(self.agents[2])


def main(render):
    env = DemoEnv()
    if render:
        env.render()
    _, rewards, _, _ = env.step({0: Actions.STILL, 1: Actions.STILL, 2: Actions.STILL})
    print(rewards)
    if render:
        env.render()
    env.nodes[0].add_object(Victim(env, InjuryIndex.A, is_stabilized=False))
    if render:
        env.render()
    _, rewards, _, _ = env.step({0: Actions.STABILIZE, 1: Actions.STILL, 2: Actions.STILL})
    print(rewards)
    if render:
        env.render()
    env.nodes[0].add_object(
            Victim(env, InjuryIndex.C, is_stabilized=False))
    if render:
        env.render()
    _, rewards, _, _ = env.step({0: Actions.STABILIZE, 1: Actions.STILL, 2: Actions.STILL})
    print(rewards)
    if render:
        env.render()
    _, rewards, _, _ = env.step({0: Actions.STILL, 1: Actions.STILL, 2: 0})
    print(rewards)
    if render:
        env.render()
    _, rewards, _, _ = env.step({0: Actions.STABILIZE, 1: Actions.STILL, 2: 0})
    print(rewards)
    if render:
        env.render()
    input('Successfully installed, press any key to exit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    main(args.render)
