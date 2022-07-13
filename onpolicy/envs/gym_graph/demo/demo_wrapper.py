from graph_env.env.saturn import PORight_Heterogeneous_Merged
from graph_env.utils import Actions
from graph_env.graph_env import FlattenObservation

if __name__ == '__main__':
    env = FlattenObservation(PORight_Heterogeneous_Merged())

    obs_rllib, rewards_rllib, done_rllib, info_rllib = env.step(
            {agent_id: Actions.STILL for agent_id in env.get_real_agent_ids()})

    env.render()
    t = 1
