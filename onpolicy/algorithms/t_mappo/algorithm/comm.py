import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gumbel softmax
class MAC(nn.Module):
    """
    Multi-Agent Communication Module.
    Facilitates emergent communication among agents
    """
    def __init__(self, args, device=torch.device("cpu")):
        super(MAC, self).__init__()
        self.num_agents = args.num_agents
        self.b = args.n_rollout_threads
        self.args = args
        self.comm_dim = self.args.hidden_size
        self.comm_action_one = True
        self.comm_action_zero = False
        self.mha_comm = False
        self.comm_mode = 'avg'
        # self.comm_passes = args.comm_passes

        self.tpdv = dict(dtype=torch.float32, device=device)

        # Mask for communication
        if self.comm_action_zero:
            self.comm_mask = torch.zeros(self.num_agents, self.num_agents).to(**self.tpdv)
        else:
            # this just prohibits self communication
            self.comm_mask = torch.ones(self.num_agents, self.num_agents) \
                            - torch.eye(self.num_agents, self.num_agents).to(**self.tpdv)

        self.comm_head = nn.Linear(self.comm_dim, self.comm_dim)


    def forward(self, comm, info={}):
        input_comm = comm
        self.b = comm.shape[0] // self.num_agents
        comm = comm.reshape(self.b, self.num_agents, self.comm_dim)
        comm, comm_prob, comm_mask = self.communicate(comm, info)
        if self.mha_comm:
            comm = self.attend_comm(comm.view(self.b, self.num_agents, self.num_agents ,self.comm_dim).transpose(1,0), mask=comm_mask, is_comm=True)
        else:
            comm_sum = comm.sum(dim=1)
            comm_sum = comm_sum.reshape(self.b * self.num_agents, self.comm_dim)
            comm = self.comm_head(comm_sum)
            comm = torch.tanh(comm)
        return input_comm + comm

    def communicate(self, comm, info={}):
        n = self.num_agents
        '''Mask Communication'''
        # mask 1) input communication
        mask = self.comm_mask.view(n, n)

        # 2) Mask communcation from dead agents, 3) communication to dead agents
        num_agents_alive, agent_mask = self.get_agent_mask(info)
        # gating sparsity mask
        agent_mask, comm_action, comm_prob = self.get_gating_mask(comm, agent_mask)
        if info != None:
            info['comm_action'] = comm_action.detach().numpy()

        # Mask 1) input communication 2) Mask communcation from dead agents, 3) communication to dead agents
        comm_out_mask = mask * agent_mask * agent_mask.transpose(0, 1)

        '''Perform communication'''
        # doing cts communication vectors only right now
        comm  = comm.view(self.b, n, self.comm_dim)
        comm = comm.unsqueeze(-2).expand(self.b, n, n, self.comm_dim)
        comm_out_mask = comm_out_mask.unsqueeze(0).expand(self.b, n, n)
        comm = comm * comm_out_mask.unsqueeze(-1).expand_as(comm)

        if not self.mha_comm:
            if self.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

        return comm, comm_prob, comm_out_mask

    def get_gating_mask(self, x, agent_mask):
        # Gating
        # Hard Attention - action whether an agent communicates or not
        comm_prob = None
        if self.comm_action_one:
            comm_action = torch.ones(self.num_agents).to(**self.tpdv)
        elif self.comm_action_zero:
            comm_action = torch.zeros(self.num_agents).to(**self.tpdv)
        else:
            x = x.view(self.num_agents, self.hid_size)
            comm_prob = F.log_softmax(self.gating_head(x), dim=-1)[0]
            comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
            comm_prob = comm_prob[:, 1].reshape(self.num_agents)
            comm_action = comm_prob

        comm_action_mask = comm_action.expand(self.num_agents, self.num_agents)
        # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
        agent_mask = agent_mask * comm_action_mask
        return agent_mask, comm_action, comm_prob

    def get_agent_mask(self, info):
        n = self.num_agents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask']).to(**self.tpdv)
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n).to(**self.tpdv)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, n)
        agent_mask = agent_mask.expand(n, n)

        return num_agents_alive, agent_mask
