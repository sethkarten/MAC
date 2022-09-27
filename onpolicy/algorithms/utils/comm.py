import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import init
from onpolicy.algorithms.utils.transformer import Attention
# import gumbel softmax
import numpy as np

def _t2n(x):
    return x.detach().cpu().numpy()

class MAC(nn.Module):
    """
    Multi-Agent Communication Module.
    Facilitates emergent communication among agents
    """
    def __init__(self, args, active_func, gain, device=torch.device("cpu")):
        super(MAC, self).__init__()
        self.num_agents = args.num_agents
        self.b = args.n_rollout_threads
        self.args = args
        self.comm_dim = self.args.hidden_size
        self.comm_action_one = True
        self.comm_action_zero = False
        self.mha_comm = args.mha_comm
        self.comm_mode = 'avg'
        # self.comm_passes = args.comm_passes
        self.active_func = active_func
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.tpdv = dict(dtype=torch.float32, device=device)

        # create communication action
        if self.args.use_vib or self.args.use_vqvib:
            self.comm_act = self.vib_forward
            self.fc_mu = init_(nn.Linear(self.args.hidden_size, self.comm_dim))
            self.fc_var = init_(nn.Linear(self.args.hidden_size, self.comm_dim))
            if self.args.use_vqvib:
                self.message_vocabulary = nn.Parameter(data=torch.Tensor(self.args.vocab_size, self.comm_dim)).to(**self.tpdv)
                nn.init.uniform_(self.message_vocabulary, -2, 2)
        elif self.args.use_compositional:
            self.comm_act = self.compositional_forward
            # choose transformer or GRU to predict tokens sequentially, mu and var
            self.to_Q = init_(nn.Linear(self.args.hidden_size, self.args.hidden_size))
            self.to_V = init_(nn.Linear(self.args.hidden_size, self.args.hidden_size))
            self.to_K = init_(nn.Linear(self.comm_dim, self.args.hidden_size))
            self.fc_mu = init_(nn.Linear(self.args.hidden_size, self.args.composition_dim))
            self.fc_var = init_(nn.Linear(self.args.hidden_size, self.args.composition_dim))
            # self.EOS_token = nn.Parameter(data=torch.Tensor(self.args.composition_dim)).to(**self.tpdv)
            self.message_vocabulary = nn.Parameter(data=torch.Tensor(self.args.vocab_size+1, self.args.composition_dim)).to(**self.tpdv)
            nn.init.uniform_(self.message_vocabulary, -2, 2)

            # noncomp for independence
            self.fc_inde_mu = init_(nn.Linear(self.args.hidden_size, self.comm_dim))
            self.fc_inde_var = init_(nn.Linear(self.args.hidden_size, self.comm_dim))
        else:
            self.comm_act = init_(nn.Linear(self.args.hidden_size, self.comm_dim))
        # Mask for communication
        if self.comm_action_zero:
            self.comm_mask = torch.zeros(self.num_agents, self.num_agents).to(**self.tpdv)
        else:
            # this just prohibits self communication
            self.comm_mask = torch.ones(self.num_agents, self.num_agents).to(**self.tpdv) \
                            - torch.eye(self.num_agents, self.num_agents).to(**self.tpdv)

        if self.mha_comm:
            self.comm_self_att = Attention(args.transformer_heads, self.comm_dim, active_func, gain, args, dropout=0, hidden_size=self.args.hidden_size)
        else:
            self.comm_sum_head = init_(nn.Linear(self.comm_dim, self.args.hidden_size))
        self.comm_head = init_(nn.Linear(self.args.hidden_size, self.args.hidden_size))

        self.EPISILON = 1e-6
        self.norm_factor = 1 / np.sqrt(self.args.hidden_size)
        torch.autograd.set_detect_anomaly(True)

    def vib_forward(self, hidden_state):
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_var(hidden_state)
        #reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = eps * std + mu

        self.decoding_mu = mu.squeeze().clone()
        self.decoding_log_var = log_var.squeeze().clone()

        return out

    def vqvib_forward(self, hidden_state):
        return torch.min((vib_forward(hidden_state) - self.message_vocabulary).square(), 1)

    def compositional_forward(self, hidden_state):
        # predict VIB tokens until repeat or EOS token
        batch_size = len(hidden_state)
        self.decoding_mus = []
        self.decoding_log_vars = []
        Q, V = self.to_Q(hidden_state) * self.norm_factor, self.to_K(hidden_state)
        total_tokens = self.comm_dim // self.args.composition_dim
        mask = torch.ones(batch_size).to(**self.tpdv)
        mask.requires_grad = False
        finished_messages = torch.zeros(batch_size).to(**self.tpdv)
        EOS_token = self.message_vocabulary[-1]
        message = torch.zeros(batch_size, total_tokens, self.args.composition_dim).to(**self.tpdv)
        # print("total_tokens", total_tokens, self.args.composition_dim, batch_size)
        for i in range(total_tokens):
            # predict token
            m = message.reshape(batch_size, -1).clone()
            K = self.to_K(m) * self.norm_factor
            dot = Q * K
            attn = F.softmin(dot, dim=-1)
            hidden_attn = attn * V
            token_mu = self.fc_mu(hidden_attn)
            token_log_var = self.fc_var(hidden_attn)
            token_std = torch.exp(0.5 * token_log_var)
            self.decoding_mus.append(token_mu.squeeze().clone())
            self.decoding_log_vars.append(token_log_var.squeeze().clone())
            eps = torch.randn_like(token_std)
            token = eps * token_std + token_mu

            # discretization layer
            token_mse = (token.reshape(batch_size,1,self.args.composition_dim) - self.message_vocabulary).square()
            token = torch.min(token_mse, 1)[0]
            # need to do gumbel here? to pass through argmin to get actual vocab words and pass through gradient
            token = token * mask.reshape(-1,1).clone()    # ignore predictions once EOS already reached

            # mask EOS for future
            eos_mask_index = (token.reshape(batch_size,1,self.args.composition_dim) - self.message_vocabulary[-1]).square().sum(-1) < self.EPISILON
            # mask repeat tokens
            mask_index = (token.reshape(batch_size,1,self.args.composition_dim) - message[:,:i]).square().sum(-1) < self.EPISILON
            if mask_index.shape[1] != 0:
                mask_index = mask_index.sum(1) > 0 # logical and through time
                mask_index = torch.logical_or(mask_index, eos_mask_index.reshape(-1))
            else:
                mask_index = eos_mask_index
            mask_index = mask_index.reshape(-1)
            if mask_index.any():
                mask[mask_index] = 0
                EOS_update_indices = torch.logical_and(mask_index, torch.logical_not(finished_messages))
                self.EOS_token_mse_loss = (token.reshape(batch_size,1,self.args.composition_dim) - self.message_vocabulary[-1]).square().sum(-1).mean()
                token[EOS_update_indices] = EOS_token
                # message[EOS_update_indices, i] = EOS_token
                finished_messages = torch.logical_or(mask_index, finished_messages)

            message[:,i] = token
            if finished_messages.sum() == batch_size:# all tokens masked:
                break

        # vartiational for predicting the messages without composition
        inde_mu = self.fc_inde_mu(hidden_state)
        inde_var = self.fc_inde_var(hidden_state)
        inde_std = torch.exp(0.5 * inde_var)
        self.decoding_inde_mu = inde_var.squeeze().clone()
        self.decoding_inde_log_var = inde_std.squeeze().clone()
        inde_eps = torch.randn_like(inde_std)
        self.noncomp_message = inde_eps * inde_std + inde_mu

        self.message = message.reshape(batch_size, -1)
        return self.message.clone()

    def forward(self, hidden_state, info={}):
        # ================== DECODING PHASE BEGINNING ===================
        # communication embedding
        comm = self.comm_act(hidden_state)
        self.b = comm.shape[0] // self.num_agents
        comm = comm.reshape(self.b, self.num_agents, self.comm_dim)
        comm, comm_prob, comm_mask = self.communicate(comm, info)
        # decode communication
        if self.mha_comm:
            # Reshape to account for number of agents in batch
            comm = comm.view(self.b, self.num_agents, self.num_agents, self.comm_dim).transpose(2,1)
            comm = comm.reshape(self.b * self.num_agents, self.num_agents, self.comm_dim)
            comm_mask = comm_mask.reshape(self.b * self.num_agents, self.num_agents, 1)
            comm = self.comm_self_att(comm, comm, comm, mask=None, is_comm=True)
        else:
            comm_sum = comm.sum(dim=1)
            comm = comm_sum.reshape(self.b * self.num_agents, self.comm_dim)
            # resize
            comm = self.comm_sum_head(comm)
        comm = self.active_func(self.comm_head(comm)) # add nonlinearity between message rounds
        return comm

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
            info['comm_action'] = _t2n(comm_action)

        # Mask 1) input communication 2) Mask communcation from dead agents, 3) communication to dead agents
        comm_out_mask = mask * agent_mask * agent_mask.transpose(1, 2)

        '''Perform communication'''
        # doing cts communication vectors only right now
        comm  = comm.view(self.b, n, self.comm_dim)
        comm = comm.unsqueeze(-2).expand(self.b, n, n, self.comm_dim)
        comm = comm #* comm_out_mask.reshape(self.b, n, n, 1)

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
            comm_action = torch.ones(self.b, self.num_agents).to(**self.tpdv)
        elif self.comm_action_zero:
            comm_action = torch.zeros(self.b, self.num_agents).to(**self.tpdv)
        else:
            x = x.view(self.b, self.num_agents, self.hid_size)
            comm_prob = F.log_softmax(self.gating_head(x), dim=-1)[0]
            comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
            comm_prob = comm_prob[:, 1].reshape(self.num_agents)
            comm_action = comm_prob

        comm_action_mask = comm_action.unsqueeze(-1).expand(self.b, self.num_agents, self.num_agents)
        # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
        agent_mask = agent_mask * comm_action_mask
        return agent_mask, comm_action, comm_prob

    def get_agent_mask(self, info):
        n = self.num_agents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask']).to(**self.tpdv)
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(self.b, n).to(**self.tpdv)
            num_agents_alive = n

        agent_mask = agent_mask.view(self.b, 1, n)
        agent_mask = agent_mask.expand(self.b, n, n)

        return num_agents_alive, agent_mask
