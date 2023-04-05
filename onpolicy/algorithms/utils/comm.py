import torch
import torch.nn.functional as F
from torch import nn
from onpolicy.algorithms.utils.transformer import Attention
# from network_utils import gumbel_softmax
import numpy as np

def _t2n(x):
    return x.detach().cpu().numpy()

class MAC(nn.Module):
    """
    Multi-Agent Communication.
    Uses communication vector to communicate info between agents
    """
    def __init__(self, args, num_inputs, train_mode=True, device=torch.device("cpu")):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MAC {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(MAC, self).__init__()
        args.hid_size = args.hidden_size
        # args.comm_dim = 64
        self.args = args
        self.args.comp_beta = 0.1
        self.nagents = args.num_agents
        self.hid_size = args.hidden_size
        self.comm_passes = args.comm_rounds
        # self.max_len = min(args.composition_dim, 10)
        self.dropout = 0.
        self.vocab_size = args.vocab_size
        self.composition_dim = 32
        self.EPISILON = 1e-9
        self.comm_dim = self.args.comm_dim
        self.norm_factor = 1 / np.sqrt(self.hid_size)

        self.tpdv = dict(dtype=torch.float32, device=device)
        # embedding for one hot encoding of inputs
        self.embed = nn.Linear(num_inputs, args.hid_size)

        # message action generation
        # self.init_hidden(args.batch_size)
        # # self.message_gru = nn.GRU(args.hid_size, args.hid_size)
        # self.message_gru = nn.Linear(args.hid_size, args.hid_size)

        if self.args.vae or self.args.use_vqvib:
            self.fc_mu = nn.Linear(args.hid_size, args.comm_dim)
            self.fc_var = nn.Linear(args.hid_size, args.comm_dim)
            self.message_generation = self.vib_forward
            if self.args.use_vqvib:
                self.message_generation = self.vqvib_forward
                self.message_vocabulary = nn.Parameter(data=torch.Tensor(self.vocab_size, self.args.comm_dim)).to(**self.tpdv)
                nn.init.uniform_(self.message_vocabulary, -2, 2)
        elif self.args.use_compositional:
            self.message_generation = self.compositional_forward
            # choose transformer or GRU to predict tokens sequentially, mu and var
            self.to_Q = nn.Linear(self.hid_size, self.hid_size)
            self.to_V = nn.Linear(self.hid_size, self.hid_size)
            self.to_K = nn.Linear(self.composition_dim, self.hid_size) # expand message to hidden size
            self.fc_mu = nn.Linear(self.hid_size, self.composition_dim)
            self.fc_var = nn.Linear(self.hid_size, self.composition_dim)

            # self.EOS_token = nn.Parameter(data=torch.Tensor(self.composition_dim))
            self.message_vocabulary = nn.Parameter(data=torch.Tensor(self.vocab_size+1, self.composition_dim)).to(**self.tpdv)
            nn.init.uniform_(self.message_vocabulary, -2, 2)

            # noncomp for independence
            self.fc_inde_mu = nn.Linear(self.hid_size, self.comm_dim)
            self.fc_inde_var = nn.Linear(self.hid_size, self.comm_dim)
        else:
            self.message_generation = nn.Linear(args.hid_size, args.comm_dim)


        # attend to communications to determine relevant information
        if args.mha_comm:
            self.attend_comm = Attention(args.num_heads, args.comm_dim, dropout=self.dropout)


        self.action_gru = nn.GRU(args.comm_dim + args.hid_size, args.hid_size)


        # Decoder for information maximizing communication
        if self.args.use_ae:
            self.decoder_head = nn.Linear(args.comm_dim, num_inputs)

        # attend to latent obs/intent/comm to produce action
        # self.attend_action = SelfAttention(args.num_heads, args.hid_size)
        # print(args.num_actions)
        # self.action_head = nn.Linear(self.hid_size, args.num_actions[0])

        # Gate sparse communication
        self.gating_head = nn.Linear(self.hid_size, 2)

        # Critic
        self.value_head = nn.Linear(self.hid_size, 1)

        # Mask for communication
        # if self.args.comm_mask_zero:
        # self.comm_mask = torch.zeros(self.nagents, self.nagents).to(**self.tpdv)
        # else:
        # this just prohibits self communication
        self.comm_mask = (torch.ones(self.nagents, self.nagents) \
                        - torch.eye(self.nagents, self.nagents)).to(**self.tpdv)


        self.apply(self.init_weights)
        torch.nn.init.zeros_(self.embed.weight)
        # if self.args.vae:
        #     torch.nn.init.zeros_(self.fc_mu.weight)
        #     torch.nn.init.zeros_(self.fc_var.weight)

    def forward(self, comm, info={}):
        comm_encoding = self.default_forward(comm)
        return self.communicate(comm_encoding, info)

    def communicate(self, comm, info={}):
        # print('before',comm, comm.shape)
        n = self.nagents
        comm = comm.reshape(-1, n, self.args.comm_dim)
        # print('reshape', comm)
        b = len(comm)
        # print(comm.shape)
        '''Mask Communication'''
        # mask 1) input communication
        mask = self.comm_mask.view(n, n)

        # 2) Mask communcation from dead agents, 3) communication to dead agents
        num_agents_alive, agent_mask = self.get_agent_mask(b, info)
        # gating sparsity mask
        # agent_mask, comm_action, comm_prob = self.get_gating_mask(comm, agent_mask)
        # info['comm_action'] = comm_action.detach().numpy()

        # Mask 1) input communication 2) Mask communcation from dead agents, 3) communication to dead agents
        comm_out_mask = mask * agent_mask * agent_mask.transpose(0, 1)

        '''Perform communication'''
        # doing cts communication vectors only right now
        comm  = comm.view(b, n, self.args.comm_dim)
        comm = comm.unsqueeze(-2).expand(b, n, n, self.args.comm_dim)
        # print(comm_out_mask.unsqueeze(-1).shape, comm.shape)
        comm = comm * comm_out_mask.unsqueeze(-1).expand_as(comm)
        # print(comm.shape)
        if self.args.mha_comm:
            comm = self.attend_comm(comm.view(b,n,n,self.args.comm_dim).transpose(2,1), mask=self.comm_mask.view(n, n), is_comm=True)
        else:
            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)
            comm = comm.sum(dim=1)
        # print(comm.shape)
        comm = comm.reshape(-1, self.comm_dim)
        # print('after', comm, comm.shape)
        return comm #, comm_prob, comm_out_mask

    def do_embed(self, x):
        x, extras = x
        x = self.embed(x)
        hidden_state_message, hidden_state_action = extras
        return x, hidden_state_message, hidden_state_action

    def default_forward(self, hidden_state):
        self.message = self.message_generation(hidden_state)
        return self.message

    def ae_forward(self, hidden_state):
        self.message = self.message_generation(hidden_state)
        return self.message

    def vib_forward(self, hidden_state):
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_var(hidden_state)
        #reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = eps * std + mu

        # n, e = mu.shape[-2], mu.shape[-1]
        self.decoding_mu = mu.squeeze().clone()#.reshape((1,n,e))
        self.decoding_log_var = log_var.squeeze().clone()#.reshape((1,n,e))
        # print(self.decoding_mu.shape)
        self.message = out.reshape(self.nagents, -1)
        return out

    def vqvib_forward(self, hidden_state):
        token_close = self.vib_forward(hidden_state).reshape(self.nagents, 1, -1)
        token_mse = (token_close - self.message_vocabulary).square()
        token = torch.min(token_mse, 1)[0]
        return token.reshape(-1)

    def compositional_forward(self, hidden_state):
        # predict VIB tokens until repeat or EOS token
        batch_size = self.nagents
        if self.args.env_name == 'PascalVoc':
            batch_size = 1
        self.decoding_mus = []
        self.decoding_log_vars = []
        Q, V = self.to_Q(hidden_state) * self.norm_factor, self.to_V(hidden_state)
        total_tokens = self.comm_dim // self.composition_dim
        mask = torch.ones(batch_size).to(**self.tpdv)
        # mask.requires_grad = False
        finished_messages = torch.zeros(batch_size).to(**self.tpdv)
        EOS_token = self.message_vocabulary[-1]
        message = torch.zeros(batch_size, total_tokens, self.composition_dim).to(**self.tpdv)
        # print("total_tokens", total_tokens, self.composition_dim, batch_size)
        for i in range(total_tokens):
            # predict token
            m = message.reshape(batch_size*total_tokens, self.composition_dim).clone()
            K = self.norm_factor * self.to_K(m)
            K =  K.reshape(batch_size, total_tokens, self.hid_size).mean(1)
            # K = self.to_K(m) * self.norm_factor
            dot = Q * K
            attn = F.softmin(dot, dim=-1)
            hidden_attn = attn * V
            token_mu = self.fc_mu(hidden_attn)
            token_log_var = torch.zeros_like(token_mu).to(**self.tpdv)
            # token_log_var = self.fc_var(hidden_attn)
            # token_std = torch.exp(0.5 * token_log_var)
            self.decoding_mus.append(token_mu.squeeze().clone())
            self.decoding_log_vars.append(token_log_var.squeeze().clone())
            # eps = torch.randn_like(token_std)
            # token = eps * token_std + token_mu
            token = token_mu
            # discretization layer
            # token_mse = (token.reshape(batch_size,1,self.composition_dim) - self.message_vocabulary).square()
            # token = self.message_vocabulary[torch.min(token_mse, 1)[1][:,1]]
            # token = torch.min(token_mse, 1)[0]
            # need to do gumbel here? to pass through argmin to get actual vocab words and pass through gradient
            token = token * mask.reshape(-1,1).clone()    # ignore predictions once EOS already reached

            self.EOS_token_mse_loss = 0
            '''
            if i != 0:
                # mask EOS for future
                eos_mask_index = (token.reshape(batch_size,1,self.composition_dim) - self.message_vocabulary[-1]).square().sum(-1) < self.EPISILON
                # mask repeat tokens
                mask_index = (token.reshape(batch_size,1,self.composition_dim) - message[:,:i]).square().sum(-1) < self.EPISILON
                if mask_index.shape[1] != 0:
                    mask_index = mask_index.sum(1) > 0 # logical and through time
                    mask_index = torch.logical_or(mask_index, eos_mask_index.reshape(-1))
                else:
                    mask_index = eos_mask_index
                mask_index = mask_index.reshape(-1)
                print(mask_index)
                if mask_index.any():
                    mask[mask_index] = 0
                    EOS_update_indices = torch.logical_and(mask_index, torch.logical_not(finished_messages))
                    if EOS_update_indices.any():
                        self.EOS_token_mse_loss = (token[EOS_update_indices].reshape(batch_size,1,self.composition_dim) - self.message_vocabulary[-1]).square().sum(-1).mean()
                        token[EOS_update_indices] = EOS_token
                        # message[EOS_update_indices, i] = EOS_token
                        finished_messages = torch.logical_or(mask_index, finished_messages)
            '''
            # # need to mask unfinished sequences with EOS at end
            #
            message[:,i] = token
            # if finished_messages.sum() == batch_size:# all tokens masked:
            #     break
        # print((message != 0).type(torch.FloatTensor).mean(-1))
        # vartiational for predicting the messages without composition
        inde_mu = self.fc_inde_mu(hidden_state)
        inde_var = torch.zeros_like(inde_mu).to(**self.tpdv)
        # inde_var = self.fc_inde_var(hidden_state)
        # inde_std = torch.exp(0.5 * inde_var)
        self.decoding_inde_mu = inde_mu.squeeze().clone()
        self.decoding_inde_log_var = inde_var.squeeze().clone()
        # inde_eps = torch.randn_like(inde_std)
        # self.noncomp_message = inde_eps * inde_std + inde_mu
        self.noncomp_message = inde_mu

        self.message = message.reshape(-1)
        # self.message = message.reshape(batch_size, -1)
        return self.message

    def compositional_loss(self):
        # sum losses for all tokens, EOS loss
        # loss = torch.tensor(0.).to(**self.tpdv)
        # loss for noncomp net
        loss = nn.functional.mse_loss(self.message.squeeze(), self.noncomp_message.squeeze())
        # independence term
        mu_comp = self.decoding_inde_mu
        var_comp = self.decoding_inde_log_var
        # individual message entropy term
        for i, (mu, log_var) in enumerate(zip(self.decoding_mus, self.decoding_log_vars)):
            # discreteness
            token_mse = (mu.reshape(self.nagents,1,self.composition_dim) - self.message_vocabulary).square()
            loss += torch.min(token_mse, 1)[0].mean()
            # nonrandomness
            loss += torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
            # indepdence term
            # mu_comp_i = mu_comp[:,i*self.composition_dim:(i+1)*self.composition_dim]
            # var_comp_i = var_comp[:,i*self.composition_dim:(i+1)*self.composition_dim]
            mu_comp_i = mu_comp[i*self.composition_dim:(i+1)*self.composition_dim]
            var_comp_i = var_comp[i*self.composition_dim:(i+1)*self.composition_dim]
            loss += torch.mean( torch.log(log_var.exp().sqrt() / var_comp_i.exp().sqrt()) + (var_comp_i.exp() + (mu_comp_i - mu)**2) / (2 * log_var.exp()) - 0.5 )
        # self-supervised EOS loss
        loss += self.EOS_token_mse_loss
        # return torch.tensor(0)
        return self.args.comp_beta * loss

    def decode(self):
        if self.args.vae or self.args.use_vqvib:
            return self.decoder_head(self.encoded_info), self.decoding_mu, self.decoding_log_var
        return self.decoder_head(self.encoded_info)

    # def get_gating_mask(self, x, agent_mask):
    #     # Gating
    #     # Hard Attention - action whether an agent communicates or not
    #     comm_prob = None
    #     if self.args.comm_action_one:
    #         comm_action = torch.ones(self.nagents)
    #     elif self.args.comm_action_zero:
    #         comm_action = torch.zeros(self.nagents)
    #     else:
    #         x = x.view(self.nagents, self.hid_size)
    #         comm_prob = F.log_softmax(self.gating_head(x), dim=-1)[0]
    #         comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
    #         comm_prob = comm_prob[:, 1].reshape(self.nagents)
    #         comm_action = comm_prob
    #
    #     comm_action_mask = comm_action.expand(self.nagents, self.nagents)
    #     # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
    #     agent_mask = agent_mask * comm_action_mask.double()
    #     return agent_mask, comm_action, comm_prob

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask']).to(**self.tpdv)
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n).to(**self.tpdv)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, n)
        agent_mask = agent_mask.expand(n, n)

        return num_agents_alive, agent_mask

    def reset_layers(self):
        # reset output layers
        self.init_layer(self.action_head)
        # self.init_layer(self.value_head)
        # self.init_layer(self.attend_obs_intent.unifyheads)

    def init_layer(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        m.bias.data.fill_(0.01)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(1, batch_size * self.nagents, self.hid_size, requires_grad=True).to(**self.tpdv),
                       torch.zeros(1, batch_size * self.nagents, self.hid_size, requires_grad=True).to(**self.tpdv)))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.init_layer(m)

if __name__ == '__main__':

    comm = torch.tensor([[1,1],[2,2],[3,3]])
    n = 3
    comm_dim = 2
    comm = comm.reshape(-1, n, comm_dim)
    b = len(comm)
    # print(comm.shape)
    '''Mask Communication'''
    # mask 1) input communication
    comm_mask = (torch.ones(n, n) \
                        - torch.eye(n, n))
    mask = comm_mask.view(n, n)

    # 2) Mask communcation from dead agents, 3) communication to dead agents
    agent_mask = torch.ones(n)
    num_agents_alive = n
    agent_mask = agent_mask.view(1, n)
    agent_mask = agent_mask.expand(n, n)

    # Mask 1) input communication 2) Mask communcation from dead agents, 3) communication to dead agents
    comm_out_mask = mask * agent_mask * agent_mask.transpose(0, 1)

    '''Perform communication'''
    # doing cts communication vectors only right now
    comm  = comm.view(b, n, comm_dim)
    comm = comm.unsqueeze(-2).expand(b, n, n, comm_dim)
    # print(comm_out_mask.unsqueeze(-1).shape, comm.shape)
    comm = comm * comm_out_mask.unsqueeze(-1).expand_as(comm)
    # print(comm.shape)
    comm = comm.sum(dim=1)
    # print(comm.shape)
    print( comm.reshape(-1, comm_dim) )
    