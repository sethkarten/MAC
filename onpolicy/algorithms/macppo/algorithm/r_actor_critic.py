import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase, MLPLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.comm import MAC
import torch.nn.functional as F
import torchvision


class MAC_R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        self.comm_dim = args.comm_dim
        self.use_cnn = True

        super(MAC_R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._active_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.num_agents = args.num_agents
        self.args = args

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_shape = obs_shape
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        # self.base = base(args, obs_shape)
        self.base = nn.Linear(obs_shape[0], self.hidden_size)

        self.rnn = RNNLayer(self.hidden_size + args.comm_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.decodeBase = nn.Linear(self.hidden_size + args.comm_dim, self.hidden_size)

        self.communicate = MAC(args, self.hidden_size, device=device)


        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        if self.args.env_name == 'MPE':
            
            self.reconstruct = MLPLayer(self.hidden_size, args.env_size**2, 3, True, True)

        # autoencoder
        if self.args.use_ae:
            self.decode = nn.Linear(self.comm_dim, obs_shape[0])

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        # communicate
        comm_encoding = self.communicate.default_forward(actor_features)
        messages = self.communicate.communicate(comm_encoding)
        # print(messages.shape, actor_features.shape)
        actor_features = torch.cat((messages, actor_features), -1)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)


        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        if self.args.env_name == 'MPE':
            # reconstruction of env
            reconstruction = self.reconstruct(actor_features)
            return actions, action_log_probs, rnn_states, reconstruction
        else:
            return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        # communicate
        comm_encoding = self.communicate.default_forward(actor_features)
        messages = self.communicate.communicate(comm_encoding)
        if self.args.use_ae:
            decoded = self.decode(comm_encoding)
        actor_features = torch.cat((messages, actor_features), -1)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        loss = torch.tensor(0).to(**self.tpdv)
        if self.args.use_ae:
            loss = nn.functional.mse_loss(decoded, obs)

        # if self.args.use_vib or self.args.use_vqvib: # KLD
        #     mu = self.communicate.decoding_mu
        #     log_var = self.communicate.decoding_log_var
        #     loss = self.args.beta * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))

        #     mu1 = self.communicate2.decoding_mu
        #     log_var1 = self.communicate2.decoding_log_var
        #     loss += self.args.beta * torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=-1))

        # if self.args.use_compositional:
        #     loss += self.communicate.compositional_loss()

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy, loss

    def critic(self, obs, rnn_states, masks, r_obs=None, f_obs=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        # communicate
        comm_encoding = self.communicate.default_forward(critic_features)
        messages = self.communicate.communicate(comm_encoding)
        critic_features = torch.cat((messages, critic_features), -1)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        if self.args.contrastive and r_obs is not None:
            r_obs = check(r_obs).to(**self.tpdv)
            r_enc = self.base(r_obs)
            contrast_rand_loss = -torch.log(1-torch.sigmoid(critic_features.T @ r_enc) + 1e-9)
            f_obs = check(f_obs).to(**self.tpdv)
            f_obs = self.base(f_obs)
            contrast_future_loss = -torch.log(torch.sigmoid(critic_features.T @ f_obs) + 1e-9)
        else:
            contrast_rand_loss = torch.tensor(0).to(**self.tpdv)
            contrast_future_loss = torch.tensor(0).to(**self.tpdv)

        return values, rnn_states, contrast_rand_loss, contrast_future_loss
        # return values, rnn_states

class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._active_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)
        # self.base = nn.Linear(cent_obs_shape[0], self.hidden_size)

        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # self.communicate = MAC(args, self.hidden_size, device=device)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        # communicate
        # critic_features = self.communicate(critic_features)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
