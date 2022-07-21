import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.transformer import TransformerEncoder, TransformerDecoder
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.comm import MAC

class MAC_T_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(MAC_T_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._active_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = args.num_agents
        self.args = args

        obs_shape = get_shape_from_obs_space(obs_space)
        # embed input state
        self.embed = MLPBase(args, obs_shape)

        self.encoder = TransformerEncoder(args, self._active_func, self._gain, device=device)

        self.communicate = MAC(args, self._active_func, self._gain, device)

        self.decoder = TransformerDecoder(args, self._active_func, self._gain)

        self.action_head = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        # autoencoder
        self.decode_head = nn.Linear(self.hidden_size, obs_shape[0])

        self.to(device)

    def forward(self, obs, seq_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param seq_states: (np.ndarray / torch.Tensor) seq states for transformer.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return seq_states: (torch.Tensor) updated seq hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        seq_states[:, :-1] = seq_states[:, 1:]      # update buffer for new autoregressive observation
        seq_states = check(seq_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.embed(obs)
        seq_states[:, -1] = actor_features.clone()  # add embedded obs
        actor_features = self.encoder(seq_states)
        print("forward", actor_features.shape, seq_states.shape)
        # repeat for R communication rounds (multi-round comm)
        c = actor_features
        for i in range(self.args.comm_rounds):
            c = self.communicate(c)
        actor_features = self.decoder(c, actor_features)
        seq_states[:, -1] = actor_features.clone()  # update for embedded obs + comm

        actions, action_log_probs = self.action_head(actor_features, available_actions, deterministic)
        return actions, action_log_probs, seq_states

    def evaluate_actions(self, obs, seq_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param seq_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        # seq_states[:, :-1] = seq_states[:, 1:]      # update buffer for new autoregressive observation
        seq_states = check(seq_states).to(**self.tpdv)
        print(seq_states.shape)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        print(available_actions.shape)
        # actor_features = self.embed(obs)
        actor_features = self.encoder(seq_states)
        print("eval_act", obs.shape, actor_features.shape, seq_states.shape)
        # repeat for R communication rounds (multi-round comm)
        c = actor_features
        for i in range(self.args.comm_rounds):
            c = self.communicate(c)
        actor_features = self.decoder(c, actor_features)

        action_log_probs, dist_entropy = self.action_head.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        ae_loss = nn.functional.mse_loss(decoded, obs)
        return action_log_probs, dist_entropy, ae_loss


class T_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(T_Critic, self).__init__()
        self.args = args
        self._active_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.embed = MLPBase(args, cent_obs_shape)

        self.encoder = TransformerEncoder(args, self._active_func, self._gain, device=device)

        self.communicate = MAC(args, self._active_func, self._gain, device)

        self.decoder = TransformerDecoder(args, self._active_func, self._gain)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, seq_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param seq_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return seq_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        seq_states[:, :-1] = seq_states[:, 1:]      # update buffer for new autoregressive observation
        seq_states = check(seq_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.embed(cent_obs)
        seq_states[:, -1] = critic_features.clone()  # add embedded obs
        critic_features = self.encoder(seq_states)
        # repeat for R communication rounds (multi-round comm)
        # c = critic_features
        # for i in range(self.args.comm_rounds):
        #     c = self.communicate(c)
        # critic_features = self.decoder(c, critic_features)
        # seq_states[:, -1] = critic_features.clone()  # update for embedded obs + comm

        values = self.v_out(critic_features)

        return values, seq_states
