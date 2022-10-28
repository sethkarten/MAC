import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.comm import MAC
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MAC_R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
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
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        # self.base = base(args, obs_shape)
        use_cnn = False
        if args.env_name == 'PascalVoc' and use_cnn == True:
            self.base = SimpleConv(self.hidden_size)
        else:
            self.base = nn.Linear(obs_shape[0], self.hidden_size)

        self.rnn = RNNLayer(self.hidden_size*2, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.decodeBase = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.communicate = MAC(args, self._active_func, self._gain, device)


        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        if args.env_name == 'PascalVoc':
            self.base2 = nn.Linear(obs_shape[0], self.hidden_size)
            # self.rnn2 = RNNLayer(self.hidden_size*2, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.decodeBase2 = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.communicate2 = MAC(args, self._active_func, self._gain, device)
            self.act2 = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        # autoencoder
        if self.args.use_ae:
            self.decode = nn.Linear(self.hidden_size, obs_shape[0])

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
        if self.args.env_name == 'PascalVoc':
            #AGENT 0
            # obs_copy, rnn_states_copy, masks_copy, available_actions_copy, deterministic_copy = obs, rnn_states, masks, available_actions, deterministic
            obs0 = check(obs[0]).to(**self.tpdv)
            rnn_states0 = check(rnn_states[0]).to(**self.tpdv)
            masks0 = check(masks[0]).to(**self.tpdv)
            if available_actions[0] is not None:
                available_actions0 = check(available_actions[0]).to(**self.tpdv)

            actor_features0 = self.base(obs0)

            #AGENT 1
            obs1 = check(obs[1]).to(**self.tpdv)
            rnn_states1 = check(rnn_states[1]).to(**self.tpdv)
            masks1 = check(masks[1]).to(**self.tpdv)
            if available_actions[1] is not None:
                available_actions1 = check(available_actions[1]).to(**self.tpdv)

            actor_features1 = self.base2(obs1)

            # message encoding

            if self.args.use_compositional:
                agent0_message = self.communicate.compositional_forward(actor_features0)
                agent1_message = self.communicate2.compositional_forward(actor_features1)
            elif self.args.use_vqvib:
                agent0_message = self.communicate.vqvib_forward(actor_features0)
                agent1_message = self.communicate2.vqvib_forward(actor_features1)
            else:
                agent0_message = self.communicate.ae_forward(actor_features0)
                agent1_message = self.communicate2.ae_forward(actor_features1)

            # communicate
            agent0_message, agent1_message = agent1_message, agent0_message
            # comm_encoding = self.communicate(actor_features0)
            # comm_encoding = self.communicate2(actor_features1)

            # AGENT 0
            actor_features0 = torch.cat((comm_encoding, agent0_message), -1)
            actor_features0 = self.decodeBase(actor_features0)
            actions0, action_log_probs0 = self.act(actor_features0, available_actions0, deterministic)

            # AGENT 1
            actor_features1 = torch.cat((comm_encoding, agent1_message), -1)
            actor_features1 = self.decodeBase2(actor_features1)
            actions1, action_log_probs1 = self.act2(actor_features1, available_actions1, deterministic)

            #STACK AND RETURN
            actions, action_log_probs, rnn_states = torch.stack((actions0, actions1)),
            torch.stack((action_log_probs0, action_log_probs1)),
            torch.stack((rnn_states0, rnn_states1))
            return actions, action_log_probs, rnn_states
        else:
            obs = check(obs).to(**self.tpdv)
            rnn_states = check(rnn_states).to(**self.tpdv)
            masks = check(masks).to(**self.tpdv)
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)

            actor_features = self.base(obs)

            # communicate
            comm_encoding = self.communicate(actor_features)
            actor_features = torch.cat((comm_encoding, actor_features), -1)
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)


            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

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
        comm_encoding = self.communicate(actor_features)
        if self.args.use_ae:
            decoded = self.decode(actor_features)
            if self.args.env_name == 'PascalVoc':
                decoded1 = self.decode2(actor_features1)
        actor_features = torch.cat((comm_encoding, actor_features), -1)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        loss = torch.tensor(0).to(**self.tpdv)
        if self.args.use_ae:
            loss = nn.functional.mse_loss(decoded, obs[0])
            if self.args.env_name == 'PascalVoc':
                loss += nn.functional.mse_loss(decoded1, obs[1])
        if self.args.use_vib or self.args.use_vqvib: # KLD
            mu = self.communicate.decoding_mu
            log_var = self.communicate.decoding_log_var
            loss = self.args.beta * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))

            mu1 = self.communicate2.decoding_mu
            log_var1 = self.communicate2.decoding_log_var
            loss += self.args.beta * torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=-1))

        if self.args.use_compositional:
            loss += self.communicate.compositional_loss()
            if self.args.env_name == 'PascalVoc':
                loss += self.communicate2.compositional_loss()

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
        comm_encoding = self.communicate(critic_features)
        critic_features = torch.cat((comm_encoding, critic_features), -1)
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

        self.communicate = MAC(args, self._active_func, self._gain, device)

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
        critic_features = self.communicate(critic_features)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
