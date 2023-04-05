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
import torchvision


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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_resnet(name, pretrained=False):
    resnets = {
        'resnet18': torchvision.models.resnet18(pretrained=pretrained),
        'resnet34': torchvision.models.resnet34(pretrained=pretrained),
        'resnet50': torchvision.models.resnet50(pretrained=pretrained),
        'resnet101': torchvision.models.resnet101(pretrained=pretrained),
        'resnet152': torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f'{name} is not a valid ResNet version')
    return resnets[name]

def get_vgg(name, pretrained=False):
    vggs = {
        'vgg11': torchvision.models.vgg11(pretrained=pretrained),
        'vgg11_bn': torchvision.models.vgg11_bn(pretrained=pretrained),
        'vgg13': torchvision.models.vgg13(pretrained=pretrained),
        'vgg13_bn': torchvision.models.vgg13_bn(pretrained=pretrained),
        'vgg16': torchvision.models.vgg16(pretrained=pretrained),
        'vgg16_bn': torchvision.models.vgg16_bn(pretrained=pretrained),
        'vgg19': torchvision.models.vgg19(pretrained=pretrained),
        'vgg19_bn': torchvision.models.vgg19_bn(pretrained=pretrained),
    }
    if name not in vggs.keys():
        raise KeyError(f'{name} is not a valid VGG version')
    return vggs[name]

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016)
    to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the
    average pooling layer.
    """

    def __init__(self, encoder_name='resnet18', projection_dim=128):
        super(SimCLR, self).__init__()

        if encoder_name.startswith('resnet'):
            self.encoder = get_resnet(encoder_name, pretrained=True)
        elif encoder_name.startswith('vgg'):
            self.encoder = get_vgg(encoder_name, pretrained=True)
        else:
            raise NotImplementedError

        #Freeze weights
        # self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = True

        #Eval mode for BN layers
        # def set_bn_eval(module):
        #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #         module.eval()
        # self.encoder.apply(set_bn_eval)

        # get dimensions of last fully-connected layer of encoder
        # (2048 for resnet50, 512 for resnet18)
        if encoder_name.startswith('resnet'):
            self.n_features = self.encoder.fc.in_features
        elif encoder_name.startswith('vgg'):
            self.n_features = self.encoder.classifier[-1].in_features
        else:
            raise NotImplementedError

        # replace the fc layer with an Identity function
        if encoder_name.startswith('resnet'):
            self.encoder.fc = Identity()
        elif encoder_name.startswith('vgg'):
            self.encoder.classifier[-1] = Identity()
        else:
            raise NotImplementedError

        # use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i):
        h_i = self.encoder(x_i)
        #h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        #z_j = self.projector(h_j)
        # return h_i, z_i
        return z_i

class MAC_R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        args.comm_dim = 128
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
        if args.env_name == 'PascalVoc' and self.use_cnn == True:
            # self.base = SimpleConv(self.hidden_size)
            self.base = SimCLR(projection_dim=self.hidden_size, encoder_name='resnet18')
        else:
            self.base = nn.Linear(obs_shape[0], self.hidden_size)

        self.rnn = RNNLayer(self.hidden_size*2, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.decodeBase = nn.Linear(self.hidden_size + args.comm_dim, self.hidden_size)

        self.communicate = MAC(args, self.hidden_size, device=device)


        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        if args.env_name == 'PascalVoc':
            if args.env_name == 'PascalVoc' and self.use_cnn == True:
                # self.base2 = SimpleConv(self.hidden_size)
                self.base2 = SimCLR(projection_dim=self.hidden_size, encoder_name='resnet18')
            else:
                self.base2 = nn.Linear(obs_shape[0], self.hidden_size)
            # self.rnn2 = RNNLayer(self.hidden_size*2, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.decodeBase2 = nn.Linear(self.hidden_size + args.comm_dim, self.hidden_size)
            self.communicate2 = MAC(args, self.hidden_size, device=device)
            self.act2 = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        # autoencoder
        if self.args.use_ae:
            self.decode = nn.Linear(self.comm_dim, obs_shape[0])
            if args.env_name == 'PascalVoc':
                self.decode2 = nn.Linear(self.comm_dim, obs_shape[0])

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
            if self.use_cnn == True:
                obs = obs.reshape(2*self.args.n_rollout_threads, 1, 3, 32, 32) if self.obs_shape[0] == 32*32*3 else obs.reshape(2*self.args.n_rollout_threads, 1, 3, 375, 500)
            #AGENT 0
            # obs_copy, rnn_states_copy, masks_copy, available_actions_copy, deterministic_copy = obs, rnn_states, masks, available_actions, deterministic
            obs0 = check(obs[0::2]).to(**self.tpdv)
            rnn_states0 = check(rnn_states[0::2]).to(**self.tpdv)
            masks0 = check(masks[0::2]).to(**self.tpdv)
            if available_actions[0::2] is not None:
                available_actions0 = check(available_actions[0::2]).to(**self.tpdv)

            actor_features0 = self.base(obs0) if self.use_cnn == False else self.base(obs0.reshape((-1, 3, 32, 32)))

            #AGENT 1
            obs1 = check(obs[1::2]).to(**self.tpdv)
            rnn_states1 = check(rnn_states[1::2]).to(**self.tpdv)
            masks1 = check(masks[1::2]).to(**self.tpdv)
            if available_actions[1::2] is not None:
                available_actions1 = check(available_actions[1::2]).to(**self.tpdv)

            actor_features1 = self.base2(obs1) if self.use_cnn == False else self.base2(obs1.reshape((-1, 3, 32, 32)))

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
            actor_features0 = torch.cat((actor_features0, agent0_message), -1)
            actor_features0 = self.decodeBase(actor_features0)
            actions0, action_log_probs0 = self.act(actor_features0, available_actions0, deterministic)

            # AGENT 1
            actor_features1 = torch.cat((actor_features1, agent1_message), -1)
            actor_features1 = self.decodeBase2(actor_features1)
            actions1, action_log_probs1 = self.act2(actor_features1, available_actions1, deterministic)

            #STACK AND RETURN
            actions, action_log_probs, rnn_states = torch.stack((actions0, actions1)), torch.stack((action_log_probs0, action_log_probs1), dim=1), torch.stack((rnn_states0, rnn_states1))
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
        if self.args.env_name == 'PascalVoc':
            if self.use_cnn == True:
                obs0_prior = check(obs[0::2]).to(**self.tpdv)
                obs1_prior = check(obs[1::2]).to(**self.tpdv)
                obs = obs.reshape(2*self.args.n_rollout_threads, 1, 3, 32, 32) if self.obs_shape[0] == 32*32*3 else obs.reshape(2*self.args.n_rollout_threads, 1, 3, 375, 500)
            action = check(action).to(**self.tpdv)
            if available_actions is not None:
                available_actions = check(available_actions).to(**self.tpdv)
            if active_masks is not None:
                active_masks = check(active_masks).to(**self.tpdv)

            #AGENT 0
            obs0 = check(obs[0::2]).to(**self.tpdv)
            #rnn_states0 = check(rnn_states[0]).to(**self.tpdv)
            masks0 = check(masks[0::2]).to(**self.tpdv)
            # if available_actions[0] is not None:
            #     available_actions0 = check(available_actions[0]).to(**self.tpdv)

            actor_features0 = self.base(obs0) if self.use_cnn == False else self.base(obs0.reshape((-1, 3, 32, 32)))

            #AGENT 1
            obs1 = check(obs[1::2]).to(**self.tpdv)
            #rnn_states1 = check(rnn_states[1]).to(**self.tpdv)
            masks1 = check(masks[1::2]).to(**self.tpdv)
            # if available_actions[1] is not None:
            #     available_actions1 = check(available_actions[1]).to(**self.tpdv)

            actor_features1 = self.base2(obs1) if self.use_cnn == False else self.base2(obs1.reshape((-1, 3, 32, 32)))

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

            # loss then communicate
            loss = torch.tensor(0).to(**self.tpdv)
            if self.args.use_ae:
                decoded = self.decode(agent0_message)
                decoded1 = self.decode2(agent1_message)
                loss = nn.functional.mse_loss(decoded, obs0_prior)
                loss += nn.functional.mse_loss(decoded1, obs1_prior)

            agent0_message, agent1_message = agent1_message, agent0_message

            # AGENT 0
            actor_features0 = torch.cat((actor_features0, agent0_message), -1)
            actor_features0 = self.decodeBase(actor_features0)
            # AGENT 1
            actor_features1 = torch.cat((actor_features1, agent1_message), -1)
            actor_features1 = self.decodeBase2(actor_features1)

            # actor_features = torch.stack((actor_features0, actor_features1))
            actor_features = torch.cat((actor_features0, actor_features1))
        else:
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
            actor_features = torch.cat((comm_encoding, actor_features), -1)
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

            loss = torch.tensor(0).to(**self.tpdv)
            if self.args.use_ae:
                loss = nn.functional.mse_loss(decoded, obs)

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
        if self.args.env_name == 'PascalVoc':
            if self.use_cnn == True:
                obs = obs.reshape(2*self.args.n_rollout_threads, 1, 3, 32, 32) if self.obs_shape[0] == 32*32*3 else obs.reshape(2*self.args.n_rollout_threads, 1, 3, 375, 500)
            #AGENT 0
            obs0 = check(obs[0::2]).to(**self.tpdv)
            # rnn_states0 = check(rnn_states[0]).to(**self.tpdv)
            masks0 = check(masks[0::2]).to(**self.tpdv)

            critic_features0 = self.base(obs0) if self.use_cnn == False else self.base(obs0.reshape((-1, 3, 32, 32)))

            #AGENT 1
            obs1 = check(obs[1::2]).to(**self.tpdv)
            # rnn_states1 = check(rnn_states[1]).to(**self.tpdv)
            masks1 = check(masks[1::2]).to(**self.tpdv)

            critic_features1 = self.base2(obs1) if self.use_cnn == False else self.base2(obs1.reshape((-1, 3, 32, 32)))

            # message encoding

            if self.args.use_compositional:
                agent0_message = self.communicate.compositional_forward(critic_features0)
                agent1_message = self.communicate2.compositional_forward(critic_features1)
            elif self.args.use_vqvib:
                agent0_message = self.communicate.vqvib_forward(critic_features0)
                agent1_message = self.communicate2.vqvib_forward(critic_features1)
            else:
                agent0_message = self.communicate.ae_forward(critic_features0)
                agent1_message = self.communicate2.ae_forward(critic_features1)

            # communicate
            agent0_message, agent1_message = agent1_message, agent0_message

            # AGENT 0
            critic_features0 = torch.cat((critic_features0, agent0_message), -1)
            critic_features0 = self.decodeBase(critic_features0)

            # AGENT 1
            critic_features1 = torch.cat((critic_features1, agent1_message), -1)
            critic_features1 = self.decodeBase2(critic_features1)

            # note that these are swapped for a test
            value0 = self.v_out(critic_features1)
            value1 = self.v_out(critic_features0)

            values = torch.cat((value0, value1))

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
        else:
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

        self.communicate = MAC(args, self.hidden_size, device=device)

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
