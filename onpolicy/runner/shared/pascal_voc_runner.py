import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.util import set_lr
import copy
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def _t2n(x):
    return x.detach().cpu().numpy()

class PascalVocRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for PascalVoc. See parent class for details."""
    def __init__(self, config):
        super(PascalVocRunner, self).__init__(config)
        self.best = 0

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        success = np.zeros(self.n_rollout_threads)
        success_eps = 0
        if self.all_args.contrastive:
            rr_available_actions = self.buffer.rr_available_actions[0]
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            success_one_ep = np.ones(self.n_rollout_threads)
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                if self.all_args.contrastive:
                    rr_available_actions = rr_available_actions.sum(-1)
                    rr_available_actions = rr_available_actions.reshape(self.all_args.n_rollout_threads*self.num_agents)
                    rr_actions = np.random.randint(np.zeros_like(rr_available_actions), rr_available_actions)
                    rr_actions = rr_actions.reshape(self.all_args.n_rollout_threads, self.num_agents, 1)
                    rr_obs, rr_share_obs, rr_rewards, rr_dones, rr_infos, rr_available_actions = self.envs_contrastive.step(rr_actions)
                    rr_dones_env = np.all(rr_dones, axis=1)
                    rr_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    rr_masks[rr_dones_env == True] = np.zeros(((rr_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    data_rand_rollout = (rr_obs, rr_share_obs, rr_masks)
                    # if step+1 < self.episode_length - self.all_args.lookahead:
                    #     data_rand_rollout = self.rand_rollout(available_actions, actions, step)
                    # else:
                    #     data_rand_rollout = (None, None, None)
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic, data_rand_rollout
                else:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic
                for i, info in enumerate(infos):
                    success_one_ep[i] = min(success_one_ep[i], info[0]['success'])
                # insert data into buffer
                self.insert(data)
            success += success_one_ep
            success_eps += 1

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Difficulty {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.difficulty,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "TrafficJunction":
                    success_rate = np.mean(success / success_eps)
                    if success_rate > self.best:
                        self.best = success_rate
                        self.save_best()
                    if success_rate >= .97:
                        # decrease learning rate significantly
                        print("Setting lr to 1e-5")
                        set_lr(self.trainer.policy.actor_optimizer, 1e-5)
                        set_lr(self.trainer.policy.critic_optimizer, 1e-5)
                    success = np.zeros(self.n_rollout_threads)
                    success_eps = 0
                    print("Success rate is {}.".format(success_rate))
                    if self.use_wandb:
                        wandb.log({"success": success_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("success", {"success": success_rate}, total_num_steps)

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def rand_rollout(self, available_actions, _actions, step):
        envs = self.envs.get()
        self.envs_contrastive.set(envs)
        data_rand_rollout_obs = []
        data_rand_rollout_share_obs = []
        data_rand_rollout_masks = []
        for step_rand in range(step+1, min(step+1+self.all_args.lookahead, self.episode_length)):
            available_actions = available_actions.sum(-1)
            available_actions = available_actions.reshape(self.all_args.n_rollout_threads*self.num_agents)
            actions_rand = np.random.randint(np.zeros_like(available_actions), available_actions)
            actions_rand = actions_rand.reshape(self.all_args.n_rollout_threads, self.num_agents, 1)
            obs, share_obs, rewards, dones, infos, available_actions = self.envs_contrastive.step(actions_rand)
            dones_env = np.all(dones, axis=1)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            data_rand_rollout_obs.append(obs)
            data_rand_rollout_share_obs.append(share_obs)
            data_rand_rollout_masks.append(masks)
        data_rand_rollout_obs = np.stack(data_rand_rollout_obs)
        data_rand_rollout_share_obs = np.stack(data_rand_rollout_share_obs)
        data_rand_rollout_masks = np.stack(data_rand_rollout_masks)
        return (data_rand_rollout_obs, data_rand_rollout_share_obs, data_rand_rollout_masks)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        # print(share_obs.shape, self.buffer.share_obs[0].shape, self.buffer.obs[0].shape)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

        if self.all_args.contrastive:
            rr_obs, rr_share_obs, rr_available_actions = self.envs_contrastive.reset()
            if not self.use_centralized_V:
                rr_share_obs = rr_obs
            self.buffer.rr_share_obs[0] = rr_share_obs.copy()
            self.buffer.rr_obs[0] = rr_obs.copy()
            self.buffer.rr_available_actions[0] = rr_available_actions.copy()


    @torch.no_grad()
    def collect(self, step):
        # TODO: add transformer buffer states
        self.trainer.prep_rollout()
        # print('collect', self.buffer.share_obs[step].shape)
        if self.all_args.use_transformer_policy:
            value, action, action_log_prob, seq_state, seq_state_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.seq_states[step]),
                                                np.concatenate(self.buffer.seq_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]))
            seq_states = np.array(np.split(_t2n(seq_state), self.n_rollout_threads))
            seq_states_critic = np.array(np.split(_t2n(seq_state_critic), self.n_rollout_threads))
        else:
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.rnn_states[step]),
                                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]))
            rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        if self.all_args.use_transformer_policy:
            return values, actions, action_log_probs, seq_states, seq_states_critic
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        if self.all_args.use_transformer_policy:
            if self.all_args.contrastive:
                obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, seq_states, seq_states_critic, data_rand_rollout = data
            else:
                obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, seq_states, seq_states_critic = data
        else:
            if self.all_args.contrastive:
                obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic, data_rand_rollout = data
            else:
                obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        if self.all_args.use_transformer_policy:
            seq_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.seq_states.shape[3:]), dtype=np.float32)
            seq_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.seq_states_critic.shape[3:]), dtype=np.float32)
        else:
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        if self.all_args.contrastive:
            drr_obs, drr_share_obs, drr_masks = data_rand_rollout
        else:
            drr_obs, drr_share_obs, drr_masks = None, None, None

        if self.all_args.use_transformer_policy:
            self.buffer.insert(share_obs, obs, seq_states, seq_states_critic,
                               actions, action_log_probs, values, rewards, masks,
                               bad_masks, active_masks, available_actions, drr_obs, drr_share_obs, drr_masks)
        else:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                               actions, action_log_probs, values, rewards, masks,
                               bad_masks, active_masks, available_actions, drr_obs, drr_share_obs, drr_masks)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_episode_success = np.zeros(self.n_rollout_threads)
        one_episode_success = np.ones(self.n_rollout_threads)

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)
            for i, info in enumerate(eval_infos):
                one_episode_success[i] = min(one_episode_success[i], info[0]['success'])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    eval_episode_success += one_episode_success
                    one_episode_success = np.ones(self.n_rollout_threads)
                    one_episode_rewards = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                print(eval_episode_success)
                eval_episode_success /= eval_episode
                print(eval_episode_success)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards, 'eval_average_episode_success': eval_episode_success}
                self.log_env(eval_env_infos, total_num_steps)

                success_rate = eval_episode_success.mean()
                print("Eval success rate is {}.".format(success_rate))
                if self.use_wandb:
                    wandb.log({"eval_success": success_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_success", {"eval_success": success_rate}, total_num_steps)

                break
