#!/bin/sh
env="TrafficJunction"
algo="mappo"
difficulty="easy"
num_agents=5
episode_length=20
exp="mlp"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_traffic_junction.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} --n_training_threads 127 --n_rollout_threads 8 --num_mini_batch 1 --num_env_steps 10000000 --ppo_epoch 50 --lr 1e-3 --critic_lr 1e-3 --use_value_active_masks --use_eval --use_recurrent_policy
done
