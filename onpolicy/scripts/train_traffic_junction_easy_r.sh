#!/bin/sh
env="TrafficJunction"
algo="rmappo"
difficulty="easy"
num_agents=5
episode_length=20
exp="r"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_traffic_junction.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} \
    --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 32 --n_rollout_threads 8 --num_mini_batch 1 \
    --num_env_steps 100000 --ppo_epoch 5 --lr 1e-3 --critic_lr 1e-3 \
    --use_ReLU --gamma 1 --clip_param 0.1 \
    --use_value_active_masks --use_recurrent_policy --use_transformer_policy
done