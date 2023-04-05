#!/bin/sh
env="TrafficJunction"
algo="r_mappo_comm"
difficulty="easy"
num_agents=5
episode_length=20
exp="rl_comm"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_pascal_voc.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} \
    --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 16 --n_rollout_threads 16 --num_mini_batch 1 \
    --num_env_steps 10000000 --ppo_epoch 5 --lr 1e-5 --critic_lr 1e-5 \
    --use_ReLU --gamma 1 --clip_param 0.2  --mha_comm \
    --use_value_active_masks --hidden_size 128 --use_recurrent_policy
done
