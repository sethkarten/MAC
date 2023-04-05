#!/bin/sh
env="TrafficJunction"
algo="r_mappo_comm"
difficulty="medium"
num_agents=10
episode_length=40
exp="test_iIMAC"
seed_max=10

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_traffic_junction.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} \
    --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 32 --n_rollout_threads 16 --num_mini_batch 1 \
    --num_env_steps 1000000 --ppo_epoch 5 --lr 1e-3 --critic_lr 1e-3 \
    --use_ReLU --gamma 1 --clip_param 0.2 \
    --use_value_active_masks --hidden_size 64 --mha_comm \
    --use_vib --use_recurrent_policy --use_transformer_policy
done
