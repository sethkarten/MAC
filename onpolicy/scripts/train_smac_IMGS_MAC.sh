#!/bin/sh
env="StarCraft2"
map="8m"
algo="r_mappo"
exp="r_mappo"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 32 --n_rollout_threads 16 \
    --num_mini_batch 1 --episode_length 120 --num_env_steps 10000000 --ppo_epoch 5 \
    --use_value_active_masks --use_ReLU --mha_comm \
    --use_recurrent_policy --use_transformer_policy
done
