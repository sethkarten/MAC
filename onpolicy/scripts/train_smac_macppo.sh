#!/bin/sh
env="StarCraft2"
map="8m"
algo="macppo"
exp="macppo"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
    --num_mini_batch 1 --episode_length 120 --num_env_steps 20000000 --ppo_epoch 5 \
    --use_value_active_masks --comm_dim 16 
done
