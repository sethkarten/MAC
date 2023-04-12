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
    --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 \
    --use_value_active_masks --lr 5e-4 --critic_lr 5e-4 --clip_param 0.1 \
    --use_ReLU --hidden_size 64 --use_recurrent_policy 
done
