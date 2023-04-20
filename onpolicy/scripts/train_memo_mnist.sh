#!/bin/sh
env="MNISTMemorization"
algo="memo_ppo"
exp="memo_ppo"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_mnist.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --n_training_threads 16 --n_rollout_threads 16 \
    --num_mini_batch 1 --episode_length 10 --num_env_steps 2000000000 --ppo_epoch 5 \
    --use_value_active_masks --lr 7e-4 --critic_lr 7e-4 
done
