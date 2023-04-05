#!/bin/sh
env="Hanabi"
hanabi="Hanabi"
num_agents=2
algo="mappo"
exp="mlp_critic1e-3_entropy0.015_v0belief"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_hanabi_forward.py --env_name "Hanabi" --algorithm_name "mappo" --experiment_name "exp1" --hanabi_name "Hanabi-Very-Small"\
    --num_agents 2 --seed 4 --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 100001\
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --user_name 'milkkarten' --num_agents 2
    echo "training is done!"
done
