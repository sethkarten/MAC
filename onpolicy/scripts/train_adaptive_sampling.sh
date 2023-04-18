#!/bin/sh
env="MPE"
scenario="simple_adaptive_sampling"  # simple_speaker_listener # simple_reference
num_landmarks=3
num_agents=3
algo="macppo"
exp="adaptive_sampling"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_mpe.py --use_valuenorm --use_popart --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 25 \
    --num_env_steps 20000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 \
    --critic_lr 7e-4 --use_recurrent_policy
done