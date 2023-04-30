#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=3
num_agents=3
algo="macppo"
exp="adaptive_sampling"
seed_max=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python render/render_mpe.py --use_valuenorm --use_popart --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 20000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 1e-3 \
    --critic_lr 1e-3 --use_recurrent_policy --mha_comm \
    --use_render --render_episodes 1 --save_gifs \
    --model_dir "/home/skailas/CMU/Research/MAC/MAC_v2/MAC/onpolicy/scripts/results/MPE/simple_spread/macppo/adaptive_sampling_reconstruction/wandb/run-20230430_053526-2kx9bk7j/files"

done