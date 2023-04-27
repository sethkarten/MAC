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
    python render/render_mpe.py --use_valuenorm --use_popart --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 100 \
    --use_recurrent_policy --use_render --mha_comm --render_episodes 5 \
    --model_dir "/home/milkkarten/research/MAC/onpolicy/scripts/results/MPE/simple_adaptive_sampling/macppo/adaptive_sampling/wandb/latest_run/files"
done