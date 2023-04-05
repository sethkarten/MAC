#!/bin/sh
env="TrafficJunction"
algo="macppo"
difficulty="easy"
num_agents=5
episode_length=20
exp="r"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python eval/eval_traffic_junction.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} \
    --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 32 --n_rollout_threads 1024 --num_mini_batch 1 \
    --num_env_steps 100000000 --ppo_epoch 5 --lr 1e-3 --critic_lr 1e-3 \
    --use_ReLU --gamma 1 --clip_param 0.1 --comm_dim 1 --mha_comm --use_eval \
    --use_value_active_masks --use_recurrent_policy --use_transformer_policy \
    --n_eval_rollout_threads 16 --eval_episodes 64 \
    --model_dir /home/milkkarten/research/MAC/onpolicy/scripts/results/TrafficJunction/easy/rmappo/mlp/wandb/run-20220714_221718-2vnrki9b/files
done