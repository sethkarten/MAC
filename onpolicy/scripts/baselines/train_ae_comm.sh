#!/bin/sh
env="FindGoal"
algo="r_mappo_comm"
difficulty="hard"
num_agents=3
episode_length=80
exp="ae_comm"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train/train_find_goal.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --difficulty ${difficulty} \
    --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 \
    --num_env_steps 1000000 --ppo_epoch 5 --lr 1e-3 --critic_lr 1e-3 \
    --use_ReLU --gamma 1 --clip_param 0.1 \
    --use_value_active_masks --hidden_size 128 --mha_comm \
    --use_ae --use_recurrent_policy --use_transformer_policy \
    --cuda --set num_workers 1 env_cfg.comm_len 10
done
