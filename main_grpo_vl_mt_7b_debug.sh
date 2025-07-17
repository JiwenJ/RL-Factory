set -e -x
# source /mnt/dolphinfs/hdd_pool/docker/share/jjw/miniconda3/bin/activate
# conda activate rl
# ray stop --force
# export HF_HUB_OFFLINE=1
# QwenVL 3B -> 90 GB 1 GPU
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Models/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/share/jjw/huggingface.co/Qwen/Qwen2.5-0.5B
MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Models/Qwen2.5-VL-7B-Instruct
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/share/jjw/miniconda3/lib:$LD_LIBRARY_PATH 
export REWARD_MODEL_PATH=/your/path/to/Qwen/QwQ-32B
export HYDRA_FULL_ERROR=1
DATE=$(date +"%Y-%m-%d-%H:%M:%S")
export RAY_DEBUG=legacy
# export http_proxy=http://172.18.128.99:8420
# export https_proxy=https://172.18.128.99:8420
export RAY_DEDUP_LOGS=0
export WANDB_API_KEY="76ecf2334073036f76da7b9e4eb5bbe934767728"
# export WANDB_MODE="offline"
export WANDB_DIR=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/wandb
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Data/geo3kv14
DATA=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Data/textvqav21
# export CUDA_VISIBLE_DEVICES="6,7"
# export RAY_memory_usage_threshold
# export RAY_memory_monitor_refresh_ms=0
# export CUDA_VISIBLE_DEVICES=0,1
TP=2
Multiple=128
Val_Multiple=64
MINI=4
# RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
python3 -m verl.trainer.main_ppo\
    algorithm.adv_estimator=grpo\
    trainer.default_local_dir=/mnt/dolphinfs/hdd_pool/docker/share/JJW/RL/temp\
    data.train_files=$DATA/train.parquet\
    data.val_files=$DATA/test.parquet\
    data.train_batch_size=$((TP * Multiple))\
    data.val_batch_size=$((TP * Val_Multiple))\
    data.max_prompt_length=8192\
    data.max_response_length=8192\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.model.enable_activation_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.ref.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TP * MINI))\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.001\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP}\
    actor_rollout_ref.rollout.max_num_batched_tokens=32768\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.top_p=0.999\
    actor_rollout_ref.rollout.top_k=-1\
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1\
    actor_rollout_ref.rollout.val_kwargs.n=1\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7\
    actor_rollout_ref.rollout.n=2\
    actor_rollout_ref.rollout.max_turns=3\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.rollout.multi_turn.enable=True\
    actor_rollout_ref.env.name=search_vl\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen2_5_vl\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/RL-Factory/envs/configs/mcp_tools1.pydata\
    actor_rollout_ref.env.use_process_reward=False\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=${TP}\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['console','tensorboard','wandb']\
    trainer.project_name='GRPO_Visual'\
    trainer.experiment_name="Visual_7B_${DATE}"\
    trainer.n_gpus_per_node=8\
    trainer.nnodes=1\
    trainer.val_before_train=False\
    trainer.default_local_dir=/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/RL-Factory/ckpts/${DATE}_7B_Qwen25VL\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=50\
    trainer.test_freq=3\
    trainer.total_epochs=10 $@ 2>&1 | tee /mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/RL-Factory/tmp/logs/${DATE}_grpo.log
# | tee /mnt/dolphinfs/hdd_pool/docker/share/JJW/RL/temp/grpo.log
    # actor_rollout_ref.model.enable_gradient_checkpointing=True\
    # actor_rollout_ref.model.enable_activation_offload=True\
    # critic.model.enable_activation_offload=True\
    #     actor_rollout_ref.ref.ulysses_sequence_parallel_size=${TP}\
    # +ulysses_sequence_parallel_size=1\     trainer.logger=['tensorboard','wandb']\
    #     actor_rollout_ref.actor.fsdp_config.param_offload=True\
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    # # +actor_rollout_ref.ref.entropy_from_logits_with_chunking=True\
    # +actor_rollout_ref.actor.entropy_checkpointing=True\