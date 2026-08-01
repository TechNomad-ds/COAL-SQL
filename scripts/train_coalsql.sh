#!/bin/bash
# COAL-SQL full training script:
#   Coverage-Guided Augmentation (complemented data) + Failure-Driven Learning
#   (step-level extra SFT on Solve-None examples + epoch-level error-aware retrieval).

set -x

# Always run from the project root (COAL-SQL/), where .env and ./core live.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# Ray configuration (single node).
# ray stop
# ray start --head --num-cpus=50

# Load machine-specific configuration from .env
source .env
export MODEL_PATH DATA_DIR SQL_DATASET_DIR_DEV SQL_DATASET_DIR_TRAIN SQL_DATASET_DIR_QB WANDB_API_KEY RESULT_DIR
export ERROR_AWARE_LLM_BASE_URL ERROR_AWARE_LLM_API_KEY ERROR_AWARE_LLM_MODEL

# Retrieval augmentation paths (reuse DATA_DIR)
RETRIEVAL_DIR=$DATA_DIR

export EXP_NAME=coalsql_7b
export WANDB_PROJECT="coalsql"

python -u -m verl.coalsql.main_ppo \
    +actor_rollout_ref.actor.sft.sft_loss_type="v0" \
    actor_rollout_ref.actor.sft.sft_epochs=1 \
    actor_rollout_ref.actor.sft.sft_data_size=64 \
    actor_rollout_ref.actor.sft.sft_mini_batch_size=64 \
    actor_rollout_ref.actor.sft.sft_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.sft.entropy_coeff=0.0001 \
    actor_rollout_ref.actor.sft.grad_clip=0.5 \
    actor_rollout_ref.actor.optim.sft.lr=1e-6 \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train/train_random3000_tag3_3575.parquet \
    data.val_files=$DATA_DIR/train/dev.parquet \
    data.truncation='left' \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    data.max_target_length=2048 \
    data.target_key="target" \
    reward_model.reward_manager="batch" \
    custom_reward_function.path="./core/reward/text2sql_reward_batch.py" \
    custom_reward_function.name="text2sql_reward_batch_func" \
    custom_reward_function.overlong_buffer.enable=False \
    custom_reward_function.overlong_buffer.len=4096 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.sft_param_offload=False \
    actor_rollout_ref.actor.fsdp_config.sft_optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.sft_use_lora=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=14336 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    trainer.critic_warmup=0 \
    +trainer.error_log_dir=${RESULT_DIR}/${WANDB_PROJECT}/${EXP_NAME}/error_logs \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.test_freq=20 \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${RESULT_DIR}/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.retrieval_augmentation.enable=True \
    trainer.retrieval_augmentation.faiss_index_path=$RETRIEVAL_DIR/skeleton_index/skeleton.index \
    trainer.retrieval_augmentation.metadata_path=$RETRIEVAL_DIR/skeleton_index/skeleton_metadata.json \
    trainer.retrieval_augmentation.question_bank_parquet=$RETRIEVAL_DIR/question_bank/question_bank_filtered.parquet \
    trainer.retrieval_augmentation.train_embeddings_path=$RETRIEVAL_DIR/train/train_skeleton_embeddings.npy \
    trainer.retrieval_augmentation.train_idx_mapping_path=$RETRIEVAL_DIR/train/bird_idx_to_parquet_row.json \
    trainer.retrieval_augmentation.qb_embeddings_path=$RETRIEVAL_DIR/skeleton_index/qb_skeleton_query_embeddings.npy \
    trainer.retrieval_augmentation.top_k=5 \
    trainer.retrieval_augmentation.num_per_epoch=512 \
    trainer.retrieval_augmentation.rank_weights=[0.35,0.25,0.20,0.10,0.10] \
    trainer.retrieval_augmentation.seed=42 \
    trainer.retrieval_augmentation.random_ratio=0.5 \
    trainer.retrieval_augmentation.mode=extra_sft \
    trainer.retrieval_augmentation.extra_sft.eval_before=True \
    trainer.retrieval_augmentation.extra_sft.eval_after=True \
    trainer.retrieval_augmentation.extra_sft.batch_size=64 \
    trainer.retrieval_augmentation.error_aware.enable=True \
    trainer.retrieval_augmentation.error_aware.llm_api.base_url="${ERROR_AWARE_LLM_BASE_URL}" \
    trainer.retrieval_augmentation.error_aware.llm_api.api_key="${ERROR_AWARE_LLM_API_KEY}" \
    trainer.retrieval_augmentation.error_aware.llm_api.model="${ERROR_AWARE_LLM_MODEL}" \
    trainer.retrieval_augmentation.error_aware.llm_api.max_tokens=16384 \
    trainer.retrieval_augmentation.error_aware.llm_api.timeout=600 \
    trainer.retrieval_augmentation.error_aware.max_workers=128 \
    trainer.retrieval_augmentation.error_aware.max_retries=3 \
    trainer.retrieval_augmentation.error_aware.structural_error_ids="[17,18,19,20,21,22,23,24,25,26,27,28,29,30]" \
    trainer.retrieval_augmentation.error_aware.error_aware_ratio=0.3 \
    trainer.retrieval_augmentation.error_aware.rerank_batch_size=20 \
    trainer.retrieval_augmentation.error_aware.rerank_multiplier=3 \
    trainer.total_epochs=8 \
    trainer.total_training_steps=240 $@ 2>&1 | tee logs/${EXP_NAME}.log
