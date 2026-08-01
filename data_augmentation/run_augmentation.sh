#!/bin/bash
# Coverage-Guided Augmentation pipeline.
#
# Prerequisites:
#   - A user-provided SQL collection at ./data/sql_pool.json
#     (see data_augmentation/README.md for the expected format).
#   - A seed set (e.g. BIRD train) at ./data/train/bird_train.json.
#   - Target databases available for execution verification.
#   - One or more OpenAI-compatible LLM endpoints for synthesis.
#
# Run from the project root:  bash data_augmentation/run_augmentation.sh
set -e

conda activate coalsql

DA=data_augmentation
DATA_DIR=./data
POOL=$DATA_DIR/sql_pool.json
BIRD_TRAIN=$DATA_DIR/train/bird_train.json
BIRD_DATA_DIR=$DATA_DIR/bird/train
AUG_DIR=$DATA_DIR/skeleton_augmentation
TARGET_K=${TARGET_K:-3500}

# LLM endpoints used for synthesis (override via environment variables).
API_URLS=${SYNTH_API_URLS:-http://localhost:8001/v1}
API_MODEL=${SYNTH_API_MODEL:-teacher-model}
API_KEY=${SYNTH_API_KEY:-EMPTY}

# 1. Build a SQL skeleton pool from the user-provided SQL collection.
python $DA/build_skeleton_pool.py \
  --data_path $POOL \
  --seed 42 \
  --num_workers 96

# 2. Embed the deduplicated skeleton pool.
python $DA/embed_skeleton_pool.py \
  --input $AUG_DIR/skeleton_pool_unique.json \
  --output $AUG_DIR/skeleton_pool_embeddings.npy \
  --batch_size 256 \
  --placeholder_style underscore

# 3. Compute kNN-based outlier scores over the pool.
python $DA/analyze_outliers.py --k 10 --top_n 300

# 4. Select complementary skeletons with K-Center Greedy.
python $DA/select_by_kcenter.py \
  --target_k $TARGET_K \
  --outlier_threshold 0.1

# 5. Synthesize verified question-SQL examples from the selected skeletons.
python $DA/synthesize_from_selected.py \
  --selected_skeletons $AUG_DIR/selected_${TARGET_K}_skeletons.json \
  --bird_train $BIRD_TRAIN \
  --bird_data_dir $BIRD_DATA_DIR \
  --output_dir $AUG_DIR/synthesis \
  --target_count $TARGET_K \
  --api_urls $API_URLS \
  --model $API_MODEL \
  --api_key $API_KEY \
  --seed 42
