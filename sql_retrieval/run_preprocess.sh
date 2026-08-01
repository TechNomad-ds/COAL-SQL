#!/bin/bash
# Build the SQL skeleton retrieval index from a user-provided question bank.
#
# Prerequisite: prepare a question bank at ./data/question_bank/question_bank.json
#   Each entry must contain at least: a gold SQL query, its db_id, and the
#   natural-language question. All SQLs should be executable on the target
#   databases. See sql_retrieval/README.md ("Question Bank input") for the format.
set -e

conda activate coalsql

# 1. SQL -> skeleton
python build_skeletons.py

# 2. Build the FAISS skeleton index
python build_index.py --gpu_id 0 --batch_size 256

# 3. Precompute skeleton embeddings for the training set and the question bank
python precompute_embeddings.py --num_gpus 8 --procs_per_gpu 2 --batch_size 256
