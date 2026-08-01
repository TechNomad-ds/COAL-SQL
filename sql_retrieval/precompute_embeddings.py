#!/usr/bin/env python3
"""
Precompute skeleton embeddings for the training set and the question bank.

For the training set:
  1. Extract SQL from train.parquet -> sql2skeleton -> skeleton
  2. Symmetric encoding with Qwen3-Embedding (no Instruct prefix) -> train_skeleton_embeddings.npy
  3. Generate the bird_idx_to_parquet_row.json mapping

For the question bank:
  1. Read existing skeletons directly from skeletons.json
  2. Symmetric encoding with Qwen3-Embedding -> qb_skeleton_query_embeddings.npy

Outputs:
  ./data/train/
    ├── train_skeletons.json
    ├── train_skeleton_embeddings.npy
    └── bird_idx_to_parquet_row.json

  ./data/skeleton_index/
    └── qb_skeleton_query_embeddings.npy

Usage:
    conda activate coalsql
    python precompute_embeddings.py [--num_gpus 8] [--procs_per_gpu 2] [--batch_size 256]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

# Add the sql_retrieval directory to sys.path so skeleton_utils can be imported.
SQL_RETRIEVAL_DIR = os.path.dirname(os.path.abspath(__file__))  # sql_retrieval/
sys.path.insert(0, SQL_RETRIEVAL_DIR)

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = "./data"
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")

# Training set
TRAIN_PARQUET = os.path.join(DATA_ROOT, "train/train.parquet")
TRAIN_OUTPUT_DIR = os.path.join(DATA_ROOT, "train")

# Question bank skeletons
QB_SKELETONS_FILE = os.path.join(DATA_ROOT, "skeleton_index/skeletons.json")
QB_OUTPUT_DIR = os.path.join(DATA_ROOT, "skeleton_index")

# BIRD schema (used by sql2skeleton)
BIRD_TABLES_FILE = "./data/bird/train/train_tables.json"


# ─── Worker ──────────────────────────────────────────────────────────────────

def encode_worker(
    worker_id: int,
    gpu_id: int,
    skeleton_chunk: list,
    batch_size: int,
    model_path: str,
    gpu_mem_util: float,
    result_dict: dict,
    progress_counter,          # multiprocessing.Value
    progress_lock,             # multiprocessing.Lock
):
    """Each worker loads its own vLLM instance on one GPU and encodes a chunk."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM  # import inside worker to avoid CUDA init in main process

    # NOTE: max_num_seqs=1 is a workaround for vllm v1 engine bug
    model = LLM(
        model=model_path,
        task="embed",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
    )

    all_embs = []
    for start in range(0, len(skeleton_chunk), batch_size):
        end = min(start + batch_size, len(skeleton_chunk))
        batch = skeleton_chunk[start:end]
        outputs = model.embed(batch)
        batch_embs = [o.outputs.embedding for o in outputs]
        all_embs.extend(batch_embs)
        # Update global progress counter
        with progress_lock:
            progress_counter.value += len(batch)

    result_dict[worker_id] = all_embs


# ─── Progress monitor ───────────────────────────────────────────────────────

def progress_monitor(total: int, progress_counter, progress_lock, stop_event,
                     desc: str = "Encoding"):
    """Prints a tqdm-style progress bar from a shared counter, runs in a thread."""
    from tqdm import tqdm

    pbar = tqdm(total=total, desc=desc, unit="skel", dynamic_ncols=True)
    last = 0
    while not stop_event.is_set():
        with progress_lock:
            current = progress_counter.value
        delta = current - last
        if delta > 0:
            pbar.update(delta)
            last = current
        if current >= total:
            break
        stop_event.wait(timeout=0.5)
    # Final sync
    with progress_lock:
        current = progress_counter.value
    if current > last:
        pbar.update(current - last)
    pbar.close()


def parallel_encode(
    skeletons: list,
    num_gpus: int,
    procs_per_gpu: int,
    batch_size: int,
    gpu_mem_util: float,
    desc: str = "Encoding",
):
    """
    Encode a list of skeleton strings using multi-GPU data parallelism.
    Returns numpy array of normalized embeddings.
    """
    import faiss
    import numpy as np

    total = len(skeletons)
    num_workers = num_gpus * procs_per_gpu

    # For very small datasets, use fewer workers (no point having empty chunks)
    num_workers = min(num_workers, total)
    if num_workers == 0:
        return np.zeros((0, 1), dtype=np.float32)

    # Split data across workers
    chunks = []
    chunk_size = (total + num_workers - 1) // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunks.append(skeletons[start:end])

    # Map worker_id -> GPU id
    worker_gpu_map = []
    for gpu_id in range(num_gpus):
        for _ in range(procs_per_gpu):
            worker_gpu_map.append(gpu_id)
    # Trim to actual num_workers
    worker_gpu_map = worker_gpu_map[:num_workers]

    print(f"  Workers: {num_workers} ({num_gpus} GPUs × {procs_per_gpu} procs)")
    print(f"  Data split: {[len(c) for c in chunks]}")

    # Use 'spawn' to avoid fork+CUDA deadlock
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()

    # Start progress monitor thread
    import threading
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(total, progress_counter, progress_lock, stop_event, desc),
        daemon=True,
    )
    monitor_thread.start()

    # Start worker processes
    t0 = time.time()
    processes = []
    for worker_id in range(num_workers):
        if len(chunks[worker_id]) == 0:
            continue
        p = ctx.Process(
            target=encode_worker,
            args=(
                worker_id,
                worker_gpu_map[worker_id],
                chunks[worker_id],
                batch_size,
                MODEL_PATH,
                gpu_mem_util,
                result_dict,
                progress_counter,
                progress_lock,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    stop_event.set()
    monitor_thread.join(timeout=5)

    elapsed = time.time() - t0
    print(f"  Encoding done in {elapsed:.1f}s ({total / max(elapsed, 0.1):.0f} skel/s)")

    # Merge embeddings in order
    all_embeddings = []
    for worker_id in range(num_workers):
        if worker_id in result_dict:
            all_embeddings.extend(result_dict[worker_id])

    embeddings = np.array(all_embeddings, dtype=np.float32)
    if embeddings.shape[0] != total:
        print(f"  WARNING: expected {total} embeddings, got {embeddings.shape[0]}")

    faiss.normalize_L2(embeddings)
    return embeddings


def load_schema_map(tables_file: str) -> dict:
    """Load tables.json and build db_id -> schema mapping."""
    print(f"  Loading schemas from {tables_file} ...")
    with open(tables_file, "r", encoding="utf-8") as f:
        tables = json.load(f)
    schema_map = {t["db_id"]: t for t in tables}
    print(f"  Loaded {len(schema_map)} database schemas.")
    return schema_map


def process_train_set(num_gpus, procs_per_gpu, batch_size, gpu_mem_util):
    """Process the training set: extract skeletons -> encode query embeddings."""
    import numpy as np
    import pandas as pd
    from skeleton_utils import sql2skeleton

    print("\n" + "=" * 60)
    print("STEP 1: Process Training Set")
    print("=" * 60)

    # 1. Load train parquet
    print(f"\n[1.1] Loading train parquet from {TRAIN_PARQUET} ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"  Loaded {len(df)} rows.")

    # 2. Load BIRD schema
    print(f"\n[1.2] Loading BIRD schema ...")
    schema_map = load_schema_map(BIRD_TABLES_FILE)

    # 3. Extract skeletons
    print(f"\n[1.3] Extracting skeletons from SQL ...")
    t0 = time.time()
    train_skeletons = []
    bird_idx_to_parquet_row = {}
    failed_count = 0

    for parquet_row in range(len(df)):
        row = df.iloc[parquet_row]
        gt = row["reward_model"]["ground_truth"]
        db_id = gt["db_id"]
        sql = gt["sql"]
        bird_idx = row["extra_info"]["index"]

        # Map bird_idx -> parquet_row
        bird_idx_to_parquet_row[str(bird_idx)] = parquet_row

        # Convert SQL to skeleton
        skeleton = None
        error = None
        if db_id in schema_map:
            try:
                skeleton = sql2skeleton(sql, schema_map[db_id])
            except Exception as e:
                error = str(e)
                failed_count += 1
        else:
            error = f"db_id '{db_id}' not in BIRD schema"
            failed_count += 1

        # Fallback: use a simple placeholder if skeleton extraction failed
        if skeleton is None:
            skeleton = "select _"  # minimal fallback

        train_skeletons.append({
            "parquet_row": parquet_row,
            "bird_idx": bird_idx,
            "db_id": db_id,
            "sql": sql,
            "skeleton": skeleton,
            "error": error,
        })

    elapsed = time.time() - t0
    print(f"  Extracted {len(train_skeletons)} skeletons in {elapsed:.1f}s")
    print(f"  Failed: {failed_count} (using fallback skeleton)")

    # 4. Save train_skeletons.json
    skeletons_path = os.path.join(TRAIN_OUTPUT_DIR, "train_skeletons.json")
    print(f"\n[1.4] Saving train_skeletons.json to {skeletons_path} ...")
    with open(skeletons_path, "w", encoding="utf-8") as f:
        json.dump(train_skeletons, f, ensure_ascii=False, indent=2)

    # 5. Save bird_idx_to_parquet_row.json
    mapping_path = os.path.join(TRAIN_OUTPUT_DIR, "bird_idx_to_parquet_row.json")
    print(f"  Saving bird_idx_to_parquet_row.json to {mapping_path} ...")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(bird_idx_to_parquet_row, f, ensure_ascii=False)

    # 6. Encode query embeddings (multi-GPU parallel)
    skeleton_texts = [item["skeleton"] for item in train_skeletons]
    print(f"\n[1.5] Encoding {len(skeleton_texts)} train skeletons (Query mode) ...")
    embeddings = parallel_encode(
        skeleton_texts, num_gpus, procs_per_gpu, batch_size, gpu_mem_util,
        desc="Train encoding",
    )
    print(f"  Shape: {embeddings.shape}")

    # 7. Save embeddings
    emb_path = os.path.join(TRAIN_OUTPUT_DIR, "train_skeleton_embeddings.npy")
    print(f"  Saving to {emb_path} ...")
    np.save(emb_path, embeddings)
    print(f"  Saved! Size: {os.path.getsize(emb_path) / 1024 / 1024:.1f} MB")

    return len(train_skeletons)


def process_question_bank(num_gpus, procs_per_gpu, batch_size, gpu_mem_util):
    """Process the question bank: read existing skeletons -> encode query embeddings."""
    import numpy as np

    print("\n" + "=" * 60)
    print("STEP 2: Process Question Bank")
    print("=" * 60)

    # 1. Load skeletons.json
    print(f"\n[2.1] Loading skeletons from {QB_SKELETONS_FILE} ...")
    with open(QB_SKELETONS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records.")

    # 2. Sort by idx to ensure alignment with question_bank_filtered.parquet
    records.sort(key=lambda x: x["idx"])

    # Verify idx ordering (may not be contiguous if some records failed skeleton extraction)
    idxs = [r["idx"] for r in records]
    if idxs != list(range(len(records))):
        print(f"  Note: idx not contiguous (min={min(idxs)}, max={max(idxs)}, count={len(idxs)}). "
              f"Proceeding with sorted order.")

    # 3. Extract skeleton texts
    skeleton_texts = [r["skeleton"] for r in records]
    print(f"  Total skeletons to encode: {len(skeleton_texts)}")

    # 4. Encode query embeddings (multi-GPU parallel)
    print(f"\n[2.2] Encoding {len(skeleton_texts)} QB skeletons (Query mode) ...")
    embeddings = parallel_encode(
        skeleton_texts, num_gpus, procs_per_gpu, batch_size, gpu_mem_util,
        desc="QB encoding",
    )
    print(f"  Shape: {embeddings.shape}")

    # 5. Save embeddings
    emb_path = os.path.join(QB_OUTPUT_DIR, "qb_skeleton_query_embeddings.npy")
    print(f"  Saving to {emb_path} ...")
    np.save(emb_path, embeddings)
    print(f"  Saved! Size: {os.path.getsize(emb_path) / 1024 / 1024:.1f} MB")

    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute skeleton embeddings for the training set and question bank (multi-GPU)"
    )
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--procs_per_gpu", type=int, default=2,
                        help="Number of vLLM processes per GPU (default: 2)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Embedding batch size per worker")
    parser.add_argument("--gpu_mem_util", type=float, default=0.95,
                        help="GPU memory utilization per vLLM instance "
                             "(default: 0.95 for 2 procs/GPU)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training-set processing")
    parser.add_argument("--skip_qb", action="store_true",
                        help="Skip question-bank processing")
    args = parser.parse_args()

    print(f"Config: {args.num_gpus} GPUs × {args.procs_per_gpu} procs/GPU")
    print(f"  batch_size={args.batch_size}, gpu_mem_util={args.gpu_mem_util}")

    # Ensure output dirs exist
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(QB_OUTPUT_DIR, exist_ok=True)

    # Process train set
    train_count = 0
    if not args.skip_train:
        train_count = process_train_set(
            args.num_gpus, args.procs_per_gpu, args.batch_size, args.gpu_mem_util
        )
    else:
        print("\n[SKIP] Training set processing skipped.")

    # Process question bank
    qb_count = 0
    if not args.skip_qb:
        qb_count = process_question_bank(
            args.num_gpus, args.procs_per_gpu, args.batch_size, args.gpu_mem_util
        )
    else:
        print("\n[SKIP] Question bank processing skipped.")

    # Summary
    print("\n" + "=" * 60)
    print("PRECOMPUTE SUMMARY")
    print("=" * 60)
    if train_count > 0:
        print(f"  Train: {train_count} skeletons → train_skeleton_embeddings.npy")
        print(f"         + train_skeletons.json + bird_idx_to_parquet_row.json")
    if qb_count > 0:
        print(f"  QB:    {qb_count} skeletons → qb_skeleton_query_embeddings.npy")
    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()
