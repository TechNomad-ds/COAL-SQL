#!/usr/bin/env python3
"""
Build FAISS index from skeleton embeddings.

1. Load skeletons.json (output of build_skeletons.py)
2. Encode unique skeletons with Qwen3-Embedding-0.6B via vLLM
   - 8 GPU × 2 processes per GPU = 16 parallel workers
   - Each worker holds an independent vLLM instance (max_num_seqs=1)
3. Build a FAISS index over the embeddings
4. Save index + metadata (skeleton -> list of original SQLs)

Usage:
    conda run -n coalsql python build_index.py [--num_gpus 8] [--procs_per_gpu 2]
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

# ─── Paths ───────────────────────────────────────────────────────────────────
SKELETONS_FILE = "./data/skeleton_index/skeletons.json"
OUTPUT_DIR = "./data/skeleton_index"
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")

INDEX_FILE = os.path.join(OUTPUT_DIR, "skeleton.index")
METADATA_FILE = os.path.join(OUTPUT_DIR, "skeleton_metadata.json")


# ─── Worker ──────────────────────────────────────────────────────────────────

def encode_worker(
    worker_id: int,
    gpu_id: int,
    skeleton_chunk: list[str],
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
    # (KeyError: None in num_scheduled_tokens) triggered when chunked
    # prefill + concurrent requests interact during embed tasks.
    # See: https://github.com/vllm-project/vllm/issues/23223
    #      https://github.com/vllm-project/vllm/issues/25991
    model = LLM(
        model=model_path,
        task="embed",
        dtype="bfloat16",
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

def progress_monitor(total: int, progress_counter, progress_lock, stop_event):
    """Prints a tqdm-style progress bar from a shared counter, runs in a thread."""
    from tqdm import tqdm

    pbar = tqdm(total=total, desc="Encoding", unit="skel", dynamic_ncols=True)
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


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index with multi-GPU data parallelism"
    )
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--procs_per_gpu", type=int, default=2,
                        help="Number of vLLM processes per GPU (default: 2)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size per worker for embedding")
    parser.add_argument("--gpu_mem_util", type=float, default=0.95,
                        help="GPU memory utilization per vLLM instance (default: 0.95, "
                             "set lower when running multiple procs per GPU)")
    args = parser.parse_args()

    num_workers = args.num_gpus * args.procs_per_gpu
    print(f"Config: {args.num_gpus} GPUs × {args.procs_per_gpu} procs/GPU = {num_workers} workers")
    print(f"  batch_size={args.batch_size}, gpu_mem_util={args.gpu_mem_util}")

    # ─── Load skeletons ──────────────────────────────────────────────────────
    print(f"\nLoading skeletons from {SKELETONS_FILE} ...")
    with open(SKELETONS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records.")

    # ─── Group by skeleton ───────────────────────────────────────────────────
    skeleton_to_entries = defaultdict(list)
    for rec in records:
        skeleton_to_entries[rec["skeleton"]].append({
            "idx": rec["idx"],
            "db_id": rec["db_id"],
            "sql": rec["sql"],
            "question": rec.get("question", ""),
        })

    unique_skeletons = list(skeleton_to_entries.keys())
    unique_skeletons = [s for s in unique_skeletons if s and s.strip()]
    total = len(unique_skeletons)
    print(f"  Unique skeletons (non-empty): {total}")

    # ─── Split data across workers ───────────────────────────────────────────
    chunks = []
    chunk_size = (total + num_workers - 1) // num_workers  # ceil division
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunks.append(unique_skeletons[start:end])

    worker_gpu_map = []
    for gpu_id in range(args.num_gpus):
        for _ in range(args.procs_per_gpu):
            worker_gpu_map.append(gpu_id)

    print(f"\n  Data split: {[len(c) for c in chunks]}")
    print(f"  Worker→GPU: {worker_gpu_map}")

    # ─── Launch workers ──────────────────────────────────────────────────────
    print(f"\nLaunching {num_workers} encoding workers ...")
    t0 = time.time()

    # Use 'spawn' to avoid fork+CUDA deadlock (torch import lock issue)
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
        args=(total, progress_counter, progress_lock, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    # Start worker processes
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
                args.batch_size,
                MODEL_PATH,
                args.gpu_mem_util,
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
    print(f"\n  All workers done in {elapsed:.1f}s")
    print(f"  Throughput: {total / elapsed:.0f} skeletons/s")

    # ─── Merge embeddings in order ───────────────────────────────────────────
    # Import faiss/numpy AFTER workers finish to avoid CUDA init in main process
    # before fork (spawn workers don't inherit main's imports anyway, but this
    # keeps the main process clean).
    import faiss
    import numpy as np

    print("Merging embeddings ...")
    all_embeddings = []
    for worker_id in range(num_workers):
        if worker_id in result_dict:
            all_embeddings.extend(result_dict[worker_id])

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    if embeddings.shape[0] != total:
        print(f"  WARNING: expected {total} embeddings, got {embeddings.shape[0]}")

    # ─── Normalize for cosine similarity ─────────────────────────────────────
    faiss.normalize_L2(embeddings)

    # ─── Build FAISS index ───────────────────────────────────────────────────
    dim = embeddings.shape[1]
    print(f"Building FAISS index (dim={dim}, n={embeddings.shape[0]}) ...")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index built, total vectors: {index.ntotal}")

    # ─── Save index ──────────────────────────────────────────────────────────
    print(f"Saving FAISS index to {INDEX_FILE} ...")
    faiss.write_index(index, INDEX_FILE)

    # ─── Build and save metadata ─────────────────────────────────────────────
    metadata = []
    for skel in unique_skeletons:
        entries = skeleton_to_entries[skel]
        metadata.append({
            "skeleton": skel,
            "count": len(entries),
            "entries": entries,
        })

    print(f"Saving metadata to {METADATA_FILE} ...")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDEX BUILD SUMMARY")
    print("=" * 60)
    print(f"Total records:        {len(records)}")
    print(f"Unique skeletons:     {len(unique_skeletons)}")
    print(f"Embedding dimension:  {dim}")
    print(f"Workers:              {num_workers} ({args.num_gpus} GPUs × {args.procs_per_gpu} procs)")
    print(f"Encoding time:        {elapsed:.1f}s ({total / elapsed:.0f} skel/s)")
    print(f"Index file:           {INDEX_FILE}")
    print(f"Metadata file:        {METADATA_FILE}")
    print(f"Index file size:      {os.path.getsize(INDEX_FILE) / 1024 / 1024:.1f} MB")
    print(f"Metadata file size:   {os.path.getsize(METADATA_FILE) / 1024 / 1024:.1f} MB")
    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()
