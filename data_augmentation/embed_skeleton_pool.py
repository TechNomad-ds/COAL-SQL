#!/usr/bin/env python3
"""Encode unique skeletons with Qwen3-Embedding and save aligned embeddings."""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time


DEFAULT_INPUT = "./data/skeleton_augmentation/skeleton_pool_unique.json"
DEFAULT_OUTPUT = "./data/skeleton_augmentation/skeleton_pool_embeddings.npy"
DEFAULT_MODEL = "./models/Qwen3-Embedding-0.6B"


def to_embedding_skeleton(skeleton, placeholder_style):
    if placeholder_style == "underscore":
        return (
            skeleton.replace("<TABLE>", "_")
            .replace("<COLUMN>", "_")
            .replace("<LITERAL>", "_")
        )
    if placeholder_style == "typed":
        return skeleton
    raise ValueError(f"unknown placeholder_style: {placeholder_style}")


def encode_worker(
    worker_id,
    gpu_id,
    skeleton_chunk,
    batch_size,
    model_path,
    gpu_mem_util,
    dtype,
    max_model_len,
    result_dict,
    progress_counter,
    progress_lock,
    shard_dir,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    import numpy as np
    from vllm import LLM

    model = LLM(
        model=model_path,
        task="embed",
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
    )

    all_embeddings = []
    for start in range(0, len(skeleton_chunk), batch_size):
        batch = skeleton_chunk[start:start + batch_size]
        outputs = model.embed(batch, use_tqdm=False)
        all_embeddings.extend([o.outputs.embedding for o in outputs])
        with progress_lock:
            progress_counter.value += len(batch)

    shard = np.asarray(all_embeddings, dtype=np.float32)
    shard_path = os.path.join(shard_dir, f"worker_{worker_id:03d}.npy")
    np.save(shard_path, shard)
    result_dict[worker_id] = {
        "path": shard_path,
        "count": int(shard.shape[0]),
    }


def progress_monitor(total, progress_counter, progress_lock, stop_event):
    from tqdm import tqdm

    pbar = tqdm(total=total, desc="Encoding skeletons", unit="skel", dynamic_ncols=True)
    last = 0
    while not stop_event.is_set():
        with progress_lock:
            current = progress_counter.value
        if current > last:
            pbar.update(current - last)
            last = current
        if current >= total:
            break
        stop_event.wait(0.5)
    with progress_lock:
        current = progress_counter.value
    if current > last:
        pbar.update(current - last)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--procs_per_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument(
        "--placeholder_style",
        choices=["underscore", "typed"],
        default="underscore",
        help="View used only for embedding. 'underscore' converts old <TABLE>/<COLUMN>/<LITERAL> records to '_'.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Embed Skeleton Pool")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"model_path={args.model_path}")
    print(f"num_gpus={args.num_gpus} procs_per_gpu={args.procs_per_gpu} batch_size={args.batch_size} max_model_len={args.max_model_len}")
    print(f"placeholder_style={args.placeholder_style}")
    print("=" * 80)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    records.sort(key=lambda x: x["pool_id"])
    skeletons = [to_embedding_skeleton(r["skeleton"], args.placeholder_style) for r in records]
    total = len(skeletons)
    print(f"Loaded {total:,} unique skeletons")

    num_workers = min(total, args.num_gpus * args.procs_per_gpu)
    chunk_size = (total + num_workers - 1) // num_workers
    chunks = [
        skeletons[i * chunk_size:min((i + 1) * chunk_size, total)]
        for i in range(num_workers)
    ]
    worker_gpu_map = []
    for gpu_id in range(args.num_gpus):
        for _ in range(args.procs_per_gpu):
            worker_gpu_map.append(gpu_id)
    worker_gpu_map = worker_gpu_map[:num_workers]

    print(f"workers={num_workers} split={[len(c) for c in chunks]}")
    print(f"worker_gpu_map={worker_gpu_map}")

    shard_dir = f"{args.output}.shards"
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()

    import threading
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=progress_monitor,
        args=(total, progress_counter, progress_lock, stop_event),
        daemon=True,
    )
    monitor.start()

    t0 = time.time()
    processes = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=encode_worker,
            args=(
                worker_id,
                worker_gpu_map[worker_id],
                chunk,
                args.batch_size,
                args.model_path,
                args.gpu_mem_util,
                args.dtype,
                args.max_model_len,
                result_dict,
                progress_counter,
                progress_lock,
                shard_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker failed with exitcode={p.exitcode}")

    stop_event.set()
    monitor.join(timeout=5)

    import faiss
    import numpy as np

    shard_arrays = []
    for worker_id in range(num_workers):
        if worker_id not in result_dict:
            raise RuntimeError(f"missing shard from worker_id={worker_id}")
        shard_info = result_dict[worker_id]
        shard = np.load(shard_info["path"])
        expected = len(chunks[worker_id])
        if shard.shape[0] != expected:
            raise RuntimeError(f"worker {worker_id} shard has {shard.shape[0]} rows, expected {expected}")
        shard_arrays.append(shard)
    embeddings = np.vstack(shard_arrays).astype(np.float32, copy=False)
    if embeddings.shape[0] != total:
        raise RuntimeError(f"expected {total} embeddings, got {embeddings.shape[0]}")
    faiss.normalize_L2(embeddings)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"Saved {args.output}")
    print(f"shape={embeddings.shape} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
