#!/usr/bin/env python3
"""Select augmentation skeletons with K-Center Greedy."""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import faiss
import numpy as np

# Import skeleton_utils from the sibling sql_retrieval/ package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_RETRIEVAL_DIR = os.path.join(PROJECT_ROOT, "sql_retrieval")
sys.path.insert(0, SQL_RETRIEVAL_DIR)

from skeleton_utils import SkeletonParseError, sql2skeleton  # noqa: E402


DEFAULT_BIRD = "./data/train/bird_train_3000.json"
DEFAULT_POOL = "./data/skeleton_augmentation/skeleton_pool_unique.json"
DEFAULT_POOL_EMB = "./data/skeleton_augmentation/skeleton_pool_embeddings.npy"
DEFAULT_OUTLIER_SCORES = "./data/skeleton_augmentation/outlier_scores.npy"
DEFAULT_OUTPUT_DIR = "./data/skeleton_augmentation"
DEFAULT_MODEL = "./models/Qwen3-Embedding-0.6B"


def to_underscore_skeleton(skeleton):
    return (
        skeleton.replace("<TABLE>", "_")
        .replace("<COLUMN>", "_")
        .replace("<LITERAL>", "_")
    )


def _extract_sql(record):
    for key in ("SQL", "sql", "query", "gold_sql"):
        value = record.get(key)
        if value:
            return value, key
    return "", None


def _process_bird_record(item):
    idx, record = item
    sql, sql_key = _extract_sql(record)
    try:
        skeleton = sql2skeleton(sql, None)
        error = None
    except SkeletonParseError as exc:
        skeleton = None
        error = str(exc)

    return {
        "source_idx": idx,
        "db_id": record.get("db_id"),
        "question": record.get("question", ""),
        "sql": sql,
        "sql_key": sql_key,
        "skeleton": skeleton,
        "error": error,
    }


def extract_bird_skeletons(bird_path, output_path, num_workers):
    with open(bird_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunksize = max(1, len(records) // (num_workers * 4))
        processed = list(executor.map(_process_bird_record, enumerate(records), chunksize=chunksize))

    successes = [r for r in processed if r["error"] is None and r["skeleton"]]
    errors = [r for r in processed if r["error"] is not None or not r["skeleton"]]

    skeleton_to_entry = {}
    for record in successes:
        skeleton = record["skeleton"]
        if skeleton not in skeleton_to_entry:
            skeleton_to_entry[skeleton] = {
                "bird_skeleton_id": len(skeleton_to_entry),
                "skeleton": skeleton,
                "count": 0,
                "first_source_idx": record["source_idx"],
                "first_db_id": record["db_id"],
                "first_question": record.get("question", ""),
                "first_sql": record.get("sql", ""),
                "source_indices": [],
            }
        entry = skeleton_to_entry[skeleton]
        entry["count"] += 1
        if len(entry["source_indices"]) < 20:
            entry["source_indices"].append(record["source_idx"])

    unique_records = list(skeleton_to_entry.values())
    payload = {
        "bird_path": bird_path,
        "total_records": len(records),
        "success_count": len(successes),
        "error_count": len(errors),
        "unique_skeleton_count": len(unique_records),
        "records": unique_records,
        "errors": errors,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _encode_worker(
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
    result_dict[worker_id] = {"path": shard_path, "count": int(shard.shape[0])}


def embed_skeletons(
    skeletons,
    output_path,
    model_path,
    num_gpus,
    procs_per_gpu,
    batch_size,
    gpu_mem_util,
    dtype,
    max_model_len,
):
    if os.path.exists(output_path):
        embeddings = np.load(output_path)
        if embeddings.shape[0] == len(skeletons):
            return embeddings.astype(np.float32, copy=False)
        raise RuntimeError(f"existing embedding row mismatch: {output_path} has {embeddings.shape[0]}, expected {len(skeletons)}")

    if not skeletons:
        raise RuntimeError("no skeletons to embed")

    total = len(skeletons)
    num_workers = min(total, num_gpus * procs_per_gpu)
    chunk_size = (total + num_workers - 1) // num_workers
    chunks = [
        skeletons[i * chunk_size:min((i + 1) * chunk_size, total)]
        for i in range(num_workers)
    ]
    worker_gpu_map = []
    for gpu_id in range(num_gpus):
        for _ in range(procs_per_gpu):
            worker_gpu_map.append(gpu_id)
    worker_gpu_map = worker_gpu_map[:num_workers]

    shard_dir = f"{output_path}.shards"
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()

    t0 = time.time()
    processes = []
    for worker_id, chunk in enumerate(chunks):
        process = ctx.Process(
            target=_encode_worker,
            args=(
                worker_id,
                worker_gpu_map[worker_id],
                chunk,
                batch_size,
                model_path,
                gpu_mem_util,
                dtype,
                max_model_len,
                result_dict,
                progress_counter,
                progress_lock,
                shard_dir,
            ),
        )
        process.start()
        processes.append(process)

    while any(process.is_alive() for process in processes):
        with progress_lock:
            done = progress_counter.value
        print(f"  embedding progress={done:,}/{total:,}", flush=True)
        time.sleep(10)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"embedding worker failed with exitcode={process.exitcode}")

    shard_arrays = []
    for worker_id in range(num_workers):
        if worker_id not in result_dict:
            raise RuntimeError(f"missing shard from worker_id={worker_id}")
        shard = np.load(result_dict[worker_id]["path"])
        expected = len(chunks[worker_id])
        if shard.shape[0] != expected:
            raise RuntimeError(f"worker {worker_id} shard has {shard.shape[0]} rows, expected {expected}")
        shard_arrays.append(shard)

    embeddings = np.vstack(shard_arrays).astype(np.float32, copy=False)
    faiss.normalize_L2(embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"  saved embeddings {output_path} shape={embeddings.shape} elapsed={time.time() - t0:.1f}s")
    return embeddings


def load_pool(pool_path, pool_emb_path, outlier_scores_path):
    with open(pool_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    records.sort(key=lambda x: x["pool_id"])
    embeddings = np.load(pool_emb_path).astype(np.float32, copy=False)
    outlier_scores = np.load(outlier_scores_path).astype(np.float32, copy=False)

    if len(records) != embeddings.shape[0]:
        raise RuntimeError(f"pool/embedding mismatch: {len(records)} vs {embeddings.shape[0]}")
    if outlier_scores.shape[0] != len(records):
        raise RuntimeError(f"pool/outlier score mismatch: {len(records)} vs {outlier_scores.shape[0]}")
    if outlier_scores.ndim == 2:
        outlier_scores = outlier_scores[:, 0]
    return records, embeddings, outlier_scores


def initialize_min_dist(candidate_embeddings, existing_embeddings, batch_size):
    dim = existing_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(existing_embeddings)

    max_sim = np.empty(candidate_embeddings.shape[0], dtype=np.float32)
    for start in range(0, candidate_embeddings.shape[0], batch_size):
        batch = candidate_embeddings[start:start + batch_size]
        scores, _ = index.search(batch, 1)
        max_sim[start:start + batch.shape[0]] = scores[:, 0]
    return 1.0 - max_sim


def kcenter_select_cpu(candidate_embeddings, initial_min_dist, k, update_batch_size):
    min_dist = initial_min_dist.astype(np.float32, copy=True)
    selected = []
    selected_distances = []

    for rank in range(k):
        selected_idx = int(np.argmax(min_dist))
        selected_distance = float(min_dist[selected_idx])
        if not np.isfinite(selected_distance) or selected_distance < 0:
            break

        selected.append(selected_idx)
        selected_distances.append(selected_distance)
        selected_vec = candidate_embeddings[selected_idx]

        for start in range(0, candidate_embeddings.shape[0], update_batch_size):
            batch = candidate_embeddings[start:start + update_batch_size]
            sims = batch @ selected_vec
            distances = 1.0 - sims
            min_dist[start:start + batch.shape[0]] = np.minimum(
                min_dist[start:start + batch.shape[0]],
                distances.astype(np.float32, copy=False),
            )
        min_dist[selected_idx] = -np.inf

        if (rank + 1) % 100 == 0 or rank == 0:
            print(f"  selected={rank + 1:,}/{k:,} latest_distance={selected_distance:.6f}", flush=True)

    return selected, selected_distances


def kcenter_select_torch(candidate_embeddings, initial_min_dist, k, device):
    import torch

    candidate_tensor = torch.from_numpy(candidate_embeddings).to(device=device, dtype=torch.float32)
    min_dist = torch.from_numpy(initial_min_dist.astype(np.float32, copy=False)).to(device=device)
    selected = []
    selected_distances = []

    for rank in range(k):
        selected_idx = int(torch.argmax(min_dist).item())
        selected_distance = float(min_dist[selected_idx].item())
        if not np.isfinite(selected_distance) or selected_distance < 0:
            break

        selected.append(selected_idx)
        selected_distances.append(selected_distance)
        selected_vec = candidate_tensor[selected_idx]
        distances = 1.0 - torch.mv(candidate_tensor, selected_vec)
        min_dist = torch.minimum(min_dist, distances)
        min_dist[selected_idx] = -float("inf")

        if (rank + 1) % 100 == 0 or rank == 0:
            print(f"  selected={rank + 1:,}/{k:,} latest_distance={selected_distance:.6f}", flush=True)

    return selected, selected_distances


def kcenter_select(candidate_embeddings, initial_min_dist, k, update_batch_size, device):
    if device != "cpu":
        try:
            return kcenter_select_torch(candidate_embeddings, initial_min_dist, k, device)
        except Exception as exc:
            print(f"  CUDA K-Center failed ({exc}); falling back to CPU", flush=True)
    return kcenter_select_cpu(candidate_embeddings, initial_min_dist, k, update_batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_path", default=DEFAULT_BIRD)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--pool_embeddings", default=DEFAULT_POOL_EMB)
    parser.add_argument("--outlier_scores", default=DEFAULT_OUTLIER_SCORES)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--target_k", type=int, default=3500)
    parser.add_argument("--outlier_threshold", type=float, default=0.3)
    parser.add_argument("--extract_workers", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--procs_per_gpu", type=int, default=1)
    parser.add_argument("--embed_batch_size", type=int, default=256)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--init_batch_size", type=int, default=8192)
    parser.add_argument("--update_batch_size", type=int, default=8192)
    parser.add_argument("--kcenter_device", default="cuda:0", help="Use cuda:N for fast selection, or cpu.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bird_skeleton_path = os.path.join(args.output_dir, "bird_train_3000_skeletons.json")
    bird_embedding_path = os.path.join(args.output_dir, "bird_train_3000_embeddings.npy")
    filtered_path = os.path.join(args.output_dir, "skeleton_pool_filtered.json")
    selected_path = os.path.join(args.output_dir, f"selected_{args.target_k}_skeletons.json")
    stats_path = os.path.join(args.output_dir, f"selected_{args.target_k}_stats.json")

    print("=" * 80)
    print("Select Skeletons by K-Center")
    print(f"bird_path={args.bird_path}")
    print(f"pool={args.pool}")
    print(f"target_k={args.target_k} outlier_threshold={args.outlier_threshold}")
    print("=" * 80)

    print("\n[1/5] Extracting BIRD skeletons...")
    bird_payload = extract_bird_skeletons(args.bird_path, bird_skeleton_path, args.extract_workers)
    bird_records = bird_payload["records"]
    print(
        f"  total={bird_payload['total_records']:,} success={bird_payload['success_count']:,} "
        f"errors={bird_payload['error_count']:,} unique={bird_payload['unique_skeleton_count']:,}"
    )
    print(f"  saved {bird_skeleton_path}")

    print("\n[2/5] Embedding BIRD skeletons...")
    bird_skeletons = [record["skeleton"] for record in bird_records]
    bird_embeddings = embed_skeletons(
        bird_skeletons,
        bird_embedding_path,
        args.model_path,
        args.num_gpus,
        args.procs_per_gpu,
        args.embed_batch_size,
        args.gpu_mem_util,
        args.dtype,
        args.max_model_len,
    )
    print(f"  bird_embeddings shape={bird_embeddings.shape}")

    print("\n[3/5] Loading and filtering candidate pool...")
    pool_records, pool_embeddings, outlier_scores = load_pool(args.pool, args.pool_embeddings, args.outlier_scores)
    keep_mask = outlier_scores <= args.outlier_threshold
    candidate_indices = np.where(keep_mask)[0]
    if candidate_indices.shape[0] < args.target_k:
        raise RuntimeError(f"only {candidate_indices.shape[0]} candidates after filtering, cannot select {args.target_k}")

    candidate_records = [pool_records[int(i)] for i in candidate_indices]
    candidate_embeddings = np.ascontiguousarray(pool_embeddings[candidate_indices])
    filtered_records = []
    for record, pool_idx in zip(candidate_records, candidate_indices):
        item = dict(record)
        item["typed_skeleton"] = record["skeleton"]
        item["underscore_skeleton"] = to_underscore_skeleton(record["skeleton"])
        item["outlier_score"] = float(outlier_scores[int(pool_idx)])
        filtered_records.append(item)
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "outlier_threshold": args.outlier_threshold,
                "total_pool_count": len(pool_records),
                "filtered_out_count": int((~keep_mask).sum()),
                "candidate_count": int(candidate_indices.shape[0]),
                "records": filtered_records,
            },
            f,
            ensure_ascii=False,
        )
    print(
        f"  pool={len(pool_records):,} filtered_out={(~keep_mask).sum():,} "
        f"candidates={candidate_indices.shape[0]:,}"
    )
    print(f"  saved {filtered_path}")

    print("\n[4/5] Initializing distances to BIRD set...")
    initial_min_dist = initialize_min_dist(candidate_embeddings, bird_embeddings, args.init_batch_size)
    print(
        "  initial_min_dist "
        f"min={initial_min_dist.min():.6f} p50={np.percentile(initial_min_dist, 50):.6f} "
        f"p95={np.percentile(initial_min_dist, 95):.6f} max={initial_min_dist.max():.6f}"
    )

    print("\n[5/5] Running K-Center Greedy...")
    selected_local_indices, selected_distances = kcenter_select(
        candidate_embeddings,
        initial_min_dist,
        args.target_k,
        args.update_batch_size,
        args.kcenter_device,
    )

    selected_records = []
    for rank, (local_idx, distance) in enumerate(zip(selected_local_indices, selected_distances), start=1):
        pool_idx = int(candidate_indices[local_idx])
        record = pool_records[pool_idx]
        typed_skeleton = record["skeleton"]
        underscore_skeleton = to_underscore_skeleton(typed_skeleton)
        selected_records.append(
            {
                "selection_rank": rank,
                "pool_id": record["pool_id"],
                "skeleton": typed_skeleton,
                "typed_skeleton": typed_skeleton,
                "underscore_skeleton": underscore_skeleton,
                "selection_distance": float(distance),
                "nearest_similarity_to_existing_or_selected": float(1.0 - distance),
                "outlier_score": float(outlier_scores[pool_idx]),
                "count": record.get("count", 0),
                "first_source_idx": record.get("first_source_idx"),
                "first_db_id": record.get("first_db_id"),
                "first_sql": record.get("first_sql", ""),
                "first_question": record.get("first_question", ""),
                "source_indices": record.get("source_indices", []),
            }
        )

    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected_records, f, ensure_ascii=False, indent=2)

    stats = {
        "method": "K-Center Greedy / Furthest First over normalized Qwen3 skeleton embeddings; distance = 1 - cosine_similarity",
        "bird_path": args.bird_path,
        "pool_path": args.pool,
        "pool_embeddings_path": args.pool_embeddings,
        "outlier_scores_path": args.outlier_scores,
        "model_path": args.model_path,
        "target_k": args.target_k,
        "selected_count": len(selected_records),
        "outlier_threshold": args.outlier_threshold,
        "total_pool_count": len(pool_records),
        "filtered_out_count": int((~keep_mask).sum()),
        "candidate_count": int(candidate_indices.shape[0]),
        "bird_total_records": bird_payload["total_records"],
        "bird_success_count": bird_payload["success_count"],
        "bird_error_count": bird_payload["error_count"],
        "bird_unique_skeleton_count": bird_payload["unique_skeleton_count"],
        "initial_min_distance": {
            "min": float(np.min(initial_min_dist)),
            "p50": float(np.percentile(initial_min_dist, 50)),
            "p95": float(np.percentile(initial_min_dist, 95)),
            "max": float(np.max(initial_min_dist)),
        },
        "selected_distance": {
            "min": float(np.min(selected_distances)) if selected_distances else None,
            "p50": float(np.percentile(selected_distances, 50)) if selected_distances else None,
            "max": float(np.max(selected_distances)) if selected_distances else None,
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  saved {selected_path}")
    print(f"  saved {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
