#!/usr/bin/env python3
"""Analyze skeleton embedding outliers with approximate kNN density."""

import argparse
import json
import os

import faiss
import numpy as np


DEFAULT_POOL = "./data/skeleton_augmentation/skeleton_pool_unique.json"
DEFAULT_EMB = "./data/skeleton_augmentation/skeleton_pool_embeddings.npy"
DEFAULT_OUTPUT_DIR = "./data/skeleton_augmentation"


def percentile_dict(values):
    percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9, 100]
    return {f"p{p:g}": float(np.percentile(values, p)) for p in percentiles}


def make_hnsw_index(embeddings, m, ef_construction, ef_search):
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    return index


def search_knn(index, embeddings, k, batch_size):
    all_scores = []
    all_indices = []
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start:start + batch_size]
        scores, indices = index.search(batch, k + 1)
        all_scores.append(scores)
        all_indices.append(indices)
    return np.vstack(all_scores), np.vstack(all_indices)


def compact_record(record, score, nearest_sim, mean_sim):
    return {
        "pool_id": record["pool_id"],
        "outlier_score": float(score),
        "nearest_similarity": float(nearest_sim),
        "mean_knn_similarity": float(mean_sim),
        "count": record.get("count", 0),
        "skeleton": record["skeleton"],
        "first_source_idx": record.get("first_source_idx"),
        "first_db_id": record.get("first_db_id"),
        "first_sql": record.get("first_sql", "")[:1000],
        "first_question": record.get("first_question", "")[:500],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--embeddings", default=DEFAULT_EMB)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=80)
    parser.add_argument("--ef_search", type=int, default=128)
    parser.add_argument("--top_n", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, "outlier_stats.json")
    samples_path = os.path.join(args.output_dir, "outlier_samples.json")
    scores_path = os.path.join(args.output_dir, "outlier_scores.npy")
    score_records_path = os.path.join(args.output_dir, "outlier_scores.jsonl")

    print("=" * 80)
    print("Analyze Skeleton Outliers")
    print(f"pool={args.pool}")
    print(f"embeddings={args.embeddings}")
    print(f"k={args.k}")
    print("=" * 80)

    with open(args.pool, "r", encoding="utf-8") as f:
        records = json.load(f)
    records.sort(key=lambda x: x["pool_id"])
    embeddings = np.load(args.embeddings)
    if len(records) != embeddings.shape[0]:
        raise RuntimeError(f"metadata/embedding mismatch: {len(records)} vs {embeddings.shape[0]}")

    print(f"Loaded n={embeddings.shape[0]:,}, dim={embeddings.shape[1]}")
    index = make_hnsw_index(embeddings, args.hnsw_m, args.ef_construction, args.ef_search)
    print("HNSW index built")
    scores, indices = search_knn(index, embeddings, args.k, args.batch_size)
    print("kNN search done")

    neighbor_scores = []
    for row_idx, (score_row, index_row) in enumerate(zip(scores, indices)):
        pairs = [
            (float(score), int(idx))
            for score, idx in zip(score_row, index_row)
            if idx != -1 and idx != row_idx
        ][:args.k]
        if not pairs:
            neighbor_scores.append([0.0])
        else:
            neighbor_scores.append([score for score, _ in pairs])

    nearest_sim = np.asarray([max(row) for row in neighbor_scores], dtype=np.float32)
    mean_sim = np.asarray([sum(row) / len(row) for row in neighbor_scores], dtype=np.float32)
    nearest_distance = 1.0 - nearest_sim
    mean_knn_distance = 1.0 - mean_sim
    score_matrix = np.column_stack([mean_knn_distance, nearest_sim, mean_sim]).astype(np.float32)
    np.save(scores_path, score_matrix)

    order = np.argsort(-mean_knn_distance)
    top_outliers = [
        compact_record(records[i], mean_knn_distance[i], nearest_sim[i], mean_sim[i])
        for i in order[:args.top_n]
    ]

    bucket_samples = {}
    for pct in [90, 95, 97, 99, 99.5, 99.9]:
        threshold = float(np.percentile(mean_knn_distance, pct))
        candidate_indices = np.where(mean_knn_distance >= threshold)[0]
        candidate_indices = candidate_indices[np.argsort(-mean_knn_distance[candidate_indices])]
        bucket_samples[f"gte_p{pct:g}"] = [
            compact_record(records[i], mean_knn_distance[i], nearest_sim[i], mean_sim[i])
            for i in candidate_indices[:50]
        ]

    stats = {
        "pool_path": args.pool,
        "embeddings_path": args.embeddings,
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "k": args.k,
        "method": "HNSW approximate kNN over normalized embeddings; outlier_score = 1 - mean(top-k cosine similarity excluding self)",
        "mean_knn_distance": percentile_dict(mean_knn_distance),
        "nearest_distance": percentile_dict(nearest_distance),
        "mean_knn_similarity": percentile_dict(mean_sim),
        "nearest_similarity": percentile_dict(nearest_sim),
        "hnsw": {
            "m": args.hnsw_m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top_outliers": top_outliers,
                "bucket_samples": bucket_samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(score_records_path, "w", encoding="utf-8") as f:
        for record, score, near_sim, avg_sim in zip(records, mean_knn_distance, nearest_sim, mean_sim):
            f.write(
                json.dumps(
                    {
                        "pool_id": record["pool_id"],
                        "outlier_score": float(score),
                        "nearest_similarity": float(near_sim),
                        "mean_knn_similarity": float(avg_sim),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved {stats_path}")
    print(f"Saved {samples_path}")
    print(f"Saved {scores_path}")
    print(f"Saved {score_records_path}")
    print(json.dumps(stats["mean_knn_distance"], indent=2))


if __name__ == "__main__":
    main()
