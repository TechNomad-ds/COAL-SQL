#!/usr/bin/env python3
"""Build a SQL skeleton pool from a user-provided SQL collection.

The input is a JSON array of records, each providing at least a SQL query.
Recognized fields per record:
  - one of: "sql" / "SQL" / "query" / "gold_sql"  (required)
  - "db_id"    (optional)
  - "question" (optional)

The script extracts a structural skeleton for each SQL, deduplicates skeletons,
and writes the pool metadata used by the downstream augmentation steps.
Optionally, a random subset can be sampled with --sample_size.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import ijson

# Import skeleton_utils from the sibling sql_retrieval/ package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_RETRIEVAL_DIR = os.path.join(PROJECT_ROOT, "sql_retrieval")
sys.path.insert(0, SQL_RETRIEVAL_DIR)

from skeleton_utils import SkeletonParseError, sql2skeleton  # noqa: E402


DEFAULT_DATA_PATH = "./data/sql_pool.json"
DEFAULT_OUTPUT_DIR = "./data/skeleton_augmentation"


def _get_sql(rec: dict) -> str:
    return rec.get("sql") or rec.get("SQL") or rec.get("query") or rec.get("gold_sql") or ""


def _process_one(item):
    idx, rec = item
    db_id = rec.get("db_id")
    sql = _get_sql(rec)
    question = rec.get("question", "")

    try:
        skeleton = sql2skeleton(sql, None)
        return {
            "source_idx": idx,
            "db_id": db_id,
            "sql": sql,
            "question": question,
            "skeleton": skeleton,
            "error": None,
        }
    except SkeletonParseError as exc:
        return {
            "source_idx": idx,
            "db_id": db_id,
            "sql": sql,
            "question": question,
            "skeleton": None,
            "error": str(exc),
        }


def load_records(data_path: str) -> list[tuple[int, dict]]:
    """Stream a JSON array of records into (index, record) tuples."""
    records = []
    start = time.time()
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            records.append((idx, item))
            if idx and idx % 500000 == 0:
                print(f"  loaded={idx:,} elapsed={time.time() - start:.1f}s")
    return records


def maybe_sample(records: list[tuple[int, dict]], sample_size: int, seed: int) -> list[tuple[int, dict]]:
    if sample_size and 0 < sample_size < len(records):
        rng = random.Random(seed)
        sampled = rng.sample(records, sample_size)
        sampled.sort(key=lambda x: x[0])
        return sampled
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH,
                        help="JSON array of records, each with a SQL query (see module docstring).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Randomly sample this many records before extraction. 0 means use all records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=96)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, "skeleton_pool_raw.json")
    unique_path = os.path.join(args.output_dir, "skeleton_pool_unique.json")
    error_path = os.path.join(args.output_dir, "skeleton_pool_errors.jsonl")
    stats_path = os.path.join(args.output_dir, "skeleton_pool_stats.json")

    print("=" * 80)
    print("Build SQL Skeleton Pool")
    print(f"data_path={args.data_path}")
    print(f"output_dir={args.output_dir}")
    print(f"sample_size={args.sample_size} (0 = all) seed={args.seed}")
    print("=" * 80)

    print("\n[1/4] Loading records...")
    t0 = time.time()
    records = load_records(args.data_path)
    print(f"  loaded {len(records):,} records in {time.time() - t0:.1f}s")

    records = maybe_sample(records, args.sample_size, args.seed)
    print(f"  using {len(records):,} records")

    print("\n[2/4] Extracting skeletons...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        chunksize = max(1, len(records) // (args.num_workers * 8))
        results = list(executor.map(_process_one, records, chunksize=chunksize))
    print(f"  processed {len(results):,} records in {time.time() - t0:.1f}s")

    successes = [r for r in results if r["error"] is None and r["skeleton"]]
    errors = [r for r in results if r["error"] is not None or not r["skeleton"]]

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(successes, f, ensure_ascii=False)
    if errors:
        with open(error_path, "w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
    print(f"  success={len(successes):,} errors={len(errors):,}")
    print(f"  saved {raw_path}")
    if errors:
        print(f"  saved {error_path}")

    print("\n[3/4] Deduplicating skeletons...")
    skeleton_to_entries = {}
    skeleton_counter = Counter()
    for rec in successes:
        skeleton = rec["skeleton"]
        skeleton_counter[skeleton] += 1
        if skeleton not in skeleton_to_entries:
            skeleton_to_entries[skeleton] = {
                "pool_id": len(skeleton_to_entries),
                "skeleton": skeleton,
                "count": 0,
                "first_source_idx": rec["source_idx"],
                "first_db_id": rec["db_id"],
                "first_sql": rec["sql"],
                "first_question": rec.get("question", ""),
                "source_indices": [],
            }
        entry = skeleton_to_entries[skeleton]
        entry["count"] += 1
        if len(entry["source_indices"]) < 20:
            entry["source_indices"].append(rec["source_idx"])

    unique_records = list(skeleton_to_entries.values())
    with open(unique_path, "w", encoding="utf-8") as f:
        json.dump(unique_records, f, ensure_ascii=False)

    print("\n[4/4] Writing statistics...")
    freq_values = list(skeleton_counter.values())
    stats = {
        "data_path": args.data_path,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "used_records": len(records),
        "success_count": len(successes),
        "error_count": len(errors),
        "unique_skeleton_count": len(unique_records),
        "avg_sql_per_skeleton": round(len(successes) / max(len(unique_records), 1), 4),
        "max_frequency": max(freq_values) if freq_values else 0,
        "frequency_buckets": {
            "1": sum(1 for x in freq_values if x == 1),
            "2_5": sum(1 for x in freq_values if 2 <= x <= 5),
            "6_10": sum(1 for x in freq_values if 6 <= x <= 10),
            "11_plus": sum(1 for x in freq_values if x >= 11),
        },
        "top50_most_frequent": [
            {"skeleton": skel, "count": cnt}
            for skel, cnt in skeleton_counter.most_common(50)
        ],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  unique={len(unique_records):,}")
    print(f"  saved {unique_path}")
    print(f"  saved {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
