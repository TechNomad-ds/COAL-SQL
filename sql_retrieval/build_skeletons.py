#!/usr/bin/env python3
"""
Build skeletons from SQL data using DAIL-SQL's sql2skeleton method.

Reads question_bank_filtered.json and tables.json, converts each SQL to skeleton
using multiprocessing for speed, outputs skeletons.json and statistics.

Usage:
    conda run -n coalsql python build_skeletons.py
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from skeleton_utils import sql2skeleton

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_FILE = "./data/question_bank/question_bank.json"
TABLES_FILE = "./data/question_bank/tables.json"
OUTPUT_DIR = "./data/skeleton_index"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SKELETONS_FILE = os.path.join(OUTPUT_DIR, "skeletons.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "skeleton_stats.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "skeleton_errors.log")

NUM_WORKERS = 96

# ─── Global schema_map for worker processes (shared via fork COW) ────────────
_schema_map = None


def _init_worker(schema_map):
    """Initialize each worker process with the shared schema_map."""
    global _schema_map
    _schema_map = schema_map


def _process_one(item):
    """Process a single data item: convert SQL to skeleton.
    
    Returns a dict with keys: idx, db_id, sql, question, skeleton, error.
    """
    idx = item["_idx"]
    gt = item["reward_model"]["ground_truth"]
    db_id = gt["db_id"]
    sql = gt["sql"]
    # Extract question from prompt (second message = user)
    question = ""
    if "prompt" in item and len(item["prompt"]) >= 2:
        question = item["prompt"][1].get("content", "")[:200]

    if db_id not in _schema_map:
        return {
            "idx": idx, "db_id": db_id, "sql": sql, "question": question,
            "skeleton": None, "error": f"db_id '{db_id}' not found in tables.json"
        }

    try:
        skeleton = sql2skeleton(sql, _schema_map[db_id])
        return {
            "idx": idx, "db_id": db_id, "sql": sql, "question": question,
            "skeleton": skeleton, "error": None
        }
    except Exception as e:
        return {
            "idx": idx, "db_id": db_id, "sql": sql, "question": question,
            "skeleton": None, "error": str(e)
        }


def load_schema_map(tables_file: str) -> dict:
    """Load tables.json and build db_id -> schema mapping."""
    print(f"Loading schemas from {tables_file} ...")
    with open(tables_file, "r", encoding="utf-8") as f:
        tables = json.load(f)
    schema_map = {}
    for t in tables:
        schema_map[t["db_id"]] = t
    print(f"  Loaded {len(schema_map)} database schemas.")
    return schema_map


def main():
    # Load schema
    schema_map = load_schema_map(TABLES_FILE)

    # Load data
    print(f"Loading data from {DATA_FILE} ...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"  Loaded {total} records.")

    # Inject index for tracking
    for i, item in enumerate(data):
        item["_idx"] = i

    # ─── Multiprocess conversion ─────────────────────────────────────────────
    print(f"Converting SQL to skeletons with {NUM_WORKERS} workers ...")
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(schema_map,)
    ) as executor:
        # Use chunksize to reduce IPC overhead for large dataset
        chunksize = max(1, total // (NUM_WORKERS * 4))
        raw_results = list(executor.map(_process_one, data, chunksize=chunksize))

    elapsed = time.time() - t0

    # Separate successes and errors
    results = []
    errors = []
    skeleton_counter = Counter()

    for r in raw_results:
        if r["error"] is not None:
            errors.append(r)
        else:
            results.append({
                "idx": r["idx"],
                "db_id": r["db_id"],
                "sql": r["sql"],
                "skeleton": r["skeleton"],
                "question": r["question"],
            })
            skeleton_counter[r["skeleton"]] += 1

    print(f"Done! {len(results)} succeeded, {len(errors)} failed in {elapsed:.1f}s")

    # ─── Save skeletons.json ─────────────────────────────────────────────────
    print(f"Saving skeletons to {SKELETONS_FILE} ...")
    with open(SKELETONS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(results)} records.")

    # ─── Save error log ──────────────────────────────────────────────────────
    if errors:
        print(f"Saving {len(errors)} errors to {ERROR_LOG_FILE} ...")
        with open(ERROR_LOG_FILE, "w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    # ─── Statistics ──────────────────────────────────────────────────────────
    unique_count = len(skeleton_counter)
    total_success = len(results)

    freq_values = list(skeleton_counter.values())
    freq_counter = Counter(freq_values)

    top50 = skeleton_counter.most_common(50)

    stats = {
        "total_records": total,
        "success_count": total_success,
        "error_count": len(errors),
        "unique_skeleton_count": unique_count,
        "avg_sql_per_skeleton": round(total_success / unique_count, 2) if unique_count > 0 else 0,
        "max_frequency": top50[0][1] if top50 else 0,
        "skeletons_with_1_sql": freq_counter.get(1, 0),
        "skeletons_with_2to5_sql": sum(freq_counter.get(i, 0) for i in range(2, 6)),
        "skeletons_with_6to10_sql": sum(freq_counter.get(i, 0) for i in range(6, 11)),
        "skeletons_with_10plus_sql": sum(v for k, v in freq_counter.items() if k > 10),
        "top50_most_frequent": [
            {"skeleton": skel, "count": cnt} for skel, cnt in top50
        ],
    }

    print(f"\nSaving statistics to {STATS_FILE} ...")
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ─── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SKELETON STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total records:            {total}")
    print(f"Successfully converted:   {total_success}")
    print(f"Errors:                   {len(errors)}")
    print(f"Unique skeletons:         {unique_count}")
    print(f"Avg SQL per skeleton:     {stats['avg_sql_per_skeleton']}")
    print(f"Max frequency:            {stats['max_frequency']}")
    print(f"Skeletons with 1 SQL:     {stats['skeletons_with_1_sql']}")
    print(f"Skeletons with 2-5 SQL:   {stats['skeletons_with_2to5_sql']}")
    print(f"Skeletons with 6-10 SQL:  {stats['skeletons_with_6to10_sql']}")
    print(f"Skeletons with 10+ SQL:   {stats['skeletons_with_10plus_sql']}")
    print("=" * 60)
    print("\nTop 10 most frequent skeletons:")
    for i, (skel, cnt) in enumerate(top50[:10]):
        print(f"  {i+1}. [{cnt}x] {skel}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
