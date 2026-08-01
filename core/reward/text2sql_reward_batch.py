"""
Text2SQL Reward Functions for COAL-SQL (Batch Version).

Batch variant adapted for BatchRewardManager. Derived from text2sql_reward.py.
"""

import os
import re
import sqlite3
import time
import random
from pathlib import Path
from typing import List, Tuple

from func_timeout import FunctionTimedOut, func_timeout
from tqdm.contrib.concurrent import process_map
from functools import lru_cache

# Database path configuration via environment variables (same as the naive version).
SQL_DATASET_DIR_DEV = os.getenv("SQL_DATASET_DIR_DEV", "./data/bird/dev_databases")
SQL_DATASET_DIR_TRAIN = os.getenv("SQL_DATASET_DIR_TRAIN", "./data/bird/train_databases")
SQL_DATASET_DIR_QB = os.getenv("SQL_DATASET_DIR_QB", "./data/question_bank/databases")


def format_reward(completions, **kwargs):
    """Reward function that checks if reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = completions
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]

    num_matches = sum(1 for match in matches if match)
    total_completions = len(completion_contents)

    if total_completions > 0:
        match_proportion = num_matches / total_completions

    return [1.0 if match else 0.0 for match in matches]


def now_str(time_format="%Y-%m-%d %H:%M:%S"):
    return time.strftime(time_format, time.localtime())


def extract_sql(gen: str):
    """Extract SQL from model output."""
    gen = gen.replace("\n", " ")
    gen = gen.replace(";", "")
    gen = gen.replace("</s>", "")
    gen = gen.replace("```sql ", "")
    gen = gen.strip().replace("```", "")

    # Extract from <answer> tags if present
    if gen.find("</think>") > -1:
        gen = gen[gen.find("</think>") + len("</think>"):]

    gen = extract_xml_answer(gen)
    gen = gen.replace("</answer>", "")
    gen = gen.replace("<answer>", "")

    return gen


def extract_xml_answer(text: str) -> str:
    """Extract content from <answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


@lru_cache(maxsize=1000000)
def cache_gold_sql(db_file, gold_sql):
    """Cache gold_sql execution result. Returns a tuple of tuples for hashability."""
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(gold_sql)
        # Convert to tuple of tuples for hashability (lru_cache requirement)
        return tuple(tuple(row) for row in cursor.fetchall())


def execute_sql(db_file, gen_sql, gold_sql):
    """Execute SQL and compare results for bird dataset."""
    # gold_sql result from cache
    ground_truth_res = cache_gold_sql(db_file, gold_sql)

    # gen_sql execute each time
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(gen_sql)
        predicted_res = cursor.fetchall()

    # For bird dataset, use simple set comparison
    if set(predicted_res) == set(ground_truth_res):
        return 1
    return 0


def execute_model(packed):
    """Execute SQL model with timeout handling."""
    q_id, db_file, gen_sql, gold_sql, exec_time_out = packed

    status = "failed"
    detail = None
    try:
        res = func_timeout(exec_time_out, execute_sql, args=(db_file, gen_sql, gold_sql))
        status = "success"
    except FunctionTimedOut:
        status = "timeout"
        res = 0
    except Exception as e:
        detail = str(e)
        status = "error"
        # print(f"[error] {detail} [sql] {gen_sql}")
        res = 0

    result = {"id": q_id, "res": res, "status": status, "detail": detail}
    return result


def run_sqls_parallel(packed_payload, num_workers=64, exec_time_out=5.0):
    """Run SQL execution in parallel."""
    ret = process_map(
        execute_model,
        [(*payload, exec_time_out) for payload in packed_payload],
        max_workers=num_workers,
        chunksize=10,
    )
    return ret


def execute_accuracy_reward(completions, ground_truth, method='strict', score=1.,
                            num_workers=256, exec_time_out=10.0, **kwargs):
    """Execute accuracy reward function for bird dataset.

    Args:
        completions: the solution text list (batched)
        ground_truth: dictionary containing target information
        method: the method to extract the solution
        score: the score for the correct answer
        num_workers: number of parallel workers
        exec_time_out: timeout for SQL execution

    Returns:
        list of rewards (one score per sample)
    """
    contents = completions

    predict_sqls = []
    gold_sqls = []
    db_path_list = []
    question_ids = []
    db_id_list = []

    for content, gt in zip(contents, ground_truth):
        predict_sql = extract_sql(content)
        predict_sqls.append(predict_sql)

        db_split = gt.get('split', 'dev')
        question_id = gt['question_id']
        db_id = gt['db_id']
        gold_sql = gt['sql']

        # Use configured database paths
        if db_split == "dev":
            db_root = SQL_DATASET_DIR_DEV
        elif db_split == "question_bank":
            db_root = SQL_DATASET_DIR_QB
        else:
            db_root = SQL_DATASET_DIR_TRAIN

        db_file = Path(db_root).joinpath(db_id).joinpath(f"{db_id}.sqlite").resolve()
        db_file = str(db_file)

        question_ids.append(question_id)
        db_id_list.append(db_id)
        db_path_list.append(db_file)
        gold_sqls.append(gold_sql)

    packed_payload = zip(question_ids, db_path_list, predict_sqls, gold_sqls)

    exec_result = run_sqls_parallel(
        packed_payload,
        num_workers=num_workers,
        exec_time_out=exec_time_out,
    )

    rewards = [score if res["res"] == 1 else 0 for res in exec_result]

    # Aggregate timeout / error / correct ratios
    total_count = len(exec_result)
    timeout_count = sum(1 for res in exec_result if res["status"] == "timeout")
    error_count = sum(1 for res in exec_result if res["status"] == "error")
    correct_count = sum(1 for res in exec_result if res["res"] == 1)

    timeout_ratio = timeout_count / total_count if total_count > 0 else 0
    error_ratio = error_count / total_count if total_count > 0 else 0
    correct_ratio = correct_count / total_count if total_count > 0 else 0

    # Print statistics
    current_time = now_str()
    # Cache statistics
    cache_info = cache_gold_sql.cache_info()
    cache_hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
    print(f"[reward statistics] {current_time} | total={total_count}, timeout={timeout_ratio:.2%}, error={error_ratio:.2%}, acc_reward={correct_ratio:.2%}")
    print(f"[gold_sql cache] hits={cache_info.hits}, misses={cache_info.misses}, hit_rate={cache_hit_rate:.2%}, size={cache_info.currsize}")

    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"{current_time} - Question_id: {question_ids[0]} | db_id: {db_id_list[0]}")
        print(f"Predict sql correct: {rewards[0]}")
        print(f"Gold sql: {gold_sqls[0]}")
        print(f"Extracted sql: {predict_sqls[0]}")
        print(f"Ex detail: {exec_result[0]['detail']}")
        print(f"Solution string: {contents[0]}")
        if rewards[0] == 0:
            print(f"Invalid SQL !!!")

    return rewards


def text2sql_reward_batch_func(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Batch Text2SQL reward function adapted for the BatchRewardManager of COAL-SQL.

    Args:
        data_sources: list of data source identifiers (unused)
        solution_strs: list of model generated outputs (batched)
        ground_truths: list of ground truth dictionaries
        extra_infos: list of extra information (unused)

    Returns:
        list of scores (one score per sample)
    """
    # 1. Format reward (batched)
    format_scores = format_reward(solution_strs)

    # 2. Execution accuracy reward (batched, multi-worker)
    # execute_accuracy_reward's first parameter is named `completions`; pass solution_strs positionally.
    accuracy_scores = execute_accuracy_reward(
        solution_strs,
        ground_truths,
        score=1.0,
        num_workers=256,
        exec_time_out=10.0,
    )

    # 3. Combine rewards (format weight 1.0, execution accuracy weight 3.0)
    combined_scores = [1.0 * f + 3.0 * a for f, a in zip(format_scores, accuracy_scores)]

    # Statistics
    total = len(solution_strs)
    avg_format = sum(format_scores) / total if total > 0 else 0
    avg_acc = sum(accuracy_scores) / total if total > 0 else 0
    avg_combined = sum(combined_scores) / total if total > 0 else 0
    current_time = now_str()
    print(f"[batch reward summary] {current_time} | total={total}, format_pass={avg_format:.2%}, acc_pass={avg_acc:.2%}, avg_score={avg_combined:.2f}")

    # 4. Build return value: one dict per sample
    results = [
        {
            "score": combined_score,
            "acc": (format_score > 0) and (accuracy_score > 0),  # acc=True only if both format and SQL execution are correct
            "format_score": format_score,
            "accuracy_score": accuracy_score,
        }
        for combined_score, format_score, accuracy_score in zip(combined_scores, format_scores, accuracy_scores)
    ]

    return results


if __name__ == "__main__":
    # Test the batch reward function
    import json
    import os

    print(f"Testing Batch Text2SQL reward function...")
    print(f"DB DEV PATH: {SQL_DATASET_DIR_DEV}")
    print(f"DB TRAIN PATH: {SQL_DATASET_DIR_TRAIN}")

    # Read the first few examples from BIRD dev.json
    dev_json_path = os.getenv("BIRD_DEV_JSON", "./data/bird/dev.json")
    if not os.path.exists(dev_json_path):
        print(f"Warning: {dev_json_path} not found, skipping real data test.")
    else:
        with open(dev_json_path, 'r') as f:
            dev_data = json.load(f)

        test_solutions = []
        test_ground_truths = []

        for i in range(3):
            item = dev_data[i]
            # Build a flat ground_truth as expected by the reward function
            gt = {
                "split": "dev",
                "question_id": item["question_id"],
                "db_id": item["db_id"],
                "sql": item["SQL"]
            }
            test_ground_truths.append(gt)

            # Build model outputs
            if i == 0:
                # Correct example
                sol = f"<think>\nAnalyze {item['db_id']}\n</think>\n<answer>\n{item['SQL']}\n</answer>"
            elif i == 1:
                # Wrong-SQL example
                sol = f"<think>\nAnalyze {item['db_id']}\n</think>\n<answer>\nSELECT * FROM table_not_exists\n</answer>"
            else:
                # Bad-format example
                sol = f"Just the SQL: {item['SQL']}"
            test_solutions.append(sol)

        print(f"\n--- Testing batch of {len(test_solutions)} samples ---")
        results = text2sql_reward_batch_func(["bird"]*3, test_solutions, test_ground_truths, [None]*3)

        for i, res in enumerate(results):
            status = "Correct" if i == 0 else ("Wrong SQL" if i == 1 else "Bad Format")
            print(f"Sample {i} ({status}): score={res['score']}, acc={res['acc']}, format_score={res['format_score']}, accuracy_score={res['accuracy_score']}")
