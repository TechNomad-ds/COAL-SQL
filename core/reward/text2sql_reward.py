"""
Text2SQL Reward Functions for COAL-SQL.

Simplified version with only format and execution rewards.
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

# Database path configuration via environment variables.
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


def execute_sql(db_file, gen_sql, gold_sql):
    """Execute SQL and compare results for bird dataset."""
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        cursor.execute(gen_sql)
        predicted_res = cursor.fetchall()

        cursor.execute(gold_sql)
        ground_truth_res = cursor.fetchall()

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
    # print(result["detail"]) # TODO:
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
                            num_workers=64, exec_time_out=5.0, **kwargs):
    """Execute accuracy reward function for bird dataset.

    Args:
        completions: the solution text
        ground_truth: dictionary containing target information
        method: the method to extract the solution
        score: the score for the correct answer
        num_workers: number of parallel workers
        exec_time_out: timeout for SQL execution
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
    print(f"{current_time} - [reward statistics] total={total_count}, timeout={timeout_ratio:.2%}, error={error_ratio:.2%}, correct={correct_ratio:.2%}")

    do_print = random.randint(1, 64) == 1
    # do_print = 1 # TODO
    if do_print:
        # current_time = now_str()
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


def text2sql_reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    """
    Text2SQL reward function adapted for the NaiveRewardManager of COAL-SQL.

    Args:
        data_source: data source identifier (unused)
        solution_str: full model generated output
        ground_truth: dict containing the ground truth, in the form:
            {
                "split": "train" | "dev",
                "question_id": int,
                "db_id": str,
                "sql": str  # gold SQL query
            }
        extra_info: extra information (unused)

    Returns:
        dict with "score" and "acc" keys
    """
    # 1. Format reward
    format_score = format_reward([solution_str])[0]  # 1.0 if format correct else 0.0

    # 2. Execution accuracy reward (single-worker, since naive is per-sample)
    accuracy_score = execute_accuracy_reward(
        [solution_str],
        [ground_truth],
        score=1.0,
        num_workers=1,
        exec_time_out=5.0,
    )[0]

    # 3. Combine rewards (format weight 1.0, execution accuracy weight 3.0)
    combined_score = 1.0 * format_score + 3.0 * accuracy_score

    return {
        "score": combined_score,
        "acc": accuracy_score > 0,  # correct if SQL executes to the right result
        "format_score": format_score,
        "accuracy_score": accuracy_score,
    }


if __name__ == "__main__":
    # Test code
    import json
    import os

    # Print paths for confirmation
    print(f"Testing Text2SQL reward function...")
    print(f"DB DEV PATH: {SQL_DATASET_DIR_DEV}")
    print(f"DB TRAIN PATH: {SQL_DATASET_DIR_TRAIN}")

    # Read the first few examples from BIRD dev.json
    dev_json_path = os.getenv("BIRD_DEV_JSON", "./data/bird/dev.json")
    with open(dev_json_path, 'r') as f:
        dev_data = json.load(f)

    for i in range(2):
        item = dev_data[i]
        print(f"\n--- Testing Case {i} ---")

        # Build a ground_truth as expected by the reward function
        # Note: BIRD uses 'SQL', while the reward function expects 'sql'
        gt = {
            "split": "dev",
            "question_id": item["question_id"],
            "db_id": item["db_id"],
            "sql": item["SQL"]
        }

        # Simulate a model output (with think and answer tags)
        test_solution = f"""<think>
The user wants to find: {item['question']}
I will use the {item['db_id']} database.
Reference info: {item.get('evidence', 'None')}
I will generate the SQL.
</think>
<answer>
{item['SQL']}
</answer>
"""

        result = text2sql_reward_func("_", test_solution, gt, None)
        print(f"Result (Correct): {result}")

        # Simulate a wrong SQL
        wrong_solution = f"""<think>\nSearching...\n</think>\n<answer>\nSELECT * FROM table_not_exists\n</answer>"""
        result_wrong = text2sql_reward_func("_", wrong_solution, gt, None)
        print(f"Result (Wrong): {result_wrong}")

        # Simulate a bad-format output
        bad_format_solution = f"""Just the SQL: {item['SQL']}"""
        result_format = text2sql_reward_func("_", bad_format_solution, gt, None)
        print(f"Result (Bad Format): {result_format}")
