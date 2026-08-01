#!/usr/bin/env python3
"""Synthesize BIRD-style text2sql data from selected SQL skeletons."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SQL_RETRIEVAL_DIR = PROJECT_ROOT / "sql_retrieval"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SQL_RETRIEVAL_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from core.llm_client import LLMClient  # noqa: E402
from execution_utils import validate_sql_non_empty  # noqa: E402
from schema_utils import BirdSchemaProvider  # noqa: E402


DEFAULT_SELECTED = os.environ.get("SELECTED_SKELETONS", "./data/skeleton_augmentation/selected_skeletons.json")
DEFAULT_BIRD_TRAIN = os.environ.get("BIRD_TRAIN_JSON", "./data/train/bird_train.json")
DEFAULT_BIRD_DATA_DIR = os.environ.get("BIRD_DATA_DIR", "./data/bird/train")
DEFAULT_OUTPUT_DIR = os.environ.get("SYNTHESIS_OUTPUT_DIR", "./data/skeleton_augmentation/synthesis")
DEFAULT_PROMPT_DIR = str(CURRENT_DIR / "prompts")

TABLE_PLACEHOLDER = "TABLE_PLACEHOLDER"
COLUMN_PLACEHOLDER = "COLUMN_PLACEHOLDER"
LITERAL_PLACEHOLDER = "LITERAL_PLACEHOLDER"


@dataclass(frozen=True)
class DemoExample:
    db_id: str
    skeleton: str
    sql: str
    question: str
    evidence: str


@dataclass
class StageResult:
    content: str | None
    api_base: str | None = None
    error: str | None = None


class PromptTemplates:
    def __init__(self, prompt_dir: str):
        prompt_dir_path = Path(prompt_dir)
        self.skeleton_instantiate = (prompt_dir_path / "sql_instantiation.txt").read_text(encoding="utf-8")
        self.backward_question_evidence = (prompt_dir_path / "nl_question_generation.txt").read_text(encoding="utf-8")
        self.forward_verification_cot = (prompt_dir_path / "semantic_consistency_verification.txt").read_text(encoding="utf-8")


def _strip_sql_comments(sql: str) -> str:
    result = []
    i = 0
    in_single = False
    in_double = False
    in_bracket = False

    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if in_single:
            result.append(ch)
            if ch == "'" and nxt == "'":
                result.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            result.append(ch)
            if ch == '"' and nxt == '"':
                result.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue
        if in_bracket:
            result.append(ch)
            if ch == "]":
                in_bracket = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            result.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            result.append(ch)
            i += 1
            continue
        if ch == "[":
            in_bracket = True
            result.append(ch)
            i += 1
            continue
        if ch == "-" and nxt == "-":
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            result.append(" ")
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < len(sql) else 0
            result.append(" ")
            continue
        result.append(ch)
        i += 1

    return "".join(result)


def sql2typed_skeleton(sql: str) -> str:
    """Convert SQL to typed placeholders (<TABLE>/<COLUMN>/<LITERAL>) for few-shot demos."""
    parsed_sql = sqlglot.parse_one(_strip_sql_comments(sql).strip().rstrip(";"), dialect="sqlite")

    for alias in list(parsed_sql.find_all(exp.Alias)):
        alias.replace(alias.this)
    for cte in parsed_sql.find_all(exp.CTE):
        cte.set("alias", exp.TableAlias(this=exp.to_identifier(TABLE_PLACEHOLDER)))
    for subquery in parsed_sql.find_all(exp.Subquery):
        subquery.set("alias", None)

    for table in list(parsed_sql.find_all(exp.Table)):
        table.replace(sqlglot.to_table(TABLE_PLACEHOLDER))
    for column in list(parsed_sql.find_all(exp.Column)):
        column.replace(sqlglot.to_column(COLUMN_PLACEHOLDER))
    for literal in list(parsed_sql.find_all(exp.Literal)):
        literal.replace(sqlglot.to_column(LITERAL_PLACEHOLDER))

    skeleton = parsed_sql.sql(dialect="sqlite")
    return (
        skeleton.replace(f"-{LITERAL_PLACEHOLDER}", "<LITERAL>")
        .replace(COLUMN_PLACEHOLDER, "<COLUMN>")
        .replace(TABLE_PLACEHOLDER, "<TABLE>")
        .replace(LITERAL_PLACEHOLDER, "<LITERAL>")
    )


def extract_tag(text: str | None, tag: str) -> str | None:
    if not text:
        return None
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def clean_sql_answer(sql: str | None) -> str | None:
    if not sql:
        return None
    sql = sql.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        sql = fence.group(1).strip()
    return sql.strip().strip(";").strip() or None


def parse_backward_response(text: str | None) -> tuple[str | None, str | None]:
    return extract_tag(text, "question"), extract_tag(text, "evidence")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, payload: Any, indent: int | None = 2) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, path)


def append_jsonl(path: str, records: list[dict]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_sql(record: dict) -> str:
    for key in ("SQL", "output_sql", "sql", "query", "gold_sql"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def load_demonstrations(bird_train_path: str) -> tuple[dict[str, list[DemoExample]], dict[str, int]]:
    records = load_json(bird_train_path)
    demos_by_db: dict[str, list[DemoExample]] = defaultdict(list)
    stats = {
        "total": len(records),
        "skeleton_parse_success": 0,
        "skeleton_parse_error": 0,
        "missing_sql_or_db": 0,
    }
    for record in records:
        db_id = record.get("db_id")
        sql = extract_sql(record)
        if not db_id or not sql:
            stats["missing_sql_or_db"] += 1
            continue
        try:
            skeleton = sql2typed_skeleton(sql)
        except Exception:
            stats["skeleton_parse_error"] += 1
            continue
        stats["skeleton_parse_success"] += 1
        demos_by_db[str(db_id)].append(
            DemoExample(
                db_id=str(db_id),
                skeleton=skeleton,
                sql=sql,
                question=str(record.get("question", "")),
                evidence=str(record.get("evidence", "")),
            )
        )
    return dict(demos_by_db), stats


def sample_demonstrations(
    demos_by_db: dict[str, list[DemoExample]],
    db_id: str,
    n_shot: int,
    rng: random.Random,
) -> list[DemoExample]:
    demos = demos_by_db.get(db_id, [])
    if n_shot <= 0 or not demos:
        return []
    return rng.sample(demos, min(n_shot, len(demos)))


def build_instantiate_demos(demos: list[DemoExample]) -> str:
    parts = []
    for demo in demos:
        parts.append(f"Skeleton: {demo.skeleton}\nSQL: <answer>{demo.sql}</answer>")
    return "\n\n".join(parts)


def build_backward_demos(demos: list[DemoExample]) -> str:
    parts = []
    for demo in demos:
        parts.append(
            "SQL: "
            f"{demo.sql}\n"
            f"<question>\n{demo.question}\n</question>\n"
            f"<evidence>\n{demo.evidence}\n</evidence>"
        )
    return "\n\n".join(parts)


def render_instantiate_prompt(
    templates: PromptTemplates,
    schema: str,
    skeleton: str,
    demonstrations: list[DemoExample],
) -> str:
    return (
        templates.skeleton_instantiate
        .replace("{DATABASE_SCHEMA}", schema)
        .replace("{SKELETON}", skeleton)
        .replace("{DEMONSTRATIONS}", build_instantiate_demos(demonstrations))
    )


def render_backward_prompt(
    templates: PromptTemplates,
    schema: str,
    sql: str,
    demonstrations: list[DemoExample],
) -> str:
    return (
        templates.backward_question_evidence
        .replace("{DATABASE_SCHEMA}", schema)
        .replace("{SQL}", sql)
        .replace("{DEMONSTRATIONS}", build_backward_demos(demonstrations))
    )


def render_forward_prompt(
    templates: PromptTemplates,
    schema: str,
    question: str,
    evidence: str,
) -> str:
    return (
        templates.forward_verification_cot
        .replace("{DATABASE_SCHEMA}", schema)
        .replace("{QUESTION}", question)
        .replace("{EVIDENCE}", evidence)
    )


def call_llm(
    client: LLMClient,
    prompt: str,
    step: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
    presence_penalty: float | None = None,
) -> StageResult:
    try:
        with client.token_tracker.step(step):
            optional_kwargs: dict = {}
            if extra_body is not None:
                optional_kwargs["extra_body"] = extra_body
            if presence_penalty is not None:
                optional_kwargs["presence_penalty"] = presence_penalty
            response = client.chat(
                query=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **optional_kwargs,
            )
        return StageResult(content=response.content, api_base=response.api_base)
    except Exception as exc:
        return StageResult(content=None, error=str(exc))


def validation_to_fail_reason(stage: str, reason: str) -> str:
    if reason == "empty_result":
        return f"{stage}_sql_empty"
    return f"{stage}_sql_invalid"


def truncate_text(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def make_failure_record(
    skeleton_record: dict,
    db_id: str | None,
    attempt_id: int,
    stage: str,
    reason: str,
    raw_response: str | None = None,
    error: str | None = None,
    validation_reason: str | None = None,
) -> dict:
    return {
        "selection_rank": skeleton_record.get("selection_rank"),
        "pool_id": skeleton_record.get("pool_id"),
        "db_id": db_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "reason": reason,
        "validation_reason": validation_reason,
        "error": error,
        "raw_response": truncate_text(raw_response, 2000),
        "typed_skeleton": skeleton_record.get("typed_skeleton") or skeleton_record.get("skeleton"),
        "underscore_skeleton": skeleton_record.get("underscore_skeleton"),
        "timestamp": time.time(),
    }


def synthesize_one_skeleton(
    skeleton_record: dict,
    db_ids: list[str],
    demos_by_db: dict[str, list[DemoExample]],
    schema_provider: BirdSchemaProvider,
    templates: PromptTemplates,
    client: LLMClient,
    args: argparse.Namespace,
) -> dict:
    selection_rank = int(skeleton_record.get("selection_rank", 0))
    rng = random.Random(args.seed + selection_rank * 1000003)
    failures = []
    typed_skeleton = skeleton_record.get("typed_skeleton") or skeleton_record.get("skeleton")
    underscore_skeleton = skeleton_record.get("underscore_skeleton")

    if not typed_skeleton:
        failures.append(
            make_failure_record(
                skeleton_record,
                None,
                0,
                "setup",
                "skeleton_missing",
            )
        )
        return {"ok": False, "failures": failures, "attempts": 0, "selection_rank": selection_rank}

    for attempt_id in range(1, args.max_attempts_per_skeleton + 1):
        db_id = rng.choice(db_ids)
        try:
            db_path = schema_provider.get_db_path(db_id)
            schema = schema_provider.get_schema(db_id)

            instantiate_demos = sample_demonstrations(demos_by_db, db_id, args.n_shot_instantiate, rng)
            instantiate_prompt = render_instantiate_prompt(templates, schema, typed_skeleton, instantiate_demos)
            instantiate_response = call_llm(
                client,
                instantiate_prompt,
                "skeleton_instantiate",
                args.instantiate_temperature,
                args.instantiate_max_tokens,
                args.extra_body,
                args.presence_penalty,
            )
            if instantiate_response.error:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "instantiate",
                        "llm_request_fail",
                        error=instantiate_response.error,
                    )
                )
                continue
            initial_sql = clean_sql_answer(extract_tag(instantiate_response.content, "answer"))
            if not initial_sql:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "instantiate",
                        "instantiate_parse_fail",
                        raw_response=instantiate_response.content,
                    )
                )
                continue
            initial_validation = validate_sql_non_empty(
                db_path,
                initial_sql,
                timeout=args.sql_timeout,
                fetch_limit=args.sql_fetch_limit,
            )
            if not initial_validation.ok:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "instantiate",
                        validation_to_fail_reason("instantiate", initial_validation.reason),
                        raw_response=instantiate_response.content,
                        error=initial_validation.result_str,
                        validation_reason=initial_validation.reason,
                    )
                )
                continue

            backward_demos = sample_demonstrations(demos_by_db, db_id, args.n_shot_backward, rng)
            backward_prompt = render_backward_prompt(templates, schema, initial_sql, backward_demos)
            backward_response = call_llm(
                client,
                backward_prompt,
                "backward_question_evidence",
                args.backward_temperature,
                args.backward_max_tokens,
                args.extra_body,
                args.presence_penalty,
            )
            if backward_response.error:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "backward",
                        "llm_request_fail",
                        error=backward_response.error,
                    )
                )
                continue
            question, evidence = parse_backward_response(backward_response.content)
            if not question and not evidence:
                reason = "backward_parse_fail"
            elif not question:
                reason = "backward_question_missing"
            elif not evidence:
                reason = "backward_evidence_missing"
            else:
                reason = None
            if reason:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "backward",
                        reason,
                        raw_response=backward_response.content,
                    )
                )
                continue

            forward_prompt = render_forward_prompt(templates, schema, question, evidence)
            forward_response = call_llm(
                client,
                forward_prompt,
                "forward_verification_cot",
                args.forward_temperature,
                args.forward_max_tokens,
                args.extra_body,
                args.presence_penalty,
            )
            if forward_response.error:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "forward",
                        "llm_request_fail",
                        error=forward_response.error,
                    )
                )
                continue
            final_sql = clean_sql_answer(extract_tag(forward_response.content, "answer"))
            reasoning = extract_tag(forward_response.content, "reasoning")
            if not final_sql:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "forward",
                        "forward_parse_fail",
                        raw_response=forward_response.content,
                    )
                )
                continue
            if not reasoning:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "forward",
                        "forward_reasoning_missing",
                        raw_response=forward_response.content,
                    )
                )
                continue
            final_validation = validate_sql_non_empty(
                db_path,
                final_sql,
                timeout=args.sql_timeout,
                fetch_limit=args.sql_fetch_limit,
            )
            if not final_validation.ok:
                failures.append(
                    make_failure_record(
                        skeleton_record,
                        db_id,
                        attempt_id,
                        "forward",
                        validation_to_fail_reason("forward", final_validation.reason),
                        raw_response=forward_response.content,
                        error=final_validation.result_str,
                        validation_reason=final_validation.reason,
                    )
                )
                continue

            sample = {
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "SQL": final_sql,
                "output_reasoning": reasoning,
                "output_sql": final_sql,
                "typed_skeleton": typed_skeleton,
                "underscore_skeleton": underscore_skeleton,
                "pool_id": skeleton_record.get("pool_id"),
                "selection_rank": selection_rank,
                "outlier_score": skeleton_record.get("outlier_score"),
                "selection_distance": skeleton_record.get("selection_distance"),
                "source": "skeleton_augmentation",
            }
            debug = {
                "selection_rank": selection_rank,
                "pool_id": skeleton_record.get("pool_id"),
                "db_id": db_id,
                "attempt_id": attempt_id,
                "initial_sql": initial_sql,
                "final_sql": final_sql,
                "instantiate_api_base": instantiate_response.api_base,
                "backward_api_base": backward_response.api_base,
                "forward_api_base": forward_response.api_base,
                "instantiate_prompt": instantiate_prompt,
                "backward_prompt": backward_prompt,
                "forward_prompt": forward_prompt,
                "instantiate_response": instantiate_response.content,
                "backward_response": backward_response.content,
                "forward_response": forward_response.content,
                "initial_sql_result": initial_validation.result_str,
                "final_sql_result": final_validation.result_str,
            }
            return {
                "ok": True,
                "sample": sample,
                "full": {
                    "status": "success",
                    "sample": sample,
                    "debug": debug,
                    "timestamp": time.time(),
                },
                "failures": failures,
                "attempts": attempt_id,
                "selection_rank": selection_rank,
            }
        except Exception as exc:
            failures.append(
                make_failure_record(
                    skeleton_record,
                    db_id if "db_id" in locals() else None,
                    attempt_id,
                    "unexpected",
                    "unexpected_error",
                    error=str(exc),
                )
            )

    return {
        "ok": False,
        "failures": failures,
        "attempts": args.max_attempts_per_skeleton,
        "selection_rank": selection_rank,
    }


def load_success_records(full_path: str) -> dict[int, dict]:
    successes = {}
    for record in iter_jsonl(full_path) or []:
        if record.get("status") != "success":
            continue
        sample = record.get("sample") or {}
        rank = sample.get("selection_rank")
        if rank is None:
            continue
        successes[int(rank)] = record
    return successes


def load_failure_counts(failure_path: str) -> Counter:
    counter = Counter()
    for record in iter_jsonl(failure_path) or []:
        counter[record.get("reason", "unknown")] += 1
    return counter


def materialize_main_output(output_json: str, success_records: dict[int, dict], target_count: int) -> list[dict]:
    samples = [
        record["sample"]
        for _, record in sorted(success_records.items())
        if record.get("sample")
    ][:target_count]
    dump_json(output_json, samples, indent=2)
    return samples


def write_stats(
    stats_path: str,
    selected_count: int,
    target_count: int,
    success_records: dict[int, dict],
    failure_counts: Counter,
    demo_stats: dict[str, int],
    token_usage: dict | None,
) -> None:
    samples = [record["sample"] for record in success_records.values() if record.get("sample")]
    db_counts = Counter(sample.get("db_id") for sample in samples)
    stats = {
        "selected_count": selected_count,
        "target_count": target_count,
        "success_count": len(samples),
        "failure_count": int(sum(failure_counts.values())),
        "failure_reason_counts": dict(failure_counts),
        "db_success_counts": dict(sorted(db_counts.items())),
        "demo_stats": demo_stats,
        "token_usage": token_usage or {},
        "updated_at": time.time(),
    }
    dump_json(stats_path, stats, indent=2)


def parse_api_urls(api_urls: str) -> list[str]:
    return [url.strip() for url in api_urls.split(",") if url.strip()]


def run_generation(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, args.output_json)
    full_jsonl = os.path.join(args.output_dir, args.full_jsonl)
    failures_jsonl = os.path.join(args.output_dir, args.failures_jsonl)
    stats_json = os.path.join(args.output_dir, args.stats_json)

    selected = load_json(args.selected_skeletons)
    selected = selected[: args.max_selected] if args.max_selected else selected
    demos_by_db, demo_stats = load_demonstrations(args.bird_train)
    db_ids = sorted(demos_by_db)
    if not db_ids:
        raise RuntimeError("no BIRD demonstrations with successfully parsed typed skeletons")

    print("=" * 80)
    print("Skeleton-guided BIRD synthesis")
    print(f"selected_skeletons={args.selected_skeletons} count={len(selected):,}")
    print(f"bird_train={args.bird_train}")
    print(f"demo_parse_success={demo_stats['skeleton_parse_success']:,}/{demo_stats['total']:,}")
    print(f"db_count_with_demos={len(db_ids):,}")
    print(f"output_dir={args.output_dir}")
    print("=" * 80)

    # Build qwen3.6-specific extra_body to align with the vLLM deployment
    # (--reasoning-parser qwen3). Thinking is disabled by default for fast,
    # cheap synthesis; pass --enable_thinking to turn it on.
    args.extra_body = {
        "top_k": args.top_k,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }

    if args.dry_run:
        print("Dry run only. No LLM requests will be sent.")
        print(f"first_db={db_ids[0]} demo_count={len(demos_by_db[db_ids[0]])}")
        return

    templates = PromptTemplates(args.prompt_dir)
    schema_provider = BirdSchemaProvider(
        bird_data_dir=args.bird_data_dir,
        enable_comment=True,
        enable_values=True,
        num_sampled_values=args.num_sampled_values,
    )
    client = LLMClient(
        api_urls=parse_api_urls(args.api_urls),
        api_key=args.api_key,
        model=args.model,
        temperature=args.forward_temperature,
        max_tokens=args.forward_max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    success_records = load_success_records(full_jsonl)
    failure_counts = load_failure_counts(failures_jsonl)
    done_ranks = set(success_records)
    current_success = len(success_records)
    current_attempts = current_success + sum(failure_counts.values())
    print(f"Resume: success={current_success:,} attempts_seen={current_attempts:,}")

    def persist_result(result: dict) -> None:
        nonlocal current_success, current_attempts, success_records, failure_counts
        failures = result.get("failures") or []
        append_jsonl(failures_jsonl, failures)
        for failure in failures:
            failure_counts[failure.get("reason", "unknown")] += 1
        current_attempts += len(failures)
        if result.get("ok"):
            append_jsonl(full_jsonl, [result["full"]])
            rank = int(result["selection_rank"])
            success_records[rank] = result["full"]
            current_success = len(success_records)
            current_attempts += 1

    def run_batch(candidates: list[dict]) -> list[dict]:
        failed = []
        if not candidates:
            return failed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    synthesize_one_skeleton,
                    skeleton_record,
                    db_ids,
                    demos_by_db,
                    schema_provider,
                    templates,
                    client,
                    args,
                )
                for skeleton_record in candidates
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                persist_result(result)
                if not result.get("ok"):
                    rank = int(result.get("selection_rank", 0))
                    if rank:
                        failed.append(rank)
                if idx % args.progress_every == 0 or result.get("ok"):
                    print(
                        f"progress batch={idx:,}/{len(candidates):,} "
                        f"success={len(success_records):,}/{args.target_count:,} "
                        f"attempts~={current_attempts:,}"
                    )
        materialize_main_output(output_json, success_records, args.target_count)
        write_stats(
            stats_json,
            len(selected),
            args.target_count,
            success_records,
            failure_counts,
            demo_stats,
            client.token_tracker.summary(),
        )
        return [rank for rank in failed if rank not in success_records]

    rank_to_record = {int(record.get("selection_rank", i + 1)): record for i, record in enumerate(selected)}
    pending = [
        record
        for rank, record in sorted(rank_to_record.items())
        if rank not in done_ranks
    ]
    pending = pending[: max(args.target_count - len(success_records), 0)]
    failed_ranks = run_batch(pending)

    while len(success_records) < args.target_count and failed_ranks and current_attempts < args.max_total_attempts:
        remaining = args.target_count - len(success_records)
        candidates = [rank_to_record[rank] for rank in failed_ranks if rank in rank_to_record]
        random.Random(args.seed + current_attempts).shuffle(candidates)
        candidates = candidates[:remaining]
        print(f"Compensation round: candidates={len(candidates):,} remaining={remaining:,}")
        failed_ranks = run_batch(candidates)

    samples = materialize_main_output(output_json, success_records, args.target_count)
    write_stats(
        stats_json,
        len(selected),
        args.target_count,
        success_records,
        failure_counts,
        demo_stats,
        client.token_tracker.summary(),
    )
    print("=" * 80)
    print(f"Finished. success={len(samples):,}/{args.target_count:,}")
    print(f"main_output={output_json}")
    print(f"full_jsonl={full_jsonl}")
    print(f"failures_jsonl={failures_jsonl}")
    print(f"stats_json={stats_json}")
    print("=" * 80)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_skeletons", default=DEFAULT_SELECTED)
    parser.add_argument("--bird_train", default=DEFAULT_BIRD_TRAIN)
    parser.add_argument("--bird_data_dir", default=DEFAULT_BIRD_DATA_DIR)
    parser.add_argument("--prompt_dir", default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_json", default="synthetic_coalsql.json")
    parser.add_argument("--full_jsonl", default="synthetic_coalsql_full.jsonl")
    parser.add_argument("--failures_jsonl", default="synthetic_coalsql_failures.jsonl")
    parser.add_argument("--stats_json", default="synthetic_coalsql_stats.json")
    parser.add_argument("--target_count", type=int, default=3500)
    parser.add_argument("--max_selected", type=int, default=0)
    parser.add_argument("--api_urls", default="http://localhost:8088/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable qwen3.6 thinking mode (default: disabled for nonthinking synthesis)")
    parser.add_argument("--presence_penalty", type=float, default=1.5,
                        help="Presence penalty for qwen3.6 nonthinking mode")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling for qwen3.6")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--max_workers", type=int, default=256)
    parser.add_argument("--max_attempts_per_skeleton", type=int, default=8)
    parser.add_argument("--max_total_attempts", type=int, default=42000)
    parser.add_argument("--n_shot_instantiate", type=int, default=6)
    parser.add_argument("--n_shot_backward", type=int, default=6)
    parser.add_argument("--num_sampled_values", type=int, default=10)
    parser.add_argument("--sql_timeout", type=int, default=20)
    parser.add_argument("--sql_fetch_limit", type=int, default=30)
    parser.add_argument("--instantiate_temperature", type=float, default=1.0)
    parser.add_argument("--backward_temperature", type=float, default=1.0)
    parser.add_argument("--forward_temperature", type=float, default=0.7)
    parser.add_argument("--instantiate_max_tokens", type=int, default=4096)
    parser.add_argument("--backward_max_tokens", type=int, default=4096)
    parser.add_argument("--forward_max_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
