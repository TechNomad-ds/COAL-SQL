#!/usr/bin/env python3
"""SQLite execution utilities for skeleton-guided synthesis."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT = 20
DEFAULT_FETCH_LIMIT = 30
MAX_RESULT_LEN = 300


@dataclass
class SQLExecutionResult:
    """Result of a SQLite query execution."""

    success: bool
    result: Any = None
    error: str | None = None
    timed_out: bool = False

    @property
    def is_non_empty(self) -> bool:
        return self.success and bool(self.result)

    @property
    def result_str(self) -> str:
        if self.error:
            return self.error
        text = str(self.result)
        if len(text) > MAX_RESULT_LEN:
            text = text[:MAX_RESULT_LEN] + "..."
        return text


@dataclass
class SQLValidationResult:
    """Structured validity decision for synthesis filtering."""

    ok: bool
    reason: str
    execution: SQLExecutionResult

    @property
    def rows(self) -> Any:
        return self.execution.result

    @property
    def result_str(self) -> str:
        return self.execution.result_str


def resolve_bird_db_path(db_root: str, db_id: str, mode: str = "train") -> Path:
    """Resolve BIRD-style sqlite paths under train/dev database directories."""
    return Path(db_root) / f"{mode}_databases" / db_id / f"{db_id}.sqlite"


def execute_sql(
    db_path: str | Path,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> SQLExecutionResult:
    """Execute a SQL query with evol-sql-style thread timeout protection."""
    db_path = Path(db_path)
    if not db_path.exists():
        return SQLExecutionResult(success=False, error=f"Database not found: {db_path}")
    if not sql or not sql.strip():
        return SQLExecutionResult(success=False, error="SQL is empty")

    query_result = {"rows": None, "error": None}

    def _run_query(connection: sqlite3.Connection, sql_query: str) -> None:
        try:
            cursor = connection.cursor()
            cursor.execute(sql_query)
            query_result["rows"] = cursor.fetchmany(fetch_limit)
        except Exception as exc:
            query_result["error"] = exc

    try:
        conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")

        thread = threading.Thread(target=_run_query, args=(conn, sql))
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            conn.interrupt()
            thread.join()
            conn.close()
            return SQLExecutionResult(
                success=False,
                error="Query execution timed out",
                timed_out=True,
            )

        conn.close()

        if query_result["error"]:
            return SQLExecutionResult(success=False, error=str(query_result["error"]))

        return SQLExecutionResult(success=True, result=query_result["rows"])
    except Exception as exc:
        return SQLExecutionResult(success=False, error=str(exc))


def execute_sql_with_timeout(
    db_path: str | Path,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> SQLExecutionResult:
    """Alias kept for compatibility with evol-sql naming."""
    return execute_sql(db_path, sql, timeout=timeout)


def validate_sql_non_empty(
    db_path: str | Path,
    sql: str,
    timeout: int = DEFAULT_TIMEOUT,
    fetch_limit: int = DEFAULT_FETCH_LIMIT,
) -> SQLValidationResult:
    """Validate that SQL executes successfully and returns at least one row."""
    if not sql or not sql.strip():
        execution = SQLExecutionResult(success=False, error="SQL is empty")
        return SQLValidationResult(ok=False, reason="sql_empty", execution=execution)

    execution = execute_sql(
        db_path=db_path,
        sql=sql,
        timeout=timeout,
        fetch_limit=fetch_limit,
    )
    if execution.success and execution.is_non_empty:
        return SQLValidationResult(ok=True, reason="ok", execution=execution)
    if execution.success:
        return SQLValidationResult(ok=False, reason="empty_result", execution=execution)
    if execution.timed_out:
        return SQLValidationResult(ok=False, reason="timeout", execution=execution)
    if execution.error and execution.error.startswith("Database not found:"):
        return SQLValidationResult(ok=False, reason="db_not_found", execution=execution)
    return SQLValidationResult(ok=False, reason="execution_error", execution=execution)
