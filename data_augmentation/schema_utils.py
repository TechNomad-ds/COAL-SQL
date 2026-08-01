#!/usr/bin/env python3
"""SQLite schema formatting utilities for LLM prompts."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any


DEFAULT_BIRD_DATA_DIR = "./data/bird/train"
DEFAULT_BIRD_DESCRIPTION_FILE = str(Path(__file__).parent / "resources" / "BIRD_description.json")


def quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def dump_db_json_schema(db_file: str) -> list[dict]:
    """Return schema metadata (table/column ordering and shape)."""
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("pragma foreign_keys=ON")
        cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        tables = []
        primary_keys_map = {}

        for table_name, _ in cursor.fetchall():
            cur = conn.execute(f"PRAGMA table_info({quote_sql_literal(table_name)})")
            primary_keys = {}
            for col in cur.fetchall():
                if col[5]:
                    primary_keys[col[5]] = col[1]
            primary_keys_map[table_name] = [
                primary_keys[pk_id + 1]
                for pk_id in range(len(primary_keys))
            ]

        cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        for table_name, sql in cursor.fetchall():
            if table_name == "sqlite_sequence":
                continue
            fks = conn.execute(f"PRAGMA foreign_key_list({quote_sql_literal(table_name)})").fetchall()
            foreign_keys = []
            fk_holder = [[(fk[0], fk[1]), (fk[3],), (fk[2], fk[4])] for fk in fks]
            fk_grouped = {}
            for (fk_id, sub_id), (src_fk,), (tgt_tbl, tgt_col) in fk_holder:
                if fk_id not in fk_grouped:
                    fk_grouped[fk_id] = {}
                fk_grouped[fk_id][sub_id] = [(src_fk,), (tgt_tbl, tgt_col)]
            for fk_id in range(len(fk_grouped)):
                fk = [fk_grouped[fk_id][sub_id] for sub_id in range(len(fk_grouped[fk_id]))]
                foreign_keys.append(
                    {
                        "fk": [src_fk for (src_fk,), (tgt_tbl, tgt_col) in fk],
                        "ref": fk[0][1][0],
                        "ref_key": [
                            tgt_col if tgt_col is not None else primary_keys_map[fk[0][1][0]][0]
                            for (src_fk,), (tgt_tbl, tgt_col) in fk
                        ],
                    }
                )

            columns = []
            primary_keys = {}
            cur = conn.execute(f"PRAGMA table_info({quote_sql_literal(table_name)})")
            for col in cur.fetchall():
                columns.append((col[1], col[2]))
                if col[5]:
                    primary_keys[col[5]] = col[1]
            tables.append(
                {
                    "name": table_name,
                    "sql": sql,
                    "columns": columns,
                    "primary_keys": [
                        primary_keys[pk_id + 1]
                        for pk_id in range(len(primary_keys))
                    ],
                    "foreign_keys": list(reversed(foreign_keys)),
                }
            )
        return tables
    finally:
        conn.close()


def format_db_value(value: Any) -> str:
    return "NULL" if value is None else repr(value)


def format_db_values(values: list[Any] | None) -> str:
    if not values:
        return ""
    return f"Examples: {', '.join([format_db_value(value) for value in values])}"


def format_comment(
    comment: str | None,
    values: list[Any] | None,
    enable_comment: bool = True,
    enable_values: bool = True,
) -> str | None:
    if not enable_comment:
        comment = None
    if not enable_values:
        values = None
    if comment is None and not values:
        return None
    if comment is None:
        return format_db_values(values)
    if not values:
        return comment
    return f"{comment} {format_db_values(values)}"


def build_sql(
    table_name: str,
    columns: list,
    primary_keys: list[str],
    foreign_keys: list[dict],
) -> str:
    body_parts = []
    for col_name, col_type, col_comment in columns:
        comment = f" /*{col_comment}*/" if col_comment else ""
        body_parts.append(f"`{col_name}` {col_type}{comment}")
    if primary_keys:
        body_parts.append(
            f"PRIMARY KEY ({', '.join([f'`{pk}`' for pk in primary_keys])})"
        )
    for fk in foreign_keys:
        fk_cols = ", ".join([f"`{key}`" for key in fk["fk"]])
        ref_cols = ", ".join([f"`{key}`" for key in fk["ref_key"]])
        body_parts.append(
            f"FOREIGN KEY ({fk_cols}) REFERENCES `{fk['ref']}` ({ref_cols})"
        )
    body = ",\n".join(body_parts)
    return f"CREATE TABLE `{table_name}` (\n{body}\n);"


def convert_to_table_schemas(
    db_schema: list[dict],
    col_schema: list,
    enable_comment: bool = True,
    enable_values: bool = True,
) -> list[str]:
    pks_map, fks_map, pks_orig = {}, {}, {}
    for table in db_schema:
        pks_map[table["name"].lower()] = [pk.lower() for pk in table["primary_keys"]]
        pks_orig[table["name"]] = table["primary_keys"]
        fks_map[table["name"].lower()] = table["foreign_keys"]

    schemas = {}
    for table in db_schema:
        schemas[table["name"]] = []

    for tname, col_name, col_type, comment, values in col_schema:
        if tname not in schemas:
            schemas[tname] = []
        col_comment = format_comment(comment, values, enable_comment, enable_values)
        schemas[tname].append((col_name, col_type, col_comment))

    valid_fk_map = {}
    for table in schemas:
        for pk in pks_map[table.lower()]:
            if not any(pk.lower() == col[0].lower() for col in schemas[table]):
                schemas[table].append((pk, "TEXT", None))
        valid_fk_map[table] = []
        for fk in fks_map[table.lower()]:
            ref_table = fk["ref"]
            for col in fk["fk"]:
                if not any(col.lower() == candidate[0].lower() for candidate in schemas[table]):
                    schemas[table].append((col, "TEXT", None))
            if ref_table in schemas:
                for col in fk["ref_key"]:
                    if not any(col.lower() == candidate[0].lower() for candidate in schemas[ref_table]):
                        schemas[ref_table].append((col, "TEXT", None))
            valid_fk_map[table].append(fk)

    results = []
    for table, cols in schemas.items():
        results.append(
            build_sql(
                table,
                cols,
                pks_orig[table],
                valid_fk_map[table],
            )
        )
    return results


def load_bird_descriptions(description_file: str) -> dict:
    with open(description_file, encoding="utf-8") as f:
        return json.load(f)


def get_sqlite_schema(
    db_file: str,
    enable_comment: bool = True,
    enable_values: bool = True,
    dataset: str = "BIRD",
    db_description_file: str | None = DEFAULT_BIRD_DESCRIPTION_FILE,
    num_sampled_values: int = 10,
) -> str:
    """Format a sqlite database schema as DDL for LLM prompts.

    The bundled BIRD description file is resolved from this package.
    """
    db_name = os.path.splitext(os.path.basename(db_file))[0]
    if db_description_file and dataset == "BIRD":
        descriptions = load_bird_descriptions(db_description_file).get(db_name.lower(), {})
    else:
        descriptions = {}

    db_schema = dump_db_json_schema(db_file)
    col_schema = []
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        for table in db_schema:
            tname = table["name"]
            for column_name, column_type in table["columns"]:
                comment = descriptions.get(tname.lower(), {}).get(column_name.lower(), None)
                values = []
                if enable_values:
                    cursor.execute(
                        f"SELECT DISTINCT {quote_sql_identifier(column_name)} "
                        f"FROM {quote_sql_identifier(tname)} "
                        f"WHERE {quote_sql_identifier(column_name)} IS NOT NULL "
                        f"LIMIT {int(num_sampled_values)};"
                    )
                    result = cursor.fetchall()
                    if result:
                        values = [row[0] for row in result]
                        if len(str(values[0])) > 1000:
                            values = None
                    else:
                        values = None
                col_schema.append((tname, column_name, column_type, comment, values))

    final_schemas = convert_to_table_schemas(
        db_schema,
        col_schema,
        enable_comment=enable_comment,
        enable_values=enable_values,
    )
    return "\n\n".join(final_schemas)


def bird_db_path(bird_data_dir: str, db_id: str) -> str:
    return os.path.join(bird_data_dir, "train_databases", db_id, f"{db_id}.sqlite")


class BirdSchemaProvider:
    """Cached BIRD train schema provider with DDL-style formatting."""

    def __init__(
        self,
        bird_data_dir: str = DEFAULT_BIRD_DATA_DIR,
        db_description_file: str | None = DEFAULT_BIRD_DESCRIPTION_FILE,
        enable_comment: bool = True,
        enable_values: bool = True,
        num_sampled_values: int = 10,
    ):
        self.bird_data_dir = bird_data_dir
        self.db_description_file = db_description_file
        self.enable_comment = enable_comment
        self.enable_values = enable_values
        self.num_sampled_values = num_sampled_values
        self._schema_cache: dict[str, str] = {}
        self._db_path_cache: dict[str, str] = {}
        self._cache_lock = threading.RLock()

    def get_db_path(self, db_id: str) -> str:
        if db_id in self._db_path_cache:
            return self._db_path_cache[db_id]
        with self._cache_lock:
            if db_id in self._db_path_cache:
                return self._db_path_cache[db_id]
            path = bird_db_path(self.bird_data_dir, db_id)
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing sqlite database for db_id={db_id}: {path}")
            self._db_path_cache[db_id] = path
        return self._db_path_cache[db_id]

    def get_schema(self, db_id: str) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        with self._cache_lock:
            if db_id in self._schema_cache:
                return self._schema_cache[db_id]
            self._schema_cache[db_id] = get_sqlite_schema(
                db_file=self.get_db_path(db_id),
                enable_comment=self.enable_comment,
                enable_values=self.enable_values,
                dataset="BIRD",
                db_description_file=self.db_description_file,
                num_sampled_values=self.num_sampled_values,
            )
        return self._schema_cache[db_id]
