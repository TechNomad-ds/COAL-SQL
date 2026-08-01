"""
Skeleton utility functions.

The public sql2skeleton API uses sqlglot AST rewriting. It replaces tables,
columns, literals, CTE names, and aliases with "_" while preserving SQL
operators and clause structure.
"""

import collections
import re

import sqlglot
from sqlglot import exp, parse_one
from sql_metadata import Parser


class SkeletonParseError(Exception):
    """Raised when SQL cannot be parsed into a reliable skeleton."""


_TABLE_PLACEHOLDER = "TABLE_PLACEHOLDER"
_COLUMN_PLACEHOLDER = "COLUMN_PLACEHOLDER"
_LITERAL_PLACEHOLDER = "LITERAL_PLACEHOLDER"


# ─── SQL keywords kept during fallback skeleton extraction ───────────────────
_SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "between", "like",
    "is", "null", "exists", "case", "when", "then", "else", "end",
    "join", "inner", "left", "right", "outer", "full", "cross", "on",
    "group", "by", "having", "order", "asc", "desc", "limit", "offset",
    "union", "all", "intersect", "except", "as", "distinct", "with",
    "insert", "into", "values", "update", "set", "delete", "create",
    "table", "index", "view", "drop", "alter", "add", "column",
    "count", "sum", "avg", "min", "max", "cast",
    "over", "partition", "row_number", "rank", "dense_rank", "coalesce",
    "nulls", "last", "first", "recursive", "lateral", "any", "some",
    # Operators / punctuation kept as-is
    "(", ")", ",", "=", "!=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/",
}


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments without touching quoted string literals."""
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


def _normalize_identifier(identifier: str) -> str:
    identifier = identifier.strip().strip(";")
    parts = []
    for part in identifier.split("."):
        part = part.strip()
        if len(part) >= 2 and (
            (part[0] == part[-1] == "`")
            or (part[0] == part[-1] == '"')
            or (part[0] == "[" and part[-1] == "]")
        ):
            part = part[1:-1]
        parts.append(part.lower())
    return ".".join(parts)


def _identifier_suffixes(identifier: str) -> set:
    parts = [p for p in _normalize_identifier(identifier).split(".") if p]
    return {".".join(parts[i:]) for i in range(len(parts))}


def _schema_identifier_sets(db_schema: dict) -> tuple[set, set, set]:
    table_names = set()
    column_names = {"*"}
    table_dot_column_names = set()

    raw_tables = db_schema.get("table_names_original") or db_schema.get("table_names") or []
    for table_name in raw_tables:
        table_norm = _normalize_identifier(str(table_name))
        table_names.add(table_norm)
        table_names.update(_identifier_suffixes(table_norm))
        table_dot_column_names.add(f"{table_norm}.*")

    for column_id, column_name in db_schema.get("column_names_original", []):
        column_norm = _normalize_identifier(str(column_name))
        column_names.add(column_norm)
        if isinstance(column_id, int) and 0 <= column_id < len(raw_tables):
            table_norm = _normalize_identifier(str(raw_tables[column_id]))
            table_dot_column_names.add(f"{table_norm}.{column_norm}")

    return table_names, column_names, table_dot_column_names


def _collect_parser_names(parsed_sql: Parser) -> tuple[set, set]:
    """Collect CTE/table aliases and column aliases reported by sql_metadata."""
    table_like_names = set()
    column_aliases = set()

    def is_real_table_alias(name: str) -> bool:
        normalized = _normalize_identifier(str(name))
        compact = normalized.replace(" ", "")
        return bool(normalized) and compact not in _SQL_KEYWORDS and " " not in normalized

    def is_real_column_alias(name: str) -> bool:
        normalized = _normalize_identifier(str(name))
        compact = normalized.replace(" ", "")
        return bool(normalized) and compact not in _SQL_KEYWORDS

    for attr in ("with_names",):
        for name in getattr(parsed_sql, attr, None) or []:
            if is_real_table_alias(name):
                table_like_names.add(_normalize_identifier(str(name)))

    try:
        for alias in (parsed_sql.tables_aliases or {}).keys():
            if is_real_table_alias(alias):
                table_like_names.add(_normalize_identifier(str(alias)))
    except Exception:
        pass

    try:
        for alias in parsed_sql.columns_aliases_names or []:
            if is_real_column_alias(alias):
                column_aliases.add(_normalize_identifier(str(alias)))
    except Exception:
        pass

    return table_like_names, column_aliases


def _augment_names_from_parser_columns(
    parsed_sql: Parser,
    table_names: set,
    column_names: set,
    table_dot_column_names: set,
) -> None:
    """Add parser-resolved columns/tables that may be absent from external schema."""
    try:
        for table in parsed_sql.tables or []:
            for suffix in _identifier_suffixes(str(table)):
                table_names.add(suffix)
    except Exception:
        pass

    try:
        for column in parsed_sql.columns or []:
            column_norm = _normalize_identifier(str(column))
            parts = [p for p in column_norm.split(".") if p]
            if not parts:
                continue
            column_names.add(parts[-1])
            if len(parts) >= 2:
                table_dot_column_names.add(".".join(parts[-2:]))
    except Exception:
        pass


def _augment_table_aliases_from_tokens(parsed_sql: Parser, table_like_names: set) -> None:
    """Pick up aliases after FROM/JOIN targets, including CTE aliases."""
    try:
        tokens = [token.value.strip() for token in parsed_sql.tokens]
    except Exception:
        return

    join_markers = {
        "from", "join", "inner join", "left join", "right join",
        "full join", "cross join", "left outer join", "right outer join",
        "full outer join",
    }
    clause_boundaries = {
        "on", "where", "group by", "order by", "having", "limit", "offset",
        "union", "union all", "intersect", "except", ",", ")",
    }
    for i, token in enumerate(tokens[:-2]):
        if token.lower() not in join_markers:
            continue
        alias = tokens[i + 2]
        alias_norm = _normalize_identifier(alias)
        compact = alias_norm.replace(" ", "")
        if not alias_norm or compact in _SQL_KEYWORDS or alias_norm in clause_boundaries:
            continue
        table_like_names.add(alias_norm)

    for i, token in enumerate(tokens[:-1]):
        if token != ")":
            continue
        alias = tokens[i + 1]
        alias_norm = _normalize_identifier(alias)
        compact = alias_norm.replace(" ", "")
        if not alias_norm or compact in _SQL_KEYWORDS or alias_norm in clause_boundaries:
            continue
        table_like_names.add(alias_norm)


def _strip_alias_definitions(tokens: list[str]) -> list[str]:
    """Remove `AS alias` definitions after alias tokens have been masked."""
    stripped = []
    i = 0
    clause_keywords = {
        "from", "where", "join", "inner join", "left join", "right join",
        "full join", "cross join", "group by", "order by", "having", "limit",
        "offset", "union", "intersect", "except", ")", ",",
    }
    while i < len(tokens):
        tok = tokens[i].lower()
        if tok == "as" and i + 1 < len(tokens) and tokens[i + 1] == "_":
            prev_tok = stripped[-1].lower() if stripped else ""
            next_tok = tokens[i + 2].lower() if i + 2 < len(tokens) else ""
            if prev_tok not in {"with", "recursive"} and next_tok not in {"("}:
                i += 2
                continue
        stripped.append(tokens[i])
        i += 1

    for i in range(len(stripped) - 2):
        if stripped[i].lower() == "as":
            alias = stripped[i + 1].lower()
            next_tok = stripped[i + 2].lower()
            if alias not in {"(", "_"} and next_tok in {",", "from", ")", "order by", "partition by"}:
                stripped[i + 1] = "_"

    # Some malformed parser outputs leave natural-language alias tokens after AS.
    for i, tok in enumerate(stripped):
        if tok.lower() == "as" and i + 1 < len(stripped):
            next_tok = stripped[i + 1].lower()
            if next_tok == "(":
                continue
            if next_tok not in clause_keywords and next_tok != "_":
                raise SkeletonParseError(f"unmasked alias after AS: {stripped[i + 1]}")
    return stripped


def _strip_table_alias_tokens(tokens: list[str]) -> list[str]:
    """Remove aliases in `FROM _ alias` / `JOIN _ alias` skeleton fragments."""
    stripped = []
    join_markers = {
        "from", "join", "inner join", "left join", "right join",
        "full join", "cross join", "left outer join", "right outer join",
        "full outer join",
    }
    clause_boundaries = {
        "on", "where", "group by", "order by", "having", "limit", "offset",
        "union", "union all", "intersect", "except", ",", ")",
    }
    boundaries = clause_boundaries | join_markers
    i = 0
    while i < len(tokens):
        stripped.append(tokens[i])
        marker = tokens[i].lower()
        if marker in join_markers and i + 2 < len(tokens) and tokens[i + 1] == "_":
            candidate = tokens[i + 2].lower()
            if candidate == "_":
                i += 2
            elif candidate not in boundaries and candidate != "as":
                i += 2
            else:
                i += 1
        else:
            i += 1
    return stripped


def _should_mask_identifier(
    value: str,
    table_names: set,
    column_names: set,
    table_dot_column_names: set,
    table_like_names: set,
    column_aliases: set,
) -> bool:
    value_norm = _normalize_identifier(value)
    if not value_norm:
        return False

    if value_norm in table_names or value_norm in column_names:
        return True
    if value_norm in table_dot_column_names or value_norm in table_like_names:
        return True
    if value_norm in column_aliases:
        return True

    parts = [p for p in value_norm.split(".") if p]
    if not parts:
        return False

    # table.column, db.table, db.table.column, alias.column, cte.column
    if parts[-1] in column_names or parts[-1] in column_aliases:
        return True
    if parts[-1] in table_names or parts[-1] in table_like_names:
        return True
    if len(parts) >= 2 and ".".join(parts[-2:]) in table_dot_column_names:
        return True
    if len(parts) >= 2 and parts[-2] in table_names and parts[-1] in column_names:
        return True
    if len(parts) >= 2 and parts[-2] in table_like_names:
        return True

    return False


def _post_process_skeleton(sql_skeleton: str) -> str:
    sql_skeleton = sql_skeleton.lower()

    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while "_ , _" in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    # remove clauses in WHERE keywords
    ops = ["=", "!=", "<>", ">", ">=", "<", "<="]
    for op in ops:
        sql_skeleton = sql_skeleton.replace(f"_ {op} _", "_")

    while "where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    # double check for order by
    split_skeleton = sql_skeleton.split(" ")
    for i in range(2, len(split_skeleton)):
        if split_skeleton[i - 2] == "order" and split_skeleton[i - 1] == "by" and split_skeleton[i] != "_":
            split_skeleton[i] = "_"
    return " ".join(split_skeleton).strip()


def _validate_skeleton_quality(sql_skeleton: str) -> None:
    """Reject skeletons that still contain unmasked identifiers or aliases."""
    allowed = _SQL_KEYWORDS | {
        "_", "order", "by", "not", "null", "not null", "order by",
        "inner join", "left join", "right join", "full join", "cross join",
        "left outer join", "right outer join", "full outer join",
        "||", "%", ".", ";",
    }
    operators = {
        "(", ")", ",", "=", "!=", "<>", "<", ">", "<=", ">=",
        "+", "-", "*", "/", "||", "%",
    }
    tokens = sql_skeleton.split()
    for i, token in enumerate(tokens):
        if token in allowed or token in operators:
            continue
        if isFloat(token) or isNegativeInt(token):
            continue
        # SQL function names are meaningful structure; aliases are not.
        if re.match(r"^[a-z_][a-z0-9_]*$", token) and i + 1 < len(tokens) and tokens[i + 1] == "(":
            continue
        if re.search(r"[a-zA-Z]", token):
            raise SkeletonParseError(f"unmasked identifier remains in skeleton: {token}")


def _strip_sqlglot_aliases(parsed_sql) -> None:
    """Remove aliases that leak natural-language or schema-specific names."""
    for alias in list(parsed_sql.find_all(exp.Alias)):
        alias.replace(alias.this)

    for cte in parsed_sql.find_all(exp.CTE):
        cte.set(
            "alias",
            exp.TableAlias(this=exp.to_identifier(_TABLE_PLACEHOLDER)),
        )

    for subquery in parsed_sql.find_all(exp.Subquery):
        subquery.set("alias", None)


def _format_sqlglot_skeleton(parsed_sql, ignore_bracket: bool = False) -> str:
    skeleton = parsed_sql.sql(dialect="sqlite")
    if ignore_bracket:
        return (
            skeleton.replace(f"-{_LITERAL_PLACEHOLDER}", "_")
            .replace(_COLUMN_PLACEHOLDER, "_")
            .replace(_TABLE_PLACEHOLDER, "_")
            .replace(_LITERAL_PLACEHOLDER, "_")
        )
    return (
        skeleton.replace(f"-{_LITERAL_PLACEHOLDER}", "_")
        .replace(_COLUMN_PLACEHOLDER, "_")
        .replace(_TABLE_PLACEHOLDER, "_")
        .replace(_LITERAL_PLACEHOLDER, "_")
    )


def _sqlglot_skeleton(sql: str, ignore_bracket: bool = False) -> str:
    parsed_sql = parse_one(sql, dialect="sqlite")
    _strip_sqlglot_aliases(parsed_sql)

    for table in list(parsed_sql.find_all(exp.Table)):
        table.replace(sqlglot.to_table(_TABLE_PLACEHOLDER))
    for column in list(parsed_sql.find_all(exp.Column)):
        column.replace(sqlglot.to_column(_COLUMN_PLACEHOLDER))
    for literal in list(parsed_sql.find_all(exp.Literal)):
        literal.replace(sqlglot.to_column(_LITERAL_PLACEHOLDER))

    skeleton = _format_sqlglot_skeleton(parsed_sql, ignore_bracket=ignore_bracket)
    if not skeleton:
        raise SkeletonParseError("empty skeleton")
    return skeleton


def _fallback_skeleton(sql: str) -> str:
    """Regex-based fallback skeleton for SQL that sql_metadata can't parse.
    
    Keeps SQL keywords and replaces identifiers/values with '_', then applies
    the same post-processing as sql2skeleton.
    """
    # Basic normalisation
    sql = _strip_sql_comments(sql).strip().rstrip(";")
    sql = sql.replace('"', "'")

    # Tokenise: split on whitespace and punctuation while keeping punctuation
    tokens = re.findall(
        r"""'[^']*'|"[^"]*"|`[^`]*`|\[[^\]]*\]|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|\d+(?:\.\d+)?|<>|!=|<=|>=|[(),=<>+\-*/]""",
        sql,
        re.DOTALL,
    )

    new_tokens = []
    for tok in tokens:
        low = tok.lower()
        if low in _SQL_KEYWORDS:
            new_tokens.append(low)
        elif tok in ("(", ")", ",", "=", "!=", "<>", "<", ">", "<=", ">=",
                      "+", "-", "*", "/"):
            new_tokens.append(tok)
        else:
            new_tokens.append("_")

    return _post_process_skeleton(" ".join(new_tokens))


def sql_normalization(sql):
    sql = _strip_sql_comments(sql).strip()

    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])
        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*'
        )
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")
        return s

    def sql_split(s):
        while "  " in s:
            s = s.replace("  ", " ")
        s = s.strip()
        i = 0
        toks = []
        while i < len(s):
            tok = ""
            if s[i] == "'":
                tok = tok + s[i]
                i += 1
                while i < len(s) and s[i] != "'":
                    tok = tok + s[i]
                    i += 1
                if i < len(s):
                    tok = tok + s[i]
                    i += 1
            else:
                while i < len(s) and s[i] != " ":
                    tok = tok + s[i]
                    i += 1
                while i < len(s) and s[i] == " ":
                    i += 1
            toks.append(tok)
        return toks

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
        table_names = []
        for tok in sql_split(s):
            if '.' in tok:
                table_names.append(tok.split('.')[0])
        for table_name in table_names:
            if table_name in tables_aliases.keys():
                new_tables_aliases[table_name] = tables_aliases[table_name]
        tables_aliases = new_tables_aliases

        new_s = []
        pre_tok = ""
        for tok in sql_split(s):
            if tok in tables_aliases.keys():
                if pre_tok == 'as':
                    new_s = new_s[:-1]
                elif pre_tok != tables_aliases[tok]:
                    new_s.append(tables_aliases[tok])
            elif '.' in tok:
                split_toks = tok.split('.')
                for i in range(len(split_toks)):
                    if len(split_toks[i]) > 2 and split_toks[i][0] == "'" and split_toks[i][-1] == "'":
                        split_toks[i] = split_toks[i].replace("'", "")
                        split_toks[i] = split_toks[i].lower()
                    if split_toks[i] in tables_aliases.keys():
                        split_toks[i] = tables_aliases[split_toks[i]]
                new_s.append('.'.join(split_toks))
            else:
                new_s.append(tok)
            pre_tok = tok

        # remove as
        s = new_s
        new_s = []
        for i in range(len(s)):
            if s[i] == "as":
                continue
            if i > 0 and s[i - 1] == "as":
                continue
            new_s.append(s[i])
        new_s = ' '.join(new_s)

        return new_s

    processing_func = lambda x: remove_table_alias(
        add_asc(lower(white_space_fix(double2single(remove_semicolon(x)))))
    )

    return processing_func(sql.strip())


def sql2skeleton(sql: str, db_schema: dict | None = None) -> str:
    """Convert a SQL query to its skeleton form with sqlglot AST rewriting.

    Args:
        sql: The SQL query string.
        db_schema: Kept for backward compatibility. sqlglot-based extraction
                   does not need external schema metadata.

    Returns:
        The skeleton string with tables, columns, and values masked as "_".

    Raises:
        SkeletonParseError: If sqlglot cannot parse the SQL.
    """
    try:
        sql_clean = _strip_sql_comments(sql).strip().rstrip(";")
        return _sqlglot_skeleton(sql_clean)
    except Exception as exc:
        raise SkeletonParseError(f"sqlglot parsing failed: {exc}") from exc


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True


def jaccard_similarity(skeleton1: str, skeleton2: str) -> float:
    """Compute Jaccard similarity between two skeleton strings."""
    tokens1 = skeleton1.strip().split(" ")
    tokens2 = skeleton2.strip().split(" ")

    def list_to_dict(tokens):
        token_dict = collections.defaultdict(int)
        for t in tokens:
            token_dict[t] += 1
        return token_dict

    token_dict1 = list_to_dict(tokens1)
    token_dict2 = list_to_dict(tokens2)

    intersection = 0
    for t in token_dict1:
        if t in token_dict2:
            intersection += min(token_dict1[t], token_dict2[t])
    union = (len(tokens1) + len(tokens2)) - intersection
    return float(intersection) / union
