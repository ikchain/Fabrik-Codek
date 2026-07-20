"""SQL escaping helpers for LanceDB .where() filters.

LanceDB uses DuckDB-flavored SQL and does not expose parametric filters in its
Python API as of the version pinned by this project. Until that changes, callsites
must escape user-controlled values manually.
"""

from __future__ import annotations


def escape_sql_literal(value: str) -> str:
    """Escape a string value for safe inclusion in a SQL string literal.

    Doubles single quotes per ANSI SQL. The result is meant to be wrapped in
    single quotes by the caller, e.g. f"col = '{escape_sql_literal(val)}'".
    """
    if not isinstance(value, str):
        raise TypeError(f"escape_sql_literal expects str, got {type(value).__name__}")
    return value.replace("'", "''")
