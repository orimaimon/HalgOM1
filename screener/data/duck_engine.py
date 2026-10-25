"""
DuckDB Parquet Query Engine.

Provides lightweight DuckDB SQL querying over Parquet files returning pandas DataFrames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd


def _is_sql(s: str) -> bool:
    """Check if a string looks like a SQL query rather than a file path."""
    s_strip = s.strip()
    if "\n" in s_strip or ";" in s_strip:
        return True
    first_word = s_strip.split(maxsplit=1)[0].upper() if s_strip else ""
    sql_keywords = {
        "SELECT",
        "WITH",
        "EXPLAIN",
        "DESCRIBE",
        "SHOW",
        "FROM",
        "PRAGMA",
        "CREATE",
        "INSERT",
        "UPDATE",
        "DELETE",
    }
    return first_word in sql_keywords


def query_parquet_df(
    sql_or_path: str | Path | Sequence[str | Path] | None = None,
    sql: str | None = None,
    params: list | tuple | dict | None = None,
    *,
    parquet_path: str | Path | Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """Execute a DuckDB query over Parquet file(s) and return the result as a pandas DataFrame.

    Args:
        sql_or_path: SQL query string, or path to a Parquet file, or sequence of paths.
        sql: SQL query string (when passed as a keyword or second positional argument).
        params: Optional parameters for parameterized DuckDB queries.
        parquet_path: Path to Parquet file or sequence of paths (keyword argument).

    Returns:
        pd.DataFrame containing query results.
    """
    target_path = parquet_path
    query_str = sql

    if sql_or_path is not None:
        if isinstance(sql_or_path, Path) or (
            isinstance(sql_or_path, (list, tuple, set)) and not isinstance(sql_or_path, (str, bytes))
        ):
            if target_path is None:
                target_path = sql_or_path
            elif query_str is None and isinstance(sql_or_path, str):
                query_str = sql_or_path
        elif isinstance(sql_or_path, str):
            if query_str is not None or target_path is not None:
                if query_str is None:
                    query_str = sql_or_path
                elif target_path is None:
                    target_path = sql_or_path
            else:
                if _is_sql(sql_or_path):
                    query_str = sql_or_path
                else:
                    target_path = sql_or_path

    if query_str is None and target_path is not None:
        query_str = "SELECT * FROM read_parquet(?)"

    if query_str is None:
        raise ValueError("Either a SQL query or a parquet_path must be provided.")

    # Convert target_path to parameter if needed when query contains '?'
    if target_path is not None and "?" in query_str and (params is None or len(params) == 0):
        if isinstance(target_path, (list, tuple, set)) and not isinstance(target_path, (str, bytes, Path)):
            formatted_paths = [str(p) for p in target_path]
            params = [formatted_paths]
        else:
            params = [str(target_path)]

    conn = duckdb.connect()
    try:
        if params is not None:
            rel = conn.execute(query_str, params)
        else:
            rel = conn.execute(query_str)
        return rel.df()
    finally:
        conn.close()
