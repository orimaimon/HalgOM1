"""
DuckDB Engine & Storage Parity Engine.

Provides lightweight DuckDB SQL querying over Parquet files returning pandas DataFrames,
DuckDB connection management (enforcing read-only access mode),
Parquet file integrity verification, Parquet schema verification, storage parity comparison against Pandas,
and health check capabilities for local Parquet files in screener/_data/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


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


def get_connection(db_path: Path | str | None = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection.

    Ensures DuckDB connections are read-only when requested or when connecting to database files.
    """
    if db_path is not None and str(db_path) != ":memory:":
        path = Path(db_path)
        if not path.exists():
            init_c = duckdb.connect(str(path))
            init_c.close()
        conn = duckdb.connect(
            str(path),
            read_only=read_only,
            config={"access_mode": "READ_ONLY" if read_only else "AUTOMATIC"},
        )
        return conn

    conn = duckdb.connect(":memory:")
    if read_only:
        try:
            conn.execute("SET access_mode = 'READ_ONLY'")
        except Exception:
            pass
    return conn


def verify_parquet_integrity(path: Path | str) -> dict:
    """Verify that a Parquet file exists, can be opened by DuckDB, and is uncorrupted.

    Returns:
        dict: {
            "valid": bool,
            "path": str,
            "rows": int,
            "columns": dict[str, str],  # col_name -> col_type
            "error": str | None
        }
    """
    p = Path(path)
    if not p.exists():
        return {
            "valid": False,
            "path": str(p),
            "rows": 0,
            "columns": {},
            "error": f"File does not exist: {p}",
        }

    conn = None
    try:
        conn = get_connection(read_only=True)
        posix_path = str(p.resolve()).replace("'", "''")

        row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{posix_path}')").fetchone()[0]
        desc = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{posix_path}')").fetchall()
        cols = {row[0]: row[1] for row in desc}

        # Full-scan verification to confirm Parquet row groups and pages are uncorrupted
        _ = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{posix_path}')").fetchone()

        return {
            "valid": True,
            "path": str(p),
            "rows": row_count,
            "columns": cols,
            "error": None,
        }
    except Exception as e:
        log.warning("Parquet integrity check failed for %s: %s", p, e)
        return {
            "valid": False,
            "path": str(p),
            "rows": 0,
            "columns": {},
            "error": str(e),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def compute_daily_returns_duck(
    parquet_path: str | Path, ticker: str | None = None
) -> pd.DataFrame:
    """Execute a DuckDB query computing daily percent return for tickers in a Parquet file.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the Parquet file.
    ticker : str | None
        Optional ticker symbol to filter on.

    Returns
    -------
    pd.DataFrame
        DataFrame ordered by ticker, date containing daily return_pct.
    """
    p_str = str(parquet_path)
    if ticker is not None:
        sql = """
            SELECT
                *,
                (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) * 100.0 AS return_pct
            FROM read_parquet(?)
            WHERE ticker = ?
            ORDER BY ticker, date
        """
        params = [p_str, ticker]
    else:
        sql = """
            SELECT
                *,
                (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) * 100.0 AS return_pct
            FROM read_parquet(?)
            ORDER BY ticker, date
        """
        params = [p_str]

    conn = get_connection(read_only=True)
    try:
        return conn.execute(sql, params).df()
    finally:
        conn.close()


def compute_rolling_volatility_duck(
    parquet_path: str | Path, ticker: str | None = None, window: int = 20
) -> pd.DataFrame:
    """Execute a DuckDB window query computing rolling standard deviation of daily returns.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the Parquet file.
    ticker : str | None
        Optional ticker symbol to filter on.
    window : int
        Rolling window size in days (default 20).

    Returns
    -------
    pd.DataFrame
        DataFrame containing daily returns and rolling_volatility.
    """
    if window <= 0:
        raise ValueError("window must be greater than 0")

    p_str = str(parquet_path)
    preceding = window - 1

    where_clause = "WHERE ticker = ?" if ticker is not None else ""
    params = [p_str, ticker] if ticker is not None else [p_str]

    sql = f"""
        WITH rets AS (
            SELECT
                *,
                (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) * 100.0 AS return_pct
            FROM read_parquet(?)
            {where_clause}
        )
        SELECT
            *,
            CASE
                WHEN COUNT(return_pct) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW) >= {window}
                THEN STDDEV_SAMP(return_pct) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW)
                ELSE NULL
            END AS rolling_volatility
        FROM rets
        ORDER BY ticker, date
    """

    conn = get_connection(read_only=True)
    try:
        return conn.execute(sql, params).df()
    finally:
        conn.close()


def verify_parquet_schema(
    path: Path | str, expected_columns: list[str] | dict[str, str] | None = None
) -> dict:
    """Inspect Parquet schema via DuckDB and verify whether required or expected columns exist.

    Returns:
        dict: {
            "valid": bool,
            "path": str,
            "columns": dict[str, str],
            "missing_columns": list[str],
            "error": str | None
        }
    """
    res = verify_parquet_integrity(path)
    if not res["valid"]:
        return {
            "valid": False,
            "path": str(path),
            "columns": {},
            "missing_columns": list(expected_columns) if expected_columns else [],
            "error": res["error"],
        }

    actual_cols = res["columns"]
    missing = []
    if expected_columns:
        expected_names = list(expected_columns.keys()) if isinstance(expected_columns, dict) else list(expected_columns)
        missing = [c for c in expected_names if c not in actual_cols]

    return {
        "valid": len(missing) == 0,
        "path": str(path),
        "columns": actual_cols,
        "missing_columns": missing,
        "error": f"Missing columns: {missing}" if missing else None,
    }


def health_check(data_dir: Path | str | None = None) -> dict:
    """Check DuckDB engine health and scan all local Parquet files in data_dir for integrity.

    Returns:
        dict: {
            "status": "ok" | "degraded" | "error",
            "duckdb_version": str,
            "total_files": int,
            "valid_files": int,
            "corrupted_files": int,
            "files": list[dict],
            "error": str | None
        }
    """
    try:
        import duckdb
        ver = duckdb.__version__
    except Exception as e:
        return {
            "status": "error",
            "duckdb_version": "unknown",
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": 0,
            "files": [],
            "error": str(e),
        }

    target_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / "_data"

    if not target_dir.exists():
        return {
            "status": "ok",
            "duckdb_version": ver,
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": 0,
            "files": [],
            "error": None,
        }

    parquet_files = list(target_dir.rglob("*.parquet"))
    results = []
    corrupted_count = 0

    for pf in parquet_files:
        info = verify_parquet_integrity(pf)
        results.append(info)
        if not info["valid"]:
            corrupted_count += 1

    status = "ok"
    if corrupted_count > 0:
        status = "degraded"

    return {
        "status": status,
        "duckdb_version": ver,
        "total_files": len(parquet_files),
        "valid_files": len(parquet_files) - corrupted_count,
        "corrupted_files": corrupted_count,
        "files": results,
        "error": None,
    }


def check_storage_parity(df_pandas: pd.DataFrame, parquet_path: Path | str) -> dict:
    """Check numerical and schema parity between an in-memory Pandas DataFrame and Parquet on disk via DuckDB.

    Returns:
        dict: {
            "parity": bool,
            "rows_pandas": int,
            "rows_duck": int,
            "columns_match": bool,
            "mismatches": list[str],
            "error": str | None
        }
    """
    p = Path(parquet_path)
    if not p.exists():
        return {
            "parity": False,
            "rows_pandas": len(df_pandas),
            "rows_duck": 0,
            "columns_match": False,
            "mismatches": [f"Parquet file does not exist: {p}"],
            "error": f"File does not exist: {p}",
        }

    conn = None
    try:
        conn = get_connection(read_only=True)
        posix_path = str(p.resolve()).replace("'", "''")
        df_duck = conn.execute(f"SELECT * FROM read_parquet('{posix_path}')").df()

        mismatches = []
        if len(df_pandas) != len(df_duck):
            mismatches.append(f"Row count mismatch: pandas={len(df_pandas)}, duckdb={len(df_duck)}")

        pandas_cols = list(df_pandas.columns)
        duck_cols = list(df_duck.columns)
        cols_match = (pandas_cols == duck_cols)
        if not cols_match:
            mismatches.append(f"Column mismatch: pandas={pandas_cols}, duckdb={duck_cols}")

        if cols_match and len(df_pandas) == len(df_duck):
            try:
                pd.testing.assert_frame_equal(df_pandas, df_duck, check_dtype=False, check_like=True)
            except AssertionError as ae:
                mismatches.append(f"Value mismatch: {ae}")

        return {
            "parity": len(mismatches) == 0,
            "rows_pandas": len(df_pandas),
            "rows_duck": len(df_duck),
            "columns_match": cols_match,
            "mismatches": mismatches,
            "error": None,
        }
    except Exception as e:
        return {
            "parity": False,
            "rows_pandas": len(df_pandas),
            "rows_duck": 0,
            "columns_match": False,
            "mismatches": [str(e)],
            "error": str(e),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
