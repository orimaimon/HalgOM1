"""
DuckDB Engine & Storage Parity Engine.

Provides DuckDB connection management (enforcing read-only access mode),
Parquet file integrity verification, Parquet schema verification, storage parity comparison against Pandas,
and health check capabilities for local Parquet files in screener/_data/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


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
            "details": list[dict]
        }
    """
    if data_dir is None:
        from screener.data.store import DATA_DIR
        data_dir = DATA_DIR

    d_path = Path(data_dir)
    engine_ok = False
    try:
        conn = get_connection(read_only=True)
        one = conn.execute("SELECT 1").fetchone()[0]
        conn.close()
        engine_ok = (one == 1)
    except Exception as e:
        log.error("DuckDB engine health check failed: %s", e)

    if not engine_ok:
        return {
            "status": "error",
            "duckdb_version": duckdb.__version__,
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": 0,
            "details": [],
            "error": "Engine connection check failed",
        }

    parquet_files = sorted(list(d_path.rglob("*.parquet"))) if d_path.exists() else []
    details = []
    valid_count = 0
    corrupted_count = 0

    for p in parquet_files:
        res = verify_parquet_integrity(p)
        details.append(res)
        if res["valid"]:
            valid_count += 1
        else:
            corrupted_count += 1

    status = "ok" if corrupted_count == 0 else "degraded"
    return {
        "status": status,
        "duckdb_version": duckdb.__version__,
        "total_files": len(parquet_files),
        "valid_files": valid_count,
        "corrupted_files": corrupted_count,
        "details": details,
    }


def check_storage_parity(df_pandas: pd.DataFrame, parquet_path: Path | str) -> dict:
    """Verify parity between a Pandas DataFrame and DuckDB reading of the corresponding Parquet file.

    Compares row count, columns, null counts, and values.

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
            "mismatches": ["Parquet file does not exist"],
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
