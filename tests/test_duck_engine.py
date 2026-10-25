"""
Tests for DuckDB engine and Parquet storage integrity and parity suite (screener/data/duck_engine.py).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from screener.data import duck_engine as DE


def test_get_connection_readonly(tmp_path):
    db_file = tmp_path / "test.duckdb"

    # 1. Connection to file-backed DB is read-only
    conn = DE.get_connection(db_file, read_only=True)
    access_mode = conn.execute("SELECT current_setting('access_mode')").fetchone()[0]
    assert access_mode == "read_only"

    # 2. Writing to read-only DB raises an exception
    with pytest.raises(duckdb.Error):
        conn.execute("CREATE TABLE forbidden (x INT)")
    conn.close()

    # 3. In-memory connection test
    mem_conn = DE.get_connection(":memory:", read_only=True)
    assert mem_conn is not None
    mem_conn.close()


def test_verify_parquet_integrity_valid(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "price": [180.5, 400.2, 140.0],
        "volume": [1000000, 2000000, 1500000],
        "pe": [30.5, np.nan, 25.0],
    })
    p_path = tmp_path / "valid.parquet"
    df.to_parquet(p_path, index=False)

    res = DE.verify_parquet_integrity(p_path)
    assert res["valid"] is True
    assert res["path"] == str(p_path)
    assert res["rows"] == 3
    assert "ticker" in res["columns"]
    assert "price" in res["columns"]
    assert res["error"] is None


def test_verify_parquet_integrity_nonexistent(tmp_path):
    p_path = tmp_path / "does_not_exist.parquet"
    res = DE.verify_parquet_integrity(p_path)
    assert res["valid"] is False
    assert res["rows"] == 0
    assert "File does not exist" in res["error"]


def test_verify_parquet_integrity_corrupted(tmp_path):
    p_path = tmp_path / "corrupted.parquet"
    p_path.write_bytes(b"INVALID_PARQUET_HEADER_DATA_123456789")

    res = DE.verify_parquet_integrity(p_path)
    assert res["valid"] is False
    assert res["rows"] == 0
    assert res["error"] is not None


def test_verify_parquet_integrity_empty(tmp_path):
    df = pd.DataFrame(columns=["ticker", "price", "volume"])
    p_path = tmp_path / "empty.parquet"
    df.to_parquet(p_path, index=False)

    res = DE.verify_parquet_integrity(p_path)
    assert res["valid"] is True
    assert res["rows"] == 0
    assert set(res["columns"].keys()) == {"ticker", "price", "volume"}


def test_verify_parquet_schema(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAPL"],
        "price": [150.0],
        "sector": ["Technology"],
    })
    p_path = tmp_path / "schema.parquet"
    df.to_parquet(p_path, index=False)

    # Valid schema check
    res = DE.verify_parquet_schema(p_path, expected_columns=["ticker", "price"])
    assert res["valid"] is True
    assert res["missing_columns"] == []

    # Missing column check
    res_missing = DE.verify_parquet_schema(p_path, expected_columns=["ticker", "market_cap", "pe"])
    assert res_missing["valid"] is False
    assert res_missing["missing_columns"] == ["market_cap", "pe"]


def test_health_check(tmp_path):
    # Setup valid parquet
    df1 = pd.DataFrame({"a": [1, 2]})
    p1 = tmp_path / "f1.parquet"
    df1.to_parquet(p1, index=False)

    # Setup corrupted parquet
    p2 = tmp_path / "f2.parquet"
    p2.write_bytes(b"BAD_BYTES")

    hc = DE.health_check(data_dir=tmp_path)
    assert hc["status"] == "degraded"
    assert hc["total_files"] == 2
    assert hc["valid_files"] == 1
    assert hc["corrupted_files"] == 1
    assert hc["duckdb_version"] == duckdb.__version__


def test_check_storage_parity_matching(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "price": [180.5, 400.2],
        "volume": [1000000, 2000000],
        "pe": [30.5, np.nan],
    })
    p_path = tmp_path / "parity.parquet"
    df.to_parquet(p_path, index=False)

    res = DE.check_storage_parity(df, p_path)
    assert res["parity"] is True
    assert res["rows_pandas"] == 2
    assert res["rows_duck"] == 2
    assert res["columns_match"] is True
    assert res["mismatches"] == []


def test_check_storage_parity_mismatches(tmp_path):
    df_pandas = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "price": [180.5, 400.2, 140.0],
    })
    df_disk = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "price": [180.5, 400.2],
    })
    p_path = tmp_path / "mismatch.parquet"
    df_disk.to_parquet(p_path, index=False)

    res = DE.check_storage_parity(df_pandas, p_path)
    assert res["parity"] is False
    assert res["rows_pandas"] == 3
    assert res["rows_duck"] == 2
    assert any("Row count mismatch" in m for m in res["mismatches"])


def test_parity_across_provider_datasets(tmp_path):
    """Test Parquet parity across provider data formats (prices, boi, fred, sec, massive)."""
    # 1. Price history format
    prices_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [104.0, 105.0, 106.0, 107.0, 108.0],
        "adj_close": [104.0, 105.0, 106.0, 107.0, 108.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })
    p1 = tmp_path / "prices_sample.parquet"
    prices_df.to_parquet(p1, index=False)
    res1 = DE.check_storage_parity(prices_df, p1)
    assert res1["parity"] is True

    # 2. BOI exchange rates format
    boi_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "usd_ils": [3.70, 3.71, 3.69],
    })
    p2 = tmp_path / "boi_sample.parquet"
    boi_df.to_parquet(p2, index=False)
    res2 = DE.check_storage_parity(boi_df, p2)
    assert res2["parity"] is True
