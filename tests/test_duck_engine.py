from pathlib import Path

import duckdb
import pandas as pd
import pytest

from screener.data.duck_engine import query_parquet_df


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "sample.parquet"
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "price": [150.0, 280.0, 120.0, 130.0],
            "volume": [1000, 2000, 1500, 1800],
        }
    )
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def sample_parquet_2(tmp_path: Path) -> Path:
    p = tmp_path / "sample_2.parquet"
    df = pd.DataFrame(
        {
            "ticker": ["NVDA", "META"],
            "price": [400.0, 300.0],
            "volume": [2500, 2200],
        }
    )
    df.to_parquet(p, index=False)
    return p


def test_query_parquet_df_single_file_str(sample_parquet: Path) -> None:
    res = query_parquet_df(str(sample_parquet))
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4
    assert set(res.columns) == {"ticker", "price", "volume"}


def test_query_parquet_df_single_file_path(sample_parquet: Path) -> None:
    res = query_parquet_df(sample_parquet)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4


def test_query_parquet_df_explicit_parquet_path(sample_parquet: Path) -> None:
    res = query_parquet_df(parquet_path=sample_parquet)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4


def test_query_parquet_df_multiple_files(sample_parquet: Path, sample_parquet_2: Path) -> None:
    res = query_parquet_df([sample_parquet, sample_parquet_2])
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 6
    assert set(res["ticker"]) == {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"}


def test_query_parquet_df_custom_sql(sample_parquet: Path) -> None:
    sql = "SELECT ticker, price FROM read_parquet(?) WHERE price > 140 ORDER BY price DESC"
    res = query_parquet_df(sample_parquet, sql=sql)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert list(res["ticker"]) == ["MSFT", "AAPL"]


def test_query_parquet_df_parameterized(sample_parquet: Path) -> None:
    sql = "SELECT ticker, price FROM read_parquet(?) WHERE price > ?"
    res = query_parquet_df(sql=sql, params=[str(sample_parquet), 200.0])
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert res["ticker"].iloc[0] == "MSFT"


def test_query_parquet_df_direct_sql(sample_parquet: Path) -> None:
    sql = f"SELECT count(*) AS total_count FROM read_parquet('{sample_parquet.as_posix()}')"
    res = query_parquet_df(sql)
    assert isinstance(res, pd.DataFrame)
    assert res["total_count"].iloc[0] == 4


def test_query_parquet_df_empty_result(sample_parquet: Path) -> None:
    sql = "SELECT * FROM read_parquet(?) WHERE price > 1000"
    res = query_parquet_df(sample_parquet, sql=sql)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert set(res.columns) == {"ticker", "price", "volume"}


def test_query_parquet_df_missing_arguments() -> None:
    with pytest.raises(ValueError, match="Either a SQL query or a parquet_path must be provided"):
        query_parquet_df()


def test_query_parquet_df_missing_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.parquet"
    with pytest.raises((duckdb.Error, Exception)):
        query_parquet_df(non_existent)


def test_query_parquet_df_invalid_sql(sample_parquet: Path) -> None:
    sql = "SELECT INVALID SYNTAX FROM read_parquet(?)"
    with pytest.raises((duckdb.Error, Exception)):
        query_parquet_df(sample_parquet, sql=sql)
