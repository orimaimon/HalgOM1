from pathlib import Path

import duckdb
import pandas as pd
import pytest

import numpy as np

from screener.data.duck_engine import (
    compute_daily_returns_duck,
    compute_rolling_volatility_duck,
    query_parquet_df,
)


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


@pytest.fixture
def price_history_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "price_history.parquet"
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30)
    df1 = pd.DataFrame({"ticker": "AAPL", "date": dates, "close": np.random.uniform(100, 200, 30)})
    df2 = pd.DataFrame({"ticker": "MSFT", "date": dates, "close": np.random.uniform(200, 300, 30)})
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_parquet(p, index=False)
    return p


def test_compute_daily_returns_duck_multi_ticker(price_history_parquet: Path) -> None:
    res = compute_daily_returns_duck(price_history_parquet)
    assert isinstance(res, pd.DataFrame)
    assert "return_pct" in res.columns
    assert set(res["ticker"].unique()) == {"AAPL", "MSFT"}

    # Compute pandas expected
    raw_df = pd.read_parquet(price_history_parquet)
    raw_df["expected_return"] = raw_df.groupby("ticker")["close"].pct_change() * 100.0

    pd.testing.assert_series_equal(
        res["return_pct"], raw_df["expected_return"], check_names=False
    )


def test_compute_daily_returns_duck_single_ticker(price_history_parquet: Path) -> None:
    res = compute_daily_returns_duck(price_history_parquet, ticker="AAPL")
    assert isinstance(res, pd.DataFrame)
    assert list(res["ticker"].unique()) == ["AAPL"]
    assert len(res) == 30

    raw_df = pd.read_parquet(price_history_parquet)
    aapl_df = raw_df[raw_df["ticker"] == "AAPL"].copy()
    aapl_df["expected_return"] = aapl_df["close"].pct_change() * 100.0

    pd.testing.assert_series_equal(
        res["return_pct"], aapl_df["expected_return"].reset_index(drop=True), check_names=False
    )


def test_compute_daily_returns_duck_nonexistent_ticker(price_history_parquet: Path) -> None:
    res = compute_daily_returns_duck(price_history_parquet, ticker="NONEXISTENT")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0


def test_compute_rolling_volatility_duck_multi_ticker(price_history_parquet: Path) -> None:
    res = compute_rolling_volatility_duck(price_history_parquet, window=20)
    assert isinstance(res, pd.DataFrame)
    assert "rolling_volatility" in res.columns

    raw_df = pd.read_parquet(price_history_parquet)
    raw_df["return_pct"] = raw_df.groupby("ticker")["close"].pct_change() * 100.0
    raw_df["expected_vol"] = (
        raw_df.groupby("ticker")["return_pct"].rolling(20).std().reset_index(level=0, drop=True)
    )

    pd.testing.assert_series_equal(
        res["rolling_volatility"], raw_df["expected_vol"], check_names=False
    )


def test_compute_rolling_volatility_duck_single_ticker_custom_window(
    price_history_parquet: Path,
) -> None:
    res = compute_rolling_volatility_duck(price_history_parquet, ticker="MSFT", window=5)
    assert isinstance(res, pd.DataFrame)
    assert list(res["ticker"].unique()) == ["MSFT"]
    assert len(res) == 30

    raw_df = pd.read_parquet(price_history_parquet)
    msft_df = raw_df[raw_df["ticker"] == "MSFT"].copy()
    msft_df["return_pct"] = msft_df["close"].pct_change() * 100.0
    msft_df["expected_vol"] = msft_df["return_pct"].rolling(5).std()

    pd.testing.assert_series_equal(
        res["rolling_volatility"],
        msft_df["expected_vol"].reset_index(drop=True),
        check_names=False,
    )


def test_compute_rolling_volatility_duck_invalid_window(price_history_parquet: Path) -> None:
    with pytest.raises(ValueError, match="window must be greater than 0"):
        compute_rolling_volatility_duck(price_history_parquet, window=0)
