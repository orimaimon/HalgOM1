"""Tests for screener.seasonality module."""

import time
import numpy as np
import pandas as pd
import pytest

from screener.seasonality import best_worst, monthly_seasonality


def test_constructed_series_jan_jul():
    """Test constructed series with +10% every January and -5% every July."""
    # Build monthly dates from 2020-12-31 to 2023-12-31
    # Month ends: Dec 2020 (100), Jan 2021 (110), Feb 2021 (110)... Jul 2021 (104.5)
    # We will expand month ends into daily series
    dates = pd.date_range("2020-12-31", "2023-12-31", freq="B")

    # Determine month-end targets
    m_ends = pd.date_range("2020-12-31", "2023-12-31", freq="ME")
    m_prices = {}
    m_prices[m_ends[0]] = 100.0
    for i in range(1, len(m_ends)):
        prev_p = m_prices[m_ends[i - 1]]
        m = m_ends[i].month
        if m == 1:
            m_prices[m_ends[i]] = prev_p * 1.10
        elif m == 7:
            m_prices[m_ends[i]] = prev_p * 0.95
        else:
            m_prices[m_ends[i]] = prev_p

    # Assign daily prices so month-end matches m_prices
    daily_prices = []
    for d in dates:
        # find closest month end on or after d
        m_end = pd.date_range(d, d + pd.Timedelta(days=31), freq="ME")[0]
        daily_prices.append(m_prices[m_end])

    series = pd.Series(daily_prices, index=dates)

    res = monthly_seasonality(series)

    assert len(res) == 12
    assert list(res.columns) == ["month", "avg_return", "median_return", "win_rate", "n_years"]

    # Month 1 (Jan): avg +10%, median +10%, win_rate 100%, n_years 3
    jan = res[res["month"] == 1].iloc[0]
    assert pytest.approx(jan["avg_return"], abs=1e-5) == 10.0
    assert pytest.approx(jan["median_return"], abs=1e-5) == 10.0
    assert pytest.approx(jan["win_rate"], abs=1e-5) == 100.0
    assert jan["n_years"] == 3

    # Month 7 (Jul): avg -5%, median -5%, win_rate 0%, n_years 3
    jul = res[res["month"] == 7].iloc[0]
    assert pytest.approx(jul["avg_return"], abs=1e-5) == -5.0
    assert pytest.approx(jul["median_return"], abs=1e-5) == -5.0
    assert pytest.approx(jul["win_rate"], abs=1e-5) == 0.0
    assert jul["n_years"] == 3

    # Other months: avg 0%, median 0%, win_rate 0%
    feb = res[res["month"] == 2].iloc[0]
    assert pytest.approx(feb["avg_return"], abs=1e-5) == 0.0
    assert pytest.approx(feb["win_rate"], abs=1e-5) == 0.0


def test_unsorted_input():
    """Test that unsorted DatetimeIndex input produces identical result as sorted."""
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="B")
    np.random.seed(123)
    p_values = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))
    series_sorted = pd.Series(p_values, index=dates)

    shuffled_idx = np.random.permutation(len(dates))
    series_unsorted = pd.Series(p_values[shuffled_idx], index=dates[shuffled_idx])

    res_sorted = monthly_seasonality(series_sorted)
    res_unsorted = monthly_seasonality(series_unsorted)

    pd.testing.assert_frame_equal(res_sorted, res_unsorted)


def test_nan_gaps():
    """Test input with NaN gaps in prices."""
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="B")
    np.random.seed(456)
    p_values = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))
    # Introduce NaN gaps
    p_values[10:20] = np.nan
    p_values[100:150] = np.nan

    series = pd.Series(p_values, index=dates)
    res = monthly_seasonality(series)

    assert len(res) == 12
    assert res["month"].tolist() == list(range(1, 13))


def test_fewer_than_2_months():
    """Test input spanning fewer than 2 months returns 12-row table with NaNs and n_years 0."""
    dates = pd.date_range("2023-01-01", "2023-01-20", freq="B")
    series = pd.Series(100.0, index=dates)

    res = monthly_seasonality(series)

    assert len(res) == 12
    assert res["n_years"].sum() == 0
    assert res["avg_return"].isna().all()
    assert res["median_return"].isna().all()
    assert res["win_rate"].isna().all()


def test_non_datetime_index():
    """Test non-DatetimeIndex returns empty 12-row table."""
    series = pd.Series([100.0, 105.0, 110.0], index=[1, 2, 3])
    res = monthly_seasonality(series)

    assert len(res) == 12
    assert res["n_years"].sum() == 0
    assert res["avg_return"].isna().all()

    # Non-Series
    res_list = monthly_seasonality([100.0, 105.0])  # type: ignore
    assert len(res_list) == 12
    assert res_list["n_years"].sum() == 0


def test_best_worst():
    """Test best_worst helper function."""
    dates = pd.date_range("2020-12-31", "2023-12-31", freq="B")
    m_ends = pd.date_range("2020-12-31", "2023-12-31", freq="ME")
    m_prices = {m_ends[0]: 100.0}
    for i in range(1, len(m_ends)):
        prev_p = m_prices[m_ends[i - 1]]
        m = m_ends[i].month
        if m == 1:
            m_prices[m_ends[i]] = prev_p * 1.10
        elif m == 7:
            m_prices[m_ends[i]] = prev_p * 0.95
        else:
            m_prices[m_ends[i]] = prev_p

    daily_prices = []
    for d in dates:
        m_end = pd.date_range(d, d + pd.Timedelta(days=31), freq="ME")[0]
        daily_prices.append(m_prices[m_end])

    s = pd.Series(daily_prices, index=dates)

    res = monthly_seasonality(s)
    bw = best_worst(res)

    assert bw == {"best_month": 1, "worst_month": 7}

    # Empty / NaN table test
    empty_res = monthly_seasonality(pd.Series(dtype=float))
    assert best_worst(empty_res) == {"best_month": None, "worst_month": None}

    # None input
    assert best_worst(None) == {"best_month": None, "worst_month": None}


def test_20_years_speed_benchmark():
    """Timed test: 20 years of daily data must run in under 0.3s."""
    dates = pd.date_range("2004-01-01", "2023-12-31", freq="B")
    np.random.seed(42)
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates)))), index=dates)

    t0 = time.perf_counter()
    res = monthly_seasonality(prices)
    elapsed = time.perf_counter() - t0

    assert len(res) == 12
    assert elapsed < 0.3, f"Execution took {elapsed:.4f}s, expected < 0.3s"
