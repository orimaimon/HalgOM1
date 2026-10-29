"""Unit tests for screener.portfolio_benchmark module."""

import time
import numpy as np
import pandas as pd
import pytest

from screener.portfolio_benchmark import compare, rolling_beta


def test_portfolio_equals_benchmark():
    """When portfolio returns equal benchmark returns:

    beta=1, alpha_annual~0, tracking_error~0, correlation=1, captures=1.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    b_returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    p_returns = b_returns.copy()

    res = compare(p_returns, b_returns, risk_free_annual=0.02)

    assert res["n_obs"] == 100
    assert pytest.approx(res["beta"], abs=1e-6) == 1.0
    assert pytest.approx(res["alpha_annual"], abs=1e-6) == 0.0
    assert pytest.approx(res["correlation"], abs=1e-6) == 1.0
    assert pytest.approx(res["tracking_error"], abs=1e-6) == 0.0
    assert pytest.approx(res["up_capture"], abs=1e-6) == 1.0
    assert pytest.approx(res["down_capture"], abs=1e-6) == 1.0


def test_portfolio_leveraged_2x():
    """When portfolio returns equal 2 * benchmark returns:

    beta=2, correlation=1, up_capture=2, down_capture=2.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    # Generate mixture with both positive and negative returns
    b_returns = pd.Series(
        np.array([0.01 if i % 2 == 0 else -0.01 for i in range(100)]), index=dates
    )
    p_returns = 2.0 * b_returns

    res = compare(p_returns, b_returns)

    assert res["n_obs"] == 100
    assert pytest.approx(res["beta"], abs=1e-6) == 2.0
    assert pytest.approx(res["correlation"], abs=1e-6) == 1.0
    assert pytest.approx(res["up_capture"], abs=1e-6) == 2.0
    assert pytest.approx(res["down_capture"], abs=1e-6) == 2.0


def test_uncorrelated_returns():
    """Uncorrelated returns result in correlation ~ 0 and beta ~ 0."""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=500)
    p_returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
    b_returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)

    res = compare(p_returns, b_returns)

    assert res["n_obs"] == 500
    assert abs(res["beta"]) < 0.2
    assert abs(res["correlation"]) < 0.2


def test_fewer_than_20_obs_returns_none():
    """Fewer than 20 observations must return None for computed statistics."""
    dates = pd.date_range("2023-01-01", periods=19)
    p_returns = pd.Series([0.01] * 19, index=dates)
    b_returns = pd.Series([0.01] * 19, index=dates)

    res = compare(p_returns, b_returns)

    assert res["n_obs"] == 19
    assert res["beta"] is None
    assert res["alpha_annual"] is None
    assert res["correlation"] is None
    assert res["tracking_error"] is None
    assert res["up_capture"] is None
    assert res["down_capture"] is None


def test_misaligned_indexes_and_nans_dropped():
    """Date indexes are aligned and NaN values are dropped before calculation."""
    np.random.seed(42)
    p_dates = pd.date_range("2023-01-01", periods=50)
    b_dates = pd.date_range("2023-01-10", periods=50)

    p_returns = pd.Series(np.random.normal(0.001, 0.01, 50), index=p_dates)
    b_returns = pd.Series(np.random.normal(0.001, 0.01, 50), index=b_dates)

    # Insert NaNs in p_returns
    p_returns.iloc[15] = np.nan

    res = compare(p_returns, b_returns)

    # Overlapping dates: Jan 10 to Jan 19 (41 days overlap minus 1 NaN = 40 days)
    assert res["n_obs"] == 40
    assert res["beta"] is not None


def test_constant_benchmark_zero_variance():
    """Zero variance in benchmark returns returns None for beta and alpha."""
    dates = pd.date_range("2023-01-01", periods=30)
    p_returns = pd.Series(np.random.normal(0.01, 0.02, 30), index=dates)
    b_returns = pd.Series([0.005] * 30, index=dates)  # zero variance

    res = compare(p_returns, b_returns)

    assert res["n_obs"] == 30
    assert res["beta"] is None
    assert res["alpha_annual"] is None


def test_no_up_or_down_days():
    """If benchmark has no up or no down days, corresponding capture ratio is None."""
    dates = pd.date_range("2023-01-01", periods=30)
    p_returns = pd.Series([0.01] * 30, index=dates)
    b_returns = pd.Series([0.01] * 30, index=dates)  # Only up days (b > 0)

    res = compare(p_returns, b_returns)

    assert res["n_obs"] == 30
    assert res["up_capture"] == 1.0
    assert res["down_capture"] is None


def test_rolling_beta():
    """rolling_beta computes moving window beta and returns NaNs before window fills."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    b_returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    p_returns = 1.5 * b_returns

    r_beta = rolling_beta(p_returns, b_returns, window=60)

    assert len(r_beta) == 100
    assert r_beta.iloc[:59].isna().all()
    assert pytest.approx(r_beta.iloc[59], abs=1e-6) == 1.5
    assert pytest.approx(r_beta.iloc[-1], abs=1e-6) == 1.5


def test_execution_speed_benchmark():
    """2,500 daily points must run compare and rolling_beta in under 0.5 seconds."""
    np.random.seed(99)
    dates = pd.date_range("2010-01-01", periods=2500)
    b_returns = pd.Series(np.random.normal(0.0005, 0.01, 2500), index=dates)
    p_returns = 1.2 * b_returns + pd.Series(
        np.random.normal(0.0, 0.005, 2500), index=dates
    )

    t0 = time.perf_counter()
    res = compare(p_returns, b_returns, risk_free_annual=0.03)
    r_beta = rolling_beta(p_returns, b_returns, window=60)
    elapsed = time.perf_counter() - t0

    assert res["n_obs"] == 2500
    assert len(r_beta) == 2500
    assert elapsed < 0.5, f"Execution took {elapsed:.4f}s, expected < 0.5s"
