"""
Unit tests for drawdown episode statistics module (screener/drawdown_stats.py).
"""
import time
import numpy as np
import pandas as pd
import pytest

from screener.drawdown_stats import current_drawdown, drawdown_episodes


def test_known_episodes_recovered_and_unrecovered():
    """Test hand-built series with two known episodes: one recovered and one unrecovered."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    # Peak 1 = 100.0 on 2025-01-01 (idx 0)
    # Trough 1 = 80.0 on 2025-01-03 (idx 2) -> Depth = -20.0%
    # Recovery 1 = 100.0 on 2025-01-05 (idx 4) -> days_to_trough = 2, days_to_recover = 4
    # Peak 2 = 120.0 on 2025-01-06 (idx 5)
    # Trough 2 = 90.0 on 2025-01-08 (idx 7) -> Depth = (90 - 120)/120 * 100 = -25.0%
    # Unrecovered through 2025-01-10 (idx 9) -> recovery_date = NaT, days_to_recover = NaN
    prices = [100.0, 90.0, 80.0, 95.0, 100.0, 120.0, 105.0, 90.0, 100.0, 110.0]
    s = pd.Series(prices, index=dates)

    ep = drawdown_episodes(s, min_depth=5.0, top_n=5)
    assert len(ep) == 2

    # Sorted by depth ascending (deepest -25% first, then -20%)
    ep1 = ep.iloc[0]
    assert ep1["peak_date"] == pd.Timestamp("2025-01-06")
    assert ep1["trough_date"] == pd.Timestamp("2025-01-08")
    assert pd.isna(ep1["recovery_date"])
    assert pytest.approx(ep1["depth"], abs=1e-4) == -25.0
    assert ep1["days_to_trough"] == 2
    assert np.isnan(ep1["days_to_recover"])

    ep2 = ep.iloc[1]
    assert ep2["peak_date"] == pd.Timestamp("2025-01-01")
    assert ep2["trough_date"] == pd.Timestamp("2025-01-03")
    assert ep2["recovery_date"] == pd.Timestamp("2025-01-05")
    assert pytest.approx(ep2["depth"], abs=1e-4) == -20.0
    assert ep2["days_to_trough"] == 2
    assert ep2["days_to_recover"] == 4

    # Current drawdown at last point (110 vs peak 120): (110-120)/120 * 100 = -8.3333%
    curr_dd = current_drawdown(s)
    assert curr_dd is not None
    assert pytest.approx(curr_dd, abs=1e-3) == -8.333


def test_min_depth_filter():
    """Test min_depth filtering excludes minor drawdowns."""
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    # Peak 100 -> Trough 97 (-3%) -> Recover 100
    prices = [100.0, 98.0, 97.0, 100.0, 105.0, 105.0]
    s = pd.Series(prices, index=dates)

    ep_default = drawdown_episodes(s, min_depth=5.0)
    assert ep_default.empty

    ep_small = drawdown_episodes(s, min_depth=2.0)
    assert len(ep_small) == 1
    assert pytest.approx(ep_small.iloc[0]["depth"], abs=1e-4) == -3.0


def test_top_n_limit():
    """Test top_n limits the number of returned drawdown episodes."""
    dates = pd.date_range("2025-01-01", periods=15, freq="D")
    # 3 episodes: -30%, -20%, -10%
    prices = [
        100, 70, 100,  # -30%
        100, 80, 100,  # -20%
        100, 90, 100,  # -10%
        100, 100, 100, 100, 100, 100,
    ]
    s = pd.Series(prices, index=dates)

    ep_top2 = drawdown_episodes(s, min_depth=5.0, top_n=2)
    assert len(ep_top2) == 2
    assert pytest.approx(ep_top2.iloc[0]["depth"], abs=1e-4) == -30.0
    assert pytest.approx(ep_top2.iloc[1]["depth"], abs=1e-4) == -20.0


def test_flat_and_monotonically_rising():
    """Test flat and monotonically rising series give 0 episodes and 0.0 current drawdown."""
    dates = pd.date_range("2025-01-01", periods=5, freq="D")

    # Flat series
    flat_s = pd.Series([50.0] * 5, index=dates)
    assert drawdown_episodes(flat_s).empty
    assert current_drawdown(flat_s) == 0.0

    # Monotonically rising
    rising_s = pd.Series([10.0, 12.0, 15.0, 20.0, 25.0], index=dates)
    assert drawdown_episodes(rising_s).empty
    assert current_drawdown(rising_s) == 0.0


def test_nan_and_zero_values_dropped():
    """Test NaN and non-positive (zero/negative) values are dropped correctly."""
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    prices = [100.0, np.nan, 0.0, -10.0, 80.0, np.nan, 100.0, 120.0]
    s = pd.Series(prices, index=dates)

    # Valid points after dropping NaN and <=0:
    # 2025-01-01: 100.0
    # 2025-01-05: 80.0 (trough, depth -20%)
    # 2025-01-07: 100.0 (recovered)
    # 2025-01-08: 120.0 (new peak)
    ep = drawdown_episodes(s, min_depth=5.0)
    assert len(ep) == 1
    assert ep.iloc[0]["peak_date"] == pd.Timestamp("2025-01-01")
    assert ep.iloc[0]["trough_date"] == pd.Timestamp("2025-01-05")
    assert ep.iloc[0]["recovery_date"] == pd.Timestamp("2025-01-07")
    assert pytest.approx(ep.iloc[0]["depth"], abs=1e-4) == -20.0

    assert current_drawdown(s) == 0.0


def test_fewer_than_two_valid_points():
    """Test series with fewer than 2 valid points returns None for current_drawdown and empty frame for episodes."""
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    s_empty = pd.Series([np.nan, 0.0, -5.0], index=dates)
    assert current_drawdown(s_empty) is None
    assert drawdown_episodes(s_empty).empty

    s_one = pd.Series([100.0, np.nan, 0.0], index=dates)
    assert current_drawdown(s_one) is None
    assert drawdown_episodes(s_one).empty


def test_performance_timing():
    """Test 5,000 points processed under 0.2s."""
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=5000, freq="D")
    returns = np.random.normal(0.0002, 0.015, size=5000)
    price_curve = 100.0 * np.exp(np.cumsum(returns))
    s = pd.Series(price_curve, index=dates)

    t0 = time.perf_counter()
    ep = drawdown_episodes(s, min_depth=5.0, top_n=10)
    curr_dd = current_drawdown(s)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    assert elapsed < 0.2, f"Execution took {elapsed:.4f}s >= 0.2s"
    assert isinstance(ep, pd.DataFrame)
    assert curr_dd is not None or curr_dd == 0.0
