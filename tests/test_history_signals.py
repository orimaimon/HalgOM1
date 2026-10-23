"""Tests for screener.history_signals."""
import time
import numpy as np
import pandas as pd
import pytest

from screener import history_signals, signals


def test_cross_up_sma200_within():
    """A ticker crossing above SMA200 three bars ago: cross_up_sma200 True for days=5, False for days=2, days_since_cross_up_sma200 == 3."""
    # Build 250 bars where SMA200 is constant 100
    # Last 5 bars (index 245..249):
    # 245: price = 95 (below 100)
    # 246: price = 105 (crossed above 100 -> 3 bars ago relative to bar 249)
    # 247: price = 106
    # 248: price = 107
    # 249: price = 108
    n = 250
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.full(n, 100.0)
    prices[:246] = 95.0
    prices[246:] = 105.0 + np.arange(4)

    closes = pd.DataFrame({"AAPL": prices}, index=dates)

    # days = 5 -> crossed 3 bars ago -> True
    res5 = history_signals.signals_within(closes, days=5, prepared=True)
    assert res5.loc["AAPL", "cross_up_sma200"] is True or res5.loc["AAPL", "cross_up_sma200"] == True
    assert res5.loc["AAPL", "days_since_cross_up_sma200"] == 3.0

    # days = 2 -> window is bars 248 and 249, cross was at 246 -> False
    res2 = history_signals.signals_within(closes, days=2, prepared=True)
    assert res2.loc["AAPL", "cross_up_sma200"] is False or res2.loc["AAPL", "cross_up_sma200"] == False
    assert res2.loc["AAPL", "days_since_cross_up_sma200"] == 3.0


def test_days_1_matches_signals_cross_up():
    """On the last bar only, the result for days=1 matches signals.cross_up('price', 'sma200')
    applied to a one-row frame built from the last two bars (price, sma200, close_prev, sma200_prev).
    """
    n = 250
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.full(n, 100.0)
    prices[-2] = 95.0
    prices[-1] = 105.0

    closes = pd.DataFrame({"AAPL": prices}, index=dates)

    # Calculate via history_signals
    res = history_signals.signals_within(closes, days=1, prepared=True)

    # Calculate via signals.cross_up on last two bars
    sma200 = closes["AAPL"].rolling(200, min_periods=200).mean()
    frame = pd.DataFrame({
        "price": [closes["AAPL"].iloc[-1]],
        "sma200": [sma200.iloc[-1]],
        "close_prev": [closes["AAPL"].iloc[-2]],
        "sma200_prev": [sma200.iloc[-2]]
    })
    sig_val = signals.cross_up("price", "sma200")(frame).iloc[0]

    res_val = res.loc["AAPL", "cross_up_sma200"]
    assert (res_val == sig_val) or (bool(res_val) == bool(sig_val))


def test_golden_cross_and_death_cross():
    """Golden cross and death cross on constructed series."""
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    p_aapl = np.full(n, 100.0)
    p_aapl[200:] = 200.0   # SMA50 moves up rapidly, SMA200 moves up slowly

    p_msft = np.full(n, 200.0)
    p_msft[200:] = 50.0    # SMA50 moves down rapidly

    closes = pd.DataFrame({"AAPL": p_aapl, "MSFT": p_msft}, index=dates)
    res = history_signals.signals_within(closes, days=100, prepared=True)

    assert res.loc["AAPL", "golden_cross"] == True
    assert res.loc["AAPL", "death_cross"] == False

    assert res.loc["MSFT", "golden_cross"] == False
    assert res.loc["MSFT", "death_cross"] == True


def test_breakouts():
    """Breakouts: a new 20-day high two bars ago -> breakout_20 True; a high that equals (not exceeds) the prior max -> False."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # AAPL: last 20 bars prior to bar 47 were <= 100. Bar 47 = 105 (new high 2 bars ago relative to bar 49)
    p_aapl = np.full(n, 100.0)
    p_aapl[47] = 105.0
    p_aapl[48] = 102.0
    p_aapl[49] = 102.0

    # MSFT: prior 20 bars max was 100. Bar 49 = 100 (equals prior max, not exceeds)
    p_msft = np.full(n, 100.0)

    closes = pd.DataFrame({"AAPL": p_aapl, "MSFT": p_msft}, index=dates)
    res = history_signals.signals_within(closes, days=5, prepared=True)

    assert res.loc["AAPL", "breakout_20"] == True
    assert res.loc["MSFT", "breakout_20"] == False


def test_short_history_na_sma200():
    """A ticker with 150 bars -> SMA200 signals are NA, SMA50 signals are computed."""
    n = 150
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.full(n, 100.0)
    prices[-10:] = 110.0

    closes = pd.DataFrame({"SHORT": prices}, index=dates)
    res = history_signals.signals_within(closes, days=5, prepared=True)

    assert pd.isna(res.loc["SHORT", "cross_up_sma200"])
    assert pd.isna(res.loc["SHORT", "days_since_cross_up_sma200"])
    assert res.loc["SHORT", "cross_up_sma50"] == False or res.loc["SHORT", "cross_up_sma50"] == True


def test_split_in_middle_prepared():
    """A 2:1 split in the middle: no false cross or breakout caused by the jump (because of prepare)."""
    n = 250
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Ticker was trading around 200, then 2:1 split to 100 at bar 100
    prices = np.full(n, 100.0)
    prices[:100] = 200.0

    closes = pd.DataFrame({"SPLIT": prices}, index=dates)

    # prepare undoes the 2:1 split jump
    prepared_closes = history_signals.prepare(closes)
    # Check that prepared_closes near bar 100 has no 2x jump
    assert abs(prepared_closes["SPLIT"].iloc[99] - prepared_closes["SPLIT"].iloc[100]) < 5.0

    res = history_signals.signals_within(closes, days=250, prepared=False)
    # No false golden cross or breakout_250 on the split day
    val = res.loc["SPLIT", "breakout_250"]
    assert pd.isna(val) or val == False


def test_nan_gaps_do_not_raise():
    """NaN gaps do not raise."""
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.full(n, 100.0)
    prices[20:25] = np.nan
    prices[50:60] = np.nan

    closes = pd.DataFrame({"GAP": prices}, index=dates)
    res = history_signals.signals_within(closes, days=5, prepared=False)
    assert "cross_up_sma50" in res.columns


def test_hebrew_labels_complete():
    """Every signal column has a Hebrew label."""
    closes = pd.DataFrame({"AAPL": [100.0] * 50}, index=pd.date_range("2025-01-01", periods=50, freq="B"))
    res = history_signals.signals_within(closes, days=5, prepared=True)

    for col in res.columns:
        assert col in history_signals.SIGNAL_LABELS, f"Missing Hebrew label for column {col}"
        assert isinstance(history_signals.SIGNAL_LABELS[col], str)
        assert len(history_signals.SIGNAL_LABELS[col]) > 0


def test_speed_benchmark():
    """Speed: 5,000 tickers x 500 days must run in under 2 seconds on CI."""
    n_days = 500
    n_tickers = 5000
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    np.random.seed(42)
    # Generate random walk closes around 100
    returns = np.random.normal(0, 0.02, size=(n_days, n_tickers))
    price_matrix = 100.0 * np.exp(np.cumsum(returns, axis=0))

    closes = pd.DataFrame(price_matrix, index=dates, columns=[f"T{i}" for i in range(n_tickers)])

    start_time = time.perf_counter()
    res = history_signals.signals_within(closes, days=5, prepared=True)
    elapsed = time.perf_counter() - start_time

    assert res.shape == (n_tickers, 12)
    assert elapsed < 2.0, f"Execution took {elapsed:.2f} seconds, expected < 2.0s"


def test_breakout_needs_a_full_window():
    import numpy as np
    import pandas as pd
    from screener import history_signals as hs
    idx = pd.bdate_range("2024-01-01", periods=300)
    old = np.linspace(10, 20, 300)                       # a steady riser: new highs every day
    young = np.r_[np.full(270, np.nan), np.linspace(5, 8, 30)]   # listed 30 bars ago, also rising
    res = hs.signals_within(pd.DataFrame({"OLD": old, "NEW": young}, index=idx), days=5)
    assert res.loc["OLD", "breakout_250"] == True       # noqa: E712
    assert res.loc["NEW", "breakout_20"] == True        # noqa: E712
    assert pd.isna(res.loc["NEW", "breakout_250"])
