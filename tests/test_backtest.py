"""Tests for screener.backtest."""
import time
import numpy as np
import pandas as pd
import pytest

from screener import backtest, history_signals


def test_event_matrix_agrees_with_signals_within():
    """event_matrix fired within last 5 bars matches signals_within(closes, days=5)."""
    n_days = 260
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Generate synthetic price series for multiple tickers
    np.random.seed(42)
    p_aapl = np.full(n_days, 100.0)
    p_aapl[-5] = 90.0
    p_aapl[-1] = 110.0  # Crosses SMA or breaks out

    p_msft = np.linspace(50, 150, n_days)  # Steady rise

    p_goog = np.full(n_days, 100.0)
    p_goog[200:] = 80.0  # Death cross

    closes = pd.DataFrame({"AAPL": p_aapl, "MSFT": p_msft, "GOOG": p_goog}, index=dates)

    boolean_signals = [
        "cross_up_sma50",
        "cross_down_sma50",
        "cross_up_sma200",
        "cross_down_sma200",
        "golden_cross",
        "death_cross",
        "breakout_20",
        "breakout_60",
        "breakout_120",
        "breakout_250",
    ]

    sw = history_signals.signals_within(closes, days=5, prepared=False)

    for sig in boolean_signals:
        em = history_signals.event_matrix(closes, sig, prepared=False)
        fired_5 = em.tail(5).any(axis=0)

        for ticker in closes.columns:
            sw_val = sw.loc[ticker, sig]
            if not pd.isna(sw_val):
                assert fired_5[ticker] == bool(sw_val), f"Mismatch for {sig} on {ticker}: matrix {fired_5[ticker]} vs sw {sw_val}"


def test_sma200_cross_up_known_day_and_pending():
    """Constructed SMA200 cross-up on known day gives exact return; recent event is pending."""
    n_days = 250
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # AAPL: SMA200 is 100. At day 200, price crosses from 95 to 100.
    # Day 200 price = 100.0
    # Day 220 (day 200 + 20) price = 110.0 -> 20-day return = 10.0%
    prices = np.full(n_days, 100.0)
    prices[199] = 95.0
    prices[200] = 100.0
    prices[220] = 110.0

    # Also add a cross-up on day 245 (last 10 days) -> pending for 20 and 60
    prices[244] = 90.0
    prices[245] = 120.0

    closes = pd.DataFrame({"AAPL": prices}, index=dates)

    res = backtest.run(closes, "cross_up_sma200", horizons=(5, 20, 60), cooldown=5)

    assert res["events"] == 2
    assert res["tickers"] == 1
    table = res["table"]

    # First event at day 200
    row0 = table.iloc[0]
    assert row0["ticker"] == "AAPL"
    assert abs(row0["ret_20"] - 10.0) < 1e-5

    # Second event at day 245 -> pending for 20 and 60
    row1 = table.iloc[1]
    assert pd.isna(row1["ret_20"])
    assert pd.isna(row1["ret_60"])

    # Summary
    h20 = res["horizons"][20]
    assert h20["pending"] == 1
    assert h20["n"] == 1
    assert h20["median"] == 10.0


def test_cooldown():
    """Three fires on consecutive days -> 1 event with cooldown=5, 3 with cooldown=0."""
    n_days = 50
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    prices = np.full(n_days, 100.0)
    # Breakout 20 on days 25, 26, 27
    prices[25] = 105.0
    prices[26] = 110.0
    prices[27] = 115.0

    closes = pd.DataFrame({"XYZ": prices}, index=dates)

    res_cd5 = backtest.run(closes, "breakout_20", cooldown=5)
    assert res_cd5["events"] == 1

    res_cd0 = backtest.run(closes, "breakout_20", cooldown=0)
    assert res_cd0["events"] == 3


def test_split_in_forward_window():
    """A 2:1 split inside the forward window does not create a -50% return."""
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Breakout on day 25 at price 105
    prices = np.full(n_days, 100.0)
    prices[25] = 105.0  # breakout
    # 2:1 split on day 35 -> price drops from 100.0 to 50.0
    prices[35:] = 50.0

    closes = pd.DataFrame({"SPLT": prices}, index=dates)

    res = backtest.run(closes, "breakout_20", horizons=(5, 20), cooldown=5)
    table = res["table"]
    assert len(table) == 1
    ret_20 = table.iloc[0]["ret_20"]

    # Prepared closes undoes the 2:1 split jump, so 20-day return is ~ -4.76%, NOT -50%
    assert not pd.isna(ret_20)
    assert abs(ret_20) < 5.0  # Close to 0%, not -50%


def test_benchmark():
    """SPY series excess uses SPY return; without benchmark, uses universe median."""
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    p_a = np.full(n_days, 100.0)
    p_a[25] = 105.0  # breakout day 25
    p_a[30] = 115.5  # day 25+5 price -> +10% return for stock A

    p_b = np.full(n_days, 100.0)
    p_b[25] = 100.0
    p_b[30] = 100.0  # 0% return for stock B

    closes = pd.DataFrame({"A": p_a, "B": p_b}, index=dates)

    # Benchmark SPY with +4% return between day 25 and 30
    spy = pd.Series(100.0, index=dates)
    spy.iloc[25] = 100.0
    spy.iloc[30] = 104.0

    # With SPY benchmark: stock A return = +10%, SPY = +4%, excess = +6%
    res_spy = backtest.run(closes, "breakout_20", horizons=(5,), cooldown=5, benchmark=spy)
    assert res_spy["horizons"][5]["excess"] == 6.0

    # Without benchmark: universe median return over day 25->30 is median(10%, 0%) = 5%
    # Stock A return = 10%, excess over universe median = 10% - 5% = 5%
    res_no_spy = backtest.run(closes, "breakout_20", horizons=(5,), cooldown=5, benchmark=None)
    assert res_no_spy["horizons"][5]["excess"] == 5.0


def test_no_events_and_unknown_signal():
    """No events -> events == 0, every statistic None, no exception. Unknown signal -> ValueError."""
    n_days = 50
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    closes = pd.DataFrame({"FLAT": np.full(n_days, 100.0)}, index=dates)

    res = backtest.run(closes, "cross_up_sma200")
    assert res["events"] == 0
    assert res["tickers"] == 0
    assert res["first_day"] is None
    assert res["last_day"] is None
    assert res["table"].empty

    for h in (5, 20, 60):
        h_sum = res["horizons"][h]
        assert h_sum["n"] == 0
        assert h_sum["pending"] == 0
        assert h_sum["median"] is None
        assert h_sum["mean"] is None
        assert h_sum["pct_positive"] is None
        assert h_sum["excess"] is None

    with pytest.raises(ValueError, match="Unknown boolean signal"):
        backtest.run(closes, "invalid_signal_name")


def test_speed_benchmark():
    """Speed test: 3,000 tickers x 750 days in under 3 seconds."""
    n_days = 750
    n_tickers = 3000
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    np.random.seed(42)
    returns = np.random.normal(0, 0.02, size=(n_days, n_tickers))
    price_matrix = 100.0 * np.exp(np.cumsum(returns, axis=0))

    closes = pd.DataFrame(price_matrix, index=dates, columns=[f"T{i}" for i in range(n_tickers)])

    start_time = time.perf_counter()
    res = backtest.run(closes, "breakout_20", horizons=(5, 20, 60), cooldown=5)
    elapsed = time.perf_counter() - start_time

    assert res["events"] > 0
    assert elapsed < 3.0, f"Backtest took {elapsed:.2f}s, expected < 3.0s"


def test_labels_dict():
    """LABELS dictionary is complete and contains Hebrew labels."""
    assert "disclaimer" in backtest.LABELS
    assert backtest.LABELS["disclaimer"] == "תוצאות עבר על נתוני היסטוריה; אינן מבטיחות תוצאות בעתיד ואינן ייעוץ השקעות."
    for k in ("signal", "events", "tickers", "first_day", "last_day", "n", "pending", "median", "mean", "pct_positive", "excess"):
        assert k in backtest.LABELS
