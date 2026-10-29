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


def test_duckdb_compute_returns_parity():
    """Verify compute_returns DuckDB implementation exact match with Pandas logic."""
    prices_list = [100.0, 105.0, 102.0, 108.0, 95.0, 90.0, 100.0, 110.0]
    p_series = pd.Series(prices_list, index=pd.date_range("2024-01-01", periods=8))

    # Test forward returns
    for h in (1, 2, 3):
        # Pandas expected
        exp_fwd = (p_series.shift(-h) / p_series - 1.0) * 100.0
        res_fwd = backtest.compute_returns(p_series, horizon=h, kind="forward")
        pd.testing.assert_series_equal(res_fwd, exp_fwd)

        # Test trailing returns
        exp_trail = (p_series / p_series.shift(h) - 1.0) * 100.0
        res_trail = backtest.compute_returns(p_series, horizon=h, kind="trailing")
        pd.testing.assert_series_equal(res_trail, exp_trail)

    # DataFrame test
    df_prices = pd.DataFrame({"AAPL": p_series, "MSFT": p_series * 1.5})
    res_df = backtest.compute_returns(df_prices, horizon=2, kind="forward")
    exp_df = (df_prices.shift(-2) / df_prices - 1.0) * 100.0
    pd.testing.assert_frame_equal(res_df, exp_df)

    # Numpy array test
    arr_prices = np.array(prices_list)
    res_arr = backtest.compute_returns(arr_prices, horizon=1, kind="forward")
    exp_arr = ((np.roll(arr_prices, -1) / arr_prices - 1.0) * 100.0)
    exp_arr[-1] = np.nan
    np.testing.assert_allclose(res_arr[:-1], exp_arr[:-1])
    assert np.isnan(res_arr[-1])

    # Exception checks
    with pytest.raises(ValueError, match="Horizon must be a positive integer"):
        backtest.compute_returns(p_series, horizon=0)
    with pytest.raises(ValueError, match="Unknown kind"):
        backtest.compute_returns(p_series, kind="invalid")


def test_duckdb_compute_sharpe_parity():
    """Verify compute_sharpe DuckDB implementation exact match with Pandas logic."""
    np.random.seed(42)
    rets = np.random.normal(0.001, 0.02, 100)

    # Pandas baseline Sharpe
    s_series = pd.Series(rets)
    rf = 0.0001
    excess = s_series - rf
    exp_sharpe = float(excess.mean() / excess.std(ddof=1))
    exp_sharpe_ann = exp_sharpe * np.sqrt(252)

    res_sharpe = backtest.compute_sharpe(rets, rf=rf)
    assert res_sharpe is not None
    assert np.isclose(res_sharpe, exp_sharpe)

    res_sharpe_ann = backtest.compute_sharpe(rets, rf=rf, annualize=True, period_factor=252)
    assert res_sharpe_ann is not None
    assert np.isclose(res_sharpe_ann, exp_sharpe_ann)

    # Edge cases
    assert backtest.compute_sharpe([]) is None
    assert backtest.compute_sharpe([1.0]) is None
    assert backtest.compute_sharpe([2.0, 2.0, 2.0]) is None  # zero std
    assert backtest.compute_sharpe([np.nan, np.nan]) is None


def test_duckdb_compute_drawdown_parity():
    """Verify compute_drawdown DuckDB implementation exact match with Pandas logic."""
    prices_list = [100.0, 105.0, 102.0, 108.0, 95.0, 90.0, 100.0, 110.0]
    p_series = pd.Series(prices_list)

    # Pandas baseline drawdown
    peak = p_series.cummax()
    exp_dd = float(((p_series - peak) / peak * 100.0).min())

    res_dd = backtest.compute_drawdown(p_series)
    assert res_dd is not None
    assert np.isclose(res_dd, exp_dd)

    # List input
    res_dd_list = backtest.compute_drawdown(prices_list)
    assert res_dd_list is not None
    assert np.isclose(res_dd_list, exp_dd)

    # Edge cases
    assert backtest.compute_drawdown([]) is None
    assert backtest.compute_drawdown([np.nan, np.nan]) is None


def test_load_price_history_date_filtering_parity(tmp_path):
    """Verify DuckDB date filtering produces identical results to Pandas filtering."""
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="B")
    np.random.seed(42)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "AAPL": np.random.normal(150, 5, len(dates)),
        "MSFT": np.random.normal(300, 10, len(dates)),
    })
    parquet_path = tmp_path / "prices.parquet"
    df.to_parquet(parquet_path)

    start_date = "2022-03-01"
    end_date = "2022-09-30"

    # DuckDB predicate pushdown load
    res_duck = backtest.load_price_history(parquet_path, start_date=start_date, end_date=end_date)

    # Standard pandas load and filter
    df_full = pd.read_parquet(parquet_path)
    mask = (df_full["date"] >= start_date) & (df_full["date"] <= end_date)
    res_pandas = df_full[mask].reset_index(drop=True)

    pd.testing.assert_frame_equal(res_duck.reset_index(drop=True), res_pandas)

    # Test with datetime / date objects
    res_obj = backtest.load_price_history(
        parquet_path,
        start_date=pd.Timestamp(start_date).date(),
        end_date=pd.Timestamp(end_date),
    )
    pd.testing.assert_frame_equal(res_obj.reset_index(drop=True), res_pandas)


def test_load_price_history_empty_range(tmp_path):
    """Verify empty date ranges produce an empty DataFrame with proper columns."""
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "price": range(len(dates)),
    })
    parquet_path = tmp_path / "prices.parquet"
    df.to_parquet(parquet_path)

    # Query range outside available data
    res = backtest.load_price_history(parquet_path, start_date="2025-01-01", end_date="2025-06-01")
    assert isinstance(res, pd.DataFrame)
    assert res.empty
    assert list(res.columns) == ["date", "price"]


def test_load_price_history_missing_file(tmp_path):
    """Verify missing file handling raises appropriate error."""
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(Exception):
        backtest.load_price_history(missing_path, start_date="2022-01-01")


def test_load_price_history_fallback(tmp_path, monkeypatch):
    """Verify graceful fallback to pd.read_parquet when DuckDB query fails."""
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "price": range(len(dates)),
    })
    parquet_path = tmp_path / "prices.parquet"
    df.to_parquet(parquet_path)

    # Force query_parquet_df to raise exception
    def mock_query_parquet_df(*args, **kwargs):
        raise RuntimeError("Simulated DuckDB error")

    monkeypatch.setattr("screener.data.duck_engine.query_parquet_df", mock_query_parquet_df)

    start_date = "2022-06-01"
    end_date = "2022-08-31"

    res_fallback = backtest.load_price_history(parquet_path, start_date=start_date, end_date=end_date)

    df_full = pd.read_parquet(parquet_path)
    mask = (df_full["date"] >= start_date) & (df_full["date"] <= end_date)
    res_pandas = df_full[mask]

    pd.testing.assert_frame_equal(res_fallback, res_pandas)
