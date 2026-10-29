import time
import numpy as np
import pandas as pd
import pytest

from screener.portfolio import PList, PORTFOLIO
from screener.portfolio_correlation import (
    STRESS_COLUMNS,
    high_correlation_pairs,
    return_correlations,
    stress_scenarios,
)


def make_history(prices_list: list[float], dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    n = len(prices_list)
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "close": prices_list,
        "adj_close": prices_list,
    }, index=dates)


def test_return_correlations_and_high_correlation_pairs(monkeypatch):
    # Generate 100 days of data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # Base return
    base_ret = np.random.normal(0.001, 0.02, 100)

    # NVDA and SMH: nearly identical
    nvda_ret = base_ret + np.random.normal(0, 0.001, 100)
    smh_ret = base_ret + np.random.normal(0, 0.001, 100)

    # AAPL: uncorrelated
    aapl_ret = np.random.normal(0.001, 0.02, 100)

    # Short history ticker (only 10 overlapping days with others)
    dates_short = dates[-10:]
    short_ret = np.random.normal(0.001, 0.02, 10)

    # Convert returns to prices
    nvda_prices = 100.0 * np.exp(np.cumsum(nvda_ret))
    smh_prices = 200.0 * np.exp(np.cumsum(smh_ret))
    aapl_prices = 150.0 * np.exp(np.cumsum(aapl_ret))
    short_prices = 50.0 * np.exp(np.cumsum(short_ret))

    histories = {
        "NVDA": make_history(nvda_prices.tolist(), dates),
        "SMH": make_history(smh_prices.tolist(), dates),
        "AAPL": make_history(aapl_prices.tolist(), dates),
        "SHORT": make_history(short_prices.tolist(), dates_short),
    }

    monkeypatch.setattr("screener.data.prices.history", lambda ticker: histories.get(ticker, pd.DataFrame()))

    # Test return_correlations
    corr = return_correlations(["NVDA", "SMH", "AAPL", "SHORT"], days=90, min_overlap=20)
    assert not corr.empty
    assert corr.loc["NVDA", "SMH"] > 0.95
    # SHORT should have NaN correlations with others due to min_overlap=20 > 10 points
    assert pd.isna(corr.loc["NVDA", "SHORT"])

    # Test high_correlation_pairs
    plist = PList(
        name="Tech Fund",
        kind=PORTFOLIO,
        rows=[
            {"ticker": "NVDA", "date": "2024-01-01", "price": 100.0, "quantity": 100, "side": "buy", "fees": 0},
            {"ticker": "SMH", "date": "2024-01-01", "price": 200.0, "quantity": 100, "side": "buy", "fees": 0},
            {"ticker": "AAPL", "date": "2024-01-01", "price": 150.0, "quantity": 100, "side": "buy", "fees": 0},
            {"ticker": "SHORT", "date": "2024-01-01", "price": 50.0, "quantity": 100, "side": "buy", "fees": 0},
        ],
    )

    universe = pd.DataFrame([
        {"ticker": "NVDA", "company": "NVIDIA", "price": 100.0, "sector": "Technology", "beta": 1.5},
        {"ticker": "SMH", "company": "Semiconductor ETF", "price": 200.0, "sector": "Technology", "beta": 1.4},
        {"ticker": "AAPL", "company": "Apple", "price": 150.0, "sector": "Technology", "beta": 1.0},
        {"ticker": "SHORT", "company": "Short Co", "price": 50.0, "sector": "Technology", "beta": 1.0},
    ])

    pairs = high_correlation_pairs(plist, universe, threshold=0.80, days=90, min_overlap=20)
    assert len(pairs) == 1
    pair = pairs[0]

    # SMH ($200*100=$20,000) + NVDA ($100*100=$10,000) out of total $50,000
    # SMH weight = 40%, NVDA weight = 20% -> combined weight = 60%
    assert pair["ticker_a"] == "SMH"
    assert pair["ticker_b"] == "NVDA"
    assert pair["correlation"] > 0.95
    assert pair["combined_weight"] == pytest.approx(60.0)
    assert "SMH ו-NVDA זזות יחד (מתאם" in pair["warning"]
    assert "ומהוות יחד 60% מהתיק" in pair["warning"]


def test_stress_scenarios():
    # Exposure DataFrame with total value $10,000 and weighted beta 1.5
    exp = pd.DataFrame([
        {"ticker": "NVDA", "value": 5000.0, "weight": 50.0, "beta": 2.0},
        {"ticker": "AAPL", "value": 5000.0, "weight": 50.0, "beta": 1.0},
    ])

    df_stress = stress_scenarios(exp, market_moves_pct=(-10.0, 5.0))
    assert list(df_stress.columns) == STRESS_COLUMNS
    assert len(df_stress) == 2

    # Market move -10%: eff_beta = (5000*2 + 5000*1)/10000 = 1.5
    # est_portfolio_pct = -10 * 1.5 = -15.0%
    # est_pnl_dollars = 10000 * -0.15 = -$1,500
    # est_value_dollars = 10000 - 1500 = 8500
    row_down = df_stress[df_stress["market_move_pct"] == -10.0].iloc[0]
    assert row_down["est_portfolio_pct"] == pytest.approx(-15.0)
    assert row_down["est_pnl_dollars"] == pytest.approx(-1500.0)
    assert row_down["est_value_dollars"] == pytest.approx(8500.0)
    assert "בירידת שוק של 10%" in row_down["description_he"]
    assert "-15.0%" in row_down["description_he"]
    assert "$-1,500" in row_down["description_he"]

    # Test missing beta fallback to 1.0
    exp_missing_beta = pd.DataFrame([
        {"ticker": "NVDA", "value": 5000.0, "weight": 50.0, "beta": np.nan},
        {"ticker": "AAPL", "value": 5000.0, "weight": 50.0, "beta": 1.0},
    ])
    df_missing = stress_scenarios(exp_missing_beta, market_moves_pct=(-10.0,))
    # eff_beta = (5000*1.0 + 5000*1.0)/10000 = 1.0
    row_missing = df_missing.iloc[0]
    assert row_missing["est_portfolio_pct"] == pytest.approx(-10.0)
    assert row_missing["est_pnl_dollars"] == pytest.approx(-1000.0)

    # Test empty exposure frame or value <= 0
    empty_df = stress_scenarios(pd.DataFrame(columns=["ticker", "value", "beta"]))
    assert list(empty_df.columns) == STRESS_COLUMNS
    assert empty_df.empty

    zero_val_df = stress_scenarios(pd.DataFrame([{"ticker": "A", "value": 0.0, "beta": 1.0}]))
    assert list(zero_val_df.columns) == STRESS_COLUMNS
    assert zero_val_df.empty


def test_speed_performance(monkeypatch):
    # 30 tickers x 250 days must run in < 0.5s
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    tickers = [f"TICK{i}" for i in range(30)]

    histories = {}
    for t in tickers:
        rets = np.random.normal(0.0005, 0.015, 250)
        prices_arr = 100.0 * np.exp(np.cumsum(rets))
        histories[t] = make_history(prices_arr.tolist(), dates)

    monkeypatch.setattr("screener.data.prices.history", lambda ticker: histories.get(ticker, pd.DataFrame()))

    t0 = time.perf_counter()
    corr = return_correlations(tickers, days=250, min_overlap=20)
    elapsed = time.perf_counter() - t0

    assert corr.shape == (30, 30)
    assert elapsed < 0.5, f"Execution took {elapsed:.3f}s, exceeding limit of 0.5s"
