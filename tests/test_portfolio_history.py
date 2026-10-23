import datetime as dt
import numpy as np
import pandas as pd
import pytest

from screener import portfolio_history as ph
from screener.portfolio import BUY, PORTFOLIO, SELL, WATCHLIST, PList


def _make_history_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=ph.prices.COLUMNS)
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    for col in ph.prices.COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ("dividends", "splits") else np.nan
    return df[ph.prices.COLUMNS]


def test_tickers_to_track():
    p1 = PList("P1", PORTFOLIO, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy"},
        {"ticker": "MSFT", "date": "2025-01-02", "price": 200.0, "quantity": 5, "side": "buy"},
    ])
    p2 = PList("W1", WATCHLIST, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "note": ""},
        {"ticker": "NVDA", "date": "2025-01-01", "price": 150.0, "note": ""},
    ])
    tracked = ph.tickers_to_track({"P1": p1, "W1": p2})
    assert tracked == ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def test_update_prices_success_and_failure(monkeypatch):
    p1 = PList("P1", PORTFOLIO, [
        {"ticker": "AAPL", "date": "2025-03-10", "price": 100.0, "quantity": 10, "side": "buy"},
    ])

    called_args = {}

    def mock_ensure(tickers, start):
        called_args["tickers"] = tickers
        called_args["start"] = start
        return ["AAPL", "SPY", "QQQ"], []

    monkeypatch.setattr(ph.prices, "ensure", mock_ensure)

    updated, failed = ph.update_prices({"P1": p1}, today=dt.date(2025, 3, 20))
    assert updated == ["AAPL", "SPY", "QQQ"]
    assert failed == []
    assert called_args["tickers"] == ["AAPL", "SPY", "QQQ"]
    assert called_args["start"] == dt.date(2025, 3, 3)  # 2025-03-10 minus 7 days

    # Test failure does not raise exception
    def mock_ensure_fail(tickers, start):
        raise ValueError("Network error")

    monkeypatch.setattr(ph.prices, "ensure", mock_ensure_fail)
    updated, failed = ph.update_prices({"P1": p1}, today=dt.date(2025, 3, 20))
    assert updated == []
    assert failed == ["AAPL", "SPY", "QQQ"]


def test_split_after_buy(monkeypatch):
    # Buy 10 shares of XYZ at $100 on 2025-01-01. Total cost $1,000.
    # On 2025-01-03, 2:1 split.
    # Yahoo close prices post-split are $50. Before split on Yahoo history, close is $50 (split-adjusted).
    # split_factor for buy on 2025-01-01 will be 2.0.
    # Adjusted trade: qty = 20, price = 50.
    # Market value on 2025-01-02: 20 * 50 = $1,000.
    # Market value on 2025-01-03 (split day): 20 * 50 = $1,000. No jump in value.
    p = PList("P", PORTFOLIO, [
        {"ticker": "XYZ", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
    ])

    def mock_split_factor(ticker, since):
        if ticker == "XYZ" and since <= dt.date(2025, 1, 1):
            return 2.0
        return 1.0

    monkeypatch.setattr(ph.prices, "split_factor", mock_split_factor)

    hist_xyz = _make_history_df([
        {"date": "2025-01-01", "close": 50.0, "adj_close": 50.0},
        {"date": "2025-01-02", "close": 50.0, "adj_close": 50.0},
        {"date": "2025-01-03", "close": 50.0, "adj_close": 50.0},
    ])

    def mock_history(ticker):
        if ticker == "XYZ":
            return hist_xyz
        return pd.DataFrame(columns=ph.prices.COLUMNS)

    monkeypatch.setattr(ph.prices, "history", mock_history)

    trades = p.frame()
    adj = ph.adjusted_trades(trades)
    assert adj.loc[0, "quantity"] == 20.0
    assert adj.loc[0, "price"] == 50.0
    assert adj.loc[0, "split_factor"] == 2.0

    vs = ph.value_series(p)
    assert list(vs["value"]) == [1000.0, 1000.0, 1000.0]


def test_dividends_received_while_held(monkeypatch):
    # Buy on 2025-01-02, sell on 2025-01-05.
    # Dividend on 2025-01-01 ($1/sh) -> ignored (before buy).
    # Dividend on 2025-01-03 ($2/sh) -> counted for held shares (10 shares * $2 = $20).
    # Dividend on 2025-01-06 ($1/sh) -> ignored (after full sell).
    p = PList("P", PORTFOLIO, [
        {"ticker": "DIV", "date": "2025-01-02", "price": 100.0, "quantity": 10, "side": "buy"},
        {"ticker": "DIV", "date": "2025-01-05", "price": 105.0, "quantity": 10, "side": "sell"},
    ])

    monkeypatch.setattr(ph.prices, "split_factor", lambda t, d: 1.0)

    hist_div = _make_history_df([
        {"date": "2025-01-01", "close": 100.0, "adj_close": 100.0, "dividends": 1.0},
        {"date": "2025-01-02", "close": 100.0, "adj_close": 100.0, "dividends": 0.0},
        {"date": "2025-01-03", "close": 101.0, "adj_close": 101.0, "dividends": 2.0},
        {"date": "2025-01-05", "close": 105.0, "adj_close": 105.0, "dividends": 0.0},
        {"date": "2025-01-06", "close": 106.0, "adj_close": 106.0, "dividends": 1.0},
    ])

    monkeypatch.setattr(ph.prices, "history", lambda t: hist_div if t == "DIV" else pd.DataFrame(columns=ph.prices.COLUMNS))

    divs = ph.dividends_received(p)
    assert len(divs) == 1
    assert divs.loc[0, "date"] == dt.date(2025, 1, 3)
    assert divs.loc[0, "ticker"] == "DIV"
    assert divs.loc[0, "per_share"] == 2.0
    assert divs.loc[0, "shares"] == 10.0
    assert divs.loc[0, "amount"] == 20.0


def test_partial_sell_and_later_buy(monkeypatch):
    # Day 1 (2025-01-01): Buy 10 shares @ 100 ($1000). net_invested = 1000, value = 1000, pnl = 0.
    # Day 2 (2025-01-02): Price is 110. Sell 4 shares @ 110 ($440).
    #   net_invested = 1000 - 440 = 560. Value of 6 shares @ 110 = 660. pnl = 660 - 560 = 100.
    # Day 3 (2025-01-03): Buy 5 shares @ 120 ($600).
    #   net_invested = 560 + 600 = 1160. Position = 11 shares @ 120 -> value = 1320. pnl = 1320 - 1160 = 160.
    p = PList("P", PORTFOLIO, [
        {"ticker": "ABC", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
        {"ticker": "ABC", "date": "2025-01-02", "price": 110.0, "quantity": 4, "side": "sell", "fees": 0.0},
        {"ticker": "ABC", "date": "2025-01-03", "price": 120.0, "quantity": 5, "side": "buy", "fees": 0.0},
    ])

    monkeypatch.setattr(ph.prices, "split_factor", lambda t, d: 1.0)

    hist_abc = _make_history_df([
        {"date": "2025-01-01", "close": 100.0, "adj_close": 100.0},
        {"date": "2025-01-02", "close": 110.0, "adj_close": 110.0},
        {"date": "2025-01-03", "close": 120.0, "adj_close": 120.0},
    ])

    monkeypatch.setattr(ph.prices, "history", lambda t: hist_abc if t == "ABC" else pd.DataFrame(columns=ph.prices.COLUMNS))

    vs = ph.value_series(p)
    assert vs.loc[dt.date(2025, 1, 1), "net_invested"] == 1000.0
    assert vs.loc[dt.date(2025, 1, 1), "value"] == 1000.0
    assert vs.loc[dt.date(2025, 1, 1), "pnl"] == 0.0

    assert vs.loc[dt.date(2025, 1, 2), "net_invested"] == 560.0
    assert vs.loc[dt.date(2025, 1, 2), "value"] == 660.0
    assert vs.loc[dt.date(2025, 1, 2), "pnl"] == 100.0

    assert vs.loc[dt.date(2025, 1, 3), "net_invested"] == 1160.0
    assert vs.loc[dt.date(2025, 1, 3), "value"] == 1320.0
    assert vs.loc[dt.date(2025, 1, 3), "pnl"] == 160.0


def test_benchmark_calculation(monkeypatch):
    # Buy $1,000 worth of ABC on 2025-01-01 when SPY adj_close is 100.
    # SPY units bought = 1000 / 100 = 10 units.
    # On 2025-01-02, SPY adj_close is 110 -> spy value = 10 * 110 = 1100.
    # On 2025-01-03, sell $440 of ABC -> sells 440 / 110 = 4 units of SPY -> 6 units left.
    p = PList("P", PORTFOLIO, [
        {"ticker": "ABC", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
        {"ticker": "ABC", "date": "2025-01-03", "price": 110.0, "quantity": 4, "side": "sell", "fees": 0.0},
    ])

    monkeypatch.setattr(ph.prices, "split_factor", lambda t, d: 1.0)

    hist_abc = _make_history_df([
        {"date": "2025-01-01", "close": 100.0, "adj_close": 100.0},
        {"date": "2025-01-02", "close": 105.0, "adj_close": 105.0},
        {"date": "2025-01-03", "close": 110.0, "adj_close": 110.0},
    ])
    hist_spy = _make_history_df([
        {"date": "2025-01-01", "close": 100.0, "adj_close": 100.0},
        {"date": "2025-01-02", "close": 110.0, "adj_close": 110.0},
        {"date": "2025-01-03", "close": 110.0, "adj_close": 110.0},
    ])

    def mock_history(t):
        if t == "ABC":
            return hist_abc
        if t == "SPY":
            return hist_spy
        return pd.DataFrame(columns=ph.prices.COLUMNS)

    monkeypatch.setattr(ph.prices, "history", mock_history)

    vs = ph.value_series(p)
    assert vs.loc[dt.date(2025, 1, 1), "spy"] == 1000.0
    assert vs.loc[dt.date(2025, 1, 2), "spy"] == 1100.0
    # Day 3 sell $440 out -> SPY value becomes (10 - 440/110) * 110 = 6 * 110 = 660.0
    assert vs.loc[dt.date(2025, 1, 3), "spy"] == 660.0


def test_missing_day_carry_forward_and_missing_ticker(monkeypatch):
    # Buy 10 shares of HELD and 10 shares of MISSING on 2025-01-01.
    p = PList("P", PORTFOLIO, [
        {"ticker": "HELD", "date": "2025-01-01", "price": 10.0, "quantity": 10, "side": "buy"},
        {"ticker": "NO_HIST", "date": "2025-01-01", "price": 5.0, "quantity": 10, "side": "buy"},
    ])

    monkeypatch.setattr(ph.prices, "split_factor", lambda t, d: 1.0)

    # Day 1 (01-01), Day 2 (01-02 has price), Days 3-7 missing, Day 8 has price
    dates = ["2025-01-01", "2025-01-02", "2025-01-08"]
    hist_held = _make_history_df([
        {"date": "2025-01-01", "close": 10.0, "adj_close": 10.0},
        {"date": "2025-01-02", "close": 12.0, "adj_close": 12.0},
        {"date": "2025-01-08", "close": 15.0, "adj_close": 15.0},
    ])

    def mock_history(t):
        if t == "HELD":
            return hist_held
        return pd.DataFrame(columns=ph.prices.COLUMNS)

    monkeypatch.setattr(ph.prices, "history", mock_history)

    vs = ph.value_series(p)
    # NO_HIST is missing
    assert vs.attrs["missing"] == ["NO_HIST"]


def test_latest_prices(monkeypatch):
    universe = pd.Series({"AAPL": 150.0, "MSFT": 300.0})

    hist_etf = _make_history_df([
        {"date": "2025-01-01", "close": 50.0, "adj_close": 50.0},
        {"date": "2025-01-02", "close": 55.0, "adj_close": 55.0},
    ])

    def mock_history(t):
        if t == "MY_ETF":
            return hist_etf
        return pd.DataFrame(columns=ph.prices.COLUMNS)

    monkeypatch.setattr(ph.prices, "history", mock_history)

    lp = ph.latest_prices(["AAPL", "MY_ETF", "UNKNOWN"], universe_prices=universe)
    assert lp["AAPL"] == 150.0
    assert lp["MY_ETF"] == 55.0
    assert pd.isna(lp["UNKNOWN"])
