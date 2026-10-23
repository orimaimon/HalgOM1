"""Tests for screener.home: "הבוקר שלי" home page data module."""
import datetime as dt
import sys
import pandas as pd
import pytest

from screener.data.store import Store
from screener import home, portfolio
from screener.fields import COLUMNS

DAY0 = dt.date(2026, 3, 10)
PREV_DAY = dt.date(2026, 3, 9)


def make_snapshot(rows: dict[str, dict]) -> pd.DataFrame:
    """Helper to build a synthetic universe / snapshot DataFrame from a row dictionary."""
    tickers = list(rows)
    n = len(tickers)
    data = {k: [10.0] * n for k, c in COLUMNS.items() if c.source == "provider" and c.kind not in ("text", "date")}
    df = pd.DataFrame(data)
    df["ticker"] = tickers
    df["company"] = [f"Co {t}" for t in tickers]
    df["exchange"] = "NASDAQ"
    df["sector"] = "Technology"
    df["industry"] = "Software"
    df["country"] = "United States"
    df["sec_type"] = "stock"
    df["market_cap"], df["avg_volume"] = 1e9, 1e6

    for t, vals in rows.items():
        idx = df[df["ticker"] == t].index[0]
        for col, val in vals.items():
            df.loc[idx, col] = val

    return df


def test_new_in_screens(tmp_path):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    # Yesterday: AAPL and MSFT pass "pe < 15"
    prev_rows = {
        "AAPL": {"pe": 10.0, "price": 150.0},
        "MSFT": {"pe": 12.0, "price": 300.0},
        "GOOG": {"pe": 20.0, "price": 100.0},
    }
    make_snapshot(prev_rows).to_parquet(store.snapshot_path(PREV_DAY))

    # Today: AAPL, MSFT, and GOOG pass "pe < 15" (GOOG PE changed to 14.0 -> total=3, new=1: GOOG)
    today_rows = {
        "AAPL": {"pe": 10.0, "price": 150.0},
        "MSFT": {"pe": 12.0, "price": 300.0},
        "GOOG": {"pe": 14.0, "price": 105.0},
        "AMZN": {"pe": 25.0, "price": 200.0},
    }
    today_univ = make_snapshot(today_rows)
    make_snapshot(today_rows).to_parquet(store.snapshot_path(DAY0))

    screens = {
        "Builtin Screen": {"pe": "Under 15"},
        "★ Saved Screen": {"pe": "Under 15"},
    }

    m = home.morning(today_univ, store, DAY0, lists={}, screens=screens)
    df = m.new_in_screens

    # Saved screen comes first
    assert list(df["screen"]) == ["★ Saved Screen", "Builtin Screen"]
    for row in df.itertuples():
        assert row.total == 3
        assert row.new == 1

    # Test no previous snapshot day
    empty_store = Store(tmp_path / "empty_store")
    m_no_prev = home.morning(today_univ, empty_store, DAY0, lists={}, screens=screens)
    for row in m_no_prev.new_in_screens.itertuples():
        assert row.total == 3
        assert row.new is None


def test_movers(tmp_path):
    store = Store(tmp_path)

    # Universe with various tickers & changes
    univ = make_snapshot({
        "AAPL": {"change": 5.0, "price": 150.0, "company": "Apple"},
        "TSLA": {"change": -8.0, "price": 200.0, "company": "Tesla"},   # -8% beats +5%
        "MSFT": {"change": 2.0, "price": 300.0, "company": "Microsoft"},
        "NVDA": {"change": 7.0, "price": 120.0, "company": "NVIDIA"},
        "AMZN": {"change": -3.0, "price": 180.0, "company": "Amazon"},
        "META": {"change": 1.0, "price": 500.0, "company": "Meta"},
    })

    # Watchlist 1: AAPL, TSLA, GONE (GONE not in universe)
    w1 = portfolio.PList("List A", portfolio.WATCHLIST, [
        {"ticker": "AAPL", "date": "2026-01-01", "price": 140.0, "note": ""},
        {"ticker": "TSLA", "date": "2026-01-01", "price": 210.0, "note": ""},
        {"ticker": "GONE", "date": "2026-01-01", "price": 50.0, "note": ""},
    ])

    # Watchlist 2: TSLA, MSFT, NVDA, AMZN, META
    w2 = portfolio.PList("List B", portfolio.WATCHLIST, [
        {"ticker": "TSLA", "date": "2026-01-01", "price": 210.0, "note": ""},
        {"ticker": "MSFT", "date": "2026-01-01", "price": 290.0, "note": ""},
        {"ticker": "NVDA", "date": "2026-01-01", "price": 110.0, "note": ""},
        {"ticker": "AMZN", "date": "2026-01-01", "price": 185.0, "note": ""},
        {"ticker": "META", "date": "2026-01-01", "price": 490.0, "note": ""},
    ])

    lists = {"List A": w1, "List B": w2}

    m = home.morning(univ, store, DAY0, lists=lists, screens={})
    movers = m.movers

    assert len(movers) == 5
    assert "GONE" not in list(movers["ticker"])

    # Top mover is TSLA (-8%), followed by NVDA (7%), AAPL (5%), AMZN (-3%), MSFT (2%)
    assert list(movers["ticker"]) == ["TSLA", "NVDA", "AAPL", "AMZN", "MSFT"]
    assert movers.iloc[0]["lists"] == ["List A", "List B"]
    assert movers.iloc[2]["lists"] == ["List A"]


def test_earnings(tmp_path):
    store = Store(tmp_path)

    # Earnings dates:
    # AAPL: today + 3 days (in range)
    # MSFT: today + 10 days (in range)
    # NVDA: today + 20 days (out of 14 day range)
    # AMZN: today + 5 days (in range, but fully sold position)
    # TSLA: today + 2 days (in range, but watchlist only)
    univ = make_snapshot({
        "AAPL": {"earnings_date": "2026-03-13", "company": "Apple"},
        "MSFT": {"earnings_date": "2026-03-20", "company": "Microsoft"},
        "NVDA": {"earnings_date": "2026-03-30", "company": "NVIDIA"},
        "AMZN": {"earnings_date": "2026-03-15", "company": "Amazon"},
        "TSLA": {"earnings_date": "2026-03-12", "company": "Tesla"},
    })

    # Portfolio with AAPL (held), MSFT (held), NVDA (held), AMZN (buy then sell -> qty=0)
    p1 = portfolio.PList("My Portfolio", portfolio.PORTFOLIO, [
        {"ticker": "AAPL", "date": "2026-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "MSFT", "date": "2026-01-01", "price": 300.0, "quantity": 5, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "NVDA", "date": "2026-01-01", "price": 100.0, "quantity": 20, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "AMZN", "date": "2026-01-01", "price": 180.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "AMZN", "date": "2026-02-01", "price": 190.0, "quantity": 10, "side": "sell", "fees": 0, "note": ""},
    ])

    # Watchlist with TSLA
    w1 = portfolio.PList("Watchlist", portfolio.WATCHLIST, [
        {"ticker": "TSLA", "date": "2026-01-01", "price": 200.0, "note": ""},
    ])

    lists = {"My Portfolio": p1, "Watchlist": w1}

    m = home.morning(univ, store, DAY0, lists=lists, screens={})
    earn = m.earnings

    # Only AAPL (in 3 days) and MSFT (in 10 days) should be included, sorted by date
    assert list(earn["ticker"]) == ["AAPL", "MSFT"]
    assert list(earn["days"]) == [3, 10]
    assert list(earn["earnings_date"]) == [dt.date(2026, 3, 13), dt.date(2026, 3, 20)]


def test_exception_handling(tmp_path, monkeypatch):
    store = Store(tmp_path)
    univ = make_snapshot({"AAPL": {"change": 2.0, "price": 150.0}})

    def bad_screen(*args, **kwargs):
        raise RuntimeError("Simulated failure in filters.screen")

    monkeypatch.setattr(home.filters, "screen", bad_screen)

    m = home.morning(univ, store, DAY0, lists={}, screens={"Test": {}})

    assert m.new_in_screens.empty
    assert len(m.errors) == 1
    assert "new_in_screens" in m.errors[0]
    # Rest of morning result is still valid/empty without crashing
    assert isinstance(m.movers, pd.DataFrame)
    assert isinstance(m.earnings, pd.DataFrame)


def test_no_streamlit_and_no_network(tmp_path, monkeypatch):
    # Verify screener.home module does not import or use streamlit
    assert "streamlit" not in vars(home)
    assert "st" not in vars(home)

    def fail_net(*args, **kwargs):
        raise RuntimeError("No network allowed")

    monkeypatch.setattr("requests.get", fail_net, raising=False)
    monkeypatch.setattr("requests.post", fail_net, raising=False)
    try:
        import yfinance
        monkeypatch.setattr(yfinance, "download", fail_net, raising=False)
        monkeypatch.setattr(yfinance, "Ticker", fail_net, raising=False)
    except ImportError:
        pass

    store = Store(tmp_path)
    univ = make_snapshot({"AAPL": {"change": 1.0, "price": 150.0}})

    m = home.morning(univ, store, DAY0, lists={}, screens={})
    assert isinstance(m, home.Morning)
    assert m.errors == []
