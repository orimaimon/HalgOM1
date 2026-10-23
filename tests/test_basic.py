import datetime as dt
import logging
import numpy as np
import pandas as pd
import pytest

from screener.data import basic, guard
from screener.fields import provider_columns


def test_yahoo_wins_over_nasdaq_and_fills_missing():
    # AAPL in both, Yahoo price 180, Nasdaq price 170 -> Yahoo wins (180)
    # MSFT in both, Yahoo pe NaN, Nasdaq pe 30 -> Nasdaq fills (30)
    # NVDA only in Nasdaq, Yahoo missing -> Nasdaq price 120
    nasdaq_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 170.0, "pe": 25.0, "sector": "Technology"},
        {"ticker": "MSFT", "price": 400.0, "pe": 30.0, "sector": "Technology"},
        {"ticker": "NVDA", "price": 120.0, "pe": 40.0, "sector": "Technology"},
    ])

    yahoo_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 180.0, "pe": 28.0},
        {"ticker": "MSFT", "price": 405.0, "pe": np.nan},
    ])

    res = basic.combine(nasdaq_df, yahoo_df)

    res_indexed = res.set_index("ticker")
    assert res_indexed.loc["AAPL", "price"] == 180.0
    assert res_indexed.loc["AAPL", "pe"] == 28.0
    assert res_indexed.loc["MSFT", "price"] == 405.0
    assert res_indexed.loc["MSFT", "pe"] == 30.0
    assert res_indexed.loc["NVDA", "price"] == 120.0
    assert res_indexed.loc["NVDA", "pe"] == 40.0


def test_sector_from_snapshot_and_fallback():
    # AAPL in snapshot with sector "Technology Services", Nasdaq has "Technology" -> Snapshot wins ("Technology Services")
    # MSFT not in snapshot -> Nasdaq's sector ("Technology") used as fallback
    nasdaq_df = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology"},
        {"ticker": "MSFT", "sector": "Technology"},
    ])

    snap_df = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology Services"},
    ])

    res = basic.combine(nasdaq_df, snapshot=snap_df, snapshot_day=dt.date(2026, 10, 20))
    res_indexed = res.set_index("ticker")

    assert res_indexed.loc["AAPL", "sector"] == "Technology Services"
    assert res_indexed.loc["MSFT", "sector"] == "Technology"


def test_snapshot_only_ticker_dropped_ipo_year_kept_all_provider_columns_exist():
    # TSLA only in snapshot -> should be dropped
    # AAPL in Nasdaq with ipo_year 1980 -> ipo_year kept
    nasdaq_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "ipo_year": 1980.0},
    ])

    snap_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 140.0, "sector": "Technology Services"},
        {"ticker": "TSLA", "price": 200.0, "sector": "Consumer Cyclical"},
    ])

    res = basic.combine(nasdaq_df, snapshot=snap_df)

    tickers = set(res["ticker"])
    assert "AAPL" in tickers
    assert "TSLA" not in tickers

    assert "ipo_year" in res.columns
    assert res.set_index("ticker").loc["AAPL", "ipo_year"] == 1980.0

    for col in provider_columns():
        assert col in res.columns


def test_attrs_source_metadata():
    nasdaq_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "sector": "Technology"},
    ])
    snap_df = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology Services", "rsi": 55.0},
    ])

    snap_day = dt.date(2026, 10, 23)
    res = basic.combine(nasdaq_df, snapshot=snap_df, snapshot_day=snap_day)

    meta = res.attrs["source"]
    assert meta["provider"] == "basic"
    assert meta["basic"] is True
    assert meta["snapshot_day"] == "2026-10-23"
    assert "rsi" in meta["stale"]
    assert "sector" in meta["stale"]

    # Missing columns check: e.g. 'sma20' is non-FRESH and not in snapshot or nasdaq
    assert "sma20" in meta["missing"]


def test_snapshot_and_yahoo_none_cases():
    nasdaq_df = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "sector": "Technology"},
    ])

    res = basic.combine(nasdaq_df, yahoo_df=None, snapshot=None, snapshot_day=None)

    assert len(res) == 1
    assert res.iloc[0]["ticker"] == "AAPL"
    assert res.iloc[0]["price"] == 150.0
    assert res.attrs["source"]["snapshot_day"] is None
    assert res.attrs["source"]["stale"] == []


def test_fetch_basic_behavior(monkeypatch, caplog):
    class DummyStore:
        def snapshot_dates(self):
            return [dt.date(2026, 10, 22)]

        def load_snapshot(self, day):
            return pd.DataFrame([{"ticker": "AAPL", "sector": "Technology Services"}])

    dummy_store = DummyStore()

    nasdaq_dummy = pd.DataFrame([{"ticker": "AAPL", "price": 170.0, "sector": "Technology"}])

    def mock_guard_call(provider, fn, *args, **kwargs):
        if provider == "nasdaq":
            return nasdaq_dummy
        if provider == "yahoo":
            raise guard.Wait("yahoo", dt.datetime.now(), "backoff")
        raise RuntimeError(f"Unexpected provider {provider}")

    monkeypatch.setattr(guard, "call", mock_guard_call)

    with caplog.at_level(logging.WARNING):
        res = basic.fetch_basic(dummy_store)

    assert len(res) == 1
    assert res.iloc[0]["ticker"] == "AAPL"
    assert res.attrs["source"]["basic"] is True
    assert "Yahoo screener failed in basic mode" in caplog.text

    # Test Nasdaq raising exception propagates
    def mock_guard_call_nasdaq_fails(provider, fn, *args, **kwargs):
        if provider == "nasdaq":
            raise guard.Wait("nasdaq", dt.datetime.now(), "block")
        return None

    monkeypatch.setattr(guard, "call", mock_guard_call_nasdaq_fails)

    with pytest.raises(guard.Wait) as exc_info:
        basic.fetch_basic(dummy_store)
    assert exc_info.value.provider == "nasdaq"
