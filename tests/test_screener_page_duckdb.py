"""
Tests for DuckDB predicate pushdown and universe snapshot loading in screener/ui/screener_page.py.
"""
from __future__ import annotations

import datetime as dt
import json
import numpy as np
import pandas as pd
import pytest

from screener.data.store import Store
from screener.filters import ANY, FILTERS, hide_junk
from screener.ui import screener_page as SP
from screener.ui.common import ss


@pytest.fixture
def sample_store(tmp_path):
    store = Store(tmp_path)
    data = {
        "ticker": ["AAPL", "MSFT", "PENNY", "LOWVOL", "TINYCAP", "SPAC1"],
        "company": [
            "Apple Inc.",
            "Microsoft Corp.",
            "Penny Stock Inc.",
            "Low Vol Co.",
            "Tiny Cap Ltd.",
            "Acquisition Corp.",
        ],
        "exchange": ["NASDAQ", "NASDAQ", "NASDAQ", "NASDAQ", "NASDAQ", "NASDAQ"],
        "price": [150.0, 300.0, 0.5, 10.0, 20.0, 10.0],
        "avg_volume": [1e6, 2e6, 100e3, 10e3, 100e3, 100e3],
        "market_cap": [2e12, 2.5e12, 10e6, 1e9, 10e6, 100e6],
        "industry": ["Tech", "Tech", "Tech", "Tech", "Tech", "Financial Conglomerates"],
        "sector": ["Technology", "Technology", "Technology", "Technology", "Technology", "Financials"],
        "country": ["USA", "USA", "USA", "USA", "USA", "USA"],
        "close_w": [150.0, 300.0, 0.5, 10.0, 20.0, 10.0],
        "sma50_w": [140.0, 290.0, 0.4, 9.0, 19.0, 9.5],
        "sma20": [145.0, 295.0, 0.4, 9.5, 19.5, 9.8],
        "sma50": [140.0, 290.0, 0.4, 9.0, 19.0, 9.5],
        "sma150": [135.0, 280.0, 0.3, 8.5, 18.5, 9.2],
        "sma200": [130.0, 270.0, 0.3, 8.0, 18.0, 9.0],
        "high52": [160.0, 310.0, 1.0, 12.0, 25.0, 11.0],
        "low52": [120.0, 250.0, 0.2, 8.0, 15.0, 9.5],
        "high_1m": [155.0, 305.0, 0.6, 11.0, 22.0, 10.5],
        "low_1m": [145.0, 295.0, 0.4, 9.0, 18.0, 9.8],
        "high_5d": [152.0, 302.0, 0.55, 10.5, 21.0, 10.2],
        "low_5d": [148.0, 298.0, 0.45, 9.8, 19.0, 9.9],
        "eps_next_fy": [7.0, 12.0, 0.05, 0.5, 1.0, 0.2],
        "target": [170.0, 350.0, 0.8, 12.0, 25.0, 11.0],
    }
    df_raw = pd.DataFrame(data)
    source_meta = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    df_raw.attrs["source"] = source_meta
    now = dt.datetime.now().astimezone()
    store.save(df_raw, now)
    return store


def test_resolve_universe_duckdb_parquet(sample_store):
    res = SP.resolve_universe(sample_store, hide_junk=True)
    assert res is not None
    df, fetched_at, meta = res

    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    assert df.attrs.get("source") == meta
    assert meta.get("provider") == "tradingview"


def test_resolve_universe_without_hide_junk(sample_store):
    res = SP.resolve_universe(sample_store, hide_junk=False)
    assert res is not None
    df, fetched_at, meta = res

    assert len(df) == 6
    assert "PENNY" in list(df["ticker"])


def test_resolve_universe_fallback_on_exception(sample_store, monkeypatch):
    def mock_load_filtered_universe(*args, **kwargs):
        raise RuntimeError("Simulated DuckDB error")

    monkeypatch.setattr(SP, "load_filtered_universe", mock_load_filtered_universe)

    res = SP.resolve_universe(sample_store, hide_junk=True)
    assert res is not None
    df, fetched_at, meta = res

    assert set(df["ticker"]) == {"AAPL", "MSFT"}


def test_resolve_universe_fallback_when_no_parquet(tmp_path):
    store = Store(tmp_path)
    res = SP.resolve_universe(store)
    assert res is None


def test_resolve_universe_parity_with_standard_loader(sample_store):
    std_res = sample_store.load_latest()
    assert std_res is not None
    df_std, ts_std, meta_std = std_res
    df_std_junk = hide_junk(df_std)
    df_std_junk.attrs["source"] = meta_std

    duck_res = SP.resolve_universe(sample_store, hide_junk=True)
    assert duck_res is not None
    df_duck, ts_duck, meta_duck = duck_res

    assert meta_duck == meta_std
    assert df_duck.attrs.get("source") == df_std_junk.attrs.get("source")
    pd.testing.assert_frame_equal(
        df_std_junk.reset_index(drop=True),
        df_duck.reset_index(drop=True),
    )


def test_resolve_universe_with_custom_where(sample_store):
    res = SP.resolve_universe(sample_store, hide_junk=True, where="price >= 200.0")
    assert res is not None
    df, fetched_at, meta = res

    assert list(df["ticker"]) == ["MSFT"]


def test_filter_panel_default_resolves_universe(sample_store, monkeypatch):
    ss["lang"] = "he"
    ss["hide_junk"] = True
    for f in FILTERS:
        ss[f"f_{f.key}"] = ANY

    monkeypatch.setattr(SP, "get_store", lambda: sample_store)
    called = False

    def mock_resolve_universe(*args, **kwargs):
        nonlocal called
        called = True
        df = sample_store.load_latest()[0]
        for f in FILTERS:
            if f.column not in df.columns:
                df[f.column] = np.nan
        return (df, dt.datetime.now(), {})

    monkeypatch.setattr(SP, "resolve_universe", mock_resolve_universe)

    SP.filter_panel()
    assert called
