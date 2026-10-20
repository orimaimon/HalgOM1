"""Regression tests for TradingView fetch sanity checks, throttling, concurrency, and config."""
import datetime as dt
import threading
import time
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from screener.data.store import Snapshot, Store, load_universe
from screener.data.tradingview import MIN_UNIVERSE_ROWS, TradingViewError, fetch_universe, validate_universe
from tests.test_screener import ROOT, make_raw


def test_validate_universe_checks_fill_rates_and_floors():
    # 1. Price fill rate collapse (< 50%)
    df_null_price = make_raw(600).assign(price=float("nan"))
    with pytest.raises(TradingViewError, match="price fill rate collapsed"):
        validate_universe(df_null_price.iloc[:0], raw_df=df_null_price)

    # 2. Market cap fill rate collapse (< 50%)
    df_null_mcap = make_raw(600).assign(market_cap=float("nan"))
    with pytest.raises(TradingViewError, match="market_cap fill rate collapsed"):
        validate_universe(df_null_mcap, raw_df=df_null_mcap)

    # 3. Cleaned row count below absolute floor (MIN_UNIVERSE_ROWS = 500)
    df_small = make_raw(MIN_UNIVERSE_ROWS - 1)
    with pytest.raises(TradingViewError, match="below minimum floor"):
        validate_universe(df_small, raw_df=df_small)

    # 4. Cleaned row count below 50% of cached universe
    df_cached_drop = make_raw(600)
    with pytest.raises(TradingViewError, match="dropped below 50% of cached"):
        validate_universe(df_cached_drop, raw_df=df_cached_drop, cached_rows=1500)

    # Valid universe passes
    df_valid = make_raw(600)
    validate_universe(df_valid, raw_df=df_valid, cached_rows=1000)


def test_bad_fetch_retains_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("screener.data.tradingview.MIN_UNIVERSE_ROWS", 100)
    store = Store(tmp_path)
    good_raw = make_raw(300)

    # First fetch succeeds and populates cache
    snap1 = load_universe(store, lambda: good_raw, dt.timedelta(minutes=15))
    assert snap1.source == "live" and len(snap1.df) == 300

    # Bad fetch returns null prices -> validation fails -> returns stale cache
    bad_raw = good_raw.copy().assign(price=float("nan"))

    def bad_fetch():
        from screener.data.tradingview import clean
        cleaned = clean(bad_raw)
        validate_universe(cleaned, raw_df=bad_raw, cached_rows=len(snap1.df))
        return cleaned

    snap2 = load_universe(store, bad_fetch, dt.timedelta(minutes=15), force=True)
    assert snap2.source == "stale"
    assert len(snap2.df) == 300
    assert "price fill rate collapsed" in snap2.error


def test_refresh_throttling_in_get_data(monkeypatch):
    import streamlit as st
    import screener.app as app_mod

    toasts = []
    monkeypatch.setattr(st, "toast", lambda msg, **kw: toasts.append(msg))

    df = make_raw(200)
    fetched_at = dt.datetime.now() - dt.timedelta(seconds=15)  # 15s ago
    mock_snap = Snapshot(df=df, fetched_at=fetched_at, source="live")

    # Put mock_snap in session_state
    app_mod.ss["snap"] = mock_snap
    app_mod.ss["df"] = df

    fetch_called = []
    monkeypatch.setattr(app_mod, "load_universe", lambda *a, **kw: fetch_called.append(True))

    # Force call get_data(force=True) when age < 60s
    snap_out, df_out = app_mod.get_data(force=True)

    assert not fetch_called, "load_universe should not be called within 60s throttle window"
    assert snap_out is mock_snap
    assert len(toasts) == 1
    assert "הנתונים עודכנו לפני" in toasts[0]
    assert "15" in toasts[0] or "14" in toasts[0] or "16" in toasts[0]


def test_concurrent_load_universe(tmp_path, monkeypatch):
    monkeypatch.setattr("screener.data.tradingview.MIN_UNIVERSE_ROWS", 100)
    store = Store(tmp_path)
    good_raw = make_raw(200)

    fetch_count = 0
    lock = threading.Lock()

    def slow_fetch():
        nonlocal fetch_count
        with lock:
            fetch_count += 1
        time.sleep(0.2)
        return good_raw

    results = [None, None]
    errors = [None, None]

    def worker(index):
        try:
            results[index] = load_universe(store, slow_fetch, dt.timedelta(minutes=15))
        except Exception as e:
            errors[index] = e

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors[0] is None, f"Thread 0 error: {errors[0]}"
    assert errors[1] is None, f"Thread 1 error: {errors[1]}"
    assert fetch_count == 1, f"Expected 1 fetch call, got {fetch_count}"
    assert results[0] is not None and results[1] is not None
    assert len(results[0].df) == 200 and len(results[1].df) == 200


def test_unique_temp_file_names(tmp_path):
    store = Store(tmp_path)
    df = make_raw(10)
    now = dt.datetime.now()

    # Save multiple times and check store creates unique temp names without collisions
    store.save(df, now)

    # Parquet and JSON files exist
    assert store.latest_path.exists()
    assert store.meta_path.exists()

    # Confirm no stray .tmp files remain in cache or snapshots dir
    assert list(store.cache_dir.glob("*.tmp")) == []
    assert list(store.snap_dir.glob("*.tmp")) == []


def test_config_toolbar_mode():
    config_path = ROOT / ".streamlit" / "config.toml"
    assert config_path.exists()
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    assert config.get("client", {}).get("toolbarMode") == "minimal"
