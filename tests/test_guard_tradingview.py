"""Tests for TradingView request guard integration."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from screener import export, lifecycle
from screener import portfolio as PF
from screener import presets as P
from screener.data import guard, store as store_mod, tradingview as tv
from screener.data.store import Store
from screener.data.tradingview import clean
from screener.ui.common import _market, get_data
from tests.test_screener import ROOT, make_raw


class HTTP429Error(Exception):
    def __init__(self, msg: str = "HTTP 429 Too Many Requests"):
        super().__init__(msg)
        self.status_code = 429


class FakeClock:
    def __init__(self, start: dt.datetime | None = None):
        self.now_dt = start or dt.datetime(2026, 10, 24, 12, 0, 0, tzinfo=dt.timezone.utc)

    def __call__(self) -> dt.datetime:
        return self.now_dt

    def advance(self, **kwargs):
        self.now_dt += dt.timedelta(**kwargs)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Clean data directory and state isolated from real files."""
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PRESETS_PATH", tmp_path / "presets.json")
    monkeypatch.setattr(PF, "PORTFOLIOS_PATH", tmp_path / "portfolios.json")
    monkeypatch.setattr(export, "EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(lifecycle, "ensure_idle_watchdog_started", lambda: None)
    st.cache_data.clear()
    st.cache_resource.clear()
    guard._DEFAULT_GUARD = guard.Guard()
    return tmp_path


def _save_sample_cache(data_dir: Path, fetched_at: dt.datetime | None = None) -> pd.DataFrame:
    raw = clean(make_raw(50).assign(is_primary=True, currency="USD"))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    ts = fetched_at or (dt.datetime.now().astimezone() - dt.timedelta(hours=2))
    Store(data_dir).save(raw, ts)
    return raw


def _sample_market() -> pd.DataFrame:
    return pd.DataFrame({
        "close": [500.0, 400.0, 80.0, 15.0],
        "change": 0.5,
        "sma20": [490.0, 390.0, 79.0, 16.0],
        "sma50": 480.0,
        "sma200": [450.0, 350.0, 78.0, 17.0],
        "adx": [30.0, 20.0, 15.0, 10.0],
        "di_plus": 25.0,
        "di_minus": 15.0
    }, index=["SPY", "QQQ", "HYG", "VIX"])


def test_block_no_retry(clean_env, monkeypatch):
    """Block, no retry: tv.fetch_universe raises HTTP 429 on first call -> stale snapshot from cache.
    Next get_data(force=True) and a new AppTest run (new tab) within 60 min do NOT call fetch_universe
    again (count calls == 1), and page shows the Hebrew wait message."""
    _save_sample_cache(clean_env)
    market = _sample_market()
    monkeypatch.setattr(tv, "fetch_market", lambda: market)

    calls = 0

    def failing_fetch(**kwargs):
        nonlocal calls
        calls += 1
        raise HTTP429Error("429 Too Many Requests")

    monkeypatch.setattr(tv, "fetch_universe", failing_fetch)

    # First call: triggers 429 -> stale snapshot from cache
    snap, df = get_data(force=True)
    assert calls == 1
    assert snap.source == "stale"
    assert "TradingView חסם זמנית — מוצגים נתונים שמורים" in (snap.error or "")

    # Next get_data(force=True): guard blocks call, no retry
    snap2, df2 = get_data(force=True)
    assert calls == 1
    assert snap2.source == "stale"

    # New AppTest run (a new tab): must NOT call fetch_universe again
    at = AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert calls == 1

    # Page shows the Hebrew wait message
    wait_text = "TradingView חסם זמנית — מוצגים נתונים שמורים"
    assert any(wait_text in w.value for w in at.warning)


def test_survives_restart(clean_env, monkeypatch):
    """Survives a restart: after the block, reset the guard's in-memory state:
    still no call until the wait is over (fake clock)."""
    clock = FakeClock()
    monkeypatch.setattr(guard, "_default_now", clock)

    _save_sample_cache(clean_env, fetched_at=clock() - dt.timedelta(hours=2))
    monkeypatch.setattr(tv, "fetch_market", _sample_market)

    calls = 0

    def failing_fetch(**kwargs):
        nonlocal calls
        calls += 1
        raise HTTP429Error("429 Too Many Requests")

    monkeypatch.setattr(tv, "fetch_universe", failing_fetch)

    # Trigger block
    get_data(force=True)
    assert calls == 1

    # Simulate restart: reset guard's in-memory instance and state
    guard._DEFAULT_GUARD = guard.Guard()

    # Still within 60 minutes: guard must still block from persisted state
    clock.advance(minutes=30)
    snap, _ = get_data(force=True)
    assert calls == 1
    assert snap.source == "stale"

    # After wait is over (60 minutes past block): next call is allowed
    clock.advance(minutes=31)

    def ok_fetch(**kwargs):
        nonlocal calls
        calls += 1
        raw = clean(make_raw(50).assign(is_primary=True, currency="USD"))
        raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
        return raw

    monkeypatch.setattr(tv, "fetch_universe", ok_fetch)
    snap_ok, _ = get_data(force=True)
    assert calls == 2
    assert snap_ok.source == "live"


def test_market_block_and_last_good(clean_env, monkeypatch):
    """Market: tv.fetch_market raising 429 -> subsequent reruns do not call it again
    during the wait, and the header shows the last good market data when there was one."""
    clock = FakeClock()
    monkeypatch.setattr(guard, "_default_now", clock)

    _save_sample_cache(clean_env, fetched_at=clock() - dt.timedelta(hours=2))

    # First fetch succeeds and populates cache and last good holder
    market_df = _sample_market()
    monkeypatch.setattr(tv, "fetch_market", lambda: market_df)
    monkeypatch.setattr(tv, "fetch_universe", lambda **kw: _save_sample_cache(clean_env, fetched_at=clock() - dt.timedelta(hours=2)))

    m, regime = _market()
    assert m is not None
    assert "SPY" in m.index

    # Advance past min_interval (120 s)
    clock.advance(seconds=125)

    # Now market fails with 429
    market_calls = 0

    def failing_market():
        nonlocal market_calls
        market_calls += 1
        raise HTTP429Error("429 Too Many Requests")

    monkeypatch.setattr(tv, "fetch_market", failing_market)
    _market.clear()  # Clear cache to trigger next fetch

    # First failing call records block and falls back to last good
    m1, regime1 = _market()
    assert market_calls == 1
    assert m1 is not None
    assert "SPY" in m1.index

    # Subsequent reruns (even with min_interval passed, but within 60 min block)
    clock.advance(minutes=5)
    _market.clear()
    m2, regime2 = _market()
    assert market_calls == 1  # Not called again!
    assert m2 is not None
    assert "SPY" in m2.index

    # AppTest run shows header with SPY metric instead of 'Market data unavailable'
    at = AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert market_calls == 1
    # Check that market metrics are present (not unavailable)
    assert not any("Market data unavailable" in c.value for c in at.caption)


def test_no_cache_at_all_wait(clean_env, monkeypatch):
    """No cache at all + Wait -> the Hebrew wait message, not the internet-connection error,
    and no exception."""
    monkeypatch.setattr(tv, "fetch_market", _sample_market)

    # Pre-block tradingview in guard state so guard raises Wait immediately
    now = dt.datetime.now().astimezone()
    guard._DEFAULT_GUARD.call("tradingview", lambda: None)  # creates initial state
    with guard._STATE_LOCK:
        state = guard._load_state()
        state["tradingview"]["wait_until"] = (now + dt.timedelta(minutes=60)).isoformat()
        state["tradingview"]["wait_reason"] = "block"
        guard._save_state(state)

    calls = 0

    def should_not_be_called(**kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame()

    monkeypatch.setattr(tv, "fetch_universe", should_not_be_called)

    at = AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert calls == 0

    # Shows the Hebrew wait message, NOT the internet-connection error
    wait_text = "TradingView חסם זמנית — מוצגים נתונים שמורים"
    assert any(wait_text in err.value for err in at.error)
    assert not any("בדוק את החיבור לאינטרנט" in err.value for err in at.error)
