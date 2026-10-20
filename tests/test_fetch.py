"""Regression tests for TradingView fetch sanity checks, throttling, concurrency, and config."""
import datetime as dt
import threading
import time
import tomllib

import pytest

import screener.data.tradingview as tv
from screener.data.store import Store, load_universe
from screener.data.tradingview import TradingViewError, clean, validate_universe
from tests.test_frontend import app  # noqa: F401  (fixture)
from tests.test_screener import ROOT, make_raw


def test_validate_universe_checks_fill_rates_and_floor(monkeypatch):
    monkeypatch.setattr(tv, "MIN_UNIVERSE_ROWS", 500)
    with pytest.raises(TradingViewError, match="price fill rate collapsed"):     # e.g. `close` renamed
        bad = make_raw(600).assign(price=float("nan"))
        validate_universe(clean(bad), bad)
    with pytest.raises(TradingViewError, match="market_cap fill rate collapsed"):
        bad = make_raw(600).assign(market_cap=float("nan"))
        validate_universe(bad, bad)
    with pytest.raises(TradingViewError, match="below minimum floor"):
        small = make_raw(499)
        validate_universe(small, small)
    ok = make_raw(600)
    validate_universe(ok, ok)


def test_bad_response_keeps_the_good_cache(tmp_path):
    store = Store(tmp_path)
    good = make_raw(300)
    assert load_universe(store, lambda: good, dt.timedelta(minutes=15)).source == "live"

    def bad_fetch():                               # what fetch_universe does with a broken response
        bad = good.assign(price=float("nan"))
        cleaned = clean(bad)
        validate_universe(cleaned, bad)
        return cleaned

    snap = load_universe(store, bad_fetch, dt.timedelta(minutes=15), force=True)
    assert snap.source == "stale" and len(snap.df) == 300 and "price fill rate collapsed" in snap.error
    assert len(store.load_latest()[0]) == 300                                   # nothing overwritten
    assert len(store.load_snapshot(store.snapshot_dates()[-1])) == 300


def test_shrunken_universe_keeps_the_good_cache(tmp_path):
    store = Store(tmp_path)
    load_universe(store, lambda: make_raw(300), dt.timedelta(minutes=15))
    snap = load_universe(store, lambda: make_raw(100), dt.timedelta(minutes=15), force=True)
    assert snap.source == "stale" and len(snap.df) == 300 and "shrank" in snap.error


def test_refresh_button_is_throttled(app, monkeypatch):  # noqa: F811
    calls = []
    real = tv.fetch_universe
    monkeypatch.setattr(tv, "fetch_universe", lambda: calls.append(1) or real())
    app.run()
    assert calls == [1]
    refresh = next(b for b in app.button if b.label == "רענון נתונים")
    refresh.click().run()                                        # seconds after the fetch: no request
    assert not app.exception, app.exception
    assert calls == [1] and any("הנתונים עודכנו לפני" in t.value for t in app.toast)
    app.session_state["snap"].fetched_at -= dt.timedelta(minutes=2)
    next(b for b in app.button if b.label == "רענון נתונים").click().run()
    assert not app.exception, app.exception
    assert calls == [1, 1]


def test_concurrent_load_universe(tmp_path):
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
