"""Regression tests for TradingView fetch sanity checks, throttling, concurrency, and config."""
import datetime as dt
import threading
import time
import tomllib

import pytest

import screener.data.tradingview as tv
from screener.data.store import Snapshot, Store, load_universe
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


def test_market_fetch_failure_not_cached(monkeypatch):
    from screener import app as app_module
    from screener.data.tradingview import TradingViewError

    calls = 0

    def mock_fetch_market():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TradingViewError("network down")
        # Return valid market DataFrame
        import pandas as pd
        return pd.DataFrame({"close": [100.0], "change": [1.0], "sma20": [95.0], "sma50": [90.0],
                            "sma200": [85.0], "adx": [25.0], "di_plus": [20.0], "di_minus": [15.0]},
                           index=["SPY"])

    monkeypatch.setattr(app_module, "fetch_market", mock_fetch_market)

    # Clear market cache before testing
    app_module._market.clear()

    # First attempt fails -> returns (None, None)
    m1, r1 = app_module._market()
    assert m1 is None and r1 is None
    assert calls == 1

    # Second attempt succeeds -> must retry fetch instead of serving cached (None, None)
    m2, r2 = app_module._market()
    assert m2 is not None
    assert calls == 2


def test_unreadable_latest_falls_back_to_snapshot(tmp_path):
    store = Store(tmp_path)
    good = make_raw(100)
    save_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    store.save(good, save_time)

    # Corrupt latest.parquet
    store.latest_path.write_bytes(b"corrupted parquet file")

    def failing_fetch():
        raise TradingViewError("network error")

    # Offline load_universe should fall back to snapshot and return flagged stale
    snap = load_universe(store, failing_fetch, dt.timedelta(minutes=15))
    assert snap.source == "stale"
    assert len(snap.df) == 100
    assert "network error" in snap.error


def test_timeout_is_passed_only_when_asked(tmp_path):
    store = Store(tmp_path)
    good = make_raw(100)
    timeouts = []

    def fetch(timeout=60):
        timeouts.append(timeout)
        return good

    load_universe(store, fetch, dt.timedelta(minutes=15))                        # normal: the provider's own
    load_universe(store, fetch, dt.timedelta(minutes=15), force=True, timeout=20)  # the app's offline retry
    assert timeouts == [60, 20]


def test_fetch_time_is_shown_in_local_time(tmp_path):
    """Stored tz-aware, so the header ("שעת הנתונים") must still show the PC's clock, not UTC."""
    store = Store(tmp_path)
    snap = load_universe(store, lambda: make_raw(100), dt.timedelta(minutes=15))
    local_now = dt.datetime.now()
    for ts in (snap.fetched_at, store.load_latest()[1]):
        assert ts.tzinfo is not None
        assert abs(ts.replace(tzinfo=None) - local_now) < dt.timedelta(minutes=1)


def test_validate_universe_logs_empty_provider_columns(caplog, monkeypatch):
    monkeypatch.setattr(tv, "MIN_UNIVERSE_ROWS", 100)
    ok = make_raw(100)
    # Set one provider column ('pe') to all NaNs
    bad_pe = ok.assign(pe=float("nan"))

    with caplog.at_level("INFO"):
        validate_universe(bad_pe, bad_pe)

    # Should log warning for empty provider column 'pe'
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("provider column pe is completely empty" in msg for msg in warnings)


def test_clock_skew_and_negative_age(tmp_path):
    from screener.data.store import market_date
    store = Store(tmp_path)
    good = make_raw(100)

    # 1. Test future timestamp (negative age) is treated as stale and re-fetches
    future_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)
    store.save(good, future_time)

    # Future cache has negative age
    snap_future = Snapshot(good, future_time, "cache")
    assert snap_future.age < dt.timedelta(0)

    # load_universe should treat negative age as stale and run fetch
    fetched = False
    def mock_fetch():
        nonlocal fetched
        fetched = True
        return good

    snap = load_universe(store, mock_fetch, dt.timedelta(minutes=15))
    assert fetched is True
    assert snap.source == "live"

    # 2. Test reading old naive timestamp vs new tz-aware timestamp
    naive_time = dt.datetime.now() - dt.timedelta(minutes=5)
    snap_naive = Snapshot(good, naive_time, "cache")
    assert dt.timedelta(0) < snap_naive.age < dt.timedelta(minutes=10)

    # 3. Test market_date works for both naive and aware datetimes
    d1 = market_date(dt.datetime(2026, 10, 25, 14, 0, 0))
    d2 = market_date(dt.datetime(2026, 10, 25, 14, 0, 0, tzinfo=dt.timezone.utc))
    assert isinstance(d1, dt.date)
    assert isinstance(d2, dt.date)


def test_config_toolbar_mode():
    config_path = ROOT / ".streamlit" / "config.toml"
    assert config_path.exists()
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    assert config.get("client", {}).get("toolbarMode") == "minimal"
