"""
Unit tests for screener/breadth.py
"""
import datetime as dt
import json
import re
import time
import numpy as np
import pandas as pd
import pytest

from screener import breadth
from screener import derive
from screener import signals
from screener.data.store import Store, _atomic_parquet


def test_breadth_hand_counted():
    # 5 stocks
    # S1: price 100, sma50 90 (above), sma200 80 (above), change +2.0, high52_prev 90 (high), low52_prev 50
    # S2: price 50, sma50 60 (below), sma200 40 (above), change -1.0, high52_prev 70, low52_prev 60 (low)
    # S3: price 20, sma50 20 (not above), sma200 NaN (no sma200), change 0.0, high52_prev 30, low52_prev 10
    # S4: price 10, sma50 NaN, sma200 15 (below), change NaN (no change), high52_prev 12, low52_prev 8
    # S5: price 0.5 (junk: < $1), sma50 0.4 (above), sma200 0.4 (above), change +0.1, high52_prev 0.4 (high), low52_prev 0.1
    df = pd.DataFrame([
        {"ticker": "S1", "price": 100.0, "sma50": 90.0, "sma200": 80.0, "change": 2.0, "high52_prev": 90.0, "low52_prev": 50.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "S2", "price": 50.0, "sma50": 60.0, "sma200": 40.0, "change": -1.0, "high52_prev": 70.0, "low52_prev": 60.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "S3", "price": 20.0, "sma50": 20.0, "sma200": np.nan, "change": 0.0, "high52_prev": 30.0, "low52_prev": 10.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "S4", "price": 10.0, "sma50": np.nan, "sma200": 15.0, "change": np.nan, "high52_prev": 12.0, "low52_prev": 8.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "S5", "price": 0.5, "sma50": 0.4, "sma200": 0.4, "change": 0.1, "high52_prev": 0.4, "low52_prev": 0.1, "avg_volume": 100e3, "market_cap": 100e6},
    ])

    # When junk=True (default): S5 excluded (price < $1)
    # Remaining stocks: S1, S2, S3, S4 (4 stocks)
    # sma50:
    #   valid: S1 (100 vs 90 -> above), S2 (50 vs 60 -> below), S3 (20 vs 20 -> equal/not above) -> 3 valid, 1 above (S1) -> 1/3 = 33.33333333333333%
    # sma200:
    #   valid: S1 (100 vs 80 -> above), S2 (50 vs 40 -> above), S4 (10 vs 15 -> below). S3 is NaN (left out).
    #   3 valid, 2 above (S1, S2) -> 2/3 = 66.66666666666666%
    # change:
    #   S1 (+2.0 -> advancer), S2 (-1.0 -> decliner), S3 (0.0 -> unchanged), S4 (NaN -> ignored).
    #   advancers = 1, decliners = 1, unchanged = 1
    #   ad_ratio = 1 / 1 = 1.0
    #   pct_advancing = 1 / 3 * 100 = 33.33333333333333%
    # new_highs:
    #   S1 (100 > 90 -> high) -> 1
    # new_lows:
    #   S2 (50 < 60 -> low) -> 1
    # net_highs: 1 - 1 = 0
    res_junk = breadth.breadth(df, junk=True)
    assert res_junk["stocks"] == 4
    assert res_junk["pct_above_sma50"] == pytest.approx(33.3333333)
    assert res_junk["pct_above_sma200"] == pytest.approx(66.6666666)
    assert res_junk["advancers"] == 1
    assert res_junk["decliners"] == 1
    assert res_junk["unchanged"] == 1
    assert res_junk["ad_ratio"] == pytest.approx(1.0)
    assert res_junk["pct_advancing"] == pytest.approx(33.3333333)
    assert res_junk["new_highs"] == 1
    assert res_junk["new_lows"] == 1
    assert res_junk["net_highs"] == 0

    # When junk=False: S5 included (5 stocks)
    # S5: price 0.5, sma50 0.4 (above), sma200 0.4 (above), change 0.1 (advancer), high52_prev 0.4 (high)
    # sma50 valid: S1, S2, S3, S5 (4 valid). S1, S5 above -> 2/4 = 50.0%
    # sma200 valid: S1, S2, S4, S5 (4 valid). S1, S2, S5 above -> 3/4 = 75.0%
    # advancers = 2 (S1, S5), decliners = 1 (S2), unchanged = 1 (S3) -> ad_ratio = 2.0, pct_advancing = 2/4 * 100 = 50.0%
    # new_highs = 2 (S1, S5), new_lows = 1 (S2) -> net_highs = 1
    res_no_junk = breadth.breadth(df, junk=False)
    assert res_no_junk["stocks"] == 5
    assert res_no_junk["pct_above_sma50"] == 50.0
    assert res_no_junk["pct_above_sma200"] == 75.0
    assert res_no_junk["advancers"] == 2
    assert res_no_junk["decliners"] == 1
    assert res_no_junk["ad_ratio"] == 2.0
    assert res_no_junk["pct_advancing"] == 50.0
    assert res_no_junk["new_highs"] == 2
    assert res_no_junk["new_lows"] == 1
    assert res_no_junk["net_highs"] == 1


def test_breadth_no_decliners_and_all_nan():
    # Test zero decliners -> ad_ratio is None
    df = pd.DataFrame([
        {"ticker": "A", "price": 10.0, "change": 1.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "B", "price": 20.0, "change": 0.0, "avg_volume": 100e3, "market_cap": 100e6},
    ])
    res = breadth.breadth(df, junk=True)
    assert res["advancers"] == 1
    assert res["decliners"] == 0
    assert res["ad_ratio"] is None
    assert res["pct_advancing"] == 50.0
    assert res["pct_above_sma50"] is None
    assert res["pct_above_sma200"] is None


def test_new_highs_lows_match_signals():
    df = pd.DataFrame([
        {"ticker": "T1", "price": 105.0, "high52_prev": 100.0, "low52_prev": 80.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "T2", "price": 75.0, "high52_prev": 100.0, "low52_prev": 80.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "T3", "price": 90.0, "high52_prev": 100.0, "low52_prev": 80.0, "avg_volume": 100e3, "market_cap": 100e6},
    ])
    res = breadth.breadth(df, junk=False)

    bo_fn = next(s.fn for s in signals.BREAKOUTS if s.label == "New 52-week high (breakout)")
    bd_fn = next(s.fn for s in signals.BREAKDOWNS if s.label == "New 52-week low (breakdown)")

    expected_highs = int(bo_fn(df).fillna(False).astype(bool).sum())
    expected_lows = int(bd_fn(df).fillna(False).astype(bool).sum())

    assert res["new_highs"] == expected_highs == 1
    assert res["new_lows"] == expected_lows == 1


def test_by_sector():
    df = pd.DataFrame([
        # Tech: 2 stocks. T1: price 100 > sma200 80 (100%), chg +1 (adv); T2: price 50 > sma200 40 (100%), chg +2 (adv). pct_above_sma200 = 100%, pct_advancing = 100%
        {"ticker": "T1", "sector": "Technology", "price": 100.0, "sma200": 80.0, "change": 1.0, "high52_prev": 90.0, "low52_prev": 50.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "T2", "sector": "Technology", "price": 50.0, "sma200": 40.0, "change": 2.0, "high52_prev": 60.0, "low52_prev": 30.0, "avg_volume": 100e3, "market_cap": 100e6},
        # Finance: 2 stocks. F1: price 10 < sma200 20 (0%), chg -1 (dec); F2: price 30 > sma200 25 (100%), chg +1 (adv). pct_above_sma200 = 50%, pct_advancing = 50%
        {"ticker": "F1", "sector": "Finance", "price": 10.0, "sma200": 20.0, "change": -1.0, "high52_prev": 25.0, "low52_prev": 15.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "F2", "sector": "Finance", "price": 30.0, "sma200": 25.0, "change": 1.0, "high52_prev": 35.0, "low52_prev": 5.0, "avg_volume": 100e3, "market_cap": 100e6},
        # Energy: 1 stock. E1: price 10 < sma200 30 (0%), chg -2 (dec). pct_above_sma200 = 0%
        {"ticker": "E1", "sector": "Energy", "price": 10.0, "sma200": 30.0, "change": -2.0, "high52_prev": 40.0, "low52_prev": 12.0, "avg_volume": 100e3, "market_cap": 100e6},
    ])

    sec_df = breadth.by_sector(df, junk=True)

    assert list(sec_df.columns) == ["stocks", "pct_above_sma200", "pct_advancing", "new_highs", "new_lows"]
    # Sorted by pct_above_sma200 descending: Technology (100%), Finance (50%), Energy (0%)
    assert list(sec_df.index) == ["Technology", "Finance", "Energy"]
    assert sec_df.loc["Technology", "stocks"] == 2
    assert sec_df.loc["Technology", "pct_above_sma200"] == 100.0
    assert sec_df.loc["Finance", "pct_above_sma200"] == 50.0
    assert sec_df.loc["Energy", "pct_above_sma200"] == 0.0


def test_history_caching_and_updates(tmp_path, monkeypatch):
    store = Store(root=tmp_path)
    monkeypatch.setattr(breadth.store_module, "DATA_DIR", tmp_path)

    d1 = dt.date(2025, 1, 1)
    d2 = dt.date(2025, 1, 2)
    d3 = dt.date(2025, 1, 3)

    raw_df1 = pd.DataFrame([{"ticker": "A", "price": 100.0, "sma50": 90.0, "sma200": 80.0, "change": 1.0, "high52_prev": 90.0, "low52_prev": 50.0, "avg_volume": 100e3, "market_cap": 100e6}])
    raw_df2 = pd.DataFrame([{"ticker": "A", "price": 105.0, "sma50": 91.0, "sma200": 81.0, "change": 5.0, "high52_prev": 90.0, "low52_prev": 50.0, "avg_volume": 100e3, "market_cap": 100e6}])
    raw_df3 = pd.DataFrame([{"ticker": "A", "price": 95.0, "sma50": 92.0, "sma200": 82.0, "change": -10.0, "high52_prev": 110.0, "low52_prev": 50.0, "avg_volume": 100e3, "market_cap": 100e6}])

    store.snap_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(raw_df1, store.snapshot_path(d1))
    _atomic_parquet(raw_df2, store.snapshot_path(d2))
    _atomic_parquet(raw_df3, store.snapshot_path(d3))

    derive_call_count = 0
    orig_add_derived = derive.add_derived

    def mock_add_derived(raw, today=None):
        nonlocal derive_call_count
        derive_call_count += 1
        return orig_add_derived(raw, today=today)

    monkeypatch.setattr(breadth.derive, "add_derived", mock_add_derived)

    # First call: computes 3 snapshot days -> add_derived called 3 times
    h1 = breadth.history(store, days=60, junk=True)
    assert len(h1) == 3
    assert list(h1.index) == [d1, d2, d3]
    assert derive_call_count == 3

    # Second call: reads cache -> 0 calls to add_derived
    derive_call_count = 0
    h2 = breadth.history(store, days=60, junk=True)
    assert len(h2) == 3
    assert derive_call_count == 0

    # Touch snapshot d2 (update mtime_ns) -> recomputes d2 only (add_derived called 1 time)
    p2 = store.snapshot_path(d2)
    stat2 = p2.stat()
    # Modify mtime by setting utime
    new_mtime = stat2.st_mtime + 10.0
    import os
    os.utime(p2, (new_mtime, new_mtime))

    derive_call_count = 0
    h3 = breadth.history(store, days=60, junk=True)
    assert len(h3) == 3
    assert derive_call_count == 1

    # Damaged cache JSON -> rebuilt completely
    cache_file = tmp_path / "breadth" / "daily.json"
    cache_file.write_text("CORRUPTED JSON {", encoding="utf-8")

    derive_call_count = 0
    h4 = breadth.history(store, days=60, junk=True)
    assert len(h4) == 3
    assert derive_call_count == 3

    # Unreadable snapshot file -> skipped gracefully with a warning
    bad_snap = store.snapshot_path(d3)
    bad_snap.write_bytes(b"INVALID PARQUET CONTENT")

    derive_call_count = 0
    h5 = breadth.history(store, days=60, junk=True)
    # d1 and d2 loaded from cache or computed; d3 snapshot load fails and d3 is skipped
    # Since cache was rewritten by h4, d1 and d2 come from cache
    assert len(h5) == 2
    assert list(h5.index) == [d1, d2]


def test_labels_hebrew_and_english_allowed():
    ALLOWED_ENGLISH = re.compile(r"\b(SMA\d*|ATR|RSI|EPS|PEG|ROA|ROE|ROIC|FCF|EV|EBITDA|P|E|S|B|M|K|x|"
                                 r"S&P|500|Nasdaq|100|ADX|SPY|QQQ|HYG|VIX|USD|US|B2B|B2C|AI|IT|TTM|Q/Q|YTD|P/L|P&L)\b")

    for key, label in breadth.LABELS.items():
        assert isinstance(label, str) and len(label.strip()) > 0
        # Clean allowed English terms and check for leftover english letters
        cleaned = ALLOWED_ENGLISH.sub("", label)
        leftover_english = re.findall(r"[A-Za-z]+", cleaned)
        assert not leftover_english, f"Unexpected English in breadth LABELS[{key!r}] = {label!r}: {leftover_english}"


def test_speed_benchmark():
    n_stocks = 5000
    np.random.seed(42)

    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n_stocks)],
        "price": np.random.uniform(5.0, 500.0, size=n_stocks),
        "sma50": np.random.uniform(5.0, 500.0, size=n_stocks),
        "sma200": np.random.uniform(5.0, 500.0, size=n_stocks),
        "change": np.random.uniform(-5.0, 5.0, size=n_stocks),
        "high52_prev": np.random.uniform(5.0, 500.0, size=n_stocks),
        "low52_prev": np.random.uniform(5.0, 500.0, size=n_stocks),
        "avg_volume": np.full(n_stocks, 200e3),
        "market_cap": np.full(n_stocks, 500e6),
    })

    t0 = time.perf_counter()
    b = breadth.breadth(df, junk=True)
    elapsed = time.perf_counter() - t0

    assert b["stocks"] == n_stocks
    assert elapsed < 0.2, f"breadth() on 5000 stocks took {elapsed:.4f}s (expected < 0.2s)"


def test_history_cache_is_per_junk_setting(tmp_path):
    store = Store(root=tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)
    day = dt.date(2025, 1, 2)
    raw = pd.DataFrame([
        {"ticker": "A", "price": 100.0, "sma50": 90.0, "sma200": 80.0, "change": 1.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "J", "price": 0.5, "sma50": 0.4, "sma200": 0.4, "change": 1.0, "avg_volume": 100e3, "market_cap": 100e6},
    ])
    _atomic_parquet(raw, store.snapshot_path(day))
    assert breadth.history(store, junk=True).loc[day, "advancers"] == 1
    assert breadth.history(store, junk=False).loc[day, "advancers"] == 2
