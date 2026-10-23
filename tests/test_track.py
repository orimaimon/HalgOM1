"""Tests for screener/track.py track record engine (offline on synthetic data)."""
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from screener.data.store import Store
from screener.derive import add_derived
from screener.fields import COLUMNS
import screener.track as track


def _make_snapshot(day: dt.date, tickers: list[str], prices: dict[str, float]) -> pd.DataFrame:
    n = len(tickers)
    data = {}
    for key, col in COLUMNS.items():
        if col.source != "provider" or col.kind == "text":
            continue
        if col.kind == "date":
            data[key] = [pd.Timestamp(day)] * n
        else:
            data[key] = [10.0] * n

    df = pd.DataFrame(data)
    df["ticker"] = tickers
    df["company"] = [f"Co {t}" for t in tickers]
    df["exchange"] = "NASDAQ"
    df["sector"] = "Technology"
    df["industry"] = "Software"
    df["country"] = "United States"
    df["sec_type"] = "stock"
    df["tv_symbol"] = [f"NASDAQ:{t}" for t in tickers]
    df["price"] = [prices.get(t, 100.0) for t in tickers]
    df["market_cap"] = 1e9
    df["pe"] = 12.0  # passes "Under 15" filter
    df["avg_volume"] = 1e6
    df["high52"] = df["price"] * 1.1
    df["low52"] = df["price"] * 0.9
    return df


@pytest.fixture
def mock_no_network(monkeypatch):
    """Ensure no network calls are made during track testing."""
    monkeypatch.setattr(track.prices, "ensure", lambda tickers, start: (tickers, []))
    monkeypatch.setattr(track.prices, "total_return", lambda ticker, start, end: None)
    monkeypatch.setattr(track.prices, "close_on", lambda ticker, day: None)


def test_track_synthetic_30_days(tmp_path, mock_no_network):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    base_date = dt.date(2026, 1, 1)
    dates = [base_date + dt.timedelta(days=i) for i in range(30)]

    # Price trajectories:
    # T001 enters D0 (p=100), D5 (p=110 -> +10%), D20 (p=120 -> +20%)
    # T002 enters D0 (p=50),  D5 (p=45 -> -10%), D20 (p=60 -> +20%)
    # SPLT enters D10 (p=100), D15 (p=50 due to 2:1 split -> 0% return adjusted)
    # T025 enters D25 (p=100), not enough days for +5, +20, +60 -> "not yet"

    for i, d in enumerate(dates):
        tickers = ["T001", "T002"]
        price_map = {"T001": 100.0, "T002": 50.0}

        if i >= 5:
            price_map["T001"] = 110.0
            price_map["T002"] = 45.0
        if i >= 20:
            price_map["T001"] = 120.0
            price_map["T002"] = 60.0

        if i >= 10:
            tickers.append("SPLT")
            price_map["SPLT"] = 100.0 if i < 15 else 50.0  # 2:1 split on D15

        if i >= 25:
            tickers.append("T025")
            price_map["T025"] = 100.0

        snap = _make_snapshot(d, tickers, price_map)
        snap.to_parquet(store.snap_dir / f"{d.isoformat()}.parquet")

    filters = {"pe": "Under 15"}
    res = track.track_screen(filters=filters, screen_name="TestScreen", store=store)

    assert res["screen_name"] == "TestScreen"
    entries = res["entries"]

    # Entries detected:
    # D0: T001, T002
    # D10: SPLT
    # D25: T025
    d0_entries = [e for e in entries if e["entry_date"] == dates[0].isoformat()]
    assert set(e["ticker"] for e in d0_entries) == {"T001", "T002"}

    t001 = next(e for e in entries if e["ticker"] == "T001" and e["entry_date"] == dates[0].isoformat())
    t002 = next(e for e in entries if e["ticker"] == "T002" and e["entry_date"] == dates[0].isoformat())

    # 5-day and 20-day returns exact:
    assert t001["returns"]["5"] == 10.0   # +10%
    assert t001["returns"]["20"] == 20.0  # +20%
    assert t002["returns"]["5"] == -10.0  # -10%
    assert t002["returns"]["20"] == 20.0  # +20%

    # Split handling: SPLT enters on D10 at 100, on D15 price drops to 50 due to 2:1 split.
    # Adjusted exit price = 50 * 2 = 100 => 0.0% return (not -50%)
    splt = next(e for e in entries if e["ticker"] == "SPLT")
    assert splt["returns"]["5"] == 0.0

    # "not yet" at the end: T025 enters on D25 (index 25). 25 + 5 = 30 >= 30 days -> "not yet"
    t025 = next(e for e in entries if e["ticker"] == "T025")
    assert t025["returns"]["5"] == track.NOT_YET
    assert t025["returns"]["20"] == track.NOT_YET
    assert t025["returns"]["60"] == track.NOT_YET

    # Summary checks:
    h5 = res["horizons"][5]
    assert h5["entries"] == 3  # T001, T002, SPLT
    assert h5["median_return"] == 0.0  # median(10, -10, 0) == 0.0
    assert h5["pct_positive"] == pytest.approx(33.3, abs=0.1)  # 1 out of 3 positive (T001)


def test_track_incremental_update(tmp_path, mock_no_network):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    base_date = dt.date(2026, 1, 1)
    dates = [base_date + dt.timedelta(days=i) for i in range(25)]

    for d in dates:
        snap = _make_snapshot(d, ["T001"], {"T001": 100.0})
        snap.to_parquet(store.snap_dir / f"{d.isoformat()}.parquet")

    filters = {"pe": "Under 15"}

    # Initial run with 25 days
    res1 = track.track_screen(filters=filters, screen_name="IncrementalTest", store=store)
    c_path = track._cache_path("IncrementalTest", filters, root_dir=store.root / "track")
    assert c_path.exists()

    with open(c_path, "r", encoding="utf-8") as f:
        cache1 = json.load(f)
    assert len(cache1["daily_results"]) == 25

    # Add 26th day
    d25 = base_date + dt.timedelta(days=25)
    snap25 = _make_snapshot(d25, ["T001"], {"T001": 150.0})
    snap25.to_parquet(store.snap_dir / f"{d25.isoformat()}.parquet")

    processed_snapshots = []
    orig_load = store.load_snapshot

    def load_snapshot_logging(d):
        processed_snapshots.append(d)
        return orig_load(d)

    with patch.object(store, "load_snapshot", side_effect=load_snapshot_logging):
        res2 = track.track_screen(filters=filters, screen_name="IncrementalTest", store=store)

    # Verify only the new snapshot was loaded during incremental processing
    assert d25 in processed_snapshots
    assert len(processed_snapshots) <= 2  # d25 for screening + d25 on demand for snapshot exit price lookup

    with open(c_path, "r", encoding="utf-8") as f:
        cache2 = json.load(f)
    assert len(cache2["daily_results"]) == 26


def test_track_cache_invalidation_on_filter_change(tmp_path, mock_no_network):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    d0 = dt.date(2026, 1, 1)
    snap = _make_snapshot(d0, ["T001"], {"T001": 100.0})
    snap.to_parquet(store.snap_dir / f"{d0.isoformat()}.parquet")

    filters1 = {"pe": "Under 15"}
    filters2 = {"pe": "Under 10"}

    res1 = track.track_screen(filters=filters1, screen_name="CacheTest", store=store)
    path1 = track._cache_path("CacheTest", filters1, root_dir=store.root / "track")
    assert path1.exists()

    # Change filters -> cache path for new filters differs or invalidates
    res2 = track.track_screen(filters=filters2, screen_name="CacheTest", store=store)
    path2 = track._cache_path("CacheTest", filters2, root_dir=store.root / "track")
    assert path2.exists()
    assert path1 != path2


def test_track_all_screens(tmp_path, mock_no_network):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    d0 = dt.date(2026, 1, 1)
    snap = _make_snapshot(d0, ["T001"], {"T001": 100.0})
    snap.to_parquet(store.snap_dir / f"{d0.isoformat()}.parquet")

    all_res = track.track_all_screens(store=store)
    assert isinstance(all_res, dict)
    assert "Value — cheap & profitable" in all_res
    assert "disclaimer" in all_res["Value — cheap & profitable"]
