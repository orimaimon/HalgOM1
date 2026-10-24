import datetime as dt
from pathlib import Path
import pytest
import pandas as pd
import requests

from screener.data import guard, store
from screener.data import boi


def test_guard_boi_limits_exist():
    assert "boi" in guard.LIMITS
    assert guard.LIMITS["boi"].min_interval == 60.0
    assert guard.LIMITS["boi"].daily_cap == 30
    assert guard.DISPLAY_NAMES["boi"] == "בנק ישראל"


def test_parse_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "boi" / "usd_ils.csv"
    csv_text = fixture_path.read_text(encoding="utf-8")
    s = boi._parse_csv(csv_text)
    assert len(s) == 3
    assert s.loc[dt.date(2026, 10, 1)] == 3.08
    assert s.loc[dt.date(2026, 10, 2)] == 3.06
    assert s.loc[dt.date(2026, 10, 5)] == 3.059


def test_rate_on_lookback():
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,  # Thursday
            dt.date(2026, 10, 2): 3.06,  # Friday
            dt.date(2026, 10, 5): 3.059, # Monday
        }
    )
    # Saturday 2026-10-03 -> Friday 2026-10-02
    assert boi.rate_on(dt.date(2026, 10, 3), rates) == 3.06
    # Sunday 2026-10-04 -> Friday 2026-10-02
    assert boi.rate_on(dt.date(2026, 10, 4), rates) == 3.06

    # Test Thursday fallback if Friday missing
    rates_no_fri = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,  # Thursday
            dt.date(2026, 10, 5): 3.059, # Monday
        }
    )
    assert boi.rate_on(dt.date(2026, 10, 3), rates_no_fri) == 3.08

    # More than 7 days without rate -> None
    assert boi.rate_on(dt.date(2026, 10, 15), rates) is None


def test_429_raises_with_status_code(monkeypatch):
    def mock_get(url, timeout=30):
        resp = requests.Response()
        resp.status_code = 429
        return resp

    monkeypatch.setattr(requests, "get", mock_get)
    with pytest.raises(boi.BoiError) as exc_info:
        boi.fetch_rates(dt.date(2026, 10, 1), dt.date(2026, 10, 5))
    assert exc_info.value.status_code == 429


def test_update_missing_range_and_once_per_day(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    # Setup initial cache
    initial_rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,
            dt.date(2026, 10, 2): 3.06,
        }
    )
    boi_dir = tmp_path / "boi"
    boi_dir.mkdir(parents=True, exist_ok=True)
    boi._atomic_parquet(initial_rates, boi_dir / "usd_ils.parquet")

    requested_ranges = []

    def mock_raw_fetch(start, end, timeout=30):
        requested_ranges.append((start, end))
        return pd.Series({dt.date(2026, 10, 5): 3.059})

    monkeypatch.setattr(boi, "_raw_fetch", mock_raw_fetch)

    today = dt.date(2026, 10, 5)
    # First update: should request from 2026-10-03 (day after last cached 2026-10-02) to 2026-10-05
    res1 = boi.update(start=dt.date(2026, 10, 1), today=today)
    assert res1 is True
    assert len(requested_ranges) == 1
    assert requested_ranges[0] == (dt.date(2026, 10, 3), dt.date(2026, 10, 5))

    # Second update on same day: should do nothing and return False
    res2 = boi.update(start=dt.date(2026, 10, 1), today=today)
    assert res2 is False
    assert len(requested_ranges) == 1  # No new fetch request sent


def test_update_guard_wait_and_500_error_keeps_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    boi_dir = tmp_path / "boi"
    boi_dir.mkdir(parents=True, exist_ok=True)

    initial_rates = pd.Series({dt.date(2026, 10, 1): 3.08})
    boi._atomic_parquet(initial_rates, boi_dir / "usd_ils.parquet")

    # 1. guard.Wait
    def mock_fetch_wait(start, end):
        raise guard.Wait("boi", dt.datetime.now(), "backoff")

    monkeypatch.setattr(boi, "fetch_rates", mock_fetch_wait)
    res_wait = boi.update(start=dt.date(2026, 10, 1), today=dt.date(2026, 10, 5))
    assert res_wait is False
    assert len(boi._load_cache(boi_dir)) == 1

    # Reset state
    (boi_dir / "state.json").unlink(missing_ok=True)

    # 2. HTTP 500 error
    def mock_fetch_500(start, end):
        raise boi.BoiError("Server Error", status_code=500)

    monkeypatch.setattr(boi, "fetch_rates", mock_fetch_500)
    res_500 = boi.update(start=dt.date(2026, 10, 1), today=dt.date(2026, 10, 5))
    assert res_500 is False
    assert len(boi._load_cache(boi_dir)) == 1


def test_empty_csv_never_overwrites_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    boi_dir = tmp_path / "boi"
    boi_dir.mkdir(parents=True, exist_ok=True)

    initial_rates = pd.Series({dt.date(2026, 10, 1): 3.08})
    boi._atomic_parquet(initial_rates, boi_dir / "usd_ils.parquet")

    def mock_fetch_empty(start, end):
        return pd.Series(dtype=float)

    monkeypatch.setattr(boi, "fetch_rates", mock_fetch_empty)
    res = boi.update(start=dt.date(2026, 10, 1), today=dt.date(2026, 10, 5))
    assert res is False
    assert len(boi._load_cache(boi_dir)) == 1
    assert boi._load_cache(boi_dir).loc[dt.date(2026, 10, 1)] == 3.08
