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


def test_update_fetches_an_older_range_when_start_predates_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    cached = pd.Series([3.5, 3.6], index=[dt.date(2026, 1, 5), dt.date(2026, 1, 6)])
    boi._atomic_parquet(cached, tmp_path / "boi" / "usd_ils.parquet")
    asked = []

    def fake_fetch(start, end):
        asked.append((start, end))
        return pd.Series([3.9, 3.5, 3.6], index=[dt.date(2024, 3, 1), dt.date(2026, 1, 5), dt.date(2026, 1, 6)])

    monkeypatch.setattr(boi, "fetch_rates", fake_fetch)
    assert boi.update(dt.date(2024, 3, 1), today=dt.date(2026, 1, 7)) is True
    assert asked == [(dt.date(2024, 3, 1), dt.date(2026, 1, 7))]
    assert boi.rate_on(dt.date(2024, 3, 2)) == 3.9


def test_get_rate_for_date_weekends_and_holidays():
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,   # Thursday
            dt.date(2026, 10, 2): 3.06,   # Friday
            dt.date(2026, 10, 5): 3.059,  # Monday (2026-10-06 is a holiday)
            dt.date(2026, 10, 7): 3.050,  # Wednesday
        }
    )

    # Friday rate
    assert boi.get_rate_for_date(dt.date(2026, 10, 2), rates) == 3.06

    # Saturday (2026-10-03) and Sunday (2026-10-04) -> Friday's rate
    assert boi.get_rate_for_date(dt.date(2026, 10, 3), rates) == 3.06
    assert boi.get_rate_for_date(dt.date(2026, 10, 4), rates) == 3.06

    # Holiday gap: Tuesday (2026-10-06) -> Monday's rate (3.059)
    assert boi.get_rate_for_date(dt.date(2026, 10, 6), rates) == 3.059


def test_get_rate_for_date_earliest_and_future():
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,
            dt.date(2026, 10, 2): 3.06,
        }
    )

    # Before earliest known date -> NaN
    res_before = boi.get_rate_for_date(dt.date(2026, 9, 30), rates)
    assert pd.isna(res_before)

    # Future date after latest known rate -> forward-fill latest rate (3.06)
    assert boi.get_rate_for_date(dt.date(2026, 11, 15), rates) == 3.06


def test_get_rates_for_range():
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.08,   # Thursday
            dt.date(2026, 10, 2): 3.06,   # Friday
            dt.date(2026, 10, 5): 3.059,  # Monday (Tuesday Oct 6 holiday)
            dt.date(2026, 10, 7): 3.050,  # Wednesday
        }
    )

    # Range covering weekend and holiday (Oct 2 to Oct 7)
    s_range = boi.get_rates_for_range(dt.date(2026, 10, 2), dt.date(2026, 10, 7), rates)
    expected_dates = [
        dt.date(2026, 10, 2),  # Fri: 3.06
        dt.date(2026, 10, 3),  # Sat: 3.06 (weekend)
        dt.date(2026, 10, 4),  # Sun: 3.06 (weekend)
        dt.date(2026, 10, 5),  # Mon: 3.059
        dt.date(2026, 10, 6),  # Tue: 3.059 (holiday gap)
        dt.date(2026, 10, 7),  # Wed: 3.050
    ]
    assert list(s_range.index) == expected_dates
    assert s_range.loc[dt.date(2026, 10, 2)] == 3.06
    assert s_range.loc[dt.date(2026, 10, 3)] == 3.06
    assert s_range.loc[dt.date(2026, 10, 4)] == 3.06
    assert s_range.loc[dt.date(2026, 10, 5)] == 3.059
    assert s_range.loc[dt.date(2026, 10, 6)] == 3.059
    assert s_range.loc[dt.date(2026, 10, 7)] == 3.050


def test_get_rates_for_range_earliest_and_future(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    rates = pd.Series(
        {
            dt.date(2026, 10, 2): 3.06,
            dt.date(2026, 10, 5): 3.059,
        }
    )
    boi_dir = tmp_path / "boi"
    boi_dir.mkdir(parents=True, exist_ok=True)
    boi._atomic_parquet(rates, boi_dir / "usd_ils.parquet")

    # Range completely before earliest date
    s_before = boi.get_rates_for_range(dt.date(2026, 9, 28), dt.date(2026, 9, 30))
    assert len(s_before) == 3
    assert s_before.isna().all()

    # Range straddling earliest date
    s_straddle = boi.get_rates_for_range(dt.date(2026, 10, 1), dt.date(2026, 10, 3))
    assert pd.isna(s_straddle.loc[dt.date(2026, 10, 1)])
    assert s_straddle.loc[dt.date(2026, 10, 2)] == 3.06
    assert s_straddle.loc[dt.date(2026, 10, 3)] == 3.06

    # Range completely in the future
    s_future = boi.get_rates_for_range(dt.date(2026, 11, 1), dt.date(2026, 11, 3))
    assert len(s_future) == 3
    assert (s_future == 3.059).all()
