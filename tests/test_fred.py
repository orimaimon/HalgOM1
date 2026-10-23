"""Tests for FRED data provider module."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

from screener.data import fred, guard, store

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "fred"


def test_build_url():
    url_default = fred.build_url()
    assert "id=VIXCLS,BAMLH0A0HYM2" in url_default

    url_single = fred.build_url("VIXCLS")
    assert url_single == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

    url_list = fred.build_url(["VIXCLS", "BAMLH0A0HYM2"])
    assert url_list == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS,BAMLH0A0HYM2"


def test_parse_csv_header_styles_and_nans():
    csv_new = (FIXTURES_DIR / "fred_header_new.csv").read_text(encoding="utf-8")
    df_new = fred._parse_csv(csv_new)

    assert isinstance(df_new, pd.DataFrame)
    assert list(df_new.columns) == ["vix", "hy_spread"]
    assert df_new.index[0] == dt.date(2026, 1, 1)
    assert np.isnan(df_new.loc[dt.date(2026, 1, 1), "vix"])
    assert np.isnan(df_new.loc[dt.date(2026, 1, 1), "hy_spread"])
    assert df_new.loc[dt.date(2026, 1, 2), "vix"] == 15.2
    assert df_new.loc[dt.date(2026, 1, 2), "hy_spread"] == 3.45
    assert np.isnan(df_new.loc[dt.date(2026, 1, 3), "vix"])
    assert df_new.loc[dt.date(2026, 1, 3), "hy_spread"] == 3.50

    csv_old = (FIXTURES_DIR / "fred_header_old.csv").read_text(encoding="utf-8")
    df_old = fred._parse_csv(csv_old)
    assert df_old.loc[dt.date(2026, 1, 1), "vix"] == 16.1
    assert df_old.loc[dt.date(2026, 1, 1), "hy_spread"] == 3.50
    assert np.isnan(df_old.loc[dt.date(2026, 1, 2), "vix"])
    assert np.isnan(df_old.loc[dt.date(2026, 1, 2), "hy_spread"])


def test_parse_csv_missing_date_column():
    invalid_csv = "col1,col2\n1,2"
    with pytest.raises(fred.FredError, match="Missing date column"):
        fred._parse_csv(invalid_csv)


def test_429_raises_fred_error_and_guard_is_block(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_resp)

    with pytest.raises(fred.FredError) as exc_info:
        fred._fetch_fred()

    exc = exc_info.value
    assert exc.status_code == 429
    assert guard._is_block(exc) is True


def test_update_downloads_once_per_ny_day(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    fake_now = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(guard, "_default_now", lambda: fake_now)

    csv_sample = (FIXTURES_DIR / "fred_sample.csv").read_text(encoding="utf-8")
    fetch_count = 0

    def mock_get(url, timeout):
        nonlocal fetch_count
        fetch_count += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.text = csv_sample
        return resp

    monkeypatch.setattr(requests, "get", mock_get)

    day1 = dt.date(2026, 10, 23)
    day2 = dt.date(2026, 10, 24)

    # First call on day 1: downloads
    res1 = fred.update(today=day1)
    assert res1 is True
    assert fetch_count == 1

    # Second call on day 1: cached, no new download
    res2 = fred.update(today=day1)
    assert res2 is False
    assert fetch_count == 1

    # Advance fake time by 1 day for day 2
    fake_now = dt.datetime(2026, 10, 24, 12, 0, tzinfo=dt.timezone.utc)

    # Call on day 2: downloads again
    res3 = fred.update(today=day2)
    assert res3 is True
    assert fetch_count == 2


def test_update_handles_guard_wait_and_http_500(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    fake_now = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(guard, "_default_now", lambda: fake_now)

    # First, save a valid file
    csv_sample = (FIXTURES_DIR / "fred_sample.csv").read_text(encoding="utf-8")
    mock_ok = MagicMock(status_code=200, text=csv_sample)
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_ok)

    day1 = dt.date(2026, 10, 23)
    assert fred.update(today=day1) is True

    d_dir = tmp_path / "fred"
    parquet_path = d_dir / "fred.parquet"
    original_mtime = parquet_path.stat().st_mtime

    day2 = dt.date(2026, 10, 24)
    fake_now = dt.datetime(2026, 10, 24, 12, 0, tzinfo=dt.timezone.utc)

    # Test HTTP 500 error on day 2
    mock_500 = MagicMock(status_code=500, text="Internal Server Error")
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_500)

    res_500 = fred.update(today=day2)
    assert res_500 is False
    assert "FRED update failed" in caplog.text or "FRED returned status code 500" in caplog.text
    # Check that previous parquet file is preserved
    assert parquet_path.exists()
    assert parquet_path.stat().st_mtime == original_mtime

    # Test guard.Wait on day 2
    def mock_guard_wait(*args, **kwargs):
        raise guard.Wait("fred", dt.datetime.now().astimezone(), "min_interval")

    monkeypatch.setattr(guard, "call", mock_guard_wait)

    res_wait = fred.update(today=day2)
    assert res_wait is False
    assert parquet_path.exists()


def test_latest_and_history_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    s_vix = fred.history("vix", data_dir=tmp_path / "fred")
    assert isinstance(s_vix, pd.Series)
    assert s_vix.empty

    latest_data = fred.latest(data_dir=tmp_path / "fred")
    assert latest_data == {
        "vix": {"value": None, "date": None},
        "hy_spread": {"value": None, "date": None},
        "hy_spread_sma20": None,
        "hy_spread_above_sma20": None,
    }


def test_latest_and_history_with_data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    csv_sample = (FIXTURES_DIR / "fred_sample.csv").read_text(encoding="utf-8")
    mock_resp = MagicMock(status_code=200, text=csv_sample)
    monkeypatch.setattr(requests, "get", lambda url, timeout: mock_resp)

    # Run update to write data
    assert fred.update(today=dt.date(2026, 10, 23)) is True

    d_dir = tmp_path / "fred"

    # Test history()
    s_vix = fred.history("vix", data_dir=d_dir)
    assert len(s_vix) == 26
    assert s_vix.loc[dt.date(2026, 1, 24)] == 17.3

    s_hy = fred.history("hy_spread", data_dir=d_dir)
    assert len(s_hy) == 26
    assert s_hy.loc[dt.date(2026, 1, 25)] == 3.24

    s_unknown = fred.history("non_existent", data_dir=d_dir)
    assert s_unknown.empty

    # Test latest()
    latest_data = fred.latest(data_dir=d_dir)

    # vix: last non-NaN is 17.3 on 2026-01-24
    assert latest_data["vix"] == {"value": 17.3, "date": "2026-01-24"}

    # hy_spread: last non-NaN is 3.24 on 2026-01-25
    assert latest_data["hy_spread"] == {"value": 3.24, "date": "2026-01-25"}

    # hy_spread_sma20: trailing 20 non-NaN values (rows 6 to 25: 3.05 to 3.24)
    # mean of 3.05..3.24 is 3.145
    assert latest_data["hy_spread_sma20"] == pytest.approx(3.145)

    # 3.24 > 3.145 -> True
    assert latest_data["hy_spread_above_sma20"] is True
