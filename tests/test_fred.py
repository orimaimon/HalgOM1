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


def _sample_csv_with_yield_curve() -> str:
    raw = (FIXTURES_DIR / "fred_sample.csv").read_text(encoding="utf-8").splitlines()
    lines = []
    for i, line in enumerate(raw):
        if i == 0:
            lines.append(line + ",T10Y2Y")
        elif i == len(raw) - 1:
            lines.append(line + ",.")
        else:
            lines.append(f"{line},{0.50 + i * 0.01:.2f}")
    return "\n".join(lines)


def test_build_url():
    url_default = fred.build_url()
    assert "id=VIXCLS,BAMLH0A0HYM2,T10Y2Y" in url_default

    url_single = fred.build_url("VIXCLS")
    assert url_single == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

    url_list = fred.build_url(["VIXCLS", "BAMLH0A0HYM2", "T10Y2Y"])
    assert url_list == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS,BAMLH0A0HYM2,T10Y2Y"


def test_parse_csv_header_styles_and_nans():
    csv_text = (
        "observation_date,VIXCLS,BAMLH0A0HYM2,T10Y2Y\n"
        "2026-01-01,.,.,.\n"
        "2026-01-02,15.2,3.45,0.65\n"
        "2026-01-03,.,3.50,-0.10\n"
    )
    df_new = fred._parse_csv(csv_text)

    assert isinstance(df_new, pd.DataFrame)
    assert list(df_new.columns) == ["vix", "hy_spread", "yield_curve"]
    assert df_new.index[0] == dt.date(2026, 1, 1)
    assert np.isnan(df_new.loc[dt.date(2026, 1, 1), "vix"])
    assert np.isnan(df_new.loc[dt.date(2026, 1, 1), "hy_spread"])
    assert np.isnan(df_new.loc[dt.date(2026, 1, 1), "yield_curve"])
    assert df_new.loc[dt.date(2026, 1, 2), "vix"] == 15.2
    assert df_new.loc[dt.date(2026, 1, 2), "hy_spread"] == 3.45
    assert df_new.loc[dt.date(2026, 1, 2), "yield_curve"] == 0.65
    assert df_new.loc[dt.date(2026, 1, 3), "yield_curve"] == -0.10

    csv_old = "DATE,VIXCLS,BAMLH0A0HYM2,T10Y2Y\n2026-01-01,16.1,3.50,0.40\n2026-01-02,.,.,.\n"
    df_old = fred._parse_csv(csv_old)
    assert df_old.loc[dt.date(2026, 1, 1), "vix"] == 16.1
    assert df_old.loc[dt.date(2026, 1, 1), "hy_spread"] == 3.50
    assert df_old.loc[dt.date(2026, 1, 1), "yield_curve"] == 0.40
    assert np.isnan(df_old.loc[dt.date(2026, 1, 2), "vix"])


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

    csv_sample = _sample_csv_with_yield_curve()
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
    csv_sample = _sample_csv_with_yield_curve()
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
        "yield_curve": {"value": None, "date": None},
        "hy_spread_sma20": None,
        "hy_spread_above_sma20": None,
    }


def test_latest_and_history_with_data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    csv_sample = _sample_csv_with_yield_curve()
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

    s_yc = fred.history("yield_curve", data_dir=d_dir)
    assert len(s_yc) == 26
    assert s_yc.loc[dt.date(2026, 1, 24)] == pytest.approx(0.74)

    s_unknown = fred.history("non_existent", data_dir=d_dir)
    assert s_unknown.empty

    # Test latest()
    latest_data = fred.latest(data_dir=d_dir)

    # vix: last non-NaN is 17.3 on 2026-01-24
    assert latest_data["vix"] == {"value": 17.3, "date": "2026-01-24"}

    # hy_spread: last non-NaN is 3.24 on 2026-01-25
    assert latest_data["hy_spread"] == {"value": 3.24, "date": "2026-01-25"}

    # yield_curve: last non-NaN is 0.75 on 2026-01-25
    assert latest_data["yield_curve"] == {"value": pytest.approx(0.75), "date": "2026-01-25"}

    # hy_spread_sma20: trailing 20 non-NaN values (rows 6 to 25: 3.05 to 3.24)
    assert latest_data["hy_spread_sma20"] == pytest.approx(3.145)

    assert latest_data["hy_spread_above_sma20"] is True


def test_answer_without_data_keeps_the_previous_file(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    good = pd.DataFrame(
        {"vix": [15.0], "hy_spread": [3.1], "yield_curve": [0.55]},
        index=[pd.Timestamp("2026-10-22").date()],
    )
    monkeypatch.setattr(fred.guard, "call", lambda provider, fn, *a, **k: good)
    assert fred.update(today=pd.Timestamp("2026-10-22").date()) is True

    # The next day FRED answers with missing yield_curve series
    incomplete = pd.DataFrame({"vix": [16.0], "hy_spread": [3.2]}, index=[pd.Timestamp("2026-10-23").date()])
    monkeypatch.setattr(fred.guard, "call", lambda provider, fn, *a, **k: incomplete)
    assert fred.update(today=pd.Timestamp("2026-10-23").date()) is False
    assert fred.latest()["hy_spread"]["value"] == 3.1
    assert fred.latest()["yield_curve"]["value"] == 0.55


def test_compute_macro_summary_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    summary = fred.compute_macro_summary(data_dir=tmp_path / "fred")
    assert summary["regime"] == "Unknown"
    assert summary["latest"]["yield_curve"] is None
    assert summary["changes_1w"]["yield_curve"] is None


def test_compute_macro_summary_normal_and_inverted_regimes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    d_dir = tmp_path / "fred"

    # 1. Normal Yield Curve (Expansion)
    dates = [dt.date(2026, 1, i) for i in range(1, 15)]
    df_normal = pd.DataFrame(
        {
            "vix": [15.0 + i * 0.1 for i in range(14)],
            "hy_spread": [3.20 + i * 0.02 for i in range(14)],
            "yield_curve": [0.60 + i * 0.01 for i in range(14)],
        },
        index=dates,
    )
    fred._atomic_parquet(df_normal, d_dir / "fred.parquet")

    summary_normal = fred.compute_macro_summary(data_dir=d_dir)
    assert summary_normal["regime"] == "Expansion"
    assert summary_normal["latest"]["vix"] == pytest.approx(16.3)
    assert summary_normal["latest"]["hy_spread"] == pytest.approx(3.46)
    assert summary_normal["latest"]["yield_curve"] == pytest.approx(0.73)
    # 1-week change (2026-01-14 minus 2026-01-07): 0.73 - 0.66 = 0.07
    assert summary_normal["yield_curve"]["change_1w"] == pytest.approx(0.07)
    assert summary_normal["changes_1w"]["yield_curve"] == pytest.approx(0.07)
    assert summary_normal["vix"]["change_1w"] == pytest.approx(0.7)

    # 2. Inverted Yield Curve (Inversion)
    df_inverted = pd.DataFrame(
        {
            "vix": [18.0] * 14,
            "hy_spread": [3.80] * 14,
            "yield_curve": [-0.35] * 14,
        },
        index=dates,
    )
    fred._atomic_parquet(df_inverted, d_dir / "fred.parquet")

    summary_inverted = fred.compute_macro_summary(data_dir=d_dir)
    assert summary_inverted["regime"] == "Inversion"
    assert summary_inverted["latest"]["yield_curve"] == pytest.approx(-0.35)

    # 3. Flat Yield Curve or Elevated Credit Spread (Caution)
    df_caution_spread = pd.DataFrame(
        {
            "vix": [20.0] * 14,
            "hy_spread": [5.50] * 14,
            "yield_curve": [0.50] * 14,
        },
        index=dates,
    )
    fred._atomic_parquet(df_caution_spread, d_dir / "fred.parquet")

    summary_caution1 = fred.compute_macro_summary(data_dir=d_dir)
    assert summary_caution1["regime"] == "Caution"

    df_caution_flat = pd.DataFrame(
        {
            "vix": [16.0] * 14,
            "hy_spread": [3.50] * 14,
            "yield_curve": [0.10] * 14,
        },
        index=dates,
    )
    fred._atomic_parquet(df_caution_flat, d_dir / "fred.parquet")

    summary_caution2 = fred.compute_macro_summary(data_dir=d_dir)
    assert summary_caution2["regime"] == "Caution"
