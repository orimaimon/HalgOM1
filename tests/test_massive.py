import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from screener.data import guard, massive, store


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "DATA_DIR", data_dir)
    return data_dir


@pytest.fixture
def key_file(tmp_data_dir):
    k_file = tmp_data_dir / "massive_key.txt"
    k_file.write_text("test_secret_key_999\n", encoding="utf-8")
    return k_file


@pytest.fixture
def grouped_fixture_json():
    p = Path(__file__).parent / "fixtures" / "massive" / "grouped_daily.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_no_key_file(tmp_data_dir, monkeypatch):
    no_key_path = tmp_data_dir / "nonexistent_key.txt"

    requests_called = []

    def mock_get(*args, **kwargs):
        requests_called.append(args)
        raise RuntimeError("Should not be called")

    monkeypatch.setattr("requests.get", mock_get)

    res = massive.update(data_dir=tmp_data_dir / "market_history", key_path=no_key_path)
    assert res == 0
    assert len(requests_called) == 0

    df = massive.fetch_day(dt.date(2024, 1, 2), data_dir=tmp_data_dir / "market_history", key_path=no_key_path)
    assert df.empty
    assert len(requests_called) == 0


def test_key_sent_only_in_authorization_header(tmp_data_dir, key_file, caplog, monkeypatch):
    captured_urls = []
    captured_headers = []

    class MockResponse:
        status_code = 200

        def json(self):
            return {"status": "OK", "resultsCount": 0}

    def mock_get(url, headers=None, timeout=30):
        captured_urls.append(url)
        captured_headers.append(headers or {})
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr(guard, "call", lambda provider, fn, *args, **kwargs: fn())

    massive.fetch_day(dt.date(2024, 1, 2), data_dir=tmp_data_dir / "market_history", key_path=key_file)

    assert len(captured_headers) == 1
    assert captured_headers[0].get("Authorization") == "Bearer test_secret_key_999"

    for url in captured_urls:
        assert "test_secret_key_999" not in url

    for record in caplog.records:
        assert "test_secret_key_999" not in record.getMessage()


def test_fetch_day_parses_fixture_and_empty_day(tmp_data_dir, key_file, grouped_fixture_json, monkeypatch):
    d_dir = tmp_data_dir / "market_history"
    requests_count = 0

    class MockResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    def mock_get(url, headers=None, timeout=30):
        nonlocal requests_count
        requests_count += 1
        if "2024-01-02" in url:
            return MockResponse(grouped_fixture_json)
        elif "2024-01-01" in url:
            return MockResponse({"status": "OK", "resultsCount": 0})
        return MockResponse({"status": "OK", "resultsCount": 0})

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr(guard, "call", lambda provider, fn, *args, **kwargs: fn())

    # 1. Fetch normal day
    df = massive.fetch_day(dt.date(2024, 1, 2), data_dir=d_dir, key_path=key_file)
    assert list(df.columns) == massive.COLUMNS
    assert len(df) == 3
    assert set(df["ticker"]) == {"AAPL", "BRK.B", "MSFT"}
    assert (d_dir / "2024-01-02.parquet").exists()

    # 2. Fetch holiday (empty day)
    df_empty = massive.fetch_day(dt.date(2024, 1, 1), data_dir=d_dir, key_path=key_file)
    assert df_empty.empty
    assert list(df_empty.columns) == massive.COLUMNS

    empty_days_path = d_dir / "empty_days.json"
    assert empty_days_path.exists()
    empty_days = json.loads(empty_days_path.read_text(encoding="utf-8"))
    assert "2024-01-01" in empty_days

    # 3. Call holiday again: should skip network request
    req_before = requests_count
    df_empty_cached = massive.fetch_day(dt.date(2024, 1, 1), data_dir=d_dir, key_path=key_file)
    assert df_empty_cached.empty
    assert requests_count == req_before


def test_update_with_fake_guard(tmp_data_dir, key_file, grouped_fixture_json, monkeypatch):
    d_dir = tmp_data_dir / "market_history"
    d_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate existing file for 2024-01-08 (Monday)
    pd.DataFrame([{"ticker": "AAPL", "close": 150.0}]).to_parquet(d_dir / "2024-01-08.parquet")

    # Pre-populate empty_days.json with 2024-01-05 (Friday)
    (d_dir / "empty_days.json").write_text(json.dumps(["2024-01-05"]), encoding="utf-8")

    fetched_dates = []

    class MockResponse:
        status_code = 200

        def json(self):
            return grouped_fixture_json

    def mock_get(url, headers=None, timeout=30):
        date_str = url.split("/")[-1].split("?")[0]
        fetched_dates.append(date_str)
        return MockResponse()

    def fake_guard_call(provider, fn, *args, **kwargs):
        # Extract date from url inside fn closure or test date
        # If calling fn executes mock_get, inspect date
        res = fn(*args, **kwargs)
        if fetched_dates and fetched_dates[-1] == "2024-01-04":
            raise guard.Wait("massive", dt.datetime.now(), "daily_cap")
        return res

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr(guard, "call", fake_guard_call)

    ref_today = dt.date(2024, 1, 10)  # Wednesday
    # Loop from yesterday back 10 days:
    # 2024-01-09 (Tue) -> fetched
    # 2024-01-08 (Mon) -> exists -> skipped
    # 2024-01-07 (Sun) -> weekend -> skipped
    # 2024-01-06 (Sat) -> weekend -> skipped
    # 2024-01-05 (Fri) -> empty_days -> skipped
    # 2024-01-04 (Thu) -> raises guard.Wait -> stops quietly!

    count = massive.update(days_back=10, today=ref_today, data_dir=d_dir, key_path=key_file)

    assert count == 1
    assert fetched_dates == ["2024-01-09", "2024-01-04"]
    assert "2024-01-10" not in fetched_dates  # Never asks for today


def test_429_status_code(tmp_data_dir, key_file, monkeypatch):
    d_dir = tmp_data_dir / "market_history"

    class MockResponse:
        status_code = 429
        text = "429 Too Many Requests"

        def json(self):
            return {"error": "rate limit"}

    def mock_get(url, headers=None, timeout=30):
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

    with pytest.raises(massive.MassiveError) as exc_info:
        massive.fetch_day(dt.date(2024, 1, 2), data_dir=d_dir, key_path=key_file)

    err = exc_info.value
    assert err.status_code == 429
    assert guard._is_block(err) is True


def test_load_closes(tmp_data_dir):
    d_dir = tmp_data_dir / "market_history"
    d_dir.mkdir(parents=True, exist_ok=True)

    df1 = pd.DataFrame([{"ticker": "AAPL", "close": 150.0}, {"ticker": "MSFT", "close": 400.0}])
    df2 = pd.DataFrame([{"ticker": "AAPL", "close": 152.0}, {"ticker": "MSFT", "close": 402.0}])
    df3 = pd.DataFrame([{"ticker": "AAPL", "close": 155.0}, {"ticker": "MSFT", "close": 405.0}])

    df1.to_parquet(d_dir / "2024-01-02.parquet")
    df2.to_parquet(d_dir / "2024-01-03.parquet")
    df3.to_parquet(d_dir / "2024-01-04.parquet")

    # Full range
    wide = massive.load_closes(data_dir=d_dir)
    assert list(wide.index) == [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert wide.loc[dt.date(2024, 1, 2), "AAPL"] == 150.0
    assert wide.loc[dt.date(2024, 1, 4), "MSFT"] == 405.0

    # With start filter
    wide_start = massive.load_closes(start=dt.date(2024, 1, 3), data_dir=d_dir)
    assert list(wide_start.index) == [dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
