import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from screener.data import prices

@pytest.fixture
def mock_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "PAUSE", 0)
    with patch("screener.data.prices.DATA_DIR", tmp_path):
        with patch("screener.data.prices.STATE_FILE", tmp_path / "update_state.json"):
            yield tmp_path

@pytest.fixture
def mock_yf_download():
    with patch("screener.data.prices.yf.download") as mock_dl:
        yield mock_dl

def create_mock_df(dates, close_vals, adj_close_vals, splits=None, tz_aware=False):
    if splits is None:
        splits = [0.0] * len(dates)

    idx = pd.DatetimeIndex(dates, name="Date")
    if tz_aware:
        idx = idx.tz_localize("America/New_York")

    df = pd.DataFrame({
        "Open": close_vals,
        "High": close_vals,
        "Low": close_vals,
        "Close": close_vals,
        "Adj Close": adj_close_vals,
        "Volume": [1000] * len(dates),
        "Dividends": [0.0] * len(dates),
        "Stock Splits": splits,
    }, index=idx)
    return df

def test_first_download(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(
        ["2023-01-03", "2023-01-04"],
        [100.0, 101.0],
        [100.0, 101.0]
    )

    updated, failed = prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    assert updated == ["AAPL"]
    assert failed == []

    df = prices.history("AAPL")
    assert not df.empty
    assert list(df.index) == [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")]
    assert df.iloc[0]["close"] == 100.0

def test_incremental_update(mock_data_dir, mock_yf_download):
    # First download
    mock_yf_download.return_value = create_mock_df(["2023-01-03"], [100.0], [100.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    # Simulate a day passing by resetting the state JSON
    state_file = mock_data_dir / "update_state.json"
    state_file.write_text("{}")

    # Incremental update
    mock_yf_download.return_value = create_mock_df(["2023-01-04"], [101.0], [101.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    df = prices.history("AAPL")
    assert len(df) == 2
    assert df.iloc[-1]["close"] == 101.0

def test_already_checked_today(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(["2023-01-03"], [100.0], [100.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))
    assert mock_yf_download.call_count == 1

    # Call again, should be skipped
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))
    assert mock_yf_download.call_count == 1

def test_failed_download(mock_data_dir, mock_yf_download):
    # Setup initial data
    mock_yf_download.return_value = create_mock_df(["2023-01-03"], [100.0], [100.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    # Clear state to force update
    state_file = mock_data_dir / "update_state.json"
    state_file.write_text("{}")

    # Simulate failure
    mock_yf_download.side_effect = Exception("Network error")
    updated, failed = prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    assert updated == []
    assert failed == ["AAPL"]

    # Old data should remain
    df = prices.history("AAPL")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.0

def test_close_on(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(
        ["2023-01-03", "2023-01-04", "2023-01-06"],
        [100.0, 101.0, 103.0],
        [100.0, 101.0, 103.0]
    )
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    assert prices.close_on("AAPL", dt.date(2023, 1, 3)) == 100.0
    assert prices.close_on("AAPL", dt.date(2023, 1, 5)) == 101.0 # weekend/holiday fallback
    assert prices.close_on("AAPL", dt.date(2023, 1, 6)) == 103.0
    assert prices.close_on("AAPL", dt.date(2022, 12, 31)) is None

def test_split_factor(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        [100.0, 25.0, 26.0],
        [25.0, 25.0, 26.0],
        splits=[0.0, 4.0, 0.0]
    )
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    # Purchase on Jan 3, split on Jan 4
    assert prices.split_factor("AAPL", dt.date(2023, 1, 3)) == 4.0
    # Purchase on Jan 4, split already happened
    assert prices.split_factor("AAPL", dt.date(2023, 1, 4)) == 1.0

def test_ticker_mapping(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(["2023-01-03"], [100.0], [100.0])
    prices.ensure(["BRK.B"], dt.date(2023, 1, 1))

    mock_yf_download.assert_called_once_with(
        "BRK-B", start="2023-01-01", actions=True, auto_adjust=False, threads=False, progress=False
    )

    # Data is saved under BRK.B
    assert (mock_data_dir / "BRK.B.parquet").exists()

def test_total_return(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        [100.0, 25.0, 26.0],
        [25.0, 25.0, 26.0],
        splits=[0.0, 4.0, 0.0]
    )
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))

    # Return from Jan 3 to Jan 5. Adj close goes from 25.0 to 26.0
    ret = prices.total_return("AAPL", dt.date(2023, 1, 3), dt.date(2023, 1, 5))
    assert pytest.approx(ret) == (26.0 / 25.0) - 1.0


# ── review fixes ──────────────────────────────────────────────────────────────

def test_last_stored_bar_is_downloaded_again(mock_data_dir, mock_yf_download):
    """A bar stored during the session is not final: the next update replaces it."""
    mock_yf_download.return_value = create_mock_df(["2023-01-03", "2023-01-04"], [100.0, 101.0], [100.0, 101.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))
    (mock_data_dir / "update_state.json").write_text("{}")
    mock_yf_download.return_value = create_mock_df(["2023-01-04", "2023-01-05"], [102.5, 103.0], [102.5, 103.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))
    assert mock_yf_download.call_args.kwargs["start"] == "2023-01-04"
    df = prices.history("AAPL")
    assert df["close"].tolist() == [100.0, 102.5, 103.0]


def test_new_dividend_reloads_the_whole_adjusted_history(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = create_mock_df(["2023-01-03", "2023-01-04"], [100.0, 100.0], [100.0, 100.0])
    prices.ensure(["AAPL"], dt.date(2023, 1, 1))
    (mock_data_dir / "update_state.json").write_text("{}")
    recent = create_mock_df(["2023-01-04", "2023-01-05"], [100.0, 99.0], [100.0, 99.0])
    recent.loc[pd.Timestamp("2023-01-05"), "Dividends"] = 1.0
    full = create_mock_df(["2023-01-03", "2023-01-04", "2023-01-05"], [100.0, 100.0, 99.0], [99.0, 99.0, 99.0])
    mock_yf_download.side_effect = [recent, full]
    assert prices.ensure(["AAPL"], dt.date(2023, 1, 1)) == (["AAPL"], [])
    assert mock_yf_download.call_args.kwargs["start"] == "2023-01-03"          # from the first stored day
    assert prices.history("AAPL")["adj_close"].tolist() == [99.0, 99.0, 99.0]  # old rows re-adjusted


def test_empty_first_download_is_a_failure_and_is_retried(mock_data_dir, mock_yf_download):
    mock_yf_download.return_value = pd.DataFrame()             # what yfinance returns when rate-limited
    assert prices.ensure(["AAPL"], dt.date(2023, 1, 1)) == ([], ["AAPL"])
    mock_yf_download.return_value = create_mock_df(["2023-01-03"], [100.0], [100.0])
    assert prices.ensure(["AAPL"], dt.date(2023, 1, 1)) == (["AAPL"], [])      # same day: not marked done
