from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "_data" / "history"
STATE_FILE = DATA_DIR / "update_state.json"
NY_TZ = ZoneInfo("America/New_York")


def _read_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(state: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = STATE_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temp_file, STATE_FILE)


def _parquet_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def _ticker_to_yf(ticker: str) -> str:
    if ticker == "BRK.B":
        return "BRK-B"
    return ticker.replace(".", "-")


def history(ticker: str) -> pd.DataFrame:
    """Return the stored history for the ticker, or an empty DataFrame if none exists."""
    path = _parquet_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def close_on(ticker: str, day: dt.date) -> float | None:
    """Return the close on that day or the last trading day before it."""
    df = history(ticker)
    if df.empty:
        return None
    # Filter to dates <= day
    # df.index is tz-aware or tz-naive? We'll make it naive Date in index, or dt.datetime.
    # Usually we can compare with strings or datetime objects.
    day_ts = pd.Timestamp(day)
    df_past = df[df.index <= day_ts]
    if df_past.empty:
        return None
    return float(df_past["close"].iloc[-1])


def split_factor(ticker: str, since: dt.date) -> float:
    """The product of splits after `since`, to correct an old purchase price and quantity."""
    df = history(ticker)
    if df.empty:
        return 1.0
    since_ts = pd.Timestamp(since)
    df_future = df[df.index > since_ts]
    if df_future.empty:
        return 1.0

    factor = 1.0
    for split in df_future["splits"]:
        if split > 0.0 and split != 1.0:
            factor *= split
    return factor


def total_return(ticker: str, start: dt.date, end: dt.date) -> float | None:
    """From adj_close. Return between start and end (using the close on or before these dates)."""
    df = history(ticker)
    if df.empty:
        return None

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    df_start = df[df.index <= start_ts]
    df_end = df[df.index <= end_ts]

    if df_start.empty or df_end.empty:
        return None

    adj_start = df_start["adj_close"].iloc[-1]
    adj_end = df_end["adj_close"].iloc[-1]

    if pd.isna(adj_start) or adj_start == 0.0:
        return None

    return float(adj_end / adj_start) - 1.0


def ensure(tickers: list[str], start: dt.date) -> tuple[list[str], list[str]]:
    """
    Brings each ticker up to date from `start` (or from its last stored day).
    Returns (updated, failed).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state()
    today_ny = dt.datetime.now(NY_TZ).date().isoformat()

    to_update = {}
    for ticker in tickers:
        if state.get(ticker) == today_ny:
            continue

        df = history(ticker)
        if df.empty:
            ticker_start = start
        else:
            # Get the last date in history and add 1 day
            last_date = df.index.max().date()
            ticker_start = last_date + dt.timedelta(days=1)

        # We only fetch if ticker_start <= today.
        # But even if it is today, there might be new data.
        # We'll just ask yfinance for start=ticker_start.
        # If ticker_start > today_ny, it might just return empty, which is fine.
        to_update[ticker] = ticker_start

    updated = []
    failed = []

    if not to_update:
        return updated, failed

    # Process in batches
    batch_size = 20
    tickers_to_update = list(to_update.keys())

    for i in range(0, len(tickers_to_update), batch_size):
        batch = tickers_to_update[i:i + batch_size]

        for ticker in batch:
            ticker_start = to_update[ticker]
            yf_ticker = _ticker_to_yf(ticker)

            try:
                # yf.download signature requires start as a string or datetime
                # auto_adjust=False, actions=True needed to get Adj Close, Dividends, Stock Splits
                start_str = ticker_start.isoformat()
                dl = yf.download(
                    yf_ticker,
                    start=start_str,
                    actions=True,
                    auto_adjust=False,
                    threads=False,
                    progress=False
                )

                if not dl.empty:
                    # Rename columns to our format
                    # Columns might be a MultiIndex if yf.download thinks we requested multiple,
                    # but for a single ticker it's usually flat or we can flatten it.
                    if isinstance(dl.columns, pd.MultiIndex):
                        # Drop the ticker level (level 1)
                        dl.columns = dl.columns.droplevel(1)

                    dl = dl.rename(columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                        "Dividends": "dividends",
                        "Stock Splits": "splits"
                    })

                    # Ensure index is named "date" and we only keep dates
                    dl.index.name = "date"
                    # Remove timezone if any, just keep naive datetime
                    if dl.index.tz is not None:
                        dl.index = dl.index.tz_localize(None)

                    # Ensure only the correct columns exist, ignoring others
                    expected_cols = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
                    missing_cols = [c for c in expected_cols if c not in dl.columns]
                    for c in missing_cols:
                        dl[c] = 0.0 if c in ["dividends", "splits"] else pd.NA
                    dl = dl[expected_cols]

                    df_old = history(ticker)
                    if not df_old.empty:
                        # Ensure old index is naive
                        if df_old.index.tz is not None:
                            df_old.index = df_old.index.tz_localize(None)
                        df_new = pd.concat([df_old, dl])
                        # Remove duplicates, keeping the last (newest) downloaded data
                        df_new = df_new[~df_new.index.duplicated(keep='last')]
                        df_new = df_new.sort_index()
                    else:
                        df_new = dl

                    # Write to parquet atomically
                    temp_file = _parquet_path(ticker).with_suffix(".parquet.tmp")
                    df_new.to_parquet(temp_file)
                    os.replace(temp_file, _parquet_path(ticker))

                # Mark as updated today
                state[ticker] = today_ny
                updated.append(ticker)

            except Exception as e:
                log.error(f"Failed to update {ticker}: {e}")
                failed.append(ticker)

        # Save state after each batch
        _write_state(state)

        # Pause between batches
        if i + batch_size < len(tickers_to_update):
            time.sleep(1.0)

    return updated, failed
