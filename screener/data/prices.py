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


COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
PAUSE = 0.5          # seconds between downloads: Yahoo rate-limits, never download in parallel


def _download(ticker: str, start: dt.date) -> pd.DataFrame:
    """Daily bars from `start`, in our column names, naive date index. Empty when Yahoo has nothing."""
    dl = yf.download(_ticker_to_yf(ticker), start=start.isoformat(), actions=True, auto_adjust=False,
                     threads=False, progress=False)
    if dl is None or dl.empty:
        return pd.DataFrame(columns=COLUMNS)
    if isinstance(dl.columns, pd.MultiIndex):             # (field, ticker) even for one ticker
        dl.columns = dl.columns.droplevel(1)
    dl = dl.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends",
                            "Stock Splits": "splits"})
    for c in COLUMNS:
        if c not in dl.columns:
            dl[c] = 0.0 if c in ("dividends", "splits") else pd.NA
    dl.index.name = "date"
    if dl.index.tz is not None:
        dl.index = dl.index.tz_localize(None)
    return dl[COLUMNS]


def _has_actions(df: pd.DataFrame) -> bool:
    """A dividend or a split: Yahoo then re-adjusts the whole history (close for splits, adj_close for both)."""
    return bool((df["dividends"].fillna(0) > 0).any() or (~df["splits"].fillna(0).isin([0.0, 1.0])).any())


def _write(ticker: str, df: pd.DataFrame) -> None:
    tmp = _parquet_path(ticker).with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, _parquet_path(ticker))              # a crash mid-write never leaves a broken file


def ensure(tickers: list[str], start: dt.date) -> tuple[list[str], list[str]]:
    """Bring each ticker up to date. Returns (updated, failed); never raises.

    - At most one check per ticker per New York day (update_state.json).
    - The last stored day is downloaded again: a bar stored during the session was not final.
    - Yahoo's history is back-adjusted: a new dividend or split changes adj_close (and, for a split,
      close) of every earlier day. Then the whole history is downloaded again instead of appended.
    - One ticker at a time with a pause in between; nothing downloaded for a new ticker is a failure
      (yfinance returns an empty frame rather than raising when it is rate-limited).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state()
    today_ny = dt.datetime.now(NY_TZ).date().isoformat()
    updated: list[str] = []
    failed: list[str] = []
    todo = [t for t in dict.fromkeys(tickers) if state.get(t) != today_ny]
    for n, ticker in enumerate(todo):
        if n:
            time.sleep(PAUSE)
        try:
            old = history(ticker)
            if old.empty:
                new = _download(ticker, start)
                if new.empty:
                    raise ValueError("no data")
            else:
                last = old.index.max()
                recent = _download(ticker, last.date())
                if _has_actions(recent[recent.index > last]):
                    new = _download(ticker, old.index.min().date())
                    if new.empty:
                        raise ValueError("no data on the full re-download")
                else:
                    new = pd.concat([old, recent]) if not recent.empty else old
                    new = new[~new.index.duplicated(keep="last")].sort_index()   # re-downloaded day wins
            _write(ticker, new)
            state[ticker] = today_ny
            updated.append(ticker)
        except Exception as e:                           # keep what is stored, try again next time
            log.warning("price history: %s not updated: %s", ticker, e)
            failed.append(ticker)
        _write_state(state)
    return updated, failed
