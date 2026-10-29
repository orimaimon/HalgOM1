"""Bank of Israel USD/ILS exchange rates provider.

Fetches daily representative USD/ILS exchange rates from Bank of Israel SDMX endpoint
via the request guard (`screener.data.guard`).
Data is cached in `_data/boi/usd_ils.parquet` and state tracked in `_data/boi/state.json`.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from screener.data import guard

log = logging.getLogger(__name__)

BOI_URL_TEMPLATE = (
    "https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/"
    "BOI.STATISTICS/EXR/1.0/RER_USD_ILS?startperiod={start}&endperiod={end}&format=csv"
)


class BoiError(RuntimeError):
    """Bank of Israel HTTP, network, or format error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _boi_dir() -> Path:
    from screener.data import store
    return Path(store.DATA_DIR) / "boi"


def _parse_csv(csv_text: str) -> pd.Series:
    try:
        df = pd.read_csv(io.StringIO(csv_text), dtype=str)
    except Exception as e:
        raise BoiError(f"Failed to parse BOI CSV: {e}") from e

    if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise BoiError("BOI CSV missing required columns (TIME_PERIOD, OBS_VALUE) or empty")

    dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce").dt.date
    s_clean = df["OBS_VALUE"].astype(str).str.strip().replace({".": np.nan, "": np.nan})
    values = pd.to_numeric(s_clean, errors="coerce")

    valid_mask = dates.notna() & values.notna()
    if not valid_mask.any():
        raise BoiError("BOI CSV contains no valid date/value rows")

    s = pd.Series(values[valid_mask].values, index=dates[valid_mask], dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        raise BoiError("BOI CSV contains no valid rows after deduplication")
    return s


def _raw_fetch(start: dt.date, end: dt.date, timeout: float = 30) -> pd.Series:
    url = BOI_URL_TEMPLATE.format(start=start.isoformat(), end=end.isoformat())
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        resp_obj = getattr(e, "response", None)
        status_code = getattr(resp_obj, "status_code", None)
        raise BoiError(f"BOI network error: {e}", status_code=status_code) from e

    if resp.status_code != 200:
        raise BoiError(f"BOI returned status code {resp.status_code}", status_code=resp.status_code)

    return _parse_csv(resp.text)


def fetch_rates(start: dt.date, end: dt.date) -> pd.Series:
    """Fetch USD/ILS exchange rates for date range through the request guard."""
    return guard.call("boi", _raw_fetch, start, end)


def _atomic_parquet(s: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"rate": s})
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        df.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_cache(b_dir: Path | None = None) -> pd.Series:
    d = b_dir or _boi_dir()
    p = d / "usd_ils.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_parquet(p)
        if "rate" in df.columns:
            s = df["rate"]
            # Ensure index elements are dt.date
            idx = [d.date() if isinstance(d, (dt.datetime, pd.Timestamp)) else d for d in s.index]
            s.index = idx
            return s.sort_index()
        return pd.Series(dtype=float)
    except Exception as e:
        log.warning("Failed to load BOI parquet cache: %s", e)
        return pd.Series(dtype=float)


def update(start: dt.date, today: dt.date | None = None) -> bool:
    """Fetch missing USD/ILS rates starting from `start` (or after last cached date) up to `today`.

    Runs at most once per New York day.
    Returns True if new rates were fetched and saved; False otherwise.
    Never raises on guard.Wait or errors, preserving existing cache.
    """
    from screener.data import store
    if today is None:
        ny_today = store.market_date(dt.datetime.now().astimezone())
    else:
        ny_today = today

    b_dir = _boi_dir()
    state_path = b_dir / "state.json"
    parquet_path = b_dir / "usd_ils.parquet"

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("last_check") == ny_today.isoformat():
                return False
        except Exception:
            pass

    cached = _load_cache(b_dir)
    fetch_start = start
    # Only the missing tail, unless `start` is older than the cache (a trade imported later with an older date):
    # then the whole range again, still one request (a few hundred rows a year).
    if not cached.empty and start >= min(cached.index):
        next_day = max(cached.index) + dt.timedelta(days=1)
        if next_day > fetch_start:
            fetch_start = next_day

    if fetch_start > ny_today:
        # Mark check for today and return
        _atomic_write_text(json.dumps({"last_check": ny_today.isoformat()}, indent=2), state_path)
        return False

    try:
        new_rates = fetch_rates(fetch_start, ny_today)
    except guard.Wait as w:
        log.warning("BOI update waiting: %s", w)
        return False
    except Exception as e:
        log.warning("BOI update failed: %s", e)
        return False

    if new_rates.empty:
        log.warning("BOI response had no rates")
        return False

    # Filter out any rates already in cache if needed, or combine
    if not cached.empty:
        combined = pd.concat([cached, new_rates])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        # Check if combined actually has new rows beyond cached
        if len(combined) <= len(cached) and (combined.index == cached.index).all():
            _atomic_write_text(json.dumps({"last_check": ny_today.isoformat()}, indent=2), state_path)
            return False
    else:
        combined = new_rates

    try:
        _atomic_parquet(combined, parquet_path)
        _atomic_write_text(json.dumps({"last_check": ny_today.isoformat()}, indent=2), state_path)
        return True
    except Exception as e:
        log.warning("Failed to save BOI rates cache: %s", e)
        return False


def rate_on(day: dt.date, rates: pd.Series | None = None) -> float | None:
    """Return the USD/ILS rate on `day`, or closest preceding business day up to 7 days back."""
    s = rates if rates is not None else _load_cache()
    if s.empty:
        return None

    # Check up to 7 days back
    for offset in range(8):
        check_day = day - dt.timedelta(days=offset)
        if check_day in s.index and pd.notna(s.loc[check_day]):
            return float(s.loc[check_day])
    return None


def load_rates() -> pd.Series:
    """Return cached USD/ILS exchange rates Series indexed by date."""
    return _load_cache()


def latest_rate() -> tuple[dt.date, float] | None:
    """Return tuple of (latest_date, latest_rate) from cached USD/ILS rates, or None."""
    s = _load_cache()
    if s.empty:
        return None
    valid = s.dropna()
    if valid.empty:
        return None
    last_d = max(valid.index)
    return (last_d, float(valid.loc[last_d]))


def get_rate_for_date(date: dt.date, rates: pd.Series | None = None) -> float:
    """Return official USD/ILS rate on `date`, or forward-fill from the most recent preceding valid rate."""
    s = rates if rates is not None else load_rates()
    if s.empty:
        return float("nan")

    valid = s.dropna()
    if valid.empty:
        return float("nan")

    target_date = date.date() if isinstance(date, dt.datetime) else date

    if len(valid) > 0 and isinstance(valid.index[0], (dt.datetime, pd.Timestamp)):
        idx = [d.date() if isinstance(d, (dt.datetime, pd.Timestamp)) else d for d in valid.index]
        valid = pd.Series(valid.values, index=idx, dtype=float)

    past = valid[valid.index <= target_date]
    if past.empty:
        return float("nan")

    return float(past.iloc[-1])


def get_rates_for_range(
    start_date: dt.date,
    end_date: dt.date,
    rates: pd.Series | None = None,
) -> pd.Series:
    """Return daily reindexed series with forward-filled rates for all calendar days in range."""
    s = rates if rates is not None else load_rates()

    start = start_date.date() if isinstance(start_date, dt.datetime) else start_date
    end = end_date.date() if isinstance(end_date, dt.datetime) else end_date

    if start > end:
        return pd.Series(dtype=float)

    n_days = (end - start).days + 1
    all_dates = [start + dt.timedelta(days=i) for i in range(n_days)]

    if s.empty:
        return pd.Series(np.nan, index=all_dates, dtype=float)

    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=all_dates, dtype=float)

    if len(valid) > 0 and isinstance(valid.index[0], (dt.datetime, pd.Timestamp)):
        idx = [d.date() if isinstance(d, (dt.datetime, pd.Timestamp)) else d for d in valid.index]
        valid = pd.Series(valid.values, index=idx, dtype=float)

    valid = valid[~valid.index.duplicated(keep="last")].sort_index()

    relevant_valid = valid[valid.index <= end]
    if relevant_valid.empty:
        return pd.Series(np.nan, index=all_dates, dtype=float)

    combined_idx = sorted(set(relevant_valid.index) | set(all_dates))

    filled = relevant_valid.reindex(combined_idx).ffill()
    result = filled.reindex(all_dates)
    return result
