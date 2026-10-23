"""FRED provider for the market header: VIX and high-yield bond spread from St. Louis Fed.

Fetches VIXCLS and BAMLH0A0HYM2 from https://fred.stlouisfed.org/graph/fredgraph.csv
through the request guard, cached once per New York day in _data/fred/fred.parquet.
Module only; not connected to the UI or store yet.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from screener.data import guard

log = logging.getLogger(__name__)

SERIES = {
    "vix": "VIXCLS",
    "hy_spread": "BAMLH0A0HYM2",
}


class FredError(RuntimeError):
    """FRED HTTP or network error. `status_code` lets the request guard see 403/429 blocks."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _default_dir() -> Path:
    from screener.data import store  # resolved at call time
    return store.DATA_DIR / "fred"


def build_url(ids: str | Sequence[str] | None = None) -> str:
    """Build FRED CSV graph export URL.

    Note: Multi-id support (requesting multiple series in a single call via comma-separated IDs)
    must be checked live before wiring.
    """
    if ids is None:
        series_ids = list(SERIES.values())
    elif isinstance(ids, str):
        series_ids = [ids]
    else:
        series_ids = list(ids)

    id_param = ",".join(series_ids)
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id_param}"


def _parse_csv(csv_text: str) -> pd.DataFrame:
    """Parse FRED CSV string into a DataFrame indexed by date with series keys as columns."""
    try:
        df_raw = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise FredError(f"Failed to parse FRED CSV: {e}") from e

    if df_raw.empty:
        return pd.DataFrame(columns=list(SERIES.keys()))

    date_col = None
    if "observation_date" in df_raw.columns:
        date_col = "observation_date"
    elif "DATE" in df_raw.columns:
        date_col = "DATE"

    if date_col is None:
        raise FredError("Missing date column in FRED CSV (expected 'observation_date' or 'DATE')")

    df_out = pd.DataFrame()
    df_out["date"] = pd.to_datetime(df_raw[date_col], errors="coerce").dt.date

    inv_series = {v: k for k, v in SERIES.items()}
    for series_id, key in inv_series.items():
        if series_id in df_raw.columns:
            s_clean = df_raw[series_id].astype(str).str.strip().replace({".": np.nan, "": np.nan})
            df_out[key] = pd.to_numeric(s_clean, errors="coerce")
        else:
            df_out[key] = np.nan

    df_out = df_out.dropna(subset=["date"]).set_index("date")
    return df_out


def _fetch_fred(ids: str | Sequence[str] | None = None, timeout: float = 30) -> pd.DataFrame:
    url = build_url(ids)
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        resp_obj = getattr(e, "response", None)
        status_code = getattr(resp_obj, "status_code", None)
        raise FredError(f"FRED network error: {e}", status_code=status_code) from e

    if resp.status_code != 200:
        raise FredError(f"FRED returned status code {resp.status_code}", status_code=resp.status_code)

    return _parse_csv(resp.text)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def update(today: dt.date | None = None) -> bool:
    """Fetch and cache FRED series if not updated today (New York day).

    Returns True when a new download occurred. Returns False when already cached today
    or when a guard/network error occurs (logging a warning without raising).
    """
    from screener.data import store
    if today is None:
        ny_today = store.market_date(dt.datetime.now().astimezone())
    else:
        ny_today = today

    d_dir = _default_dir()
    json_path = d_dir / "fred.json"
    parquet_path = d_dir / "fred.parquet"

    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("day") == ny_today.isoformat():
                return False
        except Exception:
            pass

    try:
        df = guard.call("fred", _fetch_fred)
    except guard.Wait as w:
        log.warning("FRED update waiting: %s", w)
        return False
    except Exception as e:
        log.warning("FRED update failed: %s", e)
        return False

    # An answer without data (an empty CSV, or a series missing from it) must not replace a good file.
    missing = [k for k in SERIES if k not in df.columns or df[k].dropna().empty]
    if df.empty or missing:
        log.warning("FRED answer has no data for %s; keeping the previous file", missing or list(SERIES))
        return False

    try:
        _atomic_parquet(df, parquet_path)
        meta_data = {
            "day": ny_today.isoformat(),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _atomic_write_text(json.dumps(meta_data, indent=2), json_path)
        return True
    except Exception as e:
        log.warning("Failed to save FRED data: %s", e)
        return False


def history(key: str, data_dir: Path | None = None) -> pd.Series:
    """Return stored series for key (vix or hy_spread). Returns empty Series if unavailable."""
    d_dir = data_dir or _default_dir()
    path = d_dir / "fred.parquet"
    if not path.exists():
        return pd.Series(dtype=float)

    try:
        df = pd.read_parquet(path)
        if key in df.columns:
            return df[key]
        return pd.Series(dtype=float)
    except Exception as e:
        log.warning("Failed to read FRED history: %s", e)
        return pd.Series(dtype=float)


def latest(data_dir: Path | None = None) -> dict:
    """Return latest observations and computed HY spread SMA20 metrics."""
    d_dir = data_dir or _default_dir()
    path = d_dir / "fred.parquet"

    res = {
        "vix": {"value": None, "date": None},
        "hy_spread": {"value": None, "date": None},
        "hy_spread_sma20": None,
        "hy_spread_above_sma20": None,
    }

    if not path.exists():
        return res

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.warning("Failed to read FRED parquet: %s", e)
        return res

    for key in SERIES:
        if key in df.columns:
            s = df[key].dropna()
            if not s.empty:
                last_idx = s.index[-1]
                date_str = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
                res[key] = {
                    "value": float(s.iloc[-1]),
                    "date": date_str,
                }

    if "hy_spread" in df.columns:
        s_hy = df["hy_spread"].dropna()
        if len(s_hy) >= 20:
            sma20 = float(s_hy.iloc[-20:].mean())
            res["hy_spread_sma20"] = sma20
            hy_val = res["hy_spread"]["value"]
            if hy_val is not None:
                res["hy_spread_above_sma20"] = bool(hy_val > sma20)

    return res
