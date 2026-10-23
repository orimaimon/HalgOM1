"""
Massive (formerly Polygon.io) grouped daily EOD market history provider.

Fetches grouped daily aggregates (the whole US market in one request per day) to build a local EOD history.
adjusted=true is applied by Massive when each day is fetched, so days fetched before a later split are not
adjusted for it. Consumers undo split jumps with track.split_adjusted() (already in the repo).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from screener.data import guard

log = logging.getLogger(__name__)

# Massive is the new name of Polygon.io; base URL must be checked live before the module is wired in.
BASE_URL = "https://api.polygon.io"

COLUMNS = ["ticker", "open", "high", "low", "close", "volume", "vwap", "trades"]
# An empty answer for a recent day may only mean the data is not published yet: record a day as a holiday
# (never asked again) only when it is at least this many days old.
EMPTY_FINAL_DAYS = 3


class MassiveError(RuntimeError):
    """Massive API error. `status_code` is set for HTTP errors so the request guard sees blocks (403/429)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def get_api_key(key_path: Path | str | None = None) -> str | None:
    """Read API key from %USERPROFILE%\\.halgom\\massive_key.txt (first non-empty line)."""
    if key_path is None:
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            k_path = Path(user_profile) / ".halgom" / "massive_key.txt"
        else:
            k_path = Path.home() / ".halgom" / "massive_key.txt"
    else:
        k_path = Path(key_path)

    if not k_path.exists():
        return None

    try:
        content = k_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            s = line.strip()
            if s:
                return s
    except OSError as e:
        log.warning("Failed to read Massive API key at %s: %s", k_path, e)
        return None

    return None


def _default_dir() -> Path:
    from screener.data import store
    return store.DATA_DIR / "market_history"


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_empty_days(data_dir: Path) -> set[str]:
    path = data_dir / "empty_days.json"
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return set(str(item) for item in raw)
    except Exception as e:
        log.warning("Failed to load empty_days.json at %s: %s", path, e)
    return set()


def _add_empty_day(data_dir: Path, day_str: str) -> None:
    days = _load_empty_days(data_dir)
    days.add(day_str)
    _atomic_write_json(sorted(days), data_dir / "empty_days.json")


def fetch_day(
    day: dt.date,
    data_dir: Path | None = None,
    key_path: Path | str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch grouped daily stock aggregate for one day from Massive.

    Returns parsed DataFrame on success or an empty DataFrame for a holiday / missing key.
    Writes results to _data/market_history/YYYY-MM-DD.parquet or records in empty_days.json.
    """
    d_dir = data_dir or _default_dir()
    day_str = day.isoformat()

    empty_days = _load_empty_days(d_dir)
    if day_str in empty_days:
        return pd.DataFrame(columns=COLUMNS)

    parquet_path = d_dir / f"{day_str}.parquet"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            log.warning("Failed to read existing parquet for %s: %s", day_str, e)

    key = get_api_key(key_path)
    if not key:
        log.warning("Massive API key missing or empty.")
        return pd.DataFrame(columns=COLUMNS)

    url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{day_str}?adjusted=true"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def _do_request() -> dict[str, Any]:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise MassiveError(f"network error: {e}") from e

        if resp.status_code != 200:
            raise MassiveError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )

        try:
            payload = resp.json()
        except Exception as e:
            raise MassiveError(f"invalid JSON response: {e}") from e

        if not isinstance(payload, dict):
            raise MassiveError("invalid response structure: not a dict")

        return payload

    payload = guard.call("massive", _do_request)

    results_count = payload.get("resultsCount", 0)
    results = payload.get("results")

    if results_count == 0 or not results:
        from screener.data import store
        ny_today = store.market_date(dt.datetime.now().astimezone())
        if (ny_today - day).days >= EMPTY_FINAL_DAYS:
            _add_empty_day(d_dir, day_str)
        return pd.DataFrame(columns=COLUMNS)

    parsed_rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        parsed_rows.append({
            "ticker": str(r.get("T", "") or "").strip(),
            "open": r.get("o"),
            "high": r.get("h"),
            "low": r.get("l"),
            "close": r.get("c"),
            "volume": r.get("v"),
            "vwap": r.get("vw"),
            "trades": r.get("n"),
        })

    df = pd.DataFrame(parsed_rows, columns=COLUMNS)
    _atomic_parquet(df, parquet_path)
    return df


def update(
    days_back: int = 500,
    today: dt.date | None = None,
    data_dir: Path | None = None,
    key_path: Path | str | None = None,
) -> int:
    """Fill missing weekdays from yesterday (New York date) back to days_back days, newest first.

    Returns the number of days fetched. Stops quietly on guard.Wait and re-raises other exceptions.
    """
    key = get_api_key(key_path)
    if not key:
        log.warning("Massive API key missing or empty.")
        return 0

    from screener.data import store

    if today is not None:
        ref_today = today
    else:
        ref_today = store.market_date(dt.datetime.now().astimezone())

    d_dir = data_dir or _default_dir()
    empty_days = _load_empty_days(d_dir)

    fetched = 0
    for i in range(1, days_back + 1):
        target_day = ref_today - dt.timedelta(days=i)
        if target_day.weekday() in (5, 6):
            continue

        t_str = target_day.isoformat()
        if t_str in empty_days:
            continue

        if (d_dir / f"{t_str}.parquet").exists():
            continue

        try:
            fetch_day(target_day, data_dir=d_dir, key_path=key_path)
            fetched += 1
        except guard.Wait as w:
            log.info("Massive update stopped by guard: %s", w)
            break
        except Exception:
            raise

    return fetched


def load_closes(
    start: dt.date | None = None,
    field: str = "close",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Build wide DataFrame (index = sorted trading days, columns = tickers) for field."""
    d_dir = data_dir or _default_dir()
    if not d_dir.exists():
        return pd.DataFrame()

    day_files: list[tuple[dt.date, Path]] = []
    for p in d_dir.glob("*.parquet"):
        try:
            d = dt.date.fromisoformat(p.stem)
            if start is None or d >= start:
                day_files.append((d, p))
        except ValueError:
            pass

    if not day_files:
        return pd.DataFrame()

    day_files.sort(key=lambda x: x[0])

    data_dict: dict[dt.date, pd.Series] = {}
    for day, path in day_files:
        try:
            df = pd.read_parquet(path, columns=["ticker", field])
            if not df.empty and "ticker" in df.columns and field in df.columns:
                data_dict[day] = df.drop_duplicates("ticker", keep="last").set_index("ticker")[field]
        except Exception as e:
            log.warning("Failed to load parquet %s for load_closes: %s", path, e)

    if not data_dict:
        return pd.DataFrame()

    res = pd.DataFrame(data_dict).T
    res.index = pd.DatetimeIndex(pd.to_datetime(res.index), name="date")     # what history_signals expects
    return res.sort_index()
