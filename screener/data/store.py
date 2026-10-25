"""
Local storage: latest-snapshot cache, one snapshot per day, and offline fallback.

  _data/cache/latest.parquet      last successful fetch (+ latest_meta.json with the timestamp)
  _data/snapshots/YYYY-MM-DD.parquet   last fetch of each day — builds a private history over time
  _data/snapshots/YYYY-MM-DD.json      metadata of that snapshot (fetched_at, rows, data source)

The data source ({"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}) comes
from the provider via df.attrs["source"].

load_universe() returns the cache while it is fresh, fetches otherwise, and falls back to the
last cache (flagged stale) when the provider fails, so the app opens even without internet.
A cache saved by an older version that lacks required columns is refetched (still usable offline).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("HALGOM_DATA_DIR", Path(__file__).resolve().parent.parent / "_data"))
MARKET_TZ = "America/New_York"      # snapshot days are US market days


@dataclass
class Snapshot:
    df: pd.DataFrame
    fetched_at: dt.datetime
    source: str          # "live" | "cache" | "stale"
    error: str | None = None
    meta: dict | None = None   # data source metadata (provider, update_mode, delay_min)

    @property
    def age(self) -> dt.timedelta:
        """Negative when the PC clock went back since the fetch: callers treat that as stale."""
        now = dt.datetime.now().astimezone() if self.fetched_at.tzinfo is not None else dt.datetime.now()
        return now - self.fetched_at


class Store:
    def __init__(self, root: Path = DATA_DIR):
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.snap_dir = self.root / "snapshots"
        self.latest_path = self.cache_dir / "latest.parquet"
        self.meta_path = self.cache_dir / "latest_meta.json"

    def load_latest(self) -> tuple[pd.DataFrame, dt.datetime, dict] | None:
        if self.latest_path.exists() and self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                return (pd.read_parquet(self.latest_path), _local(meta["fetched_at"]), meta.get("source") or {})
            except Exception as e:  # corrupted cache must never stop the app
                log.warning("cannot read cache: %s", e)

        # Fall back to the newest readable snapshot in _data/snapshots/
        for day in reversed(self.snapshot_dates()):
            try:
                df = self.load_snapshot(day)
                meta = self.snapshot_meta(day)
                if "fetched_at" in meta:
                    fetched_at = _local(meta["fetched_at"])
                else:
                    fetched_at = dt.datetime(day.year, day.month, day.day)
                source = meta.get("source") or {}
                log.info("falling back to snapshot from %s", day)
                return (df, fetched_at, source)
            except Exception as e:
                log.warning("cannot read snapshot %s: %s", day, e)

        return None

    def save(self, df: pd.DataFrame, fetched_at: dt.datetime) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"fetched_at": fetched_at.isoformat(timespec="seconds"), "rows": len(df),
                           "source": df.attrs.get("source") or {}})
        _atomic_parquet(df, self.latest_path)
        _atomic_write_text(meta, self.meta_path)
        day = session_date(fetched_at).isoformat()
        _atomic_parquet(df, self.snap_dir / f"{day}.parquet")
        _atomic_write_text(meta, self.snap_dir / f"{day}.json")

    def latest_fetched_at(self) -> dt.datetime | None:
        """Time of the last successful fetch, from the small meta file (no parquet read)."""
        try:
            return dt.datetime.fromisoformat(json.loads(self.meta_path.read_text(encoding="utf-8"))["fetched_at"])
        except (OSError, ValueError, KeyError):
            return None

    def snapshot_dates(self) -> list[dt.date]:
        if not self.snap_dir.exists():
            return []
        out = []
        for p in self.snap_dir.glob("*.parquet"):
            try:
                out.append(dt.date.fromisoformat(p.stem))
            except ValueError:
                pass
        return sorted(out)

    def snapshot_path(self, day: dt.date) -> Path:
        return self.snap_dir / f"{day.isoformat()}.parquet"

    def load_snapshot(self, day: dt.date, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.snapshot_path(day), columns=columns)

    def snapshot_meta(self, day: dt.date) -> dict:
        """Metadata of a snapshot; {} for snapshots saved before metadata existed."""
        try:
            return json.loads((self.snap_dir / f"{day.isoformat()}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def previous_snapshot_date(self, before: dt.date) -> dt.date | None:
        """The last snapshot day strictly before `before` (Friday's, on a Monday)."""
        days = [d for d in self.snapshot_dates() if d < before]
        return days[-1] if days else None


def market_date(ts: dt.datetime) -> dt.date:
    """New York date of a fetch time (naive = local time). A fetch after midnight in Israel still
    belongs to the same US session, so it must not start a new snapshot day."""
    try:
        return ts.astimezone(ZoneInfo(MARKET_TZ)).date()
    except Exception:            # no time-zone database: the local date
        return ts.date()


def session_date(ts: dt.datetime | dt.date) -> dt.date:
    """Trading session date of a fetch timestamp or date.
    On a weekday from the 09:30 NY open onwards, the session is that NY date.
    Before 09:30 NY open, or on a weekend, the session is the previous weekday.
    US market holidays are ignored."""
    if isinstance(ts, dt.date) and not isinstance(ts, dt.datetime):
        ts = dt.datetime(ts.year, ts.month, ts.day, 12, 0, tzinfo=ZoneInfo(MARKET_TZ))
    try:
        if ts.tzinfo is None:
            ts = ts.astimezone().astimezone(ZoneInfo(MARKET_TZ))
        else:
            ts = ts.astimezone(ZoneInfo(MARKET_TZ))
    except Exception:
        pass

    d = ts.date()
    open_time = dt.time(9, 30)

    # weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    if ts.weekday() in (5, 6):
        days_back = 1 if ts.weekday() == 5 else 2
        return d - dt.timedelta(days=days_back)
    else:
        if ts.time() >= open_time:
            return d
        else:
            if ts.weekday() == 0:
                return d - dt.timedelta(days=3)
            else:
                return d - dt.timedelta(days=1)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    df = df.copy()
    df.attrs = {}        # metadata lives in the JSON next to the file
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_text(text: str, path: Path) -> None:
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


FETCH_LOCK = threading.Lock()   # one fetch at a time for the whole server (all browser tabs)


def _local(iso: str) -> dt.datetime:
    """A stored fetch time: tz-aware in the PC's time zone (the header shows local time), or naive as
    written by versions before 2026-10-20."""
    ts = dt.datetime.fromisoformat(iso)
    return ts.astimezone() if ts.tzinfo is not None else ts


def load_universe(store: Store, fetch: Callable[..., pd.DataFrame], max_age: dt.timedelta,
                  force: bool = False, required: Iterable[str] = (), timeout: float | None = None) -> Snapshot:
    """`timeout` (seconds) is passed to fetch() when given: the app shortens it for retries while offline."""
    from screener.data import guard

    def fresh(cached) -> bool:
        complete = cached is not None and set(required) <= set(cached[0].columns)
        if not complete or force:
            return False
        age = Snapshot(cached[0], cached[1], "cache").age
        return dt.timedelta(0) <= age < max_age

    cached = store.load_latest()
    if fresh(cached):
        return Snapshot(cached[0], cached[1], "cache", meta=cached[2])
    with FETCH_LOCK:
        cached = store.load_latest()         # another tab may have fetched while this one waited
        if fresh(cached):
            return Snapshot(cached[0], cached[1], "cache", meta=cached[2])
        now = dt.datetime.now().astimezone()     # tz-aware: a clock change (DST) cannot make it "fresh"
        try:
            df = fetch() if timeout is None else fetch(timeout=timeout)
            if cached is not None and len(df) < 0.5 * len(cached[0]):
                raise ValueError(f"universe shrank to {len(df)} rows from {len(cached[0])}")
        except guard.Wait as w:
            if cached is None:
                raise
            return Snapshot(cached[0], cached[1], "stale", error=guard.wait_message(w), meta=cached[2])
        except Exception as e:
            log.exception("fetch failed")
            if cached is None:
                raise
            w = guard.check_wait("tradingview")
            err_msg = guard.wait_message(w) if w else str(e)
            return Snapshot(cached[0], cached[1], "stale", error=err_msg, meta=cached[2])
        store.save(df, now)
        return Snapshot(df, now, "live", meta=df.attrs.get("source") or {})
