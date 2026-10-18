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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "_data"


@dataclass
class Snapshot:
    df: pd.DataFrame
    fetched_at: dt.datetime
    source: str          # "live" | "cache" | "stale"
    error: str | None = None
    meta: dict | None = None   # data source metadata (provider, update_mode, delay_min)

    @property
    def age(self) -> dt.timedelta:
        return dt.datetime.now() - self.fetched_at


class Store:
    def __init__(self, root: Path = DATA_DIR):
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.snap_dir = self.root / "snapshots"
        self.latest_path = self.cache_dir / "latest.parquet"
        self.meta_path = self.cache_dir / "latest_meta.json"

    def load_latest(self) -> tuple[pd.DataFrame, dt.datetime, dict] | None:
        if not (self.latest_path.exists() and self.meta_path.exists()):
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return (pd.read_parquet(self.latest_path), dt.datetime.fromisoformat(meta["fetched_at"]),
                    meta.get("source") or {})
        except Exception as e:  # corrupted cache must never stop the app
            log.warning("cannot read cache: %s", e)
            return None

    def save(self, df: pd.DataFrame, fetched_at: dt.datetime) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"fetched_at": fetched_at.isoformat(timespec="seconds"), "rows": len(df),
                           "source": df.attrs.get("source") or {}})
        _atomic_parquet(df, self.latest_path)
        self.meta_path.write_text(meta, encoding="utf-8")
        day = fetched_at.date().isoformat()
        _atomic_parquet(df, self.snap_dir / f"{day}.parquet")
        (self.snap_dir / f"{day}.json").write_text(meta, encoding="utf-8")

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

    def load_snapshot(self, day: dt.date) -> pd.DataFrame:
        return pd.read_parquet(self.snap_dir / f"{day.isoformat()}.parquet")

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


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df = df.copy()
    df.attrs = {}        # metadata lives in the JSON next to the file
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_universe(store: Store, fetch: Callable[[], pd.DataFrame], max_age: dt.timedelta,
                  force: bool = False, required: Iterable[str] = ()) -> Snapshot:
    cached = store.load_latest()
    now = dt.datetime.now()
    complete = cached is not None and set(required) <= set(cached[0].columns)
    if cached is not None and complete and not force and now - cached[1] < max_age:
        return Snapshot(cached[0], cached[1], "cache", meta=cached[2])
    try:
        df = fetch()
    except Exception as e:
        log.exception("fetch failed")
        if cached is None:
            raise
        return Snapshot(cached[0], cached[1], "stale", str(e), meta=cached[2])
    store.save(df, now)
    return Snapshot(df, now, "live", meta=df.attrs.get("source") or {})
