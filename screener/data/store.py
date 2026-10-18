"""
Local storage: latest-snapshot cache, one snapshot per day, and offline fallback.

  _data/cache/latest.parquet      last successful fetch (+ latest_meta.json with the timestamp)
  _data/snapshots/YYYY-MM-DD.parquet   last fetch of each day — builds a private history over time

load_universe() returns the cache while it is fresh, fetches otherwise, and falls back to the
last cache (flagged stale) when the provider fails, so the app opens even without internet.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "_data"


@dataclass
class Snapshot:
    df: pd.DataFrame
    fetched_at: dt.datetime
    source: str          # "live" | "cache" | "stale"
    error: str | None = None

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

    def load_latest(self) -> tuple[pd.DataFrame, dt.datetime] | None:
        if not (self.latest_path.exists() and self.meta_path.exists()):
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return pd.read_parquet(self.latest_path), dt.datetime.fromisoformat(meta["fetched_at"])
        except Exception as e:  # corrupted cache must never stop the app
            log.warning("cannot read cache: %s", e)
            return None

    def save(self, df: pd.DataFrame, fetched_at: dt.datetime) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(df, self.latest_path)
        self.meta_path.write_text(json.dumps({"fetched_at": fetched_at.isoformat(timespec="seconds"),
                                              "rows": len(df)}), encoding="utf-8")
        _atomic_parquet(df, self.snap_dir / f"{fetched_at.date().isoformat()}.parquet")

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


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_universe(store: Store, fetch: Callable[[], pd.DataFrame], max_age: dt.timedelta,
                  force: bool = False) -> Snapshot:
    cached = store.load_latest()
    now = dt.datetime.now()
    if cached is not None and not force and now - cached[1] < max_age:
        return Snapshot(cached[0], cached[1], "cache")
    try:
        df = fetch()
    except Exception as e:
        log.exception("fetch failed")
        if cached is None:
            raise
        return Snapshot(cached[0], cached[1], "stale", str(e))
    store.save(df, now)
    return Snapshot(df, now, "live")
