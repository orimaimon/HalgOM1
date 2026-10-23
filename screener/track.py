"""Track record of a screen: how stocks did after they entered it. Past performance, not a promise.

Built only from the daily snapshots (store.py), plus SPY/QQQ closes from the price-history layer
(prices.py: two tickers, at most one update a day). No TradingView requests.

- Entry: a ticker in the screen's results on snapshot day D that was not in them on the previous
  snapshot day. The first snapshot day has no previous day, so it has no entries.
- Forward return after 5 / 20 / 60 snapshot days, from the snapshot `price`, in percent (5.0 == 5%).
  "Pending" while there are not enough later snapshots; no return when the ticker is gone by then.
- Splits: a day-over-day move of more than 45% that matches a common split ratio (2:1, 3:1, 1:10, ...)
  within 2% is undone. A real 50% one-day crash looks the same and is undone too (rare; accepted).
- Benchmark: SPY over the same snapshot days (excess = stock - SPY). Without SPY prices (Yahoo not
  reachable), the median return of the whole universe over the same days, labelled UNIVERSE.

Cache in _data/track/ (per machine), grown by new or rewritten snapshot days only:
  prices.parquet + prices.json   snapshot price of every ticker on every snapshot day (long format)
  <name>_<hash>.json            per screen: the tickers in its results on each snapshot day
A snapshot is rewritten during its day (store.py keeps the day's last fetch), so each cached day
remembers the file's modification time and is redone when it changes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from screener import presets as P
from screener.data import prices
from screener.data.store import Store
from screener.derive import add_derived
from screener.filters import active, screen

log = logging.getLogger(__name__)

HORIZONS = (5, 20, 60)                       # in snapshot days
SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50)   # n-for-1; reverse splits are 1/n
SPLIT_MOVE = 0.45                            # only a one-day move bigger than this can be a split
SPLIT_TOLERANCE = 0.02
BENCHMARKS = ("SPY", "QQQ")
UNIVERSE = "universe median"
FEW_SESSIONS = 20                            # below this the UI says the numbers are not stable yet
VERSION = 1                                  # bump when the membership logic changes: old caches are redone

_benchmarks_checked: dt.date | None = None   # prices.ensure at most once a day per process
_LOCK = threading.Lock()                     # two browser tabs must not write the cache at the same time


# ── small file helpers ─────────────────────────────────────────────────────

def _dir(store: Store) -> Path:
    return store.root / "track"


def _replace(path: Path, write) -> None:
    """Write via a temp file + os.replace: a crash mid-write never leaves a broken cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    _replace(path, lambda tmp: tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8"))


def _stamp(store: Store, day: dt.date) -> int | None:
    try:
        return store.snapshot_path(day).stat().st_mtime_ns
    except OSError:
        return None


def _screen_path(store: Store, name: str, filters: dict[str, str]) -> Path:
    digest = hashlib.sha256(json.dumps(active(filters), sort_keys=True).encode()).hexdigest()[:12]
    slug = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_")[:40] or "screen"
    return _dir(store) / f"{slug}_{digest}.json"


# ── shared inputs: snapshot prices and benchmark closes ────────────────────

def snapshot_prices(store: Store) -> pd.DataFrame:
    """Snapshot price of every ticker: index = snapshot days (dt.date), columns = tickers."""
    path, meta_path = _dir(store) / "prices.parquet", _dir(store) / "prices.json"
    stamps: dict[str, int] = (_read_json(meta_path) or {}).get("stamps", {})
    try:
        long = pd.read_parquet(path) if stamps else None
    except Exception as e:                      # a damaged cache is rebuilt from the snapshots
        log.warning("track: cannot read %s: %s", path, e)
        long, stamps = None, {}
    if long is None:
        long = pd.DataFrame({"day": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str),
                             "price": pd.Series(dtype=float)})
    days = store.snapshot_dates()
    redo = [d for d in days if stamps.get(d.isoformat()) != _stamp(store, d)]
    gone = set(stamps) - {d.isoformat() for d in days}
    if redo or gone:
        drop = {pd.Timestamp(d) for d in redo} | {pd.Timestamp(d) for d in gone}
        parts = [long[~long["day"].isin(drop)]]
        for d in redo:
            stamp = _stamp(store, d)
            try:
                snap = store.load_snapshot(d, columns=["ticker", "price"])
            except Exception as e:               # retried on the next call
                log.warning("track: cannot read snapshot %s: %s", d, e)
                stamps.pop(d.isoformat(), None)
                continue
            parts.append(snap.assign(day=pd.Timestamp(d))[["day", "ticker", "price"]])
            stamps[d.isoformat()] = stamp
        for k in gone:
            stamps.pop(k, None)
        long = pd.concat(parts, ignore_index=True)
        _replace(path, long.to_parquet)
        _write_json(meta_path, {"stamps": stamps})
    long = long.dropna(subset=["ticker"]).drop_duplicates(["day", "ticker"], keep="last")
    wide = long.pivot(index="day", columns="ticker", values="price").sort_index()
    wide.index = [ts.date() for ts in wide.index]
    return wide.astype(float)


def split_adjusted(wide: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prices with split jumps undone (see the module docstring), and how many jumps were undone."""
    ratio = wide / wide.shift(1)
    undo = pd.DataFrame(1.0, index=wide.index, columns=wide.columns)
    for n in SPLIT_RATIOS:
        for r in (n, 1 / n):
            if abs(r - 1) > SPLIT_MOVE:
                undo = undo.mask((ratio / r - 1).abs() <= SPLIT_TOLERANCE, 1 / r)
    return wide * undo.cumprod(), int((undo != 1).sum().sum())


def benchmark_closes(days: list[dt.date]) -> pd.DataFrame:
    """Close of each benchmark on each snapshot day (NaN when unknown). Updates them at most once a day."""
    global _benchmarks_checked
    today = dt.date.today()
    if days and _benchmarks_checked != today:
        _benchmarks_checked = today
        try:
            prices.ensure(list(BENCHMARKS), days[0] - dt.timedelta(days=7))
        except Exception as e:                   # ensure never raises; this is only a guard
            log.warning("track: benchmark update failed: %s", e)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    out = pd.DataFrame(index=days)
    for t in BENCHMARKS:
        try:
            h = prices.history(t)
        except Exception as e:
            log.warning("track: cannot read %s history: %s", t, e)
            h = pd.DataFrame()
        if h.empty or "close" not in h:
            out[t] = math.nan
            continue
        s = h["close"].sort_index()
        # a close up to 4 calendar days back (weekend, holiday); an older one means the history is stale
        out[t] = s.reindex(idx, method="ffill", tolerance=pd.Timedelta(days=4)).to_numpy()
    return out


# ── per screen: which tickers were in its results on each day ───────────────

def _members(store: Store, screens: dict[str, dict[str, str]]) -> dict[str, dict[dt.date, set[str]]]:
    """Results of each screen (Hide junk on, as in the app) on each snapshot day, from the cache when it
    is current. A snapshot is loaded and derived once per day for all the screens that need it."""
    days = store.snapshot_dates()
    stamps = {d.isoformat(): _stamp(store, d) for d in days}
    cache: dict[str, dict] = {}
    for name, filters in screens.items():
        c = _read_json(_screen_path(store, name, filters)) or {}
        if c.get("version") != VERSION or c.get("filters") != active(filters):
            c = {}
        c.setdefault("days", {})
        cache[name] = {"version": VERSION, "filters": active(filters),
                       "days": {k: v for k, v in c["days"].items()
                                if k in stamps and isinstance(v, dict) and v.get("stamp") == stamps[k]
                                and isinstance(v.get("tickers"), list)}}
    changed: set[str] = set()
    for d in days:
        k = d.isoformat()
        need = [n for n in screens if k not in cache[n]["days"]]
        if not need:
            continue
        try:
            df = add_derived(store.load_snapshot(d), today=d)
        except Exception as e:                   # retried on the next call
            log.warning("track: cannot read snapshot %s: %s", d, e)
            continue
        for n in need:
            tickers = screen(df, screens[n], junk=True)["ticker"].dropna().astype(str)
            cache[n]["days"][k] = {"stamp": stamps[k], "tickers": sorted(set(tickers))}
            changed.add(n)
    for n in changed:
        try:
            _write_json(_screen_path(store, n, screens[n]), cache[n])
        except OSError as e:                     # the result is still right, only not cached
            log.warning("track: cannot write the cache of %s: %s", n, e)
    return {n: {dt.date.fromisoformat(k): set(v["tickers"]) for k, v in c["days"].items()}
            for n, c in cache.items()}


# ── entries, returns, summary ──────────────────────────────────────────────

def entries(members: dict[dt.date, set[str]], adj: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """One row per entry: day, ticker, and per horizon h: ret_h (stock, %), spy_h, univ_h (%), pending_h.
    `adj` is split-adjusted snapshot prices (index = snapshot days in order)."""
    days = list(adj.index)
    n = len(days)
    col = {t: j for j, t in enumerate(adj.columns)}
    px = adj.to_numpy(dtype=float)               # numpy: per-cell pandas lookups are far too slow here
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = {h: np.vstack([px[h:] / px[:-h] * 100 - 100, np.full((min(h, n), px.shape[1]), np.nan)])[:n]
               for h in HORIZONS}
    # median of the whole universe per day (NaN where no ticker has a return yet)
    univ = {h: pd.DataFrame(np.where(np.isfinite(fwd[h]), fwd[h], np.nan)).median(axis=1).to_numpy()
            for h in HORIZONS}
    spy = bench["SPY"].to_numpy(dtype=float) if "SPY" in bench else np.full(n, np.nan)
    rows = []
    for i in range(1, n):
        d, before = days[i], members.get(days[i - 1])
        now = members.get(d)
        if now is None or before is None:        # a day without data has no entries, and neither has the next
            continue
        for t in sorted(now - before):
            j = col.get(t)
            if j is None or not px[i, j] > 0:
                continue
            row = {"day": d, "ticker": t}
            for h in HORIZONS:
                pending = i + h >= n
                row[f"pending_{h}"] = pending
                row[f"ret_{h}"] = math.nan if pending else fwd[h][i, j]
                row[f"univ_{h}"] = math.nan if pending else univ[h][i]
                row[f"spy_{h}"] = math.nan if pending else spy[i + h] / spy[i] * 100 - 100
            rows.append(row)
    cols = ["day", "ticker"] + [f"{p}_{h}" for h in HORIZONS for p in ("ret", "spy", "univ", "pending")]
    return pd.DataFrame(rows, columns=cols)


def _r(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 2)


def summarize(ent: pd.DataFrame, sessions: int) -> dict:
    """Per horizon: entries with a return, still pending, median / mean return, % positive, mean excess
    over SPY (or over the universe median when there is no SPY data). None where there is no data."""
    out = {"sessions": sessions, "few_sessions": sessions < FEW_SESSIONS, "entries": len(ent), "horizons": {}}
    for h in HORIZONS:
        has = ent[ent[f"ret_{h}"].notna()] if len(ent) else ent
        r = has[f"ret_{h}"]
        bench = "SPY" if has[f"spy_{h}"].notna().any() else UNIVERSE if len(has) else None
        base = has["spy_" + str(h)] if bench == "SPY" else has[f"univ_{h}"]
        out["horizons"][h] = {
            "entries": len(has),
            "pending": int(ent[f"pending_{h}"].sum()) if len(ent) else 0,
            "median": _r(r.median()) if len(has) else None,
            "mean": _r(r.mean()) if len(has) else None,
            "pct_positive": _r((r > 0).mean() * 100) if len(has) else None,
            "excess": _r((r - base).mean()) if len(has) else None,
            "benchmark": bench,
        }
    return out


def _run(screens: dict[str, dict[str, str]], store: Store | None) -> dict[str, dict]:
    with _LOCK:
        return _run_locked(screens, store or Store())


def _run_locked(screens: dict[str, dict[str, str]], store: Store) -> dict[str, dict]:
    wide = snapshot_prices(store)
    adj, n_splits = split_adjusted(wide)
    bench = benchmark_closes(list(adj.index))
    out = {}
    for name, days in _members(store, screens).items():
        ent = entries(days, adj, bench)
        out[name] = {**summarize(ent, len(adj.index)), "splits_undone": n_splits, "table": ent}
    return out


def track_screen(filters: dict[str, str], name: str = "custom", store: Store | None = None) -> dict:
    """Track record of one screen (past performance). The summary from `summarize`, plus "table":
    the entries (see `entries`) and "splits_undone": split jumps undone in the whole universe."""
    return _run({name: filters}, store)[name]


def all_screens(user_prefix: str = "★ ") -> dict[str, dict[str, str]]:
    """The built-in screens and the user's saved screens (named as in the app's screen picker)."""
    screens = {name: p.filters for name, p in P.BUILTIN.items()}
    screens |= {user_prefix + name: p.filters for name, p in P.load_user().items()}
    return screens


def track_all_screens(store: Store | None = None) -> dict[str, dict]:
    """Track record of every built-in and saved screen (past performance), sharing one pass per day."""
    return _run(all_screens(), store)
