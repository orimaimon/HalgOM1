"""
Market breadth calculations — per sector and over time from daily snapshots.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

from screener import derive
from screener import filters
from screener import signals as S
from screener.data import store as store_module
from screener.data.store import Store, _atomic_write_text

log = logging.getLogger(__name__)

LABELS: dict[str, str] = {
    "stocks": "מניות",
    "pct_above_sma50": "% מעל SMA50",
    "pct_above_sma200": "% מעל SMA200",
    "advancers": "עולות",
    "decliners": "יורדות",
    "unchanged": "ללא שינוי",
    "ad_ratio": "יחס עולות/יורדות",
    "pct_advancing": "% עולות",
    "new_highs": "שיאים חדשים",
    "new_lows": "שפלים חדשים",
    "net_highs": "שיאים פחות שפלים",
}

_BO_52W = next(s for s in S.BREAKOUTS if s.label == "New 52-week high (breakout)")
_BD_52W = next(s for s in S.BREAKDOWNS if s.label == "New 52-week low (breakdown)")


def _pct_above(price: pd.Series, sma: pd.Series) -> float | None:
    valid = sma.notna() & price.notna()
    tot = int(valid.sum())
    if tot == 0:
        return None
    above = int((valid & (price > sma)).sum())
    return (above / tot) * 100.0


def breadth(universe: pd.DataFrame, junk: bool = True) -> dict:
    df = filters.hide_junk(universe, junk) if junk else universe

    stocks = len(df)

    price = df["price"] if "price" in df.columns else pd.Series(dtype=float)
    sma50 = df["sma50"] if "sma50" in df.columns else pd.Series(dtype=float)
    sma200 = df["sma200"] if "sma200" in df.columns else pd.Series(dtype=float)

    pct_above_sma50 = _pct_above(price, sma50)
    pct_above_sma200 = _pct_above(price, sma200)

    if "change" in df.columns:
        chg = df["change"].dropna()
        advancers = int((chg > 0).sum())
        decliners = int((chg < 0).sum())
        unchanged = int((chg == 0).sum())
    else:
        advancers = 0
        decliners = 0
        unchanged = 0

    ad_ratio = (advancers / decliners) if decliners > 0 else None
    tot_chg = advancers + decliners + unchanged
    pct_advancing = (advancers / tot_chg * 100.0) if tot_chg > 0 else None

    if "price" in df.columns and "high52_prev" in df.columns:
        new_highs = int(_BO_52W.fn(df).fillna(False).astype(bool).sum())
    else:
        new_highs = 0

    if "price" in df.columns and "low52_prev" in df.columns:
        new_lows = int(_BD_52W.fn(df).fillna(False).astype(bool).sum())
    else:
        new_lows = 0

    net_highs = new_highs - new_lows

    return {
        "stocks": stocks,
        "pct_above_sma50": pct_above_sma50,
        "pct_above_sma200": pct_above_sma200,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "ad_ratio": ad_ratio,
        "pct_advancing": pct_advancing,
        "new_highs": new_highs,
        "new_lows": new_lows,
        "net_highs": net_highs,
    }


def by_sector(universe: pd.DataFrame, junk: bool = True) -> pd.DataFrame:
    df = filters.hide_junk(universe, junk) if junk else universe

    cols = ["stocks", "pct_above_sma200", "pct_advancing", "new_highs", "new_lows"]
    rows = []

    if "sector" in df.columns:
        for sector, group in df.groupby("sector", observed=True):
            if pd.isna(sector) or not str(sector).strip():
                continue
            b = breadth(group, junk=False)
            rows.append({
                "sector": str(sector),
                "stocks": b["stocks"],
                "pct_above_sma200": b["pct_above_sma200"],
                "pct_advancing": b["pct_advancing"],
                "new_highs": b["new_highs"],
                "new_lows": b["new_lows"],
            })

    if not rows:
        res = pd.DataFrame(columns=cols)
        res.index.name = "sector"
        return res

    res = pd.DataFrame(rows).set_index("sector")
    res = res.sort_values(by="pct_above_sma200", ascending=False, na_position="last")
    return res


def history(store: Store, days: int = 60, junk: bool = True) -> pd.DataFrame:
    dates = store.snapshot_dates()
    if days > 0:
        dates = dates[-days:]

    cols = ["pct_above_sma50", "pct_above_sma200", "advancers", "decliners", "new_highs", "new_lows", "net_highs"]
    if not dates:
        res = pd.DataFrame(columns=cols)
        res.index.name = "day"
        return res

    data_dir = getattr(store, "root", store_module.DATA_DIR)
    cache_dir = Path(data_dir) / "breadth"
    cache_file = cache_dir / "daily.json"

    cache = {}
    if cache_file.exists():
        try:
            raw_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(raw_cache, dict):
                cache = raw_cache
        except Exception as e:
            log.warning("damaged breadth cache file ignored: %s", e)
            cache = {}

    cache_changed = False
    result_rows = {}

    for day in dates:
        snap_path = store.snapshot_path(day)
        try:
            mtime_ns = snap_path.stat().st_mtime_ns
        except Exception as e:
            log.warning("unreadable snapshot file %s: %s", snap_path, e)
            continue

        day_str = day.isoformat()
        cached_entry = cache.get(day_str)
        if isinstance(cached_entry, dict) and cached_entry.get("mtime_ns") == mtime_ns and "metrics" in cached_entry:
            metrics = cached_entry["metrics"]
        else:
            try:
                raw_df = store.load_snapshot(day)
            except Exception as e:
                log.warning("unreadable snapshot data %s: %s", day, e)
                continue

            try:
                derived_df = derive.add_derived(raw_df, today=day)
                b = breadth(derived_df, junk=junk)
                metrics = {c: b[c] for c in cols}
                cache[day_str] = {
                    "mtime_ns": mtime_ns,
                    "metrics": metrics,
                }
                cache_changed = True
            except Exception as e:
                log.warning("failed to compute breadth for snapshot %s: %s", day, e)
                continue

        result_rows[day] = metrics

    if cache_changed:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(json.dumps(cache, indent=2), cache_file)
        except Exception as e:
            log.warning("cannot write breadth cache: %s", e)

    if not result_rows:
        res = pd.DataFrame(columns=cols)
        res.index.name = "day"
        return res

    res = pd.DataFrame.from_dict(result_rows, orient="index")[cols]
    res.index.name = "day"
    return res
