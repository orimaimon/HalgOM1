"""
Track record engine for screens (past performance / ביצועי עבר).

Measures forward performance (+5, +20, +60 snapshot days) for stocks entering a screen
based on daily snapshots stored in screener/_data/snapshots/ (see store.py).

Results clearly represent historical past performance (ביצועי עבר) and are not a guarantee
of future results.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import screener.data.prices as prices
from screener.data.store import DATA_DIR, Store
from screener.derive import add_derived
from screener.filters import screen
import screener.presets as presets

log = logging.getLogger(__name__)

TRACK_DIR = DATA_DIR / "track"
HORIZONS = [5, 20, 60]
NOT_YET = "not yet"


def _filter_key(filters: dict[str, str]) -> str:
    """Deterministic hash of filter selections."""
    s = json.dumps(filters, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _cache_path(screen_name: str | None, filters: dict[str, str], root_dir: Path | None = None) -> Path:
    f_hash = _filter_key(filters)
    base_dir = root_dir if root_dir is not None else TRACK_DIR
    if screen_name:
        safe_name = "".join(c for c in screen_name if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_name = safe_name.replace(" ", "_")
        return base_dir / f"{safe_name}_{f_hash}.json"
    return base_dir / f"screen_{f_hash}.json"


def _atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _correct_split(ticker: str, start_day: dt.date, p_start: float, p_end: float) -> float:
    """Correct exit price if a split occurred between start_day and exit day."""
    if p_start <= 0 or p_end <= 0:
        return p_end

    # Primary check: prices.split_factor
    try:
        sf = prices.split_factor(ticker, since=start_day)
        if sf > 0 and sf != 1.0:
            return p_end * sf
    except Exception:
        pass

    # Heuristic ratio check for unrecorded splits or synthetic tests:
    # A one-day or multi-day price move of >45% matching a common split ratio
    raw_ratio = p_end / p_start
    for target_ratio in [0.5, 0.333333, 0.25, 0.2, 0.1, 0.666667]:
        if abs(raw_ratio - target_ratio) / target_ratio < 0.05:
            mult = 1.0 / target_ratio
            return p_end * mult
    for target_ratio in [2.0, 3.0, 4.0, 5.0, 10.0, 1.5]:
        if abs(raw_ratio - target_ratio) / target_ratio < 0.05:
            div = target_ratio
            return p_end / div
    return p_end


def _ensure_benchmarks(start_date: dt.date) -> None:
    try:
        prices.ensure(["SPY", "QQQ"], start=start_date)
    except Exception as e:
        log.warning("cannot update benchmark prices: %s", e)


def _get_benchmark_return(ticker: str, start_day: dt.date, end_day: dt.date) -> float | None:
    try:
        ret = prices.total_return(ticker, start_day, end_day)
        if ret is not None:
            return float(ret * 100.0)
        p_start = prices.close_on(ticker, start_day)
        p_end = prices.close_on(ticker, end_day)
        if p_start is not None and p_end is not None and p_start > 0:
            return float(((p_end / p_start) - 1.0) * 100.0)
    except Exception:
        pass
    return None


def _universe_median_return_from_maps(
    p_start_map: dict[str, float], p_end_map: dict[str, float]
) -> float | None:
    common = set(p_start_map.keys()) & set(p_end_map.keys())
    valid_pairs = [
        (p_start_map[t], p_end_map[t])
        for t in common
        if p_start_map[t] > 0 and p_end_map[t] > 0
    ]
    if not valid_pairs:
        return None
    rets = [((p_end / p_start) - 1.0) * 100.0 for p_start, p_end in valid_pairs]
    return float(np.median(rets))


def track_screen(
    filters: dict[str, str],
    screen_name: str | None = None,
    store: Store | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Compute track record summary for a single screen.

    Measures historical performance (ביצועי עבר) over +5, +20, and +60 snapshot days.
    """
    store = store or Store()
    snapshot_dates = store.snapshot_dates()
    c_path = _cache_path(screen_name, filters, root_dir=store.root / "track")

    cached_data: dict[str, Any] | None = None
    if not force_recompute and c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c = json.load(f)
            if c.get("filters") == filters:
                cached_data = c
        except Exception as e:
            log.warning("cannot read track cache %s: %s", c_path, e)

    if not snapshot_dates:
        return _build_empty_summary(screen_name, filters)

    _ensure_benchmarks(snapshot_dates[0])

    daily_results: dict[str, list[str]] = (
        cached_data.get("daily_results", {}) if cached_data else {}
    )
    daily_prices: dict[str, dict[str, float]] = (
        cached_data.get("daily_prices", {}) if cached_data else {}
    )

    cached_entries_map: dict[tuple[str, str], dict[str, Any]] = {}
    if cached_data and "entries" in cached_data:
        for e in cached_data["entries"]:
            key = (e["ticker"], e["entry_date"])
            cached_entries_map[key] = e

    all_days_str = [d.isoformat() for d in snapshot_dates]
    processed_days = set(daily_results.keys())
    missing_days = [d for d in snapshot_dates if d.isoformat() not in processed_days]

    for d in missing_days:
        try:
            raw = store.load_snapshot(d)
            df_derived = add_derived(raw, today=d)
            screened = screen(df_derived, filters, junk=True)
            d_str = d.isoformat()
            if "ticker" in screened.columns:
                tickers = screened["ticker"].tolist()
                prices_dict = (
                    dict(zip(screened["ticker"], screened["price"]))
                    if "price" in screened.columns
                    else {}
                )
            else:
                tickers = []
                prices_dict = {}
            daily_results[d_str] = tickers
            daily_prices[d_str] = {
                t: float(p) for t, p in prices_dict.items() if pd.notna(p)
            }
        except Exception as e:
            log.warning("failed to process snapshot %s for track record: %s", d, e)

    # Memoize full snapshot price maps on demand for exit price lookup
    snapshot_price_maps: dict[str, dict[str, float]] = {}

    def get_snapshot_prices(day: dt.date) -> dict[str, float]:
        day_str = day.isoformat()
        if day_str not in snapshot_price_maps:
            try:
                raw_snap = store.load_snapshot(day)
                if "ticker" in raw_snap.columns and "price" in raw_snap.columns:
                    m = dict(zip(raw_snap["ticker"], raw_snap["price"]))
                    snapshot_price_maps[day_str] = {
                        t: float(p) for t, p in m.items() if pd.notna(p)
                    }
                else:
                    snapshot_price_maps[day_str] = {}
            except Exception:
                snapshot_price_maps[day_str] = {}
        return snapshot_price_maps[day_str]

    entries: list[dict[str, Any]] = []
    n_days = len(snapshot_dates)

    for i, d in enumerate(snapshot_dates):
        d_str = d.isoformat()
        curr_tickers = daily_results.get(d_str, [])
        prev_tickers = set(
            daily_results.get(snapshot_dates[i - 1].isoformat(), []) if i > 0 else []
        )
        entry_tickers = [t for t in curr_tickers if t not in prev_tickers]

        d_prices = daily_prices.get(d_str, {})
        for ticker in entry_tickers:
            p_start = d_prices.get(ticker)
            if p_start is None or p_start <= 0:
                snap_p = get_snapshot_prices(d)
                p_start = snap_p.get(ticker)
            if p_start is None or p_start <= 0:
                continue

            entry_key = (ticker, d_str)
            cached_e = cached_entries_map.get(entry_key)

            entry_record: dict[str, Any] = {
                "ticker": ticker,
                "entry_date": d_str,
                "entry_price": p_start,
                "returns": {},
                "spy_returns": {},
                "benchmarks": {},
            }

            for h in HORIZONS:
                h_str = str(h)

                # Check if return for horizon h was already computed in cache
                if (
                    cached_e
                    and isinstance(cached_e.get("returns", {}).get(h_str), (int, float))
                ):
                    entry_record["returns"][h_str] = cached_e["returns"][h_str]
                    entry_record["spy_returns"][h_str] = cached_e.get("spy_returns", {}).get(h_str, NOT_YET)
                    entry_record["benchmarks"][h_str] = cached_e.get("benchmarks", {}).get(h_str, "SPY")
                    continue

                target_idx = i + h
                if target_idx < n_days:
                    target_day = snapshot_dates[target_idx]
                    snap_end = get_snapshot_prices(target_day)
                    p_end_raw = snap_end.get(ticker)

                    if p_end_raw is None:
                        # Fallback: check prices history
                        try:
                            p_end_raw = prices.close_on(ticker, target_day)
                        except Exception:
                            p_end_raw = None

                    if p_end_raw is not None and p_end_raw > 0:
                        p_end = _correct_split(ticker, d, p_start, p_end_raw)
                        stock_ret = float(((p_end / p_start) - 1.0) * 100.0)
                        entry_record["returns"][h_str] = round(stock_ret, 2)
                    else:
                        entry_record["returns"][h_str] = NOT_YET

                    spy_ret = _get_benchmark_return("SPY", d, target_day)
                    if spy_ret is not None:
                        entry_record["spy_returns"][h_str] = round(spy_ret, 2)
                        entry_record["benchmarks"][h_str] = "SPY"
                    else:
                        snap_start = get_snapshot_prices(d)
                        med_ret = _universe_median_return_from_maps(snap_start, snap_end)
                        if med_ret is not None:
                            entry_record["spy_returns"][h_str] = round(med_ret, 2)
                            entry_record["benchmarks"][h_str] = "universe_median"
                        else:
                            entry_record["spy_returns"][h_str] = NOT_YET
                            entry_record["benchmarks"][h_str] = "none"
                else:
                    entry_record["returns"][h_str] = NOT_YET
                    entry_record["spy_returns"][h_str] = NOT_YET
                    entry_record["benchmarks"][h_str] = "none"

            entries.append(entry_record)

    summary = _calculate_summary(screen_name, filters, entries, snapshot_dates)

    cache_payload = {
        "screen_name": screen_name,
        "filters": filters,
        "last_updated": all_days_str[-1] if all_days_str else "",
        "daily_results": daily_results,
        "daily_prices": daily_prices,
        "entries": entries,
        "summary": summary,
    }
    _atomic_write_json(cache_payload, c_path)

    return summary


def _calculate_summary(
    screen_name: str | None,
    filters: dict[str, str],
    entries: list[dict[str, Any]],
    snapshot_dates: list[dt.date],
) -> dict[str, Any]:
    horizons_summary = {}

    for h in HORIZONS:
        h_str = str(h)
        valid_entries = [
            e for e in entries if isinstance(e["returns"].get(h_str), (int, float))
        ]

        if not valid_entries:
            horizons_summary[h] = {
                "horizon": h,
                "entries": 0,
                "median_return": 0.0,
                "avg_return": 0.0,
                "pct_positive": 0.0,
                "avg_excess_spy": 0.0,
                "sessions": 0,
                "benchmark": "SPY",
                "disclaimer": "ביצועי עבר בלבד (Past performance only)",
            }
            continue

        rets = [float(e["returns"][h_str]) for e in valid_entries]
        spy_rets = [
            float(e["spy_returns"][h_str])
            for e in valid_entries
            if isinstance(e["spy_returns"].get(h_str), (int, float))
        ]

        excess_rets = []
        for e in valid_entries:
            r = e["returns"].get(h_str)
            sr = e["spy_returns"].get(h_str)
            if isinstance(r, (int, float)) and isinstance(sr, (int, float)):
                excess_rets.append(float(r - sr))

        benchmarks = [
            e["benchmarks"].get(h_str, "SPY")
            for e in valid_entries
            if e["benchmarks"].get(h_str) != "none"
        ]
        bm_type = benchmarks[0] if benchmarks else "SPY"

        sessions_count = len(set(e["entry_date"] for e in valid_entries))

        horizons_summary[h] = {
            "horizon": h,
            "entries": len(valid_entries),
            "median_return": round(float(np.median(rets)), 2),
            "avg_return": round(float(np.mean(rets)), 2),
            "pct_positive": round(
                float(np.mean([100.0 if r > 0 else 0.0 for r in rets])), 1
            ),
            "avg_excess_spy": round(float(np.mean(excess_rets)), 2)
            if excess_rets
            else 0.0,
            "sessions": sessions_count,
            "benchmark": bm_type,
            "disclaimer": "ביצועי עבר בלבד (Past performance only)",
        }

    return {
        "screen_name": screen_name or "Custom Screen",
        "filters": filters,
        "last_updated": snapshot_dates[-1].isoformat() if snapshot_dates else "",
        "total_entries": len(entries),
        "total_sessions": len(snapshot_dates),
        "horizons": horizons_summary,
        "entries": entries,
        "disclaimer": "תוצאות אלו מייצגות ביצועי עבר (Past performance) ואינן מהוות התחייבות לתשואה עתידית.",
    }


def _build_empty_summary(
    screen_name: str | None, filters: dict[str, str]
) -> dict[str, Any]:
    horizons_summary = {}
    for h in HORIZONS:
        horizons_summary[h] = {
            "horizon": h,
            "entries": 0,
            "median_return": 0.0,
            "avg_return": 0.0,
            "pct_positive": 0.0,
            "avg_excess_spy": 0.0,
            "sessions": 0,
            "benchmark": "SPY",
            "disclaimer": "ביצועי עבר בלבד (Past performance only)",
        }

    return {
        "screen_name": screen_name or "Custom Screen",
        "filters": filters,
        "last_updated": "",
        "total_entries": 0,
        "total_sessions": 0,
        "horizons": horizons_summary,
        "entries": [],
        "disclaimer": "תוצאות אלו מייצגות ביצועי עבר (Past performance) ואינן מהוות התחייבות לתשואה עתידית.",
    }


def track_all_screens(
    store: Store | None = None,
    force_recompute: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute track record summary for all built-in presets and user-saved screens.

    Returns a dict mapping screen_name -> track_summary.
    Results represent past performance (ביצועי עבר).
    """
    store = store or Store()
    out: dict[str, dict[str, Any]] = {}

    # 1. Built-in presets
    for name, preset in presets.BUILTIN.items():
        try:
            out[name] = track_screen(
                filters=preset.filters,
                screen_name=name,
                store=store,
                force_recompute=force_recompute,
            )
        except Exception as e:
            log.exception("failed to compute track record for builtin screen '%s'", name)

    # 2. User screens
    user_presets = presets.load_user(path=store.root / "presets.json")
    for name, preset in user_presets.items():
        try:
            out[name] = track_screen(
                filters=preset.filters,
                screen_name=name,
                store=store,
                force_recompute=force_recompute,
            )
        except Exception as e:
            log.exception("failed to compute track record for user screen '%s'", name)

    return out
