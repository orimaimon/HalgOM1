"""Backtesting technical signals over daily history.

Calculates forward returns (5/20/60 days) and excess returns over a benchmark
(or universe median) for every day a technical signal fired.
"""
from __future__ import annotations

import datetime as dt
import math
import numpy as np
import pandas as pd

from screener import history_signals

LABELS: dict[str, str] = {
    "signal": "סיגנל",
    "events": "אירועים",
    "tickers": "מניות",
    "first_day": "יום ראשון",
    "last_day": "יום אחרון",
    "n": "נבדקו",
    "pending": "בהמתנה",
    "median": "תשואה חציונית",
    "mean": "תשואה ממוצעת",
    "pct_positive": "אחוז חיוביות",
    "excess": "תשואת יתר",
    "disclaimer": "תוצאות עבר על נתוני היסטוריה; אינן מבטיחות תוצאות בעתיד ואינן ייעוץ השקעות.",
}


def _r(x: float | None) -> float | None:
    if x is None or pd.isna(x):
        return None
    return round(float(x), 2)


def run(
    closes: pd.DataFrame,
    name: str,
    horizons: tuple[int, ...] = (5, 20, 60),
    cooldown: int = 5,
    benchmark: pd.Series | None = None,
) -> dict:
    """Run a historical backtest for a technical signal across daily closes.

    Parameters
    ----------
    closes : pd.DataFrame
        Daily close prices (index = trading days, columns = tickers).
    name : str
        Boolean signal name from history_signals.SIGNAL_LABELS.
    horizons : tuple[int, ...]
        Forward snapshot horizon days (default 5, 20, 60).
    cooldown : int
        Minimum trading days between counted events for the same ticker.
    benchmark : pd.Series | None
        Optional benchmark close series (e.g. SPY). If None, uses universe median.

    Returns
    -------
    dict
        Summary dictionary with keys: signal, events, tickers, first_day, last_day,
        horizons dict, and table DataFrame.
    """
    adj = history_signals.prepare(closes)
    events_df = history_signals.event_matrix(adj, name, prepared=True)

    n_bars, n_tickers = adj.shape
    dates = [d.date() if hasattr(d, "date") else d for d in adj.index]
    tickers = list(adj.columns)
    px = adj.to_numpy(dtype=float)
    events_arr = events_df.to_numpy(dtype=bool)

    # Extract event coordinates with cooldown filter
    rows_list: list[int] = []
    cols_list: list[int] = []
    for c in range(n_tickers):
        fire_indices = np.where(events_arr[:, c])[0]
        if len(fire_indices) == 0:
            continue
        last_r = -cooldown - 1
        for r in fire_indices:
            if r - last_r > cooldown:
                rows_list.append(r)
                cols_list.append(c)
                last_r = r

    empty_table = pd.DataFrame(columns=["day", "ticker"] + [f"ret_{h}" for h in horizons])

    if not rows_list:
        return {
            "signal": name,
            "events": 0,
            "tickers": 0,
            "first_day": None,
            "last_day": None,
            "horizons": {
                h: {
                    "n": 0,
                    "pending": 0,
                    "median": None,
                    "mean": None,
                    "pct_positive": None,
                    "excess": None,
                }
                for h in horizons
            },
            "table": empty_table,
        }

    rows_arr = np.array(rows_list, dtype=int)
    cols_arr = np.array(cols_list, dtype=int)

    # Sort events chronologically (by bar index r, then column index c)
    sort_order = np.lexsort((cols_arr, rows_arr))
    rows_arr = rows_arr[sort_order]
    cols_arr = cols_arr[sort_order]

    # Precompute forward returns matrices for each horizon
    fwd_ret: dict[int, np.ndarray] = {}
    for h in horizons:
        mat = np.full((n_bars, n_tickers), np.nan, dtype=float)
        if n_bars > h:
            with np.errstate(divide="ignore", invalid="ignore"):
                mat[:-h, :] = (px[h:, :] / px[:-h, :] - 1.0) * 100.0
        fwd_ret[h] = mat

    # Precompute benchmark returns for each horizon
    bench_ret: dict[int, np.ndarray] = {}
    if benchmark is not None and not benchmark.empty:
        b_series = benchmark.reindex(adj.index).ffill().to_numpy(dtype=float)
        for h in horizons:
            b_mat = np.full(n_bars, np.nan, dtype=float)
            if n_bars > h:
                with np.errstate(divide="ignore", invalid="ignore"):
                    b_mat[:-h] = (b_series[h:] / b_series[:-h] - 1.0) * 100.0
            bench_ret[h] = b_mat
    else:
        for h in horizons:
            b_mat = np.full(n_bars, np.nan, dtype=float)
            if n_bars > h:
                with np.errstate(all="ignore"):
                    b_mat[:-h] = np.nanmedian(fwd_ret[h][:-h, :], axis=1)
            bench_ret[h] = b_mat

    # Build event table DataFrame
    table_dict: dict[str, list] = {
        "day": [dates[r] for r in rows_arr],
        "ticker": [tickers[c] for c in cols_arr],
    }
    for h in horizons:
        vals = np.where(rows_arr + h >= n_bars, np.nan, fwd_ret[h][rows_arr, cols_arr])
        table_dict[f"ret_{h}"] = list(vals)

    table = pd.DataFrame(table_dict, columns=["day", "ticker"] + [f"ret_{h}" for h in horizons])

    # Horizons summary
    horizons_summary: dict[int, dict] = {}
    for h in horizons:
        pending_count = int(np.sum(rows_arr + h >= n_bars))
        valid_mask = rows_arr + h < n_bars
        ev_r = rows_arr[valid_mask]
        ev_c = cols_arr[valid_mask]

        vals = fwd_ret[h][ev_r, ev_c]
        b_vals = bench_ret[h][ev_r]

        non_nan = ~np.isnan(vals)
        valid_rets = vals[non_nan]

        if len(valid_rets) == 0:
            horizons_summary[h] = {
                "n": 0,
                "pending": pending_count,
                "median": None,
                "mean": None,
                "pct_positive": None,
                "excess": None,
            }
        else:
            excess_mask = non_nan & ~np.isnan(b_vals)
            excess_vals = vals[excess_mask] - b_vals[excess_mask]
            horizons_summary[h] = {
                "n": len(valid_rets),
                "pending": pending_count,
                "median": _r(float(np.median(valid_rets))),
                "mean": _r(float(np.mean(valid_rets))),
                "pct_positive": _r(float(np.mean(valid_rets > 0) * 100.0)),
                "excess": _r(float(np.mean(excess_vals))) if len(excess_vals) > 0 else None,
            }

    first_day = table_dict["day"][0]
    last_day = table_dict["day"][-1]
    unique_tickers = len(set(cols_arr))

    return {
        "signal": name,
        "events": len(rows_arr),
        "tickers": unique_tickers,
        "first_day": first_day,
        "last_day": last_day,
        "horizons": horizons_summary,
        "table": table,
    }
