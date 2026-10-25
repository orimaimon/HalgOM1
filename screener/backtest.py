"""Backtesting technical signals over daily history.

Calculates forward returns (5/20/60 days) and excess returns over a benchmark
(or universe median) for every day a technical signal fired.
"""
from __future__ import annotations

import datetime as dt
import math
import duckdb
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


def compute_returns(
    prices: pd.Series | pd.DataFrame | np.ndarray,
    horizon: int = 1,
    kind: str = "forward",
) -> pd.Series | pd.DataFrame | np.ndarray:
    """Calculate percentage returns across a series, dataframe, or numpy array using DuckDB SQL.

    Parameters
    ----------
    prices : pd.Series | pd.DataFrame | np.ndarray
        Price inputs.
    horizon : int
        Look-ahead or look-back step horizon.
    kind : str
        'forward' or 'trailing'.

    Returns
    -------
    pd.Series | pd.DataFrame | np.ndarray
        Returns calculated as percentage change.
    """
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer")

    con = duckdb.connect()

    if isinstance(prices, pd.Series):
        s = prices.astype(float)
        df_in = pd.DataFrame({"i": range(len(s)), "price": s.values})
        if kind == "forward":
            sql = "SELECT (LEAD(price, ?) OVER (ORDER BY i) / price - 1.0) * 100.0 AS ret FROM df_in ORDER BY i"
        elif kind == "trailing":
            sql = "SELECT (price / LAG(price, ?) OVER (ORDER BY i) - 1.0) * 100.0 AS ret FROM df_in ORDER BY i"
        else:
            raise ValueError(f"Unknown kind: {kind}")
        res = con.execute(sql, [horizon]).df()["ret"]
        return pd.Series(res.values, index=prices.index)

    elif isinstance(prices, pd.DataFrame):
        df = prices.astype(float)
        out = {}
        for col in df.columns:
            sub = pd.DataFrame({"i": range(len(df)), "price": df[col].values})
            if kind == "forward":
                sql = "SELECT (LEAD(price, ?) OVER (ORDER BY i) / price - 1.0) * 100.0 AS ret FROM sub ORDER BY i"
            elif kind == "trailing":
                sql = "SELECT (price / LAG(price, ?) OVER (ORDER BY i) - 1.0) * 100.0 AS ret FROM sub ORDER BY i"
            else:
                raise ValueError(f"Unknown kind: {kind}")
            out[col] = con.execute(sql, [horizon]).df()["ret"].values
        return pd.DataFrame(out, index=df.index)

    else:
        arr = np.asarray(prices, dtype=float)
        s = pd.Series(arr)
        ret = compute_returns(s, horizon=horizon, kind=kind)
        return ret.to_numpy()


def compute_sharpe(
    returns: pd.Series | pd.DataFrame | np.ndarray | list,
    rf: float = 0.0,
    annualize: bool = False,
    period_factor: int = 252,
) -> float | None:
    """Calculate Sharpe Ratio using DuckDB SQL aggregations.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame | np.ndarray | list
        Return values (percentages or decimals).
    rf : float
        Risk-free rate per period.
    annualize : bool
        Whether to multiply by sqrt(period_factor).
    period_factor : int
        Periods per year (default 252 for daily).

    Returns
    -------
    float | None
        Calculated Sharpe Ratio or None if insufficient non-NaN values or zero std.
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        s = returns.squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0] if not s.empty else pd.Series(dtype=float)
        s = pd.Series(s).dropna()
    else:
        arr = np.asarray(returns, dtype=float)
        s = pd.Series(arr[~np.isnan(arr)]) if len(arr) > 0 else pd.Series(dtype=float)

    if len(s) < 2:
        return None

    df_in = pd.DataFrame({"r": s.values})
    con = duckdb.connect()
    res = con.execute(
        """
        SELECT
            AVG(r - ?) AS mean_ex,
            STDDEV_SAMP(r - ?) AS std_ex
        FROM df_in
        """,
        [rf, rf],
    ).fetchone()

    if res is None or res[0] is None or res[1] is None or res[1] == 0:
        return None

    sharpe = float(res[0] / res[1])
    if annualize:
        sharpe *= math.sqrt(period_factor)
    return sharpe


def compute_drawdown(prices: pd.Series | pd.DataFrame | np.ndarray | list) -> float | None:
    """Calculate maximum drawdown percentage using DuckDB SQL window functions.

    Parameters
    ----------
    prices : pd.Series | pd.DataFrame | np.ndarray | list
        Price history series or array.

    Returns
    -------
    float | None
        Maximum drawdown in percent (e.g. -15.5) or None if prices empty.
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        s = prices.squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0] if not s.empty else pd.Series(dtype=float)
        s = pd.Series(s).dropna()
    else:
        arr = np.asarray(prices, dtype=float)
        s = pd.Series(arr[~np.isnan(arr)]) if len(arr) > 0 else pd.Series(dtype=float)

    if len(s) == 0:
        return None

    df_in = pd.DataFrame({"i": range(len(s)), "price": s.values})
    con = duckdb.connect()
    res = con.execute(
        """
        WITH peak_tb AS (
            SELECT
                price,
                MAX(price) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
            FROM df_in
        )
        SELECT MIN((price - peak) / peak * 100.0) AS max_dd
        FROM peak_tb
        """
    ).fetchone()

    if res is None or res[0] is None:
        return None

    return float(res[0])


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
