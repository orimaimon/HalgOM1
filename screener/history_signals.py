"""Pure functions for history-based signals ("within the last N days").

Calculates crosses of SMA50/SMA200 and breakouts over multiple windows
from wide daily price history (index=trading days, columns=tickers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screener import track

SIGNAL_LABELS: dict[str, str] = {
    "cross_up_sma50": "חצתה את ממוצע 50 כלפי מעלה",
    "cross_down_sma50": "חצתה את ממוצע 50 כלפי מטה",
    "cross_up_sma200": "חצתה את ממוצע 200 כלפי מעלה",
    "cross_down_sma200": "חצתה את ממוצע 200 כלפי מטה",
    "golden_cross": "ממוצע 50 חצה את ממוצע 200 כלפי מעלה (גולדן קרוס)",
    "death_cross": "ממוצע 50 חצה את ממוצע 200 כלפי מטה (דת' קרוס)",
    "days_since_cross_up_sma200": "ימים מאז חציית ממוצע 200 כלפי מעלה",
    "days_since_cross_down_sma200": "ימים מאז חציית ממוצע 200 כלפי מטה",
    "breakout_20": "פריצת שיא 20 ימים",
    "breakout_60": "פריצת שיא 60 ימים",
    "breakout_120": "פריצת שיא 120 ימים",
    "breakout_250": "פריצת שיא 250 ימים",
}


def prepare(closes: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw daily closes for history signal calculations.

    Undoes split jumps and forward-fills gaps up to 5 days.
    """
    if closes.empty:
        return closes.copy()
    adj, _ = track.split_adjusted(closes)
    return adj.ffill(limit=5)


def _fast_sma(arr: np.ndarray, w: int) -> np.ndarray:
    """Fast rolling simple moving average over 2D numpy array (axis=0).

    Requires min_periods = w (all w bars in window must be non-NaN).
    """
    n_bars, n_tickers = arr.shape
    sma = np.full_like(arr, np.nan, dtype=float)
    if n_bars < w:
        return sma

    nan_mask = np.isnan(arr)
    valid_count = np.cumsum(~nan_mask, axis=0)
    window_valid = valid_count[w - 1:] - np.vstack([np.zeros((1, n_tickers), dtype=int), valid_count[:-w]])

    filled = np.nan_to_num(arr, nan=0.0)
    cs = np.cumsum(filled, axis=0)
    window_sum = cs[w - 1:] - np.vstack([np.zeros((1, n_tickers), dtype=float), cs[:-w]])

    sma[w - 1:] = np.where(window_valid == w, window_sum / w, np.nan)
    return sma


def _cross_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (cross_up, cross_down, valid) boolean 2D matrices."""
    a_prev = np.vstack([np.full((1, a.shape[1]), np.nan), a[:-1]])
    b_prev = np.vstack([np.full((1, b.shape[1]), np.nan), b[:-1]])

    valid = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(a_prev) & ~np.isnan(b_prev)

    cross_up = (a > b) & (a_prev <= b_prev) & valid
    cross_down = (a < b) & (a_prev >= b_prev) & valid

    return cross_up, cross_down, valid


def _eval_within_matrix(events: np.ndarray, valid: np.ndarray, days: int, index: pd.Index) -> pd.Series:
    """Evaluate event within last `days` bars for 2D matrices.

    Returns pd.Series with 'boolean' dtype (True / False / pd.NA).
    """
    n_bars, n_tickers = events.shape
    if n_bars == 0 or n_tickers == 0:
        return pd.Series(pd.NA, index=index, dtype="boolean")

    w_events = events[-days:]
    w_valid = valid[-days:]

    has_event = np.any(w_events, axis=0)
    has_valid = np.any(w_valid, axis=0)

    arr = np.full(n_tickers, pd.NA, dtype=object)
    arr[has_valid] = False
    arr[has_event] = True

    return pd.Series(arr, index=index, dtype="boolean")


def _days_since_matrix(events: np.ndarray, index: pd.Index) -> pd.Series:
    """Calculate bars since most recent True event across history."""
    n_bars, n_tickers = events.shape
    if n_bars == 0 or n_tickers == 0:
        return pd.Series(np.nan, index=index, dtype=float)

    indices = np.arange(n_bars)[:, None]
    match = np.where(events, indices, -1)
    last_idx = np.max(match, axis=0)

    days_since = np.where(last_idx >= 0, (n_bars - 1) - last_idx, np.nan)
    return pd.Series(days_since, index=index, dtype=float)


def _eval_breakout_within(arr: np.ndarray, w: int, days: int, index: pd.Index) -> pd.Series:
    """Evaluate breakout over window `w` within last `days` bars directly."""
    n_bars, n_tickers = arr.shape
    if n_bars == 0 or n_tickers == 0:
        return pd.Series(pd.NA, index=index, dtype="boolean")

    # We need sub-array starting from bar -days - w
    start = max(0, n_bars - days - w)
    sub = arr[start:]
    sub_bars = len(sub)

    has_event = np.zeros(n_tickers, dtype=bool)
    has_valid = np.zeros(n_tickers, dtype=bool)

    # For each bar k in the last `days` bars (relative to end of sub)
    lookback_bars = min(days, sub_bars)
    for offset in range(lookback_bars, 0, -1):
        idx = sub_bars - offset
        if idx - w < 0:
            continue
        prev_w = sub[idx - w : idx]
        with np.errstate(all="ignore"):
            max_w = np.nanmax(prev_w, axis=0)
        # A full window only: a stock listed 30 days ago has no 250-day high to break (NA, not True).
        full = (~np.isnan(prev_w)).sum(axis=0) == w

        curr = sub[idx]
        valid_k = ~np.isnan(curr) & full
        bo_k = (curr > max_w) & valid_k

        has_event |= bo_k
        has_valid |= valid_k

    res_arr = np.full(n_tickers, pd.NA, dtype=object)
    res_arr[has_valid] = False
    res_arr[has_event] = True

    return pd.Series(res_arr, index=index, dtype="boolean")


def signals_within(closes: pd.DataFrame, days: int = 5, prepared: bool = False) -> pd.DataFrame:
    """Compute 'within last N days' signals from wide daily price closes.

    Returns pd.DataFrame indexed by ticker (closes.columns) with nullable boolean
    and float columns for days_since.
    """
    if not prepared:
        closes = prepare(closes)

    tickers = closes.columns
    if closes.empty:
        df_out = pd.DataFrame(index=tickers)
        for col in SIGNAL_LABELS:
            if col.startswith("days_since_"):
                df_out[col] = pd.Series(dtype=float)
            else:
                df_out[col] = pd.Series(dtype="boolean")
        return df_out

    arr = closes.to_numpy(dtype=float)

    # Rolling SMAs
    sma50 = _fast_sma(arr, 50)
    sma200 = _fast_sma(arr, 200)

    # Crosses
    px_up_50, px_down_50, valid_px_50 = _cross_matrix(arr, sma50)
    px_up_200, px_down_200, valid_px_200 = _cross_matrix(arr, sma200)
    golden, death, valid_sma_sma = _cross_matrix(sma50, sma200)

    res = {
        "cross_up_sma50": _eval_within_matrix(px_up_50, valid_px_50, days, tickers),
        "cross_down_sma50": _eval_within_matrix(px_down_50, valid_px_50, days, tickers),
        "cross_up_sma200": _eval_within_matrix(px_up_200, valid_px_200, days, tickers),
        "cross_down_sma200": _eval_within_matrix(px_down_200, valid_px_200, days, tickers),
        "golden_cross": _eval_within_matrix(golden, valid_sma_sma, days, tickers),
        "death_cross": _eval_within_matrix(death, valid_sma_sma, days, tickers),
        "days_since_cross_up_sma200": _days_since_matrix(px_up_200, tickers),
        "days_since_cross_down_sma200": _days_since_matrix(px_down_200, tickers),
    }

    # Breakouts
    for window in (20, 60, 120, 250):
        res[f"breakout_{window}"] = _eval_breakout_within(arr, window, days, tickers)

    return pd.DataFrame(res, index=tickers)


def event_matrix(closes: pd.DataFrame, name: str, prepared: bool = False) -> pd.DataFrame:
    """Compute boolean event matrix for a single technical signal across daily history.

    Returns pd.DataFrame with same shape (index=trading days, columns=tickers) as closes,
    with True on days the signal fired and False otherwise.

    Raises ValueError if name is not a valid boolean signal.
    """
    boolean_signals = {
        "cross_up_sma50",
        "cross_down_sma50",
        "cross_up_sma200",
        "cross_down_sma200",
        "golden_cross",
        "death_cross",
        "breakout_20",
        "breakout_60",
        "breakout_120",
        "breakout_250",
    }
    if name not in boolean_signals:
        raise ValueError(f"Unknown boolean signal: {name}")

    if not prepared:
        closes = prepare(closes)

    if closes.empty:
        return pd.DataFrame(False, index=closes.index, columns=closes.columns, dtype=bool)

    arr = closes.to_numpy(dtype=float)

    if name in {"cross_up_sma50", "cross_down_sma50"}:
        sma50 = _fast_sma(arr, 50)
        px_up, px_down, _ = _cross_matrix(arr, sma50)
        matrix = px_up if name == "cross_up_sma50" else px_down
        return pd.DataFrame(matrix, index=closes.index, columns=closes.columns, dtype=bool)

    if name in {"cross_up_sma200", "cross_down_sma200"}:
        sma200 = _fast_sma(arr, 200)
        px_up, px_down, _ = _cross_matrix(arr, sma200)
        matrix = px_up if name == "cross_up_sma200" else px_down
        return pd.DataFrame(matrix, index=closes.index, columns=closes.columns, dtype=bool)

    if name in {"golden_cross", "death_cross"}:
        sma50 = _fast_sma(arr, 50)
        sma200 = _fast_sma(arr, 200)
        golden, death, _ = _cross_matrix(sma50, sma200)
        matrix = golden if name == "golden_cross" else death
        return pd.DataFrame(matrix, index=closes.index, columns=closes.columns, dtype=bool)

    # Breakout signals (breakout_20, breakout_60, breakout_120, breakout_250)
    w = int(name.split("_")[1])
    shifted = closes.shift(1)
    rolled_max = shifted.rolling(w, min_periods=w).max()
    rolled_count = shifted.rolling(w, min_periods=w).count()
    full = rolled_count == w
    bo = (closes > rolled_max) & full & closes.notna()
    return bo.astype(bool)
