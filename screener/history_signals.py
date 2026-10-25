"""Pure functions for history-based signals ("within the last N days").

Calculates crosses of SMA50/SMA200 and breakouts over multiple windows
from wide daily price history (index=trading days, columns=tickers) using DuckDB SQL.
"""
from __future__ import annotations

import duckdb
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

    df_long = closes.reset_index().rename(columns={closes.index.name or "index": "date"})

    con = duckdb.connect()
    sql = f"""
    WITH unpivoted AS (
        UNPIVOT df_long
        ON COLUMNS(* EXCLUDE (date))
        INTO NAME ticker VALUE close
    ),
    ordered AS (
        SELECT ticker, date, close,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) - 1 AS bar_idx,
               COUNT(*) OVER (PARTITION BY ticker) AS total_bars
        FROM unpivoted
    ),
    smas AS (
        SELECT *,
               CASE WHEN COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) = 50
                    THEN AVG(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
                    ELSE NULL END AS sma50,
               CASE WHEN COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) = 200
                    THEN AVG(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)
                    ELSE NULL END AS sma200
        FROM ordered
    ),
    lags AS (
        SELECT *,
               LAG(close) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_close,
               LAG(sma50) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_sma50,
               LAG(sma200) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_sma200
        FROM smas
    ),
    events AS (
        SELECT *,
               (close IS NOT NULL AND sma50 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma50 IS NOT NULL) AS valid_px_50,
               ((close IS NOT NULL AND sma50 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma50 IS NOT NULL) AND (close > sma50) AND (prev_close <= prev_sma50)) AS cross_up_sma50,
               ((close IS NOT NULL AND sma50 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma50 IS NOT NULL) AND (close < sma50) AND (prev_close >= prev_sma50)) AS cross_down_sma50,

               (close IS NOT NULL AND sma200 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma200 IS NOT NULL) AS valid_px_200,
               ((close IS NOT NULL AND sma200 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma200 IS NOT NULL) AND (close > sma200) AND (prev_close <= prev_sma200)) AS cross_up_sma200,
               ((close IS NOT NULL AND sma200 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma200 IS NOT NULL) AND (close < sma200) AND (prev_close >= prev_sma200)) AS cross_down_sma200,

               (sma50 IS NOT NULL AND sma200 IS NOT NULL AND prev_sma50 IS NOT NULL AND prev_sma200 IS NOT NULL) AS valid_sma_sma,
               ((sma50 IS NOT NULL AND sma200 IS NOT NULL AND prev_sma50 IS NOT NULL AND prev_sma200 IS NOT NULL) AND (sma50 > sma200) AND (prev_sma50 <= prev_sma200)) AS golden_cross,
               ((sma50 IS NOT NULL AND sma200 IS NOT NULL AND prev_sma50 IS NOT NULL AND prev_sma200 IS NOT NULL) AND (sma50 < sma200) AND (prev_sma50 >= prev_sma200)) AS death_cross,

               -- Breakouts
               (close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) = 20) AS valid_bo_20,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) = 20) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)) AS breakout_20,

               (close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) = 60) AS valid_bo_60,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) = 60) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING)) AS breakout_60,

               (close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) = 120) AS valid_bo_120,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) = 120) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING)) AS breakout_120,

               (close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING) = 250) AS valid_bo_250,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING) = 250) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING)) AS breakout_250
        FROM lags
    )
    SELECT ticker,
           CASE WHEN BOOL_OR(cross_up_sma50) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_px_50) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS cross_up_sma50,
           CASE WHEN BOOL_OR(cross_down_sma50) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_px_50) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS cross_down_sma50,
           CASE WHEN BOOL_OR(cross_up_sma200) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_px_200) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS cross_up_sma200,
           CASE WHEN BOOL_OR(cross_down_sma200) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_px_200) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS cross_down_sma200,
           CASE WHEN BOOL_OR(golden_cross) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_sma_sma) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS golden_cross,
           CASE WHEN BOOL_OR(death_cross) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_sma_sma) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS death_cross,

           CASE WHEN MAX(bar_idx) FILTER (WHERE cross_up_sma200) IS NOT NULL THEN (MAX(total_bars) - 1) - MAX(bar_idx) FILTER (WHERE cross_up_sma200) ELSE NULL END AS days_since_cross_up_sma200,
           CASE WHEN MAX(bar_idx) FILTER (WHERE cross_down_sma200) IS NOT NULL THEN (MAX(total_bars) - 1) - MAX(bar_idx) FILTER (WHERE cross_down_sma200) ELSE NULL END AS days_since_cross_down_sma200,

           CASE WHEN BOOL_OR(breakout_20) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_bo_20) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS breakout_20,
           CASE WHEN BOOL_OR(breakout_60) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_bo_60) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS breakout_60,
           CASE WHEN BOOL_OR(breakout_120) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_bo_120) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS breakout_120,
           CASE WHEN BOOL_OR(breakout_250) FILTER (WHERE bar_idx >= total_bars - {days}) THEN TRUE WHEN BOOL_OR(valid_bo_250) FILTER (WHERE bar_idx >= total_bars - {days}) THEN FALSE ELSE NULL END AS breakout_250
    FROM events
    GROUP BY ticker
    """
    raw_res = con.execute(sql).df().set_index("ticker").reindex(tickers)
    con.close()

    df_out = pd.DataFrame(index=tickers)
    for col in SIGNAL_LABELS:
        if col.startswith("days_since_"):
            df_out[col] = raw_res[col].astype(float)
        else:
            df_out[col] = raw_res[col].astype("boolean")

    return df_out


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

    df_long = closes.reset_index().rename(columns={closes.index.name or "index": "date"})

    con = duckdb.connect()
    sql = f"""
    WITH unpivoted AS (
        UNPIVOT df_long
        ON COLUMNS(* EXCLUDE (date))
        INTO NAME ticker VALUE close
    ),
    ordered AS (
        SELECT ticker, date, close,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) - 1 AS bar_idx
        FROM unpivoted
    ),
    smas AS (
        SELECT *,
               CASE WHEN COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) = 50
                    THEN AVG(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
                    ELSE NULL END AS sma50,
               CASE WHEN COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) = 200
                    THEN AVG(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)
                    ELSE NULL END AS sma200
        FROM ordered
    ),
    lags AS (
        SELECT *,
               LAG(close) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_close,
               LAG(sma50) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_sma50,
               LAG(sma200) OVER (PARTITION BY ticker ORDER BY bar_idx) AS prev_sma200
        FROM smas
    ),
    events AS (
        SELECT ticker, date,
               ((close IS NOT NULL AND sma50 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma50 IS NOT NULL) AND (close > sma50) AND (prev_close <= prev_sma50)) AS cross_up_sma50,
               ((close IS NOT NULL AND sma50 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma50 IS NOT NULL) AND (close < sma50) AND (prev_close >= prev_sma50)) AS cross_down_sma50,

               ((close IS NOT NULL AND sma200 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma200 IS NOT NULL) AND (close > sma200) AND (prev_close <= prev_sma200)) AS cross_up_sma200,
               ((close IS NOT NULL AND sma200 IS NOT NULL AND prev_close IS NOT NULL AND prev_sma200 IS NOT NULL) AND (close < sma200) AND (prev_close >= prev_sma200)) AS cross_down_sma200,

               ((sma50 IS NOT NULL AND sma200 IS NOT NULL AND prev_sma50 IS NOT NULL AND prev_sma200 IS NOT NULL) AND (sma50 > sma200) AND (prev_sma50 <= prev_sma200)) AS golden_cross,
               ((sma50 IS NOT NULL AND sma200 IS NOT NULL AND prev_sma50 IS NOT NULL AND prev_sma200 IS NOT NULL) AND (sma50 < sma200) AND (prev_sma50 >= prev_sma200)) AS death_cross,

               -- Breakouts
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) = 20) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)) AS breakout_20,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) = 60) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING)) AS breakout_60,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) = 120) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING)) AS breakout_120,
               ((close IS NOT NULL AND COUNT(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING) = 250) AND close > MAX(close) OVER (PARTITION BY ticker ORDER BY bar_idx ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING)) AS breakout_250
        FROM lags
    )
    SELECT date, ticker, COALESCE({name}, FALSE) AS fired FROM events
    """
    raw_df = con.execute(sql).df()
    con.close()

    pivoted = raw_df.pivot(index="date", columns="ticker", values="fired").fillna(False).astype(bool)
    return pivoted.reindex(index=closes.index, columns=closes.columns, fill_value=False)
