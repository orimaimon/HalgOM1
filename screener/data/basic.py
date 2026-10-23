"""
Basic mode data provider: combines Nasdaq + Yahoo prices with slow fields from the last snapshot.

Used when TradingView blocks us or is unavailable.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from screener.data import guard, nasdaq, yahoo
from screener.fields import provider_columns

if TYPE_CHECKING:
    from screener.data.store import Store

log = logging.getLogger(__name__)

FRESH: list[str] = [
    "price",
    "change",
    "change_abs",
    "volume",
    "avg_volume",
    "market_cap",
    "shares_out",
    "pe",
    "eps",
    "pb",
    "sma50",
    "sma200",
    "high52",
    "low52",
    "earnings_date",
    "div_yield",
]


def combine(
    nasdaq_df: pd.DataFrame,
    yahoo_df: pd.DataFrame | None = None,
    snapshot: pd.DataFrame | None = None,
    snapshot_day: dt.date | None = None,
) -> pd.DataFrame:
    """Pure combination of Nasdaq, Yahoo, and snapshot DataFrames into basic mode universe."""
    if nasdaq_df is None or nasdaq_df.empty or "ticker" not in nasdaq_df.columns:
        nasdaq_indexed = pd.DataFrame()
        tickers = pd.Index([])
    else:
        nasdaq_indexed = nasdaq_df.drop_duplicates(subset=["ticker"]).set_index("ticker")
        tickers = nasdaq_indexed.index

    yahoo_indexed = (
        yahoo_df.drop_duplicates(subset=["ticker"]).set_index("ticker")
        if (yahoo_df is not None and not yahoo_df.empty and "ticker" in yahoo_df.columns)
        else None
    )

    snap_indexed = (
        snapshot.drop_duplicates(subset=["ticker"]).set_index("ticker")
        if (snapshot is not None and not snapshot.empty and "ticker" in snapshot.columns)
        else None
    )

    res_dict: dict[str, pd.Series] = {}

    # 1. Ticker column
    res_dict["ticker"] = pd.Series(tickers.values, index=tickers, dtype=object)

    # 2. Extra ipo_year column from Nasdaq if present
    if "ipo_year" in nasdaq_indexed.columns:
        res_dict["ipo_year"] = nasdaq_indexed["ipo_year"]

    # 3. FRESH columns: Yahoo > Nasdaq > NaN
    for col in FRESH:
        y_s = (
            yahoo_indexed[col].reindex(tickers)
            if (yahoo_indexed is not None and col in yahoo_indexed.columns)
            else None
        )
        n_s = nasdaq_indexed[col] if col in nasdaq_indexed.columns else None

        if y_s is not None and n_s is not None:
            combined = y_s.fillna(n_s)
        elif y_s is not None:
            combined = y_s
        elif n_s is not None:
            combined = n_s
        else:
            combined = pd.Series(np.nan, index=tickers, dtype=float)

        res_dict[col] = combined

    # 4. Non-FRESH provider columns (excluding ticker): snapshot > Nasdaq fallback > NaN
    all_prov = provider_columns()
    non_fresh = [c for c in all_prov if c not in FRESH and c != "ticker"]

    for col in non_fresh:
        if snap_indexed is not None and col in snap_indexed.columns:
            s = snap_indexed[col].reindex(tickers)
        else:
            s = pd.Series(np.nan, index=tickers, dtype=object)

        # Fallback to Nasdaq for sector, industry, country, company, sec_type, etc.
        if col in nasdaq_indexed.columns:
            s = s.fillna(nasdaq_indexed[col])

        res_dict[col] = s

    df = pd.DataFrame(res_dict, index=tickers)
    df = df.reset_index(drop=True)

    # Ensure every fields.provider_columns() key exists in the result
    for col in all_prov:
        if col not in df.columns:
            df[col] = np.nan

    # Metadata for attrs["source"]
    snap_day_str = snapshot_day.isoformat() if snapshot_day is not None else None
    stale_cols = (
        [c for c in non_fresh if c in snapshot.columns]
        if snapshot is not None
        else []
    )
    missing_cols = [c for c in all_prov if df[c].isna().all()]

    df.attrs["source"] = {
        "provider": "basic",
        "update_mode": "delayed",
        "delay_min": None,
        "raw": None,
        "basic": True,
        "snapshot_day": snap_day_str,
        "stale": stale_cols,
        "missing": missing_cols,
    }

    return df


def fetch_basic(store: Store) -> pd.DataFrame:
    """Fetch basic universe using Nasdaq and Yahoo via request guard, combined with store snapshot."""
    nasdaq_df = guard.call("nasdaq", nasdaq.fetch_universe)

    yahoo_df = None
    try:
        yahoo_df = guard.call("yahoo", yahoo.fetch_universe)
    except Exception as e:
        log.warning("Yahoo screener failed in basic mode: %s", e)

    snapshot_df = None
    snapshot_day = None
    if store is not None:
        dates = store.snapshot_dates()
        if dates:
            snapshot_day = dates[-1]
            try:
                snapshot_df = store.load_snapshot(snapshot_day)
            except Exception as e:
                log.warning("cannot read snapshot for basic mode from %s: %s", snapshot_day, e)
                snapshot_df = None
                snapshot_day = None

    return combine(nasdaq_df, yahoo_df, snapshot_df, snapshot_day)
