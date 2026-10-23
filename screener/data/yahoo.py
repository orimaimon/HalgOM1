"""
Yahoo Screener provider (yfinance.screen fallback).

Fetches stock universe data from Yahoo Finance via `yfinance.screen` and maps
quotes to canonical provider columns defined in `screener.fields.COLUMNS`.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from screener.fields import COLUMNS, provider_columns

log = logging.getLogger(__name__)

MAX_REQUESTS = 40
PAGE_SIZE = 250
_sleep = time.sleep


class YahooError(RuntimeError):
    """Raised when Yahoo screener returns no data or fails catastrophically."""
    pass


# Map quote dictionary keys returned by yf.screen to canonical column names
# and specify unit conversion or normalization logic.
# Only canonical provider keys from fields.COLUMNS are mapped here.
MAPPING: dict[str, str] = {
    # Ticker & descriptive
    "symbol": "ticker",                      # String (BRK-B converted to BRK.B)
    "longName": "company",                   # String (fallback to shortName if missing)
    "exchange": "exchange",                  # String (NMS/NGM/NCM -> NASDAQ, NYQ -> NYSE, ASE -> AMEX)

    # Price & Volume
    "regularMarketPrice": "price",           # Money ($)
    "regularMarketChange": "change_abs",     # Change ($)
    "regularMarketChangePercent": "change",  # Percent (%) e.g. 5.0 == 5%
    "regularMarketVolume": "volume",        # Count (shares)
    "averageDailyVolume3Month": "avg_volume",# Count (shares)
    "marketCap": "market_cap",               # Money ($)
    "sharesOutstanding": "shares_out",       # Count (shares)

    # Valuation & Fundamentals
    "trailingPE": "pe",                      # Ratio
    "epsTrailingTwelveMonths": "eps",        # Money ($ per share)
    "priceToBook": "pb",                     # Ratio

    # Moving averages & 52-week extremes
    "fiftyDayAverage": "sma50",              # Price ($)
    "twoHundredDayAverage": "sma200",        # Price ($)
    "fiftyTwoWeekHigh": "high52",            # Price ($)
    "fiftyTwoWeekLow": "low52",              # Price ($)

    # Calendar & Dividends
    "earningsTimestamp": "earnings_date",    # Timestamp (epoch seconds -> datetime64[ns])
    "dividendYield": "div_yield",            # Percent (%) e.g. 2.31 == 2.31% (if fraction, x100)
}

EXCHANGE_MAP = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "AMEX",
}


def default_query() -> EquityQuery:
    """Default EquityQuery: US region, major exchanges, market cap > $50M."""
    return EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("is-in", ["exchange", "NMS", "NYQ", "ASE", "NGM", "NCM"]),
            EquityQuery("gt", ["intradaymarketcap", 50000000]),
        ],
    )


def source_meta() -> dict:
    """Return provider metadata for snapshot/store compatibility."""
    return {
        "provider": "yahoo",
        "update_mode": "delayed",
        "delay_min": None,
        "raw": None,
    }


def clean_quotes(quotes: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform quote dicts into a canonical provider DataFrame."""
    if not quotes:
        return pd.DataFrame()

    df = pd.DataFrame(quotes)

    # Normalize ticker: BRK-B -> BRK.B
    if "symbol" in df.columns:
        df["ticker"] = df["symbol"].astype(str).str.replace("-", ".", regex=False)
    else:
        df["ticker"] = np.nan

    # Company name: longName with fallback to shortName
    company = pd.Series(index=df.index, dtype=object)
    if "longName" in df.columns:
        company = company.fillna(df["longName"])
    if "shortName" in df.columns:
        company = company.fillna(df["shortName"])
    df["company"] = company

    # Exchange mapping
    if "exchange" in df.columns:
        df["exchange"] = df["exchange"].map(lambda e: EXCHANGE_MAP.get(e, e))

    # Single-field simple mappings
    for raw_key, canon_key in MAPPING.items():
        if raw_key in ("symbol", "longName", "exchange"):
            continue
        if raw_key in df.columns:
            df[canon_key] = df[raw_key]

    # Unit check & fix for div_yield:
    # yfinance quotes sometimes provide dividendYield as percentage (e.g. 2.31) or trailingAnnualDividendYield as fraction (0.0231).
    if "div_yield" in df.columns:
        df["div_yield"] = pd.to_numeric(df["div_yield"], errors="coerce")
        mask = df["div_yield"].notna() & (df["div_yield"] > 0) & (df["div_yield"] <= 1.0)
        df.loc[mask, "div_yield"] = df.loc[mask, "div_yield"] * 100.0

    # earnings_date conversion: epoch seconds -> Eastern-normalized datetime64[ns]
    if "earnings_date" in df.columns:
        s = pd.to_numeric(df["earnings_date"], errors="coerce")
        dt_series = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        df["earnings_date"] = (
            dt_series.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
    else:
        df["earnings_date"] = pd.to_datetime(pd.Series(np.nan, index=df.index))

    # Build dictionary of missing provider columns to avoid DataFrame fragmentation
    prov_cols = provider_columns()
    missing_dict = {}
    for col_key in prov_cols:
        if col_key not in df.columns:
            col_def = COLUMNS[col_key]
            if col_def.kind in ("price", "pct", "ratio", "money", "count", "int", "days"):
                missing_dict[col_key] = pd.Series(np.nan, index=df.index, dtype=float)
            elif col_def.kind == "date":
                missing_dict[col_key] = pd.to_datetime(pd.Series(np.nan, index=df.index))
            else:
                missing_dict[col_key] = pd.Series(np.nan, index=df.index, dtype=object)
        else:
            col_def = COLUMNS[col_key]
            if col_def.kind in ("price", "pct", "ratio", "money", "count", "int", "days"):
                df[col_key] = pd.to_numeric(df[col_key], errors="coerce").astype(float)

    if missing_dict:
        missing_df = pd.DataFrame(missing_dict, index=df.index)
        df = pd.concat([df, missing_df], axis=1)

    # Deduplicate by ticker if necessary
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    # Keep ONLY provider columns
    df = df[prov_cols]

    return df


def fetch_universe(
    query: EquityQuery | None = None,
    max_requests: int = MAX_REQUESTS,
) -> pd.DataFrame:
    """Fetch US stocks universe from Yahoo screener via yf.screen.

    Pages through results up to `max_requests` (40 requests max by default).
    Returns a DataFrame with all provider columns.
    """
    if query is None:
        query = default_query()

    all_quotes: list[dict[str, Any]] = []
    offset = 0
    total = None

    for request_num in range(max_requests):
        if request_num > 0:
            _sleep(1)

        res = yf.screen(
            query,
            offset=offset,
            size=PAGE_SIZE,
            sortField="intradaymarketcap",
            sortAsc=False,
        )

        if not isinstance(res, dict):
            break

        total = res.get("total", total)
        quotes = res.get("quotes", [])
        if not quotes:
            break

        all_quotes.extend(quotes)
        offset += PAGE_SIZE

        if (total is not None and offset >= total) or len(quotes) < PAGE_SIZE:
            break
    else:
        if total is not None and len(all_quotes) < total:
            log.warning(
                "Yahoo screener query cut off at max_requests (%d). Fetched %d of %d.",
                max_requests,
                len(all_quotes),
                total,
            )

    if not all_quotes:
        raise YahooError("empty universe returned from Yahoo screener")

    df = clean_quotes(all_quotes)
    if len(df) == 0:
        raise YahooError("empty universe returned after cleaning")

    df.attrs["source"] = source_meta()
    return df
