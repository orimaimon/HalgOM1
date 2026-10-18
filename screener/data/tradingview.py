"""
TradingView scanner provider.

Talks to TradingView's public scanner endpoint directly with `requests` (the same endpoint
the `tradingview-screener` package wraps). It is unofficial and ~15 minutes delayed, so it
is kept behind this module: swapping providers means writing another `fetch_universe()`
that returns the same canonical columns (see fields.COLUMNS).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from screener.fields import COLUMNS, EXTRA_TV_FIELDS, tv_fields

log = logging.getLogger(__name__)

SCAN_URL = "https://scanner.tradingview.com/{market}/scan"
HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "origin": "https://www.tradingview.com",
    "referer": "https://www.tradingview.com/",
    "accept": "application/json",
}
US_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]

# Benchmarks for the market-regime header (same inputs as DB/build_simulation_db.py).
MARKET_SYMBOLS = {"SPY": "AMEX:SPY", "QQQ": "NASDAQ:QQQ", "HYG": "AMEX:HYG", "VIX": "CBOE:VIX"}


class TradingViewError(RuntimeError):
    pass


def _scan(market: str, query: dict, timeout: float) -> list[dict]:
    try:
        r = requests.post(SCAN_URL.format(market=market), json=query, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        raise TradingViewError(f"network error: {e}") from e
    if r.status_code != 200:
        raise TradingViewError(f"HTTP {r.status_code}: {r.text[:200]}")
    payload = r.json()
    if "data" not in payload or payload["data"] is None:
        raise TradingViewError(f"unexpected response: {str(payload)[:200]}")
    return payload["data"]


def universe_query(fields: list[str]) -> dict:
    """US primary listings: common stocks + ADRs (no ETFs, preferreds, warrants, OTC)."""
    return {
        "markets": ["america"],
        "columns": fields,
        "filter": [
            {"left": "exchange", "operation": "in_range", "right": US_EXCHANGES},
            {"left": "is_primary", "operation": "equal", "right": True},
        ],
        "filter2": {"operator": "or", "operands": [
            {"operation": {"operator": "and", "operands": [
                {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}},
            ]}},
            {"operation": {"operator": "and", "operands": [
                {"expression": {"left": "type", "operation": "equal", "right": "dr"}},
            ]}},
        ]},
        "options": {"lang": "en"},
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, 20000],
    }


def fetch_universe(timeout: float = 60) -> pd.DataFrame:
    """Fetch the full US stock universe as a DataFrame keyed by canonical column names."""
    mapping = tv_fields()
    fields = list(mapping) + list(EXTRA_TV_FIELDS)
    rows = _scan("america", universe_query(fields), timeout)
    if not rows:
        raise TradingViewError("empty universe returned")
    df = pd.DataFrame([r["d"] for r in rows], columns=fields)
    df.insert(0, "tv_symbol", [r["s"] for r in rows])
    df = df.rename(columns=mapping)
    log.info("TradingView universe: %d rows", len(df))
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and drop rows that would make results look untrustworthy."""
    df = df.copy()
    for key, col in COLUMNS.items():
        if key not in df.columns:
            continue
        if col.kind == "date":
            df[key] = pd.to_datetime(pd.to_numeric(df[key], errors="coerce"), unit="s", utc=True)
            df[key] = df[key].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        elif col.kind != "text":
            df[key] = pd.to_numeric(df[key], errors="coerce").astype(float)

    n0 = len(df)
    if "is_primary" in df.columns:
        df = df[df["is_primary"].fillna(False).astype(bool)]
    if "currency" in df.columns:
        df = df[df["currency"].isna() | (df["currency"] == "USD")]
    df = df[df["price"].notna() & (df["price"] > 0)]
    # Same ticker twice (share classes listed oddly) — keep the larger one.
    df = df.sort_values("market_cap", ascending=False, na_position="last")
    df = df.drop_duplicates("ticker", keep="first")
    df = df.drop(columns=[c for c in EXTRA_TV_FIELDS if c in df.columns])
    if len(df) < n0:
        log.info("clean: dropped %d rows", n0 - len(df))
    return df.reset_index(drop=True)


def fetch_market(timeout: float = 20) -> pd.DataFrame:
    """SPY / QQQ / HYG / VIX quotes with moving averages, indexed by short name."""
    query = {
        "symbols": {"tickers": list(MARKET_SYMBOLS.values())},
        "columns": ["close", "change", "SMA20", "SMA50", "SMA200"],
        "range": [0, len(MARKET_SYMBOLS)],
    }
    rows = _scan("global", query, timeout)
    by_symbol = {r["s"]: r["d"] for r in rows}
    data = {name: by_symbol.get(sym, [np.nan] * 5) for name, sym in MARKET_SYMBOLS.items()}
    return pd.DataFrame.from_dict(data, orient="index", columns=["close", "change", "sma20", "sma50", "sma200"]
                                  ).apply(pd.to_numeric, errors="coerce")
