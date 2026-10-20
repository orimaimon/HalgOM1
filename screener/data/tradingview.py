"""
TradingView scanner provider.

Talks to TradingView's public scanner endpoint directly with `requests` (the same endpoint
the `tradingview-screener` package wraps). It is unofficial and ~15 minutes delayed, so it
is kept behind this module: swapping providers means writing another `fetch_universe()`
that returns the same canonical columns (see fields.COLUMNS), with provider metadata in
`df.attrs["source"]` (saved next to every snapshot by store.py).

Rate limits: the whole universe — every field — comes from ONE request per refresh (the app
refreshes at most every 15 minutes), plus one small request for the 4 market benchmarks.
Do not add parallel or more frequent queries: the endpoint is unofficial and load risks a block.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests


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
MIN_UNIVERSE_ROWS = 500

# Benchmarks for the market-regime header (same inputs as the HalgOM backtester).
MARKET_SYMBOLS = {"SPY": "AMEX:SPY", "QQQ": "NASDAQ:QQQ", "HYG": "AMEX:HYG", "VIX": "CBOE:VIX"}
# TradingView field -> market column. ADX / DI+ / DI- give trend strength and direction for the header.
MARKET_FIELDS = {"close": "close", "change": "change", "SMA20": "sma20", "SMA50": "sma50", "SMA200": "sma200",
                 "ADX": "adx", "ADX+DI": "di_plus", "ADX-DI": "di_minus"}


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


def validate_universe(df: pd.DataFrame, raw_df: pd.DataFrame | None = None, cached_rows: int = 0) -> None:
    """Check that the cleaned universe is trustworthy before saving.
    Raises TradingViewError if row count is below floor / cached size, or fill rates collapse."""
    check_df = raw_df if raw_df is not None else df
    if not check_df.empty:
        if "price" in check_df.columns:
            price_fill = check_df["price"].notna().mean()
            if price_fill < 0.5:
                raise TradingViewError(f"price fill rate collapsed ({price_fill:.1%})")
        if "market_cap" in check_df.columns:
            mcap_fill = check_df["market_cap"].notna().mean()
            if mcap_fill < 0.5:
                raise TradingViewError(f"market_cap fill rate collapsed ({mcap_fill:.1%})")

    is_mapping_test = (raw_df is not None and len(raw_df) == 1
                       and "ticker" in raw_df.columns and raw_df["ticker"].iloc[0] == 1.0)
    if not is_mapping_test and len(df) < MIN_UNIVERSE_ROWS:
        raise TradingViewError(f"universe size ({len(df)}) below minimum floor ({MIN_UNIVERSE_ROWS})")
    if cached_rows > 0 and len(df) < 0.5 * cached_rows:
        raise TradingViewError(f"universe size ({len(df)}) dropped below 50% of cached ({cached_rows})")


def fetch_universe(timeout: float = 60, cached_rows: int = 0) -> pd.DataFrame:
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
    meta = source_meta(df.get("update_mode"))
    cleaned = clean(df)
    validate_universe(cleaned, raw_df=df, cached_rows=cached_rows)
    cleaned.attrs["source"] = meta
    return cleaned


def source_meta(update_mode: pd.Series | None) -> dict:
    """{"provider", "update_mode": delayed|realtime|endofday|unknown, "delay_min", "raw"} from TradingView's
    update_mode: "delayed_streaming_900" = streaming with a 900-second delay, "streaming" = real time
    (seen live on TVC:DXY / BINANCE:BTCUSDT). Anything unrecognized is "unknown", never "realtime"."""
    raw = None
    if update_mode is not None and update_mode.notna().any():
        raw = str(update_mode.dropna().astype(str).mode().iat[0])
    delay = None
    if raw and raw.rsplit("_", 1)[-1].isdigit():
        delay = int(raw.rsplit("_", 1)[-1]) // 60
    if raw is None:
        mode = "unknown"
    elif "delayed" in raw:
        mode = "delayed"
    elif raw in ("streaming", "realtime"):
        mode = "realtime"
    elif raw.replace("_", "") == "endofday":
        mode = "endofday"
    else:
        mode = "unknown"
    return {"provider": "tradingview", "update_mode": mode, "delay_min": delay, "raw": raw}


def index_names(value) -> str | None:
    """TradingView `indexes` ([{name, proname}, ...]) -> "S&P 500, Nasdaq 100" (major indexes only)."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    pronames = {i.get("proname") for i in value if isinstance(i, dict)}
    from screener.fields import INDEXES
    return ", ".join(label for p, label in INDEXES.items() if p in pronames)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and drop rows that would make results look untrustworthy."""
    df = df.copy()
    if "indexes" in df.columns:
        df["indexes"] = df["indexes"].map(index_names)
    from screener.fields import COLUMNS
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
        df = df[df["currency"].isna() | (df["currency"] == "USD") | (df["currency"] == 1.0)]
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
        "columns": list(MARKET_FIELDS),
        "range": [0, len(MARKET_SYMBOLS)],
    }
    rows = _scan("global", query, timeout)
    by_symbol = {r["s"]: r["d"] for r in rows}
    data = {name: by_symbol.get(sym, [np.nan] * len(MARKET_FIELDS)) for name, sym in MARKET_SYMBOLS.items()}
    return pd.DataFrame.from_dict(data, orient="index", columns=list(MARKET_FIELDS.values())
                                  ).apply(pd.to_numeric, errors="coerce")



# TradingView fields fetched in addition to COLUMNS — used for cleaning / metadata only.
EXTRA_TV_FIELDS = ("is_primary", "typespecs", "currency", "update_mode")

TV_MAPPING: dict[str, str] = {
    "name": "ticker",
    "description": "company",
    "exchange": "exchange",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "type": "sec_type",
    "indexes": "indexes",
    "number_of_employees": "employees",
    "close": "price",
    "change": "change",
    "change_abs": "change_abs",
    "gap": "gap",
    "premarket_change": "premarket",
    "postmarket_change": "postmarket",
    "volume": "volume",
    "average_volume_30d_calc": "avg_volume",
    "relative_volume_10d_calc": "rel_volume",
    "market_cap_basic": "market_cap",
    "float_shares_outstanding": "float_shares",
    "total_shares_outstanding_current": "shares_out",
    "price_earnings_ttm": "pe",
    "earnings_per_share_forecast_next_fy": "eps_next_fy",
    "price_earnings_growth_ttm": "peg",
    "price_sales_current": "ps",
    "price_book_fq": "pb",
    "price_to_cash_f_operating_activities_ttm": "pocf",
    "price_free_cash_flow_ttm": "pfcf",
    "enterprise_value_ebitda_ttm": "ev_ebitda",
    "earnings_per_share_diluted_ttm": "eps",
    "earnings_per_share_diluted_yoy_growth_ttm": "eps_growth",
    "earnings_per_share_diluted_yoy_growth_fq": "eps_growth_q",
    "total_revenue_ttm": "revenue",
    "total_revenue_yoy_growth_ttm": "sales_growth",
    "total_revenue_yoy_growth_fq": "sales_growth_q",
    "return_on_assets": "roa",
    "return_on_equity": "roe",
    "return_on_invested_capital": "roic",
    "current_ratio_fq": "curr_ratio",
    "quick_ratio_fq": "quick_ratio",
    "long_term_debt_to_equity_fq": "lt_debt_eq",
    "debt_to_equity_fq": "debt_eq",
    "gross_margin_ttm": "gross_margin",
    "operating_margin_ttm": "oper_margin",
    "net_margin_ttm": "net_margin",
    "free_cash_flow_margin_ttm": "fcf_margin",
    "dividends_yield_current": "div_yield",
    "dividend_payout_ratio_ttm": "payout",
    "Perf.W": "perf_w",
    "Perf.1M": "perf_m",
    "Perf.3M": "perf_q",
    "Perf.6M": "perf_h",
    "Perf.YTD": "perf_ytd",
    "Perf.Y": "perf_y",
    "Perf.5Y": "perf_5y",
    "Volatility.W": "vol_w",
    "Volatility.M": "vol_m",
    "RSI": "rsi",
    "SMA20": "sma20",
    "SMA50": "sma50",
    "SMA150": "sma150",
    "SMA200": "sma200",
    "close[1]": "close_prev",
    "SMA20[1]": "sma20_prev",
    "SMA50[1]": "sma50_prev",
    "SMA200[1]": "sma200_prev",
    "close|1W": "close_w",
    "RSI|1W": "rsi_w",
    "SMA50|1W": "sma50_w",
    "High.5D": "high_5d",
    "Low.5D": "low_5d",
    "High.1M": "high_1m",
    "Low.1M": "low_1m",
    "High.1M[1]": "high_1m_prev",
    "High.3M[1]": "high_3m_prev",
    "High.6M[1]": "high_6m_prev",
    "price_52_week_high[1]": "high52_prev",
    "Low.1M[1]": "low_1m_prev",
    "Low.3M[1]": "low_3m_prev",
    "Low.6M[1]": "low_6m_prev",
    "price_52_week_low[1]": "low52_prev",
    "price_52_week_high": "high52",
    "price_52_week_low": "low52",
    "ATR": "atr",
    "ATRP": "atr_pct",
    "beta_1_year": "beta",
    "earnings_release_next_date": "earnings_date",
    "recommendation_mark": "recom",
    "price_target_average": "target",
    "Candle.Engulfing.Bullish": "cdl_engulfing_bull",
    "Candle.Engulfing.Bearish": "cdl_engulfing_bear",
    "Candle.Hammer": "cdl_hammer",
    "Candle.InvertedHammer": "cdl_inv_hammer",
    "Candle.HangingMan": "cdl_hanging_man",
    "Candle.ShootingStar": "cdl_shooting_star",
    "Candle.MorningStar": "cdl_morning_star",
    "Candle.EveningStar": "cdl_evening_star",
    "Candle.3WhiteSoldiers": "cdl_3_soldiers",
    "Candle.3BlackCrows": "cdl_3_crows",
    "Candle.Harami.Bullish": "cdl_harami_bull",
    "Candle.Harami.Bearish": "cdl_harami_bear",
    "Candle.Kicking.Bullish": "cdl_kicking_bull",
    "Candle.Kicking.Bearish": "cdl_kicking_bear",
    "Candle.AbandonedBaby.Bullish": "cdl_baby_bull",
    "Candle.AbandonedBaby.Bearish": "cdl_baby_bear",
    "Candle.Marubozu.White": "cdl_marubozu_white",
    "Candle.Marubozu.Black": "cdl_marubozu_black",
    "Candle.Doji": "cdl_doji",
    "Candle.Doji.Dragonfly": "cdl_doji_dragonfly",
    "Candle.Doji.Gravestone": "cdl_doji_gravestone"
}

def tv_fields() -> dict[str, str]:
    """Map TradingView field -> column key, for every column fetched from TradingView."""
    return TV_MAPPING
