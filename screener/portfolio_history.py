"""
Portfolio performance over time, dividends, and split adjustments against benchmark indices (SPY, QQQ).

Compares portfolio market value, net invested cash, P&L, and dividend income against holding SPY/QQQ
with the same cash flows. Built on top of screener/portfolio.py and screener/data/prices.py.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from screener.data import prices
from screener.portfolio import BUY, PORTFOLIO, WATCHLIST, PList, sort_trades

log = logging.getLogger(__name__)


def _to_date(v: Any) -> dt.date | None:
    if v is None or pd.isna(v):
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, str):
        try:
            return dt.date.fromisoformat(v.strip())
        except ValueError:
            return None
    return None


def tickers_to_track(lists: dict[str, PList]) -> list[str]:
    """Every ticker in every portfolio and watchlist, plus "SPY" and "QQQ"; no duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for plist in lists.values():
        df = plist.frame()
        if not df.empty and "ticker" in df.columns:
            for t in df["ticker"].dropna():
                ticker_str = str(t).strip()
                if ticker_str and ticker_str not in seen:
                    seen.add(ticker_str)
                    out.append(ticker_str)
    for bench in ("SPY", "QQQ"):
        if bench not in seen:
            seen.add(bench)
            out.append(bench)
    return out


def update_prices(lists: dict[str, PList], today: dt.date | None = None) -> tuple[list[str], list[str]]:
    """Bring prices up to date for all tickers in lists plus benchmarks.

    Start date is the earliest trade or "added" date minus 7 days (one year back if none).
    Returns (updated, failed) and never raises.
    """
    today = today or dt.date.today()
    tickers = tickers_to_track(lists)

    all_dates: list[dt.date] = []
    for plist in lists.values():
        df = plist.frame()
        if not df.empty and "date" in df.columns:
            for d in df["date"].dropna():
                parsed = _to_date(d)
                if parsed:
                    all_dates.append(parsed)

    if all_dates:
        start = min(all_dates) - dt.timedelta(days=7)
    else:
        start = today - dt.timedelta(days=365)

    try:
        return prices.ensure(tickers, start)
    except Exception as e:
        log.warning("update_prices failed: %s", e)
        return [], tickers


def adjusted_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Adjust trade quantity (*= f) and price (/= f) by split_factor(ticker, trade_date).

    The original columns stay; split_factor column is added.
    """
    if trades.empty:
        out = trades.copy()
        out["split_factor"] = pd.Series(dtype=float)
        return out

    out = trades.copy()
    factors = []
    adj_quantities = []
    adj_prices = []

    for _, row in out.iterrows():
        ticker = str(row["ticker"]) if pd.notna(row.get("ticker")) else ""
        trade_date = _to_date(row.get("date"))
        f = prices.split_factor(ticker, trade_date) if ticker and trade_date else 1.0
        factors.append(f)

        qty = float(row["quantity"]) if pd.notna(row.get("quantity")) else 0.0
        price = float(row["price"]) if pd.notna(row.get("price")) else 0.0

        adj_quantities.append(qty * f)
        adj_prices.append(price / f if f != 0 else price)

    out["split_factor"] = factors
    out["quantity"] = adj_quantities
    out["price"] = adj_prices
    return out


def dividends_received(plist: PList) -> pd.DataFrame:
    """Dividends received for a portfolio: date, ticker, per_share, shares, amount.

    Calculated from prices.history(ticker)'s dividends column on days the adjusted position > 0.
    """
    cols = ["date", "ticker", "per_share", "shares", "amount"]
    if plist.kind == WATCHLIST:
        return pd.DataFrame(columns=cols)

    trades = plist.frame()
    if trades.empty:
        return pd.DataFrame(columns=cols)

    adj = adjusted_trades(trades)
    sorted_df = sort_trades(adj)

    records = []
    for ticker in sorted_df["ticker"].unique():
        t_str = str(ticker)
        hist = prices.history(t_str)
        if hist.empty or "dividends" not in hist.columns:
            continue

        div_rows = hist[hist["dividends"] > 0]
        if div_rows.empty:
            continue

        t_trades = sorted_df[sorted_df["ticker"] == ticker].copy()
        t_trades["parsed_date"] = t_trades["date"].map(_to_date)

        for idx, row in div_rows.iterrows():
            div_date = _to_date(idx)
            if not div_date:
                continue

            # Position up to and including div_date
            past_trades = t_trades[t_trades["parsed_date"] <= div_date]
            pos = 0.0
            for _, tr in past_trades.iterrows():
                q = float(tr["quantity"]) if pd.notna(tr.get("quantity")) else 0.0
                if tr.get("side") == BUY:
                    pos += q
                else:
                    pos -= q

            if pos > 1e-9:
                per_share = float(row["dividends"])
                shares = float(pos)
                amount = per_share * shares
                records.append({
                    "date": div_date,
                    "ticker": t_str,
                    "per_share": per_share,
                    "shares": shares,
                    "amount": amount,
                })

    if not records:
        return pd.DataFrame(columns=cols)

    out_df = pd.DataFrame(records, columns=cols)
    return out_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def latest_prices(tickers: Iterable[str], universe_prices: pd.Series | None = None) -> pd.Series:
    """The universe price when available, otherwise the last stored close from prices.history().

    Never downloads; NaN when there is nothing.
    """
    res = {}
    for t in tickers:
        t_str = str(t)
        price = np.nan
        if universe_prices is not None and t_str in universe_prices and pd.notna(universe_prices[t_str]):
            price = float(universe_prices[t_str])
        else:
            hist = prices.history(t_str)
            if not hist.empty and "close" in hist.columns:
                valid_closes = hist["close"].dropna()
                if not valid_closes.empty:
                    price = float(valid_closes.iloc[-1])
        res[t_str] = price
    return pd.Series(res, dtype=float)


def _benchmark_adj_close(hist: pd.DataFrame, target_date: dt.date) -> float | None:
    """Get benchmark's adj_close on target_date, or closest date on or before it."""
    if hist.empty or "adj_close" not in hist.columns:
        return None
    valid = hist["adj_close"].dropna()
    if valid.empty:
        return None

    ts = pd.Timestamp(target_date)
    past = valid[valid.index <= ts]
    if not past.empty:
        return float(past.iloc[-1])
    return float(valid.iloc[0])


def value_series(plist: PList, end: dt.date | None = None) -> pd.DataFrame:
    """One row per trading day from the first trade to end.

    A watchlist returns an empty frame.
    Attributes:
      df.attrs["missing"]: list of tickers with no price history.
    """
    empty_cols = ["value", "net_invested", "pnl", "dividends", "spy", "qqq"]
    if plist.kind == WATCHLIST:
        res = pd.DataFrame(columns=empty_cols)
        res.attrs["missing"] = []
        return res

    trades = plist.frame()
    if trades.empty:
        res = pd.DataFrame(columns=empty_cols)
        res.attrs["missing"] = []
        return res

    adj_tr = adjusted_trades(trades)
    sorted_tr = sort_trades(adj_tr)
    sorted_tr["parsed_date"] = sorted_tr["date"].map(_to_date)
    valid_tr = sorted_tr[sorted_tr["parsed_date"].notna()].copy()

    if valid_tr.empty:
        res = pd.DataFrame(columns=empty_cols)
        res.attrs["missing"] = []
        return res

    first_trade_date = valid_tr["parsed_date"].min()
    held_tickers = list(valid_tr["ticker"].unique())

    histories: dict[str, pd.DataFrame] = {}
    missing_tickers: list[str] = []
    for t in held_tickers:
        h = prices.history(str(t))
        if h.empty:
            missing_tickers.append(str(t))
        else:
            histories[str(t)] = h

    spy_hist = prices.history("SPY")
    qqq_hist = prices.history("QQQ")

    # Collect trading dates
    all_trading_dates: set[dt.date] = set()
    for h in list(histories.values()) + [spy_hist, qqq_hist]:
        if not h.empty:
            for idx in h.index:
                d = _to_date(idx)
                if d:
                    all_trading_dates.add(d)

    filtered_dates = [d for d in all_trading_dates if d >= first_trade_date]
    if end is not None:
        filtered_dates = [d for d in filtered_dates if d <= end]
    else:
        if all_trading_dates:
            max_hist_date = max(all_trading_dates)
            filtered_dates = [d for d in filtered_dates if d <= max_hist_date]

    trading_days = sorted(filtered_dates)
    if not trading_days:
        res = pd.DataFrame(columns=empty_cols)
        res.attrs["missing"] = missing_tickers
        return res

    # Build daily price lookup for each held ticker (with at most 5-day carry forward)
    price_on_day: dict[str, dict[dt.date, float]] = {}
    for t, h in histories.items():
        price_on_day[t] = {}
        date_to_close = {
            _to_date(idx): float(row["close"])
            for idx, row in h.iterrows()
            if pd.notna(row.get("close"))
        }
        last_p = None
        missing_count = 0
        for d in trading_days:
            if d in date_to_close:
                last_p = date_to_close[d]
                missing_count = 0
                price_on_day[t][d] = last_p
            else:
                missing_count += 1
                if last_p is not None and missing_count <= 5:
                    price_on_day[t][d] = last_p
                else:
                    price_on_day[t][d] = np.nan

    # Benchmark adj_close lookup for trading_days
    bench_price_on_day: dict[str, dict[dt.date, float]] = {"SPY": {}, "QQQ": {}}
    for bench_symbol, b_hist in (("SPY", spy_hist), ("QQQ", qqq_hist)):
        date_to_adj = {
            _to_date(idx): float(row["adj_close"])
            for idx, row in b_hist.iterrows()
            if pd.notna(row.get("adj_close"))
        }
        last_p = None
        for d in trading_days:
            if d in date_to_adj:
                last_p = date_to_adj[d]
                bench_price_on_day[bench_symbol][d] = last_p
            else:
                bench_price_on_day[bench_symbol][d] = last_p if last_p is not None else np.nan

    # Benchmark cash flows & units
    bench_units: dict[str, list[tuple[dt.date, float]]] = {"SPY": [], "QQQ": []}
    for bench_symbol, b_hist in (("SPY", spy_hist), ("QQQ", qqq_hist)):
        for _, tr in valid_tr.iterrows():
            tr_date = tr["parsed_date"]
            price = float(tr["price"]) if pd.notna(tr.get("price")) else 0.0
            qty = float(tr["quantity"]) if pd.notna(tr.get("quantity")) else 0.0
            fees = float(tr["fees"]) if pd.notna(tr.get("fees")) else 0.0

            if tr.get("side") == BUY:
                cf = price * qty + fees
            else:
                cf = -(price * qty - fees)

            b_price = _benchmark_adj_close(b_hist, tr_date)
            if b_price and b_price > 0:
                units = cf / b_price
                bench_units[bench_symbol].append((tr_date, units))

    # Calculate cumulative dividends
    div_df = dividends_received(plist)

    rows = []
    for d in trading_days:
        past_trades = valid_tr[valid_tr["parsed_date"] <= d]

        # Net invested
        cash_in = 0.0
        cash_out = 0.0
        for _, tr in past_trades.iterrows():
            price = float(tr["price"]) if pd.notna(tr.get("price")) else 0.0
            qty = float(tr["quantity"]) if pd.notna(tr.get("quantity")) else 0.0
            fees = float(tr["fees"]) if pd.notna(tr.get("fees")) else 0.0
            if tr.get("side") == BUY:
                cash_in += price * qty + fees
            else:
                cash_out += price * qty - fees
        net_invested = cash_in - cash_out

        # Portfolio value
        port_value = 0.0
        for t, h in histories.items():
            t_trades = past_trades[past_trades["ticker"] == t]
            pos = 0.0
            for _, tr in t_trades.iterrows():
                q = float(tr["quantity"]) if pd.notna(tr.get("quantity")) else 0.0
                if tr.get("side") == BUY:
                    pos += q
                else:
                    pos -= q
            if pos > 1e-9:
                p = price_on_day[t].get(d)
                if pd.notna(p):
                    port_value += pos * p

        pnl = port_value - net_invested

        # Cumulative dividends
        div_cum = 0.0
        if not div_df.empty:
            div_cum = float(div_df[div_df["date"] <= d]["amount"].sum())

        # Benchmark values
        bench_vals = {}
        for bench_symbol in ("SPY", "QQQ"):
            b_units = sum(u for t_d, u in bench_units[bench_symbol] if t_d <= d)
            b_p = bench_price_on_day[bench_symbol].get(d)
            if pd.notna(b_p) and b_p is not None:
                bench_vals[bench_symbol] = b_units * b_p
            else:
                bench_vals[bench_symbol] = np.nan

        rows.append({
            "date": d,
            "value": port_value,
            "net_invested": net_invested,
            "pnl": pnl,
            "dividends": div_cum,
            "spy": bench_vals["SPY"],
            "qqq": bench_vals["QQQ"],
        })

    out_df = pd.DataFrame(rows).set_index("date")
    out_df.attrs["missing"] = missing_tickers
    return out_df
