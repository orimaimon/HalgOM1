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
    known: dict[tuple[str, dt.date], float] = {}     # split_factor reads the ticker's parquet file each time

    for _, row in out.iterrows():
        ticker = str(row["ticker"]) if pd.notna(row.get("ticker")) else ""
        trade_date = _to_date(row.get("date"))
        if ticker and trade_date:
            if (ticker, trade_date) not in known:
                known[(ticker, trade_date)] = prices.split_factor(ticker, trade_date)
            f = known[(ticker, trade_date)]
        else:
            f = 1.0
        factors.append(f)

        qty = float(row["quantity"]) if pd.notna(row.get("quantity")) else 0.0
        price = float(row["price"]) if pd.notna(row.get("price")) else 0.0

        adj_quantities.append(qty * f)
        adj_prices.append(price / f if f != 0 else price)

    out["split_factor"] = factors
    out["quantity"] = adj_quantities
    out["price"] = adj_prices
    return out


def _trades(plist: PList) -> pd.DataFrame:
    """The portfolio's dated trades, split-adjusted, one row each: day (Timestamp), ticker, qty (signed:
    + buy, - sell) and cash (+ in for a buy incl. fees, - out for a sell net of fees)."""
    cols = ["day", "ticker", "qty", "cash"]
    if plist.kind == WATCHLIST:
        return pd.DataFrame(columns=cols)
    trades = plist.frame()
    if trades.empty:
        return pd.DataFrame(columns=cols)
    tr = sort_trades(adjusted_trades(trades))
    day = pd.to_datetime(tr["date"].map(_to_date), errors="coerce")
    qty = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(tr["price"], errors="coerce").fillna(0.0)
    fees = pd.to_numeric(tr["fees"], errors="coerce").fillna(0.0) if "fees" in tr else 0.0
    buy = tr["side"] == BUY
    out = pd.DataFrame({
        "day": day,
        "ticker": tr["ticker"].astype(str),
        "qty": qty.where(buy, -qty),
        "cash": (price * qty + fees).where(buy, -(price * qty - fees)),
    })
    return out[out["day"].notna()].reset_index(drop=True)


def _norm(hist: pd.DataFrame) -> pd.DataFrame:
    """History with a naive, day-normalized, unique DatetimeIndex."""
    if hist.empty:
        return hist
    h = hist.copy()
    idx = pd.to_datetime(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    return h[~h.index.duplicated(keep="last")].sort_index()


def _as_of(totals: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Amounts keyed by day -> their running total as of each of `days`. An amount on a non-trading day
    (a weekend trade) counts from the next day in `days`."""
    if totals.empty:
        return pd.Series(0.0, index=days)
    run = totals.groupby(level=0).sum().sort_index().cumsum()
    return run.reindex(run.index.union(days)).ffill().reindex(days).fillna(0.0)


def _column_on(hist: pd.DataFrame, col: str, days: pd.DatetimeIndex, limit: int | None = None) -> pd.Series:
    """`col` on each of `days`, carrying the last value forward (at most `limit` days when given)."""
    if hist.empty or col not in hist.columns:
        return pd.Series(np.nan, index=days)
    s = hist[col].dropna()
    return s.reindex(s.index.union(days)).ffill(limit=limit).reindex(days)


def dividends_received(plist: PList) -> pd.DataFrame:
    """Dividends received for a portfolio: date, ticker, per_share, shares, amount.

    From prices.history(ticker)'s dividends column (Yahoo dates a dividend by its ex-date): the shares held
    at the end of the day before the ex-date get it, so a buy on the ex-date does not and a sale on the
    ex-date still does. Yahoo's per-share dividends are split-adjusted, like the adjusted quantity.
    """
    cols = ["date", "ticker", "per_share", "shares", "amount"]
    tr = _trades(plist)
    records = []
    for ticker, t_tr in tr.groupby("ticker"):
        hist = _norm(prices.history(ticker))
        if hist.empty or "dividends" not in hist.columns:
            continue
        divs = hist.loc[hist["dividends"] > 0, "dividends"]
        if divs.empty:
            continue
        held = t_tr.groupby("day")["qty"].sum().sort_index().cumsum()
        for ex_day, per_share in divs.items():
            before = held[held.index < ex_day]
            shares = float(before.iloc[-1]) if not before.empty else 0.0
            if shares > 1e-9:
                records.append({"date": ex_day.date(), "ticker": ticker, "per_share": float(per_share),
                                "shares": shares, "amount": float(per_share) * shares})
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records, columns=cols).sort_values(["date", "ticker"]).reset_index(drop=True)


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


CARRY_DAYS = 5      # a held ticker with no bar on a day keeps its last close for at most this many days


def value_series(plist: PList, end: dt.date | None = None) -> pd.DataFrame:
    """One row per trading day (index: date) from the first trade to `end` (default: the last day with prices).

    value         holdings at the close: adjusted quantity x Yahoo's split-adjusted close
    net_invested  cash in (buys incl. fees) minus cash out (sells net of fees), cumulative
    pnl           value - net_invested
    dividends     cumulative dividends received (value + dividends is the total)
    spy, qqq      the same cash flows in the benchmark: each buy amount buys units at that day's adj_close,
                  each sell amount sells units worth the same; valued by adj_close (dividends included)

    A watchlist returns an empty frame. df.attrs["missing"]: held tickers with no price history (left out
    of `value`). Vectorized: a few years of daily rows for dozens of tickers take well under a second.
    """
    empty_cols = ["value", "net_invested", "pnl", "dividends", "spy", "qqq"]

    def empty(missing: list[str]) -> pd.DataFrame:
        res = pd.DataFrame(columns=empty_cols)
        res.attrs["missing"] = missing
        return res

    tr = _trades(plist)
    if tr.empty:
        return empty([])

    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for t in tr["ticker"].unique():
        h = _norm(prices.history(t))
        if h.empty:
            missing.append(t)
        else:
            histories[t] = h
    bench = {"spy": _norm(prices.history("SPY")), "qqq": _norm(prices.history("QQQ"))}

    all_days = pd.DatetimeIndex([])
    for h in [*histories.values(), *bench.values()]:
        if not h.empty:
            all_days = all_days.union(h.index)
    last = pd.Timestamp(end) if end is not None else (all_days.max() if len(all_days) else None)
    if last is None:
        return empty(missing)
    days = all_days[(all_days >= tr["day"].min()) & (all_days <= last)]
    if days.empty:
        return empty(missing)

    out = pd.DataFrame(index=days)
    out["net_invested"] = _as_of(tr.set_index("day")["cash"], days)
    value = pd.Series(0.0, index=days)
    for t, h in histories.items():
        held = _as_of(tr.loc[tr["ticker"] == t].set_index("day")["qty"], days)
        close = _column_on(h, "close", days, limit=CARRY_DAYS)
        value += (held.where(held > 1e-9, 0.0) * close).fillna(0.0)
    out["value"] = value
    out["pnl"] = out["value"] - out["net_invested"]

    div = dividends_received(plist)
    div_by_day = (div.assign(day=pd.to_datetime(div["date"])).set_index("day")["amount"]
                  if not div.empty else pd.Series(dtype=float))
    out["dividends"] = _as_of(div_by_day, days)

    for name, h in bench.items():
        if h.empty or "adj_close" not in h.columns:
            out[name] = np.nan
            continue
        adj = h["adj_close"].dropna()
        # The benchmark price of each trade: that day's adj_close, or the last one before it
        # (the first one when the trade is older than the benchmark's history).
        at_trade = adj.reindex(adj.index.union(tr["day"])).ffill().bfill().reindex(tr["day"]).to_numpy()
        units = pd.Series(np.where(at_trade > 0, tr["cash"].to_numpy() / at_trade, 0.0), index=tr["day"])
        out[name] = _as_of(units, days) * _column_on(h, "adj_close", days)

    out = out[empty_cols]
    out.index = pd.Index([d.date() for d in days], name="date")
    out.attrs["missing"] = missing
    return out
