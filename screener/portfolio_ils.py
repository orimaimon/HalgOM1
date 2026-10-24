"""Shekel P&L calculation module for portfolios.

Provides position valuation in shekels (ILS), splitting P&L into stock price movement
and USD/ILS currency exchange movement, as well as realized shekel P&L per sell trade.
Pure calculation module: accepts rates as an argument and performs no network/I/O.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from screener.data.boi import rate_on
from screener.portfolio import BUY, SELL, PList, sort_trades

if TYPE_CHECKING:
    pass


def ils_positions(
    plist: PList,
    prices: pd.Series,
    rates: pd.Series,
    today: dt.date,
) -> pd.DataFrame:
    """Calculate open position metrics in ILS, splitting P&L into stock and currency components.

    Columns returned:
      ticker, quantity, cost_usd, value_usd, cost_ils, value_ils, pnl_ils,
      pnl_ils_pct, stock_part_ils, fx_part_ils

    Positions where a trade date or today has no USD/ILS exchange rate have NaN in ILS columns
    and their tickers are listed in df.attrs["no_rate"].
    """
    trades = plist.frame()
    cols = [
        "ticker", "quantity", "cost_usd", "value_usd", "cost_ils",
        "value_ils", "pnl_ils", "pnl_ils_pct", "stock_part_ils", "fx_part_ils"
    ]
    if trades.empty:
        df_out = pd.DataFrame(columns=cols)
        df_out.attrs["no_rate"] = []
        return df_out

    sorted_trades = sort_trades(trades).assign(fees=lambda d: d["fees"].fillna(0.0))
    today_rate = rate_on(today, rates)

    rows = []
    no_rate_tickers = []

    for t, g in sorted_trades.groupby("ticker", sort=False):
        qty = 0.0
        cost_usd = 0.0
        cost_ils = 0.0
        missing_rate = False

        for _, tr in g.iterrows():
            rate = rate_on(tr["date"], rates)
            if tr["side"] == BUY:
                spent_usd = tr["price"] * tr["quantity"] + tr["fees"]
                if rate is None:
                    missing_rate = True
                    spent_ils = np.nan
                else:
                    spent_ils = spent_usd * rate

                qty += tr["quantity"]
                cost_usd += spent_usd
                if not np.isnan(spent_ils) and not np.isnan(cost_ils):
                    cost_ils += spent_ils
                else:
                    cost_ils = np.nan
            else:  # SELL
                part = min(tr["quantity"], qty)
                if qty > 0:
                    avg_cost_usd = cost_usd / qty
                    avg_cost_ils = cost_ils / qty if not np.isnan(cost_ils) else np.nan

                    qty -= part
                    cost_usd -= avg_cost_usd * part
                    if not np.isnan(avg_cost_ils):
                        cost_ils -= avg_cost_ils * part
                    else:
                        cost_ils = np.nan

                if abs(qty) < 1e-9:
                    qty = 0.0
                    cost_usd = 0.0
                    cost_ils = 0.0
                    missing_rate = False

        if qty > 0:
            px = float(prices.get(t, np.nan)) if prices is not None else np.nan
            value_usd = px * qty if pd.notna(px) else np.nan

            if missing_rate or today_rate is None or np.isnan(cost_ils):
                no_rate_tickers.append(t)
                c_ils = np.nan
                v_ils = np.nan
                p_ils = np.nan
                p_ils_pct = np.nan
                s_part = np.nan
                f_part = np.nan
            else:
                c_ils = cost_ils
                v_ils = value_usd * today_rate if pd.notna(value_usd) else np.nan
                p_ils = v_ils - c_ils if pd.notna(v_ils) else np.nan
                p_ils_pct = (p_ils / c_ils * 100) if c_ils > 0 and pd.notna(p_ils) else np.nan

                avg_buy_rate = c_ils / cost_usd if cost_usd > 0 else np.nan
                if pd.notna(value_usd) and pd.notna(avg_buy_rate):
                    s_part = (value_usd - cost_usd) * avg_buy_rate
                    f_part = value_usd * (today_rate - avg_buy_rate)
                else:
                    s_part = np.nan
                    f_part = np.nan

            rows.append({
                "ticker": t,
                "quantity": qty,
                "cost_usd": cost_usd,
                "value_usd": value_usd,
                "cost_ils": c_ils,
                "value_ils": v_ils,
                "pnl_ils": p_ils,
                "pnl_ils_pct": p_ils_pct,
                "stock_part_ils": s_part,
                "fx_part_ils": f_part,
            })

    df_out = pd.DataFrame(rows, columns=cols)
    df_out.attrs["no_rate"] = no_rate_tickers
    return df_out


def realized_ils(plist: PList, rates: pd.Series) -> pd.DataFrame:
    """Calculate realized shekel P&L per sell trade using trade-date exchange rates and average ILS cost.

    Note: Groundwork for a future tax summary. This is NOT an official tax calculation.

    Columns returned:
      date, ticker, quantity, proceeds_ils, cost_ils, pnl_ils
    """
    trades = plist.frame()
    cols = ["date", "ticker", "quantity", "proceeds_ils", "cost_ils", "pnl_ils"]
    if trades.empty:
        return pd.DataFrame(columns=cols)

    sorted_trades = sort_trades(trades).assign(fees=lambda d: d["fees"].fillna(0.0))
    sell_rows = []

    for t, g in sorted_trades.groupby("ticker", sort=False):
        qty = 0.0
        cost_ils = 0.0

        for _, tr in g.iterrows():
            rate = rate_on(tr["date"], rates)
            if tr["side"] == BUY:
                spent_usd = tr["price"] * tr["quantity"] + tr["fees"]
                spent_ils = spent_usd * rate if rate is not None else np.nan
                qty += tr["quantity"]
                if pd.notna(spent_ils) and pd.notna(cost_ils):
                    cost_ils += spent_ils
                else:
                    cost_ils = np.nan
            else:  # SELL
                part = min(tr["quantity"], qty)
                avg_cost_ils = cost_ils / qty if qty > 0 and pd.notna(cost_ils) else np.nan
                c_ils_sold = avg_cost_ils * part if pd.notna(avg_cost_ils) else np.nan

                proceeds_usd = tr["price"] * part - tr["fees"]
                proceeds_ils = proceeds_usd * rate if rate is not None else np.nan
                pnl_ils = proceeds_ils - c_ils_sold if pd.notna(proceeds_ils) and pd.notna(c_ils_sold) else np.nan

                sell_rows.append({
                    "date": tr["date"],
                    "ticker": t,
                    "quantity": part,
                    "proceeds_ils": proceeds_ils,
                    "cost_ils": c_ils_sold,
                    "pnl_ils": pnl_ils,
                })

                if qty > 0:
                    qty -= part
                    if pd.notna(cost_ils) and pd.notna(c_ils_sold):
                        cost_ils -= c_ils_sold
                    else:
                        cost_ils = np.nan

                if abs(qty) < 1e-9:
                    qty = 0.0
                    cost_ils = 0.0

    return pd.DataFrame(sell_rows, columns=cols)
