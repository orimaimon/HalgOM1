"""
Trade journal and closed-trade P&L analytics (pure module).
Analyzes closed trades, win rate, payoff ratio (R:R), profit factor,
holding period, and realized P&L by sector.
"""
from __future__ import annotations

import datetime as dt
import logging
import numpy as np
import pandas as pd

from screener.portfolio import BUY, WATCHLIST, PList, sort_trades

log = logging.getLogger(__name__)

CLOSED_COLUMNS = [
    "ticker",
    "date",
    "quantity",
    "avg_cost",
    "sell_price",
    "fees",
    "cost_basis",
    "proceeds",
    "pnl",
    "pnl_pct",
    "hold_days",
    "sector",
    "note",
]


def closed_trades(plist: PList, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extract closed trade records from a portfolio using average-cost accounting.

    Returns a DataFrame with CLOSED_COLUMNS sorted by date descending.
    If plist is a WATCHLIST or has no trades, returns an empty DataFrame with CLOSED_COLUMNS.
    """
    if plist.kind == WATCHLIST or not plist.rows:
        return pd.DataFrame(columns=CLOSED_COLUMNS)

    df = plist.frame()
    if df.empty:
        return pd.DataFrame(columns=CLOSED_COLUMNS)

    # Sector mapping from universe
    sector_map = {}
    if universe is not None and not universe.empty and "ticker" in universe.columns and "sector" in universe.columns:
        for _, u_row in universe.drop_duplicates("ticker").iterrows():
            sec = u_row["sector"]
            if pd.notna(sec) and str(sec).strip():
                sector_map[str(u_row["ticker"])] = str(sec).strip()

    sorted_df = sort_trades(df)
    if "fees" in sorted_df.columns:
        sorted_df = sorted_df.assign(fees=lambda d: d["fees"].fillna(0.0))
    else:
        sorted_df = sorted_df.assign(fees=0.0)

    rows = []
    for t, g in sorted_df.groupby("ticker", sort=False):
        qty = cost = 0.0
        weighted_days = 0.0
        for _, tr in g.iterrows():
            tr_date = tr["date"]
            if isinstance(tr_date, (pd.Timestamp, dt.datetime)):
                tr_date = tr_date.date()

            fees = float(tr["fees"]) if pd.notna(tr.get("fees")) else 0.0
            price = float(tr["price"]) if pd.notna(tr.get("price")) else 0.0
            tr_qty = float(tr["quantity"]) if pd.notna(tr.get("quantity")) else 0.0
            side = str(tr.get("side", "")).lower()

            if side == BUY:
                spent = price * tr_qty + fees
                weighted_days += spent * tr_date.toordinal()
                qty += tr_qty
                cost += spent
            else:  # sell
                if qty > 1e-9:
                    avg = cost / qty
                    part = min(tr_qty, qty)
                    cost_basis = avg * part
                    proceeds = price * part - fees
                    pnl = proceeds - cost_basis
                    pnl_pct = (pnl / cost_basis * 100.0) if cost_basis > 0 else np.nan
                    avg_buy_ord = round(weighted_days / cost) if cost > 0 else tr_date.toordinal()
                    buy_date = dt.date.fromordinal(avg_buy_ord)
                    hold_days = max(0, (tr_date - buy_date).days)
                    note = str(tr["note"]) if pd.notna(tr.get("note")) and str(tr.get("note")).strip() else ""
                    sector = sector_map.get(t, "Other")

                    rows.append({
                        "ticker": t,
                        "date": tr_date,
                        "quantity": part,
                        "avg_cost": avg,
                        "sell_price": price,
                        "fees": fees,
                        "cost_basis": cost_basis,
                        "proceeds": proceeds,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "hold_days": hold_days,
                        "sector": sector,
                        "note": note,
                    })

                    weighted_days *= (qty - part) / qty
                    qty -= part
                    cost -= cost_basis

                if abs(qty) < 1e-9:
                    qty = cost = 0.0
                    weighted_days = 0.0

    if not rows:
        return pd.DataFrame(columns=CLOSED_COLUMNS)

    res = pd.DataFrame(rows, columns=CLOSED_COLUMNS)
    return res.sort_values("date", ascending=False).reset_index(drop=True)


def journal_summary(closed_df: pd.DataFrame) -> dict:
    """Compute summary statistics and Hebrew insights for closed trades.

    Returns dict with keys: trades, wins, losses, win_rate, total_pnl, avg_win_pct,
    avg_loss_pct, payoff_ratio, profit_factor, avg_hold_days, insights.
    """
    if closed_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "payoff_ratio": None,
            "profit_factor": None,
            "avg_hold_days": None,
            "insights": [],
        }

    wins_df = closed_df[closed_df["pnl"] > 0]
    losses_df = closed_df[closed_df["pnl"] < 0]

    trades = len(closed_df)
    wins = len(wins_df)
    losses = len(losses_df)
    win_rate = (wins / trades * 100.0) if trades > 0 else None
    total_pnl = float(closed_df["pnl"].sum())

    avg_win_pct = float(wins_df["pnl_pct"].mean()) if not wins_df.empty else None
    avg_loss_pct = float(losses_df["pnl_pct"].mean()) if not losses_df.empty else None

    payoff_ratio = (
        (avg_win_pct / abs(avg_loss_pct))
        if avg_win_pct is not None and avg_loss_pct is not None and abs(avg_loss_pct) > 1e-9
        else None
    )

    gross_profit = float(wins_df["pnl"].sum()) if not wins_df.empty else 0.0
    gross_loss = abs(float(losses_df["pnl"].sum())) if not losses_df.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else None

    avg_hold_days = float(closed_df["hold_days"].mean()) if trades > 0 else None

    insights = []
    if trades > 0 and win_rate is not None:
        insights.append(f"אחוז הצלחה של {round(win_rate)}% ב-{trades} עסקאות סגורות (רווח נטו ${total_pnl:+,.0f})")
    if payoff_ratio is not None and avg_win_pct is not None and avg_loss_pct is not None:
        insights.append(
            f"יחס רווח/הפסד (R:R) של {payoff_ratio:.2f} (רווח ממוצע {avg_win_pct:+.1f}%, הפסד ממוצע {avg_loss_pct:+.1f}%)"
        )
    elif avg_win_pct is not None:
        insights.append(f"רווח ממוצע לעסקה מרוויחה: {avg_win_pct:+.1f}%")
    elif avg_loss_pct is not None:
        insights.append(f"הפסד ממוצע לעסקה מפסידה: {avg_loss_pct:+.1f}%")

    if avg_hold_days is not None:
        insights.append(f"זמן החזקה ממוצע לעסקה: {round(avg_hold_days)} ימים")

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold_days,
        "insights": insights,
    }


def by_sector_pnl(closed_df: pd.DataFrame) -> pd.DataFrame:
    """Group closed trades by sector and return statistics sorted by P&L descending."""
    cols = ["sector", "trades", "win_rate", "pnl"]
    if closed_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for sector, g in closed_df.groupby("sector", sort=False):
        trades = len(g)
        wins = len(g[g["pnl"] > 0])
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        pnl = float(g["pnl"].sum())
        rows.append({
            "sector": sector,
            "trades": trades,
            "win_rate": win_rate,
            "pnl": pnl,
        })

    res = pd.DataFrame(rows, columns=cols)
    return res.sort_values("pnl", ascending=False).reset_index(drop=True)
