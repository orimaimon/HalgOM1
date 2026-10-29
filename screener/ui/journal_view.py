"""Trade journal & performance analytics UI component.

Renders performance metrics, sector breakdown chart, and closed trades log
for closed positions in a portfolio.

Public entry point: render(plist: PList, universe: pd.DataFrame) -> None
"""
from __future__ import annotations

import datetime as dt
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from screener import i18n, portfolio, theme
from screener.portfolio import BUY, SELL, WATCHLIST, PList
from screener.ui.common import L, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _compute_closed_trades(plist: PList) -> list[dict]:
    """Calculate closed trades from portfolio trades using average-cost method."""
    df = plist.frame()
    if df.empty:
        return []

    sorted_df = portfolio.sort_trades(df)
    closed_trades = []

    # Track running inventory per ticker
    # ticker -> {"qty": float, "cost": float, "weighted_days": float}
    inventory: dict[str, dict] = {}

    for _, tr in sorted_df.iterrows():
        tkr = tr["ticker"]
        side = tr.get("side", BUY)
        date_val = tr.get("date")
        if not isinstance(date_val, dt.date):
            if isinstance(date_val, (dt.datetime, pd.Timestamp)):
                date_val = date_val.date()
            elif isinstance(date_val, str) and date_val:
                try:
                    date_val = dt.date.fromisoformat(date_val[:10])
                except ValueError:
                    date_val = dt.date.today()
            else:
                date_val = dt.date.today()

        price = float(tr.get("price", 0.0) or 0.0)
        quantity = float(tr.get("quantity", 0.0) or 0.0)
        fees = float(tr.get("fees", 0.0) or 0.0)
        note = str(tr.get("note", "") or "")

        inv = inventory.setdefault(tkr, {"qty": 0.0, "cost": 0.0, "weighted_days": 0.0})

        if side == BUY:
            spent = price * quantity + fees
            inv["weighted_days"] += spent * date_val.toordinal()
            inv["qty"] += quantity
            inv["cost"] += spent
        elif side == SELL:
            if inv["qty"] <= 0:
                continue

            avg_cost = inv["cost"] / inv["qty"] if inv["qty"] > 0 else 0.0
            part = min(quantity, inv["qty"])

            if inv["cost"] > 0:
                avg_buy_ordinal = inv["weighted_days"] / inv["cost"]
                avg_buy_date = dt.date.fromordinal(round(avg_buy_ordinal))
            else:
                avg_buy_date = date_val

            holding_days = max(0, (date_val - avg_buy_date).days)
            cost_sold = avg_cost * part
            proceeds = price * part - fees
            realized_pnl = proceeds - cost_sold
            pnl_pct = (realized_pnl / cost_sold * 100.0) if cost_sold > 0 else 0.0

            closed_trades.append({
                "ticker": tkr,
                "date": date_val,
                "quantity": part,
                "avg_cost": avg_cost,
                "sell_price": price,
                "realized_pnl": realized_pnl,
                "pnl_pct": pnl_pct,
                "holding_days": holding_days,
                "note": note,
            })

            # Reduce inventory
            ratio = (inv["qty"] - part) / inv["qty"] if inv["qty"] > 0 else 0.0
            inv["weighted_days"] *= ratio
            inv["cost"] -= avg_cost * part
            inv["qty"] -= part

            if abs(inv["qty"]) < 1e-9:
                inv["qty"] = 0.0
                inv["cost"] = 0.0
                inv["weighted_days"] = 0.0

    return closed_trades


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Render trade journal and performance analytics UI."""
    try:
        if plist.kind == WATCHLIST:
            return

        closed_trades = _compute_closed_trades(plist)
        if not closed_trades:
            hebrew_note(t("No closed trades yet"))
            return

        count = len(closed_trades)
        wins = [tr for tr in closed_trades if tr["realized_pnl"] > 0]
        losses = [tr for tr in closed_trades if tr["realized_pnl"] < 0]

        win_rate = (len(wins) / count * 100.0) if count > 0 else 0.0

        if wins and losses:
            total_win_pnl = sum(tr["realized_pnl"] for tr in wins)
            total_loss_pnl = sum(abs(tr["realized_pnl"]) for tr in losses)
            profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else None

            avg_win_pnl = total_win_pnl / len(wins)
            avg_loss_pnl = total_loss_pnl / len(losses)
            payoff_ratio = (avg_win_pnl / avg_loss_pnl) if avg_loss_pnl > 0 else None
        else:
            profit_factor = None
            payoff_ratio = None

        # Top row of 4 metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Closed trades"), str(count))
        c2.metric(t("Win rate"), f"{win_rate:.1f}%")
        c3.metric(t("Profit factor"), f"{profit_factor:.2f}" if profit_factor is not None else "—")
        c4.metric(t("Payoff ratio (R:R)"), f"{payoff_ratio:.2f}" if payoff_ratio is not None else "—")

        # Summary note via hebrew_note
        total_pnl = sum(tr["realized_pnl"] for tr in closed_trades)
        pnl_str = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"

        avg_win_pct = (sum(tr["pnl_pct"] for tr in wins) / len(wins)) if wins else 0.0
        avg_loss_pct = (sum(tr["pnl_pct"] for tr in losses) / len(losses)) if losses else 0.0
        avg_holding = sum(tr["holding_days"] for tr in closed_trades) / count

        win_str = f"+{avg_win_pct:.1f}%"
        loss_str = f"{avg_loss_pct:.1f}%"
        holding_str = f"{round(avg_holding)}"

        summary_msg = f"רווח/הפסד ממומש כולל: {pnl_str} · ממוצע רווח: {win_str} · ממוצע הפסד: {loss_str} · ממוצע ימי החזקה: {holding_str}"
        hebrew_note(summary_msg)

        # Sector mapping from universe
        ticker_to_sector = {}
        if not universe.empty and "ticker" in universe.columns and "sector" in universe.columns:
            ticker_to_sector = universe.set_index("ticker")["sector"].to_dict()

        sector_pnl: dict[str, float] = {}
        for tr in closed_trades:
            raw_sec = ticker_to_sector.get(tr["ticker"])
            sec_name = i18n.value("sector", raw_sec if pd.notna(raw_sec) and raw_sec else "Other", L())
            sector_pnl[sec_name] = sector_pnl.get(sec_name, 0.0) + tr["realized_pnl"]

        chart_df = pd.DataFrame([
            {"sector_name": sec, "realized_pnl": pnl_val}
            for sec, pnl_val in sector_pnl.items()
        ])

        if not chart_df.empty:
            chart_df = chart_df.sort_values("realized_pnl", ascending=True)
            colors = [theme.GAIN if v >= 0 else theme.LOSS for v in chart_df["realized_pnl"]]

            fig = px.bar(
                chart_df,
                x="realized_pnl",
                y="sector_name",
                orientation="h",
                text_auto=".2f",
            )
            fig.update_traces(
                marker_color=colors,
                textposition="auto",
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(180, len(chart_df) * 35 + 40),
            )
            st.plotly_chart(fig, use_container_width=True, key="jr_sector_chart")

        # Closed trades table
        table_rows = []
        for tr in closed_trades:
            d_str = tr["date"].strftime("%d/%m/%Y") if isinstance(tr["date"], dt.date) else str(tr["date"])
            table_rows.append({
                t("Ticker"): tr["ticker"],
                t("Date"): d_str,
                t("Quantity"): f"{tr['quantity']:g}",
                t("Avg cost"): f"${tr['avg_cost']:.2f}",
                t("Sell price"): f"${tr['sell_price']:.2f}",
                t("Realized P&L"): f"${tr['realized_pnl']:.2f}",
                t("P&L %"): f"{tr['pnl_pct']:+.2f}%",
                t("Holding days"): str(tr["holding_days"]),
                t("Notes"): tr["note"],
            })

        table_df = pd.DataFrame(table_rows)
        st.dataframe(table_df, hide_index=True, use_container_width=True, key="jr_trades_table")

    except Exception:
        log.exception("error rendering trade journal")
        hebrew_note(t("Error displaying trade journal"))
