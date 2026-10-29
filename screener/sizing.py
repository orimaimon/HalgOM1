"""
ATR position sizer, stop-loss and target calculator, and portfolio trailing-stop monitoring.
Pure calculation module.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from screener.portfolio import WATCHLIST, PList, positions

STOP_COLUMNS = [
    "ticker",
    "company",
    "quantity",
    "avg_cost",
    "price",
    "atr",
    "ref_high",
    "stop_price",
    "distance_pct",
    "pnl_pct",
    "status",
]


def fixed_risk_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> dict[str, float]:
    """Calculate fixed dollar risk position size based on account equity and stop loss price."""
    inputs = [account_equity, risk_pct, entry_price, stop_loss_price]
    if any(v is None or pd.isna(v) for v in inputs):
        return {
            "risk_per_share": 0.0,
            "risk_usd": 0.0,
            "shares": 0.0,
            "position_cost": 0.0,
            "position_weight_pct": 0.0,
            "stop_loss_price": 0.0,
        }

    account_equity = float(account_equity)
    risk_pct = float(risk_pct)
    entry_price = float(entry_price)
    stop_loss_price = float(stop_loss_price)

    if account_equity <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_loss_price <= 0 or stop_loss_price >= entry_price:
        return {
            "risk_per_share": 0.0,
            "risk_usd": 0.0,
            "shares": 0.0,
            "position_cost": 0.0,
            "position_weight_pct": 0.0,
            "stop_loss_price": max(0.0, stop_loss_price),
        }

    risk_per_share = entry_price - stop_loss_price
    risk_usd = account_equity * (risk_pct / 100.0)
    shares = float(math.floor(risk_usd / risk_per_share))
    position_cost = shares * entry_price
    position_weight_pct = round((position_cost / account_equity) * 100.0, 6)

    return {
        "risk_per_share": float(risk_per_share),
        "risk_usd": float(risk_usd),
        "shares": float(shares),
        "position_cost": float(position_cost),
        "position_weight_pct": float(position_weight_pct),
        "stop_loss_price": float(stop_loss_price),
    }


def atr_volatility_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
) -> dict[str, float]:
    """Calculate position size based on trailing ATR stop distance."""
    if (
        entry_price is None
        or pd.isna(entry_price)
        or atr is None
        or pd.isna(atr)
        or float(atr) <= 0
        or atr_multiplier is None
        or pd.isna(atr_multiplier)
        or float(atr_multiplier) <= 0
    ):
        return {
            "risk_per_share": 0.0,
            "risk_usd": 0.0,
            "shares": 0.0,
            "position_cost": 0.0,
            "position_weight_pct": 0.0,
            "stop_loss_price": 0.0,
        }

    stop_loss_price = float(entry_price) - (float(atr) * float(atr_multiplier))
    return fixed_risk_size(account_equity, risk_pct, entry_price, stop_loss_price)


def position_plan(
    price: float,
    atr: float,
    account_size: float,
    risk_pct: float = 1.0,
    atr_mult: float = 2.0,
    target_r: float = 2.0,
) -> dict | None:
    """Calculate ATR-based position sizing, stop-loss, target, and Hebrew summary.

    Returns None if any input is missing, NaN, <= 0, or if atr * atr_mult >= price.
    """
    inputs = [price, atr, account_size, risk_pct, atr_mult, target_r]
    if any(v is None or pd.isna(v) or v <= 0 for v in inputs):
        return None

    price = float(price)
    atr = float(atr)
    account_size = float(account_size)
    risk_pct = float(risk_pct)
    atr_mult = float(atr_mult)
    target_r = float(target_r)

    stop_distance = atr * atr_mult
    if stop_distance >= price:
        return None

    risk_dollars = account_size * (risk_pct / 100.0)
    stop_price = price - stop_distance
    stop_pct = -(stop_distance / price * 100.0)

    raw_shares = int(math.floor(risk_dollars / stop_distance))
    max_shares = int(math.floor(account_size / price))
    shares = min(raw_shares, max_shares)

    position_value = shares * price
    position_weight_pct = (position_value / account_size) * 100.0
    actual_risk_dollars = shares * stop_distance
    target_price = price + stop_distance * target_r
    target_pct = (stop_distance * target_r / price) * 100.0
    reward_dollars = shares * stop_distance * target_r

    summary_he = (
        f"קנייה של {shares} מניות (${position_value:,.0f}) · "
        f"סטופ ב-${stop_price:.2f} ({stop_pct:.1f}%) · "
        f"סיכון ${actual_risk_dollars:,.0f}"
    )

    return {
        "shares": shares,
        "position_value": position_value,
        "position_weight_pct": position_weight_pct,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "target_price": target_price,
        "target_pct": target_pct,
        "risk_dollars": risk_dollars,
        "actual_risk_dollars": actual_risk_dollars,
        "reward_dollars": reward_dollars,
        "summary_he": summary_he,
    }


def trailing_stops(
    plist: PList,
    universe: pd.DataFrame,
    atr_mult: float = 2.0,
    warn_within_pct: float = 1.5,
) -> pd.DataFrame:
    """Calculate ATR-based trailing stops for open positions in a portfolio."""
    if plist.kind == WATCHLIST or not plist.rows:
        return pd.DataFrame(columns=STOP_COLUMNS)

    pos_df = positions(plist.frame())
    if pos_df.empty or "quantity" not in pos_df.columns:
        return pd.DataFrame(columns=STOP_COLUMNS)

    open_pos = pos_df[pos_df["quantity"] > 0].copy()
    if open_pos.empty:
        return pd.DataFrame(columns=STOP_COLUMNS)

    uni_map = {}
    if not universe.empty and "ticker" in universe.columns:
        uni_dedup = universe.drop_duplicates("ticker")
        for _, urow in uni_dedup.iterrows():
            uni_map[urow["ticker"]] = urow

    rows = []
    for _, prow in open_pos.iterrows():
        ticker = prow["ticker"]
        quantity = prow["quantity"]
        avg_cost = prow["avg_cost"]

        urow = uni_map.get(ticker)
        company = urow["company"] if urow is not None and "company" in urow and pd.notna(urow["company"]) else ""
        price = float(urow["price"]) if urow is not None and "price" in urow and pd.notna(urow["price"]) else np.nan
        atr = float(urow["atr"]) if urow is not None and "atr" in urow and pd.notna(urow["atr"]) else np.nan
        high_1m = float(urow["high_1m"]) if urow is not None and "high_1m" in urow and pd.notna(urow["high_1m"]) else np.nan

        if pd.notna(price):
            cand = [v for v in (avg_cost, price, high_1m) if pd.notna(v)]
            ref_high = float(max(cand)) if cand else np.nan
        else:
            ref_high = np.nan

        if pd.notna(price) and price > 0 and pd.notna(atr) and atr > 0 and atr_mult > 0:
            stop_price = ref_high - atr_mult * atr
            distance_pct = (price - stop_price) / price * 100.0
            if price <= stop_price:
                status = "hit"
            elif 0 < distance_pct <= warn_within_pct:
                status = "near"
            else:
                status = "ok"
        else:
            stop_price = np.nan
            distance_pct = np.nan
            status = "no_data"

        if pd.notna(price) and price > 0 and pd.notna(avg_cost) and avg_cost > 0:
            pnl_pct = (price - avg_cost) / avg_cost * 100.0
        else:
            pnl_pct = np.nan

        rows.append({
            "ticker": ticker,
            "company": company,
            "quantity": quantity,
            "avg_cost": avg_cost,
            "price": price,
            "atr": atr,
            "ref_high": ref_high,
            "stop_price": stop_price,
            "distance_pct": distance_pct,
            "pnl_pct": pnl_pct,
            "status": status,
        })

    df = pd.DataFrame(rows, columns=STOP_COLUMNS)
    status_order = {"hit": 0, "near": 1, "ok": 2, "no_data": 3}
    df["_rank"] = df["status"].map(lambda s: status_order.get(s, 4))
    df = df.sort_values(by=["_rank", "distance_pct"], na_position="last").drop(columns=["_rank"]).reset_index(drop=True)
    return df


def stop_warnings(stops_df: pd.DataFrame) -> list[str]:
    """Return Hebrew warning strings for 'hit' and 'near' trailing stop rows."""
    if stops_df.empty or "status" not in stops_df.columns:
        return []

    warnings = []
    for _, row in stops_df.iterrows():
        st = row["status"]
        ticker = row["ticker"]
        price = row["price"]
        stop_price = row["stop_price"]
        distance_pct = row["distance_pct"]

        if st == "hit":
            warnings.append(f"{ticker}: המחיר (${price:.2f}) חצה את הסטופ הנגרר (${stop_price:.2f})")
        elif st == "near":
            warnings.append(f"{ticker}: קרובה לסטופ הנגרר (${stop_price:.2f}, מרחק {distance_pct:.1f}%)")

    return warnings
