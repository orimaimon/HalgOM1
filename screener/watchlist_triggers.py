"""Smart Watchlist Triggers & Playbook engine.

Manages target entry prices and trader notes on watchlist items, evaluating
breakout ("above") or dip-buy ("below") triggers, near-target proximity,
and portfolio holding status.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from screener import portfolio
from screener.data import store
from screener.portfolio import WATCHLIST, PList
from screener.userfiles import load_user_file, save_user_file

log = logging.getLogger(__name__)

WATCH_TRIGGER_COLUMNS = [
    "ticker",
    "company",
    "current",
    "change",
    "target_price",
    "direction",
    "distance_pct",
    "status",
    "in_portfolio",
    "note",
]


@dataclass
class WatchTarget:
    key: str = ""
    list_name: str = ""
    ticker: str = ""
    target_price: float | None = None
    direction: str = "above"  # "above" | "below"


def targets_path() -> Path:
    """Returns the path to watchlist_targets.json, resolved at call time."""
    return store.DATA_DIR / "watchlist_targets.json"


def load_targets(path: Path | None = None) -> dict[str, WatchTarget]:
    """Load all saved watch targets."""
    p = path or targets_path()
    try:
        return load_user_file(p, WatchTarget)
    except OSError as e:
        log.warning("cannot read watchlist targets: %s", e)
        return {}


def set_target(
    list_name: str,
    ticker: str,
    target_price: float | None,
    direction: str = "above",
    path: Path | None = None,
) -> None:
    """Set or update a watch target for a (list_name, ticker) pair."""
    p = path or targets_path()
    key = f"{list_name}:{ticker}"

    def modify(items: dict[str, WatchTarget]) -> None:
        if target_price is None:
            items.pop(key, None)
        else:
            items[key] = WatchTarget(
                key=key,
                list_name=list_name,
                ticker=ticker,
                target_price=float(target_price),
                direction=direction if direction in ("above", "below") else "above",
            )

    save_user_file(p, WatchTarget, modify)


def update_note(
    plist: PList,
    ticker: str,
    note: str,
    portfolios_path: Path | None = None,
) -> bool:
    """Update the trader note for a ticker in a watchlist and save it."""
    found = False
    for r in plist.rows:
        if r.get("ticker") == ticker:
            r["note"] = note
            found = True

    if found:
        portfolio.save(plist, path=portfolios_path)
    return found


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None", "NaT")


def evaluate_watchlist(
    plist: PList,
    universe: pd.DataFrame,
    all_lists: dict[str, PList] | None = None,
    path: Path | None = None,
    near_pct: float = 2.0,
) -> pd.DataFrame:
    """Evaluate trigger status, near proximity, portfolio overlap, and notes for a watchlist."""
    if plist.kind != WATCHLIST or not plist.rows:
        return pd.DataFrame(columns=WATCH_TRIGGER_COLUMNS)

    if all_lists is None:
        all_lists = portfolio.load_all()

    # Determine held tickers across all portfolios
    held_tickers: set[str] = set()
    for l in all_lists.values():
        if l.kind == portfolio.PORTFOLIO and l.rows:
            trades_df = l.frame()
            if not trades_df.empty:
                pos = portfolio.positions(trades_df)
                if not pos.empty and "quantity" in pos.columns:
                    held_tickers.update(pos.loc[pos["quantity"] > 0, "ticker"].tolist())

    targets = load_targets(path)

    # Fast lookups from universe
    prices = (
        universe.set_index("ticker")["price"]
        if not universe.empty and "ticker" in universe.columns and "price" in universe.columns
        else pd.Series(dtype=float)
    )
    changes = (
        universe.set_index("ticker")["change"]
        if not universe.empty and "ticker" in universe.columns and "change" in universe.columns
        else pd.Series(dtype=float)
    )
    companies = (
        universe.set_index("ticker")["company"]
        if not universe.empty and "ticker" in universe.columns and "company" in universe.columns
        else pd.Series(dtype=str)
    )

    rows = []
    for r in plist.rows:
        ticker = str(r.get("ticker", "")).strip()
        if not ticker:
            continue

        company = str(companies.get(ticker, "")) if ticker in companies and not _blank(companies.get(ticker)) else ""
        current_val = prices.get(ticker) if ticker in prices else np.nan
        current = float(current_val) if pd.notna(current_val) else np.nan

        change_val = changes.get(ticker) if ticker in changes else np.nan
        change = float(change_val) if pd.notna(change_val) else np.nan

        key = f"{plist.name}:{ticker}"
        target_obj = targets.get(key)

        target_price = (
            float(target_obj.target_price)
            if target_obj and target_obj.target_price is not None and pd.notna(target_obj.target_price)
            else None
        )
        direction = target_obj.direction if target_obj and target_obj.direction in ("above", "below") else "above"

        in_portfolio = bool(ticker in held_tickers)
        note = str(r.get("note", "")) if not _blank(r.get("note")) else ""

        # Calculate trigger status and distance_pct
        if target_price is None:
            status = "no_target"
            distance_pct = np.nan
        elif pd.isna(current) or current <= 0:
            status = "waiting"
            distance_pct = np.nan
        elif direction == "above":
            dist = (target_price - current) / current * 100.0
            if current >= target_price:
                status = "triggered"
            elif 0 < dist <= near_pct:
                status = "near"
            else:
                status = "waiting"
            distance_pct = dist
        else:  # direction == "below"
            dist = (current - target_price) / current * 100.0
            if current <= target_price:
                status = "triggered"
            elif 0 < dist <= near_pct:
                status = "near"
            else:
                status = "waiting"
            distance_pct = dist

        rows.append({
            "ticker": ticker,
            "company": company,
            "current": current,
            "change": change,
            "target_price": target_price,
            "direction": direction,
            "distance_pct": distance_pct,
            "status": status,
            "in_portfolio": in_portfolio,
            "note": note,
        })

    df = pd.DataFrame(rows, columns=WATCH_TRIGGER_COLUMNS)
    if df.empty:
        return df

    # Sort "triggered" rows first, then "near", then "waiting", then "no_target"
    status_rank = {"triggered": 0, "near": 1, "waiting": 2, "no_target": 3}
    df["_rank"] = df["status"].map(lambda s: status_rank.get(s, 4))
    df = df.sort_values("_rank", kind="stable").drop(columns=["_rank"]).reset_index(drop=True)
    return df
