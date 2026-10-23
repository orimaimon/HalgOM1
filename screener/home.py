"""
"הבוקר שלי" (My Morning) home page data module.

Provides daily digest data for the home page:
- new_in_screens: matches for saved and built-in screens today vs previous snapshot day.
- movers: top 5 portfolio/watchlist stocks with biggest daily absolute price change.
- earnings: portfolio-held positions reporting earnings in the next 14 days.
- alerts: fired signals/alerts passed through.

No Streamlit imports and no network requests.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from screener import filters, history, portfolio, track
from screener.data.store import Store
from screener.portfolio import PORTFOLIO, PList

log = logging.getLogger(__name__)


@dataclass
class Morning:
    new_in_screens: pd.DataFrame
    movers: pd.DataFrame
    earnings: pd.DataFrame
    alerts: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def morning(
    universe: pd.DataFrame,
    store: Store,
    today: dt.date,
    lists: dict[str, PList],
    screens: dict[str, dict[str, str]] | None = None,
    fired: list | None = None,
    junk: bool = True,
) -> Morning:
    """Compute the daily home page digest data."""
    errors: list[str] = []

    # 1. new_in_screens
    new_in_screens_df = pd.DataFrame(columns=["screen", "total", "new"])
    try:
        if screens is None:
            screens = track.all_screens()

        prev = history.load_previous(store, today)

        rows = []
        for name, selections in screens.items():
            today_matches = filters.screen(universe, selections, junk=junk)
            today_tickers = set(today_matches["ticker"].dropna().astype(str)) if "ticker" in today_matches.columns else set()
            total_cnt = len(today_tickers)

            if prev is None:
                new_cnt = None
            else:
                prev_tickers = history.previous_tickers(prev, selections, junk=junk)
                new_cnt = len(today_tickers - prev_tickers)

            rows.append({"screen": name, "total": total_cnt, "new": new_cnt})

        saved_rows = [r for r in rows if r["screen"].startswith("★ ")]
        builtin_rows = [r for r in rows if not r["screen"].startswith("★ ")]

        if prev is not None:
            saved_rows.sort(key=lambda r: (r["new"] if r["new"] is not None else -1), reverse=True)
            builtin_rows.sort(key=lambda r: (r["new"] if r["new"] is not None else -1), reverse=True)

        sorted_rows = saved_rows + builtin_rows
        new_in_screens_df = pd.DataFrame(sorted_rows, columns=["screen", "total", "new"])
    except Exception as e:
        log.exception("Error computing new_in_screens: %s", e)
        errors.append(f"new_in_screens: {e}")

    # 2. movers
    movers_df = pd.DataFrame(columns=["ticker", "company", "price", "change", "lists"])
    try:
        ticker_to_lists: dict[str, list[str]] = {}
        for list_name, plist in lists.items():
            frame = plist.frame()
            if not frame.empty and "ticker" in frame.columns:
                for t in frame["ticker"].dropna().astype(str).unique():
                    ticker_to_lists.setdefault(t, []).append(list_name)

        if ticker_to_lists and not universe.empty and "ticker" in universe.columns:
            univ_indexed = universe.set_index("ticker")
            candidates = []
            for t, list_names in ticker_to_lists.items():
                if t in univ_indexed.index:
                    row = univ_indexed.loc[t]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    chg = row.get("change")
                    if pd.notna(chg):
                        abs_chg = abs(float(chg))
                        candidates.append({
                            "ticker": t,
                            "company": row.get("company", ""),
                            "price": row.get("price"),
                            "change": float(chg),
                            "lists": list_names,
                            "_abs_change": abs_chg,
                        })

            candidates.sort(key=lambda x: x["_abs_change"], reverse=True)
            top_5 = candidates[:5]
            for c in top_5:
                c.pop("_abs_change", None)
            movers_df = pd.DataFrame(top_5, columns=["ticker", "company", "price", "change", "lists"])
    except Exception as e:
        log.exception("Error computing movers: %s", e)
        errors.append(f"movers: {e}")

    # 3. earnings
    earnings_df = pd.DataFrame(columns=["ticker", "company", "earnings_date", "days"])
    try:
        held_tickers = set()
        for plist in lists.values():
            if plist.kind == PORTFOLIO:
                frame = plist.frame()
                if not frame.empty:
                    pos = portfolio.positions(frame)
                    if not pos.empty and "quantity" in pos.columns:
                        held = pos[pos["quantity"] > 0]["ticker"].dropna().unique()
                        held_tickers.update(held)

        if held_tickers and not universe.empty and "ticker" in universe.columns:
            univ_indexed = universe.set_index("ticker")
            earn_rows = []
            today_ts = pd.Timestamp(today)
            for t in held_tickers:
                if t in univ_indexed.index:
                    row = univ_indexed.loc[t]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    ed_raw = row.get("earnings_date")
                    if pd.notna(ed_raw):
                        ed_ts = pd.to_datetime(ed_raw, errors="coerce")
                        if pd.notna(ed_ts):
                            days_diff = (ed_ts.date() - today).days
                            if 0 <= days_diff <= 14:
                                earn_rows.append({
                                    "ticker": t,
                                    "company": row.get("company", ""),
                                    "earnings_date": ed_ts.date(),
                                    "days": int(days_diff),
                                })

            earn_rows.sort(key=lambda x: (x["earnings_date"], x["ticker"]))
            earnings_df = pd.DataFrame(earn_rows, columns=["ticker", "company", "earnings_date", "days"])
    except Exception as e:
        log.exception("Error computing earnings: %s", e)
        errors.append(f"earnings: {e}")

    # 4. alerts
    alerts_list = []
    try:
        alerts_list = list(fired) if fired is not None else []
    except Exception as e:
        log.exception("Error computing alerts: %s", e)
        errors.append(f"alerts: {e}")

    return Morning(
        new_in_screens=new_in_screens_df,
        movers=movers_df,
        earnings=earnings_df,
        alerts=alerts_list,
        errors=errors,
    )
