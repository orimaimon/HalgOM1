"""
Alerts engine for watched stocks.

Checks alert conditions ("price under X", "fell 5% today", "crossed SMA200 today",
"earnings in 3 days") against the latest universe snapshot in memory.

Evaluations are pure and read-only. Persistent storage (alerts.json) is managed
via screener/userfiles.py.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from screener.data.store import DATA_DIR
from screener import signals
from screener.userfiles import load_user_file, save_user_file

log = logging.getLogger(__name__)

ALERTS_PATH = DATA_DIR / "alerts.json"


def _get_alerts_path(user_id: str | None = None) -> Path:
    if user_id:
        return DATA_DIR / "users" / user_id / "alerts.json"
    return ALERTS_PATH


KINDS: dict[str, str] = {
    "price_below": "מחיר מתחת ל-",
    "price_above": "מחיר מעל ל-",
    "change_below": "שינוי יומי מתחת ל-",
    "change_above": "שינוי יומי מעל ל-",
    "cross_above_sma200": "חציית SMA200 כלפי מעלה",
    "cross_below_sma200": "חציית SMA200 כלפי מטה",
    "earnings_within": "דוח כספי תוך (ימים)",
}


@dataclass
class Alert:
    id: str = ""
    ticker: str = ""
    kind: str = ""                           # see KINDS
    value: float | None = None               # threshold; None for SMA200 crosses
    note: str = ""
    created: str = ""                        # ISO date (e.g. "2025-03-14")
    last_fired: str | None = None            # ISO session date it last fired


@dataclass(frozen=True)
class Fired:
    alert_id: str
    ticker: str
    message: str


AlertResult = Fired


# ── Storage ──────────────────────────────────────────────────────────────────

def load(path: Path | None = None, user_id: str | None = None) -> dict[str, Alert]:
    """Load all alerts indexed by alert id."""
    path = path or _get_alerts_path(user_id)
    try:
        return load_user_file(path, Alert)
    except OSError as e:
        log.warning("cannot read alerts file %s: %s", path, e)
        return {}


def load_alerts(user_id: str | None = None) -> list[Alert]:
    """Load all alerts for user_id as a list of Alert objects."""
    return list(load(user_id=user_id).values())


def add(alert: Alert, path: Path | None = None, user_id: str | None = None) -> Alert:
    """Add or update an alert. Fills id and created if empty."""
    path = path or _get_alerts_path(user_id)
    today_str = dt.date.today().isoformat()
    if not alert.id:
        alert.id = uuid.uuid4().hex
    if not alert.created:
        alert.created = today_str

    def modify(alerts: dict[str, Alert]) -> None:
        alerts[alert.id] = alert

    save_user_file(path, Alert, modify)
    return alert


def save_alert(alert: Alert, user_id: str | None = None) -> None:
    """Save an alert for the given user_id."""
    add(alert, user_id=user_id)


def remove(alert_id: str, path: Path | None = None, user_id: str | None = None) -> None:
    """Remove an alert by id."""
    path = path or _get_alerts_path(user_id)
    save_user_file(path, Alert, lambda alerts: alerts.pop(alert_id, None))


def delete_alert(alert_id: str, user_id: str | None = None) -> None:
    """Delete an alert by id for the given user_id."""
    remove(alert_id, user_id=user_id)


def mark_fired(
    alert_ids: str | Iterable[str],
    session_date: dt.date | str,
    path: Path | None = None,
    user_id: str | None = None,
) -> None:
    """Mark alerts as fired on a given session date."""
    path = path or _get_alerts_path(user_id)
    if isinstance(alert_ids, str):
        target_ids = {alert_ids}
    else:
        target_ids = set(alert_ids)

    s_date = (
        session_date.isoformat()
        if isinstance(session_date, (dt.date, dt.datetime))
        else str(session_date)
    )

    def modify(alerts: dict[str, Alert]) -> None:
        for aid in target_ids:
            if aid in alerts:
                alerts[aid].last_fired = s_date

    save_user_file(path, Alert, modify)


def fired_today(
    universe: pd.DataFrame,
    today: dt.date,
    user_id: str | None = None,
) -> list[AlertResult]:
    """Evaluate and return fired alerts today for a given user_id (or default)."""
    alert_dict = load(user_id=user_id)
    return evaluate(alert_dict, universe, session_date=today, today=today)


# ── Evaluation ───────────────────────────────────────────────────────────────

def _to_date(val) -> dt.date | None:
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, pd.Timestamp):
        return val.date()
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    try:
        ts = pd.to_datetime(val)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def evaluate(
    alerts: Iterable[Alert] | dict[str, Alert],
    universe: pd.DataFrame,
    session_date: dt.date,
    today: dt.date | None = None,
) -> list[Fired]:
    """
    Evaluate alerts against universe snapshot for session_date.
    Pure function: returns list of Fired alerts without mutating storage.
    """
    today = today or dt.date.today()
    if isinstance(alerts, dict):
        alert_list = list(alerts.values())
    else:
        alert_list = list(alerts)

    if universe.empty or "ticker" not in universe.columns:
        return []

    s_date_str = (
        session_date.isoformat()
        if isinstance(session_date, (dt.date, dt.datetime))
        else str(session_date)
    )

    # Pre-calculate SMA200 crosses reusing signals.cross_up and signals.cross_down
    cross_cols = {"price", "sma200", "close_prev", "sma200_prev"}
    has_cross_cols = cross_cols.issubset(universe.columns)
    if has_cross_cols:
        up_series = signals.cross_up("price", "sma200")(universe).fillna(False)
        down_series = signals.cross_down("price", "sma200")(universe).fillna(False)
        up_by_ticker = dict(zip(universe["ticker"], up_series))
        down_by_ticker = dict(zip(universe["ticker"], down_series))
    else:
        up_by_ticker = {}
        down_by_ticker = {}

    # The rows of the alerted tickers only (the universe has ~5,000 rows; a few alerts need a few rows)
    wanted = {a.ticker for a in alert_list}
    sub = universe[universe["ticker"].isin(wanted)].drop_duplicates("ticker")
    ticker_rows = {row["ticker"]: row for _, row in sub.iterrows()}

    fired_list: list[Fired] = []

    for alert in alert_list:
        if alert.last_fired == s_date_str:
            continue

        if alert.ticker not in ticker_rows:
            continue

        row = ticker_rows[alert.ticker]
        msg: str | None = None

        if alert.kind == "price_below":
            if alert.value is None or pd.isna(alert.value):
                continue
            price = row.get("price")
            if price is None or pd.isna(price):
                continue
            if price < alert.value:
                msg = f"{alert.ticker}: המחיר {price:.2f} ירד מתחת ל-{alert.value:.2f}"

        elif alert.kind == "price_above":
            if alert.value is None or pd.isna(alert.value):
                continue
            price = row.get("price")
            if price is None or pd.isna(price):
                continue
            if price > alert.value:
                msg = f"{alert.ticker}: המחיר {price:.2f} עלה מעל ל-{alert.value:.2f}"

        elif alert.kind == "change_below":
            if alert.value is None or pd.isna(alert.value):
                continue
            chg = row.get("change")
            if chg is None or pd.isna(chg):
                continue
            threshold = alert.value if alert.value < 0 else -abs(alert.value)
            if chg <= threshold:
                chg_val = round(abs(chg), 2)
                val_val = round(abs(alert.value), 2)
                msg = f"{alert.ticker}: ירדה {chg_val:g}% היום (סף: {val_val:g}%)"

        elif alert.kind == "change_above":
            if alert.value is None or pd.isna(alert.value):
                continue
            chg = row.get("change")
            if chg is None or pd.isna(chg):
                continue
            threshold = abs(alert.value)
            if chg >= threshold:
                chg_val = round(abs(chg), 2)
                val_val = round(abs(alert.value), 2)
                msg = f"{alert.ticker}: עלתה {chg_val:g}% היום (סף: {val_val:g}%)"

        elif alert.kind == "cross_above_sma200":
            if up_by_ticker.get(alert.ticker, False):
                msg = f"{alert.ticker}: חצתה את ממוצע 200 כלפי מעלה היום"

        elif alert.kind == "cross_below_sma200":
            if down_by_ticker.get(alert.ticker, False):
                msg = f"{alert.ticker}: חצתה את ממוצע 200 כלפי מטה היום"

        elif alert.kind == "earnings_within":
            if alert.value is None or pd.isna(alert.value):
                continue
            edate = _to_date(row.get("earnings_date"))
            if edate is None:
                continue
            days = (edate - today).days
            if 0 <= days <= alert.value:
                date_str = edate.strftime("%d/%m")
                if days == 0:
                    msg = f"{alert.ticker}: דוח כספי היום ({date_str})"
                else:
                    msg = f"{alert.ticker}: דוח כספי בעוד {int(days)} ימים ({date_str})"

        if msg:
            if alert.note and alert.note.strip():
                msg += f" — {alert.note.strip()}"
            fired_list.append(Fired(alert_id=alert.id, ticker=alert.ticker, message=msg))

    return fired_list
