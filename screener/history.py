"""
"New since yesterday": which of today's results did not pass the same screen in the previous
daily snapshot (_data/snapshots/, the last fetch of each day — Friday's on a Monday).

Both days go through the same pipeline (derive → hide junk → filters), so a stock is NEW when it
started to match the screen, or when it was not in the universe at all (new listing).
No network: only local snapshots are read.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from screener.data.store import Store
from screener.derive import add_derived
from screener.fields import COLUMNS
from screener.filters import screen

log = logging.getLogger(__name__)

NEW = "NEW"


@dataclass
class Previous:
    day: dt.date
    df: pd.DataFrame            # derived, like the current universe
    missing: list[str]          # provider columns the snapshot lacks (saved by an older version)


def load_previous(store: Store, today: dt.date) -> Previous | None:
    day = store.previous_snapshot_date(today)
    if day is None:
        return None
    try:
        raw = store.load_snapshot(day)
    except Exception as e:      # a damaged snapshot must not break the screener
        log.warning("cannot read snapshot %s: %s", day, e)
        return None
    from screener.fields import provider_columns
    missing = [k for k in provider_columns() if k not in raw.columns]
    return Previous(day, add_derived(raw, today=day), missing)


def previous_tickers(prev: Previous, selections: dict[str, str], junk: bool) -> set[str]:
    """Tickers the same screen returned on the previous snapshot day (the app caches this per screen)."""
    return set(screen(prev.df, selections, junk)["ticker"])


def mark_new(res: pd.DataFrame, before: set[str] | None) -> pd.Series:
    """'NEW' for rows of `res` whose ticker is not in `before`; all blank when there is no previous day."""
    if before is None:
        return pd.Series("", index=res.index)
    return res["ticker"].map(lambda t: "" if t in before else NEW)
