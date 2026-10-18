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
    missing = [k for k, c in COLUMNS.items() if c.tv and k not in raw.columns]
    return Previous(day, add_derived(raw, today=day), missing)


def mark_new(res: pd.DataFrame, prev: Previous | None, selections: dict[str, str], junk: bool) -> pd.Series:
    """'NEW' for rows of `res` that the same screen did not return on the previous snapshot day."""
    if prev is None:
        return pd.Series("", index=res.index)
    before = set(screen(prev.df, selections, junk)["ticker"])
    return res["ticker"].map(lambda t: "" if t in before else NEW)
