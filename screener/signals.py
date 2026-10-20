"""
Signals — Finviz-style events on the latest daily bar, defined once and used twice:
filters.py turns them into dropdown options, and summarize() builds the "Signals" text column
("Engulfing+, Vol 2.8x, BO 1M") plus the "# Signals" count, so a stock that passes several
signals at once stands out when sorted.

All signals use only the single universe fetch: "[1]" fields are TradingView's value as of the
previous bar, so "crossed today" = above now and not above yesterday. Breakouts compare today's
close with yesterday's N-month high, so today's own bar cannot hide the breakout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from screener.fields import CANDLES

Mask = Callable[[pd.DataFrame], pd.Series]

VOL_SPIKE = 2.0          # rel volume that counts as a spike in the Signals column
# Consolidation is relative to the stock's own volatility: a fixed "range < 10%" fires for ~20% of
# the universe (every quiet large cap); "1-month range < 3 x ATR%" fires for ~6% (live check 2026-10-18).
TIGHT_ATR = 3.0


@dataclass(frozen=True)
class Signal:
    label: str           # dropdown option label
    tag: str             # short text in the Signals column
    fn: Mask


def _prev(col: str) -> str:
    return "close_prev" if col == "price" else f"{col}_prev"


def cross_up(a: str, b: str) -> Mask:
    return lambda d: (d[a] > d[b]) & (d[_prev(a)] <= d[_prev(b)])


def cross_down(a: str, b: str) -> Mask:
    return lambda d: (d[a] < d[b]) & (d[_prev(a)] >= d[_prev(b)])


def _candle_fn(k: str) -> Mask:
    if k == "cdl_doji":
        return lambda d: (d[k] == 1) & (d["cdl_doji_dragonfly"] != 1) & (d["cdl_doji_gravestone"] != 1)
    return lambda d, k=k: d[k] == 1


CANDLE_SIGNALS: list[Signal] = [
    Signal(label, tag + bias, _candle_fn(k)) for k, (label, bias, tag, _) in CANDLES.items()
]


def any_candle(bias: str) -> Mask:
    keys = [k for k, c in CANDLES.items() if c[1] == bias]
    return lambda d: d[keys].eq(1).any(axis=1)


# Ordered strongest first: the Signals column shows only the first match of each list.
BREAKOUTS: list[Signal] = [
    Signal("New 52-week high (breakout)", "BO 52W", lambda d: d["price"] > d["high52_prev"]),
    Signal("New 6-month high (breakout)", "BO 6M", lambda d: d["price"] > d["high_6m_prev"]),
    Signal("New 3-month high (breakout)", "BO 3M", lambda d: d["price"] > d["high_3m_prev"]),
    Signal("New 1-month high (breakout)", "BO 1M", lambda d: d["price"] > d["high_1m_prev"]),
]
BREAKDOWNS: list[Signal] = [
    Signal("New 52-week low (breakdown)", "BD 52W", lambda d: d["price"] < d["low52_prev"]),
    Signal("New 6-month low (breakdown)", "BD 6M", lambda d: d["price"] < d["low_6m_prev"]),
    Signal("New 3-month low (breakdown)", "BD 3M", lambda d: d["price"] < d["low_3m_prev"]),
    Signal("New 1-month low (breakdown)", "BD 1M", lambda d: d["price"] < d["low_1m_prev"]),
]
CONSOLIDATION: list[Signal] = [
    Signal(f"Volatility contraction (1M range < {TIGHT_ATR:g}x ATR)", "Tight 1M",
           lambda d: d["range_1m_pct"] < TIGHT_ATR * d["atr_pct"]),
]
CROSSES: list[Signal] = [
    Signal("Golden Cross today (SMA50 crossed above SMA200)", "Golden Cross", cross_up("sma50", "sma200")),
    Signal("Death Cross today (SMA50 crossed below SMA200)", "Death Cross", cross_down("sma50", "sma200")),
    Signal("SMA20 crossed above SMA50 today", "SMA20↑SMA50", cross_up("sma20", "sma50")),
    Signal("SMA20 crossed below SMA50 today", "SMA20↓SMA50", cross_down("sma20", "sma50")),
    Signal("Price crossed above SMA50 today", "Px↑SMA50", cross_up("price", "sma50")),
    Signal("Price crossed below SMA50 today", "Px↓SMA50", cross_down("price", "sma50")),
    Signal("Price crossed above SMA200 today", "Px↑SMA200", cross_up("price", "sma200")),
    Signal("Price crossed below SMA200 today", "Px↓SMA200", cross_down("price", "sma200")),
]


def _volume_tag(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rv = d["rel_volume"]
    return rv >= VOL_SPIKE, "Vol " + rv.round(1).astype(str) + "x"


def summarize(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """("Engulfing+, Vol 2.8x, BO 1M", 3) per row. Missing inputs never produce a signal."""
    parts: list[tuple[np.ndarray, np.ndarray]] = []   # (mask, tag) in display order

    def add(mask: pd.Series, tag) -> None:
        m = mask.fillna(False).astype(bool).to_numpy()
        parts.append((m, np.asarray(tag if not isinstance(tag, str) else np.full(len(d), tag), dtype=object)))

    for s in CANDLE_SIGNALS:
        add(s.fn(d), s.tag)
    vol_mask, vol_tag = _volume_tag(d)
    add(vol_mask, vol_tag)
    for group in (BREAKOUTS, BREAKDOWNS, CONSOLIDATION):
        taken = np.zeros(len(d), dtype=bool)
        for s in group:
            m = s.fn(d).fillna(False).astype(bool).to_numpy() & ~taken
            taken |= m
            add(pd.Series(m, index=d.index), s.tag)
    for s in CROSSES:
        add(s.fn(d), s.tag)

    if not parts:
        return pd.Series("", index=d.index), pd.Series(0, index=d.index)
    masks = np.column_stack([m for m, _ in parts])
    tags = np.column_stack([t for _, t in parts])
    text = [", ".join(row_tags[row_mask]) for row_mask, row_tags in zip(masks, tags)]
    return pd.Series(text, index=d.index, dtype=object), pd.Series(masks.sum(axis=1), index=d.index).astype(float)
