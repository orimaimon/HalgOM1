"""
Finviz-style dropdown filters.

Every filter is a dropdown of named options ("Over 20", "Under 5%", ...). An option is a
predicate on the whole DataFrame, so options may combine columns (e.g. SMA50 above SMA200).
"Any" (the default) means the filter is off. Dynamic filters (sector, industry, ...) take
their options from the current data.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from screener import signals as S
from screener.fields import COLUMNS, INDEXES

log = logging.getLogger(__name__)

ANY = "Any"
Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Option:
    label: str
    fn: Mask


@dataclass(frozen=True)
class Filter:
    key: str
    label: str
    group: str                  # Descriptive | Fundamental | Technical | Signals | Scores
    column: str
    options: tuple[Option, ...] = ()
    dynamic: bool = False
    help: str = field(default="")

    @property
    def tooltip(self) -> str:
        return self.help or COLUMNS[self.column].help


# ─────────────────────────── option builders ───────────────────────────────

def _fmt(v: float, unit: str) -> str:
    if unit == "$":
        return f"${v:g}"
    if unit == "%":
        return f"{v:g}%"
    if unit == "K":   # share counts / volumes
        return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K"
    if unit == "$B":  # dollar amounts
        return f"${v / 1e9:g}B" if v >= 1e9 else f"${v / 1e6:g}M"
    return f"{v:g}"


class N:
    """Builds numeric options for one column. NaN never passes a numeric option."""

    def __init__(self, col: str, unit: str = ""):
        self.col, self.unit = col, unit

    def s(self, d: pd.DataFrame) -> pd.Series:
        return d[self.col]

    def over(self, *vals: float) -> list[Option]:
        return [Option(f"Over {_fmt(v, self.unit)}", lambda d, v=v: self.s(d) > v) for v in vals]

    def under(self, *vals: float) -> list[Option]:
        return [Option(f"Under {_fmt(v, self.unit)}", lambda d, v=v: self.s(d) < v) for v in vals]

    def between(self, *pairs: tuple[float, float]) -> list[Option]:
        return [Option(f"{_fmt(lo, self.unit)} to {_fmt(hi, self.unit)}",
                       lambda d, lo=lo, hi=hi: self.s(d).between(lo, hi)) for lo, hi in pairs]

    def opt(self, label: str, fn: Callable[[pd.Series], pd.Series]) -> list[Option]:
        return [Option(label, lambda d: fn(self.s(d)))]

    def pos_neg(self, pos: str = "Positive (>0%)", neg: str = "Negative (<0%)") -> list[Option]:
        return self.opt(pos, lambda s: s > 0) + self.opt(neg, lambda s: s < 0)


def _perf(col: str) -> tuple[Option, ...]:
    n = N(col, "%")
    up = [Option(f"Up {v:g}%+", lambda d, v=v: d[col] >= v) for v in (5, 10, 20, 30, 50, 100)]
    down = [Option(f"Down {v:g}%+", lambda d, v=v: d[col] <= -v) for v in (5, 10, 20, 30, 50)]
    return tuple(n.opt("Up", lambda s: s > 0) + n.opt("Down", lambda s: s < 0) + up + down)


def _ma_vs(a: str, b: str) -> list[Option]:
    """"SMA50 above SMA200" / "SMA50 below SMA200" — shared by the SMA filters and MA Cross."""
    la, lb = COLUMNS[a].label, COLUMNS[b].label
    return [Option(f"{la} above {lb}", lambda d: d[a] > d[b]), Option(f"{la} below {lb}", lambda d: d[a] < d[b])]


def _sma(n: int) -> tuple[Option, ...]:
    sma, pct = f"sma{n}", f"sma{n}_pct"
    opts = [
        Option(f"Price above SMA{n}", lambda d: d["price"] > d[sma]),
        Option(f"Price below SMA{n}", lambda d: d["price"] < d[sma]),
        *[Option(f"Price {v}% above SMA{n}", lambda d, v=v: d[pct] >= v) for v in (10, 20, 30, 50)],
        *[Option(f"Price {v}% below SMA{n}", lambda d, v=v: d[pct] <= -v) for v in (10, 20, 30, 50)],
        Option(f"Price within 2% of SMA{n}", lambda d: d[pct].abs() <= 2),
    ]
    if f"{sma}_prev" in COLUMNS:
        opts.append(Option(f"SMA{n} rising", lambda d, sma=sma: d[sma] > d[f"{sma}_prev"]))
    slower = {20: "sma50", 50: "sma200", 150: "sma200"}.get(n)
    if slower:
        opts += _ma_vs(sma, slower)
    return tuple(opts)


def _signals(sigs: list[S.Signal]) -> list[Option]:
    return [Option(s.label, s.fn) for s in sigs]


def in_index(label: str) -> Mask:
    """Membership in a major index. The column holds "S&P 500, Nasdaq 100" — whole names only."""
    return lambda d: (", " + d["indexes"].fillna("").astype(str) + ",").str.contains(f", {label},", regex=False)


def _stack(*cols: str) -> Mask:
    """cols[0] > cols[1] > ... (each pair strictly)."""
    def fn(d: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=d.index)
        for a, b in zip(cols, cols[1:]):
            m &= d[a] > d[b]
        return m
    return fn


def _margin(col: str) -> tuple[Option, ...]:
    n = N(col, "%")
    return tuple(n.pos_neg() + n.opt("High (>20%)", lambda s: s > 20)
                 + n.over(5, 10, 15, 25, 30, 40, 50, 60, 70) + n.under(0, 10, 20, 30))


def _returns(col: str) -> tuple[Option, ...]:
    n = N(col, "%")
    return tuple(n.pos_neg() + n.opt("Very positive (>30%)", lambda s: s > 30)
                 + n.over(5, 10, 15, 20, 25, 30, 40, 50) + n.under(-10, -5, 5, 10))


def _growth(col: str) -> tuple[Option, ...]:
    n = N(col, "%")
    return tuple(n.pos_neg() + n.opt("Positive, low (0-10%)", lambda s: s.between(0, 10))
                 + n.opt("High (>25%)", lambda s: s > 25) + n.over(5, 10, 15, 20, 30, 50, 100)
                 + n.under(5, 10, 20))


def _multiple(col: str, low: float, high: float, steps: tuple[float, ...]) -> tuple[Option, ...]:
    n = N(col)
    return tuple(n.opt(f"Low (<{low:g})", lambda s: (s > 0) & (s < low))
                 + n.opt("Profitable (>0)", lambda s: s > 0)
                 + n.opt(f"High (>{high:g})", lambda s: s > high)
                 + [Option(f"Under {v:g}", lambda d, v=v: (d[col] > 0) & (d[col] < v)) for v in steps]
                 + n.over(*steps))


def _ratio(col: str, steps: tuple[float, ...]) -> list[Option]:
    n = N(col)
    return n.over(*steps) + n.under(*steps)


# ─────────────────────────── filter definitions ────────────────────────────

_cap = N("market_cap", "$B")
_div = N("div_yield", "%")
_price = N("price", "$")
_avgv = N("avg_volume", "K")
_vol = N("volume", "K")
_relv = N("rel_volume")
_dv = N("dollar_volume", "$B")
_float = N("float_shares", "K")
_earn = N("days_to_earnings")
_recom = N("recom")
_tgt = N("target_upside", "%")
_payout = N("payout", "%")
_rsi = N("rsi")
_gap = N("gap", "%")
_chg = N("change", "%")
_h52 = N("high52_pct", "%")
_l52 = N("low52_pct", "%")
_beta = N("beta")
_atrp = N("atr_pct", "%")
_volw = N("vol_w", "%")
_volm = N("vol_m", "%")
_rsiw = N("rsi_w")
_smaw = N("sma50w_pct", "%")
_relvs = N("rel_volume")
_nsig = N("n_signals")

FILTERS: list[Filter] = [
    # ── Descriptive ─────────────────────────────────────────────────────────
    Filter("exchange", "Exchange", "Descriptive", "exchange", dynamic=True),
    Filter("sector", "Sector", "Descriptive", "sector", dynamic=True),
    Filter("industry", "Industry", "Descriptive", "industry", dynamic=True),
    Filter("country", "Country", "Descriptive", "country", dynamic=True),
    Filter("index", "Index", "Descriptive", "indexes", tuple(
        [Option(label, in_index(label)) for label in INDEXES.values()]
        + [Option("Not in S&P 500", lambda d: ~in_index("S&P 500")(d)),
           Option("Any major index", lambda d: d["indexes"].fillna("").astype(str) != "")])),
    Filter("market_cap", "Market Cap.", "Descriptive", "market_cap", tuple(
        _cap.opt("Mega ($200B+)", lambda s: s >= 200e9)
        + _cap.opt("Large ($10B to $200B)", lambda s: s.between(10e9, 200e9))
        + _cap.opt("Mid ($2B to $10B)", lambda s: s.between(2e9, 10e9))
        + _cap.opt("Small ($300M to $2B)", lambda s: s.between(300e6, 2e9))
        + _cap.opt("Micro ($50M to $300M)", lambda s: s.between(50e6, 300e6))
        + _cap.opt("Nano (under $50M)", lambda s: s < 50e6)
        + _cap.opt("+Large (over $10B)", lambda s: s >= 10e9)
        + _cap.opt("+Mid (over $2B)", lambda s: s >= 2e9)
        + _cap.opt("+Small (over $300M)", lambda s: s >= 300e6)
        + _cap.opt("+Micro (over $50M)", lambda s: s >= 50e6)
        + _cap.opt("-Large (under $200B)", lambda s: s < 200e9)
        + _cap.opt("-Mid (under $10B)", lambda s: s < 10e9)
        + _cap.opt("-Small (under $2B)", lambda s: s < 2e9))),
    Filter("div_yield", "Dividend Yield", "Descriptive", "div_yield", tuple(
        _div.opt("None (0%)", lambda s: s.fillna(0) == 0) + _div.opt("Positive (>0%)", lambda s: s > 0)
        + _div.opt("High (>5%)", lambda s: s > 5) + _div.opt("Very High (>10%)", lambda s: s > 10)
        + _div.over(1, 2, 3, 4, 5, 6, 8))),
    Filter("price", "Price", "Descriptive", "price", tuple(
        _price.under(1, 2, 5, 10, 20, 50) + _price.over(1, 2, 5, 10, 20, 50, 100)
        + _price.between((1, 5), (5, 10), (5, 50), (10, 20), (20, 50), (50, 100)))),
    Filter("avg_volume", "Average Volume", "Descriptive", "avg_volume", tuple(
        _avgv.under(50e3, 100e3, 500e3, 1e6) + _avgv.over(50e3, 100e3, 200e3, 300e3, 500e3, 750e3, 1e6, 2e6))),
    Filter("rel_volume", "Relative Volume", "Descriptive", "rel_volume", tuple(
        _relv.over(0.5, 1, 1.5, 2, 3, 5, 10) + _relv.under(0.5, 0.75, 1))),
    Filter("volume", "Current Volume", "Descriptive", "volume", tuple(
        _vol.under(100e3, 500e3, 1e6) + _vol.over(100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6))),
    Filter("dollar_volume", "Dollar Volume", "Descriptive", "dollar_volume", tuple(
        _dv.over(1e6, 5e6, 10e6, 20e6, 50e6, 100e6, 500e6) + _dv.under(1e6, 5e6))),
    Filter("float_shares", "Float", "Descriptive", "float_shares", tuple(
        _float.under(1e6, 5e6, 10e6, 20e6, 50e6, 100e6) + _float.over(1e6, 10e6, 50e6, 100e6, 500e6, 1e9))),
    Filter("earnings", "Earnings Date", "Descriptive", "days_to_earnings", tuple(
        _earn.opt("Today", lambda s: s == 0) + _earn.opt("Tomorrow", lambda s: s == 1)
        + _earn.opt("Next 7 days", lambda s: s.between(0, 7))
        + _earn.opt("Next 14 days", lambda s: s.between(0, 14))
        + _earn.opt("Next 30 days", lambda s: s.between(0, 30))
        + _earn.opt("Not in next 7 days", lambda s: s.notna() & ~s.between(0, 7))
        + _earn.opt("Not in next 14 days", lambda s: s.notna() & ~s.between(0, 14))),
        help="תאריך הדוח הבא. 'Not in next 7 days' מסנן מניות שעומד לצאת להן דוח — דוחות גורמים לקפיצות חדות"),
    Filter("recom", "Analyst Recom.", "Descriptive", "recom", tuple(
        _recom.opt("Strong Buy (1)", lambda s: s <= 1.5) + _recom.opt("Buy or better", lambda s: s <= 2.5)
        + _recom.opt("Hold or better", lambda s: s <= 3.5)
        + [Option("Hold", lambda d: (d["recom"] > 2.5) & (d["recom"] <= 3.5))]
        + _recom.opt("Sell or worse", lambda s: s > 3.5))),
    Filter("target_upside", "Target Price", "Descriptive", "target_upside", tuple(
        _tgt.opt("Above price", lambda s: s > 0) + _tgt.opt("Below price", lambda s: s < 0)
        + [Option(f"{v}% above price", lambda d, v=v: d["target_upside"] >= v) for v in (10, 20, 30, 50)]
        + [Option(f"{v}% below price", lambda d, v=v: d["target_upside"] <= -v) for v in (10, 20)])),

    # ── Fundamental ─────────────────────────────────────────────────────────
    Filter("pe", "P/E", "Fundamental", "pe", _multiple("pe", 15, 50, (5, 10, 15, 20, 25, 30, 40, 50))),
    Filter("fwd_pe", "Forward P/E", "Fundamental", "fwd_pe",
           _multiple("fwd_pe", 15, 50, (5, 10, 15, 20, 25, 30, 40, 50))),
    Filter("peg", "PEG", "Fundamental", "peg", _multiple("peg", 1, 2, (1, 2, 3))),
    Filter("ps", "P/S", "Fundamental", "ps", _multiple("ps", 1, 10, (1, 2, 3, 4, 5, 10))),
    Filter("pb", "P/B", "Fundamental", "pb", _multiple("pb", 1, 5, (1, 2, 3, 4, 5, 10))),
    Filter("pfcf", "Price/Free Cash Flow", "Fundamental", "pfcf",
           _multiple("pfcf", 15, 50, (5, 10, 15, 20, 25, 30, 40, 50, 100))),
    Filter("ev_ebitda", "EV/EBITDA", "Fundamental", "ev_ebitda",
           _multiple("ev_ebitda", 10, 30, (5, 10, 15, 20, 25, 30))),
    Filter("eps_growth", "EPS growth TTM", "Fundamental", "eps_growth", _growth("eps_growth")),
    Filter("eps_growth_q", "EPS growth Q/Q", "Fundamental", "eps_growth_q", _growth("eps_growth_q")),
    Filter("sales_growth", "Sales growth TTM", "Fundamental", "sales_growth", _growth("sales_growth")),
    Filter("sales_growth_q", "Sales growth Q/Q", "Fundamental", "sales_growth_q", _growth("sales_growth_q")),
    Filter("roa", "Return on Assets", "Fundamental", "roa", _returns("roa")),
    Filter("roe", "Return on Equity", "Fundamental", "roe", (
        Option("Positive (>0%)", lambda d: (d["roe"] > 0) & (~(d["debt_eq"] < 0))),
        Option("Negative (<0%)", lambda d: d["roe"] < 0),
        *N("roe", "%").opt("Very positive (>30%)", lambda s: s > 30),
        *N("roe", "%").over(5, 10, 15, 20, 25, 30, 40, 50),
        *N("roe", "%").under(-10, -5, 5, 10))),
    Filter("roic", "Return on Invested Cap.", "Fundamental", "roic", _returns("roic")),
    Filter("curr_ratio", "Current Ratio", "Fundamental", "curr_ratio", tuple(
        N("curr_ratio").opt("High (>3)", lambda s: s > 3) + N("curr_ratio").opt("Low (<1)", lambda s: s < 1)
        + _ratio("curr_ratio", (0.5, 1, 1.5, 2, 3)))),
    Filter("quick_ratio", "Quick Ratio", "Fundamental", "quick_ratio", tuple(
        N("quick_ratio").opt("High (>3)", lambda s: s > 3) + N("quick_ratio").opt("Low (<0.5)", lambda s: s < 0.5)
        + _ratio("quick_ratio", (0.5, 1, 1.5, 2, 3)))),
    Filter("lt_debt_eq", "LT Debt/Equity", "Fundamental", "lt_debt_eq", tuple(
        N("lt_debt_eq").opt("High (>0.5)", lambda s: s > 0.5)
        + [Option("Low (<0.1)", lambda d: d["lt_debt_eq"].between(0, 0.1, inclusive="left"))]
        + [Option(f"Under {v:g}", lambda d, v=v: d["lt_debt_eq"].between(0, v, inclusive="left"))
           for v in (0.1, 0.2, 0.3, 0.5, 1)]
        + N("lt_debt_eq").over(0.1, 0.2, 0.3, 0.5, 1))),
    Filter("debt_eq", "Debt/Equity", "Fundamental", "debt_eq", tuple(
        N("debt_eq").opt("High (>0.5)", lambda s: s > 0.5)
        + [Option("Low (<0.1)", lambda d: d["debt_eq"].between(0, 0.1, inclusive="left"))]
        + N("debt_eq").opt("Negative equity", lambda s: s < 0)
        + [Option(f"Under {v:g}", lambda d, v=v: d["debt_eq"].between(0, v, inclusive="left"))
           for v in (0.1, 0.2, 0.3, 0.5, 1, 2)]
        + N("debt_eq").over(0.5, 1, 2))),
    Filter("gross_margin", "Gross Margin", "Fundamental", "gross_margin", _margin("gross_margin")),
    Filter("oper_margin", "Operating Margin", "Fundamental", "oper_margin", _margin("oper_margin")),
    Filter("net_margin", "Net Profit Margin", "Fundamental", "net_margin", _margin("net_margin")),
    Filter("payout", "Payout Ratio", "Fundamental", "payout", tuple(
        _payout.opt("None (0%)", lambda s: s.fillna(0) == 0) + _payout.opt("Positive (>0%)", lambda s: s > 0)
        + _payout.opt("Low (<20%)", lambda s: s.between(0, 20, inclusive="neither"))
        + _payout.opt("High (>50%)", lambda s: s > 50)
        + [Option(f"Under {v:g}%", lambda d, v=v: d["payout"].between(0, v, inclusive="left"))
           for v in (40, 60, 80, 100)]
        + _payout.over(20, 40, 60, 80, 100))),

    # ── Technical ───────────────────────────────────────────────────────────
    Filter("change", "Change", "Technical", "change", tuple(
        _chg.opt("Up", lambda s: s > 0) + _chg.opt("Down", lambda s: s < 0)
        + [Option(f"Up {v}%+", lambda d, v=v: d["change"] >= v) for v in (1, 2, 3, 5, 10, 20)]
        + [Option(f"Down {v}%+", lambda d, v=v: d["change"] <= -v) for v in (1, 2, 3, 5, 10, 20)])),
    Filter("perf_w", "Performance (Week)", "Technical", "perf_w", _perf("perf_w")),
    Filter("perf_m", "Performance (Month)", "Technical", "perf_m", _perf("perf_m")),
    Filter("perf_q", "Performance (Quarter)", "Technical", "perf_q", _perf("perf_q")),
    Filter("perf_h", "Performance (Half)", "Technical", "perf_h", _perf("perf_h")),
    Filter("perf_ytd", "Performance (YTD)", "Technical", "perf_ytd", _perf("perf_ytd")),
    Filter("perf_y", "Performance (Year)", "Technical", "perf_y", _perf("perf_y")),
    Filter("rsi", "RSI (14)", "Technical", "rsi", tuple(
        [Option(f"Overbought ({v})", lambda d, v=v: d["rsi"] > v) for v in (90, 80, 70, 60)]
        + [Option(f"Oversold ({v})", lambda d, v=v: d["rsi"] < v) for v in (10, 20, 30, 40)]
        + _rsi.opt("Not Overbought (<60)", lambda s: s < 60) + _rsi.opt("Not Overbought (<50)", lambda s: s < 50)
        + _rsi.opt("Not Oversold (>50)", lambda s: s > 50) + _rsi.opt("Not Oversold (>40)", lambda s: s > 40))),
    Filter("sma20", "20-Day Simple Moving Avg", "Technical", "sma20_pct", _sma(20)),
    Filter("sma50", "50-Day Simple Moving Avg", "Technical", "sma50_pct", _sma(50)),
    Filter("sma150", "150-Day Simple Moving Avg", "Technical", "sma150_pct", _sma(150)),
    Filter("sma200", "200-Day Simple Moving Avg", "Technical", "sma200_pct", _sma(200)),
    Filter("high52", "52-Week High", "Technical", "high52_pct", tuple(
        _h52.opt("New High (at high)", lambda s: s >= -0.5)
        + [Option(f"0-{v}% below High", lambda d, v=v: d["high52_pct"] >= -v) for v in (3, 5, 10, 25)]
        + [Option(f"{lo}-{hi}% below High", lambda d, lo=lo, hi=hi: d["high52_pct"].between(-hi, -lo))
           for lo, hi in ((5, 10), (10, 15), (15, 20), (20, 30))]
        + [Option(f"{v}% or more below High", lambda d, v=v: d["high52_pct"] <= -v) for v in (10, 20, 30, 50)])),
    Filter("low52", "52-Week Low", "Technical", "low52_pct", tuple(
        _l52.opt("New Low (at low)", lambda s: s <= 0.5)
        + [Option(f"0-{v}% above Low", lambda d, v=v: d["low52_pct"] <= v) for v in (3, 5, 10)]
        + [Option(f"{v}% or more above Low", lambda d, v=v: d["low52_pct"] >= v) for v in (20, 30, 50, 100, 200)])),
    Filter("gap", "Gap", "Technical", "gap", tuple(
        _gap.opt("Up", lambda s: s > 0) + _gap.opt("Down", lambda s: s < 0)
        + [Option(f"Up {v}%+", lambda d, v=v: d["gap"] >= v) for v in (1, 2, 3, 5, 10)]
        + [Option(f"Down {v}%+", lambda d, v=v: d["gap"] <= -v) for v in (1, 2, 3, 5, 10)])),
    Filter("beta", "Beta", "Technical", "beta", tuple(
        _beta.under(0, 0.5, 1, 1.5, 2) + _beta.over(0, 0.5, 1, 1.5, 2, 3)
        + _beta.between((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2)))),
    Filter("atr_pct", "ATR %", "Technical", "atr_pct", tuple(_atrp.over(1, 2, 3, 4, 5, 7, 10) + _atrp.under(1, 2, 3))),
    Filter("vol_w", "Volatility (Week)", "Technical", "vol_w", tuple(_volw.over(2, 3, 4, 5, 6, 8, 10, 15) + _volw.under(1, 2, 3))),
    Filter("vol_m", "Volatility (Month)", "Technical", "vol_m", tuple(_volm.over(2, 3, 4, 5, 6, 8, 10, 15) + _volm.under(1, 2, 3))),
    Filter("rsi_w", "RSI (14) Weekly", "Technical", "rsi_w", tuple(
        [Option(f"Overbought ({v})", lambda d, v=v: d["rsi_w"] > v) for v in (80, 70, 60)]
        + [Option(f"Oversold ({v})", lambda d, v=v: d["rsi_w"] < v) for v in (20, 30, 40)]
        + _rsiw.opt("Above 50 (uptrend)", lambda s: s > 50) + _rsiw.opt("Below 50 (downtrend)", lambda s: s < 50)
        + _rsiw.between((40, 60), (50, 70)))),
    Filter("sma50_w", "50-Week Simple Moving Avg", "Technical", "sma50w_pct", tuple(
        [Option("Price above SMA50 (weekly)", lambda d: d["close_w"] > d["sma50_w"]),
         Option("Price below SMA50 (weekly)", lambda d: d["close_w"] < d["sma50_w"])]
        + [Option(f"Price {v}% above SMA50 (weekly)", lambda d, v=v: d["sma50w_pct"] >= v) for v in (10, 20, 50)]
        + [Option(f"Price {v}% below SMA50 (weekly)", lambda d, v=v: d["sma50w_pct"] <= -v) for v in (10, 20)]
        + _smaw.opt("Price within 5% of SMA50 (weekly)", lambda s: s.abs() <= 5)),
        help="ממוצע נע של 50 שבועות (כשנה) בגרף השבועי, מול הסגירה השבועית. מעליו = מגמה ארוכה עולה"),

    # ── Signals (signals.py) ────────────────────────────────────────────────
    Filter("candle", "Candlestick", "Signals", "signals", tuple(
        [Option("Any bullish pattern", S.any_candle("+")), Option("Any bearish pattern", S.any_candle("-"))]
        + _signals(S.CANDLE_SIGNALS)),
        help="תבניות נרות שזוהו על הנר היומי האחרון (זיהוי של TradingView). "
             "במהלך יום המסחר הנר עוד לא נסגר — התבנית יכולה להיעלם עד הסגירה"),
    Filter("breakout", "Breakout (N-day)", "Signals", "high_1m_prev", tuple(
        _signals(S.BREAKOUTS[::-1]) + _signals(S.BREAKDOWNS[::-1])),
        help="פריצה: הסגירה היום מעל השיא של החודש / 3 חודשים / חצי שנה / שנה כפי שהיה אתמול. "
             "שבירה: הסגירה מתחת לשפל. חלונות של TradingView: חודש ≈ 21 ימי מסחר"),
    Filter("consolidation", "Consolidation", "Signals", "range_1m_pct", tuple(
        _signals(S.CONSOLIDATION)
        + [Option(f"Tight 1-month range (<{v:g}%)", lambda d, v=v: d["range_1m_pct"] < v) for v in (10, 15, 20)]
        + [Option(f"Tight 5-day range (<{v:g}%)", lambda d, v=v: d["range_5d_pct"] < v) for v in (3, 5)]
        + [Option("Tight base near 1-month high (<15%, within 3%)",
                  lambda d: (d["range_1m_pct"] < 15) & (d["price"] >= d["high_1m"] * 0.97))]),
        help="התכנסות: המניה נסחרת בטווח צר. Volatility contraction = הטווח של החודש האחרון קטן מ-3 תנועות "
             "יומיות ממוצעות (ATR) — צר ביחס למניה עצמה. Tight range = טווח צר באחוזים (מניות גדולות ושקטות "
             "עוברות אותו בקלות). בסיס צר ליד השיא החודשי הוא מצב קלאסי לפני פריצה"),
    Filter("vol_spike", "Volume Spike", "Signals", "rel_volume", tuple(
        _relvs.opt("Volume 2x average", lambda s: s >= 2) + _relvs.opt("Volume 3x average", lambda s: s >= 3)
        + _relvs.opt("Volume 5x average", lambda s: s >= 5)
        + [Option("Up on 2x volume", lambda d: (d["rel_volume"] >= 2) & (d["change"] > 0)),
           Option("Down on 2x volume", lambda d: (d["rel_volume"] >= 2) & (d["change"] < 0))]),
        help="קפיצת מחזור: מחזור היום חלקי המחזור הממוצע של 10 הימים האחרונים. "
             "עלייה במחזור כפול = קונים גדולים נכנסים"),
    Filter("ma_cross", "MA Cross", "Signals", "sma50_pct", tuple(
        _ma_vs("sma50", "sma200") + _ma_vs("sma20", "sma50") + _signals(S.CROSSES)),
        help="מצב הממוצעים הנעים, או חצייה שקרתה היום (מול הערכים של אתמול). "
             "Golden Cross = הממוצע של 50 יום חצה מעל זה של 200 יום — סימן קלאסי לתחילת מגמת עלייה"),
    Filter("ma_stack", "MA Alignment", "Signals", "sma150_pct", (
        Option("Price > SMA50 > SMA150 > SMA200 (Stage 2)", _stack("price", "sma50", "sma150", "sma200")),
        Option("Price > SMA20 > SMA50 > SMA200", _stack("price", "sma20", "sma50", "sma200")),
        Option("Price < SMA50 < SMA200", _stack("sma200", "sma50", "price")),
        Option("Price < SMA20 < SMA50 < SMA200", _stack("sma200", "sma50", "sma20", "price"))),
        help="סדר הממוצעים הנעים. מחיר מעל ממוצע קצר שמעל ממוצע ארוך = מגמת עלייה מסודרת"),
    Filter("n_signals", "Signals Count", "Signals", "n_signals", tuple(
        _nsig.opt("None", lambda s: s == 0) + _nsig.opt("1 or more", lambda s: s >= 1)
        + _nsig.opt("2 or more", lambda s: s >= 2) + _nsig.opt("3 or more", lambda s: s >= 3))),

    # ── HalgOM scores ───────────────────────────────────────────────────────
    *[Filter(k, COLUMNS[k].label + (" Score" if k != "score" else ""), "Scores", k, tuple(
        N(k).over(50, 60, 70, 80, 90, 95) + N(k).under(20, 30, 50)))
      for k in ("score", "score_value", "score_quality", "score_momentum")],
]

FILTERS_BY_KEY: dict[str, Filter] = {f.key: f for f in FILTERS}
GROUPS = ["Descriptive", "Fundamental", "Technical", "Signals", "Scores"]

_SPAC_RE = re.compile(
    r"\b(?:acquisition|merger)\s+(?:corp|corporation|co|company|holdings|inc|ltd|limited)\b|\bblank\s+check\b",
    re.IGNORECASE,
)


def _is_spac(d: pd.DataFrame) -> pd.Series:
    """A blank-check company: the name says so, and it sits in its industry or near its $10 trust value."""
    if "company" not in d:
        return pd.Series(False, index=d.index)
    name_match = d["company"].fillna("").astype(str).str.contains(_SPAC_RE)
    industry = d["industry"] if "industry" in d else pd.Series(None, index=d.index, dtype=object)
    return name_match & ((industry == "Financial Conglomerates") | d["price"].between(9, 12))


# "Hide junk" toolbar switch (on by default): known-bad values only — a missing value is not junk.
JUNK_RULES: list[tuple[str, Mask]] = [
    ("Price under $1", lambda d: d["price"] < 1),
    ("Avg Volume under 50K", lambda d: d["avg_volume"] < 50e3),
    ("Market Cap under $50M", lambda d: d["market_cap"] < 50e6),
    ("SPAC (blank-check company)", _is_spac),
]


def junk_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    for _, fn in JUNK_RULES:
        m |= fn(df).fillna(False).astype(bool)
    return m


def hide_junk(df: pd.DataFrame, on: bool = True) -> pd.DataFrame:
    return df[~junk_mask(df)] if on else df

COUNTRY_USA, COUNTRY_FOREIGN = "USA", "Foreign (ex-USA)"


def option_labels(f: Filter, df: pd.DataFrame | None = None) -> list[str]:
    """Dropdown labels for a filter, "Any" first."""
    if not f.dynamic:
        return [ANY] + [o.label for o in f.options]
    values = [] if df is None else sorted(df[f.column].dropna().astype(str).unique())
    if f.key == "country":
        values = [COUNTRY_USA, COUNTRY_FOREIGN] + [v for v in values if v != "United States"]
    return [ANY] + values


def _mask(f: Filter, label: str, df: pd.DataFrame) -> pd.Series | None:
    if f.dynamic:
        col = df[f.column].astype(str)
        if f.key == "country" and label == COUNTRY_USA:
            return col == "United States"
        if f.key == "country" and label == COUNTRY_FOREIGN:
            return (col != "United States") & df[f.column].notna()
        return col == label
    for o in f.options:
        if o.label == label:
            return o.fn(df).fillna(False).astype(bool)
    return None


def apply_filters(df: pd.DataFrame, selections: dict[str, str]) -> pd.DataFrame:
    """Keep rows matching every active selection. Unknown filters/options are ignored (logged)."""
    mask = pd.Series(True, index=df.index)
    for key, label in selections.items():
        if not label or label == ANY:
            continue
        f = FILTERS_BY_KEY.get(key)
        m = _mask(f, label, df) if f else None
        if m is None:
            log.warning("ignoring unknown filter selection %s=%r", key, label)
            continue
        mask &= m
    return df[mask]


def screen(df: pd.DataFrame, selections: dict[str, str], junk: bool = True) -> pd.DataFrame:
    """The screener pipeline without search: hide junk (when on), then the filter selections."""
    return apply_filters(hide_junk(df, junk), selections)


def active(selections: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in selections.items() if v and v != ANY and k in FILTERS_BY_KEY}


def describe(selections: dict[str, str]) -> list[tuple[str, str]]:
    """(filter label, option) pairs for the active selections — used by exports."""
    return [(FILTERS_BY_KEY[k].label, v) for k, v in active(selections).items()]


def search(df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Ticker list ("AAPL, MSFT", "AAPL MSFT", or one ticker with a comma: "F,") → exact tickers;
    otherwise substring of ticker or company."""
    text = (text or "").strip()
    if not text:
        return df
    parts = [p.upper() for p in text.replace(";", ",").replace(",", " ").split()]
    if "," in text or ";" in text or (len(parts) > 1 and text.isupper()):
        return df[df["ticker"].fillna("").str.upper().isin(parts)]
    t = text.upper()
    return df[df["ticker"].fillna("").str.upper().str.contains(t, regex=False)
              | df["company"].fillna("").str.upper().str.contains(t, regex=False)]
