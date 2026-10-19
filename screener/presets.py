"""
Screens: built-in presets + user-saved presets (stored in _data/presets.json).

A preset = filter selections ({filter key: option label}) + the table view to open.
Option labels must match filters.py exactly — tests/test_screener.py checks the built-ins.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from screener.data.store import DATA_DIR

log = logging.getLogger(__name__)

PRESETS_PATH = DATA_DIR / "presets.json"


@dataclass
class Preset:
    filters: dict[str, str]
    view: str = "Overview"
    sort: str | None = None
    ascending: bool = False
    note: str = field(default="")


_LIQUID = {"avg_volume": "Over 500K", "price": "Over $5"}

BUILTIN: dict[str, Preset] = {
    "Value — cheap & profitable": Preset(
        {"pe": "Under 15", "roe": "Over 10%", "debt_eq": "Under 1", "market_cap": "+Small (over $300M)",
         "avg_volume": "Over 200K"},
        view="Valuation", sort="pe", ascending=True,
        note="מניות זולות (P/E מתחת ל-15) של חברות רווחיות (ROE מעל 10%) עם חוב סביר"),
    "Quality growth": Preset(
        {"eps_growth": "Over 20%", "sales_growth_q": "Over 15%", "roe": "Over 15%",
         "net_margin": "Positive (>0%)", "market_cap": "+Small (over $300M)", **_LIQUID},
        view="Financial", sort="sales_growth_q",
        note="חברות רווחיות שהרווח וההכנסות שלהן צומחים מהר"),
    "Momentum leaders": Preset(
        {"perf_h": "Up 20%+", "sma50": "Price above SMA50", "sma200": "Price above SMA200",
         "high52": "0-10% below High", "price": "Over $10", "avg_volume": "Over 500K"},
        view="Performance", sort="perf_h",
        note="מניות במגמת עלייה חזקה: מעל הממוצעים הנעים וקרובות לשיא השנתי"),
    "Oversold in uptrend": Preset(
        {"rsi": "Oversold (40)", "sma200": "Price above SMA200", "roe": "Positive (>0%)",
         "market_cap": "+Mid (over $2B)", "avg_volume": "Over 500K"},
        view="Technical", sort="rsi", ascending=True,
        note="ירידה זמנית (RSI נמוך) של מניות גדולות שעדיין במגמת עלייה ארוכה"),
    "52-week high breakout": Preset(
        {"high52": "0-3% below High", "rel_volume": "Over 1.5", **_LIQUID},
        view="Technical", sort="rel_volume",
        note="מניות שנוגעות בשיא השנתי עם מחזור גבוה מהרגיל"),
    "Dividend — stable": Preset(
        {"div_yield": "Over 3%", "payout": "Under 80%", "debt_eq": "Under 1", "market_cap": "+Mid (over $2B)",
         "roe": "Positive (>0%)"},
        view="Financial", sort="div_yield",
        note="דיבידנד מעל 3% שהחברה יכולה להרשות לעצמה (מחלקת פחות מ-80% מהרווח)"),
    "Small-cap growth": Preset(
        {"market_cap": "Small ($300M to $2B)", "sales_growth_q": "Over 30%", "eps_growth_q": "Positive (>0%)",
         "avg_volume": "Over 200K", "price": "Over $5"},
        view="Overview", sort="market_cap",
        note="חברות קטנות שההכנסות שלהן צומחות מעל 30% ברבעון"),
    "Unusual volume": Preset(
        {"rel_volume": "Over 3", "avg_volume": "Over 200K", "price": "Over $5"},
        view="Overview", sort="rel_volume",
        note="מחזור פי 3 ויותר מהרגיל — משהו קורה במניה"),
    "Stage 2 (Minervini)": Preset(
        {"ma_stack": "Price > SMA50 > SMA150 > SMA200 (Stage 2)", "low52": "30% or more above Low",
         "high52": "0-25% below High"},
        view="Technical", sort="perf_h",
        note="שלב 2 לפי Minervini: מחיר מעל SMA50 שמעל SMA150 שמעל SMA200, לפחות 30% מעל השפל השנתי "
             "ובטווח של 25% מהשיא. לא כולל את תנאי ה-RS ואת שיפוע ה-SMA200 מהשיטה המלאה"),
    "HalgOM top score": Preset(
        {"score": "Over 70", "earnings": "Not in next 7 days", **_LIQUID},
        view="Scores", sort="score",
        note="הציון המשוקלל הגבוה ביותר, בלי מניות שיוצא להן דוח השבוע"),
}


def load_user(path: Path | None = None) -> dict[str, Preset]:
    path = path or PRESETS_PATH
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {name: Preset(**p) for name, p in raw.items()}
    except Exception as e:
        log.warning("cannot read user presets: %s", e)
        return {}


def save_user(name: str, preset: Preset, path: Path | None = None) -> None:
    path = path or PRESETS_PATH
    presets = load_user(path)
    presets[name] = preset
    _write(presets, path)


def delete_user(name: str, path: Path | None = None) -> None:
    path = path or PRESETS_PATH
    presets = load_user(path)
    if presets.pop(name, None) is not None:
        _write(presets, path)


def _write(presets: dict[str, Preset], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({n: asdict(p) for n, p in presets.items()}, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)               # a crash mid-write never leaves a half-written file
