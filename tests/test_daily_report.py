"""
Tests for screener/daily_report.py.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import pandas as pd
import pytest

from screener.alerts import Fired
from screener.daily_report import Report, SPECIAL_CHARS, compose, escape_markdown_v2
from screener.home import Morning
from screener.market import Regime


def test_escape_markdown_v2_each_special_char():
    for char in SPECIAL_CHARS:
        escaped = escape_markdown_v2(char)
        assert escaped == f"\\{char}", f"Expected \\{char} for {char}, got {escaped}"


def test_escape_markdown_v2_roundtrip_sanity():
    def unescape(s: str) -> str:
        res = []
        i = 0
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                res.append(s[i + 1])
                i += 2
            else:
                res.append(s[i])
                i += 1
        return "".join(res)

    test_str = "Hello_World! Price: $100 + 5% [AAPL] (MSFT) ~test~ `code` > #tag = equal | bar {dict} . dot \\slash"
    escaped = escape_markdown_v2(test_str)
    assert unescape(escaped) == test_str


def test_full_report():
    day = dt.date(2026, 10, 23)

    df_screens = pd.DataFrame([
        {"screen": "★ My Saved Screen", "total": 10, "new": 2},
        {"screen": "Momentum leaders", "total": 12, "new": 3},
        {"screen": "Value — cheap & profitable", "total": 5, "new": 0},
    ])

    df_movers = pd.DataFrame([
        {"ticker": "NVDA", "company": "NVIDIA", "price": 135.0, "change": 4.2, "lists": ["תיק ראשי"]},
        {"ticker": "TSLA", "company": "Tesla", "price": 220.0, "change": -3.0, "lists": ["מעקב", "תיק משני"]},
    ])

    df_earnings = pd.DataFrame([
        {"ticker": "TSLA", "company": "Tesla", "earnings_date": dt.date(2026, 10, 23), "days": 0},
        {"ticker": "MSFT", "company": "Microsoft", "earnings_date": dt.date(2026, 10, 24), "days": 1},
        {"ticker": "AAPL", "company": "Apple", "earnings_date": dt.date(2026, 10, 26), "days": 3},
    ])

    alerts = [
        Fired(alert_id="a1", ticker="NVDA", message="NVDA: המחיר 135.00 עלה מעל ל-130.00"),
    ]

    morning = Morning(
        new_in_screens=df_screens,
        movers=df_movers,
        earnings=df_earnings,
        alerts=alerts,
    )

    regime = Regime(
        label="bull",
        reasons=["VIX 15.2 (<18 calm)", "SPY above SMA200"],
    )

    report = compose(morning, regime, fired=None, day=day)

    assert report.title == "הבוקר שלי · 23/10/2026"
    assert len(report.sections) == 5

    headings = [sec[0] for sec in report.sections]
    assert headings == ["מצב השוק", "חדשות בסריקות", "תנועות בולטות", "דוחות קרובים", "התראות"]

    # Market section
    mkt_items = report.sections[0][1]
    assert len(mkt_items) == 1
    assert "שוק עולה" in mkt_items[0]
    assert "VIX 15.2 (<18 רגוע)" in mkt_items[0]
    assert "SPY מעל SMA200" in mkt_items[0]

    # Screens section
    scr_items = report.sections[1][1]
    assert len(scr_items) == 2
    assert scr_items[0] == "מובילות מומנטום: 3 חדשות (מתוך 12)"
    assert scr_items[1] == "★ My Saved Screen: 2 חדשות (מתוך 10)"

    # Movers section
    mov_items = report.sections[2][1]
    assert mov_items[0] == "NVDA +4.2% (תיק ראשי)"
    assert mov_items[1] == "TSLA -3.0% (מעקב, תיק משני)"

    # Earnings section
    earn_items = report.sections[3][1]
    assert earn_items[0] == "TSLA — דוח היום (23/10)"
    assert earn_items[1] == "MSFT — דוח מחר (24/10)"
    assert earn_items[2] == "AAPL — דוח בעוד 3 ימים (26/10)"

    # Alerts section
    alt_items = report.sections[4][1]
    assert alt_items[0] == "NVDA: המחיר 135.00 עלה מעל ל-130.00"

    # Plain text
    assert report.title in report.text
    assert "נתונים בעיכוב של כ-15 דקות. אינו ייעוץ השקעות." in report.text

    # Telegram MarkdownV2
    assert "*הבוקר שלי · 23/10/2026*" in report.telegram
    assert "*מצב השוק*" in report.telegram
    assert "\\+4\\.2%" in report.telegram
    assert "\\-3\\.0%" in report.telegram


def test_every_new_zero():
    day = dt.date(2026, 10, 23)
    df_screens = pd.DataFrame([
        {"screen": "★ My Screen", "total": 10, "new": 0},
        {"screen": "Momentum leaders", "total": 12, "new": 0},
    ])
    morning = Morning(
        new_in_screens=df_screens,
        movers=pd.DataFrame(),
        earnings=pd.DataFrame(),
        alerts=[],
    )

    report = compose(morning, regime=None, fired=[], day=day)
    assert len(report.sections) == 1
    heading, items = report.sections[0]
    assert heading == "חדשות בסריקות"
    assert items == ["אין מניות חדשות בסריקות"]
    assert "אין מניות חדשות בסריקות" in report.text


def test_all_sections_empty():
    day = dt.date(2026, 10, 23)
    morning = Morning(
        new_in_screens=pd.DataFrame(),
        movers=pd.DataFrame(),
        earnings=pd.DataFrame(),
        alerts=[],
    )

    report = compose(morning, regime=None, fired=[], day=day)
    assert report.sections == []
    assert "אין חדש היום" in report.text
    assert "אין חדש היום" in report.telegram


def test_regime_none_omits_market_section():
    day = dt.date(2026, 10, 23)
    morning = Morning(
        new_in_screens=pd.DataFrame([{"screen": "★ My Screen", "total": 10, "new": 1}]),
        movers=pd.DataFrame(),
        earnings=pd.DataFrame(),
        alerts=[],
    )

    report = compose(morning, regime=None, fired=[], day=day)
    headings = [sec[0] for sec in report.sections]
    assert "מצב השוק" not in headings


def test_telegram_500_movers_truncation():
    day = dt.date(2026, 10, 23)

    movers_data = [
        {
            "ticker": f"TICK{i}",
            "company": f"Company {i}",
            "price": 100.0 + i,
            "change": 1.0 + (i % 10),
            "lists": ["תיק ראשי"],
        }
        for i in range(500)
    ]
    df_movers = pd.DataFrame(movers_data)

    morning = Morning(
        new_in_screens=pd.DataFrame(),
        movers=df_movers,
        earnings=pd.DataFrame(),
        alerts=[],
    )

    report = compose(morning, regime=None, fired=[], day=day, top_movers=500)

    assert len(report.telegram) <= 4096
    assert not report.telegram.endswith("\\")
    assert re.search(r"…ועוד \d+", report.telegram) is not None


def test_nan_movers_handling():
    day = dt.date(2026, 10, 23)

    df_movers = pd.DataFrame([
        {"ticker": "GOOD", "company": "Good Co", "price": 100.0, "change": 5.0, "lists": ["תיק ראשי"]},
        {"ticker": "BADCHG1", "company": "Bad Chg 1", "price": 50.0, "change": math.nan, "lists": ["תיק ראשי"]},
        {"ticker": "BADCHG2", "company": "Bad Chg 2", "price": 50.0, "change": None, "lists": ["תיק ראשי"]},
        {"ticker": "BADPRC", "company": "Bad Price", "price": math.nan, "change": -2.5, "lists": ["תיק ראשי"]},
    ])

    morning = Morning(
        new_in_screens=pd.DataFrame(),
        movers=df_movers,
        earnings=pd.DataFrame(),
        alerts=[],
    )

    report = compose(morning, regime=None, fired=[], day=day)

    # Confirm "nan" or "None" string doesn't leak into report text or telegram
    assert "nan" not in report.text.lower()
    assert "none" not in report.text.lower()

    mov_items = [sec[1] for sec in report.sections if sec[0] == "תנועות בולטות"][0]
    # GOOD (+5.0%) and BADPRC (-2.5%) are included; BADCHG1 and BADCHG2 are skipped.
    assert len(mov_items) == 2
    assert "GOOD +5.0% (תיק ראשי)" in mov_items[0]
    assert "BADPRC -2.5% (תיק ראשי)" in mov_items[1]
