"""Tests for screener/explain.py."""
import re
import numpy as np
import pandas as pd
import pytest

from screener import explain, i18n
from screener.derive import add_derived
from tests.test_screener import TODAY, make_raw


def make_test_row(**kwargs) -> pd.Series:
    raw = make_raw(n=1, seed=42)
    df = add_derived(raw, today=TODAY)
    row = df.iloc[0].copy()
    for k, v in kwargs.items():
        row[k] = v
    return row


def test_explain_two_selections_one_pass_one_fail():
    row = make_test_row(
        price=150.0,
        pe=12.0,
        market_cap=2.1e9,
        change=12.3,
    )
    # pe: Under 15 -> pass; price: Under 50 -> fail
    selections = {"pe": "Under 15", "price": "Under $50"}
    res = explain.explain(row, selections, lang=i18n.HE)

    assert len(res) == 2
    pe_item = next(x for x in res if x["key"] == "pe")
    price_item = next(x for x in res if x["key"] == "price")

    assert pe_item["passes"] is True
    assert pe_item["filter"] == i18n.filter_label("pe", "P/E", i18n.HE)
    assert pe_item["option"] == i18n.option("Under 15", i18n.HE)
    assert pe_item["value"] == "12.00" or pe_item["value"] == "12" or "12" in pe_item["value"]

    assert price_item["passes"] is False
    assert price_item["value"] == "$150.00"


def test_explain_formatted_values():
    row = make_test_row(
        market_cap=2.1e9,
        change=12.3,
        pe=10.0,
    )
    selections = {"market_cap": "+Mid (over $2B)", "change": "Up 10%+"}
    res = explain.explain(row, selections, lang=i18n.HE)

    cap_item = next(x for x in res if x["key"] == "market_cap")
    chg_item = next(x for x in res if x["key"] == "change")

    assert cap_item["value"] == "$2.1B"
    assert chg_item["value"] == "+12.3%"


def test_explain_inactive_selection_ignored_and_nan_value():
    row = make_test_row(pe=np.nan, price=100.0)
    selections = {"pe": "Under 15", "price": "Any", "sector": ""}

    res = explain.explain(row, selections, lang=i18n.HE)

    assert len(res) == 1
    item = res[0]
    assert item["key"] == "pe"
    assert item["value"] == "—"
    assert item["passes"] is False


def test_explain_dynamic_filter_sector():
    row = make_test_row(sector="Technology Services")
    selections = {"sector": "Technology Services"}

    res = explain.explain(row, selections, lang=i18n.HE)

    assert len(res) == 1
    item = res[0]
    assert item["key"] == "sector"
    assert item["passes"] is True
    assert item["value"] == i18n.value("sector", "Technology Services", i18n.HE)
    assert item["value"] == "שירותי טכנולוגיה"


def test_describe_large_profitable_tech_stock():
    row = make_test_row(
        market_cap=250e9,
        sector="Electronic Technology",
        price=150.0,
        sma200=100.0,
        sma200_pct=50.0,
        high52_pct=-2.0,
        sales_growth=25.0,
        net_margin=15.0,
        div_yield=3.5,
        pe_vs_industry=-25.0,
        score=85.0,
    )

    desc = explain.describe(row)

    assert "חברה ענקית (שווי $250.0B) בסקטור טכנולוגיה אלקטרונית." in desc
    assert "נסחרת במגמת עלייה מעל ממוצע 200 וקרובה לשיא השנתי." in desc
    assert "צמיחה חזקה בהכנסות (25%) ורווחית." in desc
    assert "מציעה תשואת דיבידנד של 3.5%." in desc
    assert "זולה ביחס לתעשייה (P/E נמוך ב-25% מהחציון)." in desc
    assert "ציון HalgOM גבוה (85)." in desc


def test_describe_nans_row_returns_empty_string():
    row = pd.Series({"market_cap": np.nan, "price": np.nan, "sector": np.nan})
    desc = explain.describe(row)
    assert desc == ""


def test_describe_without_pe_vs_industry():
    row = make_test_row(
        market_cap=250e9,
        sector="Electronic Technology",
        price=150.0,
        sma200=100.0,
        high52_pct=-2.0,
        sales_growth=25.0,
        net_margin=15.0,
        div_yield=3.5,
        score=85.0,
    )
    if "pe_vs_industry" in row:
        row = row.drop(labels=["pe_vs_industry"])

    desc = explain.describe(row)
    assert "זולה ביחס לתעשייה" not in desc
    assert desc != ""


def test_every_describe_sentence_contains_hebrew_and_no_nan_none():
    HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

    test_rows = [
        make_test_row(market_cap=250e9, sector="Technology Services", price=100, sma200=80, high52_pct=-1),
        make_test_row(market_cap=15e9, sector="Finance", price=50, sma200=60, high52_pct=-10, sales_growth=30, net_margin=-5),
        make_test_row(market_cap=5e9, div_yield=4.5, pe_vs_industry=-30, score=90),
        make_test_row(market_cap=500e6, net_margin=10),
        make_test_row(market_cap=10e6, sales_growth=20),
    ]

    for r in test_rows:
        desc = explain.describe(r)
        if desc:
            assert HEBREW_RE.search(desc), f"No Hebrew in: {desc}"
            assert "nan" not in desc.lower(), f"'nan' found in: {desc}"
            assert "none" not in desc.lower(), f"'none' found in: {desc}"
