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
    assert "מציעה תשואת דיבידנד של 3.5%." in desc  # one decimal
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


def test_describe_rounds_real_world_numbers():
    row = make_test_row(market_cap=15e9, sales_growth=25.123456, net_margin=7.5, div_yield=3.48712,
                        pe_vs_industry=-31.66666, score=84.6)
    desc = explain.describe(row)
    assert "(25%)" in desc and "3.5%" in desc and "ב-32%" in desc and "(85)" in desc
    assert not re.search(r"\d\.\d{2,}", desc), desc


def test_explain_stock_match_value_and_momentum():
    row = make_test_row(
        pe=12.3,
        sma200_pct=8.5,
        price=108.5,
        sma200=100.0,
    )
    active_filters = {
        "pe": "Under 15",
        "sma200": "Price above SMA200",
    }
    bullets = explain.explain_stock_match("AAPL", row, active_filters)

    assert len(bullets) == 2
    assert any("P/E 12.3 נמוך מ-15" in b for b in bullets)
    assert any("מעל ממוצע 200 ב-8.5%" in b for b in bullets)


def test_explain_stock_match_multi_factor_and_dynamic():
    row = make_test_row(
        pe=12.0,
        pe_vs_industry=-25.0,
        sales_growth=18.0,
        rel_volume=2.5,
        sector="Technology Services",
        price=150.0,
    )
    active_filters = {
        "pe": "Under 15",
        "pe_vs_industry": "20%+ below industry",
        "sales_growth": "High (>25%)",  # fail (18% < 25%)
        "vol_spike": "Volume 2x average",
        "sector": "Technology Services",
        "price": "Under $50",  # fail (150 > 50)
    }
    bullets = explain.explain_stock_match("NVDA", row, active_filters)

    assert len(bullets) == 4
    assert any("P/E 12 נמוך מ-15" in b for b in bullets)
    assert any("P/E נמוך ב-25% מהחציון בתעשייה" in b for b in bullets)
    assert any("מחזור חריג פי 2.5 מהממוצע" in b for b in bullets)
    assert any("סקטור שירותי טכנולוגיה" in b for b in bullets)


def test_explain_stock_match_empty_filters_or_no_match():
    row = make_test_row(price=100.0)
    assert explain.explain_stock_match("AAPL", row, {}) == []
    assert explain.explain_stock_match("AAPL", row, {"price": "Under $50"}) == []


def test_stock_summary_card_text_strong_factors():
    # Value + Momentum + Growth
    row_all = make_test_row(
        score_value=85.0,
        score_momentum=80.0,
        sales_growth=25.0,
    )
    card_all = explain.stock_summary_card_text(row_all)
    assert "ערך" in card_all and "מומנטום" in card_all and "צמיחה" in card_all

    # Value + Growth
    row_vg = make_test_row(
        score_value=75.0,
        score_momentum=50.0,
        sales_growth=20.0,
        sma200_pct=-5.0,
    )
    card_vg = explain.stock_summary_card_text(row_vg)
    assert "ערך" in card_vg and "צמיחה" in card_vg

    # Momentum
    row_m = make_test_row(
        score_value=40.0,
        score_momentum=80.0,
        sales_growth=5.0,
        pe=40.0,
    )
    card_m = explain.stock_summary_card_text(row_m)
    assert "מומנטום" in card_m


def test_stock_summary_card_text_nan_and_empty_edge_cases():
    assert explain.stock_summary_card_text(None) == ""
    assert explain.stock_summary_card_text(pd.Series(dtype=object)) == ""

    row_nan = pd.Series({
        "score_value": np.nan,
        "score_momentum": np.nan,
        "market_cap": 250e9,
        "price": 100.0,
        "sma200_pct": 10.0,
    })
    card_nan = explain.stock_summary_card_text(row_nan)
    assert isinstance(card_nan, str)
