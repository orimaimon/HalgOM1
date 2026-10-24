"""
Tests for screener/portfolio_risk.py
"""
import re
import numpy as np
import pandas as pd
import pytest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList
from screener.portfolio_risk import by_sector, exposure, summary

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def test_weights_add_to_100_and_closed_position_excluded():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "AAPL", "date": "2025-01-02", "price": 110.0, "quantity": 10, "side": "sell", "fees": 0, "note": ""},
        {"ticker": "MSFT", "date": "2025-01-01", "price": 200.0, "quantity": 5, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "NVDA", "date": "2025-01-01", "price": 300.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "MSFT", "price": 200.0, "company": "Microsoft Corp", "sector": "Technology Services", "beta": 1.1},
        {"ticker": "NVDA", "price": 300.0, "company": "NVIDIA Corp", "sector": "Electronic Technology", "beta": 1.7},
    ])

    exp = exposure(plist, universe)

    assert "AAPL" not in exp["ticker"].values
    assert set(exp["ticker"]) == {"MSFT", "NVDA"}
    assert round(exp["weight"].sum()) == 100


def test_ticker_missing_from_universe_no_price():
    trades = [
        {"ticker": "MSFT", "date": "2025-01-01", "price": 200.0, "quantity": 5, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "XYZ", "date": "2025-01-01", "price": 50.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "MSFT", "price": 200.0, "company": "Microsoft Corp", "sector": "Technology Services", "beta": 1.1},
    ])

    exp = exposure(plist, universe)

    assert exp.attrs["no_price"] == ["XYZ"]
    assert "XYZ" not in exp["ticker"].values
    assert round(exp["weight"].sum()) == 100

    summ = summary(exp)
    assert any("XYZ" in w for w in summ["warnings"])


def test_sector_weights_and_concentration_warning():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 300.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # $3000 -> 30%
        {"ticker": "MSFT", "date": "2025-01-01", "price": 250.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # $2500 -> 25%
        {"ticker": "JNJ", "date": "2025-01-01", "price": 100.0, "quantity": 15, "side": "buy", "fees": 0, "note": ""},   # $1500 -> 15%
        {"ticker": "PG", "date": "2025-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},    # $1500 -> 15%
        {"ticker": "XOM", "date": "2025-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},   # $1500 -> 15%
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
        {"ticker": "MSFT", "price": 250.0, "company": "Microsoft Corp", "sector": "Electronic Technology", "beta": 1.0},
        {"ticker": "JNJ", "price": 100.0, "company": "Johnson & Johnson", "sector": "Health Technology", "beta": 0.6},
        {"ticker": "PG", "price": 150.0, "company": "Procter & Gamble", "sector": "Consumer Non-Durables", "beta": 0.5},
        {"ticker": "XOM", "price": 150.0, "company": "Exxon Mobil", "sector": "Energy Minerals", "beta": 0.9},
    ])

    exp = exposure(plist, universe)
    sec_df = by_sector(exp)

    tech_row = sec_df[sec_df["sector"] == "Electronic Technology"].iloc[0]
    assert pytest.approx(tech_row["weight"], abs=1e-5) == 55.0

    summ_default = summary(exp, max_sector=40.0)
    assert any("טכנולוגיה אלקטרונית" in w and "55%" in w for w in summ_default["warnings"])

    summ_60 = summary(exp, max_sector=60.0)
    assert not any("טכנולוגיה אלקטרונית" in w for w in summ_60["warnings"])


def test_position_concentration_warning_thresholds():
    trades = [
        {"ticker": "A", "date": "2025-01-01", "price": 200.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # 20%
        {"ticker": "B", "date": "2025-01-01", "price": 200.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # 20%
        {"ticker": "C", "date": "2025-01-01", "price": 200.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # 20%
        {"ticker": "D", "date": "2025-01-01", "price": 200.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # 20%
        {"ticker": "E", "date": "2025-01-01", "price": 200.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # 20%
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": t, "price": 200.0, "company": f"Comp {t}", "sector": "Finance", "beta": 1.0}
        for t in ["A", "B", "C", "D", "E"]
    ])

    exp = exposure(plist, universe)

    summ20 = summary(exp, max_position=20.0)
    assert not any("היא 20%" in w for w in summ20["warnings"])

    trades21 = [
        {"ticker": "A", "date": "2025-01-01", "price": 210.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},  # ~21%
        {"ticker": "B", "date": "2025-01-01", "price": 197.5, "quantity": 1, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "C", "date": "2025-01-01", "price": 197.5, "quantity": 1, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "D", "date": "2025-01-01", "price": 197.5, "quantity": 1, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "E", "date": "2025-01-01", "price": 197.5, "quantity": 1, "side": "buy", "fees": 0, "note": ""},
    ]
    plist21 = PList("test_p21", PORTFOLIO, trades21)
    universe21 = pd.DataFrame([
        {"ticker": "A", "price": 210.0, "company": "Comp A", "sector": "Finance", "beta": 1.0},
        {"ticker": "B", "price": 197.5, "company": "Comp B", "sector": "Finance", "beta": 1.0},
        {"ticker": "C", "price": 197.5, "company": "Comp C", "sector": "Finance", "beta": 1.0},
        {"ticker": "D", "price": 197.5, "company": "Comp D", "sector": "Finance", "beta": 1.0},
        {"ticker": "E", "price": 197.5, "company": "Comp E", "sector": "Finance", "beta": 1.0},
    ])

    exp21 = exposure(plist21, universe21)
    summ21 = summary(exp21, max_position=20.0)
    assert any("A היא 21% מהתיק" in w for w in summ21["warnings"])


def test_weighted_beta_calculation_and_partial_missing_beta():
    trades = [
        {"ticker": "A", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # 500 value
        {"ticker": "B", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # 500 value
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "A", "price": 100.0, "company": "Comp A", "sector": "Finance", "beta": 1.2},
        {"ticker": "B", "price": 100.0, "company": "Comp B", "sector": "Finance", "beta": 0.8},
    ])

    exp = exposure(plist, universe)
    summ = summary(exp)
    assert pytest.approx(summ["beta"], abs=1e-5) == 1.0
    assert pytest.approx(summ["beta_coverage"], abs=1e-5) == 100.0

    universe_missing_b = pd.DataFrame([
        {"ticker": "A", "price": 100.0, "company": "Comp A", "sector": "Finance", "beta": 1.2},
        {"ticker": "B", "price": 100.0, "company": "Comp B", "sector": "Finance", "beta": np.nan},
    ])

    exp_m = exposure(plist, universe_missing_b)
    sum_m = summary(exp_m)
    assert pytest.approx(sum_m["beta"], abs=1e-5) == 1.2
    assert pytest.approx(sum_m["beta_coverage"], abs=1e-5) == 50.0


def test_watchlist_returns_empty_frame_and_summary_zero():
    items = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 150.0, "note": ""},
        {"ticker": "MSFT", "date": "2025-01-01", "price": 300.0, "note": ""},
    ]
    plist = PList("test_w", WATCHLIST, items)

    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
        {"ticker": "MSFT", "price": 300.0, "company": "Microsoft Corp", "sector": "Technology Services", "beta": 1.0},
    ])

    exp = exposure(plist, universe)
    assert exp.empty
    assert list(exp.columns) == ["ticker", "company", "sector", "value", "weight", "beta"]

    sec_df = by_sector(exp)
    assert sec_df.empty

    summ = summary(exp)
    assert summ["positions"] == 0
    assert summ["total_value"] == 0.0
    assert summ["beta"] is None
    assert summ["beta_coverage"] is None


def test_warnings_hebrew_content_and_no_nan_none():
    trades = [
        {"ticker": "A", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "NO_PRICE", "date": "2025-01-01", "price": 50.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "A", "price": 100.0, "company": "Comp A", "sector": "Finance", "beta": 1.2},
    ])

    exp = exposure(plist, universe)
    summ = summary(exp, max_position=20.0, max_sector=30.0)

    assert len(summ["warnings"]) > 0
    for w in summ["warnings"]:
        assert HEBREW_RE.search(w) is not None, f"Warning missing Hebrew: {w}"
        assert "nan" not in w.lower(), f"Warning contains nan: {w}"
        assert "none" not in w.lower(), f"Warning contains none: {w}"


def test_no_diversification_warning_for_an_empty_portfolio_and_singular_for_one():
    empty = summary(pd.DataFrame(columns=["ticker", "company", "sector", "value", "weight", "beta"]))
    assert not any("פיזור" in w for w in empty["warnings"])
    one = pd.DataFrame([{"ticker": "AAPL", "company": "Apple", "sector": "Electronic Technology",
                         "value": 1000.0, "weight": 100.0, "beta": 1.1}])
    assert "רק מניה אחת בתיק — פיזור נמוך" in summary(one)["warnings"]
