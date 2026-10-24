import datetime as dt
import pandas as pd
import pytest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList, positions
from screener.trade_journal import CLOSED_COLUMNS, by_sector_pnl, closed_trades, journal_summary


def test_multi_trade_aapl_matching_positions_realized():
    # Scenario: buy 10 AAPL at $100 (fee $0), sell 4 at $125 after 20 days (+25%, +$100), sell 6 at $90 after 40 days (-10%, -$60)
    d0 = dt.date(2025, 1, 1)
    d1 = d0 + dt.timedelta(days=20)
    d2 = d0 + dt.timedelta(days=40)

    rows = [
        {"ticker": "AAPL", "date": d0, "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": "buy 10"},
        {"ticker": "AAPL", "date": d1, "price": 125.0, "quantity": 4, "side": "sell", "fees": 0.0, "note": "sell 4"},
        {"ticker": "AAPL", "date": d2, "price": 90.0, "quantity": 6, "side": "sell", "fees": 0.0, "note": "sell 6"},
    ]
    plist = PList(name="Test AAPL", kind=PORTFOLIO, rows=rows)
    universe = pd.DataFrame([{"ticker": "AAPL", "sector": "Technology"}])

    c_df = closed_trades(plist, universe)
    assert not c_df.empty
    assert list(c_df.columns) == CLOSED_COLUMNS

    # Match positions(df)["realized"].sum() to the cent
    pos_df = positions(plist.frame())
    total_realized_positions = pos_df["realized"].sum()
    total_realized_closed = c_df["pnl"].sum()

    assert round(total_realized_closed, 2) == round(total_realized_positions, 2)
    assert round(total_realized_closed, 2) == 40.0

    # Verify rows sorted by date descending (newest sell d2 first)
    assert c_df.iloc[0]["date"] == d2
    assert c_df.iloc[0]["quantity"] == 6
    assert c_df.iloc[0]["pnl"] == -60.0
    assert c_df.iloc[0]["pnl_pct"] == -10.0
    assert c_df.iloc[0]["hold_days"] == 40
    assert c_df.iloc[0]["sector"] == "Technology"

    assert c_df.iloc[1]["date"] == d1
    assert c_df.iloc[1]["quantity"] == 4
    assert c_df.iloc[1]["pnl"] == 100.0
    assert c_df.iloc[1]["pnl_pct"] == 25.0
    assert c_df.iloc[1]["hold_days"] == 20
    assert c_df.iloc[1]["sector"] == "Technology"


def test_partial_buys_weighted_avg_cost_and_hold_days():
    # Buy 10 AAPL @ $100 on d0 (spent $1000)
    # Buy 10 AAPL @ $200 on d10 (spent $2000) -> total 20 AAPL @ avg_cost $150 (spent $3000)
    # Cost-weighted avg buy date: (1000 * 0 + 2000 * 10) / 3000 = 6.67 -> rounded to 7 days after d0.
    # Sell 10 AAPL @ $180 on d30 (d30 - d7 = 23 hold_days).
    d0 = dt.date(2025, 1, 1)
    d10 = d0 + dt.timedelta(days=10)
    d30 = d0 + dt.timedelta(days=30)

    rows = [
        {"ticker": "AAPL", "date": d0, "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "AAPL", "date": d10, "price": 200.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "AAPL", "date": d30, "price": 180.0, "quantity": 10, "side": "sell", "fees": 0.0, "note": "sell half"},
    ]
    plist = PList(name="Test AAPL Avg", kind=PORTFOLIO, rows=rows)
    c_df = closed_trades(plist)

    assert len(c_df) == 1
    row = c_df.iloc[0]
    assert row["avg_cost"] == 150.0
    assert row["cost_basis"] == 1500.0
    assert row["proceeds"] == 1800.0
    assert row["pnl"] == 300.0
    assert row["pnl_pct"] == 20.0
    assert row["hold_days"] == 23


def test_journal_summary_stats():
    # Building a closed_df with win_rate 50%, total_pnl 40.0, avg_win_pct 25.0, avg_loss_pct -10.0, payoff_ratio 2.5, profit_factor 100/60
    d0 = dt.date(2025, 1, 1)
    d1 = d0 + dt.timedelta(days=20)
    d2 = d0 + dt.timedelta(days=40)

    rows = [
        {"ticker": "AAPL", "date": d0, "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "AAPL", "date": d1, "price": 125.0, "quantity": 4, "side": "sell", "fees": 0.0, "note": ""},
        {"ticker": "AAPL", "date": d2, "price": 90.0, "quantity": 6, "side": "sell", "fees": 0.0, "note": ""},
    ]
    plist = PList(name="Test Summary", kind=PORTFOLIO, rows=rows)
    c_df = closed_trades(plist)

    summary = journal_summary(c_df)
    assert summary["trades"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 50.0
    assert summary["total_pnl"] == 40.0
    assert summary["avg_win_pct"] == 25.0
    assert summary["avg_loss_pct"] == -10.0
    assert summary["payoff_ratio"] == 2.5
    assert pytest.approx(summary["profit_factor"], 1e-5) == 100.0 / 60.0
    assert pytest.approx(summary["avg_hold_days"], 1e-5) == 30.0
    assert len(summary["insights"]) > 0


def test_all_wins_no_division_by_zero():
    d0 = dt.date(2025, 1, 1)
    d1 = d0 + dt.timedelta(days=10)

    rows = [
        {"ticker": "MSFT", "date": d0, "price": 200.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "MSFT", "date": d1, "price": 250.0, "quantity": 10, "side": "sell", "fees": 0.0, "note": ""},
    ]
    plist = PList(name="All Wins", kind=PORTFOLIO, rows=rows)
    c_df = closed_trades(plist)

    summary = journal_summary(c_df)
    assert summary["trades"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["win_rate"] == 100.0
    assert summary["total_pnl"] == 500.0
    assert summary["avg_win_pct"] == 25.0
    assert summary["avg_loss_pct"] is None
    assert summary["payoff_ratio"] is None
    assert summary["profit_factor"] is None


def test_empty_portfolio_or_watchlist():
    # Watchlist
    w_plist = PList(name="Watchlist", kind=WATCHLIST, rows=[{"ticker": "AAPL", "date": "2025-01-01", "price": 100}])
    c_df_w = closed_trades(w_plist)
    assert c_df_w.empty
    assert list(c_df_w.columns) == CLOSED_COLUMNS
    assert journal_summary(c_df_w)["trades"] == 0

    # Empty portfolio
    p_plist = PList(name="Empty", kind=PORTFOLIO, rows=[])
    c_df_p = closed_trades(p_plist)
    assert c_df_p.empty
    assert list(c_df_p.columns) == CLOSED_COLUMNS
    summary = journal_summary(c_df_p)
    assert summary == {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "total_pnl": 0.0,
        "avg_win_pct": None,
        "avg_loss_pct": None,
        "payoff_ratio": None,
        "profit_factor": None,
        "avg_hold_days": None,
        "insights": [],
    }


def test_by_sector_pnl():
    d0 = dt.date(2025, 1, 1)
    d1 = d0 + dt.timedelta(days=10)

    rows = [
        {"ticker": "AAPL", "date": d0, "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "AAPL", "date": d1, "price": 120.0, "quantity": 10, "side": "sell", "fees": 0.0, "note": ""},
        {"ticker": "XOM", "date": d0, "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        {"ticker": "XOM", "date": d1, "price": 80.0, "quantity": 10, "side": "sell", "fees": 0.0, "note": ""},
    ]
    plist = PList(name="Sectors", kind=PORTFOLIO, rows=rows)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology"},
        {"ticker": "XOM", "sector": "Energy"},
    ])
    c_df = closed_trades(plist, universe)
    sec_df = by_sector_pnl(c_df)

    assert len(sec_df) == 2
    assert list(sec_df.columns) == ["sector", "trades", "win_rate", "pnl"]
    assert sec_df.iloc[0]["sector"] == "Technology"
    assert sec_df.iloc[0]["pnl"] == 200.0
    assert sec_df.iloc[0]["win_rate"] == 100.0

    assert sec_df.iloc[1]["sector"] == "Energy"
    assert sec_df.iloc[1]["pnl"] == -200.0
    assert sec_df.iloc[1]["win_rate"] == 0.0

    # Empty closed_df
    empty_sec = by_sector_pnl(pd.DataFrame(columns=CLOSED_COLUMNS))
    assert empty_sec.empty
    assert list(empty_sec.columns) == ["sector", "trades", "win_rate", "pnl"]
