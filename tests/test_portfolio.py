"""screener/portfolio.py: the watchlist / portfolio file format, positions and stored lists."""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

from screener import portfolio as PF

TODAY = dt.date(2026, 10, 19)
KNOWN = {"AAPL", "MSFT", "BRK.B", "NVDA"}
PRICES = pd.Series({"AAPL": 250.0, "MSFT": 500.0, "BRK.B": 480.0, "NVDA": 180.0})


def csv(text: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    return PF.read_table(text.encode(encoding), "list.csv")


def parse(text: str, **kw) -> PF.Imported:
    return PF.parse(csv(text), known=KNOWN, today=TODAY, **kw)


def test_ticker_only_file_is_a_watchlist():
    imp = parse("ticker,note\naapl,\n BRK-B ,ברקשייר\nAAPL,\n\n")
    assert imp.ok and imp.kind == PF.WATCHLIST
    assert imp.rows["ticker"].tolist() == ["AAPL", "BRK.B"]                 # normalized, deduplicated
    assert imp.rows.loc[1, "note"] == "ברקשייר"


def test_trades_file_is_a_portfolio_with_defaults():
    imp = parse("ticker,date,price,quantity\nAAPL,2025-03-14,213.49,10\nMSFT,02/06/2025,\"$1,461.97\",5\n")
    assert imp.ok and imp.kind == PF.PORTFOLIO
    r = imp.rows
    assert r.loc[1, "date"] == dt.date(2025, 6, 2)                          # day first (Israel)
    assert r.loc[1, "price"] == 1461.97
    assert r["side"].tolist() == ["buy", "buy"] and r["fees"].tolist() == [0.0, 0.0]


def test_hebrew_headers_and_values():
    text = "סימול,תאריך קנייה,שער,כמות,פעולה,עמלה,הערה\nNVDA,14.03.2025,120,10,קנייה,1.5,ניסיון\n"
    for encoding in ("utf-8-sig", "cp1255"):                               # also old Excel's Hebrew CSV
        imp = PF.parse(csv(text, encoding), KNOWN, TODAY)
        assert imp.ok, imp.problems
        assert imp.rows.loc[0, "fees"] == 1.5 and imp.rows.loc[0, "note"] == "ניסיון"


def test_excel_dates_and_numbers(tmp_path):
    df = pd.DataFrame({"Ticker": ["AAPL"], "Date": [pd.Timestamp("2025-03-14")], "Price": [213.49],
                       "Quantity": [10]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    imp = PF.parse(PF.read_table(buf.getvalue(), "תיק.xlsx"), KNOWN, TODAY)
    assert imp.ok and imp.rows.loc[0, "date"] == dt.date(2025, 3, 14) and imp.rows.loc[0, "quantity"] == 10


def test_problems_point_at_the_excel_row():
    imp = parse("ticker,date,price,quantity,side\n"
                "AAPL,2025-03-14,213,10,buy\n"        # row 2: fine
                "XYZQ,2025-03-14,10,1,buy\n"          # row 3: not in the universe
                "MSFT,14/13/2025,10,1,buy\n"          # row 4: no such date
                "MSFT,2027-01-01,10,1,buy\n"          # row 5: future
                "MSFT,2025-01-01,-3,1,buy\n"          # row 6: negative price
                "MSFT,2025-01-01,10,,buy\n"           # row 7: no quantity
                "MSFT,2025-01-01,10,1,hold\n"         # row 8: unknown side
                "$$$,2025-01-01,10,1,buy\n")          # row 9: not a ticker
    got = {(p.row, p.code) for p in imp.problems}
    assert got == {(3, "unknown_ticker"), (4, "bad_date"), (5, "future_date"), (6, "bad_number"),
                   (7, "missing_value"), (8, "bad_side"), (9, "bad_ticker")}
    assert not imp.ok and imp.rows["ticker"].tolist() == ["AAPL"]


def test_missing_ticker_column_and_oversell():
    assert parse("name,price\nApple,1\n").problems == [PF.Problem(1, "no_ticker_column")]
    imp = parse("ticker,date,price,quantity,side\nAAPL,2025-01-02,100,5,buy\nAAPL,2025-02-02,120,8,sell\n")
    assert PF.Problem(0, "oversold", "AAPL") in imp.problems


def trades(rows):
    df = pd.DataFrame(rows, columns=["ticker", "date", "price", "quantity", "side", "fees"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def test_positions_average_cost_and_realized():
    t = trades([("AAPL", "2025-01-02", 100, 10, "buy", 10),      # cost 1010
                ("AAPL", "2025-06-02", 200, 10, "buy", 10),      # cost 2010 -> 20 sh, 3020, avg 151
                ("AAPL", "2025-09-02", 250, 5, "sell", 5),       # proceeds 1245 - 5*151 = 490 realized
                ("MSFT", "2025-01-02", 400, 1, "buy", 0),
                ("MSFT", "2025-03-02", 450, 1, "sell", 0)])      # closed: kept, realized 50
    p = PF.positions(t).set_index("ticker")
    assert p.loc["AAPL", "quantity"] == 15 and p.loc["AAPL", "avg_cost"] == pytest.approx(151)
    assert p.loc["AAPL", "cost"] == pytest.approx(2265) and p.loc["AAPL", "realized"] == pytest.approx(490)
    assert p.loc["MSFT", "quantity"] == 0 and p.loc["MSFT", "realized"] == pytest.approx(50)
    assert p.loc["AAPL", "first_buy"] == dt.date(2025, 1, 2)


def test_valuation_pnl_weights_and_annual_return():
    t = trades([("AAPL", "2025-10-19", 200, 10, "buy", 0),       # one year before TODAY: +25%
                ("MSFT", "2026-10-01", 400, 5, "buy", 0),        # 18 days: no annual figure
                ("ZZZ", "2026-01-02", 10, 1, "buy", 0)])         # not in the universe: no price
    v = PF.valuation(PF.positions(t), PRICES, TODAY).set_index("ticker")
    assert v.loc["AAPL", "value"] == 2500 and v.loc["AAPL", "pnl"] == 500
    assert v.loc["AAPL", "pnl_pct"] == pytest.approx(25)
    assert v.loc["AAPL", "annual_pct"] == pytest.approx(25, abs=0.1)
    assert np.isnan(v.loc["MSFT", "annual_pct"]) and np.isnan(v.loc["ZZZ", "value"])
    assert v.loc["AAPL", "weight"] + v.loc["MSFT", "weight"] == pytest.approx(100)    # 2500 + 2500
    assert v.loc["AAPL", "weight"] == pytest.approx(50)


def test_lists_round_trip_and_watchlist_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(PF, "PORTFOLIOS_PATH", tmp_path / "portfolios.json")
    w = PF.from_imported("מעקב", parse("ticker\nAAPL\nNVDA\n"), PRICES * 0.8, TODAY)
    assert PF.add_to_watchlist(w, "MSFT", 500.0, TODAY) and not PF.add_to_watchlist(w, "AAPL", 1.0)
    PF.save(w)
    p = PF.from_imported("תיק", parse("ticker,date,price,quantity\nAAPL,2025-03-14,213.49,10\n"), PRICES)
    PF.save(p)

    lists = PF.load_all()
    assert set(lists) == {"מעקב", "תיק"} and lists["תיק"].kind == PF.PORTFOLIO
    perf = PF.watch_performance(lists["מעקב"].frame(), PRICES).set_index("ticker")
    assert perf.loc["AAPL", "since_added_pct"] == pytest.approx(25)       # added at 80% of today's price
    assert perf.loc["MSFT", "since_added_pct"] == pytest.approx(0)

    again = PF.parse(PF.read_table(PF.to_csv(lists["תיק"]), "x.csv"), KNOWN, TODAY)
    assert again.ok and again.rows.loc[0, "date"] == dt.date(2025, 3, 14)  # export -> import is lossless
    PF.delete("מעקב")
    assert set(PF.load_all()) == {"תיק"}


def test_damaged_file_and_template(tmp_path):
    bad = tmp_path / "portfolios.json"
    bad.write_text("{not json", encoding="utf-8")
    assert PF.load_all(bad) == {}
    imp = PF.parse(PF.read_table(PF.template_csv(), "template.csv"), KNOWN, TODAY)
    assert imp.ok and imp.kind == PF.PORTFOLIO and len(imp.rows) == 2
