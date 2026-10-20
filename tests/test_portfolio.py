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
    assert not bad.exists()
    bad_files = list(tmp_path.glob("portfolios.json.bad-*"))
    assert len(bad_files) == 1
    assert bad_files[0].read_text(encoding="utf-8") == "{not json"
    imp = PF.parse(PF.read_table(PF.template_csv(), "template.csv"), KNOWN, TODAY)
    assert imp.ok and imp.kind == PF.PORTFOLIO and len(imp.rows) == 2


def test_us_and_il_date_formats():
    # US file: month-first detected because 14 is in 2nd position (03/14/2025 -> March 14)
    us_csv = "ticker,date,price,quantity\nAAPL,03/04/2025,213,10\nMSFT,03/14/2025,400,5\n"
    imp_us = parse(us_csv)
    assert imp_us.ok
    assert imp_us.date_format == "חודש/יום"
    assert imp_us.rows.loc[0, "date"] == dt.date(2025, 3, 4)
    assert imp_us.rows.loc[1, "date"] == dt.date(2025, 3, 14)

    # Israel file: day-first detected because 14 is in 1st position (14/03/2025 -> March 14)
    il_csv = "ticker,date,price,quantity\nAAPL,04/03/2025,213,10\nMSFT,14/03/2025,400,5\n"
    imp_il = parse(il_csv)
    assert imp_il.ok
    assert imp_il.date_format == "יום/חודש"
    assert imp_il.rows.loc[0, "date"] == dt.date(2025, 3, 4)
    assert imp_il.rows.loc[1, "date"] == dt.date(2025, 3, 14)

    # Ambiguous file: defaults to day-first ("יום/חודש")
    amb_csv = "ticker,date,price,quantity\nAAPL,03/04/2025,213,10\nMSFT,01/02/2025,400,5\n"
    imp_amb = parse(amb_csv)
    assert imp_amb.ok
    assert imp_amb.date_format == "יום/חודש"
    assert imp_amb.rows.loc[0, "date"] == dt.date(2025, 4, 3)

    # Date/Time string (e.g., IBKR) with time stripped
    time_csv = "ticker,date,price,quantity\nAAPL,03/14/2025 10:30:00,213,10\n"
    imp_time = parse(time_csv)
    assert imp_time.ok
    assert imp_time.rows.loc[0, "date"] == dt.date(2025, 3, 14)


def test_headerless_ticker_file_and_tradingview_export():
    tv_data = b"NASDAQ:AAPL,NYSE:BRK.B"
    imp = parse(tv_data.decode("utf-8"))
    assert imp.ok and imp.kind == PF.WATCHLIST
    assert imp.rows["ticker"].tolist() == ["AAPL", "BRK.B"]

    multiline = "NASDAQ:AAPL\nNYSE:MSFT\nNVDA"
    imp_m = parse(multiline)
    assert imp_m.ok and imp_m.kind == PF.WATCHLIST
    assert imp_m.rows["ticker"].tolist() == ["AAPL", "MSFT", "NVDA"]


def test_finviz_export_and_watchlist_round_trip():
    finviz_csv = "Ticker,Price,Company\nAAPL,213.49,Apple Inc\nMSFT,461.97,Microsoft Corp\n"
    imp = parse(finviz_csv)
    assert imp.ok and imp.kind == PF.WATCHLIST
    assert imp.rows["ticker"].tolist() == ["AAPL", "MSFT"]
    assert imp.rows["price"].tolist() == [213.49, 461.97]

    plist = PF.from_imported("Finviz WL", imp, PRICES, TODAY)
    assert plist.rows[0]["price"] == 213.49
    assert plist.rows[0]["date"] == TODAY.isoformat()

    csv_data = PF.to_csv(plist)
    imp_re = parse(csv_data.decode("utf-8-sig"))
    assert imp_re.ok and imp_re.kind == PF.WATCHLIST
    assert imp_re.rows["price"].tolist() == [213.49, 461.97]
    assert imp_re.rows["date"].tolist() == [TODAY, TODAY]


def test_parse_number_decimal_comma_and_thousands():
    assert PF.parse_number("213,49") == 213.49
    assert PF.parse_number("1,234.56") == 1234.56
    assert PF.parse_number("1,234,567.89") == 1234567.89
    assert PF.parse_number("1.234,56") is None
    assert PF.parse_number("12,34,56") is None
    assert PF.parse_number("1,2345") is None
    assert PF.parse_number("213,490") == 213490.0
    assert PF.parse_number("$213.49") == 213.49
    assert PF.parse_number("-213,49") == -213.49


def test_ibkr_and_israeli_broker_imports():
    ibkr_csv = ("Symbol,Date/Time,T. Price,Quantity,Comm/Fee\n"
                "AAPL,03/14/2025 10:30:00,213.49,10,-1.25\n"
                "AAPL,03/15/2025 14:15:00,220.00,-10,-1.25\n")
    imp_ibkr = parse(ibkr_csv)
    assert imp_ibkr.ok and imp_ibkr.kind == PF.PORTFOLIO
    r_ibkr = imp_ibkr.rows
    assert r_ibkr.loc[0, "side"] == "buy" and r_ibkr.loc[0, "quantity"] == 10 and r_ibkr.loc[0, "fees"] == 1.25
    assert r_ibkr.loc[1, "side"] == "sell" and r_ibkr.loc[1, "quantity"] == 10 and r_ibkr.loc[1, "fees"] == 1.25

    il_csv = ("שם נייר,סימול נייר,תאריך,סוג פעולה,כמות,שער ביצוע,עמלה\n"
              "אפל,AAPL,14/03/2025,קניה חול,10,213.49,2.5\n"
              "אפל,AAPL,15/03/2025,מכירה חול,10,220.00,2.5\n")
    imp_il = parse(il_csv)
    assert imp_il.ok and imp_il.kind == PF.PORTFOLIO
    r_il = imp_il.rows
    assert r_il.loc[0, "ticker"] == "AAPL" and r_il.loc[0, "side"] == "buy" and r_il.loc[0, "quantity"] == 10
    assert r_il.loc[1, "ticker"] == "AAPL" and r_il.loc[1, "side"] == "sell" and r_il.loc[1, "quantity"] == 10


def test_same_day_trade_sorting_and_fractional_precision():
    # Same day sell before buy in input (e.g. descending date file)
    t = trades([("AAPL", "2025-03-14", 200, 10, "sell", 0),
                ("AAPL", "2025-03-14", 100, 10, "buy", 0)])
    assert PF.oversold(t) == []
    p = PF.positions(t).set_index("ticker")
    assert p.loc["AAPL", "quantity"] == 0
    assert p.loc["AAPL", "realized"] == 1000.0

    # Fractional shares summing to ~0
    t_frac = trades([("NVDA", "2025-01-01", 100, 0.1, "buy", 0),
                     ("NVDA", "2025-01-02", 100, 0.2, "buy", 0),
                     ("NVDA", "2025-01-03", 150, 0.3, "sell", 0)])
    p_frac = PF.positions(t_frac).set_index("ticker")
    assert p_frac.loc["NVDA", "quantity"] == 0.0
    assert p_frac.loc["NVDA", "cost"] == 0.0
    assert np.isnan(p_frac.loc["NVDA", "avg_cost"])


def test_add_trades_refuses_an_oversell():
    plist = PF.PList("p", PF.PORTFOLIO)
    buy = parse("ticker,date,price,quantity\nAAPL,2025-01-02,10,5\n").rows
    assert PF.add_trades(plist, buy) == [] and len(plist.rows) == 1
    sell = parse("ticker,date,price,quantity,side\nAAPL,2025-02-02,12,6,sell\n").rows
    assert PF.add_trades(plist, sell) == ["AAPL"] and len(plist.rows) == 1      # nothing added
    sell = parse("ticker,date,price,quantity,side\nAAPL,2025-02-02,12,5,sell\n").rows
    assert PF.add_trades(plist, sell) == [] and len(plist.rows) == 2


def test_unknown_headers_are_not_read_as_tickers():
    raw = PF.read_table("Stock,Qty\nAAPL,10\n".encode(), "x.csv")
    imp = PF.parse(raw)
    assert [p.code for p in imp.problems] == ["no_ticker_column"]
    bare = PF.read_table(b"NASDAQ:AAPL,NYSE:BRK.B\n", "tv.txt")       # our TradingView export
    assert list(PF.parse(bare).rows["ticker"]) == ["AAPL", "BRK.B"]
