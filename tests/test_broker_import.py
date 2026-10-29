"""
Tests for screener/broker_import.py.
Pure unit tests with inline CSV strings, no network calls, no files on disk.
"""
import datetime as dt
import re

import pandas as pd
import pytest

from screener.broker_import import COLUMNS, parse_trades, parse_trades_report


def test_empty_string():
    df, reports = parse_trades_report("")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLUMNS
    assert df.empty
    assert reports == []

    df_simple = parse_trades("")
    assert list(df_simple.columns) == COLUMNS
    assert df_simple.empty


def test_ibkr_buy_and_sell_rows_with_date_formats():
    ibkr_csv = """Trades,Header,DataDiscriminator,Asset Class,Currency,Symbol,Date/Time,Quantity,T. Price,Comm/Fee
Trades,Data,Order,Stocks,USD,AAPL,"2024-03-05, 09:31:02",10,175.50,-1.0
Trades,Data,Order,Stocks,USD,MSFT,20240305;093102,-5,400.25,-2.50
Trades,Data,Order,Stocks,USD,BRK-B,2024-03-06,20,350.00,0.0
"""
    df, reports = parse_trades_report(ibkr_csv, broker="ibkr")
    assert reports == []
    assert len(df) == 3
    assert list(df["ticker"]) == ["AAPL", "MSFT", "BRK.B"]
    assert list(df["side"]) == ["buy", "sell", "buy"]
    assert list(df["shares"]) == [10.0, 5.0, 20.0]
    assert list(df["price"]) == [175.50, 400.25, 350.00]
    assert list(df["fees"]) == [1.0, 2.50, 0.0]
    assert list(df["date"]) == [dt.date(2024, 3, 5), dt.date(2024, 3, 5), dt.date(2024, 3, 6)]


def test_generic_template():
    generic_csv = """date,ticker,side,shares,price,fees
2024-03-01,NVDA,buy,15,800.50,5.0
2024-03-02,TSLA,sell,10,200.00,2.5
"""
    df = parse_trades(generic_csv, broker="generic")
    assert len(df) == 2
    assert list(df["ticker"]) == ["NVDA", "TSLA"]
    assert list(df["side"]) == ["buy", "sell"]
    assert list(df["shares"]) == [15.0, 10.0]
    assert list(df["price"]) == [800.50, 200.00]
    assert list(df["fees"]) == [5.0, 2.5]
    assert list(df["date"]) == [dt.date(2024, 3, 1), dt.date(2024, 3, 2)]


def test_auto_detection():
    ibkr_csv = """Trades,Header,Symbol,Date/Time,Quantity,T. Price,Comm/Fee
Trades,Data,AMZN,2024-03-10,5,180.00,-1.0
"""
    generic_csv = """Date,Symbol,Action,Qty,Price,Commission
2024-03-10,GOOGL,buy,8,140.00,1.5
"""
    df_ibkr = parse_trades(ibkr_csv, broker="auto")
    assert len(df_ibkr) == 1
    assert df_ibkr.iloc[0]["ticker"] == "AMZN"

    df_generic = parse_trades(generic_csv, broker="auto")
    assert len(df_generic) == 1
    assert df_generic.iloc[0]["ticker"] == "GOOGL"


def test_bad_number_row_skipped_with_hebrew_message():
    csv_text = """date,ticker,side,shares,price,fees
2024-03-01,AAPL,buy,invalid,150.0,1.0
2024-03-02,MSFT,buy,10,400.0,0.5
"""
    df, reports = parse_trades_report(csv_text, broker="generic")
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "MSFT"
    assert len(reports) == 1
    assert "שורה 2:" in reports[0]
    # Verify Hebrew characters in report message
    assert re.search(r"[\u0590-\u05FF]", reports[0]) is not None


def test_blank_lines_and_bom():
    csv_text = "\ufeff\n\ndate,ticker,side,shares,price,fees\n\n2024-03-01,AAPL,buy,10,150.0,1.0\n\n"
    df, reports = parse_trades_report(csv_text, broker="auto")
    assert reports == []
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"


test_hebrew_regex = re.compile(r"[\u0590-\u05FF]")


def test_commas_inside_quoted_fields():
    csv_text = """date,ticker,side,shares,price,fees
2024-03-01,"AAPL",buy,"1,000",150.00,"1.50"
"""
    df, reports = parse_trades_report(csv_text, broker="generic")
    assert reports == []
    assert len(df) == 1
    assert df.iloc[0]["shares"] == 1000.0
    assert df.iloc[0]["price"] == 150.00
    assert df.iloc[0]["fees"] == 1.50


def test_hebrew_headers_and_sides_in_generic():
    csv_text = """תאריך,סימול,פעולה,כמות,מחיר,עמלה
2024-03-01,META,קנייה,5,500.0,2.0
2024-03-02,AMD,מכירה,10,180.0,1.0
"""
    df, reports = parse_trades_report(csv_text, broker="generic")
    assert reports == []
    assert len(df) == 2
    assert list(df["ticker"]) == ["META", "AMD"]
    assert list(df["side"]) == ["buy", "sell"]
