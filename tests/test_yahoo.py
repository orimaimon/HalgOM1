"""
Tests for screener/data/yahoo.py
"""
import logging
import pytest
import numpy as np
import pandas as pd

import screener.data.yahoo as yahoo_mod
from screener.data.yahoo import YahooError, fetch_universe, clean_quotes
from screener.fields import COLUMNS, provider_columns


def make_fake_quote(
    symbol: str = "AAPL",
    long_name: str = "Apple Inc.",
    exchange: str = "NMS",
    price: float = 150.0,
    change_pct: float = 2.5,
    div_yield: float = 0.5,
    earnings_ts: int = 1700000000,
) -> dict:
    return {
        "symbol": symbol,
        "longName": long_name,
        "exchange": exchange,
        "regularMarketPrice": price,
        "regularMarketChange": 3.75,
        "regularMarketChangePercent": change_pct,
        "regularMarketVolume": 1000000,
        "averageDailyVolume3Month": 1200000,
        "marketCap": 2500000000000,
        "sharesOutstanding": 16000000000,
        "trailingPE": 28.5,
        "epsTrailingTwelveMonths": 5.25,
        "priceToBook": 42.0,
        "fiftyDayAverage": 145.0,
        "twoHundredDayAverage": 140.0,
        "fiftyTwoWeekHigh": 180.0,
        "fiftyTwoWeekLow": 120.0,
        "earningsTimestamp": earnings_ts,
        "dividendYield": div_yield,
    }


def test_yahoo_paging_and_sleep(monkeypatch):
    calls = []
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(yahoo_mod, "_sleep", fake_sleep)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        calls.append((offset, size))
        if offset == 0:
            quotes = [make_fake_quote(symbol=f"STOCK{i}") for i in range(250)]
        elif offset == 250:
            quotes = [make_fake_quote(symbol=f"STOCK{i}") for i in range(250, 500)]
        elif offset == 500:
            quotes = [make_fake_quote(symbol=f"STOCK{i}") for i in range(500, 600)]
        else:
            quotes = []
        return {"total": 600, "quotes": quotes}

    monkeypatch.setattr("yfinance.screen", fake_screen)

    df = fetch_universe()

    # 3 calls with offsets 0, 250, 500
    assert calls == [(0, 250), (250, 250), (500, 250)]
    # 2 sleeps of 1s between pages
    assert sleeps == [1, 1]
    # 600 unique rows
    assert len(df) == 600
    assert df["ticker"].nunique() == 600


def test_yahoo_empty_page_stops_paging(monkeypatch):
    calls = []
    monkeypatch.setattr(yahoo_mod, "_sleep", lambda _: None)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        calls.append(offset)
        if offset == 0:
            quotes = [make_fake_quote(symbol=f"STOCK{i}") for i in range(250)]
        else:
            quotes = []
        return {"total": 1000, "quotes": quotes}

    monkeypatch.setattr("yfinance.screen", fake_screen)

    df = fetch_universe()
    assert calls == [0, 250]
    assert len(df) == 250


def test_yahoo_max_requests_cap(monkeypatch, caplog):
    monkeypatch.setattr(yahoo_mod, "_sleep", lambda _: None)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        quotes = [make_fake_quote(symbol=f"STOCK_{offset}_{i}") for i in range(250)]
        return {"total": 20000, "quotes": quotes}

    monkeypatch.setattr("yfinance.screen", fake_screen)

    with caplog.at_level(logging.WARNING):
        df = fetch_universe(max_requests=3)

    assert len(df) == 750
    assert "cut off at max_requests" in caplog.text


def test_yahoo_mapping_and_units():
    raw_quotes = [
        # BRK-B ticker conversion, NMS -> NASDAQ exchange, percent yield (2.5)
        make_fake_quote(symbol="BRK-B", long_name="Berkshire Hathaway Inc.", exchange="NMS", div_yield=2.5),
        # NYQ -> NYSE, a yield under 1% stays as is (0.75 == 0.75%, like MSFT live)
        make_fake_quote(symbol="JNJ", long_name="Johnson & Johnson", exchange="NYQ", div_yield=0.75),
        # ASE -> AMEX
        make_fake_quote(symbol="GLD", long_name="SPDR Gold Trust", exchange="ASE", div_yield=0.0),
    ]

    df = clean_quotes(raw_quotes)

    # BRK-B -> BRK.B
    tickers = list(df["ticker"])
    assert "BRK.B" in tickers
    assert "BRK-B" not in tickers

    # Exchanges
    exchanges = dict(zip(df["ticker"], df["exchange"]))
    assert exchanges["BRK.B"] == "NASDAQ"
    assert exchanges["JNJ"] == "NYSE"
    assert exchanges["GLD"] == "AMEX"

    # Dividend yields: Yahoo already gives percent numbers, never multiplied
    yields = dict(zip(df["ticker"], df["div_yield"]))
    assert yields["BRK.B"] == pytest.approx(2.5)
    assert yields["JNJ"] == pytest.approx(0.75)
    assert yields["GLD"] == pytest.approx(0.0)

    # earnings_date type
    assert pd.api.types.is_datetime64_ns_dtype(df["earnings_date"]) or pd.api.types.is_datetime64_any_dtype(df["earnings_date"])
    assert df["earnings_date"].dt.tz is None


def test_yahoo_provider_and_derived_columns(monkeypatch):
    monkeypatch.setattr(yahoo_mod, "_sleep", lambda _: None)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        return {"total": 1, "quotes": [make_fake_quote(symbol="MSFT")]}

    monkeypatch.setattr("yfinance.screen", fake_screen)

    df = fetch_universe()

    # Source metadata
    assert df.attrs.get("source") == {"provider": "yahoo", "update_mode": "delayed", "delay_min": None, "raw": None}

    # All provider columns must exist
    prov_keys = provider_columns()
    for col_key in prov_keys:
        assert col_key in df.columns, f"Provider column missing: {col_key}"

    # No derived columns must be present
    derived_keys = [k for k, c in COLUMNS.items() if c.source == "derived"]
    for col_key in derived_keys:
        assert col_key not in df.columns, f"Derived column present in provider df: {col_key}"


def test_yahoo_zero_rows_error(monkeypatch):
    monkeypatch.setattr(yahoo_mod, "_sleep", lambda _: None)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        return {"total": 0, "quotes": []}

    monkeypatch.setattr("yfinance.screen", fake_screen)

    with pytest.raises(YahooError, match="empty universe returned"):
        fetch_universe()


def test_yahoo_screen_exception_propagates(monkeypatch):
    monkeypatch.setattr(yahoo_mod, "_sleep", lambda _: None)

    def fake_screen(query, offset=0, size=250, sortField="intradaymarketcap", sortAsc=False):
        raise RuntimeError("yfinance rate limit exceeded")

    monkeypatch.setattr("yfinance.screen", fake_screen)

    with pytest.raises(RuntimeError, match="yfinance rate limit exceeded"):
        fetch_universe()
