"""
Tests for screener.data.nasdaq fallback provider module.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import requests

from screener.data import nasdaq
from screener.fields import provider_columns

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "nasdaq" / "screener.json"


@pytest.fixture
def nasdaq_fixture_data():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_is_excluded_security_rules():
    """Test filtering rules for warrants, units, rights, preferred shares, and class shares."""
    # Excluded
    assert nasdaq.is_excluded_security("XYZ^A", "XYZ Corp Preferred") is True
    assert nasdaq.is_excluded_security("WARR1", "Acme Corp Warrant") is True
    assert nasdaq.is_excluded_security("WARR2", "Acme Corp Warrants") is True
    assert nasdaq.is_excluded_security("UNIT1", "Alpha Corp Unit") is True
    assert nasdaq.is_excluded_security("UNIT2", "Alpha Corp Units") is True
    assert nasdaq.is_excluded_security("RGHT1", "Beta Corp Right") is True
    assert nasdaq.is_excluded_security("RGHT2", "Beta Corp Rights") is True
    assert nasdaq.is_excluded_security("PREF1", "Gamma Corp 8.5% Preferred Stock") is True

    # Included (class shares, regular stocks, and substring edge cases)
    assert nasdaq.is_excluded_security("AAPL", "Apple Inc. Common Stock") is False
    assert nasdaq.is_excluded_security("BRK/B", "Berkshire Hathaway Inc. Class B Common Stock") is False
    assert nasdaq.is_excluded_security("UPS", "United Parcel Service, Inc. Class B Common Stock") is False
    assert nasdaq.is_excluded_security("BRGT", "Bright Horizons Family Solutions Inc. Common Stock") is False


def test_fetch_universe_mapping_and_parsing(monkeypatch, nasdaq_fixture_data):
    """Test mapping, parsing $, %, NA, symbol conversion, sec_type, and ipo_year."""
    monkeypatch.setattr(nasdaq, "MIN_ROWS", 1)

    class MockResponse:
        status_code = 200

        def json(self):
            return nasdaq_fixture_data

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockResponse())

    df = nasdaq.fetch_universe()

    # Symbol conversion BRK/B -> BRK.B
    assert "BRK.B" in df["ticker"].values
    assert "BRK/B" not in df["ticker"].values

    # AAPL row parsing
    aapl = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl["price"] == 173.89
    assert aapl["change_abs"] == -2.172
    assert aapl["change"] == -2.172
    assert aapl["volume"] == 27950000.0
    assert aapl["market_cap"] == 2795000000000.0
    assert aapl["company"] == "Apple Inc."
    assert aapl["country"] == "United States"
    assert aapl["sector"] == "Technology"
    assert aapl["industry"] == "Computer Software: Prepackaged Software"
    assert aapl["sec_type"] == "stock"
    assert aapl["ipo_year"] == 1980

    # ADR row parsing
    baba = df[df["ticker"] == "BABA"].iloc[0]
    assert baba["sec_type"] == "dr"

    # Missing value parsing
    nval = df[df["ticker"] == "NVAL"].iloc[0]
    assert pd.isna(nval["price"])
    assert pd.isna(nval["change_abs"])
    assert pd.isna(nval["change"])
    assert pd.isna(nval["market_cap"])
    assert pd.isna(nval["country"])
    assert pd.isna(nval["sector"])
    assert pd.isna(nval["industry"])
    assert pd.isna(nval["ipo_year"])

    # Excluded tickers omitted
    for excluded in ("XYZ^A", "WARR1", "WARR2", "UNIT1", "UNIT2", "RGHT1", "RGHT2", "PREF1"):
        assert excluded not in df["ticker"].values

    # Class shares and substring names included
    for included in ("BRK.B", "UPS", "BRGT", "GOOGL", "MSFT", "META"):
        assert included in df["ticker"].values


def test_provider_columns_and_attrs(monkeypatch, nasdaq_fixture_data):
    """Test that all provider columns in fields.COLUMNS exist and source metadata is set."""
    monkeypatch.setattr(nasdaq, "MIN_ROWS", 1)

    class MockResponse:
        status_code = 200

        def json(self):
            return nasdaq_fixture_data

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockResponse())

    df = nasdaq.fetch_universe()

    assert df.attrs["source"] == {
        "provider": "nasdaq",
        "update_mode": "delayed",
        "delay_min": None,
        "raw": None,
    }
    assert df.attrs["source"]["provider"] == "nasdaq"

    # Check all provider columns exist in df
    for col in provider_columns():
        assert col in df.columns, f"Provider column {col} missing from DataFrame"


def test_nasdaq_errors(monkeypatch, nasdaq_fixture_data):
    """Test NasdaqError exceptions for HTTP errors, invalid JSON, missing rows, and row counts."""
    # 403 HTTP error
    class Mock403:
        status_code = 403
        text = "Forbidden"

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: Mock403())
    with pytest.raises(nasdaq.NasdaqError, match="HTTP 403"):
        nasdaq.fetch_universe()

    # Invalid JSON
    class MockInvalidJSON:
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockInvalidJSON())
    with pytest.raises(nasdaq.NasdaqError, match="invalid JSON"):
        nasdaq.fetch_universe()

    # Missing data.rows
    class MockMissingRows:
        status_code = 200

        def json(self):
            return {"data": {}}

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockMissingRows())
    with pytest.raises(nasdaq.NasdaqError, match="missing data.rows"):
        nasdaq.fetch_universe()

    # Too few rows (real MIN_ROWS = 3000)
    class MockFixtureResponse:
        status_code = 200

        def json(self):
            return nasdaq_fixture_data

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockFixtureResponse())
    with pytest.raises(nasdaq.NasdaqError, match="below minimum floor"):
        nasdaq.fetch_universe()

    # Test threshold separately with monkeypatched MIN_ROWS
    monkeypatch.setattr(nasdaq, "MIN_ROWS", 100)
    with pytest.raises(nasdaq.NasdaqError, match=r"below minimum floor \(100\)"):
        nasdaq.fetch_universe()


def test_request_headers_and_single_call(monkeypatch, nasdaq_fixture_data):
    """Test that requests.get is called exactly once with expected browser headers."""
    monkeypatch.setattr(nasdaq, "MIN_ROWS", 1)

    calls = []

    def mock_get(url, headers, timeout):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        class MockResponse:
            status_code = 200
            def json(self):
                return nasdaq_fixture_data
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    df = nasdaq.fetch_universe(timeout=30)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&download=true"
    assert call["timeout"] == 30
    assert call["headers"]["User-Agent"] == nasdaq.HEADERS["User-Agent"]
    assert call["headers"]["Accept"] == nasdaq.HEADERS["Accept"]
    assert call["headers"]["Accept-Language"] == nasdaq.HEADERS["Accept-Language"]
    assert call["headers"]["Origin"] == nasdaq.HEADERS["Origin"]
    assert call["headers"]["Referer"] == nasdaq.HEADERS["Referer"]
