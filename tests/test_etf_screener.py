"""Tests for ETF screener feature: TradingView query, cleaning, presets, filters, AppTest UI and portfolio resolution."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.data.tradingview import clean, universe_query
import screener.fields as fields
import screener.filters as F
import screener.portfolio as PF
import screener.presets as P
from tests.test_screener import ROOT, make_raw


def test_universe_query_requests_etfs():
    """TradingView query must include typespecs has ['etf'] in filter2 operands."""
    query = universe_query(["name", "close", "aum", "expense_ratio"])
    f2 = query["filter2"]
    assert f2["operator"] == "or"
    operands = f2["operands"]
    has_etf_branch = False
    for op in operands:
        inner = op.get("operation", {})
        for expr in inner.get("operands", []):
            e = expr.get("expression", {})
            if e.get("left") == "typespecs" and e.get("operation") == "has" and e.get("right") == ["etf"]:
                has_etf_branch = True
    assert has_etf_branch, "universe_query missing typespecs has ['etf']"


def test_clean_normalizes_etf_sec_type():
    """clean() must map sec_type='fund' or typespecs containing 'etf' to sec_type='etf'."""
    df = pd.DataFrame({
        "ticker": ["SPY", "QQQ", "AAPL"],
        "price": [500.0, 400.0, 200.0],
        "sec_type": ["fund", "stock", "stock"],
        "typespecs": [["etf"], ["etf"], ["common"]],
        "aum": [500e9, 250e9, np.nan],
        "expense_ratio": [0.09, 0.20, np.nan],
        "market_cap": [np.nan, np.nan, 3e12],
        "is_primary": [True, True, True],
        "currency": ["USD", "USD", "USD"],
    })
    out = clean(df)
    assert set(out["ticker"]) == {"SPY", "QQQ", "AAPL"}
    spy = out[out["ticker"] == "SPY"].iloc[0]
    qqq = out[out["ticker"] == "QQQ"].iloc[0]
    aapl = out[out["ticker"] == "AAPL"].iloc[0]

    assert spy["sec_type"] == "etf"
    assert qqq["sec_type"] == "etf"
    assert aapl["sec_type"] == "stock"

    assert spy["aum"] == 500e9
    assert spy["expense_ratio"] == 0.09
    assert pd.isna(aapl["aum"])


def test_etf_builtin_presets_valid():
    """Every ETF preset in BUILTIN_ETF must refer to valid filters, valid options, and a valid view."""
    filters_by_key = {f.key: f for f in F.FILTERS}
    for name, p in P.BUILTIN_ETF.items():
        assert p.view in fields.VIEWS, f"{name}: view {p.view} not in VIEWS"
        if p.sort is not None:
            assert p.sort in fields.COLUMNS, f"{name}: sort column {p.sort} not in COLUMNS"
        assert p.note != "", f"{name} must have a Hebrew note"
        for f_key, opt_label in p.filters.items():
            assert f_key in filters_by_key, f"{name}: filter {f_key} not in FILTERS"
            f = filters_by_key[f_key]
            valid_opts = [o.label for o in f.options]
            assert opt_label in valid_opts, f"{name}: option {opt_label!r} not in {f_key} options"


def test_etf_screener_mode_in_app(tmp_path, monkeypatch):
    """Switching asset_type to 'etfs' in AppTest filters to ETFs and swaps presets."""
    import screener.data.store as store_mod
    import screener.data.tradingview as tv
    import screener.lifecycle as lifecycle

    raw_stocks = make_raw(10).assign(is_primary=True, currency="USD", sec_type="stock", aum=np.nan, expense_ratio=np.nan)
    raw_etfs = make_raw(5).assign(is_primary=True, currency="USD", sec_type="etf",
                                  ticker=["SPY", "QQQ", "IWM", "XLK", "SMH"],
                                  aum=[500e9, 250e9, 70e9, 80e9, 30e9],
                                  expense_ratio=[0.09, 0.20, 0.19, 0.10, 0.35])
    raw = clean(pd.concat([raw_stocks, raw_etfs], ignore_index=True))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}

    market = pd.DataFrame({"close": [500.0, 400, 80, 15], "change": 0.5, "sma20": [490.0, 390, 79, 16],
                           "sma50": 480.0, "sma200": [450.0, 350, 78, 17], "adx": [30.0, 20, 15, 10],
                           "di_plus": 25.0, "di_minus": 15.0}, index=["SPY", "QQQ", "HYG", "VIX"])

    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PRESETS_PATH", tmp_path / "presets.json")
    monkeypatch.setattr(PF, "PORTFOLIOS_PATH", tmp_path / "portfolios.json")
    monkeypatch.setattr(tv, "fetch_universe", lambda: raw)
    monkeypatch.setattr(tv, "fetch_market", lambda: market)
    monkeypatch.setattr(lifecycle, "ensure_idle_watchdog_started", lambda: None)

    app = AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=60)
    app.session_state["page"] = "Screener"
    app.run()
    assert not app.exception

    # Initially in stocks mode:
    assert app.session_state.get("asset_type", "stocks") == "stocks"

    # Switch to ETFs mode:
    app.segmented_control(key="asset_type").select("etfs").run()
    assert not app.exception
    assert app.session_state["asset_type"] == "etfs"

    # Preset dropdown should now contain ETF presets (formatted in Hebrew)
    preset_sb = app.selectbox(key="preset")
    assert any("Top AUM" in opt for opt in preset_sb.options)


def test_portfolio_resolves_etf_price(tmp_path, monkeypatch):
    """Portfolios page receives the combined universe so ETFs (SPY) get live price and value."""
    import screener.data.store as store_mod
    import screener.data.tradingview as tv
    import screener.lifecycle as lifecycle

    raw_stocks = make_raw(5).assign(is_primary=True, currency="USD", sec_type="stock", aum=np.nan)
    raw_etfs = make_raw(2).assign(is_primary=True, currency="USD", sec_type="etf",
                                  ticker=["SPY", "QQQ"],
                                  price=[550.0, 480.0],
                                  aum=[500e9, 250e9],
                                  expense_ratio=[0.09, 0.20])
    raw = clean(pd.concat([raw_stocks, raw_etfs], ignore_index=True))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}

    market = pd.DataFrame({"close": [550.0, 480, 80, 15], "change": 0.5, "sma20": [540.0, 470, 79, 16],
                           "sma50": 530.0, "sma200": [500.0, 450, 78, 17], "adx": [30.0, 20, 15, 10],
                           "di_plus": 25.0, "di_minus": 15.0}, index=["SPY", "QQQ", "HYG", "VIX"])

    # Create portfolio with SPY
    plist = PF.PList("My ETF Portfolio", PF.PORTFOLIO, rows=[
        {"ticker": "SPY", "shares": 10.0, "cost": 500.0, "date": "2026-01-15", "side": "buy"}
    ])
    PF.save(plist, tmp_path / "portfolios.json")

    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PRESETS_PATH", tmp_path / "presets.json")
    monkeypatch.setattr(PF, "PORTFOLIOS_PATH", tmp_path / "portfolios.json")
    monkeypatch.setattr(tv, "fetch_universe", lambda: raw)
    monkeypatch.setattr(tv, "fetch_market", lambda: market)
    monkeypatch.setattr(lifecycle, "ensure_idle_watchdog_started", lambda: None)

    app = AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=60)
    app.session_state["page"] = "Portfolios"
    app.run()
    assert not app.exception

    # SPY should have market value: 10 shares * $550 = $5,500
    # There should NOT be any missing price warning
    warning_notes = [e.proto.body for e in app.get("html") if "אין בסקרינר מחיר" in e.proto.body]
    assert len(warning_notes) == 0, f"Expected 0 missing price warnings, found: {warning_notes}"
