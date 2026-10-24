"""
Tests for screener/ui/breadth_panel.py component.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from screener.data.store import Store, _atomic_parquet


def _make_universe() -> pd.DataFrame:
    rows = [
        {"ticker": "AAPL", "company": "Apple Inc.", "sector": "Technology", "price": 150.0, "sma50": 140.0, "sma200": 130.0, "change": 1.5, "high52_prev": 145.0, "low52_prev": 100.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "MSFT", "company": "Microsoft Corp.", "sector": "Technology", "price": 300.0, "sma50": 310.0, "sma200": 290.0, "change": -0.5, "high52_prev": 320.0, "low52_prev": 250.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "JPM", "company": "JPMorgan Chase", "sector": "Finance", "price": 120.0, "sma50": 115.0, "sma200": 125.0, "change": 0.0, "high52_prev": 130.0, "low52_prev": 105.0, "avg_volume": 100e3, "market_cap": 100e6},
        {"ticker": "XOM", "company": "Exxon Mobil", "sector": "Energy", "price": np.nan, "sma50": np.nan, "sma200": np.nan, "change": np.nan, "high52_prev": np.nan, "low52_prev": np.nan, "avg_volume": 100e3, "market_cap": 100e6},
    ]
    return pd.DataFrame(rows)


def _script_render(universe, root_dir, monkeypatch_error=False):
    from pathlib import Path
    import pandas as pd
    import streamlit as st
    from screener.data.store import Store
    from screener.ui import breadth_panel

    if monkeypatch_error:
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated breadth error")
        import screener.breadth
        screener.breadth.breadth = _raise

    store = Store(Path(root_dir))
    breadth_panel.render(universe, store, junk=True)


def test_breadth_panel_synthetic_universe(tmp_path):
    df = _make_universe()
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)
    d1 = dt.date(2026, 3, 1)
    d2 = dt.date(2026, 3, 2)
    _atomic_parquet(df, store.snapshot_path(d1))
    _atomic_parquet(df, store.snapshot_path(d2))

    at = AppTest.from_function(_script_render, args=(df, str(tmp_path))).run()
    if at.exception:
        raise Exception(f"AppTest failed: {at.exception}")
    assert not at.exception

    # Check 4 metric values exist
    metrics = [m.value for m in at.metric]
    assert len(metrics) == 4
    # Check sector table exists
    assert len(at.dataframe) >= 1
    # Check plotly chart element exists via key or element lookup
    assert len(at.get("plotly_chart")) >= 1 or any(el.key == "br_history_chart" for el in at.get("plotly_chart"))


def test_breadth_panel_one_snapshot_shows_note(tmp_path):
    df = _make_universe()
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)
    d1 = dt.date(2026, 3, 1)
    _atomic_parquet(df, store.snapshot_path(d1))

    at = AppTest.from_function(_script_render, args=(df, str(tmp_path))).run()
    if at.exception:
        raise Exception(f"AppTest failed: {at.exception}")
    assert not at.exception

    # Note should be displayed for <2 days history
    html_texts = [h.value for h in at.html]
    assert any("צריך כמה ימים של נתונים שמורים כדי להציג מגמה" in text for text in html_texts)
    # Chart should not exist
    assert len(at.get("plotly_chart")) == 0


def test_breadth_panel_exception_resilience(tmp_path):
    df = _make_universe()

    at = AppTest.from_function(_script_render, args=(df, str(tmp_path), True)).run()
    # Should handle exception gracefully without raising
    assert not at.exception

    html_texts = [h.value for h in at.html]
    assert any("שגיאה בטעינת נתוני רוחב השוק" in text for text in html_texts)
