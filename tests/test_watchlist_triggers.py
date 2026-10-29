"""Tests for Smart Watchlist Triggers & Playbook engine and UI component."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import portfolio, watchlist_triggers
from screener.data import store
from screener.portfolio import PORTFOLIO, WATCHLIST, PList


def test_targets_path_resolves_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert watchlist_triggers.targets_path() == tmp_path / "watchlist_targets.json"
    assert watchlist_triggers._targets_path() == tmp_path / "watchlist_targets.json"


def test_targets_path_user_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert watchlist_triggers._targets_path(user_id="userA") == tmp_path / "users" / "userA" / "watchlist_targets.json"
    assert watchlist_triggers.targets_path(user_id="userB") == tmp_path / "users" / "userB" / "watchlist_targets.json"


def test_set_and_load_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    p = tmp_path / "watchlist_targets.json"

    watchlist_triggers.set_target("MyWatch", "AAPL", 150.0, "above", path=p)
    watchlist_triggers.set_target("MyWatch", "MSFT", 400.0, "below", path=p)

    targets = watchlist_triggers.load_targets(path=p)
    assert len(targets) == 2
    assert targets["MyWatch:AAPL"].target_price == 150.0
    assert targets["MyWatch:AAPL"].direction == "above"
    assert targets["MyWatch:MSFT"].target_price == 400.0
    assert targets["MyWatch:MSFT"].direction == "below"

    # Set target to None removes it
    watchlist_triggers.set_target("MyWatch", "AAPL", None, path=p)
    targets_updated = watchlist_triggers.load_targets(path=p)
    assert "MyWatch:AAPL" not in targets_updated


def test_multi_user_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    # Set targets for User A and User B
    watchlist_triggers.set_target("Watch", "AAPL", 150.0, "above", user_id="userA")
    watchlist_triggers.set_target("Watch", "AAPL", 200.0, "below", user_id="userB")

    # Save target using save_target function as well
    t_a = watchlist_triggers.WatchTarget(
        key="Watch:MSFT", list_name="Watch", ticker="MSFT", target_price=300.0, direction="above"
    )
    watchlist_triggers.save_target(t_a, user_id="userA")

    # Verify isolation on load_targets
    targets_a = watchlist_triggers.load_targets(user_id="userA")
    targets_b = watchlist_triggers.load_targets(user_id="userB")
    targets_none = watchlist_triggers.load_targets(user_id=None)

    assert "Watch:AAPL" in targets_a
    assert targets_a["Watch:AAPL"].target_price == 150.0
    assert targets_a["Watch:AAPL"].direction == "above"
    assert "Watch:MSFT" in targets_a

    assert "Watch:AAPL" in targets_b
    assert targets_b["Watch:AAPL"].target_price == 200.0
    assert targets_b["Watch:AAPL"].direction == "below"
    assert "Watch:MSFT" not in targets_b

    assert "Watch:AAPL" not in targets_none

    # Evaluate triggers for User A vs User B
    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple", "price": 160.0, "change": 1.0},
        {"ticker": "MSFT", "company": "Microsoft", "price": 310.0, "change": 2.0},
    ])

    alerts_a = watchlist_triggers.evaluate_triggers(universe, user_id="userA")
    alerts_b = watchlist_triggers.evaluate_triggers(universe, user_id="userB")
    alerts_none = watchlist_triggers.evaluate_triggers(universe, user_id=None)

    # User A: AAPL current 160 >= 150 (triggered), MSFT current 310 >= 300 (triggered)
    assert len(alerts_a) == 2
    tickers_a = [al.ticker for al in alerts_a]
    assert "AAPL" in tickers_a
    assert "MSFT" in tickers_a

    # User B: AAPL direction below, current 160 is not <= 200 (triggered? wait, 160 <= 200 IS true for below!)
    # Let's check: direction below, current 160 <= 200 -> triggered!
    # Let's test with User B target price 100.0 below so current 160 is not <= 100
    watchlist_triggers.set_target("Watch", "AAPL", 100.0, "below", user_id="userB")
    alerts_b2 = watchlist_triggers.evaluate_triggers(universe, user_id="userB")
    assert len(alerts_b2) == 0

    assert len(alerts_none) == 0

    # Evaluate watchlist for User A vs User B
    plist = PList(name="Watch", kind=WATCHLIST, rows=[{"ticker": "AAPL"}, {"ticker": "MSFT"}])
    df_a = watchlist_triggers.evaluate_watchlist(plist, universe, user_id="userA")
    df_b = watchlist_triggers.evaluate_watchlist(plist, universe, user_id="userB")
    df_none = watchlist_triggers.evaluate_watchlist(plist, universe, user_id=None)

    assert (df_a[df_a["ticker"] == "AAPL"]["status"].iloc[0]) == "triggered"
    assert (df_b[df_b["ticker"] == "AAPL"]["status"].iloc[0]) == "waiting"
    assert (df_none[df_none["ticker"] == "AAPL"]["status"].iloc[0]) == "no_target"

    # Delete target for User A
    watchlist_triggers.delete_target("AAPL", user_id="userA")
    targets_a_after = watchlist_triggers.load_targets(user_id="userA")
    targets_b_after = watchlist_triggers.load_targets(user_id="userB")

    assert "Watch:AAPL" not in targets_a_after
    assert "Watch:MSFT" in targets_a_after
    assert "Watch:AAPL" in targets_b_after


def test_update_note(tmp_path, monkeypatch):
    portfolios_file = tmp_path / "portfolios.json"
    plist = PList(
        name="Watch1",
        kind=WATCHLIST,
        rows=[{"ticker": "AAPL", "note": "Old note"}, {"ticker": "MSFT", "note": ""}],
    )
    portfolio.save(plist, path=portfolios_file)

    res = watchlist_triggers.update_note(plist, "AAPL", "New note", portfolios_path=portfolios_file)
    assert res is True
    assert plist.rows[0]["note"] == "New note"

    saved_lists = portfolio.load_all(path=portfolios_file)
    assert saved_lists["Watch1"].rows[0]["note"] == "New note"

    # Ticker not in list returns False
    res2 = watchlist_triggers.update_note(plist, "NVDA", "Note", portfolios_path=portfolios_file)
    assert res2 is False


def test_evaluate_watchlist(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    targets_p = tmp_path / "watchlist_targets.json"

    # Set targets
    watchlist_triggers.set_target("Watch", "AAPL", 150.0, "above", path=targets_p)    # current=152 >= 150 -> triggered
    watchlist_triggers.set_target("Watch", "NVDA", 150.0, "above", path=targets_p)    # current=148.5, near (1%) -> near
    watchlist_triggers.set_target("Watch", "AMZN", 150.0, "above", path=targets_p)    # current=130 -> waiting
    watchlist_triggers.set_target("Watch", "MSFT", 100.0, "below", path=targets_p)    # current=98 <= 100 -> triggered
    # META: no target -> no_target

    plist = PList(
        name="Watch",
        kind=WATCHLIST,
        rows=[
            {"ticker": "AAPL", "note": "AAPL note"},
            {"ticker": "NVDA", "note": ""},
            {"ticker": "AMZN", "note": ""},
            {"ticker": "MSFT", "note": ""},
            {"ticker": "META", "note": ""},
        ],
    )

    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple", "price": 152.0, "change": 1.5},
        {"ticker": "NVDA", "company": "Nvidia", "price": 148.5, "change": 0.5},
        {"ticker": "AMZN", "company": "Amazon", "price": 130.0, "change": -0.2},
        {"ticker": "MSFT", "company": "Microsoft", "price": 98.0, "change": -1.0},
        {"ticker": "META", "company": "Meta", "price": 500.0, "change": 2.0},
    ])

    portfolio_plist = PList(
        name="MainPort",
        kind=PORTFOLIO,
        rows=[
            {"ticker": "AAPL", "date": "2025-01-01", "price": 140.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""}
        ],
    )
    all_lists = {"Watch": plist, "MainPort": portfolio_plist}

    df = watchlist_triggers.evaluate_watchlist(plist, universe, all_lists=all_lists, path=targets_p)

    assert set(df.columns) == set(watchlist_triggers.WATCH_TRIGGER_COLUMNS)

    # AAPL and MSFT are triggered (rank 0), NVDA is near (rank 1), AMZN is waiting (rank 2), META is no_target (rank 3)
    statuses = df["status"].tolist()
    assert statuses[:2] == ["triggered", "triggered"]
    assert statuses[2] == "near"
    assert statuses[3] == "waiting"
    assert statuses[4] == "no_target"

    aapl_row = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl_row["status"] == "triggered"
    assert aapl_row["in_portfolio"] == True
    assert aapl_row["note"] == "AAPL note"

    nvda_row = df[df["ticker"] == "NVDA"].iloc[0]
    assert nvda_row["status"] == "near"
    assert nvda_row["in_portfolio"] == False

    msft_row = df[df["ticker"] == "MSFT"].iloc[0]
    assert msft_row["status"] == "triggered"

    meta_row = df[df["ticker"] == "META"].iloc[0]
    assert meta_row["status"] == "no_target"
    assert pd.isna(meta_row["distance_pct"])


def test_evaluate_watchlist_empty_or_portfolio():
    universe = pd.DataFrame([{"ticker": "AAPL", "company": "Apple", "price": 150.0}])

    # Not a watchlist
    port = PList(name="P", kind=PORTFOLIO, rows=[{"ticker": "AAPL", "quantity": 10}])
    df1 = watchlist_triggers.evaluate_watchlist(port, universe)
    assert df1.empty
    assert list(df1.columns) == watchlist_triggers.WATCH_TRIGGER_COLUMNS

    # Empty watchlist
    empty_w = PList(name="W", kind=WATCHLIST, rows=[])
    df2 = watchlist_triggers.evaluate_watchlist(empty_w, universe)
    assert df2.empty


# ── UI Tests ─────────────────────────────────────────────────────────────────

def _run_ui_app(plist, universe):
    from screener.ui import watchlist_triggers_view
    watchlist_triggers_view.render(plist, universe)


def test_ui_render_table_and_banner(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    targets_p = tmp_path / "watchlist_targets.json"
    watchlist_triggers.set_target("MyWatch", "AAPL", 150.0, "above", path=targets_p)

    plist = PList(name="MyWatch", kind=WATCHLIST, rows=[{"ticker": "AAPL", "note": "Target 150"}])
    universe = pd.DataFrame([{"ticker": "AAPL", "company": "Apple Inc", "price": 152.0, "change": 1.2}])

    portfolios_p = tmp_path / "portfolios.json"
    portfolio.save(plist, path=portfolios_p)
    monkeypatch.setattr(portfolio, "PORTFOLIOS_PATH", portfolios_p)

    at = AppTest.from_function(_run_ui_app, kwargs={"plist": plist, "universe": universe})
    at.run()
    if at.exception:
        raise at.exception[0]

    # Success banner should be shown for triggered AAPL
    assert len(at.success) >= 1
    assert "AAPL" in at.success[0].value

    # Table should be rendered
    assert len(at.dataframe) >= 1


def test_ui_edit_target_and_note(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    targets_p = tmp_path / "watchlist_targets.json"

    portfolios_p = tmp_path / "portfolios.json"
    plist = PList(name="MyWatch", kind=WATCHLIST, rows=[{"ticker": "AAPL", "note": "Old Note"}])
    portfolio.save(plist, path=portfolios_p)
    monkeypatch.setattr(portfolio, "PORTFOLIOS_PATH", portfolios_p)

    universe = pd.DataFrame([{"ticker": "AAPL", "company": "Apple", "price": 145.0, "change": 0.5}])

    at = AppTest.from_function(_run_ui_app, kwargs={"plist": plist, "universe": universe})
    at.run()
    if at.exception:
        raise at.exception[0]

    # Set target to 150.0 and note to "Updated Note"
    at.number_input(key="wt_target").set_value(150.0)
    at.text_input(key="wt_note").set_value("Updated Note")
    at.button(key="wt_save").click()
    at.run()

    # Verify target and note were persisted
    targets = watchlist_triggers.load_targets(path=targets_p)
    assert targets["MyWatch:AAPL"].target_price == 150.0

    saved_lists = portfolio.load_all(path=portfolios_p)
    assert saved_lists["MyWatch"].rows[0]["note"] == "Updated Note"


def test_ui_error_handling(monkeypatch):
    from screener.ui import watchlist_triggers_view

    def bad_eval(*args, **kwargs):
        raise RuntimeError("Simulated crash")

    monkeypatch.setattr(watchlist_triggers, "evaluate_watchlist", bad_eval)

    plist = PList(name="W", kind=WATCHLIST, rows=[{"ticker": "AAPL"}])
    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0}])

    at = AppTest.from_function(_run_ui_app, kwargs={"plist": plist, "universe": universe})
    at.run()

    if at.exception:
        raise at.exception[0]

    # Should display hebrew_note error message via st.html
    assert any("שגיאה בהצגת טריגרים" in h.value for h in at.html)
