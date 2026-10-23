"""
Tests for screener/alerts.py
"""
import datetime as dt
import re
import pandas as pd
import pytest

from screener import alerts, signals
from screener.alerts import Alert, Fired, KINDS


@pytest.fixture
def sample_universe() -> pd.DataFrame:
    """Synthetic universe dataframe with all required alert columns."""
    return pd.DataFrame([
        {
            "ticker": "AAPL",
            "price": 172.30,
            "change": -5.2,
            "sma200": 170.0,
            "close_prev": 180.0,
            "sma200_prev": 170.0,
            "earnings_date": pd.Timestamp("2025-10-28"),
        },
        {
            "ticker": "MSFT",
            "price": 420.00,
            "change": 5.2,
            "sma200": 400.0,
            "close_prev": 395.0,
            "sma200_prev": 400.0,
            "earnings_date": dt.date(2025, 10, 25),
        },
        {
            "ticker": "GOOGL",
            "price": 150.00,
            "change": 0.0,
            "sma200": 160.0,
            "close_prev": 165.0,
            "sma200_prev": 160.0,
            "earnings_date": dt.datetime(2025, 10, 29, 16, 0),
        },
        {
            "ticker": "NAN_STOCK",
            "price": float("nan"),
            "change": float("nan"),
            "sma200": float("nan"),
            "close_prev": float("nan"),
            "sma200_prev": float("nan"),
            "earnings_date": float("nan"),
        },
    ])


def test_kinds_and_messages_contain_hebrew(sample_universe):
    hebrew_regex = re.compile(r"[\u0590-\u05FF]")
    for kind, label in KINDS.items():
        assert label, f"Kind {kind} must have a non-empty label"
        assert hebrew_regex.search(label), f"Kind {kind} label must contain Hebrew"

    test_alerts = [
        Alert(id="1", ticker="AAPL", kind="price_below", value=175.0),
        Alert(id="2", ticker="MSFT", kind="price_above", value=400.0),
        Alert(id="3", ticker="AAPL", kind="change_below", value=-5.0),
        Alert(id="4", ticker="MSFT", kind="change_above", value=5.0),
        Alert(id="5", ticker="MSFT", kind="cross_above_sma200"),
        Alert(id="6", ticker="GOOGL", kind="cross_below_sma200"),
        Alert(id="7", ticker="AAPL", kind="earnings_within", value=3),
    ]

    fired = alerts.evaluate(
        test_alerts,
        sample_universe,
        session_date=dt.date(2025, 10, 25),
        today=dt.date(2025, 10, 25),
    )

    assert len(fired) > 0
    for f in fired:
        assert hebrew_regex.search(f.message), f"Message {f.message!r} must contain Hebrew"


def test_price_below(sample_universe):
    today = dt.date(2025, 10, 25)
    # Fire condition: price 172.30 < 175.00
    a1 = Alert(id="1", ticker="AAPL", kind="price_below", value=175.00, note="בדוק דוח")
    fired1 = alerts.evaluate([a1], sample_universe, session_date=today, today=today)
    assert len(fired1) == 1
    assert fired1[0].message == "AAPL: המחיר 172.30 ירד מתחת ל-175.00 — בדוק דוח"

    # Quiet condition: threshold 170.00 <= price 172.30
    a2 = Alert(id="2", ticker="AAPL", kind="price_below", value=170.00)
    fired2 = alerts.evaluate([a2], sample_universe, session_date=today, today=today)
    assert len(fired2) == 0

    # Exact threshold: price == 172.30 -> price < 172.30 is False
    a3 = Alert(id="3", ticker="AAPL", kind="price_below", value=172.30)
    fired3 = alerts.evaluate([a3], sample_universe, session_date=today, today=today)
    assert len(fired3) == 0


def test_price_above(sample_universe):
    today = dt.date(2025, 10, 25)
    # Fire condition: price 420.00 > 400.00
    a1 = Alert(id="1", ticker="MSFT", kind="price_above", value=400.00)
    fired1 = alerts.evaluate([a1], sample_universe, session_date=today, today=today)
    assert len(fired1) == 1
    assert fired1[0].message == "MSFT: המחיר 420.00 עלה מעל ל-400.00"

    # Quiet condition: threshold 450.00 >= price 420.00
    a2 = Alert(id="2", ticker="MSFT", kind="price_above", value=450.00)
    fired2 = alerts.evaluate([a2], sample_universe, session_date=today, today=today)
    assert len(fired2) == 0

    # Exact threshold: price == 420.00 -> price > 420.00 is False
    a3 = Alert(id="3", ticker="MSFT", kind="price_above", value=420.00)
    fired3 = alerts.evaluate([a3], sample_universe, session_date=today, today=today)
    assert len(fired3) == 0


def test_change_below(sample_universe):
    today = dt.date(2025, 10, 25)
    # AAPL change is -5.2
    # Threshold -5.0 (passed as -5.0 or 5.0) -> -5.2 <= -5.0 -> fires
    a1 = Alert(id="1", ticker="AAPL", kind="change_below", value=-5.0)
    fired1 = alerts.evaluate([a1], sample_universe, session_date=today, today=today)
    assert len(fired1) == 1
    assert fired1[0].message == "AAPL: ירדה 5.2% היום (סף: 5%)"

    # Exact threshold: change is -5.2, threshold value -5.2 -> fires
    a2 = Alert(id="2", ticker="AAPL", kind="change_below", value=5.2)
    fired2 = alerts.evaluate([a2], sample_universe, session_date=today, today=today)
    assert len(fired2) == 1
    assert fired2[0].message == "AAPL: ירדה 5.2% היום (סף: 5.2%)"

    # Quiet condition: change -5.2 > -6.0
    a3 = Alert(id="3", ticker="AAPL", kind="change_below", value=6.0)
    fired3 = alerts.evaluate([a3], sample_universe, session_date=today, today=today)
    assert len(fired3) == 0


def test_change_above(sample_universe):
    today = dt.date(2025, 10, 25)
    # MSFT change is 5.2
    # Threshold 5.0 -> 5.2 >= 5.0 -> fires
    a1 = Alert(id="1", ticker="MSFT", kind="change_above", value=5.0)
    fired1 = alerts.evaluate([a1], sample_universe, session_date=today, today=today)
    assert len(fired1) == 1
    assert fired1[0].message == "MSFT: עלתה 5.2% היום (סף: 5%)"

    # Exact threshold: change is 5.2, threshold 5.2 -> fires
    a2 = Alert(id="2", ticker="MSFT", kind="change_above", value=5.2)
    fired2 = alerts.evaluate([a2], sample_universe, session_date=today, today=today)
    assert len(fired2) == 1
    assert fired2[0].message == "MSFT: עלתה 5.2% היום (סף: 5.2%)"

    # Quiet condition: change 5.2 < 6.0
    a3 = Alert(id="3", ticker="MSFT", kind="change_above", value=6.0)
    fired3 = alerts.evaluate([a3], sample_universe, session_date=today, today=today)
    assert len(fired3) == 0


def test_sma200_cross_matches_signals(sample_universe):
    today = dt.date(2025, 10, 25)

    # Check signals.cross_up and signals.cross_down on sample_universe directly
    up_mask = signals.cross_up("price", "sma200")(sample_universe).fillna(False)
    down_mask = signals.cross_down("price", "sma200")(sample_universe).fillna(False)

    up_tickers = set(sample_universe.loc[up_mask, "ticker"])
    down_tickers = set(sample_universe.loc[down_mask, "ticker"])

    for ticker in sample_universe["ticker"].dropna():
        # Test cross_above_sma200
        a_up = Alert(id=f"up_{ticker}", ticker=ticker, kind="cross_above_sma200")
        f_up = alerts.evaluate([a_up], sample_universe, session_date=today, today=today)
        if ticker in up_tickers:
            assert len(f_up) == 1
            assert f_up[0].message == f"{ticker}: חצתה את ממוצע 200 כלפי מעלה היום"
        else:
            assert len(f_up) == 0

        # Test cross_below_sma200
        a_down = Alert(id=f"down_{ticker}", ticker=ticker, kind="cross_below_sma200")
        f_down = alerts.evaluate([a_down], sample_universe, session_date=today, today=today)
        if ticker in down_tickers:
            assert len(f_down) == 1
            assert f_down[0].message == f"{ticker}: חצתה את ממוצע 200 כלפי מטה היום"
        else:
            assert len(f_down) == 0


def test_earnings_within(sample_universe):
    today = dt.date(2025, 10, 25)

    # AAPL: earnings_date is Timestamp("2025-10-28"), 3 days away
    a1 = Alert(id="1", ticker="AAPL", kind="earnings_within", value=3)
    fired1 = alerts.evaluate([a1], sample_universe, session_date=today, today=today)
    assert len(fired1) == 1
    assert fired1[0].message == "AAPL: דוח כספי בעוד 3 ימים (28/10)"

    # AAPL: value=2 days -> 3 days away > 2 -> quiet
    a1_quiet = Alert(id="1q", ticker="AAPL", kind="earnings_within", value=2)
    assert len(alerts.evaluate([a1_quiet], sample_universe, session_date=today, today=today)) == 0

    # MSFT: earnings_date is date(2025, 10, 25), 0 days away (today)
    a2 = Alert(id="2", ticker="MSFT", kind="earnings_within", value=0)
    fired2 = alerts.evaluate([a2], sample_universe, session_date=today, today=today)
    assert len(fired2) == 1
    assert fired2[0].message == "MSFT: דוח כספי היום (25/10)"

    # GOOGL: earnings_date is datetime(2025, 10, 29, 16, 0), 4 days away
    a3 = Alert(id="3", ticker="GOOGL", kind="earnings_within", value=4)
    fired3 = alerts.evaluate([a3], sample_universe, session_date=today, today=today)
    assert len(fired3) == 1
    assert fired3[0].message == "GOOGL: דוח כספי בעוד 4 ימים (29/10)"


def test_last_fired_deduplication(sample_universe):
    session_date = dt.date(2025, 10, 25)

    # Already fired on this session date -> quiet
    a_fired = Alert(
        id="1",
        ticker="AAPL",
        kind="price_below",
        value=175.0,
        last_fired="2025-10-25",
    )
    fired1 = alerts.evaluate([a_fired], sample_universe, session_date=session_date, today=session_date)
    assert len(fired1) == 0

    # Fired on an older date -> fires
    a_old = Alert(
        id="2",
        ticker="AAPL",
        kind="price_below",
        value=175.0,
        last_fired="2025-10-24",
    )
    fired2 = alerts.evaluate([a_old], sample_universe, session_date=session_date, today=session_date)
    assert len(fired2) == 1


def test_missing_ticker_and_nan_values(sample_universe):
    today = dt.date(2025, 10, 25)

    # Ticker not in universe -> no fire, no exception
    a_missing = Alert(id="m1", ticker="NONEXISTENT", kind="price_below", value=100.0)
    fired_missing = alerts.evaluate([a_missing], sample_universe, session_date=today, today=today)
    assert len(fired_missing) == 0

    # Stock in universe has NaN values -> no fire, no exception for all kinds
    nan_alerts = [
        Alert(id="n1", ticker="NAN_STOCK", kind="price_below", value=100.0),
        Alert(id="n2", ticker="NAN_STOCK", kind="price_above", value=100.0),
        Alert(id="n3", ticker="NAN_STOCK", kind="change_below", value=-5.0),
        Alert(id="n4", ticker="NAN_STOCK", kind="change_above", value=5.0),
        Alert(id="n5", ticker="NAN_STOCK", kind="cross_above_sma200"),
        Alert(id="n6", ticker="NAN_STOCK", kind="cross_below_sma200"),
        Alert(id="n7", ticker="NAN_STOCK", kind="earnings_within", value=7),
    ]
    fired_nan = alerts.evaluate(nan_alerts, sample_universe, session_date=today, today=today)
    assert len(fired_nan) == 0


def test_storage_roundtrip_and_damaged_json(tmp_path):
    alerts_path = tmp_path / "alerts.json"

    # Load empty
    loaded = alerts.load(alerts_path)
    assert loaded == {}

    # Add alert
    a = Alert(ticker="AAPL", kind="price_below", value=175.0, note="test note")
    saved_a = alerts.add(a, path=alerts_path)
    assert saved_a.id != ""
    assert saved_a.created != ""

    # Load and verify
    loaded = alerts.load(alerts_path)
    assert len(loaded) == 1
    assert loaded[saved_a.id].ticker == "AAPL"
    assert loaded[saved_a.id].value == 175.0
    assert loaded[saved_a.id].note == "test note"

    # Mark fired
    session_date = dt.date(2025, 10, 25)
    alerts.mark_fired(saved_a.id, session_date, path=alerts_path)
    loaded = alerts.load(alerts_path)
    assert loaded[saved_a.id].last_fired == "2025-10-25"

    # Remove alert
    alerts.remove(saved_a.id, path=alerts_path)
    loaded = alerts.load(alerts_path)
    assert loaded == {}

    # Damaged file recovery test
    alerts_path.write_text("{ damaged json ...", encoding="utf-8")
    loaded_damaged = alerts.load(alerts_path)
    assert loaded_damaged == {}
    # Verify file was renamed to .bad-*
    bad_files = list(tmp_path.glob("alerts.json.bad-*"))
    assert len(bad_files) == 1
