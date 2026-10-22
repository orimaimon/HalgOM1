import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from screener.data.store import Snapshot, Store, session_date
from screener.history import load_previous
from screener.app import get_previous


def test_session_date():
    israel_tz = ZoneInfo("Asia/Jerusalem")
    ny_tz = ZoneInfo("America/New_York")

    # Monday 15:00 NY -> Monday
    dt_mon_15_ny = dt.datetime(2026, 10, 19, 15, 0, tzinfo=ny_tz)
    assert session_date(dt_mon_15_ny) == dt.date(2026, 10, 19)

    # Tuesday 02:00 NY -> Monday
    dt_tue_02_ny = dt.datetime(2026, 10, 20, 2, 0, tzinfo=ny_tz)
    assert session_date(dt_tue_02_ny) == dt.date(2026, 10, 19)

    # Saturday -> Friday
    dt_sat_12_ny = dt.datetime(2026, 10, 24, 12, 0, tzinfo=ny_tz)
    assert session_date(dt_sat_12_ny) == dt.date(2026, 10, 23)

    # Sunday 23:00 Israel -> Friday
    dt_sun_23_il = dt.datetime(2026, 10, 25, 23, 0, tzinfo=israel_tz)
    assert session_date(dt_sun_23_il) == dt.date(2026, 10, 23)

    # Monday 08:00 Israel -> Friday
    dt_mon_08_il = dt.datetime(2026, 10, 26, 8, 0, tzinfo=israel_tz)
    assert session_date(dt_mon_08_il) == dt.date(2026, 10, 23)


def test_synthetic_snapshots_monday_morning_israel(tmp_path):
    store = Store(root=tmp_path)

    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "price": [150.0, 300.0],
        "market_cap": [2e12, 2e12],
        "volume": [1e6, 1e6],
        "sector": ["Technology", "Technology"],
    })

    # Save Thursday snapshot (2026-10-15)
    ts_thu = dt.datetime(2026, 10, 15, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    store.save(df, ts_thu)

    # Save Friday snapshot (2026-10-16)
    ts_fri = dt.datetime(2026, 10, 16, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    store.save(df, ts_fri)

    # Save Monday intraday snapshot (2026-10-19 15:00 NY)
    ts_mon_intraday = dt.datetime(2026, 10, 19, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    store.save(df, ts_mon_intraday)

    # Simulate Monday morning Israel fetch (Mon Oct 19 08:00 Israel = Mon 01:00 NY)
    ts_mon_morning_il = dt.datetime(2026, 10, 19, 8, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))

    meta = f'{{"fetched_at": "{ts_mon_morning_il.isoformat()}"}}'
    (store.cache_dir / "latest_meta.json").write_text(meta, encoding="utf-8")

    from screener.data.store import session_date
    prev = load_previous(store, session_date(ts_mon_morning_il))
    assert prev is not None
    assert prev.day == dt.date(2026, 10, 15)


def test_get_previous_reloads_baseline(monkeypatch, tmp_path):
    store = Store(root=tmp_path)
    df = pd.DataFrame({
        "ticker": ["AAPL"],
        "price": [150.0],
        "market_cap": [2e12],
        "volume": [1e6],
        "sector": ["Technology"],
    })
    ts_thu = dt.datetime(2026, 10, 15, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    ts_fri = dt.datetime(2026, 10, 16, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    store.save(df, ts_thu)
    store.save(df, ts_fri)

    import screener.app as app
    monkeypatch.setattr(app, "_store", lambda: store)

    class MockSessionState(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, value):
            self[name] = value

    mock_ss = MockSessionState()
    monkeypatch.setattr(app, "ss", mock_ss)

    # 1. Snapshot before 07:00 Israel time on Mon Oct 19 (e.g. Mon 06:30 Israel)
    snap1 = Snapshot(df, dt.datetime(2026, 10, 19, 6, 30, tzinfo=ZoneInfo("Asia/Jerusalem")), "live")
    (store.cache_dir / "latest_meta.json").write_text(
        f'{{"fetched_at": "{snap1.fetched_at.isoformat()}"}}', encoding="utf-8"
    )
    prev1 = get_previous(snap1)
    assert prev1 is not None
    assert prev1.day == dt.date(2026, 10, 15)
    assert mock_ss["prev_for"] == dt.date(2026, 10, 16)

    # 2. Snapshot after crossing session threshold (e.g. Mon 17:00 Israel = Mon 10:00 NY)
    snap2 = Snapshot(df, dt.datetime(2026, 10, 19, 17, 0, tzinfo=ZoneInfo("Asia/Jerusalem")), "live")
    (store.cache_dir / "latest_meta.json").write_text(
        f'{{"fetched_at": "{snap2.fetched_at.isoformat()}"}}', encoding="utf-8"
    )

    prev2 = get_previous(snap2)
    assert prev2 is not None
    assert prev2.day == dt.date(2026, 10, 16)
    assert mock_ss["prev_for"] == dt.date(2026, 10, 19)
