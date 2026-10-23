"""screener/track.py: track record of a screen, offline on synthetic snapshots."""
import datetime as dt
import os

import pandas as pd
import pytest

from screener import presets as P
from screener import track
from screener.data.store import Store
from screener.fields import COLUMNS

DAY0 = dt.date(2026, 1, 1)
IN, OUT = 12.0, 30.0                 # P/E inside / outside the test screen
SCREEN = {"pe": "Under 15"}


def make_snapshot(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """rows: ticker -> (price, pe). Everything else passes Hide junk."""
    tickers = list(rows)
    n = len(tickers)
    data = {k: [10.0] * n for k, c in COLUMNS.items() if c.source == "provider" and c.kind not in ("text", "date")}
    df = pd.DataFrame(data)
    df["ticker"] = tickers
    df["company"] = [f"Co {t}" for t in tickers]
    for k, v in {"exchange": "NASDAQ", "sector": "Technology", "industry": "Software",
                 "country": "United States", "sec_type": "stock"}.items():
        df[k] = v
    df["price"] = [rows[t][0] for t in tickers]
    df["pe"] = [rows[t][1] for t in tickers]
    df["market_cap"], df["avg_volume"] = 1e9, 1e6
    return df


def write_day(store: Store, i: int, rows: dict[str, tuple[float, float]]) -> None:
    store.snap_dir.mkdir(parents=True, exist_ok=True)
    make_snapshot(rows).to_parquet(store.snapshot_path(DAY0 + dt.timedelta(days=i)))


def day(i: int) -> dt.date:
    return DAY0 + dt.timedelta(days=i)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network: benchmark updates are recorded, and there is no stored price history."""
    calls = []
    monkeypatch.setattr(track.prices, "ensure", lambda tickers, start: calls.append(tickers) or ([], tickers))
    monkeypatch.setattr(track.prices, "history", lambda t: pd.DataFrame())
    monkeypatch.setattr(track, "_benchmarks_checked", None)
    return calls


def scenario(i: int) -> dict[str, tuple[float, float]]:
    rows = {
        "ALWAYS": (50.0, IN),                                          # in the results from day 0: never an entry
        "UP": (100.0 if i < 8 else 110.0 if i < 23 else 120.0, IN if i >= 3 else OUT),
        "SPLIT": (100.0 if i < 5 else 50.0, IN if i >= 3 else OUT),       # 2-for-1 on day 5
        "DOUBLE": (100.0 + 5 * max(0, i - 3), IN if i >= 3 else OUT),    # +100% in 20 days, never a jump
        "LATE": (100.0, IN if i >= 27 else OUT),
        "BACK": (40.0, OUT if 10 <= i < 12 else IN),                     # leaves on day 10, back on day 12
        "OTHER": (20.0, OUT),
    }
    if i >= 15:
        rows.pop("UP")                                                 # gone: no 20-day return... until it is back
    if i >= 20:
        rows["UP"] = (120.0, IN)
    return rows


def test_entries_returns_splits_and_pending(tmp_path):
    store = Store(tmp_path)
    for i in range(30):
        write_day(store, i, scenario(i))
    res = track.track_screen(SCREEN, "test", store)
    t = res["table"].set_index(["ticker", "day"])

    assert set(t.index) == {("UP", day(3)), ("SPLIT", day(3)), ("DOUBLE", day(3)), ("LATE", day(27)),
                            ("BACK", day(12)), ("UP", day(20))}      # never day 0, never ALWAYS
    assert t.at[("UP", day(3)), "ret_5"] == pytest.approx(10.0)
    assert t.at[("UP", day(3)), "ret_20"] == pytest.approx(20.0)
    assert t.at[("SPLIT", day(3)), "ret_5"] == pytest.approx(0.0)       # the split is not a loss
    assert t.at[("DOUBLE", day(3)), "ret_20"] == pytest.approx(100.0)   # a gradual double is not a split
    assert t.at[("LATE", day(27)), "pending_5"] and pd.isna(t.at[("LATE", day(27)), "ret_5"])
    assert t.at[("UP", day(3)), "pending_60"]                           # day 63 does not exist yet
    assert res["splits_undone"] == 1

    h5 = res["horizons"][5]
    assert h5["entries"] == 5 and h5["pending"] == 1                    # LATE is pending
    assert res["sessions"] == 30 and not res["few_sessions"]
    assert res["horizons"][60]["entries"] == 0 and res["horizons"][60]["median"] is None


def test_a_ticker_gone_at_the_exit_day_has_no_return(tmp_path):
    store = Store(tmp_path)
    for i in range(12):
        rows = {"A": (10.0, IN if i >= 1 else OUT), "B": (10.0, IN if i >= 1 else OUT)}
        if i >= 4:
            rows.pop("B")
        write_day(store, i, rows)
    t = track.track_screen(SCREEN, "gone", store)["table"].set_index("ticker")
    assert t.at["A", "ret_5"] == pytest.approx(0.0)
    assert pd.isna(t.at["B", "ret_5"]) and not t.at["B", "pending_5"]


def test_only_new_days_are_computed(tmp_path, monkeypatch):
    store = Store(tmp_path)
    for i in range(25):
        write_day(store, i, scenario(i))
    track.track_screen(SCREEN, "inc", store)

    derived, loads = [], []
    real_derive, real_load = track.add_derived, store.load_snapshot
    monkeypatch.setattr(track, "add_derived", lambda raw, today: derived.append(today) or real_derive(raw, today))
    monkeypatch.setattr(store, "load_snapshot", lambda d, columns=None: loads.append(d) or real_load(d, columns))
    track.track_screen(SCREEN, "inc", store)
    assert derived == [] and loads == []                               # nothing new: all from the cache

    write_day(store, 25, scenario(25))
    res = track.track_screen(SCREEN, "inc", store)
    assert derived == [day(25)] and set(loads) == {day(25)}
    assert res["sessions"] == 26


def test_a_rewritten_snapshot_is_redone(tmp_path):
    store = Store(tmp_path)
    for i in range(3):
        write_day(store, i, {"A": (10.0, OUT), "B": (10.0, IN)})
    assert track.track_screen(SCREEN, "rw", store)["entries"] == 0
    write_day(store, 2, {"A": (10.0, IN), "B": (10.0, IN)})            # the day's later fetch
    path = store.snapshot_path(day(2))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    t = track.track_screen(SCREEN, "rw", store)["table"]
    assert list(t["ticker"]) == ["A"]


def test_changed_filters_do_not_use_the_old_cache(tmp_path):
    store = Store(tmp_path)
    for i in range(8):
        write_day(store, i, scenario(i))
    assert track.track_screen(SCREEN, "same name", store)["entries"] > 0
    assert track.track_screen({"pe": "Under 10"}, "same name", store)["entries"] == 0


def test_spy_is_the_benchmark_when_known(tmp_path, monkeypatch):
    store = Store(tmp_path)
    for i in range(10):
        write_day(store, i, {"A": (100.0 * 1.02 ** i, IN if i >= 1 else OUT), "Z": (10.0, OUT)})
    spy = pd.DataFrame({"close": [100.0 * 1.01 ** i for i in range(-3, 10)]},
                       index=pd.DatetimeIndex([pd.Timestamp(day(i)) for i in range(-3, 10)]))
    monkeypatch.setattr(track.prices, "history", lambda t: spy if t == "SPY" else pd.DataFrame())
    h5 = track.track_screen(SCREEN, "spy", store)["horizons"][5]
    assert h5["benchmark"] == "SPY"
    assert h5["excess"] == pytest.approx((1.02 ** 5 - 1.01 ** 5) * 100, abs=0.01)


def test_universe_median_without_benchmark_prices(tmp_path, offline):
    store = Store(tmp_path)
    for i in range(10):
        write_day(store, i, {"A": (100.0 * 1.02 ** i, IN if i >= 1 else OUT), "Z": (10.0, OUT)})
    h5 = track.track_screen(SCREEN, "offline", store)["horizons"][5]
    assert h5["benchmark"] == track.UNIVERSE
    assert h5["excess"] == pytest.approx(((1.02 ** 5 - 1) * 100) / 2, abs=0.01)    # median of A and Z (0%)
    track.track_screen(SCREEN, "offline", store)
    assert offline == [list(track.BENCHMARKS)]                         # at most one update a day


def test_all_screens_includes_saved_screens(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PRESETS_PATH", tmp_path / "presets.json")
    P.save_user("mine", P.Preset(filters=SCREEN))
    store = Store(tmp_path)
    for i in range(3):
        write_day(store, i, scenario(i))
    res = track.track_all_screens(store)
    assert set(P.BUILTIN) | {"★ mine"} == set(res)
    assert res["★ mine"]["sessions"] == 3 and res["★ mine"]["few_sessions"]


def test_no_snapshots(tmp_path):
    res = track.track_screen(SCREEN, "empty", Store(tmp_path))
    assert res["entries"] == 0 and res["sessions"] == 0 and res["horizons"][5]["median"] is None


def test_a_damaged_cache_is_rebuilt(tmp_path):
    store = Store(tmp_path)
    for i in range(8):
        write_day(store, i, scenario(i))
    first = track.track_screen(SCREEN, "dmg", store)["entries"]
    for f in (tmp_path / "track").iterdir():
        f.write_text("{not json" if f.suffix == ".json" else "garbage", encoding="utf-8")
    assert track.track_screen(SCREEN, "dmg", store)["entries"] == first
