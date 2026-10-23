import datetime as dt
import importlib
import json
import threading
import time
import pytest

from screener.data import guard


class FakeClock:
    def __init__(self, start_ts=1700000000.0):  # e.g. 2023-11-14 ~17:33 UTC
        self.now_ts = start_ts

    def now(self) -> float:
        return self.now_ts

    def sleep(self, seconds: float) -> None:
        self.now_ts += seconds


@pytest.fixture
def fake_guard(tmp_path, monkeypatch):
    clock = FakeClock()
    state_path = tmp_path / "guard.json"
    monkeypatch.setattr(guard, "STATE_PATH", state_path)
    monkeypatch.setattr(guard, "_now", clock.now)
    monkeypatch.setattr(guard, "_sleep", clock.sleep)
    return clock, state_path


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        guard.call("unknown_provider", lambda: "ok")
    with pytest.raises(ValueError, match="Unknown provider"):
        guard.status("unknown_provider")


def test_success_and_interval(fake_guard):
    clock, _ = fake_guard

    res = guard.call("yahoo", lambda: "result_1")
    assert res == "result_1"

    # Second call within min_interval (yahoo min_interval = 1.0s)
    with pytest.raises(guard.Wait) as exc_info:
        guard.call("yahoo", lambda: "result_2")
    assert exc_info.value.reason == "interval"
    assert exc_info.value.provider == "yahoo"

    # With max_sleep enough to cover remaining time (e.g. max_sleep=2.0)
    res2 = guard.call("yahoo", lambda: "result_2", max_sleep=2.0)
    assert res2 == "result_2"


def test_backoff_exponential_and_reset(fake_guard):
    clock, _ = fake_guard

    def fail_fn():
        raise RuntimeError("failed request")

    # Attempt 1 -> fails -> backoff 2 minutes (120s)
    with pytest.raises(RuntimeError, match="failed request"):
        guard.call("yahoo", fail_fn)

    # Immediate next call -> Wait("backoff")
    with pytest.raises(guard.Wait) as exc:
        guard.call("yahoo", lambda: "ok")
    assert exc.value.reason == "backoff"

    # Advance 119s -> still backoff
    clock.sleep(119)
    with pytest.raises(guard.Wait):
        guard.call("yahoo", lambda: "ok")

    # Advance 2s (total 121s) -> backoff expired, attempt 2 fails -> backoff 5 minutes (300s)
    clock.sleep(2)
    with pytest.raises(RuntimeError):
        guard.call("yahoo", fail_fn)

    # Advance 301s -> attempt 3 fails -> backoff 15 minutes (900s)
    clock.sleep(301)
    with pytest.raises(RuntimeError):
        guard.call("yahoo", fail_fn)

    # Advance 901s -> attempt 4 fails -> backoff 60 minutes (3600s)
    clock.sleep(901)
    with pytest.raises(RuntimeError):
        guard.call("yahoo", fail_fn)

    # Advance 3601s -> attempt 5 fails -> backoff 60 minutes again
    clock.sleep(3601)
    with pytest.raises(RuntimeError):
        guard.call("yahoo", fail_fn)

    st = guard.status("yahoo")
    assert st["failures"] == 5

    # Advance 3601s -> attempt 6 succeeds -> failures reset
    clock.sleep(3601)
    res = guard.call("yahoo", lambda: "ok_after_fail")
    assert res == "ok_after_fail"

    st_after = guard.status("yahoo")
    assert st_after["failures"] == 0
    assert st_after["blocked_until"] is None


class DummyHTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class DummyResponseError(Exception):
    def __init__(self, status_code):
        class Resp:
            pass
        self.response = Resp()
        self.response.status_code = status_code


class YFRateLimitError(Exception):
    pass


def test_blocked_errors_straight_to_60m(fake_guard):
    clock, _ = fake_guard

    # 429
    with pytest.raises(DummyHTTPError):
        guard.call("yahoo", lambda: (_ for _ in ()).throw(DummyHTTPError(429)))

    st = guard.status("yahoo")
    assert st["blocked_until"] == clock.now() + 3600.0

    # 403 via response.status_code
    clock.sleep(3601)
    with pytest.raises(DummyResponseError):
        guard.call("yahoo", lambda: (_ for _ in ()).throw(DummyResponseError(403)))

    st = guard.status("yahoo")
    assert st["blocked_until"] == clock.now() + 3600.0

    # YFRateLimitError
    clock.sleep(3601)
    with pytest.raises(YFRateLimitError):
        guard.call("yahoo", lambda: (_ for _ in ()).throw(YFRateLimitError()))

    st = guard.status("yahoo")
    assert st["blocked_until"] == clock.now() + 3600.0


def test_daily_cap_and_ny_midnight_reset(fake_guard):
    clock, _ = fake_guard

    # tradingview_market daily cap is 150
    # Make 150 successful calls, sleeping min_interval each time
    for i in range(150):
        res = guard.call("tradingview_market", lambda: i)
        assert res == i
        clock.sleep(121)

    # 151st call raises Wait("cap")
    with pytest.raises(guard.Wait) as exc:
        guard.call("tradingview_market", lambda: 151)
    assert exc.value.reason == "cap"

    # Advance time to next NY midnight
    ny_now = dt.datetime.fromtimestamp(clock.now(), tz=dt.timezone.utc).astimezone(guard.NY_TZ)
    next_ny_day = ny_now.date() + dt.timedelta(days=1)
    next_midnight = dt.datetime(
        next_ny_day.year, next_ny_day.month, next_ny_day.day, 0, 0, 0, tzinfo=guard.NY_TZ
    )
    sleep_needed = (next_midnight - ny_now).total_seconds() + 1.0
    clock.sleep(sleep_needed)

    # Now call should succeed
    res = guard.call("tradingview_market", lambda: "new_day")
    assert res == "new_day"


def test_state_survives_restart(fake_guard, monkeypatch):
    clock, state_path = fake_guard

    with pytest.raises(RuntimeError):
        guard.call("fred", lambda: (_ for _ in ()).throw(RuntimeError("err")))

    assert state_path.exists()

    # Simulate reload
    importlib.reload(guard)
    monkeypatch.setattr(guard, "STATE_PATH", state_path)
    monkeypatch.setattr(guard, "_now", clock.now)
    monkeypatch.setattr(guard, "_sleep", clock.sleep)

    with pytest.raises(guard.Wait) as exc:
        guard.call("fred", lambda: "ok")
    assert exc.value.reason == "backoff"


def test_damaged_guard_json(fake_guard):
    clock, state_path = fake_guard

    state_path.write_text("corrupted json {", encoding="utf-8")

    res = guard.call("fred", lambda: "recovered")
    assert res == "recovered"


def test_concurrent_threads(fake_guard):
    clock, _ = fake_guard

    fn_call_count = 0
    fn_lock = threading.Lock()

    def slow_fn():
        nonlocal fn_call_count
        with fn_lock:
            fn_call_count += 1
        time.sleep(0.05)
        return "done"

    results = []
    exceptions = []

    def worker():
        try:
            r = guard.call("sec", slow_fn)
            results.append(r)
        except Exception as e:
            exceptions.append(e)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert fn_call_count == 1
    assert len(results) == 1
    assert len(exceptions) == 1
    assert isinstance(exceptions[0], guard.Wait)
    assert exceptions[0].reason == "interval"


def test_wait_message_formatting():
    dt_until = dt.datetime(2023, 11, 14, 14, 35, tzinfo=dt.timezone.utc)
    local_time_str = dt_until.astimezone().strftime("%H:%M")

    w_backoff = guard.Wait("tradingview", dt_until, "backoff")
    msg_backoff = guard.wait_message(w_backoff)
    assert f"TradingView חסם זמנית — מוצגים נתונים שמורים. ננסה שוב ב-{local_time_str}" in msg_backoff

    w_interval = guard.Wait("tradingview_market", dt_until, "interval")
    msg_interval = guard.wait_message(w_interval)
    assert f"TradingView חסם זמנית — מוצגים נתונים שמורים. ננסה שוב ב-{local_time_str}" in msg_interval

    w_cap = guard.Wait("nasdaq", dt_until, "cap")
    msg_cap = guard.wait_message(w_cap)
    assert f"הגענו למכסת הבקשות היומית של Nasdaq — מוצגים נתונים שמורים. ננסה שוב ב-{local_time_str}" in msg_cap
