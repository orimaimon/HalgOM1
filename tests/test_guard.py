"""Tests for screener/data/guard.py."""
import datetime as dt
import json
import threading
import time
from zoneinfo import ZoneInfo

import pytest

from screener.data import guard
from screener.data.guard import Guard, Wait, call, status, wait_message


class FakeClock:
    def __init__(self, start: dt.datetime | None = None):
        self.now_dt = start or dt.datetime(2026, 10, 20, 10, 0, 0, tzinfo=dt.timezone.utc)

    def __call__(self) -> dt.datetime:
        return self.now_dt

    def advance(self, **kwargs):
        self.now_dt += dt.timedelta(**kwargs)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("screener.data.store.DATA_DIR", data_dir)
    return data_dir


def test_unknown_provider(tmp_data_dir):
    with pytest.raises(KeyError):
        call("nonexistent_provider", lambda: 42)

    with pytest.raises(KeyError):
        status("nonexistent_provider")


def test_block_429(tmp_data_dir):
    clock = FakeClock()
    calls = []

    def failing_fn():
        calls.append(clock())
        raise Exception("HTTP 429 Too Many Requests")

    # First call fails with 429
    with pytest.raises(Exception) as exc_info:
        call("tradingview", failing_fn, now=clock)
    assert "429" in str(exc_info.value)
    assert len(calls) == 1

    # Immediate second call raises Wait without executing failing_fn
    with pytest.raises(Wait) as wait_info:
        call("tradingview", failing_fn, now=clock)
    w = wait_info.value
    assert w.provider == "tradingview"
    assert w.reason == "block"
    assert w.until == clock() + dt.timedelta(minutes=60)
    assert len(calls) == 1

    # 59 minutes later: still blocked
    clock.advance(minutes=59)
    with pytest.raises(Wait):
        call("tradingview", failing_fn, now=clock)
    assert len(calls) == 1

    # 60 minutes later: fn is called again
    clock.advance(minutes=1)

    def ok_fn():
        calls.append(clock())
        return "success"

    res = call("tradingview", ok_fn, now=clock)
    assert res == "success"
    assert len(calls) == 2


def test_yfinance_rate_limit_block(tmp_data_dir):
    clock = FakeClock()

    class YFRateLimitError(Exception):
        pass

    def failing_fn():
        raise YFRateLimitError("rate limited")

    with pytest.raises(YFRateLimitError):
        call("yahoo", failing_fn, now=clock)

    with pytest.raises(Wait) as wait_info:
        call("yahoo", failing_fn, now=clock)
    assert wait_info.value.reason == "block"
    assert wait_info.value.until == clock() + dt.timedelta(minutes=60)


def test_backoff_sequence(tmp_data_dir):
    clock = FakeClock()
    calls = 0

    def failing_fn():
        nonlocal calls
        calls += 1
        raise ValueError("temporary error")

    # 1st failure -> 2 min backoff
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    assert calls == 1

    with pytest.raises(Wait) as w1:
        call("sec", failing_fn, now=clock)
    assert w1.value.reason == "backoff"
    assert w1.value.until == clock() + dt.timedelta(minutes=2)

    # 2nd failure -> 5 min backoff
    clock.advance(minutes=2)
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    assert calls == 2

    with pytest.raises(Wait) as w2:
        call("sec", failing_fn, now=clock)
    assert w2.value.until == clock() + dt.timedelta(minutes=5)

    # 3rd failure -> 15 min backoff
    clock.advance(minutes=5)
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    assert calls == 3

    with pytest.raises(Wait) as w3:
        call("sec", failing_fn, now=clock)
    assert w3.value.until == clock() + dt.timedelta(minutes=15)

    # 4th failure -> 60 min backoff
    clock.advance(minutes=15)
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    assert calls == 4

    with pytest.raises(Wait) as w4:
        call("sec", failing_fn, now=clock)
    assert w4.value.until == clock() + dt.timedelta(minutes=60)

    # 5th failure -> 60 min backoff
    clock.advance(minutes=60)
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    assert calls == 5

    with pytest.raises(Wait) as w5:
        call("sec", failing_fn, now=clock)
    assert w5.value.until == clock() + dt.timedelta(minutes=60)

    # Success resets backoff
    clock.advance(minutes=60)
    res = call("sec", lambda: "ok", now=clock)
    assert res == "ok"

    # Advance past min_interval before trying failing_fn again
    clock.advance(seconds=1)
    with pytest.raises(ValueError):
        call("sec", failing_fn, now=clock)
    with pytest.raises(Wait) as w_reset:
        call("sec", failing_fn, now=clock)
    assert w_reset.value.until == clock() + dt.timedelta(minutes=2)


def test_daily_cap_and_reset(tmp_data_dir):
    # nasdaq daily cap is 40, min_interval is 900s
    clock = FakeClock(dt.datetime(2026, 10, 20, 0, 1, 0, tzinfo=ZoneInfo("America/New_York")))
    cap = guard.LIMITS["nasdaq"].daily_cap

    for _ in range(cap):
        res = call("nasdaq", lambda: "ok", now=clock)
        assert res == "ok"
        clock.advance(seconds=900)

    # 41st call hit cap
    with pytest.raises(Wait) as wait_info:
        call("nasdaq", lambda: "ok", now=clock)

    w = wait_info.value
    assert w.reason == "daily_cap"
    expected_until = dt.datetime(2026, 10, 21, 0, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    assert w.until == expected_until

    # Advance clock to next NY day
    clock.now_dt = dt.datetime(2026, 10, 21, 0, 0, 1, tzinfo=ZoneInfo("America/New_York"))
    res = call("nasdaq", lambda: "ok", now=clock)
    assert res == "ok"


def test_min_interval_and_max_sleep(tmp_data_dir, monkeypatch):
    clock = FakeClock()
    # sec min_interval is 0.2s

    call("sec", lambda: 1, now=clock)

    # Call immediately without max_sleep -> raises Wait
    with pytest.raises(Wait) as wait_info:
        call("sec", lambda: 2, now=clock, max_sleep=0)

    assert wait_info.value.reason == "min_interval"

    # Call with max_sleep >= remaining -> sleeps and then executes
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    # Advance clock by 0.05s, remaining = 0.15s
    clock.advance(seconds=0.05)
    res = call("sec", lambda: 3, now=clock, max_sleep=0.5)
    assert res == 3
    assert len(slept) == 1
    assert abs(slept[0] - 0.15) < 1e-4


def test_persistence_and_damaged_json(tmp_data_dir, caplog):
    clock = FakeClock()

    g1 = Guard(now=clock)
    g1.call("fred", lambda: "ok_fred")

    # Verify state saved to file
    json_file = tmp_data_dir / "guard.json"
    assert json_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["fred"]["count_today"] == 1

    # New Guard instance sees existing state
    g2 = Guard(now=clock)
    st = g2.status("fred")
    assert st["count_today"] == 1

    # Reloaded module sees existing state
    st_mod = status("fred", now=clock)
    assert st_mod["count_today"] == 1

    # Damaged JSON handling
    json_file.write_text("{invalid_json...", encoding="utf-8")
    st_damaged = status("fred", now=clock)
    assert st_damaged["count_today"] == 0
    assert "cannot read guard state" in caplog.text


def test_concurrency(tmp_data_dir):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_fn():
        nonlocal active, max_active
        with lock:
            active += 1
            if active > max_active:
                max_active = active
        time.sleep(0.02)
        with lock:
            active -= 1
        return "done"

    threads = []
    # Use real clock with max_sleep=5s for provider sec (min_interval 0.2s)
    for _ in range(5):
        t = threading.Thread(target=call, args=("sec", slow_fn), kwargs={"max_sleep": 5.0})
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert max_active == 1


def test_wait_message():
    block_until = dt.datetime(2026, 10, 20, 14, 30, tzinfo=dt.timezone.utc)
    w_block = Wait("tradingview", block_until, "block")
    msg_block = wait_message(w_block)
    assert "TradingView חסם זמנית — מוצגים נתונים שמורים." in msg_block
    assert "ננסה שוב ב-" in msg_block

    w_backoff = Wait("sec", block_until, "backoff")
    msg_backoff = wait_message(w_backoff)
    assert msg_backoff.startswith("ננסה שוב ב-")
    assert "חסם זמנית" not in msg_backoff
