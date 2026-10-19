import datetime as dt
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from screener import export
from screener import history
from screener import lifecycle
from screener.data.store import Store, _atomic_write_text

# 1. file_stem
def test_file_stem_keeps_hebrew():
    when = dt.datetime(2026, 10, 18, 15, 30)
    # Hebrew name with spaces
    title = "המסך שלי 1"
    stem = export.file_stem(title, when)
    assert stem == "2026-10-18_1530_המסך שלי 1"

    # Strip invalid windows chars
    title_invalid = 'my <screen> : "best" / \\ | ? * \x01'
    stem_invalid = export.file_stem(title_invalid, when)
    assert stem_invalid == "2026-10-18_1530_my -screen- - -best- - - - - -"

# 2. lifecycle stop_server
def test_stop_server_starts_timer_before_stopping():
    stops = []
    timers = []

    class FakeRuntimeInstance:
        def stop(self):
            stops.append("stop_called")
            # Simulate a hang
            raise RuntimeError("hanging")

    class FakeRuntime:
        @classmethod
        def instance(cls):
            return FakeRuntimeInstance()

    class FakeTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or ()
            self.daemon = False

        def start(self):
            timers.append(self.function)

    with patch("screener.lifecycle.threading.Timer", FakeTimer):
        with patch.dict("sys.modules", {"streamlit.runtime": type("FakeMod", (), {"Runtime": FakeRuntime})}):
            lifecycle.stop_server("test", delay=0.0)

            # Find the _stop function created by stop_server
            stop_func = timers[0]
            timers.clear()

            # Execute _stop
            stop_func()

            assert "stop_called" in stops
            # The hard timer should have been started even though stop() raised an exception
            import os
            assert os._exit in timers

# 3. atomic_write_text
def test_atomic_write_text(tmp_path):
    target = tmp_path / "test.json"
    _atomic_write_text('{"ok": true}', target)
    assert target.exists()
    assert json.loads(target.read_text()) == {"ok": True}

    # Verify it doesn't leave .tmp files behind
    assert not list(tmp_path.glob("*.tmp"))

# 4. timezone edge case across local midnight but same NY day
def test_timezone_across_midnight(tmp_path):
    import zoneinfo
    ny_tz = zoneinfo.ZoneInfo("America/New_York")
    il_tz = zoneinfo.ZoneInfo("Asia/Jerusalem")
    store = Store(tmp_path)

    # Simulate a fetch at Friday 11:30 PM Israel time (Friday 4:30 PM NY time)
    fetch1_time = dt.datetime(2026, 10, 16, 23, 30, tzinfo=il_tz)
    assert fetch1_time.astimezone(ny_tz).date() == dt.date(2026, 10, 16)

    df1 = pd.DataFrame({"ticker": ["A", "B"], "price": [10, 20]})
    df1.attrs["source"] = {}
    store.save(df1, fetch1_time)

    # Verify the snapshot saved is for the 16th
    assert store.snapshot_dates() == [dt.date(2026, 10, 16)]

    # Simulate a fetch at Saturday 1:30 AM Israel time (Friday 6:30 PM NY time)
    # Same NY trading day!
    fetch2_time = dt.datetime(2026, 10, 17, 1, 30, tzinfo=il_tz)
    assert fetch2_time.astimezone(ny_tz).date() == dt.date(2026, 10, 16)

    df2 = pd.DataFrame({"ticker": ["A", "B"], "price": [11, 21]})
    df2.attrs["source"] = {}
    store.save(df2, fetch2_time)

    # Verify the snapshot saved is STILL for the 16th (overwritten)
    assert store.snapshot_dates() == [dt.date(2026, 10, 16)]

    # Now, history.load_previous is called with today = dt.date(2026, 10, 17)
    # It should look at the latest cache, extract the NY date (Oct 16), and look for
    # the snapshot strictly before Oct 16. It should NOT return Oct 16!
    prev = history.load_previous(store, dt.date(2026, 10, 17))
    assert prev is None # No snapshot before the 16th exists in our setup

    # Let's add a Thursday snapshot
    df0 = pd.DataFrame({"ticker": ["A", "B"], "price": [9, 19]})
    df0.attrs["source"] = {}
    fetch0_time = dt.datetime(2026, 10, 15, 23, 30, tzinfo=il_tz)
    store.save(df0, fetch0_time)

    # Re-save Friday to update the latest cache
    store.save(df2, fetch2_time)

    prev = history.load_previous(store, dt.date(2026, 10, 17))
    assert prev is not None
    assert prev.day == dt.date(2026, 10, 15)
