"""
Server lifecycle for the end user, who starts the screener by double-clicking run_screener.bat:

  stop_server()     the "Exit" button: stops Streamlit cleanly, so python exits and the black
                    console window closes by itself.
  IdleWatch         safety net: when no browser tab has been connected for IDLE_POLLS checks in a
                    row (15 minutes), the server stops, so a forgotten console does not keep
                    running (and the next launch picks up updates).

Idleness is counted in polls, not wall-clock time: after the PC wakes from sleep the clock jumps,
but the browser reconnects within seconds — the server must not be stopped by the jump.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

POLL_SECONDS = 30
IDLE_POLLS = 30                 # 30 x 30 s = 15 minutes without an open tab
FORCE_EXIT_AFTER = 5.0          # if a clean stop hangs, exit the process anyway


def stop_server(reason: str, delay: float = 1.5) -> None:
    """Stop the Streamlit server after `delay` seconds (time for the goodbye page to reach the browser)."""
    log.info("stopping server: %s", reason)

    def _stop() -> None:
        hard = threading.Timer(FORCE_EXIT_AFTER, os._exit, args=(0,))
        hard.daemon = True
        hard.start()
        try:
            from streamlit.runtime import Runtime
            Runtime.instance().stop()           # documented as safe from any thread
        except Exception:
            log.exception("clean stop failed")

    t = threading.Timer(delay, _stop)
    t.daemon = True
    t.start()


def active_sessions() -> int | None:
    """Connected browser tabs, or None when Streamlit's (internal) session manager is unavailable."""
    try:
        from streamlit.runtime import Runtime
        return Runtime.instance()._session_mgr.num_active_sessions()
    except Exception:
        return None


class IdleWatch:
    """Counts consecutive polls with no open tab; tick() returns True when the server should stop."""

    def __init__(self, idle_polls: int = IDLE_POLLS):
        self.idle_polls = idle_polls
        self.idle = 0

    def tick(self, sessions: int | None) -> bool:
        if sessions is None:            # cannot tell — never stop on a guess
            self.idle = 0
            return False
        self.idle = 0 if sessions > 0 else self.idle + 1
        return self.idle >= self.idle_polls


def start_idle_watchdog(count: Callable[[], int | None] = active_sessions,
                        stop: Callable[[str], None] = stop_server,
                        poll: float = POLL_SECONDS, idle_polls: int = IDLE_POLLS) -> threading.Thread:
    watch = IdleWatch(idle_polls)

    def run() -> None:
        while True:
            time.sleep(poll)
            if watch.tick(count()):
                stop(f"no browser tab for {idle_polls} checks")
                return

    t = threading.Thread(target=run, name="screener-idle-watchdog", daemon=True)
    t.start()
    log.info("idle watchdog started (%d x %gs)", idle_polls, poll)
    return t

