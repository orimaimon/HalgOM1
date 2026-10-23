"""
Request guard: single-flight per provider, daily cap, min interval, block & error backoff.

Limits per provider:
  tradingview        min_interval=60 s, daily_cap=200
  tradingview_market min_interval=60 s, daily_cap=200
  nasdaq             min_interval=900 s, daily_cap=40
  yahoo              min_interval=1 s, daily_cap=300
  sec                min_interval=0.2 s, daily_cap=2000
  massive            min_interval=13 s, daily_cap=300
  fred               min_interval=60 s, daily_cap=50

State persisted in _data/guard.json (resolved dynamically at call time from screener.data.store.DATA_DIR).
Every attempt counts toward the daily cap, failed ones too (the request was sent). After a failure the
provider waits 2, 5, 15, then 60 minutes; after a block (403/429) 60 minutes straight away.
Not wired into any provider yet.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo

from screener.data.store import market_date

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Limits:
    min_interval: float
    daily_cap: int


LIMITS: dict[str, Limits] = {
    "tradingview": Limits(min_interval=60.0, daily_cap=200),
    "tradingview_market": Limits(min_interval=60.0, daily_cap=200),
    "nasdaq": Limits(min_interval=900.0, daily_cap=40),
    "yahoo": Limits(min_interval=1.0, daily_cap=300),
    "sec": Limits(min_interval=0.2, daily_cap=2000),
    "massive": Limits(min_interval=13.0, daily_cap=300),
    "fred": Limits(min_interval=60.0, daily_cap=50),
}

BACKOFF = [2, 5, 15, 60]  # minutes

DISPLAY_NAMES: dict[str, str] = {
    "tradingview": "TradingView",
    "tradingview_market": "TradingView",
    "nasdaq": "Nasdaq",
    "yahoo": "Yahoo",
    "sec": "SEC",
    "massive": "Massive",
    "fred": "FRED",
}


class Wait(Exception):
    """Raised when a request cannot proceed due to backoff, daily cap, or min interval."""

    def __init__(self, provider: str, until: dt.datetime, reason: str):
        self.provider = provider
        self.until = until
        self.reason = reason
        super().__init__(f"Wait for {provider} until {until} ({reason})")


_STATE_LOCK = threading.Lock()
_PROVIDER_LOCKS: dict[str, threading.Lock] = {provider: threading.Lock() for provider in LIMITS}


def _get_json_path() -> Path:
    from screener.data.store import DATA_DIR
    return Path(DATA_DIR) / "guard.json"


def _is_block(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", None)
    if status_code in (403, 429):
        return True

    cls_name = exc.__class__.__name__
    if cls_name == "YFRateLimitError":
        return True

    msg = str(exc)
    if "429" in msg or "too many requests" in msg.lower():
        return True

    return False


def _default_now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def _parse_dt(iso_str: str | None) -> dt.datetime | None:
    if not iso_str:
        return None
    try:
        return dt.datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return None


def _format_dt(dt_val: dt.datetime | None) -> str | None:
    return dt_val.isoformat() if dt_val is not None else None


def _load_state() -> dict[str, dict[str, Any]]:
    path = _get_json_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Root JSON is not a dict")
        return data
    except Exception as e:
        log.warning("cannot read guard state from %s: %s", path, e)
        return {}


def _save_state(data: dict[str, dict[str, Any]]) -> None:
    path = _get_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        text = json.dumps(data, indent=2)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class Guard:
    def __init__(self, now: Callable[[], dt.datetime] | None = None):
        self._now_fn = now

    def _get_now(self, override_now: Callable[[], dt.datetime] | None = None) -> dt.datetime:
        fn = override_now or self._now_fn or _default_now
        return fn()

    def call(self, provider: str, fn: Callable[..., T], *args: Any, max_sleep: float = 0,
             now: Callable[[], dt.datetime] | None = None, **kwargs: Any) -> T:
        if provider not in LIMITS:
            raise KeyError(provider)

        lock = _PROVIDER_LOCKS[provider]
        limit = LIMITS[provider]

        with lock:
            # The provider lock is held from here on, so only other providers can change the state
            # between the check and the attempt: sleeping outside _STATE_LOCK is safe.
            slept = False
            while True:
                now_dt = self._get_now(now)
                with _STATE_LOCK:
                    state = _load_state()
                    pstate = state.get(provider, {})
                    remaining = self._check(provider, pstate, limit, now_dt)
                    if remaining <= 0 or slept:
                        # Count the attempt before calling fn: a request that crashes or fails was sent too.
                        pstate["last_call"] = _format_dt(now_dt)
                        pstate["count_today"] = pstate.get("count_today", 0) + 1
                        state[provider] = pstate
                        _save_state(state)
                        log.info("guard: %s %d/%d today", provider, pstate["count_today"], limit.daily_cap)
                        break
                if remaining > max_sleep:
                    raise Wait(provider, now_dt + dt.timedelta(seconds=remaining), "min_interval")
                time.sleep(remaining)
                slept = True

            try:
                res = fn(*args, **kwargs)
            except Exception as exc:
                now_dt_fail = self._get_now(now)
                with _STATE_LOCK:
                    state = _load_state()
                    pstate = state.get(provider, {})
                    pstate["last_error"] = str(exc)

                    failures = pstate.get("failures", 0) + 1
                    pstate["failures"] = failures
                    if _is_block(exc):
                        until = now_dt_fail + dt.timedelta(minutes=BACKOFF[-1])
                        pstate["wait_reason"] = "block"
                    else:
                        until = now_dt_fail + dt.timedelta(minutes=BACKOFF[min(failures - 1, len(BACKOFF) - 1)])
                        pstate["wait_reason"] = "backoff"
                    pstate["wait_until"] = _format_dt(until)

                    state[provider] = pstate
                    _save_state(state)
                    log.warning("guard: %s failed (%s), waiting until %s", provider, exc, until)
                raise

            now_dt_ok = self._get_now(now)
            with _STATE_LOCK:
                state = _load_state()
                pstate = state.get(provider, {})
                pstate["failures"] = 0
                pstate["wait_until"] = None
                pstate["wait_reason"] = None
                pstate["last_ok"] = _format_dt(now_dt_ok)
                state[provider] = pstate
                _save_state(state)

            return res

    @staticmethod
    def _check(provider: str, pstate: dict[str, Any], limit: Limits, now_dt: dt.datetime) -> float:
        """Raise Wait for a backoff or the daily cap; return the seconds left of min_interval (<= 0: go).
        Updates pstate's day and count when a new New York day starts.

        A clock moved back (Windows clock fix, wrong time zone) must not stall a provider for days: a
        stored wait longer than the longest backoff is dropped, and a last call "in the future" counts
        as just now."""
        wait_until_dt = _parse_dt(pstate.get("wait_until"))
        if wait_until_dt is not None and now_dt < wait_until_dt:
            if wait_until_dt - now_dt <= dt.timedelta(minutes=BACKOFF[-1]):
                raise Wait(provider, wait_until_dt, pstate.get("wait_reason") or "backoff")
            log.warning("guard: %s wait until %s is beyond the longest backoff (clock moved back?); dropped",
                        provider, wait_until_dt)
            pstate["wait_until"] = None

        ny_today = market_date(now_dt)
        if pstate.get("day") != ny_today.isoformat():
            pstate["count_today"] = 0
            pstate["day"] = ny_today.isoformat()

        if pstate.get("count_today", 0) >= limit.daily_cap:
            nxt = ny_today + dt.timedelta(days=1)
            raise Wait(provider, dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=ZoneInfo("America/New_York")),
                       "daily_cap")

        last_call_dt = _parse_dt(pstate.get("last_call"))
        if last_call_dt is None:
            return 0.0
        elapsed = max((now_dt - last_call_dt).total_seconds(), 0.0)
        return limit.min_interval - elapsed

    def status(self, provider: str, now: Callable[[], dt.datetime] | None = None) -> dict[str, Any]:
        if provider not in LIMITS:
            raise KeyError(provider)

        limit = LIMITS[provider]
        now_dt = self._get_now(now)

        with _STATE_LOCK:
            state = _load_state()
            pstate = state.get(provider, {})

            ny_today = market_date(now_dt)
            p_day_str = pstate.get("day")
            count_today = pstate.get("count_today", 0) if p_day_str == ny_today.isoformat() else 0

            return {
                "provider": provider,
                "count_today": count_today,
                "day": ny_today.isoformat(),
                "daily_cap": limit.daily_cap,
                "min_interval": limit.min_interval,
                "failures": pstate.get("failures", 0),
                "wait_until": pstate.get("wait_until"),
                "wait_reason": pstate.get("wait_reason"),
                "last_ok": pstate.get("last_ok"),
                "last_error": pstate.get("last_error"),
            }


_DEFAULT_GUARD = Guard()


def call(provider: str, fn: Callable[..., T], *args: Any, max_sleep: float = 0,
         now: Callable[[], dt.datetime] | None = None, **kwargs: Any) -> T:
    return _DEFAULT_GUARD.call(provider, fn, *args, max_sleep=max_sleep, now=now, **kwargs)


def status(provider: str, now: Callable[[], dt.datetime] | None = None) -> dict[str, Any]:
    return _DEFAULT_GUARD.status(provider, now=now)


def wait_message(w: Wait) -> str:
    time_str = w.until.astimezone().strftime("%H:%M") if w.until.tzinfo else w.until.strftime("%H:%M")
    disp = DISPLAY_NAMES.get(w.provider, w.provider.title())
    if w.reason == "block":
        return f"{disp} חסם זמנית — מוצגים נתונים שמורים. ננסה שוב ב-{time_str}"
    if w.reason == "daily_cap":
        return f"הגענו למכסת הבקשות היומית של {disp} — מוצגים נתונים שמורים. ננסה שוב ב-{time_str}"
    if w.reason == "backoff":
        return f"{disp} לא זמין כרגע — מוצגים נתונים שמורים. ננסה שוב ב-{time_str}"
    return f"מוצגים נתונים שמורים. העדכון הבא מ-{disp} ב-{time_str}"
