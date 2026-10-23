"""
Server-wide request guard: rate limits, daily caps, and backoff per provider.
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
from zoneinfo import ZoneInfo

from screener.data import store

log = logging.getLogger(__name__)

_now = time.time
_sleep = time.sleep

STATE_LOCK = threading.Lock()
STATE_PATH = store.DATA_DIR / "guard.json"
NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Limit:
    min_interval: float
    daily_cap: int


LIMITS: dict[str, Limit] = {
    "tradingview": Limit(120.0, 150),
    "tradingview_market": Limit(120.0, 150),
    "nasdaq": Limit(600.0, 30),
    "yahoo": Limit(1.0, 300),
    "sec": Limit(0.2, 2000),
    "massive": Limit(12.0, 600),
    "fred": Limit(60.0, 50),
}

PROVIDER_LOCKS = {p: threading.Lock() for p in LIMITS}
BACKOFF = [120.0, 300.0, 900.0, 3600.0]

DISPLAY_NAMES = {
    "tradingview": "TradingView",
    "tradingview_market": "TradingView",
    "nasdaq": "Nasdaq",
    "yahoo": "Yahoo",
    "sec": "SEC",
    "massive": "Massive",
    "fred": "FRED",
}


class Wait(Exception):
    def __init__(self, provider: str, until: dt.datetime, reason: str):
        self.provider = provider
        self.until = until
        self.reason = reason
        super().__init__(f"{provider} wait until {until} ({reason})")


def is_blocked_error(exc: Exception) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__ == "YFRateLimitError":
            return True

    code = getattr(exc, "status_code", None)
    if code is None and hasattr(exc, "response"):
        code = getattr(exc.response, "status_code", None)

    return code in (403, 429)


def _load_state() -> dict:
    path = STATE_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        log.warning("guard state is not a dict: %s", path)
        return {}
    except Exception as e:
        log.warning("cannot read guard state: %s", e)
        return {}


def _save_state(state: dict) -> None:
    path = STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        log.warning("cannot save guard state: %s", e)


def call(provider: str, fn, *args, max_sleep: float = 0.0, **kwargs):
    if provider not in LIMITS:
        raise ValueError(f"Unknown provider: {provider}")

    limit = LIMITS[provider]
    lock = PROVIDER_LOCKS[provider]

    with lock:
        now = _now()
        with STATE_LOCK:
            state = _load_state()

        p_state = state.get(provider, {})
        last_attempt = p_state.get("last_attempt")
        failures = p_state.get("failures", 0)
        blocked_until = p_state.get("blocked_until")
        stored_day = p_state.get("day")
        count = p_state.get("count", 0)

        ny_now = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).astimezone(NY_TZ)
        ny_day_str = ny_now.date().isoformat()

        if stored_day != ny_day_str:
            count = 0
            stored_day = ny_day_str

        # Check backoff
        if blocked_until is not None and now < blocked_until:
            until_dt = dt.datetime.fromtimestamp(blocked_until, tz=dt.timezone.utc)
            raise Wait(provider=provider, until=until_dt, reason="backoff")

        # Check daily cap
        if count >= limit.daily_cap:
            next_ny_day = ny_now.date() + dt.timedelta(days=1)
            next_ny_midnight_dt = dt.datetime(
                next_ny_day.year, next_ny_day.month, next_ny_day.day, 0, 0, 0, tzinfo=NY_TZ
            )
            raise Wait(provider=provider, until=next_ny_midnight_dt, reason="cap")

        # Check min interval
        if last_attempt is not None and now < last_attempt + limit.min_interval:
            remaining = (last_attempt + limit.min_interval) - now
            if remaining <= max_sleep:
                _sleep(remaining)
                now = _now()
                ny_now = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).astimezone(NY_TZ)
                ny_day_str = ny_now.date().isoformat()
                if stored_day != ny_day_str:
                    count = 0
                    stored_day = ny_day_str
            else:
                until_dt = dt.datetime.fromtimestamp(last_attempt + limit.min_interval, tz=dt.timezone.utc)
                raise Wait(provider=provider, until=until_dt, reason="interval")

        # Record attempt
        count += 1
        last_attempt = now
        p_state["last_attempt"] = last_attempt
        p_state["failures"] = failures
        p_state["blocked_until"] = blocked_until
        p_state["day"] = stored_day
        p_state["count"] = count
        state[provider] = p_state

        with STATE_LOCK:
            _save_state(state)

        log.info("guard: %s %d/%d today", provider, count, limit.daily_cap)

        try:
            res = fn(*args, **kwargs)
        except Exception as exc:
            err_now = _now()
            failures += 1
            if is_blocked_error(exc):
                blocked_until = err_now + 3600.0
            else:
                backoff_secs = BACKOFF[min(failures - 1, 3)]
                blocked_until = err_now + backoff_secs

            p_state["failures"] = failures
            p_state["blocked_until"] = blocked_until
            state[provider] = p_state

            with STATE_LOCK:
                _save_state(state)

            log.warning("guard: %s request failed: %s", provider, exc)
            raise

        # On success
        p_state["failures"] = 0
        p_state["blocked_until"] = None
        state[provider] = p_state

        with STATE_LOCK:
            _save_state(state)

        return res


def status(provider: str) -> dict:
    if provider not in LIMITS:
        raise ValueError(f"Unknown provider: {provider}")

    limit = LIMITS[provider]
    now = _now()

    with STATE_LOCK:
        state = _load_state()

    p_state = state.get(provider, {})
    last_attempt = p_state.get("last_attempt")
    failures = p_state.get("failures", 0)
    blocked_until = p_state.get("blocked_until")
    stored_day = p_state.get("day")
    count = p_state.get("count", 0)

    ny_now = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).astimezone(NY_TZ)
    ny_day_str = ny_now.date().isoformat()

    if stored_day != ny_day_str:
        count = 0

    candidates = [now]
    if blocked_until is not None and blocked_until > now:
        candidates.append(blocked_until)
    if last_attempt is not None and (last_attempt + limit.min_interval) > now:
        candidates.append(last_attempt + limit.min_interval)
    if count >= limit.daily_cap:
        next_ny_day = ny_now.date() + dt.timedelta(days=1)
        next_ny_midnight_dt = dt.datetime(
            next_ny_day.year, next_ny_day.month, next_ny_day.day, 0, 0, 0, tzinfo=NY_TZ
        )
        candidates.append(next_ny_midnight_dt.timestamp())

    next_allowed = max(candidates)

    return {
        "last_attempt": last_attempt,
        "failures": failures,
        "blocked_until": blocked_until,
        "count_today": count,
        "daily_cap": limit.daily_cap,
        "next_allowed": next_allowed,
    }


def wait_message(w: Wait) -> str:
    name = DISPLAY_NAMES.get(w.provider, w.provider)
    local_until = w.until.astimezone()
    time_str = local_until.strftime("%H:%M")
    if w.reason in ("backoff", "interval"):
        return f"{name} חסם זמנית — מוצגים נתונים שמורים. ננסה שוב ב-{time_str}"
    elif w.reason == "cap":
        return f"הגענו למכסת הבקשות היומית של {name} — מוצגים נתונים שמורים. ננסה שוב ב-{time_str}"
    return f"{name} אינו זמין. ננסה שוב ב-{time_str}"
