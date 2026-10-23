"""SEC EDGAR fundamentals (official, free): TTM revenue growth and net margin from XBRL frames.

One frame = one concept for all companies in one period (a single request). Frames are cached in
_data/sec/frames/; the result is _data/sec/fundamentals.parquet. Requests need a User-Agent with contact
details, read from %USERPROFILE%/.halgom/sec_user_agent.txt (never stored in the repository); at most
5 requests per second, one at a time. Not wired into the app yet.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

_sleep = time.sleep

MAX_STEP_BACK = 4          # quarters probed for the latest complete one
MIN_COMPANIES = 1000       # a quarter frame with fewer companies is not complete yet


class SecError(RuntimeError):
    """SEC answered 403/429 (blocked). `status_code` lets the request guard back off."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _default_dir() -> Path:
    from screener.data import store      # resolved at call time: tests point store.DATA_DIR elsewhere
    return store.DATA_DIR / "sec"


def _check_blocked(res: requests.Response, url: str) -> None:
    if res.status_code in (403, 429):
        raise SecError(f"SEC returned {res.status_code} for {url}", status_code=res.status_code)

REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
]
NET_INCOME_CONCEPT = "NetIncomeLoss"
ALL_CONCEPTS = REVENUE_CONCEPTS + [NET_INCOME_CONCEPT]

COLUMNS = [
    "ticker",
    "cik",
    "company",
    "revenue_ttm",
    "revenue_ttm_prev",
    "revenue_growth_ttm",
    "net_income_ttm",
    "net_margin_ttm",
    "last_period",
    "updated_at",
]


def get_user_agent(user_agent_path: Path | str | None = None) -> str | None:
    if user_agent_path is None:
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            ua_path = Path(user_profile) / ".halgom" / "sec_user_agent.txt"
        else:
            ua_path = Path.home() / ".halgom" / "sec_user_agent.txt"
    else:
        ua_path = Path(user_agent_path)

    if not ua_path.exists():
        return None

    try:
        content = ua_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            s = line.strip()
            if s:
                return s
    except OSError as e:
        log.warning("Failed to read SEC User-Agent file at %s: %s", ua_path, e)
        return None

    return None


def _normalize_ticker(ticker: str) -> str:
    return ticker.replace("-", ".")


def _http_get(
    url: str,
    user_agent: str,
    session: requests.Session | None = None,
) -> requests.Response:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    _sleep(0.2)
    client = session or requests
    return client.get(url, headers=headers, timeout=30)


def ticker_map(data_dir: Path | None = None) -> dict[int, str]:
    d_dir = data_dir or _default_dir()
    path = d_dir / "company_tickers.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[int, str] = {}
        for entry in raw.values():
            cik = int(entry["cik_str"])
            ticker = _normalize_ticker(str(entry["ticker"]))
            result[cik] = ticker
        return result
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        log.warning("Failed to parse ticker map at %s: %s", path, e)
        return {}


def _get_ticker_data(
    data_dir: Path,
    user_agent: str,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """One entry per ticker: {"ticker", "cik", "title"} (share classes share a CIK)."""
    path = data_dir / "company_tickers.json"
    need_download = force or not path.exists()
    if not need_download and path.exists():
        mtime = path.stat().st_mtime
        if time.time() - mtime > 30 * 86400:
            need_download = True

    if need_download:
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            res = _http_get(url, user_agent, session=session)
            _check_blocked(res, url)
            if res.status_code == 200:
                data_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
                tmp.write_text(res.text, encoding="utf-8")
                os.replace(tmp, path)
            else:
                log.warning("Failed to fetch ticker map, status %s", res.status_code)
        except SecError:
            raise
        except Exception as e:
            log.warning("Error fetching ticker map: %s", e)

    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, dict[str, Any]] = {}
        for entry in raw.values():
            ticker = _normalize_ticker(str(entry["ticker"]))
            result.setdefault(ticker, {"ticker": ticker, "cik": int(entry["cik_str"]),
                                       "title": str(entry.get("title", ""))})
        return list(result.values())
    except Exception as e:
        log.warning("Failed to parse ticker map file: %s", e)
        return []


def prev_quarter(y: int, q: int) -> tuple[int, int]:
    return (y, q - 1) if q > 1 else (y - 1, 4)


def quarter_str(y: int, q: int) -> str:
    return f"CY{y}Q{q}"


def most_recent_ended_quarter(today: dt.date | None = None) -> tuple[int, int]:
    ref = today or dt.date.today()
    y, m = ref.year, ref.month
    if m in (1, 2, 3):
        return (y - 1, 4)
    elif m in (4, 5, 6):
        return (y, 1)
    elif m in (7, 8, 9):
        return (y, 2)
    else:
        return (y, 3)


def _fetch_frame(
    concept: str,
    period: str,
    data_dir: Path,
    user_agent: str,
    is_recent: bool,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cache_path = frames_dir / f"{concept}_{period}.json"

    need_download = force or not cache_path.exists()
    if not need_download and cache_path.exists():
        if is_recent:
            age = time.time() - cache_path.stat().st_mtime
            if age > 7 * 86400:
                need_download = True

    if need_download:
        url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json"
        try:
            res = _http_get(url, user_agent, session=session)
            _check_blocked(res, url)
            if res.status_code == 200:
                data = res.json()
                tmp = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(tmp, cache_path)
                return data
            elif res.status_code == 404:
                return None
            else:
                log.warning("SEC frame %s %s returned status %s", concept, period, res.status_code)
        except SecError:
            raise
        except Exception as e:
            log.warning("Error fetching frame %s %s: %s", concept, period, e)

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to load cached frame %s: %s", cache_path, e)
            return None

    return None


def _parse_frame_vals(frame: dict[str, Any] | None) -> dict[int, float]:
    if not frame or "data" not in frame:
        return {}
    res: dict[int, float] = {}
    for item in frame["data"]:
        cik = int(item["cik"])
        val = item.get("val")
        if val is not None and cik not in res:
            res[cik] = float(val)
    return res


def update(
    force: bool = False,
    data_dir: Path | None = None,
    user_agent_path: Path | str | None = None,
    today: dt.date | None = None,
    session: requests.Session | None = None,
) -> Path | None:
    """Download what is missing, compute TTM revenue growth and net margin, write fundamentals.parquet.

    Returns None when skipped (no User-Agent, no ticker map, no complete quarter found). Raises SecError
    on a 403/429 (blocked): the run stops at once instead of sending its remaining requests.
    """
    ua = get_user_agent(user_agent_path)
    if not ua:
        log.warning("SEC EDGAR User-Agent file missing or empty. Skipping update.")
        return None

    d_dir = data_dir or _default_dir()
    d_dir.mkdir(parents=True, exist_ok=True)

    tickers_data = _get_ticker_data(d_dir, user_agent=ua, force=force, session=session)
    if not tickers_data:
        log.warning("SEC EDGAR ticker map unavailable. Skipping update.")
        return None

    # The latest complete quarter: step back over a 404 or a frame with too few companies, a few
    # quarters at most (never an endless loop of requests when SEC keeps failing).
    cand = most_recent_ended_quarter(today)
    q0: tuple[int, int] | None = None
    for _ in range(MAX_STEP_BACK):
        frame = _fetch_frame(concept="Revenues", period=quarter_str(*cand), data_dir=d_dir, user_agent=ua,
                             is_recent=True, force=force, session=session)
        if frame is not None and len(frame.get("data", [])) >= MIN_COMPANIES:
            q0 = cand
            break
        cand = prev_quarter(*cand)
    if q0 is None:
        log.warning("SEC EDGAR: no complete quarter in the last %d. Skipping update.", MAX_STEP_BACK)
        return None

    # 8 quarters: q0 and 7 preceding
    quarters: list[tuple[int, int]] = []
    curr = q0
    for _ in range(8):
        quarters.append(curr)
        curr = prev_quarter(*curr)
    q_strs = [quarter_str(*q) for q in quarters]
    last_two_strs = set(q_strs[:2])

    # Q4 = year - Q1 - Q2 - Q3 for companies that file no Q4 quarter (it is in the 10-K): every Q4 year
    # needs its annual frame and all of its first three quarters, also when they are older than the 8.
    q4_years = sorted({y for y, q in quarters if q == 4})
    extra = [quarter_str(y, q) for y in q4_years for q in (1, 2, 3) if quarter_str(y, q) not in q_strs]
    all_periods = q_strs + extra + [f"CY{y}" for y in q4_years]

    # concept -> period -> {cik: val}
    store: dict[str, dict[str, dict[int, float]]] = {c: {} for c in ALL_CONCEPTS}
    for concept in ALL_CONCEPTS:
        for period in all_periods:
            f_data = _fetch_frame(concept=concept, period=period, data_dir=d_dir, user_agent=ua,
                                  is_recent=period in last_two_strs, force=force, session=session)
            store[concept][period] = _parse_frame_vals(f_data)

    def get_revenue(cik: int, period: str) -> float | None:
        for concept in REVENUE_CONCEPTS:
            c_vals = store[concept].get(period, {})
            if cik in c_vals:
                return c_vals[cik]
        return None

    def get_net_income(cik: int, period: str) -> float | None:
        return store[NET_INCOME_CONCEPT].get(period, {}).get(cik)

    def quarter_value(get, cik: int, y: int, q: int) -> float | None:
        val = get(cik, quarter_str(y, q))
        if val is None and q == 4:
            parts = [get(cik, f"CY{y}")] + [get(cik, quarter_str(y, k)) for k in (1, 2, 3)]
            if all(p is not None for p in parts):
                val = parts[0] - parts[1] - parts[2] - parts[3]
        return val

    def ttm(vals: list[float | None]) -> float:
        return np.nan if any(v is None for v in vals) else float(sum(vals))

    def metrics(cik: int) -> dict[str, float]:
        revs = [quarter_value(get_revenue, cik, y, q) for y, q in quarters]
        nets = [quarter_value(get_net_income, cik, y, q) for y, q in quarters]
        rev_ttm, rev_ttm_prev, net_ttm = ttm(revs[:4]), ttm(revs[4:]), ttm(nets[:4])
        growth = (np.nan if np.isnan(rev_ttm) or np.isnan(rev_ttm_prev) or rev_ttm_prev <= 0
                  else (rev_ttm - rev_ttm_prev) / rev_ttm_prev * 100.0)
        margin = np.nan if np.isnan(net_ttm) or np.isnan(rev_ttm) or rev_ttm <= 0 else net_ttm / rev_ttm * 100.0
        return {"revenue_ttm": rev_ttm, "revenue_ttm_prev": rev_ttm_prev, "revenue_growth_ttm": growth,
                "net_income_ttm": net_ttm, "net_margin_ttm": margin}

    # One row per ticker: share classes (GOOG, GOOGL) share a CIK and get the same numbers.
    by_cik: dict[int, dict[str, float]] = {}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for meta in tickers_data:
        cik = meta["cik"]
        if cik not in by_cik:
            by_cik[cik] = metrics(cik)
        rows.append({"ticker": meta["ticker"], "cik": cik, "company": meta["title"], **by_cik[cik],
                     "last_period": q_strs[0], "updated_at": now_utc})

    df = pd.DataFrame(rows, columns=COLUMNS)
    out_file = d_dir / "fundamentals.parquet"
    tmp_file = out_file.with_name(f"{out_file.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, out_file)
    return out_file


def load(data_dir: Path | None = None) -> pd.DataFrame:
    d_dir = data_dir or _default_dir()
    path = d_dir / "fundamentals.parquet"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS).astype({
            "ticker": "str",
            "cik": "int64",
            "company": "str",
            "revenue_ttm": "float64",
            "revenue_ttm_prev": "float64",
            "revenue_growth_ttm": "float64",
            "net_income_ttm": "float64",
            "net_margin_ttm": "float64",
            "last_period": "str",
            "updated_at": "str",
        })
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log.warning("Failed to load fundamentals parquet at %s: %s", path, e)
        return pd.DataFrame(columns=COLUMNS)
