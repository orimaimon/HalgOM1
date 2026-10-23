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

DATA_DIR = Path(__file__).resolve().parent.parent / "_data" / "sec"

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
    d_dir = data_dir or DATA_DIR
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
) -> dict[int, dict[str, Any]]:
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
            if res.status_code == 200:
                data_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
                tmp.write_text(res.text, encoding="utf-8")
                os.replace(tmp, path)
            else:
                log.warning("Failed to fetch ticker map, status %s", res.status_code)
        except Exception as e:
            log.warning("Error fetching ticker map: %s", e)

    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result: dict[int, dict[str, Any]] = {}
        for entry in raw.values():
            cik = int(entry["cik_str"])
            ticker = _normalize_ticker(str(entry["ticker"]))
            title = str(entry.get("title", ""))
            result[cik] = {"ticker": ticker, "title": title}
        return result
    except Exception as e:
        log.warning("Failed to parse ticker map file: %s", e)
        return {}


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
    ua = get_user_agent(user_agent_path)
    if not ua:
        log.warning("SEC EDGAR User-Agent file missing or empty. Skipping update.")
        return None

    d_dir = data_dir or DATA_DIR
    d_dir.mkdir(parents=True, exist_ok=True)

    tickers_data = _get_ticker_data(d_dir, user_agent=ua, force=force, session=session)
    if not tickers_data:
        log.warning("SEC EDGAR ticker map unavailable. Skipping update.")
        return None

    cand = most_recent_ended_quarter(today)
    q0: tuple[int, int] | None = None

    while True:
        p_str = quarter_str(*cand)
        # Check Revenues concept to determine complete quarter
        frame = _fetch_frame(
            concept="Revenues",
            period=p_str,
            data_dir=d_dir,
            user_agent=ua,
            is_recent=True,
            force=force,
            session=session,
        )
        if frame is not None:
            pts = len(frame.get("data", []))
            if pts >= 1000:
                q0 = cand
                break
        cand = prev_quarter(*cand)

    # 8 quarters: q0 and 7 preceding
    quarters: list[tuple[int, int]] = []
    curr = q0
    for _ in range(8):
        quarters.append(curr)
        curr = prev_quarter(*curr)

    q_strs = [quarter_str(*q) for q in quarters]
    last_two_strs = set(q_strs[:2])

    # Identify Q4 years and annual periods needed for Q4 derivation
    q4_years: set[int] = set()
    for y, q in quarters:
        if q == 4:
            q4_years.add(y)

    annual_periods = [f"CY{y}" for y in q4_years]
    all_periods = q_strs + annual_periods

    # Fetch and store all frames: concept -> period -> {cik: val}
    store: dict[str, dict[str, dict[int, float]]] = {c: {} for c in ALL_CONCEPTS}

    for concept in ALL_CONCEPTS:
        for period in all_periods:
            is_recent = period in last_two_strs
            f_data = _fetch_frame(
                concept=concept,
                period=period,
                data_dir=d_dir,
                user_agent=ua,
                is_recent=is_recent,
                force=force,
                session=session,
            )
            store[concept][period] = _parse_frame_vals(f_data)

    # Revenue resolution per company and period
    def get_revenue(cik: int, period: str) -> float | None:
        for concept in REVENUE_CONCEPTS:
            c_vals = store[concept].get(period, {})
            if cik in c_vals:
                return c_vals[cik]
        return None

    def get_net_income(cik: int, period: str) -> float | None:
        return store[NET_INCOME_CONCEPT].get(period, {}).get(cik)

    # Calculate metrics for each company in ticker map
    rows: list[dict[str, Any]] = []
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    for cik, meta in tickers_data.items():
        ticker = meta["ticker"]
        company = meta["title"]

        revs: list[float | None] = []
        nets: list[float | None] = []

        for y, q in quarters:
            p_str = quarter_str(y, q)
            r_val = get_revenue(cik, p_str)
            n_val = get_net_income(cik, p_str)

            # Q4 derivation if missing
            if q == 4:
                year_str = f"CY{y}"
                q1_str = f"CY{y}Q1"
                q2_str = f"CY{y}Q2"
                q3_str = f"CY{y}Q3"

                if r_val is None:
                    ry = get_revenue(cik, year_str)
                    rq1 = get_revenue(cik, q1_str)
                    rq2 = get_revenue(cik, q2_str)
                    rq3 = get_revenue(cik, q3_str)
                    if ry is not None and rq1 is not None and rq2 is not None and rq3 is not None:
                        r_val = ry - rq1 - rq2 - rq3

                if n_val is None:
                    ny = get_net_income(cik, year_str)
                    nq1 = get_net_income(cik, q1_str)
                    nq2 = get_net_income(cik, q2_str)
                    nq3 = get_net_income(cik, q3_str)
                    if ny is not None and nq1 is not None and nq2 is not None and nq3 is not None:
                        n_val = ny - nq1 - nq2 - nq3

            revs.append(r_val)
            nets.append(n_val)

        # Compute TTMs
        # last 4 quarters: index 0..3
        rev_last4 = revs[:4]
        rev_prev4 = revs[4:]
        net_last4 = nets[:4]

        if any(v is None or np.isnan(v) for v in rev_last4):
            rev_ttm = np.nan
        else:
            rev_ttm = float(sum(v for v in rev_last4 if v is not None))

        if any(v is None or np.isnan(v) for v in rev_prev4):
            rev_ttm_prev = np.nan
        else:
            rev_ttm_prev = float(sum(v for v in rev_prev4 if v is not None))

        if np.isnan(rev_ttm) or np.isnan(rev_ttm_prev) or rev_ttm_prev <= 0:
            rev_growth_ttm = np.nan
        else:
            rev_growth_ttm = float((rev_ttm - rev_ttm_prev) / rev_ttm_prev * 100.0)

        if any(v is None or np.isnan(v) for v in net_last4):
            net_inc_ttm = np.nan
        else:
            net_inc_ttm = float(sum(v for v in net_last4 if v is not None))

        if np.isnan(net_inc_ttm) or np.isnan(rev_ttm) or rev_ttm <= 0:
            net_margin_ttm = np.nan
        else:
            net_margin_ttm = float(net_inc_ttm / rev_ttm * 100.0)

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "company": company,
            "revenue_ttm": rev_ttm,
            "revenue_ttm_prev": rev_ttm_prev,
            "revenue_growth_ttm": rev_growth_ttm,
            "net_income_ttm": net_inc_ttm,
            "net_margin_ttm": net_margin_ttm,
            "last_period": q_strs[0],
            "updated_at": now_utc,
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    out_file = d_dir / "fundamentals.parquet"
    tmp_file = out_file.with_name(f"{out_file.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, out_file)

    return out_file


def load(data_dir: Path | None = None) -> pd.DataFrame:
    d_dir = data_dir or DATA_DIR
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
