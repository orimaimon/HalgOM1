"""
Nasdaq stock screener fallback provider.

Fetches the whole US stock universe in a single HTTP request to Nasdaq's stock screener API.
Module only; not connected to the UI or store yet.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

from screener.fields import provider_columns

log = logging.getLogger(__name__)

NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&download=true"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

MIN_ROWS = 3000

# "Preferred" only as a share class: "Preferred Bank" (PFBC) is a common stock.
EXCLUDED_WORDS_RE = re.compile(
    r"\b(?:warrant|unit|right)s?\b"
    r"|\bpreferred\s+(?:stock|shares?|securities)\b"
    r"|\bdepositary\s+shares?\b.*\bpreferred\b",
    re.IGNORECASE,
)


class NasdaqError(RuntimeError):
    """`status_code` is set for an HTTP error, so the request guard sees a 403/429 block."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def is_excluded_security(symbol: str, name: str) -> bool:
    """Return True if security is a warrant, unit, right, or preferred share, or symbol contains ^."""
    if "^" in symbol:
        return True
    if name and EXCLUDED_WORDS_RE.search(name):
        return True
    return False


def _parse_number(val: Any) -> float:
    if val is None:
        return np.nan
    val_str = str(val).strip()
    if val_str.upper() in ("", "NA", "N/A", "NONE", "NULL", "NAN"):
        return np.nan
    val_cleaned = val_str.replace("$", "").replace("%", "").replace(",", "")
    try:
        return float(val_cleaned)
    except (ValueError, TypeError):
        return np.nan


def _clean_company_name(name: str) -> str:
    if not name:
        return ""
    name_str = str(name).strip()
    suffix = " Common Stock"
    if name_str.endswith(suffix):
        name_str = name_str[:-len(suffix)].strip()
    return name_str


def _clean_str(val: Any) -> str | float:
    if val is None:
        return np.nan
    s = str(val).strip()
    if s.upper() in ("", "NA", "N/A", "NONE", "NULL"):
        return np.nan
    return s


def fetch_universe(timeout: float = 30) -> pd.DataFrame:
    """Fetch US stock universe from Nasdaq screener API as a DataFrame."""
    try:
        resp = requests.get(NASDAQ_URL, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        raise NasdaqError(f"network error: {e}") from e

    if resp.status_code != 200:
        raise NasdaqError(f"HTTP {resp.status_code}: {resp.text[:200]}", status_code=resp.status_code)

    try:
        payload = resp.json()
    except Exception as e:
        raise NasdaqError(f"invalid JSON response: {e}") from e

    if not isinstance(payload, dict) or "data" not in payload or not isinstance(payload.get("data"), dict):
        raise NasdaqError("missing data in response")

    rows_data = payload["data"].get("rows")
    if not isinstance(rows_data, list):
        raise NasdaqError("missing data.rows in response")

    parsed_rows = []
    for r in rows_data:
        if not isinstance(r, dict):
            continue
        raw_symbol = str(r.get("symbol", "") or "")
        raw_name = str(r.get("name", "") or "")

        if is_excluded_security(raw_symbol, raw_name):
            continue

        symbol = raw_symbol.strip().upper().replace("/", ".")
        company = _clean_company_name(raw_name)
        price = _parse_number(r.get("lastsale"))
        change_abs = _parse_number(r.get("netchange"))
        change = _parse_number(r.get("pctchange"))
        volume = _parse_number(r.get("volume"))
        market_cap = _parse_number(r.get("marketCap"))
        ipo_year_num = _parse_number(r.get("ipoyear"))
        country = _clean_str(r.get("country"))
        sector = _clean_str(r.get("sector"))
        industry = _clean_str(r.get("industry"))

        sec_type = "dr" if "american depositary" in raw_name.lower() else "stock"

        parsed_rows.append({
            "ticker": symbol,
            "company": company,
            "price": price,
            "change_abs": change_abs,
            "change": change,
            "volume": volume,
            "market_cap": market_cap,
            "country": country,
            "sector": sector,
            "industry": industry,
            "sec_type": sec_type,
            "ipo_year": ipo_year_num,
        })

    df = pd.DataFrame(parsed_rows)

    if len(df) < MIN_ROWS:
        raise NasdaqError(f"universe size ({len(df)}) below minimum floor ({MIN_ROWS})")

    for col in provider_columns():
        if col not in df.columns:
            df[col] = np.nan

    df.attrs["source"] = {
        "provider": "nasdaq",
        "update_mode": "delayed",
        "delay_min": None,
        "raw": None,
    }
    return df
