"""
Broker export CSV import parsers. Pure functions that turn broker export CSV text into
a normalized trades DataFrame (Interactive Brokers and generic template).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from typing import Any

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "side", "shares", "price", "fees"]
BUY, SELL = "buy", "sell"

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,2})?$")

GENERIC_HEADERS = {
    "date": ("date", "trade date", "תאריך", "תאריך קנייה", "תאריך קניה", "date/time"),
    "ticker": ("ticker", "symbol", "סימול", "סימול נייר", "טיקר", "נייר", "מניה"),
    "side": ("side", "action", "type", "פעולה", "סוג", "סוג פעולה"),
    "shares": ("shares", "quantity", "qty", "כמות"),
    "price": ("price", "t. price", "t.price", "מחיר", "שער", "שער קנייה", "שער ביצוע"),
    "fees": ("fees", "fee", "commission", "comm/fee", "comm", "עמלה", "עמלות"),
}

IBKR_HEADERS = {
    "ticker": ("symbol", "ticker", "sym"),
    "date": ("date/time", "date", "datetime"),
    "shares": ("quantity", "qty"),
    "price": ("t. price", "t.price", "tradeprice", "price"),
    "fees": ("comm/fee", "comm / fee", "ibcommission", "commission", "fee", "fees"),
}


def parse_trades(text: str, broker: str = "auto") -> pd.DataFrame:
    """Parse broker export CSV text into a normalized trades DataFrame."""
    df, _ = parse_trades_report(text, broker)
    return df


def parse_trades_report(text: str, broker: str = "auto") -> tuple[pd.DataFrame, list[str]]:
    """Parse broker export CSV text into a normalized trades DataFrame and return skipped line reports."""
    empty_df = pd.DataFrame(columns=COLUMNS)
    if not text or not text.strip():
        return empty_df, []

    cleaned_text = text.lstrip("\ufeff")
    if broker == "auto":
        broker = _detect_broker(cleaned_text)

    if broker == "ibkr":
        return _parse_ibkr(cleaned_text)
    return _parse_generic(cleaned_text)


def _detect_broker(text: str) -> str:
    """Detect broker format ('ibkr' vs 'generic') from CSV content."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:10]:
        line_lower = line.lower()
        if '"trades"' in line_lower or "'trades'" in line_lower or line_lower.startswith("trades,"):
            return "ibkr"
        if "date/time" in line_lower or "t. price" in line_lower or "comm/fee" in line_lower or "ibcommission" in line_lower:
            return "ibkr"
    return "generic"


def _normalize_ticker(v: Any) -> str:
    s = str(v).strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    return re.sub(r"[-/]", ".", s)


def _parse_date(v: Any) -> dt.date | None:
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return v.date()
    if isinstance(v, dt.date):
        return v
    s_raw = str(v).strip()
    if not s_raw:
        return None

    # Split date part before time / comma / semicolon / T / space
    s = re.split(r"[,;\sT]+", s_raw)[0].strip()
    if not s:
        return None

    if re.match(r"^\d{8}$", s):
        try:
            return dt.datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            return None

    fmts = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_number(v: Any) -> float | None:
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return None if pd.isna(v) else float(v)
    s = str(v).strip().replace("$", "").replace(" ", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_generic(text: str) -> tuple[pd.DataFrame, list[str]]:
    reports: list[str] = []
    rows_out: list[dict[str, Any]] = []

    reader = csv.reader(io.StringIO(text))
    header_indices: dict[str, int] = {}
    found_header = False

    for row in reader:
        line_num = reader.line_num
        if not row or not any(c.strip() for c in row):
            continue

        row_clean = [c.strip() for c in row]

        if not found_header:
            col_map = {idx: c.lower() for idx, c in enumerate(row_clean)}
            for col_key, aliases in GENERIC_HEADERS.items():
                for idx, c_name in col_map.items():
                    if c_name in aliases and col_key not in header_indices:
                        header_indices[col_key] = idx

            if "ticker" in header_indices or "date" in header_indices:
                found_header = True
                continue
            else:
                reports.append(f"שורה {line_num}: חסרות עמודות חובה בבלוק הכותרת")
                continue

        # Data row parsing
        row_errs = _validate_and_extract_row(row_clean, header_indices, line_num, is_ibkr=False)
        if isinstance(row_errs, str):
            reports.append(row_errs)
        elif row_errs is not None:
            rows_out.append(row_errs)

    df = pd.DataFrame(rows_out, columns=COLUMNS) if rows_out else pd.DataFrame(columns=COLUMNS)
    return df, reports


def _parse_ibkr(text: str) -> tuple[pd.DataFrame, list[str]]:
    reports: list[str] = []
    rows_out: list[dict[str, Any]] = []

    reader = csv.reader(io.StringIO(text))
    header_indices: dict[str, int] = {}
    is_multisection = False

    for row in reader:
        line_num = reader.line_num
        if not row or not any(c.strip() for c in row):
            continue

        row_clean = [c.strip() for c in row]
        first_col_lower = row_clean[0].lower()

        # Check for multi-section format (starts with "Trades")
        if first_col_lower == "trades":
            is_multisection = True
            if len(row_clean) > 1 and row_clean[1].lower() == "header":
                header_indices = {}
                col_map = {idx: c.lower() for idx, c in enumerate(row_clean)}
                for col_key, aliases in IBKR_HEADERS.items():
                    for idx, c_name in col_map.items():
                        if c_name in aliases and col_key not in header_indices:
                            header_indices[col_key] = idx
                continue
            elif len(row_clean) > 1 and row_clean[1].lower() == "data":
                if not header_indices:
                    reports.append(f"שורה {line_num}: נתוני מסחר ללא כותרת עמודות תואמת")
                    continue
                res = _validate_and_extract_row(row_clean, header_indices, line_num, is_ibkr=True)
                if isinstance(res, str):
                    reports.append(res)
                elif res is not None:
                    rows_out.append(res)
                continue
            else:
                continue
        elif is_multisection:
            # Skip non-Trades section rows in multi-section files
            continue

        # Single-table IBKR format parsing
        if not header_indices:
            col_map = {idx: c.lower() for idx, c in enumerate(row_clean)}
            for col_key, aliases in IBKR_HEADERS.items():
                for idx, c_name in col_map.items():
                    if c_name in aliases and col_key not in header_indices:
                        header_indices[col_key] = idx

            if "ticker" in header_indices and "date" in header_indices:
                continue
            else:
                reports.append(f"שורה {line_num}: חסרות עמודות חובה בבלוק הכותרת")
                continue

        res = _validate_and_extract_row(row_clean, header_indices, line_num, is_ibkr=True)
        if isinstance(res, str):
            reports.append(res)
        elif res is not None:
            rows_out.append(res)

    df = pd.DataFrame(rows_out, columns=COLUMNS) if rows_out else pd.DataFrame(columns=COLUMNS)
    return df, reports


def _validate_and_extract_row(
    row: list[str], header_indices: dict[str, int], line_num: int, is_ibkr: bool
) -> dict[str, Any] | str:

    def _get_val(key: str) -> str:
        idx = header_indices.get(key)
        if idx is not None and idx < len(row):
            return row[idx]
        return ""

    raw_ticker = _get_val("ticker")
    ticker = _normalize_ticker(raw_ticker)
    if not ticker or not _TICKER_RE.match(ticker):
        return f"שורה {line_num}: סימול נייר לא תקין ({raw_ticker!r})"

    raw_date = _get_val("date")
    parsed_date = _parse_date(raw_date)
    if parsed_date is None:
        return f"שורה {line_num}: תאריך לא תקין ({raw_date!r})"

    raw_shares = _get_val("shares")
    shares_num = _parse_number(raw_shares)
    if shares_num is None or shares_num == 0:
        return f"שורה {line_num}: כמות מניות לא תקינה ({raw_shares!r})"

    raw_price = _get_val("price")
    price_num = _parse_number(raw_price)
    if price_num is None or price_num <= 0:
        return f"שורה {line_num}: מחיר לא תקין ({raw_price!r})"

    raw_fees = _get_val("fees")
    fees_num = _parse_number(raw_fees) if raw_fees != "" else 0.0
    if fees_num is None:
        return f"שורה {line_num}: עמלה לא תקינה ({raw_fees!r})"
    fees_num = abs(fees_num)

    if is_ibkr:
        side = BUY if shares_num > 0 else SELL
        shares = abs(shares_num)
    else:
        raw_side = _get_val("side").strip().lower()
        if raw_side in ("buy", "b", "קנייה", "קניה"):
            side = BUY
            shares = abs(shares_num)
        elif raw_side in ("sell", "s", "מכירה"):
            side = SELL
            shares = abs(shares_num)
        elif not raw_side:
            side = BUY if shares_num > 0 else SELL
            shares = abs(shares_num)
        else:
            return f"שורה {line_num}: סוג פעולה לא תקין ({raw_side!r})"

    return {
        "date": parsed_date,
        "ticker": ticker,
        "side": side,
        "shares": float(shares),
        "price": float(price_num),
        "fees": float(fees_num),
    }
