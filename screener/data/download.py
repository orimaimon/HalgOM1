"""
On-demand historical data download (the "Download" page): any tickers, any yfinance.download option.

Unlike prices.py (a per-ticker parquet store kept up to date for the app's own use), this is a one-off
request: fetch() returns what Yahoo returned plus a report of what is missing, and to_long() / to_wide()
/ to_excel() / to_csv() turn it into a file. Nothing is stored here; the app saves a copy of the file.

All of yfinance.download's arguments are exposed except `progress` (a console bar, always off),
`proxy` (deprecated in yfinance) and `session` (an object, not a user setting).
"""
from __future__ import annotations

import datetime as dt
import io
import re
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
GROUP_BY = ["column", "ticker"]
MAX_TICKERS = 1000           # one batched request; more is slow and gets rate-limited by Yahoo
EXCEL_MAX_ROWS = 1_048_575   # Excel's row limit minus the header row
# Yahoo serves intraday bars only for recent days: interval -> (max days back, max days per request)
INTRADAY_LIMITS = {"1m": (30, 8), "2m": (60, 60), "5m": (60, 60), "15m": (60, 60), "30m": (60, 60),
                   "90m": (60, 60), "60m": (730, 730), "1h": (730, 730)}
FIELD_ORDER = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits",
               "Capital Gains", "Repaired?"]


@dataclass
class Request:
    """The arguments of yfinance.download. `period` is used only when use_period is True."""
    tickers: list[str]
    use_period: bool = False
    period: str = "1y"
    start: dt.date | None = None
    end: dt.date | None = None          # exclusive, like yfinance
    interval: str = "1d"
    auto_adjust: bool = False           # False keeps both Close and Adj Close
    back_adjust: bool = False
    actions: bool = True
    prepost: bool = False
    repair: bool = False
    keepna: bool = False
    rounding: bool = False
    ignore_tz: bool | None = None       # None: yfinance decides (ignore for daily+, keep for intraday)
    group_by: str = "column"
    multi_level_index: bool = True
    threads: bool = True
    timeout: float = 10

    def kwargs(self) -> dict:
        """Keyword arguments for yf.download."""
        kw = dict(tickers=self.tickers, interval=self.interval, auto_adjust=self.auto_adjust,
                  back_adjust=self.back_adjust, actions=self.actions, prepost=self.prepost,
                  repair=self.repair, keepna=self.keepna, rounding=self.rounding, ignore_tz=self.ignore_tz,
                  group_by=self.group_by, multi_level_index=self.multi_level_index, threads=self.threads,
                  timeout=self.timeout, progress=False)
        if self.use_period:
            kw["period"] = self.period
        else:
            kw["start"] = self.start.isoformat() if self.start else None
            kw["end"] = self.end.isoformat() if self.end else None
        return kw

    def describe(self) -> list[tuple[str, str]]:
        """(parameter, value) rows for the file's "Parameters" sheet."""
        rows = [(k, str(v)) for k, v in self.kwargs().items() if k not in ("tickers", "progress")]
        return [("tickers", ", ".join(self.tickers)), ("downloaded at", f"{dt.datetime.now():%Y-%m-%d %H:%M}")] + rows


@dataclass
class Result:
    request: Request
    data: pd.DataFrame          # exactly what yfinance returned
    failed: dict[str, str]      # ticker -> reason (no rows at all)

    @property
    def ok(self) -> list[str]:
        return [t for t in self.request.tickers if t not in self.failed]


# ── input ──────────────────────────────────────────────────────────────────

_CLASS_SHARE = re.compile(r"^([A-Z0-9]+)\.([ABC])$")   # BRK.B, BF.B; exchange suffixes (.TA, .L) are kept


def parse_tickers(text: str) -> list[str]:
    """Split on commas, spaces, semicolons and new lines; upper-case; drop duplicates (keep order).

    Class shares are written with a dot elsewhere (TradingView, brokers) and with a dash on Yahoo:
    BRK.B -> BRK-B. Other dots are Yahoo exchange suffixes (TEVA.TA, VOD.L) and stay.
    """
    out = []
    for raw in re.split(r"[\s,;]+", text or ""):
        t = raw.strip().upper()
        if not t:
            continue
        t = _CLASS_SHARE.sub(r"\1-\2", t)
        out.append(t)
    return list(dict.fromkeys(out))


def validate(req: Request, today: dt.date | None = None) -> list[str]:
    """Problems that would make Yahoo return nothing. Empty list = OK to send."""
    today = today or dt.date.today()
    errors = []
    if not req.tickers:
        errors.append("Enter at least one ticker")
    if len(req.tickers) > MAX_TICKERS:
        errors.append(f"Too many tickers (max {MAX_TICKERS})")
    if req.interval not in INTERVALS:
        errors.append(f"Unknown interval: {req.interval}")
    if req.group_by not in GROUP_BY:
        errors.append(f"Unknown group_by: {req.group_by}")
    if req.use_period:
        if req.period not in PERIODS:
            errors.append(f"Unknown period: {req.period}")
    else:
        if req.start is None:
            errors.append("Choose a start date")
        if req.start and req.end and req.start >= req.end:
            errors.append("Start date must be before end date")
        if req.start and req.start > today:
            errors.append("Start date is in the future")
    if req.interval in INTRADAY_LIMITS:
        back, span = INTRADAY_LIMITS[req.interval]
        if req.use_period:
            days = {"1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "ytd": today.timetuple().tm_yday}
            if req.period in ("5y", "10y", "max") or days.get(req.period, 0) > min(back, span):
                errors.append(f"Interval {req.interval}: Yahoo keeps only the last {back} days "
                              f"(at most {span} days per request)")
        elif req.start:
            end = req.end or today
            if (today - req.start).days > back:
                errors.append(f"Interval {req.interval}: Yahoo keeps only the last {back} days")
            if (end - req.start).days > span:
                errors.append(f"Interval {req.interval}: at most {span} days per request")
    return errors


# ── download ───────────────────────────────────────────────────────────────

def fetch(req: Request) -> Result:
    """One batched yf.download call. Never raises for a bad ticker: it is reported in Result.failed."""
    data = yf.download(**req.kwargs())
    if data is None:
        data = pd.DataFrame()
    errors = dict(getattr(getattr(yf, "shared", None), "_ERRORS", None) or {})
    failed = {}
    for t in req.tickers:
        sub = _ticker_frame(data, t, req)
        if sub is None or sub.dropna(how="all").empty:
            failed[t] = _short_error(errors.get(t, "no data"))
    return Result(req, data, failed)


def _short_error(msg: str) -> str:
    msg = str(msg)
    m = re.search(r"Yahoo error = \"([^\"]+)\"", msg)
    if m:
        return m.group(1)
    msg = re.sub(r"^\w+Error\(['\"]?|['\"]?\)$", "", msg)
    return msg.replace(r"\'", "'").strip() or "no data"


def _ticker_level(data: pd.DataFrame, req: Request) -> int | None:
    if not isinstance(data.columns, pd.MultiIndex):
        return None
    names = list(data.columns.names)
    if "Ticker" in names:
        return names.index("Ticker")
    return 0 if req.group_by == "ticker" else 1


def _ticker_frame(data: pd.DataFrame, ticker: str, req: Request) -> pd.DataFrame | None:
    """One ticker's fields as plain columns (Open, High, ...), or None if it is not in the data."""
    if data.empty:
        return None
    level = _ticker_level(data, req)
    if level is None:                                   # flat columns: a single ticker
        return data if len(req.tickers) == 1 else None
    if ticker not in data.columns.get_level_values(level):
        return None
    return data.xs(ticker, axis=1, level=level)


# ── shapes for export ──────────────────────────────────────────────────────

def _fields(cols) -> list[str]:
    known = [f for f in FIELD_ORDER if f in cols]
    return known + [c for c in cols if c not in known]


def _naive(index: pd.Index) -> tuple[pd.Index, str]:
    """Excel cannot store time zones: drop the zone, and say in the column name when the clock is UTC
    (yfinance puts several tickers' intraday bars on UTC; a single ticker keeps its exchange's clock)."""
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        utc = str(index.tz) in ("UTC", "utc")
        return index.tz_localize(None), " (UTC)" if utc else ""
    return index, ""


def per_ticker(res: Result) -> dict[str, pd.DataFrame]:
    """ticker -> its own frame (Date index, field columns), in request order, failed tickers left out."""
    out = {}
    for t in res.ok:
        sub = _ticker_frame(res.data, t, res.request)
        if sub is None:
            continue
        sub = sub[_fields(list(sub.columns))]
        if not res.request.keepna:
            sub = sub.dropna(how="all")
        sub = sub.copy()
        sub.index, zone = _naive(sub.index)
        daily = res.request.interval in ("1d", "5d", "1wk", "1mo", "3mo")
        sub.index.name = "Date" if daily else "Datetime" + zone
        sub.columns.name = None
        out[t] = sub
    return out


def to_long(res: Result) -> pd.DataFrame:
    """Tidy table: one row per (date, ticker). Best for Excel pivots, databases and pandas."""
    frames = per_ticker(res)
    if not frames:
        return pd.DataFrame()
    parts = [f.reset_index().assign(Ticker=t) for t, f in frames.items()]
    long = pd.concat(parts, ignore_index=True)
    date_col = long.columns[0]
    cols = [date_col, "Ticker"] + [c for c in long.columns if c not in (date_col, "Ticker")]
    return long[cols].sort_values([date_col, "Ticker"], kind="stable").reset_index(drop=True)


def to_wide(res: Result) -> pd.DataFrame:
    """yfinance's layout: one row per date, a column per (field, ticker) - flattened to "Close AAPL"
    (group_by=column) or "AAPL Close" (group_by=ticker)."""
    frames = per_ticker(res)
    if not frames:
        return pd.DataFrame()
    by_ticker = res.request.group_by == "ticker"
    wide = pd.concat(frames, axis=1)                     # columns: (ticker, field)
    if not by_ticker:
        fields = _fields(list(dict.fromkeys(wide.columns.get_level_values(1))))
        wide = wide.swaplevel(axis=1)
        wide = wide.reindex(columns=[(f, t) for f in fields for t in frames if (f, t) in wide.columns])
    wide.columns = [f"{a} {b}" for a, b in wide.columns]
    wide.index.name = next(iter(frames.values())).index.name
    return wide.sort_index()


# ── files ──────────────────────────────────────────────────────────────────

def to_csv(res: Result, layout: str = "long") -> bytes:
    df = to_long(res) if layout == "long" else to_wide(res).reset_index()
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM so Excel opens it correctly


def _sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"
    out, n = base, 2
    while out.lower() in used:
        suffix = f" ({n})"
        out, n = base[:31 - len(suffix)] + suffix, n + 1
    used.add(out.lower())
    return out


def to_excel(res: Result, layout: str = "sheets") -> bytes:
    """layout: "sheets" (a sheet per ticker), "long" (one tidy sheet) or "wide" (yfinance's layout).
    Always adds a "Parameters" sheet and, if any, a "Failed" sheet."""
    if layout == "long":
        sheets = {"Data": to_long(res)}
    elif layout == "wide":
        sheets = {"Data": to_wide(res).reset_index()}
    else:
        sheets = {t: f.reset_index() for t, f in per_ticker(res).items()}
    too_big = [n for n, df in sheets.items() if len(df) > EXCEL_MAX_ROWS]
    if too_big:
        raise ValueError(f"Too many rows for one Excel sheet ({', '.join(too_big)}). "
                         "Choose a sheet per ticker, or CSV.")
    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        params = pd.DataFrame(res.request.describe(), columns=["Parameter", "Value"])
        params.to_excel(xl, sheet_name=_sheet_name("Parameters", used), index=False)
        if res.failed:
            pd.DataFrame(list(res.failed.items()), columns=["Ticker", "Reason"]).to_excel(
                xl, sheet_name=_sheet_name("Failed", used), index=False)
        for name, df in sheets.items():
            sheet = _sheet_name(name, used)
            df.to_excel(xl, sheet_name=sheet, index=False, freeze_panes=(1, 1))
            ws = xl.sheets[sheet]
            date_fmt = "yyyy-mm-dd" if df.columns[0] == "Date" else "yyyy-mm-dd hh:mm"
            ws.column_dimensions["A"].width = 12 if date_fmt == "yyyy-mm-dd" else 17
            for cell in ws["A"][1:]:
                cell.number_format = date_fmt
        for name in xl.sheets:
            if name.lower() in ("parameters", "failed"):
                ws = xl.sheets[name]
                ws.column_dimensions["A"].width = 18
                ws.column_dimensions["B"].width = 80
    return buf.getvalue()


def summary(res: Result) -> pd.DataFrame:
    """Per ticker: rows, first and last date, last close - shown under the download form."""
    rows = []
    for t, f in per_ticker(res).items():
        close = f["Close"].dropna() if "Close" in f else pd.Series(dtype=float)
        rows.append({"Ticker": t, "Rows": len(f),
                     "From": f.index.min() if len(f) else None, "To": f.index.max() if len(f) else None,
                     "Last close": float(close.iloc[-1]) if len(close) else None})
    return pd.DataFrame(rows)


__all__ = ["Request", "Result", "INTERVALS", "PERIODS", "GROUP_BY", "MAX_TICKERS", "parse_tickers", "validate",
           "fetch", "per_ticker", "to_long", "to_wide", "to_csv", "to_excel", "summary"]
