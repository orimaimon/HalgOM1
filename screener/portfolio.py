"""
Watchlists and portfolios. One file format for both (CSV, or the first sheet of an Excel file),
one row per trade:

  ticker     required        AAPL, BRK.B
  date       portfolio       trade date: 2025-03-14, 14/03/2025 (day first) or an Excel date
  price      portfolio       price per share, USD
  quantity   portfolio       number of shares
  side       optional        buy (default) | sell
  fees       optional        USD
  note       optional        free text

A file with only tickers is a watchlist; with date + price + quantity it is a portfolio. Headers may be
Hebrew (see HEADERS) and any case. Lists live per machine in screener/_data/portfolios.json and export
back to the same format, so an export is also the backup and the way to move a list to another PC.

Positions use the average-cost method: fees are part of the cost of a buy and reduce the proceeds
of a sell. Prices come from the screener's universe (no extra requests); a ticker that is not in the
universe keeps a NaN price. Not handled yet (needs the price-history layer): splits after the trade
date, dividends, benchmarks.

Import problems are returned as Problem(row, code, value), never raised; the UI shows them
(Hebrew text in i18n) and imports only when there are none.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from screener.data.store import DATA_DIR
from screener.userfiles import _write_user_file, load_user_file, save_user_file

log = logging.getLogger(__name__)

PORTFOLIOS_PATH = DATA_DIR / "portfolios.json"

COLUMNS = ["ticker", "date", "price", "quantity", "side", "fees", "note"]
TRADE_COLUMNS = ("date", "price", "quantity")      # all three make a row a trade
WATCHLIST, PORTFOLIO = "watchlist", "portfolio"
BUY, SELL = "buy", "sell"

# header (lower case, stripped) -> canonical column
HEADERS = {
    **{c: c for c in COLUMNS},
    "symbol": "ticker", "סימול": "ticker", "סימול נייר": "ticker", "טיקר": "ticker", "נייר": "ticker",
    "מניה": "ticker",
    "תאריך": "date", "תאריך קנייה": "date", "תאריך קניה": "date", "trade date": "date", "date/time": "date",
    "שער": "price", "מחיר": "price", "שער קנייה": "price", "שער קניה": "price", "מחיר קנייה": "price",
    "שער ביצוע": "price", "t. price": "price", "t.price": "price",
    "qty": "quantity", "shares": "quantity", "כמות": "quantity",
    "action": "side", "type": "side", "פעולה": "side", "סוג": "side", "סוג פעולה": "side",
    "fee": "fees", "commission": "fees", "comm/fee": "fees", "comm / fee": "fees", "עמלה": "fees", "עמלות": "fees",
    "notes": "note", "הערה": "note", "הערות": "note",
}
SIDES = {
    "buy": BUY, "b": BUY, "קנייה": BUY, "קניה": BUY, "קניה חול": BUY, 'קנייה חו"ל': BUY, 'קניה חו"ל': BUY,
    "sell": SELL, "s": SELL, "מכירה": SELL, "מכירה חול": SELL, 'מכירה חו"ל': SELL,
}
_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,2})?$")


# ── reading a file ───────────────────────────────────────────────────────────

def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """The raw table of an uploaded CSV / Excel file, every cell as text or as Excel gave it."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=object)
    df = None
    for encoding in ("utf-8-sig", "cp1255"):          # cp1255: Hebrew CSV saved by older Excel
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding, skipinitialspace=True,
                             header=None, skip_blank_lines=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(io.BytesIO(data), dtype=str, encoding="latin-1", header=None, skip_blank_lines=False)

    if df.empty:
        return pd.DataFrame(columns=["ticker"])

    first_row = [str(c).strip().lower() for c in df.iloc[0].dropna().tolist()]
    cells = [p for val in df.values.flatten() if pd.notna(val) for p in re.split(r"[,;\s]+", str(val)) if p]
    if not any(c in HEADERS for c in first_row) and cells and all(_TICKER.match(normalize_ticker(c)) for c in cells):
        return pd.DataFrame({"ticker": cells})       # a bare list, e.g. our TradingView export "NASDAQ:AAPL,..."
    df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.iloc[0].tolist()]
    return df.iloc[1:].reset_index(drop=True)          # unknown headers: parse() reports no_ticker_column


@dataclass(frozen=True)
class Problem:
    row: int          # 1-based row of the file, counting the header as row 1 (as Excel shows it)
    code: str         # no_ticker_column | bad_ticker | unknown_ticker | bad_date | future_date |
                      # bad_number | missing_value | bad_side | oversold
    value: str = ""


@dataclass
class Imported:
    kind: str                  # WATCHLIST | PORTFOLIO
    rows: pd.DataFrame         # canonical COLUMNS, valid rows only
    problems: list[Problem]
    date_format: str = "יום/חודש"    # "יום/חודש" or "חודש/יום"

    @property
    def ok(self) -> bool:
        return not self.problems and not self.rows.empty


def normalize_ticker(v) -> str:
    """'brk-b ' / 'BRK/B' / 'NASDAQ:AAPL' -> 'AAPL', 'BRK.B' (TradingView's class-share form)."""
    s = str(v).strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    return re.sub(r"[-/]", ".", s)


def detect_file_date_format(series: pd.Series) -> tuple[bool, str]:
    """Scans date values in a column to resolve ambiguity (day-first vs month-first).
    Returns (day_first, display_text) where display_text is 'יום/חודש' or 'חודש/יום'."""
    p1_gt_12 = False
    p2_gt_12 = False
    for v in series:
        if _blank(v) or isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
            continue
        s = re.split(r"[\s,T]+", str(v).strip())[0].strip()
        m = re.match(r"^(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})$", s)
        if m:
            p1, p2 = int(m.group(1)), int(m.group(2))
            if p1 > 12:
                p1_gt_12 = True
            if p2 > 12:
                p2_gt_12 = True
    if p1_gt_12:
        return True, "יום/חודש"
    if p2_gt_12:
        return False, "חודש/יום"
    return True, "יום/חודש"


def parse_date(v, today: dt.date, day_first: bool = True) -> dt.date | None:
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return v.date()
    if isinstance(v, dt.date):
        return v
    s = re.split(r"[\s,T]+", str(v).strip())[0].strip()
    fmts = (
        ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y")
        if day_first else
        ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y")
    )
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_number(v) -> float | None:
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return None if pd.isna(v) else float(v)
    s = str(v).strip().replace("$", "").replace("‏", "").replace("‎", "").strip()
    if not s:
        return None
    if re.match(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$", s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None
    if re.match(r"^[+-]?\d+,\d{1,2}$", s):
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return None
    if re.match(r"^[+-]?\d+(\.\d+)?$", s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "NaT", "None")


def parse(raw: pd.DataFrame, known: set[str] | None = None, today: dt.date | None = None) -> Imported:
    """Validate a raw table. `known` = tickers of the universe (unknown ones are a problem)."""
    today = today or dt.date.today()
    cols = {c: HEADERS.get(str(c).strip().lower()) for c in raw.columns}
    df = raw.rename(columns={c: k for c, k in cols.items() if k}).loc[:, lambda d: ~d.columns.duplicated()]
    df = df[[c for c in COLUMNS if c in df.columns]]
    if "ticker" not in df.columns:
        return Imported(WATCHLIST, pd.DataFrame(columns=COLUMNS), [Problem(1, "no_ticker_column")])
    df = df[~df.apply(lambda r: all(_blank(v) for v in r), axis=1)]          # skip empty rows
    kind = PORTFOLIO if ("quantity" in df.columns and not df["quantity"].map(_blank).all()) else WATCHLIST

    day_first, date_format_str = detect_file_date_format(df["date"]) if "date" in df.columns else (True, "יום/חודש")

    problems: list[Problem] = []
    out = []
    for i, r in df.iterrows():
        n = int(i) + 2                                  # header is row 1
        row_problems = len(problems)
        ticker = normalize_ticker(r["ticker"]) if not _blank(r["ticker"]) else ""
        if not ticker:
            problems.append(Problem(n, "missing_value", "ticker"))
        elif not _TICKER.match(ticker):
            problems.append(Problem(n, "bad_ticker", ticker))
        elif known is not None and ticker not in known:
            problems.append(Problem(n, "unknown_ticker", ticker))
        rec = {"ticker": ticker, "note": "" if _blank(r.get("note")) else str(r.get("note")).strip()}
        if "date" in df.columns and not _blank(r["date"]):
            d = parse_date(r["date"], today, day_first=day_first)
            if d is None:
                problems.append(Problem(n, "bad_date", str(r["date"])))
            elif d > today:
                problems.append(Problem(n, "future_date", d.isoformat()))
            rec["date"] = d

        if "price" in df.columns and not _blank(r["price"]):
            x = parse_number(r["price"])
            if x is None or x <= 0:
                problems.append(Problem(n, "bad_number", f"price: {r['price']}"))
            rec["price"] = x

        if kind == PORTFOLIO:
            for c in TRADE_COLUMNS:
                if c not in df.columns or _blank(r[c]):
                    problems.append(Problem(n, "missing_value", c))

            raw_qty = parse_number(r["quantity"]) if "quantity" in df.columns and not _blank(r["quantity"]) else None
            side_str = str(r["side"]).strip().lower() if "side" in df.columns and not _blank(r["side"]) else ""

            if raw_qty is not None:
                if raw_qty == 0:
                    problems.append(Problem(n, "bad_number", f"quantity: {r['quantity']}"))
                else:
                    qty = abs(raw_qty)
                    rec["quantity"] = qty
                    if not side_str:
                        rec["side"] = SELL if raw_qty < 0 else BUY
                    elif side_str in SIDES:
                        rec["side"] = SIDES[side_str]
                    else:
                        problems.append(Problem(n, "bad_side", side_str))
            else:
                if side_str:
                    if side_str in SIDES:
                        rec["side"] = SIDES[side_str]
                    else:
                        problems.append(Problem(n, "bad_side", side_str))

            if "fees" in df.columns and not _blank(r["fees"]):
                f = parse_number(r["fees"])
                if f is None:
                    problems.append(Problem(n, "bad_number", f"fees: {r['fees']}"))
                else:
                    rec["fees"] = abs(f)
            else:
                rec.setdefault("fees", 0.0)
        if len(problems) == row_problems:
            out.append(rec)
    rows = pd.DataFrame(out, columns=COLUMNS)
    if kind == WATCHLIST:
        rows = rows.drop_duplicates("ticker")
    elif not rows.empty:
        problems += [Problem(0, "oversold", t) for t in oversold(rows)]
    return Imported(kind, rows.reset_index(drop=True), problems, date_format=date_format_str)


# ── positions ────────────────────────────────────────────────────────────────

def sort_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Sort trades chronologically. If the file was exported newest first, reverse it first.
    Within the same date, buys come before sells."""
    if trades.empty or "date" not in trades.columns:
        return trades
    df = trades.copy()
    dates = df["date"].tolist()
    if len(dates) > 1 and all(dates[i] >= dates[i + 1] for i in range(len(dates) - 1)) and dates[0] > dates[-1]:
        df = df.iloc[::-1].reset_index(drop=True)
    side_rank = df["side"].map(lambda s: 0 if s == BUY else 1 if s == SELL else 2)
    df["_side_rank"] = side_rank
    return df.sort_values(["date", "_side_rank"], kind="stable").drop(columns=["_side_rank"]).reset_index(drop=True)


def oversold(trades: pd.DataFrame) -> list[str]:
    """Tickers where, in date order, more shares were sold than held."""
    bad = []
    sorted_df = sort_trades(trades)
    for t, g in sorted_df.groupby("ticker", sort=False):
        held = (g["quantity"] * np.where(g["side"] == SELL, -1, 1)).cumsum()
        if (held < -1e-9).any():
            bad.append(t)
    return bad


def positions(trades: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: quantity, avg_cost (incl. buy fees), cost (of the shares held), realized P&L,
    first_buy and avg_date (cost-weighted buy date, for the annual return). Closed positions stay,
    with quantity 0, so their realized P&L is not lost."""
    rows = []
    sorted_df = sort_trades(trades).assign(fees=lambda d: d["fees"].fillna(0.0))
    for t, g in sorted_df.groupby("ticker", sort=False):
        qty = cost = realized = 0.0
        weighted_days = 0.0
        for _, tr in g.iterrows():
            if tr["side"] == BUY:
                spent = tr["price"] * tr["quantity"] + tr["fees"]
                weighted_days += spent * tr["date"].toordinal()
                qty, cost = qty + tr["quantity"], cost + spent
            else:
                avg = cost / qty if qty else 0.0
                part = min(tr["quantity"], qty)
                realized += tr["price"] * part - tr["fees"] - avg * part
                weighted_days *= (qty - part) / qty if qty else 0.0
                qty, cost = qty - part, cost - avg * part
            if abs(qty) < 1e-9:
                qty = cost = 0.0
                weighted_days = 0.0
        rows.append({
            "ticker": t, "quantity": qty, "avg_cost": cost / qty if qty else np.nan, "cost": cost,
            "realized": realized, "first_buy": g.loc[g["side"] == BUY, "date"].min(),
            "avg_date": dt.date.fromordinal(round(weighted_days / cost)) if cost > 0 else None,
        })
    return pd.DataFrame(rows, columns=["ticker", "quantity", "avg_cost", "cost", "realized", "first_buy",
                                       "avg_date"])


def valuation(pos: pd.DataFrame, prices: pd.Series, today: dt.date | None = None) -> pd.DataFrame:
    """Positions + price, value, P&L ($ and %), weight (% of value) and annual return.
    `prices` maps ticker -> current price (the universe's "price"). Percent columns hold percent
    numbers (5.0 == 5%), like the rest of the screener."""
    today = today or dt.date.today()
    out = pos.copy()
    out["price"] = out["ticker"].map(prices).astype(float)
    out["value"] = out["price"] * out["quantity"]
    out["pnl"] = out["value"] - out["cost"]
    out["pnl_pct"] = np.where(out["cost"] > 0, out["pnl"] / out["cost"] * 100, np.nan)
    total = out["value"].sum(min_count=1)
    out["weight"] = out["value"] / total * 100 if total and total > 0 else np.nan
    days = out["avg_date"].map(lambda d: (today - d).days if d else np.nan).astype(float)
    growth = (out["value"] / out["cost"]).where(out["cost"] > 0)
    # Annualizing a few weeks of return gives absurd numbers: only after 90 days.
    out["annual_pct"] = np.where(days >= 90, (growth ** (365.0 / days) - 1) * 100, np.nan)
    return out


def watch_performance(items: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Watchlist rows + current price and change since added (%)."""
    out = items.copy()
    out["current"] = out["ticker"].map(prices).astype(float)
    out["since_added_pct"] = (out["current"] / out["price"] - 1) * 100
    return out


# ── stored lists ─────────────────────────────────────────────────────────────

@dataclass
class PList:
    name: str
    kind: str                                   # WATCHLIST | PORTFOLIO
    rows: list[dict] = field(default_factory=list)
    # watchlist row: ticker, date (added), price (at add), note
    # portfolio row: ticker, date, price, quantity, side, fees, note

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        if not df.empty:
            df["date"] = df["date"].map(lambda d: dt.date.fromisoformat(d) if isinstance(d, str) and d else d)
        return df


def from_imported(name: str, imp: Imported, prices: pd.Series, today: dt.date | None = None) -> PList:
    """A new list from a clean import. Watchlist tickers get today's date and current price if missing."""
    today = today or dt.date.today()
    rows = imp.rows.copy()
    if imp.kind == WATCHLIST:
        if "date" not in rows.columns or rows["date"].isna().all():
            rows["date"] = today
        else:
            rows["date"] = rows["date"].fillna(today)
        if "price" not in rows.columns or rows["price"].isna().all():
            rows["price"] = rows["ticker"].map(prices).astype(float)
        else:
            rows["price"] = rows["price"].fillna(rows["ticker"].map(prices).astype(float))
    return PList(name, imp.kind, _records(rows))


def add_to_watchlist(plist: PList, ticker: str, price: float, today: dt.date | None = None) -> bool:
    """Add a ticker (False if it is already there)."""
    if any(r["ticker"] == ticker for r in plist.rows):
        return False
    plist.rows.append({"ticker": ticker, "date": (today or dt.date.today()).isoformat(),
                       "price": None if pd.isna(price) else float(price), "note": ""})
    return True


def add_trades(plist: PList, rows: pd.DataFrame) -> list[str]:
    """Append parsed trades. If that would sell more than is held, nothing is added and the tickers
    are returned."""
    bad = oversold(pd.concat([plist.frame(), rows], ignore_index=True))
    if not bad:
        plist.rows += _records(rows)
    return bad


def _records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.to_dict("records"):
        recs.append({k: (v.isoformat() if isinstance(v, dt.date) else None if _blank(v) else v)
                     for k, v in r.items() if k in COLUMNS})
    return recs



def load_all(path: Path | None = None) -> dict[str, PList]:
    path = path or PORTFOLIOS_PATH
    try:
        return load_user_file(path, PList)
    except OSError as e:          # for display only: saving re-reads under the lock and refuses to write
        log.warning("cannot read lists: %s", e)
        return {}


def save(plist: PList, path: Path | None = None) -> None:
    path = path or PORTFOLIOS_PATH
    save_user_file(path, PList, lambda lists: lists.update({plist.name: plist}))


def delete(name: str, path: Path | None = None) -> None:
    path = path or PORTFOLIOS_PATH
    save_user_file(path, PList, lambda lists: lists.pop(name, None))


def _write(lists: dict[str, PList], path: Path) -> None:
    _write_user_file(path, lists)


# ── files out ────────────────────────────────────────────────────────────────

def to_csv(plist: PList) -> bytes:
    """The list in the import format (round trip)."""
    df = plist.frame()
    cols = ["ticker", "date", "price", "note"] if plist.kind == WATCHLIST else COLUMNS
    return df[cols].to_csv(index=False).encode("utf-8-sig")         # BOM: Excel opens Hebrew right


def template_csv() -> bytes:
    rows = [{"ticker": "AAPL", "date": "2025-03-14", "price": 213.49, "quantity": 10, "side": "buy",
             "fees": 2.5, "note": "דוגמה"},
            {"ticker": "MSFT", "date": "2025-06-02", "price": 461.97, "quantity": 5, "side": "buy",
             "fees": 0, "note": ""}]
    return pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False).encode("utf-8-sig")
