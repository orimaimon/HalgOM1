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
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from screener.data.store import DATA_DIR

log = logging.getLogger(__name__)

PORTFOLIOS_PATH = DATA_DIR / "portfolios.json"

COLUMNS = ["ticker", "date", "price", "quantity", "side", "fees", "note"]
TRADE_COLUMNS = ("date", "price", "quantity")      # all three make a row a trade
WATCHLIST, PORTFOLIO = "watchlist", "portfolio"
BUY, SELL = "buy", "sell"

# header (lower case, stripped) -> canonical column
HEADERS = {
    **{c: c for c in COLUMNS},
    "symbol": "ticker", "סימול": "ticker", "טיקר": "ticker", "נייר": "ticker", "מניה": "ticker",
    "תאריך": "date", "תאריך קנייה": "date", "תאריך קניה": "date", "trade date": "date",
    "שער": "price", "מחיר": "price", "שער קנייה": "price", "שער קניה": "price", "מחיר קנייה": "price",
    "qty": "quantity", "shares": "quantity", "כמות": "quantity",
    "action": "side", "type": "side", "פעולה": "side", "סוג": "side",
    "fee": "fees", "commission": "fees", "עמלה": "fees", "עמלות": "fees",
    "notes": "note", "הערה": "note", "הערות": "note",
}
SIDES = {"buy": BUY, "b": BUY, "קנייה": BUY, "קניה": BUY, "sell": SELL, "s": SELL, "מכירה": SELL}
_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,2})?$")


# ── reading a file ───────────────────────────────────────────────────────────

def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """The raw table of an uploaded CSV / Excel file, every cell as text or as Excel gave it."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=object)
    for encoding in ("utf-8-sig", "cp1255"):          # cp1255: Hebrew CSV saved by older Excel
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding, skipinitialspace=True)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(data), dtype=str, encoding="latin-1")


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

    @property
    def ok(self) -> bool:
        return not self.problems and not self.rows.empty


def normalize_ticker(v) -> str:
    """'brk-b ' / 'BRK/B' -> 'BRK.B' (TradingView's class-share form)."""
    return re.sub(r"[-/]", ".", str(v).strip().upper())


def parse_date(v, today: dt.date) -> dt.date | None:
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return v.date()
    if isinstance(v, dt.date):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_number(v) -> float | None:
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return None if pd.isna(v) else float(v)
    s = str(v).strip().replace("$", "").replace(",", "").replace("‏", "").replace("‎", "")
    try:
        return float(s)
    except ValueError:
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
    kind = PORTFOLIO if any(c in df.columns and not df[c].map(_blank).all() for c in TRADE_COLUMNS) \
        else WATCHLIST

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
        if kind == PORTFOLIO:
            for c in TRADE_COLUMNS:
                if c not in df.columns or _blank(r[c]):
                    problems.append(Problem(n, "missing_value", c))
            if "date" in df.columns and not _blank(r["date"]):
                d = parse_date(r["date"], today)
                if d is None:
                    problems.append(Problem(n, "bad_date", str(r["date"])))
                elif d > today:
                    problems.append(Problem(n, "future_date", d.isoformat()))
                rec["date"] = d
            for c in ("price", "quantity", "fees"):
                if c in df.columns and not _blank(r[c]):
                    x = parse_number(r[c])
                    if x is None or x < 0 or (x == 0 and c != "fees"):
                        problems.append(Problem(n, "bad_number", f"{c}: {r[c]}"))
                    rec[c] = x
            side = "buy" if "side" not in df.columns or _blank(r["side"]) else str(r["side"]).strip().lower()
            if side not in SIDES:
                problems.append(Problem(n, "bad_side", side))
            rec["side"] = SIDES.get(side, BUY)
            rec.setdefault("fees", 0.0)
        if len(problems) == row_problems:
            out.append(rec)
    rows = pd.DataFrame(out, columns=COLUMNS)
    if kind == WATCHLIST:
        rows = rows.drop_duplicates("ticker")
    elif not rows.empty:
        problems += [Problem(0, "oversold", t) for t in oversold(rows)]
    return Imported(kind, rows.reset_index(drop=True), problems)


# ── positions ────────────────────────────────────────────────────────────────

def oversold(trades: pd.DataFrame) -> list[str]:
    """Tickers where, in date order, more shares were sold than held."""
    bad = []
    for t, g in trades.sort_values("date", kind="stable").groupby("ticker", sort=False):
        held = (g["quantity"] * np.where(g["side"] == SELL, -1, 1)).cumsum()
        if (held < -1e-9).any():
            bad.append(t)
    return bad


def positions(trades: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: quantity, avg_cost (incl. buy fees), cost (of the shares held), realized P&L,
    first_buy and avg_date (cost-weighted buy date, for the annual return). Closed positions stay,
    with quantity 0, so their realized P&L is not lost."""
    rows = []
    trades = trades.assign(fees=trades["fees"].fillna(0.0))
    for t, g in trades.sort_values("date", kind="stable").groupby("ticker", sort=False):
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
    """A new list from a clean import. Watchlist tickers get today's date and current price."""
    today = today or dt.date.today()
    rows = imp.rows.copy()
    if imp.kind == WATCHLIST:
        rows["date"], rows["price"] = today, rows["ticker"].map(prices).astype(float)
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
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {name: PList(**p) for name, p in raw.items()}
    except Exception as e:                        # a damaged file must not break the screener
        log.warning("cannot read lists: %s", e)
        return {}


def save(plist: PList, path: Path | None = None) -> None:
    path = path or PORTFOLIOS_PATH
    lists = load_all(path)
    lists[plist.name] = plist
    _write(lists, path)


def delete(name: str, path: Path | None = None) -> None:
    path = path or PORTFOLIOS_PATH
    lists = load_all(path)
    if lists.pop(name, None) is not None:
        _write(lists, path)


def _write(lists: dict[str, PList], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({n: asdict(p) for n, p in lists.items()}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)                       # a crash mid-write never leaves a half-written file


# ── files out ────────────────────────────────────────────────────────────────

def to_csv(plist: PList) -> bytes:
    """The list in the import format (round trip). Watchlists keep only ticker and note."""
    df = plist.frame()
    cols = ["ticker", "note"] if plist.kind == WATCHLIST else COLUMNS
    return df[cols].to_csv(index=False).encode("utf-8-sig")         # BOM: Excel opens Hebrew right


def template_csv() -> bytes:
    rows = [{"ticker": "AAPL", "date": "2025-03-14", "price": 213.49, "quantity": 10, "side": "buy",
             "fees": 2.5, "note": "דוגמה"},
            {"ticker": "MSFT", "date": "2025-06-02", "price": 461.97, "quantity": 5, "side": "buy",
             "fees": 0, "note": ""}]
    return pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False).encode("utf-8-sig")
