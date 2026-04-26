"""
mc_database.py — SQLite persistence for Monte Carlo runs.

Schema:
  runs          - master table, one row per MC run (metrics only)
  equity_curves - daily equity snapshots (only for selected runs: top/bottom)
  trades        - individual trade records (only for selected runs)
  batches       - metadata for each MC batch (all runs in a single execution)

WAL mode is enabled for concurrent writes from multiple processes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("mc_database")

# ─── Allowed column names for ORDER BY / metric params (SQL injection guard) ───
_ALLOWED_ORDER_BY = frozenset({
    "cagr_pct", "total_return_pct", "max_drawdown_pct", "volatility_pct",
    "sharpe", "sortino", "calmar", "win_rate_gross_pct", "win_rate_net_pct", "total_trades",
    "avg_hold_days", "total_tax_paid", "alpha_vs_spy", "excess_cagr_spy",
    "alpha_vs_qqq", "excess_cagr_qqq", "period_start", "period_end",
    "final_capital", "run_id", "started_at", "completed_at",
})


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id          TEXT    PRIMARY KEY,
    strategy_name     TEXT    NOT NULL,
    n_runs_requested  INTEGER NOT NULL,
    n_runs_completed  INTEGER DEFAULT 0,
    started_at        TEXT    NOT NULL,
    completed_at      TEXT,
    data_period_start TEXT,
    data_period_end   TEXT,
    initial_capital   REAL,
    param_space_json  TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid            TEXT    UNIQUE NOT NULL,
    batch_id            TEXT    NOT NULL,
    strategy_name       TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    started_at          TEXT,
    completed_at        TEXT,
    error_message       TEXT,
    params_json         TEXT    NOT NULL,

    initial_capital     REAL,
    final_capital       REAL,
    total_return_pct    REAL,
    cagr_pct            REAL,
    max_drawdown_pct    REAL,
    volatility_pct      REAL,
    sharpe              REAL,
    sortino             REAL,
    calmar              REAL,
    total_trades        INTEGER,
    win_rate_gross_pct  REAL,
    win_rate_net_pct    REAL,
    total_tax_paid      REAL,
    avg_hold_days       REAL,

    alpha_vs_spy        REAL,
    beta_vs_spy         REAL,
    excess_cagr_spy     REAL,
    alpha_vs_qqq        REAL,
    beta_vs_qqq         REAL,
    excess_cagr_qqq     REAL,

    period_start        TEXT,
    period_end          TEXT,

    has_detail_data     INTEGER DEFAULT 0,
    detail_reason       TEXT,

    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_batch        ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_cagr         ON runs(cagr_pct);
CREATE INDEX IF NOT EXISTS idx_runs_strategy     ON runs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_runs_has_detail   ON runs(has_detail_data);
CREATE INDEX IF NOT EXISTS idx_runs_batch_cagr   ON runs(batch_id, cagr_pct);
CREATE INDEX IF NOT EXISTS idx_runs_batch_status ON runs(batch_id, status);
CREATE INDEX IF NOT EXISTS idx_runs_sharpe       ON runs(sharpe);
CREATE INDEX IF NOT EXISTS idx_runs_win_rate     ON runs(win_rate_gross_pct);

CREATE TABLE IF NOT EXISTS equity_curves (
    run_id         INTEGER NOT NULL,
    date           TEXT    NOT NULL,
    capital        REAL    NOT NULL,
    cash           REAL,
    open_positions INTEGER,
    PRIMARY KEY (run_id, date),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    ticker         TEXT    NOT NULL,
    buy_date       TEXT    NOT NULL,
    sell_date      TEXT    NOT NULL,
    hold_days      INTEGER,
    buy_price      REAL,
    sell_price     REAL,
    shares         REAL,
    gross_pnl      REAL,
    net_pnl        REAL,
    buy_reason     TEXT,
    sell_reason    TEXT,
    sector         TEXT,
    industry       TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_trades_run    ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
"""


@dataclass
class RunRecord:
    """Structured record for a single completed MC run (metrics only)."""
    run_uuid: str
    batch_id: str
    strategy_name: str
    status: str                 # 'completed' / 'failed' / 'timeout'
    started_at: str
    completed_at: str
    error_message: Optional[str]
    params: Dict[str, Any]

    # absolute metrics
    initial_capital: float
    final_capital: Optional[float]
    total_return_pct: Optional[float]
    cagr_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    volatility_pct: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    calmar: Optional[float]
    total_trades: Optional[int]
    win_rate_gross_pct: Optional[float]
    win_rate_net_pct: Optional[float]
    total_tax_paid: Optional[float]
    avg_hold_days: Optional[float]

    # relative metrics
    alpha_vs_spy: Optional[float]
    beta_vs_spy: Optional[float]
    excess_cagr_spy: Optional[float]
    alpha_vs_qqq: Optional[float]
    beta_vs_qqq: Optional[float]
    excess_cagr_qqq: Optional[float]

    period_start: Optional[str]
    period_end: Optional[str]


class MCDatabase:
    """SQLite wrapper for MC persistence. Supports concurrent writes via WAL."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as conn:
            # WAL mode — critical for concurrent writes from multiple processes
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -64000")   # 64MB
            conn.executescript(SCHEMA_SQL)
            
            # --- Auto-Migration for existing databases (adds missing columns if needed) ---
            try:
                # בדיקה האם העמודה הישנה עדיין קיימת וצריך להוסיף את החדשות (מנגנון שדרוג שקט)
                cur = conn.execute("PRAGMA table_info(runs)")
                columns = [row["name"] for row in cur.fetchall()]
                if "win_rate_gross_pct" not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN win_rate_gross_pct REAL;")
                if "win_rate_net_pct" not in columns:
                    conn.execute("ALTER TABLE runs ADD COLUMN win_rate_net_pct REAL;")
            except Exception as e:
                logger.debug(f"Schema migration skipped/failed: {e}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register_batch(self, strategy_name: str, n_runs: int,
                        param_space: Dict[str, Any],
                        initial_capital: float,
                        data_period: Optional[tuple[str, str]] = None,
                        notes: str = "") -> str:
        batch_id = str(uuid.uuid4())[:8]
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO batches (
                    batch_id, strategy_name, n_runs_requested, started_at,
                    data_period_start, data_period_end, initial_capital,
                    param_space_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                batch_id, strategy_name, n_runs, datetime.now().isoformat(),
                data_period[0] if data_period else None,
                data_period[1] if data_period else None,
                initial_capital,
                json.dumps(param_space, default=str),
                notes,
            ))
        logger.info(f"Registered batch {batch_id} | {strategy_name} | {n_runs} runs")
        return batch_id

    def finalize_batch(self, batch_id: str, n_completed: int):
        with self._conn() as conn:
            conn.execute("""
                UPDATE batches
                SET completed_at = ?, n_runs_completed = ?
                WHERE batch_id = ?
            """, (datetime.now().isoformat(), n_completed, batch_id))

    def list_batches(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql("""
                SELECT
                    b.batch_id, b.strategy_name, b.n_runs_requested, b.n_runs_completed,
                    b.started_at, b.completed_at, b.initial_capital,
                    b.data_period_start, b.data_period_end, b.notes,
                    (SELECT MAX(cagr_pct) FROM runs WHERE batch_id = b.batch_id) AS best_cagr,
                    (SELECT MIN(cagr_pct) FROM runs WHERE batch_id = b.batch_id) AS worst_cagr
                FROM batches b
                ORDER BY b.started_at DESC
            """, conn)
        return df

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def insert_run(self, rec: RunRecord) -> int:
        """Insert a run record, return assigned run_id."""
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO runs (
                    run_uuid, batch_id, strategy_name, status,
                    started_at, completed_at, error_message, params_json,
                    initial_capital, final_capital, total_return_pct, cagr_pct,
                    max_drawdown_pct, volatility_pct, sharpe, sortino, calmar,
                    total_trades, win_rate_gross_pct, win_rate_net_pct, total_tax_paid, avg_hold_days,
                    alpha_vs_spy, beta_vs_spy, excess_cagr_spy,
                    alpha_vs_qqq, beta_vs_qqq, excess_cagr_qqq,
                    period_start, period_end
                ) VALUES (?, ?, ?, ?,  ?, ?, ?, ?,
                          ?, ?, ?, ?,  ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?,
                          ?, ?, ?,  ?, ?, ?,
                          ?, ?)
            """, (
                rec.run_uuid, rec.batch_id, rec.strategy_name, rec.status,
                rec.started_at, rec.completed_at, rec.error_message,
                json.dumps(rec.params, default=str),
                rec.initial_capital, rec.final_capital, rec.total_return_pct,
                rec.cagr_pct, rec.max_drawdown_pct, rec.volatility_pct,
                rec.sharpe, rec.sortino, rec.calmar,
                rec.total_trades, rec.win_rate_gross_pct, rec.win_rate_net_pct, rec.total_tax_paid, rec.avg_hold_days,
                rec.alpha_vs_spy, rec.beta_vs_spy, rec.excess_cagr_spy,
                rec.alpha_vs_qqq, rec.beta_vs_qqq, rec.excess_cagr_qqq,
                rec.period_start, rec.period_end,
            ))
            run_id = cur.lastrowid
        return run_id

    def insert_detail_data(self, run_id: int,
                            equity_df: pd.DataFrame,
                            trades_df: pd.DataFrame,
                            reason: str = "top25"):
        with self._conn() as conn:
            if equity_df is not None and not equity_df.empty:
                eq = equity_df.copy()
                eq["Date"] = pd.to_datetime(eq["Date"]).dt.strftime("%Y-%m-%d")
                has_cash = "Cash" in eq.columns
                has_pos  = "Open_Positions" in eq.columns
                rows = [
                    (
                        run_id,
                        row.Date,
                        float(row.Capital),
                        float(row.Cash) if has_cash else 0.0,
                        int(row.Open_Positions) if has_pos else 0,
                    )
                    for row in eq.itertuples(index=False)
                ]
                conn.executemany("""
                    INSERT OR REPLACE INTO equity_curves
                    (run_id, date, capital, cash, open_positions)
                    VALUES (?, ?, ?, ?, ?)
                """, rows)

            if trades_df is not None and not trades_df.empty:
                t = trades_df.copy()
                for dc in ("Buy_Date", "Sell_Date"):
                    if dc in t.columns:
                        t[dc] = pd.to_datetime(t[dc]).dt.strftime("%Y-%m-%d")
                has_sector   = "Sector" in t.columns
                has_industry = "Industry" in t.columns
                rows = []
                for row in t.itertuples(index=False):
                    rows.append((
                        run_id,
                        str(getattr(row, "Ticker", "")),
                        str(getattr(row, "Buy_Date", "")),
                        str(getattr(row, "Sell_Date", "")),
                        int(row.Hold_Days)   if pd.notna(getattr(row, "Hold_Days",   None)) else None,
                        float(row.Buy_Price) if pd.notna(getattr(row, "Buy_Price",   None)) else None,
                        float(row.Sell_Price)if pd.notna(getattr(row, "Sell_Price",  None)) else None,
                        float(row.Shares)    if pd.notna(getattr(row, "Shares",      None)) else None,
                        float(row.Gross_PnL) if pd.notna(getattr(row, "Gross_PnL",  None)) else None,
                        float(row.Net_PnL)   if pd.notna(getattr(row, "Net_PnL",    None)) else None,
                        str(getattr(row, "Buy_Reason",  "")),
                        str(getattr(row, "Sell_Reason", "")),
                        str(row.Sector)   if has_sector   else None,
                        str(row.Industry) if has_industry else None,
                    ))
                conn.executemany("""
                    INSERT INTO trades (
                        run_id, ticker, buy_date, sell_date, hold_days,
                        buy_price, sell_price, shares, gross_pnl, net_pnl,
                        buy_reason, sell_reason, sector, industry
                    ) VALUES (?, ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, ?)
                """, rows)

            conn.execute("""
                UPDATE runs SET has_detail_data = 1, detail_reason = ?
                WHERE run_id = ?
            """, (reason, run_id))

    def query_runs(self, batch_id: Optional[str] = None,
                   strategy: Optional[str] = None,
                   min_cagr: Optional[float] = None,
                   max_dd_better_than: Optional[float] = None,
                   order_by: str = "cagr_pct",
                   descending: bool = True,
                   limit: Optional[int] = None) -> pd.DataFrame:
        if order_by not in _ALLOWED_ORDER_BY:
            raise ValueError(
                f"Invalid order_by column: '{order_by}'. "
                f"Allowed: {sorted(_ALLOWED_ORDER_BY)}"
            )
        where = ["status = 'completed'"]
        params: list = []
        if batch_id:
            where.append("batch_id = ?")
            params.append(batch_id)
        if strategy:
            where.append("strategy_name = ?")
            params.append(strategy)
        if min_cagr is not None:
            where.append("cagr_pct >= ?")
            params.append(min_cagr)
        if max_dd_better_than is not None:
            where.append("max_drawdown_pct >= ?")
            params.append(max_dd_better_than)

        direction = "DESC" if descending else "ASC"
        sql = f"""
            SELECT * FROM runs
            WHERE {' AND '.join(where)}
            ORDER BY {order_by} {direction}
        """
        if limit:
            sql += f" LIMIT {int(limit)}"

        with self._conn() as conn:
            df = pd.read_sql(sql, conn, params=params)
        if not df.empty and "params_json" in df.columns:
            df["params"] = df["params_json"].apply(lambda s: json.loads(s) if s else {})
        return df

    def get_top_and_bottom(self, batch_id: str,
                            top_n: int = 25, bottom_n: int = 10,
                            metric: str = "cagr_pct") -> tuple[list[int], list[int]]:
        if metric not in _ALLOWED_ORDER_BY:
            raise ValueError(
                f"Invalid metric: '{metric}'. "
                f"Allowed: {sorted(_ALLOWED_ORDER_BY)}"
            )
        with self._conn() as conn:
            cur = conn.execute(f"""
                SELECT run_id FROM runs
                WHERE batch_id = ? AND status = 'completed' AND {metric} IS NOT NULL
                ORDER BY {metric} DESC
                LIMIT ?
            """, (batch_id, top_n))
            top_ids = [row["run_id"] for row in cur.fetchall()]

            cur = conn.execute(f"""
                SELECT run_id FROM runs
                WHERE batch_id = ? AND {metric} IS NOT NULL
                ORDER BY {metric} ASC
                LIMIT ?
            """, (batch_id, bottom_n))
            bottom_ids = [row["run_id"] for row in cur.fetchall()]
        return top_ids, bottom_ids

    def get_run_details(self, run_id: int) -> Dict[str, Any]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            run_row = cur.fetchone()
            if not run_row:
                raise ValueError(f"run_id {run_id} not found")
            result = dict(run_row)
            result["params"] = json.loads(result["params_json"]) if result["params_json"] else {}

            eq = pd.read_sql("SELECT * FROM equity_curves WHERE run_id = ? ORDER BY date", conn, params=(run_id,))
            if not eq.empty:
                eq["date"] = pd.to_datetime(eq["date"])
                eq = eq.rename(columns={"date": "Date", "capital": "Capital",
                                         "cash": "Cash", "open_positions": "Open_Positions"})
            result["equity_curve"] = eq

            tr = pd.read_sql("SELECT * FROM trades WHERE run_id = ? ORDER BY buy_date", conn, params=(run_id,))
            if not tr.empty:
                tr = tr.rename(columns={
                    "ticker": "Ticker", "buy_date": "Buy_Date", "sell_date": "Sell_Date",
                    "hold_days": "Hold_Days", "buy_price": "Buy_Price",
                    "sell_price": "Sell_Price", "shares": "Shares",
                    "gross_pnl": "Gross_PnL", "net_pnl": "Net_PnL",
                    "buy_reason": "Buy_Reason", "sell_reason": "Sell_Reason",
                    "sector": "Sector", "industry": "Industry",
                })
                for dc in ("Buy_Date", "Sell_Date"):
                    tr[dc] = pd.to_datetime(tr[dc])
            result["trades"] = tr

        return result

    def export_run_to_xlsx(self, run_id: int, output_path: str | Path):
        details = self.get_run_details(run_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params_df = pd.DataFrame([
            {"Parameter": "strategy_class", "Value": details["strategy_name"]},
            {"Parameter": "run_uuid",       "Value": details["run_uuid"]},
            {"Parameter": "batch_id",       "Value": details["batch_id"]},
        ] + [
            {"Parameter": k, "Value": v} for k, v in details["params"].items()
        ])

        metrics_df = pd.DataFrame([{
            "Metric": k, "Value": details.get(k)
        } for k in [
            "cagr_pct", "total_return_pct", "max_drawdown_pct", "volatility_pct",
            "sharpe", "sortino", "calmar", "total_trades", "win_rate_gross_pct", "win_rate_net_pct",
            "total_tax_paid", "alpha_vs_spy", "beta_vs_spy",
        ]])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if details["equity_curve"] is not None and not details["equity_curve"].empty:
                details["equity_curve"].to_excel(writer, sheet_name="Equity", index=False)
            if details["trades"] is not None and not details["trades"].empty:
                details["trades"].to_excel(writer, sheet_name="Trades", index=False)
            params_df.to_excel(writer, sheet_name="Params", index=False)
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        logger.info(f"Exported run {run_id} → {output_path}")
        return output_path

    def vacuum(self):
        with self._conn() as conn:
            conn.execute("VACUUM")

    def stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            return {
                "batches": conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "runs_with_detail": conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE has_detail_data = 1"
                ).fetchone()[0],
                "trades": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
                "equity_points": conn.execute("SELECT COUNT(*) FROM equity_curves").fetchone()[0],
            }