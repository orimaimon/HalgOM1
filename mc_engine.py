"""
mc_engine.py — The Monte Carlo orchestrator.

Two-pass architecture:
  Pass 1 — Random search with parallel workers, metrics only.
  Pass 2 — Re-run Top-N + Bottom-M with detail tracking (equity + trades),
           saved to SQLite.

Why two passes?
  - Pass 1 is fast (metrics only, zero serialization overhead)
  - Pass 2 is deterministic (same params → same results for saved runs)
  - Memory-safe: no huge in-memory buffers of equity/trades per worker
  - Resumable: if Pass 1 crashes mid-run, the DB still has everything
"""
from __future__ import annotations

import logging
import random
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mc_database import MCDatabase, RunRecord
from mc_strategies import ParamSpace, get_strategy_config

logger = logging.getLogger("mc_engine")


# ════════════════════════════════════════════════════════════════════════════
# Worker function (runs in subprocess)
# ════════════════════════════════════════════════════════════════════════════

def _worker_execute(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single backtest in a subprocess.
    Returns metrics + optionally equity_curve + trades (for pass 2).
    """
    from core_engine import BacktestEngine
    from mc_strategies import get_strategy_config

    started_at = datetime.now().isoformat()
    run_uuid = task["run_uuid"]

    result: Dict[str, Any] = {
        "run_uuid": run_uuid,
        "status": "failed",
        "started_at": started_at,
        "completed_at": None,
        "error_message": None,
        "params": task["params"],
        "metrics": {},
        "equity_curve": None,
        "trades": None,
    }

    try:
        # Load data
        df = pd.read_parquet(task["data_path"], columns=task["cols_needed"])
        if task.get("date_filter"):
            start, end = task["date_filter"]
            if start:
                df = df[df["Date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["Date"] <= pd.to_datetime(end)]

        # Build strategy
        cfg = get_strategy_config(task["strategy_name"])
        strategy = cfg["builder"](task["params"])

        # Run backtest
        engine = BacktestEngine(
            data=df,
            strategy=strategy,
            initial_capital=task["initial_capital"],
            commission=task["commission"],
            slippage=task["slippage"],
        )
        backtest_results = engine.run()

        # Compute metrics
        metrics = _compute_metrics(
            backtest_results["Equity_Curve"],
            backtest_results["Trades"],
            task["initial_capital"],
            benchmarks_path=task.get("benchmarks_path"),
        )
        result["metrics"] = metrics

        # Return details if requested (Pass 2)
        if task.get("return_details"):
            result["equity_curve"] = backtest_results["Equity_Curve"]
            result["trades"] = backtest_results["Trades"]

        result["status"] = "completed"
    except Exception as e:
        result["error_message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    result["completed_at"] = datetime.now().isoformat()
    return result


def _compute_metrics(equity_df: pd.DataFrame,
                     trades_df: pd.DataFrame,
                     initial_capital: float,
                     benchmarks_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute all metrics for a single run."""
    metrics = {
        "initial_capital": initial_capital,
        "final_capital": None, "total_return_pct": None, "cagr_pct": None,
        "max_drawdown_pct": None, "volatility_pct": None,
        "sharpe": None, "sortino": None, "calmar": None,
        "total_trades": 0, "win_rate_gross_pct": 0.0, "win_rate_net_pct": 0.0,
        "total_tax_paid": 0.0, "avg_hold_days": None,
        "alpha_vs_spy": None, "beta_vs_spy": None, "excess_cagr_spy": None,
        "alpha_vs_qqq": None, "beta_vs_qqq": None, "excess_cagr_qqq": None,
        "period_start": None, "period_end": None,
    }

    if equity_df is None or equity_df.empty:
        return metrics

    eq = equity_df.sort_values("Date").reset_index(drop=True)
    eq["Date"] = pd.to_datetime(eq["Date"])
    metrics["period_start"] = str(eq["Date"].iloc[0].date())
    metrics["period_end"] = str(eq["Date"].iloc[-1].date())

    capital = eq["Capital"].astype(float)
    final = float(capital.iloc[-1])
    metrics["final_capital"] = round(final, 2)
    
    # מונע קריסה במקרה של הון התחלתי 0
    if initial_capital > 0:
        metrics["total_return_pct"] = round((final / initial_capital - 1) * 100, 2)

        # CAGR
        years = (eq["Date"].iloc[-1] - eq["Date"].iloc[0]).days / 365.25
        if years > 0:
            cagr = ((final / initial_capital) ** (1 / years) - 1) * 100
            metrics["cagr_pct"] = round(float(cagr), 2)

    # Max drawdown
    peak = capital.cummax()
    dd = (capital - peak) / peak * 100
    metrics["max_drawdown_pct"] = round(float(dd.min()), 2)

    # Volatility + Sharpe + Sortino
    returns = capital.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        vol_ann = float(returns.std() * np.sqrt(252) * 100)
        metrics["volatility_pct"] = round(vol_ann, 2)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
        metrics["sharpe"] = round(sharpe, 2)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(returns.mean() / downside.std() * np.sqrt(252))
            metrics["sortino"] = round(sortino, 2)

    # Calmar
    if metrics["cagr_pct"] is not None and metrics["max_drawdown_pct"] is not None:
        if metrics["max_drawdown_pct"] < 0:
            metrics["calmar"] = round(
                metrics["cagr_pct"] / abs(metrics["max_drawdown_pct"]), 2
            )

    # Trades
    if trades_df is not None and not trades_df.empty:
        metrics["total_trades"] = int(len(trades_df))
        
        # חישוב נכון של ה-Win Rate (ברוטו ונטו)
        metrics["win_rate_gross_pct"] = round(float((trades_df["Gross_PnL"] > 0).mean() * 100), 2)
        if "Net_PnL" in trades_df.columns:
            metrics["win_rate_net_pct"] = round(float((trades_df["Net_PnL"] > 0).mean() * 100), 2)
        else:
            metrics["win_rate_net_pct"] = metrics["win_rate_gross_pct"]

        if "Hold_Days" in trades_df.columns:
            metrics["avg_hold_days"] = round(float(trades_df["Hold_Days"].mean()), 1)
            
        # Sum of taxes (from Net_PnL vs Gross_PnL diff)
        if "Net_PnL" in trades_df.columns and "Gross_PnL" in trades_df.columns:
            metrics["total_tax_paid"] = round(
                float((trades_df["Gross_PnL"] - trades_df["Net_PnL"]).sum()), 2
            )

    # Benchmarks
    if benchmarks_path:
        try:
            alpha_spy, beta_spy, excess_spy = _quick_alpha_beta(
                eq, benchmarks_path, "SPY", initial_capital
            )
            alpha_qqq, beta_qqq, excess_qqq = _quick_alpha_beta(
                eq, benchmarks_path, "QQQ", initial_capital
            )
            metrics["alpha_vs_spy"] = round(alpha_spy, 2) if alpha_spy is not None else None
            metrics["beta_vs_spy"] = round(beta_spy, 3) if beta_spy is not None else None
            metrics["excess_cagr_spy"] = round(excess_spy, 2) if excess_spy is not None else None
            metrics["alpha_vs_qqq"] = round(alpha_qqq, 2) if alpha_qqq is not None else None
            metrics["beta_vs_qqq"] = round(beta_qqq, 3) if beta_qqq is not None else None
            metrics["excess_cagr_qqq"] = round(excess_qqq, 2) if excess_qqq is not None else None
        except Exception as e:
            logger.debug(f"benchmark calc failed: {e}")

    return metrics


def _quick_alpha_beta(equity_df: pd.DataFrame, benchmarks_path: str,
                      ticker_prefix: str, initial_capital: float
                      ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Quick CAPM alpha/beta vs a benchmark. Loads benchmarks parquet inline."""
    try:
        b = pd.read_parquet(benchmarks_path)
        b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
        adj_col = f"{ticker_prefix}_Adj_Close"
        if adj_col not in b.columns:
            return None, None, None

        dates_set = set(pd.to_datetime(equity_df["Date"]).dt.normalize())
        bsub = b[b["Date"].isin(dates_set)][["Date", adj_col]].copy()
        bsub[adj_col] = bsub[adj_col].ffill()
        if bsub.empty or bsub[adj_col].iloc[0] <= 0:
            return None, None, None
            
        first_px = bsub[adj_col].iloc[0]
        shares = initial_capital / first_px
        bsub["b_cap"] = bsub[adj_col] * shares

        s = equity_df[["Date", "Capital"]].copy()
        s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
        merged = s.merge(bsub[["Date", "b_cap"]], on="Date", how="inner")
        if len(merged) < 30:
            return None, None, None

        s_ret = merged["Capital"].pct_change().dropna()
        b_ret = merged["b_cap"].pct_change().dropna()
        common = s_ret.index.intersection(b_ret.index)
        s_ret, b_ret = s_ret.loc[common], b_ret.loc[common]
        if b_ret.std() == 0:
            return None, None, None

        cov = float(np.cov(s_ret, b_ret, ddof=1)[0, 1])
        varb = float(b_ret.var(ddof=1))
        beta = cov / varb

        years = (merged["Date"].iloc[-1] - merged["Date"].iloc[0]).days / 365.25
        if years > 0:
            s_cagr = (merged["Capital"].iloc[-1] / merged["Capital"].iloc[0]) ** (1/years) - 1
            b_cagr = (merged["b_cap"].iloc[-1] / merged["b_cap"].iloc[0]) ** (1/years) - 1
            alpha = (s_cagr - beta * b_cagr) * 100
            excess = (s_cagr - b_cagr) * 100
            return alpha, beta, excess
            
    except Exception as e:
        # במקום לבלוע שגיאות בשקט (pass), אנחנו מדווחים ללוג (Debug) כדי שנוכל לחקור קריסות
        logger.debug(f"Alpha/Beta calc failed for {ticker_prefix}: {e}")
        return None, None, None


# ════════════════════════════════════════════════════════════════════════════
# MCRunner — orchestrates the full MC workflow
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MCConfig:
    strategy_name: str            # 'topn' / 'momentum'
    n_runs: int = 200
    n_workers: int = 0            # 0 = auto
    seed: Optional[int] = None
    initial_capital: float = 100_000.0
    commission: float = 1.0
    slippage: float = 0.001
    data_path: str = "DB/simulation_db.parquet"
    benchmarks_path: str = "DB/benchmarks.parquet"
    db_path: str = "DB/mc_runs.sqlite"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    top_n_detail: int = 25
    bottom_n_detail: int = 10
    worker_timeout: int = 600     # seconds per worker before TimeoutError
    notes: str = ""


class MCRunner:
    def __init__(self, config: MCConfig):
        self.config = config
        self.db = MCDatabase(config.db_path)
        self.batch_id: Optional[str] = None
        self._rng = random.Random(config.seed) if config.seed is not None else random.Random()

    def run(self) -> str:
        """Execute full MC workflow. Returns batch_id."""
        cfg = self.config
        strat_cfg = get_strategy_config(cfg.strategy_name)

        # Register batch
        self.batch_id = self.db.register_batch(
            strategy_name=strat_cfg["display_name"],
            n_runs=cfg.n_runs,
            param_space=strat_cfg["param_space"].to_dict(),
            initial_capital=cfg.initial_capital,
            data_period=(cfg.start_date, cfg.end_date),
            notes=cfg.notes,
        )
        logger.info(f"Batch {self.batch_id} started | "
                    f"{cfg.strategy_name} | {cfg.n_runs} runs")

        # Pass 1 — sample params and run
        param_sets = self._sample_param_sets(cfg.n_runs)
        tasks_pass1 = self._build_tasks(param_sets, strat_cfg, return_details=False)

        logger.info(f"Pass 1: metrics only — {cfg.n_runs} runs")
        n_completed = self._run_parallel(tasks_pass1, pass_name="pass1")
        logger.info(f"Pass 1 complete: {n_completed}/{cfg.n_runs} succeeded")

        if n_completed == 0:
            logger.error("No runs succeeded in pass 1. Aborting.")
            self.db.finalize_batch(self.batch_id, 0)
            return self.batch_id

        # Pass 2 — re-run Top + Bottom with details
        top_ids, bottom_ids = self.db.get_top_and_bottom(
            self.batch_id, cfg.top_n_detail, cfg.bottom_n_detail, metric="cagr_pct"
        )
        detail_run_ids = top_ids + bottom_ids
        logger.info(f"Pass 2: re-running {len(detail_run_ids)} selected runs "
                    f"({len(top_ids)} top + {len(bottom_ids)} bottom) for detail tracking")

        if detail_run_ids:
            detail_tasks = self._build_detail_tasks(detail_run_ids, strat_cfg)
            self._run_detail_parallel(detail_tasks)

        self.db.finalize_batch(self.batch_id, n_completed)
        logger.info(f"Batch {self.batch_id} finalized.")
        return self.batch_id

    def _sample_param_sets(self, n: int) -> List[Dict[str, Any]]:
        cfg = self.config
        ps: ParamSpace = get_strategy_config(cfg.strategy_name)["param_space"]
        return [ps.sample(self._rng) for _ in range(n)]

    def _build_tasks(self, param_sets: List[Dict[str, Any]],
                     strat_cfg: Dict[str, Any],
                     return_details: bool) -> List[Dict[str, Any]]:
        cfg = self.config
        tasks = []
        for params in param_sets:
            tasks.append({
                "run_uuid": str(uuid.uuid4()),
                "strategy_name": cfg.strategy_name,
                "params": params,
                "data_path": cfg.data_path,
                "cols_needed": strat_cfg["cols_needed"],
                "initial_capital": cfg.initial_capital,
                "commission": cfg.commission,
                "slippage": cfg.slippage,
                "benchmarks_path": cfg.benchmarks_path,
                "return_details": return_details,
                "date_filter": (cfg.start_date, cfg.end_date),
            })
        return tasks

    def _build_detail_tasks(self, run_ids: List[int],
                             strat_cfg: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
        """For pass 2: get params of each run_id and build a detail task."""
        cfg = self.config
        tasks = []
        for rid in run_ids:
            details = self.db.get_run_details(rid)
            params = details["params"]
            task = {
                "run_uuid": details["run_uuid"] + "-detail",
                "_source_run_uuid": details["run_uuid"],   
                "strategy_name": cfg.strategy_name,
                "params": params,
                "data_path": cfg.data_path,
                "cols_needed": strat_cfg["cols_needed"],
                "initial_capital": cfg.initial_capital,
                "commission": cfg.commission,
                "slippage": cfg.slippage,
                "benchmarks_path": None,        
                "return_details": True,
                "date_filter": (cfg.start_date, cfg.end_date),
            }
            tasks.append((rid, task))
        return tasks

    def _run_parallel(self, tasks: List[Dict[str, Any]], pass_name: str) -> int:
        cfg = self.config
        workers = cfg.n_workers if cfg.n_workers > 0 else None
        t0 = time.time()
        n_completed = 0
        n_failed = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker_execute, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    result = fut.result(timeout=cfg.worker_timeout)
                except Exception as e:
                    logger.error(f"Worker died unexpectedly: {e}")
                    n_failed += 1
                    continue

                self._persist_pass1_result(result)
                if result["status"] == "completed":
                    n_completed += 1
                else:
                    n_failed += 1

                if i % 20 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - i) / rate if rate > 0 else 0
                    logger.info(f"[{pass_name}] {i}/{len(tasks)} done | "
                                f"OK={n_completed} FAIL={n_failed} | "
                                f"{rate:.1f}/s | ETA {eta/60:.1f}min")

        return n_completed

    def _run_detail_parallel(self, tasks: List[Tuple[int, Dict[str, Any]]]):
        cfg = self.config
        workers = cfg.n_workers if cfg.n_workers > 0 else None
        t0 = time.time()

        uuid_to_runid = {
            task["_source_run_uuid"]: rid
            for rid, task in tasks
        }
        raw_tasks = [t for _, t in tasks]

        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_uuid = {
                ex.submit(_worker_execute, t): t["_source_run_uuid"]
                for t in raw_tasks
            }
            for i, fut in enumerate(as_completed(fut_to_uuid), 1):
                source_uuid = fut_to_uuid[fut]
                run_id = uuid_to_runid[source_uuid]
                try:
                    result = fut.result(timeout=cfg.worker_timeout)
                except Exception as e:
                    logger.warning(f"Pass-2 run_id={run_id} failed: {e}")
                    continue

                if result["status"] == "completed":
                    try:
                        self.db.insert_detail_data(
                            run_id=run_id,
                            equity_df=result["equity_curve"],
                            trades_df=result["trades"],
                            reason="top_bottom",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save detail for run_id={run_id}: {e}")

                if i % 5 == 0 or i == len(raw_tasks):
                    elapsed = time.time() - t0
                    logger.info(f"[pass2] {i}/{len(raw_tasks)} detail runs done | {elapsed:.1f}s")

    def _persist_pass1_result(self, result: Dict[str, Any]):
        m = result.get("metrics", {})
        rec = RunRecord(
            run_uuid=result["run_uuid"],
            batch_id=self.batch_id,
            strategy_name=self.config.strategy_name,
            status=result["status"],
            started_at=result["started_at"],
            completed_at=result["completed_at"],
            error_message=result.get("error_message"),
            params=result["params"],

            initial_capital=m.get("initial_capital", self.config.initial_capital),
            final_capital=m.get("final_capital"),
            total_return_pct=m.get("total_return_pct"),
            cagr_pct=m.get("cagr_pct"),
            max_drawdown_pct=m.get("max_drawdown_pct"),
            volatility_pct=m.get("volatility_pct"),
            sharpe=m.get("sharpe"),
            sortino=m.get("sortino"),
            calmar=m.get("calmar"),
            total_trades=m.get("total_trades"),

            win_rate_gross_pct=m.get("win_rate_gross_pct"),
            win_rate_net_pct=m.get("win_rate_net_pct"),
            
            total_tax_paid=m.get("total_tax_paid"),
            avg_hold_days=m.get("avg_hold_days"),

            alpha_vs_spy=m.get("alpha_vs_spy"),
            beta_vs_spy=m.get("beta_vs_spy"),
            excess_cagr_spy=m.get("excess_cagr_spy"),
            alpha_vs_qqq=m.get("alpha_vs_qqq"),
            beta_vs_qqq=m.get("beta_vs_qqq"),
            excess_cagr_qqq=m.get("excess_cagr_qqq"),

            period_start=m.get("period_start"),
            period_end=m.get("period_end"),
        )
        try:
            self.db.insert_run(rec)
        except Exception as e:
            logger.error(f"Failed to insert run {rec.run_uuid}: {e}")