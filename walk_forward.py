"""
walk_forward.py — Walk-forward validation for MC strategies.

The core question this answers:
    "When I optimize parameters on past data, do they keep working on
     unseen future data, or am I just curve-fitting?"

Workflow:
    1. TRAIN phase — run N MC samples with random params on the train window
       (e.g. 2006-2016). Uses existing MCRunner.
    2. SELECT phase — pick top-K configs from train ranked by
       *excess_cagr_qqq* (not cagr_pct — alpha is what we're after).
    3. TEST phase — run those EXACT frozen configs on the test window
       (e.g. 2016-2026). Deterministic, no resampling.
    4. REPORT — for each config, compare train vs test metrics.
       Aggregate: median degradation, % maintaining alpha, etc.

If train CAGR is 22% and test CAGR is 8%, you have curve-fitting.
If train is 22% and test is 19%, you have something real.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid as uuidlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mc_database import MCDatabase, RunRecord
from mc_engine import MCConfig, MCRunner, _worker_execute
from mc_strategies import STRATEGIES, get_strategy_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("walk_forward")


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class WFConfig:
    """Walk-forward configuration."""
    strategy_name: str
    n_runs: int = 200
    n_workers: int = 0
    seed: Optional[int] = 42  # deterministic by default — important for WF

    # Selection
    top_k: int = 10
    selection_metric: str = "excess_cagr_qqq"  # what we optimize FOR

    # Date windows (must be non-overlapping; test must be AFTER train)
    train_start: str = "2006-04-21"
    train_end:   str = "2016-04-20"
    test_start:  str = "2016-04-21"
    test_end:    str = "2026-04-20"

    # Backtest mechanics
    initial_capital: float = 100_000.0
    commission: float = 1.0
    slippage: float = 0.001
    data_path: str = "DB/simulation_db.parquet"
    benchmarks_path: str = "DB/benchmarks.parquet"
    db_path: str = "DB/mc_runs.sqlite"

    worker_timeout: int = 600

    def validate(self) -> None:
        if self.strategy_name not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy_name}'. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        ts, te = pd.to_datetime(self.train_start), pd.to_datetime(self.train_end)
        vs, ve = pd.to_datetime(self.test_start),  pd.to_datetime(self.test_end)
        if ts >= te:
            raise ValueError(f"train_start ({ts.date()}) must precede train_end ({te.date()})")
        if vs >= ve:
            raise ValueError(f"test_start ({vs.date()}) must precede test_end ({ve.date()})")
        if vs < te:
            raise ValueError(
                f"Test window must start AFTER train ends. "
                f"Got train_end={te.date()}, test_start={vs.date()}. "
                f"Overlap = data leakage."
            )
        valid_metrics = {"cagr_pct", "excess_cagr_qqq", "excess_cagr_spy",
                         "sharpe", "sortino", "calmar", "alpha_vs_qqq", "alpha_vs_spy"}
        if self.selection_metric not in valid_metrics:
            raise ValueError(
                f"selection_metric must be one of {sorted(valid_metrics)}, "
                f"got '{self.selection_metric}'"
            )


# ════════════════════════════════════════════════════════════════════════════
# Walk-forward runner
# ════════════════════════════════════════════════════════════════════════════

class WalkForwardRunner:
    def __init__(self, config: WFConfig):
        config.validate()
        self.cfg = config
        self.db = MCDatabase(config.db_path)
        self.train_batch_id: Optional[str] = None
        self.test_batch_id: Optional[str] = None

    def _run_train_phase(self) -> str:
        cfg = self.cfg
        train_notes = (
            f"WF-TRAIN | window={cfg.train_start}→{cfg.train_end} | "
            f"selection_metric={cfg.selection_metric}"
        )
        mc_config = MCConfig(
            strategy_name=cfg.strategy_name,
            n_runs=cfg.n_runs,
            n_workers=cfg.n_workers,
            seed=cfg.seed,
            initial_capital=cfg.initial_capital,
            commission=cfg.commission,
            slippage=cfg.slippage,
            data_path=cfg.data_path,
            benchmarks_path=cfg.benchmarks_path,
            db_path=cfg.db_path,
            start_date=cfg.train_start,
            end_date=cfg.train_end,
            top_n_detail=cfg.top_k,
            bottom_n_detail=0,
            worker_timeout=cfg.worker_timeout,
            notes=train_notes,
        )
        logger.info(
            "═" * 70 + "\n"
            f"  PHASE 1 — TRAIN  ({cfg.train_start} → {cfg.train_end})\n"
            f"  {cfg.n_runs} random samples, strategy={cfg.strategy_name}\n"
            + "═" * 70
        )
        runner = MCRunner(mc_config)
        return runner.run()

    def _select_top_k(self, train_batch_id: str) -> pd.DataFrame:
        cfg = self.cfg
        descending = cfg.selection_metric != "max_drawdown_pct"
        selected = self.db.query_runs(
            batch_id=train_batch_id,
            order_by=cfg.selection_metric,
            descending=descending,
            limit=cfg.top_k,
        )
        
        selected = selected[selected[cfg.selection_metric].notna()].reset_index(drop=True)
        if selected.empty:
            raise RuntimeError(
                f"No train runs have a valid '{cfg.selection_metric}'. "
                f"Check that benchmarks.parquet exists and has QQQ_Adj_Close."
            )
        logger.info(
            f"Selected top {len(selected)} configs from train by {cfg.selection_metric}:"
        )
        for _, r in selected.iterrows():
            logger.info(
                f"  run_id={r['run_id']:>4} | "
                f"{cfg.selection_metric}={r[cfg.selection_metric]:>+6.2f} | "
                f"cagr={r.get('cagr_pct', np.nan):>+6.2f}% | "
                f"maxDD={r.get('max_drawdown_pct', np.nan):>6.2f}% | "
                f"params={r['params']}"
            )
        return selected

    def _run_test_phase(self, selected: pd.DataFrame) -> str:
        cfg = self.cfg
        strat_cfg = get_strategy_config(cfg.strategy_name)

        test_batch_id = self.db.register_batch(
            strategy_name=strat_cfg["display_name"] + " (WF-TEST)",
            n_runs=len(selected),
            param_space=strat_cfg["param_space"].to_dict(),
            initial_capital=cfg.initial_capital,
            data_period=(cfg.test_start, cfg.test_end),
            notes=(
                f"WF-TEST of train_batch={self.train_batch_id} | "
                f"window={cfg.test_start}→{cfg.test_end} | "
                f"selection_metric={cfg.selection_metric}"
            ),
        )

        logger.info(
            "═" * 70 + "\n"
            f"  PHASE 3 — TEST  ({cfg.test_start} → {cfg.test_end})\n"
            f"  Re-running {len(selected)} frozen configs (deterministic)\n"
            + "═" * 70
        )

        tasks: List[Dict[str, Any]] = []
        for _, row in selected.iterrows():
            tasks.append({
                "run_uuid": str(uuidlib.uuid4()),
                "_train_run_id": int(row["run_id"]),
                "strategy_name": cfg.strategy_name,
                "params": row["params"],
                "data_path": cfg.data_path,
                "cols_needed": strat_cfg["cols_needed"],
                "initial_capital": cfg.initial_capital,
                "commission": cfg.commission,
                "slippage": cfg.slippage,
                "benchmarks_path": cfg.benchmarks_path,
                "return_details": True,   
                "date_filter": (cfg.test_start, cfg.test_end),
            })

        workers = cfg.n_workers if cfg.n_workers > 0 else None
        results: List[Tuple[int, Dict[str, Any]]] = []
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_task = {}
            for task in tasks:
                worker_task = {k: v for k, v in task.items() if not k.startswith("_")}
                fut = ex.submit(_worker_execute, worker_task)
                fut_to_task[fut] = task

            for i, fut in enumerate(as_completed(fut_to_task), 1):
                task = fut_to_task[fut]
                try:
                    result = fut.result(timeout=cfg.worker_timeout)
                except Exception as e:
                    logger.error(f"Test worker died for train_run_id={task['_train_run_id']}: {e}")
                    continue
                results.append((task["_train_run_id"], result))
                logger.info(
                    f"[test] {i}/{len(tasks)} done | "
                    f"train_run_id={task['_train_run_id']} | "
                    f"status={result['status']} | "
                    f"cagr={result.get('metrics', {}).get('cagr_pct')}"
                )

        elapsed = time.time() - t0
        logger.info(f"Test phase completed in {elapsed:.1f}s")

        for train_run_id, result in results:
            self._persist_test_result(test_batch_id, train_run_id, result)
        self.db.finalize_batch(test_batch_id, len([r for _, r in results if r["status"] == "completed"]))
        return test_batch_id

    def _persist_test_result(self, test_batch_id: str, train_run_id: int,
                              result: Dict[str, Any]) -> None:
        m = result.get("metrics", {})
        
        # 🟢 התיקון הקריטי בוצע כאן: התאמת שמות המשתנים של ה-Win Rate למה שחוזר מהמנוע 🟢
        rec = RunRecord(
            run_uuid=result["run_uuid"],
            batch_id=test_batch_id,
            strategy_name=self.cfg.strategy_name,
            status=result["status"],
            started_at=result["started_at"],
            completed_at=result["completed_at"],
            error_message=result.get("error_message"),
            params=result["params"],

            initial_capital=m.get("initial_capital", self.cfg.initial_capital),
            final_capital=m.get("final_capital"),
            total_return_pct=m.get("total_return_pct"),
            cagr_pct=m.get("cagr_pct"),
            max_drawdown_pct=m.get("max_drawdown_pct"),
            volatility_pct=m.get("volatility_pct"),
            sharpe=m.get("sharpe"),
            sortino=m.get("sortino"),
            calmar=m.get("calmar"),
            total_trades=m.get("total_trades"),
            
            # השינוי (פיצול לברוטו/נטו בהתאם לקוד המנוע)
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
            run_id = self.db.insert_run(rec)
            if result["status"] == "completed" and result.get("equity_curve") is not None:
                self.db.insert_detail_data(
                    run_id=run_id,
                    equity_df=result["equity_curve"],
                    trades_df=result.get("trades"),
                    reason=f"wf_test_of_train_run_{train_run_id}",
                )
        except Exception as e:
            logger.error(f"Failed to persist test result for train_run_id={train_run_id}: {e}")

    def _build_report(self, train_selected: pd.DataFrame,
                       test_batch_id: str) -> pd.DataFrame:
        test_runs = self.db.query_runs(batch_id=test_batch_id)

        train_lookup = {
            json.dumps(r["params"], sort_keys=True): r
            for _, r in train_selected.iterrows()
        }

        def _diff(a, b):
            if a is None or b is None:
                return np.nan
            return a - b

        rows = []
        for _, t in test_runs.iterrows():
            key = json.dumps(t["params"], sort_keys=True)
            tr = train_lookup.get(key)
            if tr is None:
                continue
            rows.append({
                "params": t["params"],
                "train_run_id": int(tr["run_id"]),
                "test_run_id": int(t["run_id"]),
                "train_cagr": tr.get("cagr_pct"),
                "test_cagr":  t.get("cagr_pct"),
                "cagr_degradation": _diff(t.get("cagr_pct"), tr.get("cagr_pct")),
                "train_excess_qqq": tr.get("excess_cagr_qqq"),
                "test_excess_qqq":  t.get("excess_cagr_qqq"),
                "excess_qqq_degradation": _diff(t.get("excess_cagr_qqq"), tr.get("excess_cagr_qqq")),
                "train_maxdd": tr.get("max_drawdown_pct"),
                "test_maxdd":  t.get("max_drawdown_pct"),
                "train_sharpe": tr.get("sharpe"),
                "test_sharpe":  t.get("sharpe"),
            })
        return pd.DataFrame(rows).sort_values("test_excess_qqq", ascending=False).reset_index(drop=True)

    def _print_report(self, report: pd.DataFrame) -> None:
        if report.empty:
            print("\n[!] No matched train/test rows. Something went wrong.")
            return

        n = len(report)
        median_train_cagr = report["train_cagr"].median()
        median_test_cagr = report["test_cagr"].median()
        median_train_qqq = report["train_excess_qqq"].median()
        median_test_qqq = report["test_excess_qqq"].median()
        kept_alpha = (report["test_excess_qqq"] > 0).sum()
        kept_2pct = (report["test_excess_qqq"] >= 2).sum()

        print("\n" + "═" * 75)
        print("  WALK-FORWARD REPORT")
        print("═" * 75)
        print(f"  Strategy           : {self.cfg.strategy_name}")
        print(f"  Train window       : {self.cfg.train_start} → {self.cfg.train_end}")
        print(f"  Test window        : {self.cfg.test_start} → {self.cfg.test_end}")
        print(f"  Selection metric   : {self.cfg.selection_metric}")
        print(f"  Top-K configs      : {n}")
        print()
        print(f"  Median CAGR train→test         : "
              f"{median_train_cagr:+6.2f}%  →  {median_test_cagr:+6.2f}%  "
              f"(Δ {median_test_cagr - median_train_cagr:+6.2f} pp)")
        print(f"  Median excess vs QQQ train→test: "
              f"{median_train_qqq:+6.2f}%  →  {median_test_qqq:+6.2f}%  "
              f"(Δ {median_test_qqq - median_train_qqq:+6.2f} pp)")
        print()
        print(f"  Configs with test alpha > 0     : {kept_alpha}/{n}  ({100*kept_alpha/n:.0f}%)")
        print(f"  Configs with test alpha ≥ +2pp  : {kept_2pct}/{n}  ({100*kept_2pct/n:.0f}%)")
        print()

        print("  Per-config breakdown:")
        print("  " + "─" * 73)
        cols = ["test_cagr", "test_excess_qqq", "excess_qqq_degradation",
                "test_maxdd", "test_sharpe"]
        sub = report[["train_run_id", "test_run_id"] + cols].copy()
        for c in cols:
            sub[c] = sub[c].apply(lambda v: f"{v:+6.2f}" if pd.notna(v) else "  n/a ")
        print(sub.to_string(index=False))
        print()

        print("  VERDICT:")
        if median_test_qqq >= 2 and kept_alpha >= 0.7 * n:
            print("  ✓ Strategy holds up out-of-sample. Real alpha plausible.")
        elif median_test_qqq >= 0:
            print("  ~ Marginal. Test alpha is positive but small. Be cautious about claims.")
        else:
            print("  ✗ Strategy did NOT hold up out-of-sample.")
            print("    The train-period 'best configs' are likely curve-fit. ")
            print("    Median test alpha vs QQQ is negative — you'd be better off in QQQ.")
        print("═" * 75)

    def _save_report(self, report: pd.DataFrame, out_dir: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"wf_report_{self.cfg.strategy_name}_{ts}.csv"
        report.to_csv(path, index=False)
        sidecar = path.with_suffix(".meta.json")
        with open(sidecar, "w") as f:
            json.dump({
                "config": asdict(self.cfg),
                "train_batch_id": self.train_batch_id,
                "test_batch_id": self.test_batch_id,
                "n_configs_compared": int(len(report)),
                "generated_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Report saved: {path}")
        logger.info(f"Metadata:    {sidecar}")
        return path

    def run(self, report_dir: Path = Path("reports/walk_forward")) -> Dict[str, Any]:
        t0 = time.time()
        self.train_batch_id = self._run_train_phase()
        train_selected = self._select_top_k(self.train_batch_id)
        self.test_batch_id = self._run_test_phase(train_selected)
        report = self._build_report(train_selected, self.test_batch_id)
        self._print_report(report)
        report_path = self._save_report(report, report_dir)
        elapsed = time.time() - t0
        logger.info(f"Walk-forward complete in {elapsed/60:.1f} min")
        return {
            "train_batch_id": self.train_batch_id,
            "test_batch_id": self.test_batch_id,
            "report_path": str(report_path),
            "report": report,
        }

def main():
    p = argparse.ArgumentParser(
        description="Walk-forward validation for MC strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    p.add_argument("--n-runs", type=int, default=200,
                    help="Number of random samples in TRAIN phase (default: 200)")
    p.add_argument("--workers", type=int, default=0, help="0 = auto")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for train sampling. Fixed seed = reproducible WF.")

    p.add_argument("--top-k", type=int, default=10,
                    help="How many train winners to test out-of-sample")
    p.add_argument("--selection-metric", default="excess_cagr_qqq",
                    help="What to rank train winners by (default: excess_cagr_qqq — alpha)")

    p.add_argument("--train-start", default="2006-04-21")
    p.add_argument("--train-end",   default="2016-04-20")
    p.add_argument("--test-start",  default="2016-04-21")
    p.add_argument("--test-end",    default="2026-04-20")

    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--commission",      type=float, default=1.0)
    p.add_argument("--slippage",        type=float, default=0.001)

    p.add_argument("--data-path",       default="DB/simulation_db.parquet")
    p.add_argument("--benchmarks-path", default="DB/benchmarks.parquet")
    p.add_argument("--db-path",         default="DB/mc_runs.sqlite")
    p.add_argument("--report-dir",      default="reports/walk_forward")

    args = p.parse_args()

    cfg = WFConfig(
        strategy_name=args.strategy,
        n_runs=args.n_runs,
        n_workers=args.workers,
        seed=args.seed,
        top_k=args.top_k,
        selection_metric=args.selection_metric,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start,   test_end=args.test_end,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        data_path=args.data_path,
        benchmarks_path=args.benchmarks_path,
        db_path=args.db_path,
    )

    runner = WalkForwardRunner(cfg)
    out = runner.run(report_dir=Path(args.report_dir))

    print("\nNext steps:")
    print(f"  Build dashboard for test results:")
    print(f"    python mc_cli.py dashboard --batch-id {out['test_batch_id']} --open")
    print(f"  Compare train run details:")
    print(f"    python mc_cli.py top --batch-id {out['train_batch_id']}")


if __name__ == "__main__":
    sys.exit(main() or 0)