"""
wf_fixed.py — Multi-fold OOS validation for pre-optimized fixed configs.

Tests 4 frozen parameter sets (cluster center ±top_n from focused sweep batch 8dabac8e)
across 5 OOS time windows, each representing a different market regime.

Goal: confirm alpha is structural, not a curve-fit artifact.

Usage:
    python wf_fixed.py
    python wf_fixed.py --workers 4
    python wf_fixed.py --db-path DB/mc_runs.sqlite
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid as uuidlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pandas as pd

from mc_database import MCDatabase, RunRecord
from mc_engine import _worker_execute
from mc_strategies import STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("wf_fixed")

# ── Fixed configs ─────────────────────────────────────────────────────────────
# Cluster center ±top_n from focused sweep batch 8dabac8e (2026-05-17).
# Common denominator: sizing=equal + regime=True + hold~130d + min_price=2.0
FIXED_CONFIGS: List[Dict[str, Any]] = [
    {"name": "A-top3", "top_n": 3, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True},
    {"name": "B-top4", "top_n": 4, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True},
    {"name": "C-top5", "top_n": 5, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True},
    {"name": "D-top6", "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True},
]

# ── Extended folds — crisis + recent ──────────────────────────────────────────
EXTENDED_FOLDS: List[Dict[str, str]] = [
    {"label": "2007 (pre-crisis bull)",    "start": "2007-01-01", "end": "2007-12-31"},
    {"label": "2008 (financial crisis)",   "start": "2008-01-01", "end": "2008-12-31"},
    {"label": "2009 (V-recovery)",         "start": "2009-01-01", "end": "2009-12-31"},
    {"label": "2023 (post-bear rally)",    "start": "2023-01-01", "end": "2023-12-31"},
    {"label": "2024 (AI bull)",            "start": "2024-01-01", "end": "2024-12-31"},
    {"label": "2025 (tariff volatility)",  "start": "2025-01-01", "end": "2025-12-31"},
]

# Configs: production winner vs old baseline for direct comparison
EXTENDED_CONFIGS: List[Dict[str, Any]] = [
    {"name": "G-top6-buf1",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.01, "regime_confirm_days": 1},
    {"name": "D-top6-old",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.0, "regime_confirm_days": 1},
]

# ── V2 configs — buf=0.01 sweet spot from sweep 6470b7fd ─────────────────────
# Testing whether the in-sample winner (buf=0.01, conf=1) holds OOS.
# E=top3 (aggressive), F=top4 (balanced), G=top6 (previous OOS winner + new buf)
WF_V2_CONFIGS: List[Dict[str, Any]] = [
    {"name": "E-top3-buf1",
     "top_n": 3, "hold_days": 120, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.01, "regime_confirm_days": 1},
    {"name": "F-top4-buf1",
     "top_n": 4, "hold_days": 90, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.01, "regime_confirm_days": 1},
    {"name": "G-top6-buf1",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.01, "regime_confirm_days": 1},
    # Baseline: previous OOS winner (D-top6) with old binary filter for comparison
    {"name": "D-top6-old",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.0, "regime_confirm_days": 1},
]

# ── Regime-improvement comparison configs ─────────────────────────────────────
# D-top6 was the OOS winner. Compare old regime vs improved (buffer + confirm).
REGIME_COMPARE_CONFIGS: List[Dict[str, Any]] = [
    # Baseline — old binary filter
    {"name": "D-old-regime",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.0, "regime_confirm_days": 1},
    # 2% buffer only
    {"name": "D-buf2pct",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.02, "regime_confirm_days": 1},
    # 3% buffer only
    {"name": "D-buf3pct",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.03, "regime_confirm_days": 1},
    # 2% buffer + 3 confirmation days
    {"name": "D-buf2+conf3",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.02, "regime_confirm_days": 3},
    # 2% buffer + 5 confirmation days
    {"name": "D-buf2+conf5",
     "top_n": 6, "hold_days": 130, "min_price": 2.0,
     "sizing_method": "equal", "use_regime_filter": True,
     "regime_entry_buffer": 0.02, "regime_confirm_days": 5},
]

# ── 5 OOS folds — chosen to cover distinct market regimes ─────────────────────
FOLDS: List[Dict[str, str]] = [
    {"label": "2018 (rate-hike bear)",    "start": "2018-01-01", "end": "2018-12-31"},
    {"label": "2019 (late bull)",          "start": "2019-01-01", "end": "2019-12-31"},
    {"label": "2020 (COVID crash+rally)", "start": "2020-01-01", "end": "2020-12-31"},
    {"label": "2021 (meme/bull peak)",    "start": "2021-01-01", "end": "2021-12-31"},
    {"label": "2022 (inflation bear)",    "start": "2022-01-01", "end": "2022-12-31"},
]

STRATEGY_NAME   = "topn_focused"
DATA_PATH       = "DB/simulation_db.parquet"
BENCHMARKS_PATH = "DB/benchmarks.parquet"
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.001
WORKER_TIMEOUT  = 600


# ── Task builder ──────────────────────────────────────────────────────────────

def _build_task(cfg: Dict[str, Any], fold: Dict[str, str],
                cols_needed: List[str]) -> Dict[str, Any]:
    params = {k: v for k, v in cfg.items() if k != "name"}
    return {
        # Private metadata (stripped before sending to worker)
        "_config_name": cfg["name"],
        "_fold_label":  fold["label"],
        # Worker payload
        "run_uuid":        str(uuidlib.uuid4()),
        "strategy_name":   STRATEGY_NAME,
        "params":          params,
        "data_path":       DATA_PATH,
        "cols_needed":     cols_needed,
        "initial_capital": INITIAL_CAPITAL,
        "commission":      COMMISSION,
        "slippage":        SLIPPAGE,
        "benchmarks_path": BENCHMARKS_PATH,
        "return_details":  True,
        "date_filter":     (fold["start"], fold["end"]),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def _persist(db: MCDatabase, batch_id: str, result: Dict[str, Any],
             config_name: str, fold_label: str) -> None:
    m = result.get("metrics", {})
    rec = RunRecord(
        run_uuid=result["run_uuid"],
        batch_id=batch_id,
        strategy_name=STRATEGY_NAME,
        status=result["status"],
        started_at=result["started_at"],
        completed_at=result["completed_at"],
        error_message=result.get("error_message"),
        params=result["params"],
        initial_capital=m.get("initial_capital", INITIAL_CAPITAL),
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
        run_id = db.insert_run(rec)
        if result["status"] == "completed" and result.get("equity_curve") is not None:
            db.insert_detail_data(
                run_id=run_id,
                equity_df=result["equity_curve"],
                trades_df=result.get("trades"),
                reason=f"wf_fixed|{config_name}|{fold_label}",
            )
    except Exception as e:
        logger.error(f"Persist failed [{config_name}/{fold_label}]: {e}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(n_workers: int = 0, db_path: str = "DB/mc_runs.sqlite") -> Tuple[pd.DataFrame, str]:
    db = MCDatabase(db_path)
    strat_cfg = STRATEGIES[STRATEGY_NAME]
    cols_needed = strat_cfg["cols_needed"]

    batch_id = db.register_batch(
        strategy_name="topn_focused (WF-FIXED 4configs×5folds)",
        n_runs=len(FIXED_CONFIGS) * len(FOLDS),
        param_space=strat_cfg["param_space"].to_dict(),
        initial_capital=INITIAL_CAPITAL,
        data_period=("2018-01-01", "2022-12-31"),
        notes=(
            "Fixed-config walk-forward: configs A(top3)–D(top6), "
            "hold=130d, price=2.0, regime=True, sizing=equal. "
            "5 OOS folds 2018-2022. Source: sweep batch 8dabac8e."
        ),
    )
    logger.info(f"Batch: {batch_id}")

    tasks = [
        _build_task(cfg, fold, cols_needed)
        for cfg in FIXED_CONFIGS
        for fold in FOLDS
    ]
    total = len(tasks)
    workers = n_workers if n_workers > 0 else None

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_task: Dict[Any, Dict[str, Any]] = {}
        for task in tasks:
            worker_task = {k: v for k, v in task.items() if not k.startswith("_")}
            fut = ex.submit(_worker_execute, worker_task)
            fut_to_task[fut] = task

        for i, fut in enumerate(as_completed(fut_to_task), 1):
            task = fut_to_task[fut]
            cfg_name  = task["_config_name"]
            fold_label = task["_fold_label"]
            try:
                result = fut.result(timeout=WORKER_TIMEOUT)
            except Exception as e:
                logger.error(f"Worker crashed [{cfg_name}/{fold_label}]: {e}")
                continue

            m = result.get("metrics", {})
            logger.info(
                f"[{i:>2}/{total}] {cfg_name} | {fold_label:<28} | "
                f"cagr={m.get('cagr_pct', float('nan')):+6.1f}% | "
                f"sharpe={m.get('sharpe', float('nan')):.2f} | "
                f"excess_qqq={m.get('excess_cagr_qqq', float('nan')):+.1f}"
            )
            _persist(db, batch_id, result, cfg_name, fold_label)
            rows.append({
                "config":       cfg_name,
                "fold":         fold_label,
                "cagr_pct":     m.get("cagr_pct"),
                "sharpe":       m.get("sharpe"),
                "max_dd":       m.get("max_drawdown_pct"),
                "excess_qqq":   m.get("excess_cagr_qqq"),
                "excess_spy":   m.get("excess_cagr_spy"),
                "total_trades": m.get("total_trades"),
            })

    elapsed = time.time() - t0
    n_ok = sum(1 for r in rows if r["cagr_pct"] is not None)
    db.finalize_batch(batch_id, n_ok)
    logger.info(f"Done in {elapsed:.1f}s ({n_ok}/{total} succeeded)")

    return pd.DataFrame(rows), batch_id


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    W = 80
    print("\n" + "=" * W)
    print("  WALK-FORWARD FIXED -- STABILITY MATRIX")
    print("  4 configs x 5 OOS folds (different market regimes)")
    print("=" * W)

    def _pivot(col: str, fmt: str = "{:+.1f}") -> str:
        p = df.pivot_table(index="fold", columns="config", values=col, aggfunc="first")
        p = p[[c for c in ["A-top3", "B-top4", "C-top5", "D-top6"] if c in p.columns]]
        lines = []
        lines.append("  " + "  ".join(f"{c:>10}" for c in p.columns))
        for fold_lbl, row in p.iterrows():
            vals = "  ".join(
                fmt.format(v) if v is not None and not (isinstance(v, float) and v != v) else "     n/a"
                for v in row
            )
            lines.append(f"  {fold_lbl:<32} {vals}")
        return "\n".join(lines)

    print("\n  CAGR %\n")
    print(_pivot("cagr_pct", "{:+6.1f}"))

    print("\n\n  Sharpe\n")
    print(_pivot("sharpe", "{:+6.2f}"))

    print("\n\n  Excess CAGR vs QQQ (pp)\n")
    print(_pivot("excess_qqq", "{:+6.1f}"))

    # Summary per config
    print("\n\n  SUMMARY — median across 5 folds\n")
    summary = (
        df.groupby("config")
        .agg(
            med_cagr    =("cagr_pct",   "median"),
            med_sharpe  =("sharpe",     "median"),
            med_exqqq   =("excess_qqq", "median"),
            med_maxdd   =("max_dd",     "median"),
            pos_alpha   =("excess_qqq", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    summary["pos_alpha"] = summary["pos_alpha"].astype(str) + "/5"
    print("  " + summary.to_string(index=False))

    # Verdict
    print("\n  VERDICT:")
    best = summary.sort_values("med_exqqq", ascending=False).iloc[0]
    pos = best["pos_alpha"]
    if best["med_exqqq"] >= 2 and pos in ("4/5", "5/5"):
        print(f"  ✓ Config {best['config']} is the most stable across regimes.")
        print(f"    med_CAGR={best['med_cagr']:.1f}%, "
              f"med_excess_QQQ={best['med_exqqq']:.1f}pp, "
              f"pos_alpha={pos} folds.")
        print("    → Candidate for production.")
    elif best["med_exqqq"] >= 0:
        print(f"  ~ Config {best['config']} shows marginal OOS alpha.")
        print("    Positive but small. Be cautious before deploying.")
    else:
        print("  ✗ No config maintained positive alpha across most folds.")
        print("    Strategy is likely regime-specific or curve-fit.")
    print("=" * W)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_extended(n_workers: int = 0, db_path: str = "DB/mc_runs.sqlite") -> Tuple[pd.DataFrame, str]:
    """Run G-top6-buf1 vs D-top6-old on 6 extended folds (2007-2009, 2023-2025)."""
    db = MCDatabase(db_path)
    strat_cfg = STRATEGIES[STRATEGY_NAME]
    cols_needed = strat_cfg["cols_needed"]

    n_total = len(EXTENDED_CONFIGS) * len(EXTENDED_FOLDS)
    batch_id = db.register_batch(
        strategy_name="topn_focused_v2 (WF-EXTENDED 2007-09+2023-25)",
        n_runs=n_total,
        param_space=strat_cfg["param_space"].to_dict(),
        initial_capital=INITIAL_CAPITAL,
        data_period=("2007-01-01", "2025-12-31"),
        notes=(
            "Extended WF: G-top6-buf1 (production config) vs D-top6-old baseline. "
            "6 folds: 2007 pre-crisis, 2008 crisis, 2009 recovery, "
            "2023 post-bear, 2024 AI bull, 2025 tariff vol."
        ),
    )
    logger.info(f"Extended WF batch: {batch_id}")

    tasks = [
        _build_task(cfg, fold, cols_needed)
        for cfg in EXTENDED_CONFIGS
        for fold in EXTENDED_FOLDS
    ]
    total = len(tasks)
    workers = n_workers if n_workers > 0 else None

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_task: Dict[Any, Dict[str, Any]] = {}
        for task in tasks:
            worker_task = {k: v for k, v in task.items() if not k.startswith("_")}
            fut = ex.submit(_worker_execute, worker_task)
            fut_to_task[fut] = task

        for i, fut in enumerate(as_completed(fut_to_task), 1):
            task = fut_to_task[fut]
            cfg_name   = task["_config_name"]
            fold_label = task["_fold_label"]
            try:
                result = fut.result(timeout=WORKER_TIMEOUT)
            except Exception as e:
                logger.error(f"Worker crashed [{cfg_name}/{fold_label}]: {e}")
                continue

            m = result.get("metrics", {})
            _fmt = lambda v, f: format(v, f) if v is not None else "n/a"
            logger.info(
                f"[{i:>2}/{total}] {cfg_name:<16} | {fold_label:<28} | "
                f"cagr={_fmt(m.get('cagr_pct'), '+7.1f')}% | "
                f"sharpe={_fmt(m.get('sharpe'), '+.2f')} | "
                f"excess_qqq={_fmt(m.get('excess_cagr_qqq'), '+.1f')}"
            )
            _persist(db, batch_id, result, cfg_name, fold_label)
            rows.append({
                "config":     cfg_name,
                "fold":       fold_label,
                "cagr_pct":   m.get("cagr_pct"),
                "sharpe":     m.get("sharpe"),
                "max_dd":     m.get("max_drawdown_pct"),
                "excess_qqq": m.get("excess_cagr_qqq"),
                "excess_spy": m.get("excess_cagr_spy"),
                "n_trades":   m.get("total_trades"),
            })

    elapsed = time.time() - t0
    n_ok = sum(1 for r in rows if r["cagr_pct"] is not None)
    db.finalize_batch(batch_id, n_ok)
    logger.info(f"Done in {elapsed:.1f}s ({n_ok}/{total} succeeded)")
    return pd.DataFrame(rows), batch_id


def print_extended_report(df: pd.DataFrame) -> None:
    W = 80
    cfg_order = [c["name"] for c in EXTENDED_CONFIGS]
    fold_order = [f["label"] for f in EXTENDED_FOLDS]

    print("\n" + "=" * W)
    print("  WALK-FORWARD EXTENDED — 2007-2009 crisis + 2023-2025 recent")
    print("  G-top6-buf1 (production) vs D-top6-old (baseline)")
    print("=" * W)

    def _pivot(col: str, fmt: str) -> None:
        p = df.pivot_table(index="fold", columns="config", values=col, aggfunc="first")
        p = p[[c for c in cfg_order if c in p.columns]]
        p = p.reindex([f for f in fold_order if f in p.index])
        print("  " + "".join(f"{c:>20}" for c in p.columns))
        for fold_lbl, row in p.iterrows():
            vals = "".join(
                fmt.format(v) if v is not None and v == v else "                 n/a"
                for v in row
            )
            print(f"  {fold_lbl:<32}{vals}")

    print("\n  CAGR %\n")
    _pivot("cagr_pct", "{:>+19.1f}%")

    print("\n  Max Drawdown %\n")
    _pivot("max_dd", "{:>+19.1f}%")

    print("\n  Excess CAGR vs QQQ (pp)\n")
    _pivot("excess_qqq", "{:>+19.1f} ")

    # Per-config summary
    print("\n  SUMMARY (median across 6 extended folds)\n")
    summary = (
        df.groupby("config")
        .agg(
            med_cagr  =("cagr_pct",   "median"),
            med_sharpe=("sharpe",     "median"),
            med_maxdd =("max_dd",     "median"),
            med_exqqq =("excess_qqq", "median"),
            pos_alpha =("excess_qqq", lambda x: int((x > 0).sum())),
        )
        .reindex(cfg_order)
        .reset_index()
    )
    summary["pos_alpha"] = summary["pos_alpha"].astype(str) + "/6"
    print("  " + summary.to_string(index=False))

    # Per-fold delta (G minus D)
    print("\n  DELTA per fold: G-top6-buf1 minus D-top6-old (excess_qqq pp)\n")
    g = df[df.config == "G-top6-buf1"].set_index("fold")["excess_qqq"]
    d = df[df.config == "D-top6-old"].set_index("fold")["excess_qqq"]
    delta = (g - d).reindex(fold_order)
    for fold_lbl, val in delta.items():
        bar = "+" * int(abs(val) / 2) if val > 0 else "-" * int(abs(val) / 2)
        sign = "+" if val > 0 else ""
        print(f"  {fold_lbl:<34} {sign}{val:5.1f}pp  {bar}")

    print("=" * W)


def run_wf_v2(n_workers: int = 0, db_path: str = "DB/mc_runs.sqlite") -> Tuple[pd.DataFrame, str]:
    """WF validation for E/F/G + D baseline — buf=0.01 sweet spot vs old filter."""
    db = MCDatabase(db_path)
    strat_cfg = STRATEGIES[STRATEGY_NAME]
    cols_needed = strat_cfg["cols_needed"]

    n_total = len(WF_V2_CONFIGS) * len(FOLDS)
    batch_id = db.register_batch(
        strategy_name="topn_focused_v2 (WF-V2 E/F/G vs D-baseline)",
        n_runs=n_total,
        param_space=strat_cfg["param_space"].to_dict(),
        initial_capital=INITIAL_CAPITAL,
        data_period=("2018-01-01", "2022-12-31"),
        notes=(
            "WF V2: configs E(top3), F(top4), G(top6) all with buf=0.01 conf=1, "
            "vs D-top6 old binary filter. 5 OOS folds 2018-2022. "
            "Source: v2 sweep 6470b7fd (buf=0.01 is in-sample winner)."
        ),
    )
    logger.info(f"WF V2 batch: {batch_id}")

    tasks = [
        _build_task(cfg, fold, cols_needed)
        for cfg in WF_V2_CONFIGS
        for fold in FOLDS
    ]
    total = len(tasks)
    workers = n_workers if n_workers > 0 else None

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_task: Dict[Any, Dict[str, Any]] = {}
        for task in tasks:
            worker_task = {k: v for k, v in task.items() if not k.startswith("_")}
            fut = ex.submit(_worker_execute, worker_task)
            fut_to_task[fut] = task

        for i, fut in enumerate(as_completed(fut_to_task), 1):
            task = fut_to_task[fut]
            cfg_name   = task["_config_name"]
            fold_label = task["_fold_label"]
            try:
                result = fut.result(timeout=WORKER_TIMEOUT)
            except Exception as e:
                logger.error(f"Worker crashed [{cfg_name}/{fold_label}]: {e}")
                continue

            m = result.get("metrics", {})
            logger.info(
                f"[{i:>2}/{total}] {cfg_name:<18} | {fold_label:<28} | "
                f"cagr={m.get('cagr_pct', float('nan')):+6.1f}% | "
                f"sharpe={m.get('sharpe', float('nan')):.2f} | "
                f"excess_qqq={m.get('excess_cagr_qqq', float('nan')):+.1f}"
            )
            _persist(db, batch_id, result, cfg_name, fold_label)
            rows.append({
                "config":     cfg_name,
                "fold":       fold_label,
                "cagr_pct":   m.get("cagr_pct"),
                "sharpe":     m.get("sharpe"),
                "max_dd":     m.get("max_drawdown_pct"),
                "excess_qqq": m.get("excess_cagr_qqq"),
                "excess_spy": m.get("excess_cagr_spy"),
                "n_trades":   m.get("total_trades"),
            })

    elapsed = time.time() - t0
    n_ok = sum(1 for r in rows if r["cagr_pct"] is not None)
    db.finalize_batch(batch_id, n_ok)
    logger.info(f"Done in {elapsed:.1f}s ({n_ok}/{total} succeeded)")
    return pd.DataFrame(rows), batch_id


def print_wf_v2_report(df: pd.DataFrame) -> None:
    W = 80
    cfg_order = [c["name"] for c in WF_V2_CONFIGS]

    print("\n" + "=" * W)
    print("  WALK-FORWARD V2 — buf=0.01 sweet spot vs old filter")
    print("  E(top3) / F(top4) / G(top6-new) / D(top6-old)  x  5 OOS folds")
    print("=" * W)

    def _pivot(col: str, fmt: str) -> None:
        p = df.pivot_table(index="fold", columns="config", values=col, aggfunc="first")
        p = p[[c for c in cfg_order if c in p.columns]]
        print("  " + "".join(f"{c:>18}" for c in p.columns))
        for fold_lbl, row in p.iterrows():
            vals = "".join(
                fmt.format(v) if v is not None and v == v else "               n/a"
                for v in row
            )
            print(f"  {fold_lbl:<32}{vals}")

    print("\n  CAGR %\n")
    _pivot("cagr_pct", "{:>+17.1f}%")

    print("\n  Max Drawdown %\n")
    _pivot("max_dd", "{:>+17.1f}%")

    print("\n  Excess CAGR vs QQQ (pp)\n")
    _pivot("excess_qqq", "{:>+17.1f} ")

    print("\n  SUMMARY (median across 5 folds)\n")
    summary = (
        df.groupby("config")
        .agg(
            med_cagr  =("cagr_pct",   "median"),
            med_sharpe=("sharpe",     "median"),
            med_maxdd =("max_dd",     "median"),
            med_exqqq =("excess_qqq", "median"),
            pos_alpha =("excess_qqq", lambda x: int((x > 0).sum())),
        )
        .reindex(cfg_order)
        .reset_index()
    )
    summary["pos_alpha"] = summary["pos_alpha"].astype(str) + "/5"
    print("  " + summary.to_string(index=False))

    best = summary.sort_values("med_exqqq", ascending=False).iloc[0]
    print(f"\n  Winner: {best['config']}  |  "
          f"med_CAGR={best['med_cagr']:.1f}%  "
          f"med_excess_QQQ={best['med_exqqq']:.1f}pp  "
          f"med_MaxDD={best['med_maxdd']:.1f}%  "
          f"pos_alpha={best['pos_alpha']}")
    print("=" * W)


def run_regime_compare(n_workers: int = 0, db_path: str = "DB/mc_runs.sqlite") -> Tuple[pd.DataFrame, str]:
    """Compare old vs improved regime filters on D-top6 across 5 OOS folds."""
    db = MCDatabase(db_path)
    strat_cfg = STRATEGIES[STRATEGY_NAME]
    cols_needed = strat_cfg["cols_needed"]

    n_total = len(REGIME_COMPARE_CONFIGS) * len(FOLDS)
    batch_id = db.register_batch(
        strategy_name="topn_focused (WF-REGIME-COMPARE)",
        n_runs=n_total,
        param_space=strat_cfg["param_space"].to_dict(),
        initial_capital=INITIAL_CAPITAL,
        data_period=("2018-01-01", "2022-12-31"),
        notes=(
            "Regime filter comparison: D-top6 (top_n=6, hold=130d, price=2.0). "
            "5 variants: old binary vs buffer 2-3% vs buffer+confirm 3/5d. "
            "5 OOS folds 2018-2022."
        ),
    )
    logger.info(f"Regime compare batch: {batch_id}")

    tasks = [
        _build_task(cfg, fold, cols_needed)
        for cfg in REGIME_COMPARE_CONFIGS
        for fold in FOLDS
    ]
    total = len(tasks)
    workers = n_workers if n_workers > 0 else None

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_task: Dict[Any, Dict[str, Any]] = {}
        for task in tasks:
            worker_task = {k: v for k, v in task.items() if not k.startswith("_")}
            fut = ex.submit(_worker_execute, worker_task)
            fut_to_task[fut] = task

        for i, fut in enumerate(as_completed(fut_to_task), 1):
            task = fut_to_task[fut]
            cfg_name   = task["_config_name"]
            fold_label = task["_fold_label"]
            try:
                result = fut.result(timeout=WORKER_TIMEOUT)
            except Exception as e:
                logger.error(f"Worker crashed [{cfg_name}/{fold_label}]: {e}")
                continue

            m = result.get("metrics", {})
            logger.info(
                f"[{i:>2}/{total}] {cfg_name:<18} | {fold_label:<28} | "
                f"cagr={m.get('cagr_pct', float('nan')):+6.1f}% | "
                f"maxdd={m.get('max_drawdown_pct', float('nan')):+6.1f}%"
            )
            _persist(db, batch_id, result, cfg_name, fold_label)
            rows.append({
                "config":     cfg_name,
                "fold":       fold_label,
                "cagr_pct":   m.get("cagr_pct"),
                "sharpe":     m.get("sharpe"),
                "max_dd":     m.get("max_drawdown_pct"),
                "excess_qqq": m.get("excess_cagr_qqq"),
                "excess_spy": m.get("excess_cagr_spy"),
                "n_trades":   m.get("total_trades"),
            })

    elapsed = time.time() - t0
    n_ok = sum(1 for r in rows if r["cagr_pct"] is not None)
    db.finalize_batch(batch_id, n_ok)
    logger.info(f"Done in {elapsed:.1f}s ({n_ok}/{total} succeeded)")
    return pd.DataFrame(rows), batch_id


def print_regime_compare_report(df: pd.DataFrame) -> None:
    W = 80
    print("\n" + "=" * W)
    print("  REGIME FILTER COMPARISON -- D-top6 x 5 OOS folds")
    print("=" * W)

    config_order = [c["name"] for c in REGIME_COMPARE_CONFIGS]

    def _pivot(col: str, fmt: str) -> None:
        p = df.pivot_table(index="fold", columns="config", values=col, aggfunc="first")
        p = p[[c for c in config_order if c in p.columns]]
        header = "  " + "".join(f"{c:>16}" for c in p.columns)
        print(header)
        for fold_lbl, row in p.iterrows():
            vals = "".join(
                fmt.format(v) if v is not None and v == v else "             n/a"
                for v in row
            )
            print(f"  {fold_lbl:<34}{vals}")

    print("\n  CAGR %\n")
    _pivot("cagr_pct", "{:>+15.1f}%")

    print("\n  Max Drawdown %\n")
    _pivot("max_dd", "{:>+15.1f}%")

    print("\n  Excess CAGR vs QQQ (pp)\n")
    _pivot("excess_qqq", "{:>+15.1f} ")

    print("\n  Trades count\n")
    _pivot("n_trades", "{:>15.0f} ")

    # Summary per config
    print("\n  SUMMARY (median across 5 folds)\n")
    summary = (
        df.groupby("config")
        .agg(
            med_cagr  =("cagr_pct",   "median"),
            med_sharpe=("sharpe",     "median"),
            med_maxdd =("max_dd",     "median"),
            med_exqqq =("excess_qqq", "median"),
            pos_alpha =("excess_qqq", lambda x: int((x > 0).sum())),
            med_trades=("n_trades",   "median"),
        )
        .reindex(config_order)
        .reset_index()
    )
    summary["pos_alpha"] = summary["pos_alpha"].astype(str) + "/5"
    print("  " + summary.to_string(index=False))

    best = summary.sort_values("med_exqqq", ascending=False).iloc[0]
    print(f"\n  Best regime variant: {best['config']}")
    print(f"  med_CAGR={best['med_cagr']:.1f}%  med_excess_QQQ={best['med_exqqq']:.1f}pp  "
          f"med_MaxDD={best['med_maxdd']:.1f}%  pos_alpha={best['pos_alpha']}")
    print("=" * W)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Walk-forward validation — fixed configs A-D, 5 OOS folds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["fixed", "regime", "v2", "extended"], default="fixed",
                   help="fixed: A-D | regime: filter variants | v2: E/F/G | extended: 2007-09+2023-25")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto based on CPU count)")
    p.add_argument("--db-path", default="DB/mc_runs.sqlite",
                   help="Path to SQLite database")
    args = p.parse_args()

    if args.mode == "extended":
        df, batch_id = run_extended(n_workers=args.workers, db_path=args.db_path)
        print_extended_report(df)
    elif args.mode == "regime":
        df, batch_id = run_regime_compare(n_workers=args.workers, db_path=args.db_path)
        print_regime_compare_report(df)
    elif args.mode == "v2":
        df, batch_id = run_wf_v2(n_workers=args.workers, db_path=args.db_path)
        print_wf_v2_report(df)
    else:
        df, batch_id = run_all(n_workers=args.workers, db_path=args.db_path)
        print_report(df)

    print(f"\n  Batch ID : {batch_id}")
    print(f"  Dashboard: python mc_cli.py dashboard --batch-id {batch_id} --open")
    print(f"  Top runs : python mc_cli.py top --batch-id {batch_id}")


if __name__ == "__main__":
    main()
