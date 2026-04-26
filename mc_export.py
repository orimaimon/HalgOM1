"""
mc_export.py — Export a Monte Carlo batch to a compact JSON file.

Produces a single structured JSON containing everything needed for
external analysis (including feeding to LLMs or loading in pandas):

  - metadata: batch info, strategy name, period, notes
  - summary: aggregated stats across all runs (CAGR, MaxDD, Sharpe, Alpha)
  - all_runs: flat table of all runs with params columns expanded
  - top_runs_detail: full equity curves + trades for top 25 + bottom 10
  - regime_data: SPY/VIX regime classification + per-regime period summary

Size: typically 300-800KB for a 200-run batch (vs 11MB HTML dashboard).
Loads instantly in Python or any JSON viewer.

Usage:
    from mc_export import export_batch_to_json
    export_batch_to_json(db, batch_id="d87e03e8", output_path="batch.json")

CLI:
    python mc_cli.py export-batch --batch-id d87e03e8 --output batch.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mc_database import MCDatabase

logger = logging.getLogger("mc_export")


def _to_python_scalar(val):
    """Convert numpy/pandas scalars to plain Python types for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float):
        if np.isnan(val):
            return None
        return round(val, 4)
    return val


def _clean_dict(d: Dict) -> Dict:
    """Recursively clean a dict for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, list):
            out[k] = [_clean_dict(x) if isinstance(x, dict) else _to_python_scalar(x) for x in v]
        else:
            out[k] = _to_python_scalar(v)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_metadata(batch: Dict, runs_df: pd.DataFrame) -> Dict:
    """Header metadata about the batch."""
    metadata = {
        "batch_id": batch.get("batch_id"),
        "strategy": batch.get("strategy_name"),
        "n_runs_requested": int(batch.get("n_runs_requested", 0)),
        "n_runs_completed": int(batch.get("n_runs_completed", 0)),
        "initial_capital": float(batch.get("initial_capital", 0) or 0),
        "started_at": batch.get("started_at"),
        "completed_at": batch.get("completed_at"),
        "notes": batch.get("notes", "") or "",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Period (take from first run since batch dates may be wrong)
    if not runs_df.empty:
        ps = runs_df["period_start"].dropna().iloc[0] if not runs_df["period_start"].dropna().empty else None
        pe = runs_df["period_end"].dropna().iloc[0] if not runs_df["period_end"].dropna().empty else None
        metadata["period"] = {"start": ps, "end": pe}
    else:
        metadata["period"] = {"start": None, "end": None}

    # Param space
    if batch.get("param_space_json"):
        try:
            metadata["param_space"] = json.loads(batch["param_space_json"])
        except Exception:
            metadata["param_space"] = None

    return metadata


def _build_summary(runs_df: pd.DataFrame) -> Dict:
    """Aggregated statistics across all runs."""
    if runs_df.empty:
        return {}

    def stats(col: str) -> Dict:
        s = runs_df[col].dropna()
        if s.empty:
            return {"best": None, "worst": None, "median": None, "mean": None, "std": None}
        return {
            "best": round(float(s.max()), 2),
            "worst": round(float(s.min()), 2),
            "median": round(float(s.median()), 2),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
        }

    summary = {
        "cagr_pct": stats("cagr_pct"),
        "max_drawdown_pct": stats("max_drawdown_pct"),
        "sharpe": stats("sharpe"),
        "sortino": stats("sortino"),
        "total_trades": stats("total_trades"),
        "win_rate_pct": stats("win_rate_pct"),
        "alpha_vs_spy": stats("alpha_vs_spy"),
        "beta_vs_spy": stats("beta_vs_spy"),
        "excess_cagr_spy": stats("excess_cagr_spy"),
        "alpha_vs_qqq": stats("alpha_vs_qqq"),
        "excess_cagr_qqq": stats("excess_cagr_qqq"),
    }

    # Count helpers
    n = len(runs_df)
    summary["counts"] = {
        "total": n,
        "profitable": int((runs_df["cagr_pct"] > 0).sum()),
        "beat_spy": int((runs_df["excess_cagr_spy"].fillna(0) > 0).sum()),
        "beat_qqq": int((runs_df["excess_cagr_qqq"].fillna(0) > 0).sum()),
        "positive_alpha_spy": int((runs_df["alpha_vs_spy"].fillna(0) > 0).sum()),
        "with_detail_data": int(runs_df.get("has_detail_data", pd.Series([0] * n)).sum()),
    }

    return summary


def _build_all_runs(runs_df: pd.DataFrame) -> List[Dict]:
    """
    Flatten all runs into a list of dicts. Params are expanded as columns
    (wide format) for easier pandas/Excel loading.
    """
    if runs_df.empty:
        return []

    # Collect all param keys across runs to make uniform columns
    all_param_keys: set = set()
    for _, row in runs_df.iterrows():
        params = row.get("params") or {}
        if isinstance(params, dict):
            all_param_keys.update(params.keys())

    param_keys = sorted(all_param_keys)

    rows = []
    for _, row in runs_df.iterrows():
        params = row.get("params") or {}
        rec = {
            "run_id": int(row["run_id"]),
            "run_uuid": row.get("run_uuid"),
            "status": row.get("status"),
            "cagr_pct": _to_python_scalar(row.get("cagr_pct")),
            "total_return_pct": _to_python_scalar(row.get("total_return_pct")),
            "max_drawdown_pct": _to_python_scalar(row.get("max_drawdown_pct")),
            "volatility_pct": _to_python_scalar(row.get("volatility_pct")),
            "sharpe": _to_python_scalar(row.get("sharpe")),
            "sortino": _to_python_scalar(row.get("sortino")),
            "calmar": _to_python_scalar(row.get("calmar")),
            "total_trades": _to_python_scalar(row.get("total_trades")),
            "win_rate_pct": _to_python_scalar(row.get("win_rate_pct")),
            "avg_hold_days": _to_python_scalar(row.get("avg_hold_days")),
            "total_tax_paid": _to_python_scalar(row.get("total_tax_paid")),
            "alpha_vs_spy": _to_python_scalar(row.get("alpha_vs_spy")),
            "beta_vs_spy": _to_python_scalar(row.get("beta_vs_spy")),
            "excess_cagr_spy": _to_python_scalar(row.get("excess_cagr_spy")),
            "alpha_vs_qqq": _to_python_scalar(row.get("alpha_vs_qqq")),
            "beta_vs_qqq": _to_python_scalar(row.get("beta_vs_qqq")),
            "excess_cagr_qqq": _to_python_scalar(row.get("excess_cagr_qqq")),
            "has_detail": bool(row.get("has_detail_data", 0)),
        }
        # Expand params as prefixed columns
        for pk in param_keys:
            rec[f"param_{pk}"] = _to_python_scalar(params.get(pk))
        rows.append(rec)

    return rows


def _build_detail_runs(db: MCDatabase, runs_df: pd.DataFrame,
                         top_n: int = 25, bottom_n: int = 10) -> Dict[str, Dict]:
    """
    For top N and bottom M runs, include full equity curve + trades.
    """
    if runs_df.empty:
        return {}

    detail = {}
    top_ids = runs_df.head(top_n)["run_id"].astype(int).tolist()
    bottom_ids = runs_df.tail(bottom_n)["run_id"].astype(int).tolist()
    target_ids = list(set(top_ids + bottom_ids))

    for run_id in target_ids:
        try:
            d = db.get_run_details(run_id)
        except Exception as e:
            logger.warning(f"Failed to load detail for run {run_id}: {e}")
            continue

        if not d.get("has_detail_data"):
            continue

        eq = d.get("equity_curve")
        tr = d.get("trades")

        # Equity curve → compact list of {date, capital}
        equity_list = []
        if eq is not None and not eq.empty:
            for _, r in eq.iterrows():
                equity_list.append({
                    "date": r["Date"].strftime("%Y-%m-%d") if hasattr(r["Date"], "strftime") else str(r["Date"]),
                    "capital": round(float(r["Capital"]), 2),
                    "cash": round(float(r.get("Cash", 0) or 0), 2),
                    "open_positions": int(r.get("Open_Positions", 0) or 0),
                })

        # Trades → compact list
        trades_list = []
        if tr is not None and not tr.empty:
            for _, r in tr.iterrows():
                trades_list.append({
                    "ticker": str(r.get("Ticker", "")),
                    "buy_date": r["Buy_Date"].strftime("%Y-%m-%d") if pd.notna(r.get("Buy_Date")) else None,
                    "sell_date": r["Sell_Date"].strftime("%Y-%m-%d") if pd.notna(r.get("Sell_Date")) else None,
                    "hold_days": int(r["Hold_Days"]) if pd.notna(r.get("Hold_Days")) else None,
                    "buy_price": round(float(r["Buy_Price"]), 2) if pd.notna(r.get("Buy_Price")) else None,
                    "sell_price": round(float(r["Sell_Price"]), 2) if pd.notna(r.get("Sell_Price")) else None,
                    "shares": round(float(r["Shares"]), 4) if pd.notna(r.get("Shares")) else None,
                    "gross_pnl": round(float(r["Gross_PnL"]), 2) if pd.notna(r.get("Gross_PnL")) else None,
                    "net_pnl": round(float(r["Net_PnL"]), 2) if pd.notna(r.get("Net_PnL")) else None,
                    "buy_reason": str(r.get("Buy_Reason", "") or ""),
                    "sell_reason": str(r.get("Sell_Reason", "") or ""),
                    "sector": str(r.get("Sector", "") or "") if pd.notna(r.get("Sector")) else None,
                })

        detail[str(run_id)] = {
            "meta": {
                "run_id": run_id,
                "cagr_pct": _to_python_scalar(d.get("cagr_pct")),
                "max_drawdown_pct": _to_python_scalar(d.get("max_drawdown_pct")),
                "sharpe": _to_python_scalar(d.get("sharpe")),
                "alpha_vs_spy": _to_python_scalar(d.get("alpha_vs_spy")),
                "total_trades": _to_python_scalar(d.get("total_trades")),
                "win_rate_pct": _to_python_scalar(d.get("win_rate_pct")),
                "rank_group": "top" if run_id in top_ids else "bottom",
                "params": d.get("params", {}),
            },
            "equity_curve": equity_list,
            "trades": trades_list,
        }

    return detail


def _build_regime_data(benchmarks_path: Optional[str],
                        period_start: Optional[str],
                        period_end: Optional[str]) -> Optional[Dict]:
    """
    Add regime classification for the batch's date range.
    Returns None if mc_regime module or benchmarks are unavailable.
    """
    if not benchmarks_path:
        return None
    path = Path(benchmarks_path)
    if not path.exists():
        return None

    try:
        from mc_regime import classify_regimes, extract_regime_periods
    except ImportError:
        logger.info("mc_regime module unavailable — skipping regime data")
        return None

    try:
        bench = pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Failed to load benchmarks: {e}")
        return None

    classified = classify_regimes(bench)
    if classified.empty or "Regime" not in classified.columns:
        return None

    # Filter to period
    if period_start:
        classified = classified[classified["Date"] >= pd.to_datetime(period_start)]
    if period_end:
        classified = classified[classified["Date"] <= pd.to_datetime(period_end)]

    if classified.empty:
        return None

    # Day counts
    counts = classified["Regime"].value_counts().to_dict()
    total = int(classified["Regime"].count())

    # Periods (contiguous runs)
    periods_df = extract_regime_periods(classified, min_days=15)
    periods = []
    for _, p in periods_df.iterrows():
        periods.append({
            "regime": p["regime"],
            "start_date": p["start_date"].strftime("%Y-%m-%d"),
            "end_date": p["end_date"].strftime("%Y-%m-%d"),
            "n_days": int(p["n_days"]),
        })

    return {
        "total_days": total,
        "regime_day_counts": {k: int(v) for k, v in counts.items()},
        "regime_day_pct": {k: round(v / total * 100, 1) for k, v in counts.items()} if total > 0 else {},
        "periods": periods,
        "classification_rules": {
            "Bull-Calm":    "SPY > SMA-200 AND VIX < 20",
            "Bull-Anxious": "SPY > SMA-200 AND VIX >= 20",
            "Correction":   "SPY < SMA-200 AND VIX < 25",
            "Crisis":       "SPY < SMA-200 AND VIX >= 25",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def export_batch_to_json(db: MCDatabase, batch_id: str,
                          output_path: str | Path,
                          benchmarks_path: Optional[str] = None,
                          top_n: int = 25,
                          bottom_n: int = 10,
                          include_detail: bool = True,
                          include_regime: bool = True) -> Path:
    """
    Export everything about a batch to a single JSON file.

    Args:
        db:               MCDatabase instance
        batch_id:         Which batch to export
        output_path:      Target JSON file
        benchmarks_path:  Optional — enables regime classification
        top_n:            Include detail for top N runs
        bottom_n:         Include detail for bottom M runs
        include_detail:   Whether to include equity_curves + trades
        include_regime:   Whether to include regime data

    Returns:
        Path to the generated JSON
    """
    batch = db.get_batch(batch_id)
    if batch is None:
        raise ValueError(f"Batch {batch_id} not found")

    runs_df = db.query_runs(batch_id=batch_id, order_by="cagr_pct", descending=True)
    logger.info(f"Exporting batch {batch_id}: {len(runs_df)} runs")

    # Build each section
    metadata = _build_metadata(batch, runs_df)
    summary = _build_summary(runs_df)
    all_runs = _build_all_runs(runs_df)

    detail = {}
    if include_detail and not runs_df.empty:
        logger.info(f"Loading detail for top {top_n} + bottom {bottom_n} runs...")
        detail = _build_detail_runs(db, runs_df, top_n=top_n, bottom_n=bottom_n)
        logger.info(f"  → {len(detail)} runs have equity/trades detail")

    regime = None
    if include_regime and benchmarks_path:
        logger.info("Classifying regimes...")
        period = metadata.get("period", {})
        regime = _build_regime_data(benchmarks_path,
                                      period.get("start"), period.get("end"))
        if regime:
            logger.info(f"  → {len(regime.get('periods', []))} regime periods identified")

    # Assemble final output
    output = {
        "metadata": metadata,
        "summary": summary,
        "all_runs": all_runs,
    }
    if detail:
        output["top_runs_detail"] = detail
    if regime:
        output["regime_data"] = regime

    # Clean NaN / numpy types
    output = _clean_dict(output)

    # Write
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Exported: {output_path} ({size_kb:.0f} KB)")

    # Also log what's inside, for user feedback
    n_trades = sum(len(d.get("trades", [])) for d in detail.values())
    n_equity = sum(len(d.get("equity_curve", [])) for d in detail.values())
    logger.info(f"  Contents: {len(all_runs)} runs summarized, "
                f"{len(detail)} runs with detail ({n_trades} trades, "
                f"{n_equity} equity points)")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: quick loader for Python analysis
# ═══════════════════════════════════════════════════════════════════════════

def load_batch_from_json(path: str | Path) -> Dict:
    """Load a JSON export back into a dict. Convenience for analysis scripts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_dataframes(path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load a JSON export and convert sections to pandas DataFrames.
    Returns dict with keys: 'runs', 'regime_periods' (if present),
    and 'equity_<run_id>' and 'trades_<run_id>' per detail run.

    This is the most convenient entry point for notebook analysis.
    """
    data = load_batch_from_json(path)
    result = {}

    # All runs (flat table)
    if "all_runs" in data:
        result["runs"] = pd.DataFrame(data["all_runs"])

    # Regime periods
    if "regime_data" in data and "periods" in data["regime_data"]:
        result["regime_periods"] = pd.DataFrame(data["regime_data"]["periods"])

    # Detail data
    if "top_runs_detail" in data:
        for run_id, d in data["top_runs_detail"].items():
            if d.get("equity_curve"):
                eq = pd.DataFrame(d["equity_curve"])
                eq["date"] = pd.to_datetime(eq["date"])
                result[f"equity_{run_id}"] = eq
            if d.get("trades"):
                tr = pd.DataFrame(d["trades"])
                if "buy_date" in tr.columns:
                    tr["buy_date"] = pd.to_datetime(tr["buy_date"])
                    tr["sell_date"] = pd.to_datetime(tr["sell_date"])
                result[f"trades_{run_id}"] = tr

    return result
