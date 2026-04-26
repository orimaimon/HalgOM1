"""
mc_cli.py — Command-line interface for Monte Carlo experiments.

Usage:
  python mc_cli.py run     --strategy topn --n-runs 500 --workers 8
  python mc_cli.py list
  python mc_cli.py top     --batch-id <id> [--n 20]
  python mc_cli.py show    --run-id 42
  python mc_cli.py export  --run-id 42 --output reports/run42.xlsx
  python mc_cli.py dashboard --batch-id <id> [--open]
  python mc_cli.py stats

Sub-commands are independent — you can run a batch, inspect, then re-run with
different params. The SQLite DB is the single source of truth.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import webbrowser
from pathlib import Path

from mc_database import MCDatabase
from mc_engine import MCConfig, MCRunner
from mc_strategies import STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("mc_cli")


# ══════════════════════════════════════════════════════════════════════════════
# sub-commands
# ══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    if args.strategy not in STRATEGIES:
        print(f"ERROR: unknown strategy '{args.strategy}'. Available: {list(STRATEGIES.keys())}")
        sys.exit(1)

    config = MCConfig(
        strategy_name=args.strategy,
        n_runs=args.n_runs,
        n_workers=args.workers,
        seed=args.seed,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        data_path=args.data_path,
        benchmarks_path=args.benchmarks_path,
        db_path=args.db_path,
        start_date=args.start,
        end_date=args.end,
        top_n_detail=args.top_n_detail,
        bottom_n_detail=args.bottom_n_detail,
        notes=args.notes or "",
    )
    runner = MCRunner(config)
    batch_id = runner.run()

    print()
    print("=" * 70)
    print(f" Batch ID: {batch_id}")
    print("=" * 70)
    print(f" View top results:  python mc_cli.py top --batch-id {batch_id}")
    print(f" Build dashboard:   python mc_cli.py dashboard --batch-id {batch_id} --open")
    print("=" * 70)


def cmd_list(args):
    db = MCDatabase(args.db_path)
    df = db.list_batches()
    if df.empty:
        print("No batches yet. Run: python mc_cli.py run --strategy topn --n-runs 200")
        return
    # Pretty print
    cols = ["batch_id", "strategy_name", "n_runs_completed",
            "best_cagr", "worst_cagr", "started_at"]
    cols = [c for c in cols if c in df.columns]
    # Truncate long timestamps
    if "started_at" in df.columns:
        df["started_at"] = df["started_at"].str.slice(0, 19)
    print(df[cols].to_string(index=False))


def cmd_top(args):
    db = MCDatabase(args.db_path)
    df = db.query_runs(batch_id=args.batch_id, limit=args.n, order_by="cagr_pct")
    if df.empty:
        print(f"No runs found for batch {args.batch_id}")
        return

    cols = ["run_id", "cagr_pct", "max_drawdown_pct", "sharpe",
            "total_trades", "win_rate_pct",
            "alpha_vs_spy", "beta_vs_spy", "excess_cagr_spy"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].round(2).to_string(index=False))

    # Also show top params
    print()
    print(f"Top {min(3, len(df))} configurations (params):")
    for i, row in df.head(3).iterrows():
        print(f"  run_id={row['run_id']}  CAGR={row['cagr_pct']:.2f}%  "
              f"MaxDD={row['max_drawdown_pct']:.1f}%")
        print(f"    params: {row['params']}")


def cmd_show(args):
    db = MCDatabase(args.db_path)
    details = db.get_run_details(args.run_id)
    print(f"Run ID: {details['run_id']}")
    print(f"UUID: {details['run_uuid']}")
    print(f"Batch: {details['batch_id']}  | Strategy: {details['strategy_name']}")
    print(f"Status: {details['status']}")
    print()
    print("Parameters:")
    for k, v in details["params"].items():
        print(f"  {k} = {v}")
    print()
    print("Metrics:")
    for k in ["cagr_pct", "total_return_pct", "max_drawdown_pct",
              "sharpe", "sortino", "calmar", "volatility_pct",
              "total_trades", "win_rate_pct", "total_tax_paid",
              "alpha_vs_spy", "beta_vs_spy", "excess_cagr_spy",
              "alpha_vs_qqq", "beta_vs_qqq", "excess_cagr_qqq"]:
        v = details.get(k)
        if v is not None:
            print(f"  {k:22s} = {v}")
    print()
    print(f"Has detail data: {'YES' if details.get('has_detail_data') else 'NO'}")
    if details.get("equity_curve") is not None and not details["equity_curve"].empty:
        eq = details["equity_curve"]
        print(f"  Equity curve: {len(eq)} rows, {eq['Date'].min().date()} → {eq['Date'].max().date()}")
    if details.get("trades") is not None and not details["trades"].empty:
        print(f"  Trades: {len(details['trades'])} rows")


def cmd_export(args):
    db = MCDatabase(args.db_path)
    details = db.get_run_details(args.run_id)

    if not details.get("has_detail_data"):
        print(f"WARN: run_id {args.run_id} does not have equity+trades detail data.")
        print("  (Only Top 25 + Bottom 10 of each batch have details saved by default.)")

    out = Path(args.output) if args.output else Path("reports") / f"mc_run_{args.run_id}.xlsx"
    db.export_run_to_xlsx(args.run_id, out)
    print(f"✓ Exported: {out.absolute()}")


def cmd_dashboard(args):
    try:
        from mc_dashboard import build_mc_dashboard
    except ImportError as e:
        print(f"ERROR: mc_dashboard module not available ({e})")
        print("  Install dependencies: pip install plotly")
        sys.exit(1)

    db = MCDatabase(args.db_path)
    batch = db.get_batch(args.batch_id)
    if not batch:
        print(f"Batch {args.batch_id} not found.")
        sys.exit(1)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.output) if args.output else Path("reports/dashboards") / f"mc_{args.batch_id}_{ts}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    path = build_mc_dashboard(
        db=db,
        batch_id=args.batch_id,
        benchmarks_path=args.benchmarks_path,
        output_path=out,
    )
    print(f"✓ Dashboard: {path.absolute()}")
    if args.open:
        webbrowser.open(f"file://{path.absolute().as_posix()}")


def cmd_stats(args):
    db = MCDatabase(args.db_path)
    stats = db.stats()
    print("Database statistics:")
    for k, v in stats.items():
        print(f"  {k:22s}: {v:,}")


def cmd_export_batch(args):
    """Export an entire batch (summary + all runs + top/bottom detail + regime) to JSON."""
    try:
        from mc_export import export_batch_to_json
    except ImportError as e:
        print(f"ERROR: mc_export module not available ({e})")
        sys.exit(1)

    db = MCDatabase(args.db_path)
    batch = db.get_batch(args.batch_id)
    if not batch:
        print(f"Batch {args.batch_id} not found.")
        sys.exit(1)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.output) if args.output else Path("reports/exports") / f"mc_{args.batch_id}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    path = export_batch_to_json(
        db=db,
        batch_id=args.batch_id,
        output_path=out,
        benchmarks_path=args.benchmarks_path,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        include_detail=not args.no_detail,
        include_regime=not args.no_regime,
    )
    size_kb = path.stat().st_size / 1024
    print(f"✓ Exported: {path.absolute()}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"\n  Load in Python:")
    print(f"    from mc_export import json_to_dataframes")
    print(f"    dfs = json_to_dataframes(r'{path.absolute()}')")
    print(f"    dfs['runs']                # flat table of all runs")
    print(f"    dfs['equity_<run_id>']     # equity curve for a top run")
    print(f"    dfs['trades_<run_id>']     # trades of a top run")


def cmd_delete(args):
    db = MCDatabase(args.db_path)
    batch = db.get_batch(args.batch_id)
    if not batch:
        print(f"Batch {args.batch_id} not found.")
        return

    # Count affected rows
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        n_runs = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE batch_id = ?", (args.batch_id,)
        ).fetchone()[0]

    if not args.yes:
        print(f"Batch {args.batch_id} ({batch['strategy_name']}) contains {n_runs} runs.")
        confirm = input("Delete? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    with sqlite3.connect(db.db_path) as conn:
        # Cascade delete
        conn.execute("""
            DELETE FROM equity_curves WHERE run_id IN (
                SELECT run_id FROM runs WHERE batch_id = ?
            )
        """, (args.batch_id,))
        conn.execute("""
            DELETE FROM trades WHERE run_id IN (
                SELECT run_id FROM runs WHERE batch_id = ?
            )
        """, (args.batch_id,))
        conn.execute("DELETE FROM runs WHERE batch_id = ?", (args.batch_id,))
        conn.execute("DELETE FROM batches WHERE batch_id = ?", (args.batch_id,))
        conn.commit()
    print(f"✓ Deleted batch {args.batch_id} and {n_runs} associated runs.")


# ══════════════════════════════════════════════════════════════════════════════
# argparse
# ══════════════════════════════════════════════════════════════════════════════

def _add_common_args(p):
    p.add_argument("--db-path", default="DB/mc_runs.sqlite")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mc_cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    r = sub.add_parser("run", help="Run a Monte Carlo batch")
    r.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    r.add_argument("--n-runs", type=int, default=200)
    r.add_argument("--workers", type=int, default=0, help="0 = auto (CPU count)")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--initial-capital", type=float, default=100_000.0)
    r.add_argument("--commission", type=float, default=1.0)
    r.add_argument("--slippage", type=float, default=0.001)
    r.add_argument("--data-path", default="DB/simulation_db.parquet")
    r.add_argument("--benchmarks-path", default="DB/benchmarks.parquet")
    r.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    r.add_argument("--end",   default=None, help="YYYY-MM-DD (optional)")
    r.add_argument("--top-n-detail", type=int, default=25,
                   help="Number of top runs (by CAGR) to save detail for")
    r.add_argument("--bottom-n-detail", type=int, default=10,
                   help="Number of bottom runs to save detail for")
    r.add_argument("--notes", default=None, help="Free-form notes about this batch")
    _add_common_args(r)
    r.set_defaults(func=cmd_run)

    # list
    l = sub.add_parser("list", help="List all batches")
    _add_common_args(l)
    l.set_defaults(func=cmd_list)

    # top
    t = sub.add_parser("top", help="Show top runs of a batch")
    t.add_argument("--batch-id", required=True)
    t.add_argument("--n", type=int, default=10)
    _add_common_args(t)
    t.set_defaults(func=cmd_top)

    # show
    s = sub.add_parser("show", help="Show details of one run")
    s.add_argument("--run-id", type=int, required=True)
    _add_common_args(s)
    s.set_defaults(func=cmd_show)

    # export (single run)
    e = sub.add_parser("export", help="Export a single run to XLSX")
    e.add_argument("--run-id", type=int, required=True)
    e.add_argument("--output", default=None)
    _add_common_args(e)
    e.set_defaults(func=cmd_export)

    # export-batch (full batch to JSON)
    eb = sub.add_parser("export-batch",
                         help="Export a full batch to JSON (compact, LLM-friendly)")
    eb.add_argument("--batch-id", required=True)
    eb.add_argument("--output", default=None,
                     help="Default: reports/exports/mc_<batch_id>_<timestamp>.json")
    eb.add_argument("--benchmarks-path", default="DB/benchmarks.parquet")
    eb.add_argument("--top-n", type=int, default=25,
                     help="Include detail (equity+trades) for top N runs")
    eb.add_argument("--bottom-n", type=int, default=10,
                     help="Include detail for bottom M runs")
    eb.add_argument("--no-detail", action="store_true",
                     help="Skip equity_curve + trades (smaller file ~50KB)")
    eb.add_argument("--no-regime", action="store_true",
                     help="Skip regime classification")
    _add_common_args(eb)
    eb.set_defaults(func=cmd_export_batch)

    # dashboard
    d = sub.add_parser("dashboard", help="Build interactive HTML dashboard for a batch")
    d.add_argument("--batch-id", required=True)
    d.add_argument("--output", default=None)
    d.add_argument("--benchmarks-path", default="DB/benchmarks.parquet")
    d.add_argument("--open", action="store_true", help="Open in browser")
    _add_common_args(d)
    d.set_defaults(func=cmd_dashboard)

    # stats
    st = sub.add_parser("stats", help="Show DB statistics")
    _add_common_args(st)
    st.set_defaults(func=cmd_stats)

    # delete
    dl = sub.add_parser("delete", help="Delete a batch (all runs + details)")
    dl.add_argument("--batch-id", required=True)
    dl.add_argument("--yes", action="store_true", help="Skip confirmation")
    _add_common_args(dl)
    dl.set_defaults(func=cmd_delete)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
