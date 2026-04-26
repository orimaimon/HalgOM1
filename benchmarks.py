"""
benchmarks.py — השוואת אסטרטגיה מול SPY ו-QQQ

המודול הזה אחראי על:
  1. טעינת `benchmarks.parquet` (SPY + QQQ + VIX עם Adj_Close)
  2. בניית עקומת Buy&Hold פשוטה של SPY/QQQ מאותו הון התחלתי
  3. חישוב מדדים אבסולוטיים (CAGR, Max DD, Sharpe, Volatility)
  4. חישוב מדדים יחסיים מול SPY *ו*גם מול QQQ:
     Alpha, Beta, Correlation, Information Ratio, Tracking Error,
     Up/Down Capture

שימוש כמודול:
    from benchmarks import compare_strategy_to_benchmarks
    results = compare_strategy_to_benchmarks(
        strategy_equity = pd.DataFrame({...}),  # Date + Capital
        benchmarks_path = "DB/benchmarks.parquet",
    )
    # results['metrics_absolute'] — DataFrame להדפסה ישירה לאקסל
    # results['metrics_relative'] — Alpha/Beta טבלה
    # results['equity_curves']    — הגרף (Date × [Strategy, SPY, QQQ])

שימוש כ-CLI:
    python benchmarks.py reports/topn_20260420.xlsx
    python benchmarks.py reports/topn_20260420.xlsx --benchmarks-path DB/benchmarks.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmarks")

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE        = 0.0   # לשמירת פשטות. אפשר להזין 0.02-0.04 בעתיד.


# ══════════════════════════════════════════════════════════════════════════════
# 1. טעינה
# ══════════════════════════════════════════════════════════════════════════════

def load_benchmarks(path: str | Path) -> pd.DataFrame:
    """
    טוען את benchmarks.parquet ומחזיר DataFrame ממוין עם Date.
    עמודות צפויות: SPY_Close, SPY_Adj_Close, QQQ_Close, QQQ_Adj_Close, VIX_Close
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmarks.parquet not found at {p}")

    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    expected = ["SPY_Adj_Close", "QQQ_Adj_Close", "VIX_Close"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning(f"benchmarks.parquet missing columns: {missing}")

    logger.info(f"Loaded benchmarks: {len(df):,} rows | "
                f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. עקומת Buy & Hold
# ══════════════════════════════════════════════════════════════════════════════

def build_buy_hold_curve(benchmarks_df: pd.DataFrame,
                          strategy_dates: pd.Series,
                          initial_capital: float,
                          ticker_prefix: str) -> pd.DataFrame:
    """
    בונה Buy&Hold פשוט: קונים ביום הראשון, מחזיקים עד הסוף.
    משתמש ב-Adj_Close (כולל reinvested dividends).

    Returns: DataFrame עם Date + Capital, מסונכרן לתאריכי האסטרטגיה.
    """
    adj_col = f"{ticker_prefix}_Adj_Close"
    if adj_col not in benchmarks_df.columns:
        raise KeyError(f"Column {adj_col} not found in benchmarks")

    # תחום לתאריכי האסטרטגיה בלבד
    dates_set = set(pd.to_datetime(strategy_dates).dt.normalize())
    bench = benchmarks_df[["Date", adj_col]].copy()
    bench["Date"] = bench["Date"].dt.normalize()
    bench = bench[bench["Date"].isin(dates_set)].reset_index(drop=True)

    if bench.empty:
        raise ValueError(f"No overlap between {ticker_prefix} dates and strategy dates")

    # מילוי NaN פנימיים (חגים וכד')
    bench[adj_col] = bench[adj_col].ffill()
    first_price = bench[adj_col].iloc[0]
    if pd.isna(first_price) or first_price <= 0:
        raise ValueError(f"{ticker_prefix}: invalid first-day price ({first_price})")

    shares = initial_capital / first_price
    bench["Capital"] = (bench[adj_col] * shares).round(2)

    result = bench[["Date", "Capital"]].rename(columns={"Capital": f"{ticker_prefix}_Capital"})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. מטריקות אבסולוטיות
# ══════════════════════════════════════════════════════════════════════════════

def _returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return float(dd.min() * 100)


def _cagr(equity: pd.Series, dates: pd.Series) -> float:
    dates = pd.to_datetime(dates)
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100


def _sharpe(returns: pd.Series, rf_annual: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    return float(excess.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: pd.Series, rf_annual: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return 0.0
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_performance_metrics(equity_df: pd.DataFrame,
                                 name: str = "Strategy",
                                 equity_col: str = "Capital") -> Dict[str, float]:
    """
    מקבל DataFrame עם Date + Capital ומחזיר dict של מדדים.
    """
    equity_df = equity_df.sort_values("Date").reset_index(drop=True)
    equity    = equity_df[equity_col].astype(float)
    dates     = equity_df["Date"]
    returns   = _returns_from_equity(equity)

    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr_pct         = _cagr(equity, dates)
    max_dd_pct       = _max_drawdown_pct(equity)
    vol_annual_pct   = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
    sharpe           = _sharpe(returns)
    sortino          = _sortino(returns)

    # Calmar = CAGR / |MaxDD|
    calmar = cagr_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0.0

    return {
        "Name":            name,
        "Initial_Capital": round(equity.iloc[0], 2),
        "Final_Capital":   round(equity.iloc[-1], 2),
        "Total_Return_%":  round(total_return_pct, 2),
        "CAGR_%":          round(cagr_pct, 2),
        "Max_Drawdown_%":  round(max_dd_pct, 2),
        "Volatility_%":    round(vol_annual_pct, 2),
        "Sharpe":          round(sharpe, 2),
        "Sortino":         round(sortino, 2),
        "Calmar":          round(calmar, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. מטריקות יחסיות (Alpha, Beta, Correlation, Capture Ratios)
# ══════════════════════════════════════════════════════════════════════════════

def compute_relative_metrics(strategy_equity: pd.DataFrame,
                              benchmark_equity: pd.DataFrame,
                              benchmark_name: str = "Benchmark") -> Dict[str, float]:
    """
    מחשב מדדים יחסיים של אסטרטגיה מול בנצ'מרק.
    שני ה-DataFrames חייבים לכלול Date + Capital, ולכסות את אותם תאריכים.
    """
    # סנכרון לפי Date עם inner join
    s = strategy_equity[["Date", "Capital"]].copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    s = s.rename(columns={"Capital": "s_cap"})

    bench_cap_col = [c for c in benchmark_equity.columns if c.endswith("_Capital")][0]
    b = benchmark_equity[["Date", bench_cap_col]].copy()
    b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
    b = b.rename(columns={bench_cap_col: "b_cap"})

    merged = s.merge(b, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(merged) < 30:
        logger.warning(f"Only {len(merged)} overlapping days for relative metrics vs {benchmark_name}")
        return _empty_relative_metrics(benchmark_name)

    s_ret = merged["s_cap"].pct_change().dropna()
    b_ret = merged["b_cap"].pct_change().dropna()
    # שומרים רק תאריכים משותפים
    common = s_ret.index.intersection(b_ret.index)
    s_ret, b_ret = s_ret.loc[common], b_ret.loc[common]

    if len(s_ret) < 30 or b_ret.std() == 0:
        return _empty_relative_metrics(benchmark_name)

    # Beta = Cov(s,b) / Var(b)
    covar = float(np.cov(s_ret, b_ret, ddof=1)[0, 1])
    varb  = float(b_ret.var(ddof=1))
    beta  = covar / varb if varb > 0 else 0.0

    # Alpha (annualized, CAPM) = r_s - (rf + beta * (r_b - rf))
    s_cagr = _cagr(merged["s_cap"], merged["Date"]) / 100   # as decimal
    b_cagr = _cagr(merged["b_cap"], merged["Date"]) / 100
    rf     = RISK_FREE_RATE
    alpha_annual = (s_cagr - rf) - beta * (b_cagr - rf)

    # Correlation
    corr = float(s_ret.corr(b_ret)) if s_ret.std() > 0 and b_ret.std() > 0 else 0.0

    # Tracking Error (annualized) + Information Ratio
    diff_ret = s_ret - b_ret
    tracking_error = float(diff_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    info_ratio = float(diff_ret.mean() / diff_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) \
                 if diff_ret.std() > 0 else 0.0

    # Up/Down Capture
    up_mask   = b_ret > 0
    down_mask = b_ret < 0
    up_capture   = float(s_ret[up_mask].mean()   / b_ret[up_mask].mean()  * 100) if up_mask.sum()   > 0 and b_ret[up_mask].mean()   != 0 else 0.0
    down_capture = float(s_ret[down_mask].mean() / b_ret[down_mask].mean() * 100) if down_mask.sum() > 0 and b_ret[down_mask].mean() != 0 else 0.0

    return {
        "Benchmark":         benchmark_name,
        "Alpha_Annual_%":    round(alpha_annual * 100, 2),
        "Beta":              round(beta, 3),
        "Correlation":       round(corr, 3),
        "Tracking_Error_%":  round(tracking_error * 100, 2),
        "Information_Ratio": round(info_ratio, 2),
        "Up_Capture_%":      round(up_capture, 1),
        "Down_Capture_%":    round(down_capture, 1),
        "Excess_CAGR_%":     round((s_cagr - b_cagr) * 100, 2),
    }


def _empty_relative_metrics(name: str) -> Dict[str, float]:
    return {
        "Benchmark": name, "Alpha_Annual_%": 0.0, "Beta": 0.0,
        "Correlation": 0.0, "Tracking_Error_%": 0.0, "Information_Ratio": 0.0,
        "Up_Capture_%": 0.0, "Down_Capture_%": 0.0, "Excess_CAGR_%": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. API מרכזי — השוואה מלאה מול SPY + QQQ
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategy_to_benchmarks(strategy_equity: pd.DataFrame,
                                    benchmarks_path: str | Path,
                                    strategy_name: str = "Strategy") -> Dict[str, pd.DataFrame]:
    """
    תוצאה מוכנה להכנסה ישירה לאקסל.

    Args:
        strategy_equity: DataFrame עם Date + Capital
        benchmarks_path: נתיב ל-benchmarks.parquet

    Returns:
        dict עם 3 DataFrames:
          "metrics_absolute" — Strategy / SPY / QQQ, שורה לכל אחד
          "metrics_relative" — SPY vs / QQQ vs, שורה לכל אחד
          "equity_curves"    — Date × [Strategy, SPY, QQQ, VIX]
    """
    # הכנה
    s_eq = strategy_equity.copy()
    s_eq["Date"] = pd.to_datetime(s_eq["Date"]).dt.normalize()
    s_eq = s_eq.sort_values("Date").reset_index(drop=True)
    initial_capital = float(s_eq["Capital"].iloc[0])

    # טעינה
    bench = load_benchmarks(benchmarks_path)

    # Buy&Hold curves
    spy_curve = build_buy_hold_curve(bench, s_eq["Date"], initial_capital, "SPY")
    qqq_curve = build_buy_hold_curve(bench, s_eq["Date"], initial_capital, "QQQ")

    # מטריקות אבסולוטיות
    strat_m = compute_performance_metrics(s_eq, name=strategy_name, equity_col="Capital")
    spy_m   = compute_performance_metrics(
        spy_curve.rename(columns={"SPY_Capital": "Capital"}),
        name="SPY Buy&Hold", equity_col="Capital")
    qqq_m   = compute_performance_metrics(
        qqq_curve.rename(columns={"QQQ_Capital": "Capital"}),
        name="QQQ Buy&Hold", equity_col="Capital")

    metrics_absolute = pd.DataFrame([strat_m, spy_m, qqq_m])

    # מטריקות יחסיות
    rel_spy = compute_relative_metrics(s_eq, spy_curve, "SPY")
    rel_qqq = compute_relative_metrics(s_eq, qqq_curve, "QQQ")
    metrics_relative = pd.DataFrame([rel_spy, rel_qqq])

    # גרף equity — תאריכים משותפים בלבד
    s_small = s_eq[["Date", "Capital"]].rename(columns={"Capital": "Strategy"})
    curves = s_small.merge(spy_curve.rename(columns={"SPY_Capital": "SPY_BH"}),
                           on="Date", how="left")
    curves = curves.merge(qqq_curve.rename(columns={"QQQ_Capital": "QQQ_BH"}),
                          on="Date", how="left")

    # הוספת VIX להצגה (ללא normalization)
    vix_col = bench[["Date", "VIX_Close"]].copy()
    vix_col["Date"] = vix_col["Date"].dt.normalize()
    curves = curves.merge(vix_col, on="Date", how="left")

    return {
        "metrics_absolute": metrics_absolute,
        "metrics_relative": metrics_relative,
        "equity_curves":    curves,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. הדפסת סיכום יפה (CLI)
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_summary(results: Dict[str, pd.DataFrame]):
    """הדפסה מסודרת של הטבלאות ל-stdout."""
    ma = results["metrics_absolute"]
    mr = results["metrics_relative"]

    print()
    print("═" * 84)
    print(" Absolute Performance")
    print("═" * 84)
    print(f"{'Name':<20} {'CAGR%':>8} {'MaxDD%':>8} {'Vol%':>8} "
          f"{'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
    print("─" * 84)
    for _, r in ma.iterrows():
        print(f"{r['Name']:<20} {r['CAGR_%']:>8.2f} {r['Max_Drawdown_%']:>8.2f} "
              f"{r['Volatility_%']:>8.2f} {r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} "
              f"{r['Calmar']:>8.2f}")

    print()
    print("═" * 84)
    print(" Relative Performance (Strategy vs Benchmark)")
    print("═" * 84)
    print(f"{'Benchmark':<12} {'Alpha%':>8} {'Beta':>8} {'Corr':>8} "
          f"{'TE%':>8} {'InfoR':>8} {'Up%':>8} {'Down%':>8} {'ExCAGR%':>9}")
    print("─" * 84)
    for _, r in mr.iterrows():
        print(f"{r['Benchmark']:<12} {r['Alpha_Annual_%']:>8.2f} {r['Beta']:>8.2f} "
              f"{r['Correlation']:>8.2f} {r['Tracking_Error_%']:>8.2f} "
              f"{r['Information_Ratio']:>8.2f} {r['Up_Capture_%']:>8.1f} "
              f"{r['Down_Capture_%']:>8.1f} {r['Excess_CAGR_%']:>9.2f}")

    print()
    print("Legend:")
    print("  CAGR        = שיעור צמיחה שנתי מתכלה")
    print("  MaxDD       = נפילה מקסימלית מהשיא")
    print("  Vol         = תנודתיות שנתית")
    print("  Sharpe      = תשואה/תנודתיות (risk-adjusted)")
    print("  Sortino     = כמו Sharpe אבל רק לתנודתיות שלילית")
    print("  Calmar      = CAGR/|MaxDD|")
    print("  Alpha       = תשואה עודפת מעבר למה שצפוי מה-Beta (annualized)")
    print("  Beta        = רגישות לתנועות הבנצ'מרק (1.0 = צמוד, <1 הגנתי, >1 מוגבר)")
    print("  InfoR       = Information Ratio — alpha/tracking error")
    print("  Up/Down %   = אחוז תפיסת תנועות חיוביות/שליליות של הבנצ'מרק")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    p = argparse.ArgumentParser(description="Compare strategy equity to SPY/QQQ benchmarks")
    p.add_argument("equity_file",
                   help="Path to strategy XLSX (with 'Equity' sheet) OR CSV with Date+Capital columns")
    p.add_argument("--benchmarks-path", default="DB/benchmarks.parquet")
    p.add_argument("--sheet-name", default="Equity",
                   help="Sheet name to read from xlsx (default: Equity)")
    p.add_argument("--strategy-name", default=None,
                   help="Name to display in output (default: filename stem)")
    args = p.parse_args()

    eq_path = Path(args.equity_file)
    if not eq_path.exists():
        raise FileNotFoundError(f"Equity file not found: {eq_path}")

    strategy_name = args.strategy_name or eq_path.stem

    # טעינת equity
    if eq_path.suffix.lower() in (".xlsx", ".xls"):
        eq_df = pd.read_excel(eq_path, sheet_name=args.sheet_name)
    elif eq_path.suffix.lower() == ".csv":
        eq_df = pd.read_csv(eq_path, parse_dates=["Date"])
    else:
        raise ValueError(f"Unsupported equity file type: {eq_path.suffix}")

    if "Date" not in eq_df.columns or "Capital" not in eq_df.columns:
        raise ValueError(f"Equity file must have Date + Capital columns, got: {list(eq_df.columns)}")

    logger.info(f"Loaded strategy equity: {len(eq_df):,} rows")

    results = compare_strategy_to_benchmarks(
        strategy_equity = eq_df,
        benchmarks_path = args.benchmarks_path,
        strategy_name   = strategy_name,
    )

    print_comparison_summary(results)


if __name__ == "__main__":
    main()
