import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("Reporting")

def generate_dashboard_json(backtest_results: dict, benchmark_results: dict, strategy_name: str, parameters: dict, output_dir: Path):
    """
    לוקח את תוצאות הבק-טסט והבנצ'מרק, ומייצר קובץ JSON מובנה עבור ה-HTML Dashboard.
    """
    logger.info("Generating Dashboard JSON Data Contract...")
    
    # --- 1. חילוץ KPI (מדדי ביצוע) ---
    abs_metrics = benchmark_results["metrics_absolute"]
    strat_metrics = abs_metrics[abs_metrics["Name"] == strategy_name].iloc[0]
    
    trades_df = backtest_results["Trades"]
    win_rate = 0.0
    if not trades_df.empty:
        wins = (trades_df['Net_PnL'] > 0).sum()
        win_rate = round((wins / len(trades_df)) * 100, 2)

    kpi_data = {
        "strategy_name": strategy_name,
        "cagr": round(strat_metrics["CAGR_%"], 2),
        "max_drawdown": round(strat_metrics["Max_Drawdown_%"], 2),
        "sharpe_ratio": round(strat_metrics["Sharpe"], 2),
        "win_rate": win_rate,
        "total_trades": len(trades_df),
        "parameters": parameters
    }

    # --- 2. חילוץ סדרות זמן למאקרו (Macro Time Series) ---
    curves = benchmark_results["equity_curves"].copy()
    
    # חישוב Drawdown בזמן אמת לאסטרטגיה
    roll_max = curves["Strategy"].cummax()
    strategy_dd = ((curves["Strategy"] - roll_max) / roll_max) * 100

    # המרה בטוחה של נתונים (טיפול ב-NaN שגורמים לקריסת JSON)
    dates_str = curves["Date"].dt.strftime('%Y-%m-%d').tolist()
    
    macro_data = {
        "dates": dates_str,
        "strategy_equity": curves["Strategy"].fillna(0).round(2).tolist(),
        "spy_equity": curves["SPY_BH"].fillna(0).round(2).tolist(),
        "qqq_equity": curves["QQQ_BH"].fillna(0).round(2).tolist(),
        "vix": curves["VIX_Close"].fillna(0).round(2).tolist(),
        "strategy_drawdown": strategy_dd.fillna(0).round(2).tolist()
    }

    # --- 3. חילוץ עסקאות למיקרו (Micro Trades) ---
    if not trades_df.empty:
        # נחשב את אחוז התשואה לכל עסקה
        pnl_pct = ((trades_df["Sell_Price"] / trades_df["Buy_Price"] - 1) * 100).round(2)
        
        trades_data = {
            "tickers": trades_df["Ticker"].tolist(),
            "entry_dates": trades_df["Buy_Date"].astype(str).tolist(),
            "exit_dates": trades_df["Sell_Date"].astype(str).tolist(),
            "hold_days": trades_df["Hold_Days"].tolist(),
            "pnl_pct": pnl_pct.tolist(),
            "exit_reasons": trades_df["Sell_Reason"].tolist()
        }
    else:
        trades_data = {"tickers": [], "entry_dates": [], "exit_dates": [], "hold_days": [], "pnl_pct": [], "exit_reasons": []}

    # --- הרכבת החוזה הסופי ושמירה ---
    dashboard_data = {
        "kpi_data": kpi_data,
        "macro_data": macro_data,
        "trades_data": trades_data
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"dashboard_data_{strategy_name}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"✅ Dashboard JSON saved to: {json_path}")
    return json_path