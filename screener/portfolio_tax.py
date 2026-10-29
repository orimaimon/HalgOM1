"""Israeli capital gains tax calculation module for closed trades (pure module).

Provides functions for computing Israeli capital gains in ILS and summarizing tax years.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

TAX_DISCLAIMER_HE = "החישוב נועד לאומדן כללי בלבד ואינו מהווה ייעוץ מס או תחליף לדוח רשמי."


def _extract_date(val: Any) -> dt.date | None:
    if val is None or pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.date()
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    if isinstance(val, str):
        try:
            return dt.date.fromisoformat(val.split("T")[0].split(" ")[0])
        except ValueError:
            return None
    return None


def _get_rate(d: Any, rates: dict[str, float] | None, default_rate: float = 3.6) -> float:
    if not rates:
        return default_rate
    d_date = _extract_date(d)
    if d_date is not None:
        if d_date in rates:
            return float(rates[d_date])  # type: ignore[index]
        str_key = d_date.strftime("%Y-%m-%d")
        if str_key in rates:
            return float(rates[str_key])

    if isinstance(d, str) and d in rates:
        return float(rates[d])

    return default_rate


def compute_israeli_capital_gains(
    closed_trades_df: pd.DataFrame,
    usd_ils_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute Israeli capital gains in ILS for closed trades using transaction-date USD/ILS exchange rates.

    Calculates:
      cost_basis_ils = cost_basis * buy_date_usd_ils_rate
      proceeds_ils = proceeds * sell_date_usd_ils_rate
      nominal_gain_ils = proceeds_ils - cost_basis_ils
      estimated_tax_ils = max(0.0, nominal_gain_ils * 0.25)

    Returns enriched DataFrame with ILS columns and estimated tax.
    """
    if closed_trades_df.empty:
        df_out = closed_trades_df.copy()
        for col in ["cost_basis_ils", "proceeds_ils", "nominal_gain_ils", "estimated_tax_ils"]:
            df_out[col] = pd.Series(dtype=float)
        return df_out

    df_out = closed_trades_df.copy()

    cost_basis_ils_list = []
    proceeds_ils_list = []
    nominal_gain_ils_list = []
    estimated_tax_ils_list = []

    for _, row in df_out.iterrows():
        # Determine sell date
        sell_date_val = row.get("date") if pd.notna(row.get("date")) else row.get("sell_date")
        sell_date = _extract_date(sell_date_val)

        # Determine buy date
        buy_date_val = row.get("buy_date")
        buy_date = _extract_date(buy_date_val)
        if buy_date is None and sell_date is not None:
            hold_days = row.get("hold_days")
            if pd.notna(hold_days):
                try:
                    buy_date = sell_date - dt.timedelta(days=int(hold_days))
                except (ValueError, TypeError):
                    buy_date = sell_date
            else:
                buy_date = sell_date

        buy_rate = _get_rate(buy_date, usd_ils_rates)
        sell_rate = _get_rate(sell_date, usd_ils_rates)

        # Extract USD cost basis
        if "cost_basis" in row and pd.notna(row["cost_basis"]):
            cb_usd = float(row["cost_basis"])
        elif "cost_basis_usd" in row and pd.notna(row["cost_basis_usd"]):
            cb_usd = float(row["cost_basis_usd"])
        elif "avg_cost" in row and "quantity" in row and pd.notna(row["avg_cost"]) and pd.notna(row["quantity"]):
            cb_usd = float(row["avg_cost"]) * float(row["quantity"])
        else:
            cb_usd = 0.0

        # Extract USD proceeds
        if "proceeds" in row and pd.notna(row["proceeds"]):
            proc_usd = float(row["proceeds"])
        elif "proceeds_usd" in row and pd.notna(row["proceeds_usd"]):
            proc_usd = float(row["proceeds_usd"])
        elif "sell_price" in row and "quantity" in row and pd.notna(row["sell_price"]) and pd.notna(row["quantity"]):
            fees = float(row["fees"]) if "fees" in row and pd.notna(row["fees"]) else 0.0
            proc_usd = float(row["sell_price"]) * float(row["quantity"]) - fees
        else:
            proc_usd = 0.0

        cb_ils = cb_usd * buy_rate
        proc_ils = proc_usd * sell_rate
        nom_gain_ils = proc_ils - cb_ils
        tax_ils = max(0.0, nom_gain_ils * 0.25)

        cost_basis_ils_list.append(cb_ils)
        proceeds_ils_list.append(proc_ils)
        nominal_gain_ils_list.append(nom_gain_ils)
        estimated_tax_ils_list.append(tax_ils)

    df_out["cost_basis_ils"] = cost_basis_ils_list
    df_out["proceeds_ils"] = proceeds_ils_list
    df_out["nominal_gain_ils"] = nominal_gain_ils_list
    df_out["estimated_tax_ils"] = estimated_tax_ils_list

    return df_out


def summarize_tax_year(tax_df: pd.DataFrame, year: int) -> dict[str, float]:
    """Summarize capital gains tax for a given tax year with loss offsetting.

    Returns dict with keys:
      total_proceeds_ils, total_gain_ils, total_loss_ils, net_gain_ils, estimated_tax_ils
    """
    default_res = {
        "total_proceeds_ils": 0.0,
        "total_gain_ils": 0.0,
        "total_loss_ils": 0.0,
        "net_gain_ils": 0.0,
        "estimated_tax_ils": 0.0,
    }

    if tax_df.empty:
        return default_res

    year_rows = []
    for _, row in tax_df.iterrows():
        s_date_val = row.get("date") if pd.notna(row.get("date")) else row.get("sell_date")
        s_date = _extract_date(s_date_val)
        if s_date is not None and s_date.year == year:
            year_rows.append(row)

    if not year_rows:
        return default_res

    ydf = pd.DataFrame(year_rows)

    total_proceeds_ils = float(ydf["proceeds_ils"].sum()) if "proceeds_ils" in ydf.columns else 0.0

    if "nominal_gain_ils" in ydf.columns:
        gains = ydf[ydf["nominal_gain_ils"] > 0]["nominal_gain_ils"]
        losses = ydf[ydf["nominal_gain_ils"] < 0]["nominal_gain_ils"]

        total_gain_ils = float(gains.sum()) if not gains.empty else 0.0
        total_loss_ils = abs(float(losses.sum())) if not losses.empty else 0.0
        net_gain_ils = total_gain_ils - total_loss_ils
    else:
        total_gain_ils = 0.0
        total_loss_ils = 0.0
        net_gain_ils = 0.0

    estimated_tax_ils = max(0.0, net_gain_ils * 0.25)

    return {
        "total_proceeds_ils": total_proceeds_ils,
        "total_gain_ils": total_gain_ils,
        "total_loss_ils": total_loss_ils,
        "net_gain_ils": net_gain_ils,
        "estimated_tax_ils": estimated_tax_ils,
    }
