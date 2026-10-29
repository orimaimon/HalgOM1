"""
Dividend income projection module for stock holdings.

Units:
  div_yield is expressed as a percentage number (e.g., 0.75 represents 0.75%, NOT 0.0075 or 75%).
  10,000 portfolio value with a 0.75 div_yield yields 75.0 annual dividend income.
"""

from typing import Any
import numpy as np
import pandas as pd

COLUMNS = [
    "ticker",
    "value",
    "div_yield",
    "annual_income",
    "monthly_income",
    "yield_on_cost_unknown",
]


def projected_income(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate yearly and monthly dividend income from portfolio holdings.

    Note on units:
      `div_yield` is a percentage number where 0.75 represents 0.75% (NOT 75% or 0.0075).
      For example, value = 10,000 USD and div_yield = 0.75 results in annual_income = 75 USD.

    Parameters
    ----------
    holdings : pd.DataFrame
        DataFrame with required columns: 'ticker', 'shares', 'price', 'div_yield'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'ticker', 'value', 'div_yield', 'annual_income',
        'monthly_income', 'yield_on_cost_unknown'.
    """
    if holdings.empty or not all(col in holdings.columns for col in ["ticker", "shares", "price", "div_yield"]):
        return pd.DataFrame(columns=COLUMNS)

    df = holdings[["ticker", "shares", "price", "div_yield"]].copy()

    # Drop rows with NaN or non-positive shares or price
    valid_shares = pd.to_numeric(df["shares"], errors="coerce") > 0
    valid_price = pd.to_numeric(df["price"], errors="coerce") > 0
    df = df[valid_shares & valid_price].copy()

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    df["shares"] = pd.to_numeric(df["shares"])
    df["price"] = pd.to_numeric(df["price"])
    df["value"] = df["shares"] * df["price"]

    # NaN or negative div_yield counts as 0 income, but the row stays
    yield_num = pd.to_numeric(df["div_yield"], errors="coerce").fillna(0.0)
    effective_yield = np.where(yield_num < 0, 0.0, yield_num)

    df["annual_income"] = df["value"] * effective_yield / 100.0
    df["monthly_income"] = df["annual_income"] / 12.0
    df["yield_on_cost_unknown"] = True

    return df[COLUMNS].reset_index(drop=True)


def income_summary(income_df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize dividend income metrics from projected income DataFrame.

    Parameters
    ----------
    income_df : pd.DataFrame
        DataFrame produced by `projected_income`.

    Returns
    -------
    dict
        Dictionary containing:
        - 'total_value': total portfolio value (float)
        - 'annual_income': total annual dividend income (float)
        - 'monthly_income': total monthly dividend income (float)
        - 'portfolio_yield': portfolio dividend yield % (float or None if total_value == 0)
        - 'n_payers': number of holdings paying dividends (annual_income > 0)
        - 'top_payers': list of up to 5 (ticker, annual_income) tuples sorted descending
    """
    if income_df.empty or "annual_income" not in income_df.columns:
        return {
            "total_value": 0.0,
            "annual_income": 0.0,
            "monthly_income": 0.0,
            "portfolio_yield": None,
            "n_payers": 0,
            "top_payers": [],
        }

    total_value = float(income_df["value"].sum()) if "value" in income_df.columns else 0.0
    annual_income = float(income_df["annual_income"].sum())
    monthly_income = float(income_df["monthly_income"].sum()) if "monthly_income" in income_df.columns else annual_income / 12.0

    portfolio_yield = (annual_income / total_value * 100.0) if total_value > 0 else None

    payers_df = income_df[income_df["annual_income"] > 0]
    n_payers = int(len(payers_df))

    # Top payers (up to 5 sorted descending by annual_income)
    top_df = payers_df.sort_values(by="annual_income", ascending=False).head(5)
    top_payers = [(str(row["ticker"]), float(row["annual_income"])) for _, row in top_df.iterrows()]

    return {
        "total_value": total_value,
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "portfolio_yield": portfolio_yield,
        "n_payers": n_payers,
        "top_payers": top_payers,
    }
