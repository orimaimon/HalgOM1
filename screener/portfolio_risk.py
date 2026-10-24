"""
Portfolio concentration and risk module.

Calculates:
  - exposure: per open position weight, sector, beta, value.
  - by_sector: sector weights and position counts.
  - summary: key risk metrics (positions, total value, top 5 weight, weighted beta, beta coverage)
    and Hebrew risk warning sentences (position/sector concentration, diversification, missing prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import screener.i18n as i18n
from screener.portfolio import WATCHLIST, PList, positions, valuation

EXPOSURE_COLUMNS = ["ticker", "company", "sector", "value", "weight", "beta"]
BY_SECTOR_COLUMNS = ["sector", "value", "weight", "positions"]


def exposure(plist: PList, universe: pd.DataFrame) -> pd.DataFrame:
    """Per open position (quantity > 0): ticker, company, sector, value, weight, beta.

    - Prices, company, sector and beta come from universe by ticker.
    - Closed positions (quantity <= 0) and watchlists are excluded (watchlist returns empty frame).
    - Unpriced positions are listed in df.attrs['no_price'] and left out of weights.
    - Positions missing from universe default sector to 'Other' and beta to NaN.
    - Sorted by weight descending.
    """
    if plist.kind == WATCHLIST or not plist.rows:
        out = pd.DataFrame(columns=EXPOSURE_COLUMNS)
        out.attrs["no_price"] = []
        return out

    pos = positions(plist.frame())
    if pos.empty:
        out = pd.DataFrame(columns=EXPOSURE_COLUMNS)
        out.attrs["no_price"] = []
        return out

    pos_open = pos[pos["quantity"] > 0].copy()
    if pos_open.empty:
        out = pd.DataFrame(columns=EXPOSURE_COLUMNS)
        out.attrs["no_price"] = []
        return out

    if "ticker" in universe.columns:
        uni = universe.drop_duplicates("ticker").set_index("ticker")
    else:
        uni = universe

    prices = uni["price"] if "price" in uni.columns else pd.Series(dtype=float)
    companies = uni["company"] if "company" in uni.columns else pd.Series(dtype=object)
    sectors = uni["sector"] if "sector" in uni.columns else pd.Series(dtype=object)
    betas = uni["beta"] if "beta" in uni.columns else pd.Series(dtype=float)

    val = valuation(pos_open, prices)

    no_price_mask = val["price"].isna() | val["value"].isna()
    no_price_tickers = val.loc[no_price_mask, "ticker"].tolist()

    priced = val[~no_price_mask].copy()

    if priced.empty:
        out = pd.DataFrame(columns=EXPOSURE_COLUMNS)
        out.attrs["no_price"] = no_price_tickers
        return out

    total_val = priced["value"].sum()
    priced["weight"] = priced["value"] / total_val * 100.0 if total_val > 0 else 0.0

    priced["company"] = priced["ticker"].map(companies).fillna(priced["ticker"])
    priced["sector"] = priced["ticker"].map(sectors).fillna("Other").replace({"": "Other"})
    priced["beta"] = priced["ticker"].map(betas).astype(float)

    priced = priced.sort_values("weight", ascending=False).reset_index(drop=True)
    out = priced[EXPOSURE_COLUMNS].copy()
    out.attrs["no_price"] = no_price_tickers
    return out


def by_sector(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates exposure by sector: sector, value, weight, positions.

    - Raw English sector names retained.
    - Sorted by sector weight descending.
    """
    if exposure_df.empty:
        return pd.DataFrame(columns=BY_SECTOR_COLUMNS)

    aggregated = exposure_df.groupby("sector", as_index=False).agg(
        value=("value", "sum"),
        weight=("weight", "sum"),
        positions=("ticker", "count"),
    )
    return aggregated.sort_values("weight", ascending=False).reset_index(drop=True)[BY_SECTOR_COLUMNS]


def summary(exposure_df: pd.DataFrame, max_position: float = 20.0, max_sector: float = 40.0) -> dict:
    """Portfolio risk summary metrics and concentration warnings.

    Returns dict with keys:
      - positions: number of open priced positions
      - total_value: total dollar value of open priced positions
      - top5_weight: sum of 5 largest position weights
      - beta: value-weighted beta over positions that have a beta (or None)
      - beta_coverage: weight share (%) of positions with a beta (or None)
      - warnings: list of Hebrew warning sentences
    """
    no_price = exposure_df.attrs.get("no_price", [])
    n_positions = len(exposure_df)

    if exposure_df.empty:
        warnings: list[str] = []         # no priced position: "only 0 stocks" would be nonsense
        if no_price:
            warnings.append(f"לחלק מהמניות אין מחיר עדכני: {', '.join(no_price)}")
        return {
            "positions": 0,
            "total_value": 0.0,
            "top5_weight": 0.0,
            "beta": None,
            "beta_coverage": None,
            "warnings": warnings,
        }

    total_value = float(exposure_df["value"].sum())
    top5_weight = float(exposure_df["weight"].head(5).sum())

    valid_beta = exposure_df[exposure_df["beta"].notna()]
    if valid_beta.empty or valid_beta["value"].sum() == 0:
        weighted_beta = None
        beta_coverage = None
    else:
        weighted_beta = float((valid_beta["value"] * valid_beta["beta"]).sum() / valid_beta["value"].sum())
        beta_coverage = float(valid_beta["weight"].sum())

    warnings = []

    # Position warnings
    for _, row in exposure_df.iterrows():
        w = float(row["weight"])
        if w > max_position:
            warnings.append(f"{row['ticker']} היא {round(w)}% מהתיק (מעל {round(max_position)}%)")

    # Sector warnings
    sector_df = by_sector(exposure_df)
    for _, row in sector_df.iterrows():
        w = float(row["weight"])
        if w > max_sector:
            sec_he = i18n.value("sector", row["sector"], i18n.HE)
            warnings.append(f"סקטור {sec_he} הוא {round(w)}% מהתיק (מעל {round(max_sector)}%)")

    # Low diversification warning
    if 0 < n_positions < 5:
        warnings.append("רק מניה אחת בתיק — פיזור נמוך" if n_positions == 1
                        else f"רק {n_positions} מניות בתיק — פיזור נמוך")

    # Missing price warning
    if no_price:
        warnings.append(f"לחלק מהמניות אין מחיר עדכני: {', '.join(no_price)}")

    return {
        "positions": n_positions,
        "total_value": total_value,
        "top5_weight": top5_weight,
        "beta": weighted_beta,
        "beta_coverage": beta_coverage,
        "warnings": warnings,
    }
