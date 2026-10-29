"""
Industry and sector relative metrics calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_industry_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes industry and sector relative metrics for key financial indicators.

    Metrics computed:
      - Industry medians for key metrics (pe, forward_pe, ps, pb, roe, revenue_growth).
      - Relative ratio columns for valuation multiples:
          pe_vs_industry = pe / industry_median_pe
          forward_pe_vs_industry = forward_pe / industry_median_forward_pe
          ps_vs_industry = ps / industry_median_ps
          pb_vs_industry = pb / industry_median_pb
      - Relative difference columns for rates/percentages:
          roe_vs_industry = roe - industry_median_roe
          revenue_growth_vs_industry = revenue_growth - industry_median_revenue_growth
      - Boolean flags:
          cheaper_than_industry: pe < industry_median_pe and pe > 0
          growth_above_sector: revenue_growth > sector_median_growth

    Parameters
    ----------
    df : pd.DataFrame
        Stock universe DataFrame containing metric columns, and optionally 'industry' and 'sector'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added relative metric columns.
    """
    df = df.copy()

    # Metric tuple: (canonical_name, [candidate_cols_in_df], is_multiple)
    metrics_info: list[tuple[str, list[str], bool]] = [
        ("pe", ["pe"], True),
        ("forward_pe", ["forward_pe", "fwd_pe"], True),
        ("ps", ["ps"], True),
        ("pb", ["pb"], True),
        ("roe", ["roe"], False),
        ("revenue_growth", ["revenue_growth", "sales_growth"], False),
    ]

    has_industry = "industry" in df.columns and df["industry"].notna().any()
    has_sector = "sector" in df.columns and df["sector"].notna().any()

    ind_medians: dict[str, pd.Series] = {}

    for name, candidate_cols, is_multiple in metrics_info:
        matched_col = None
        for col in candidate_cols:
            if col in df.columns:
                matched_col = col
                break

        out_name = f"{name}_vs_industry"

        if matched_col is None or not has_industry:
            df[out_name] = np.nan
            for col in candidate_cols:
                if col != name:
                    df[f"{col}_vs_industry"] = np.nan
            continue

        s = pd.to_numeric(df[matched_col], errors="coerce").astype(float)

        if is_multiple:
            # Positive values only for valuation multiples
            valid_s = s.where(s > 0)
        else:
            valid_s = s

        # Compute industry median per group
        med = valid_s.groupby(df["industry"]).transform("median")
        ind_medians[name] = med

        if is_multiple:
            # Ratio: pe / industry_median_pe
            rel = np.where((s > 0) & (med > 0), s / med, np.nan)
        else:
            # Difference: roe - industry_median_roe
            rel = np.where(s.notna() & med.notna(), s - med, np.nan)

        rel_series = pd.Series(rel, index=df.index, dtype=float)
        df[out_name] = rel_series
        for col in candidate_cols:
            if col != name:
                df[f"{col}_vs_industry"] = rel_series

    # Calculate cheaper_than_industry: pe < industry_median_pe and pe > 0
    if "pe" in df.columns and has_industry and "pe" in ind_medians:
        pe_s = pd.to_numeric(df["pe"], errors="coerce").astype(float)
        med_pe = ind_medians["pe"]
        cheaper = (pe_s > 0) & (med_pe.notna()) & (med_pe > 0) & (pe_s < med_pe)
        df["cheaper_than_industry"] = cheaper.fillna(False).astype(bool)
    else:
        df["cheaper_than_industry"] = False

    # Calculate growth_above_sector: revenue_growth > sector_median_growth
    rev_growth_col = None
    for col in ["revenue_growth", "sales_growth"]:
        if col in df.columns:
            rev_growth_col = col
            break

    if rev_growth_col is not None and has_sector:
        rg_s = pd.to_numeric(df[rev_growth_col], errors="coerce").astype(float)
        sector_med_growth = rg_s.groupby(df["sector"]).transform("median")
        above = (rg_s.notna()) & (sector_med_growth.notna()) & (rg_s > sector_med_growth)
        df["growth_above_sector"] = above.fillna(False).astype(bool)
    else:
        df["growth_above_sector"] = False

    return df
