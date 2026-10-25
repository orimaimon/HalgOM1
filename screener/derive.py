"""
Derived columns, computed from the canonical raw columns — independent of the data provider.
Recomputed on every load so date-relative columns (days to earnings) stay correct.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

import duckdb

from screener.fields import COLUMNS
from screener.scores import add_scores
from screener.signals import summarize


def _build_where_clause(where: str | list[str] | dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Convert where specification into a SQL WHERE clause and positional parameter list."""
    if not where:
        return "", []
    if isinstance(where, str):
        w = where.strip()
        return (f" WHERE {w}", []) if w else ("", [])
    if isinstance(where, (list, tuple)):
        parts = [p.strip() for p in where if isinstance(p, str) and p.strip()]
        return (f" WHERE {' AND '.join(parts)}", []) if parts else ("", [])
    if isinstance(where, dict):
        clauses = []
        params = []
        for k, v in where.items():
            col_name = k
            if k.startswith("min_"):
                col_name = k[4:]
                clauses.append(f'"{col_name}" >= ?')
                params.append(v)
            elif k.startswith("max_"):
                col_name = k[4:]
                clauses.append(f'"{col_name}" <= ?')
                params.append(v)
            elif isinstance(v, (tuple, list)):
                if len(v) == 2 and isinstance(v[0], str) and v[0].upper() in (">", ">=", "<", "<=", "=", "!=", "LIKE", "ILIKE"):
                    clauses.append(f'"{col_name}" {v[0]} ?')
                    params.append(v[1])
                elif len(v) == 2 and isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)):
                    clauses.append(f'"{col_name}" BETWEEN ? AND ?')
                    params.extend([v[0], v[1]])
                elif v and all(isinstance(x, (str, int, float)) for x in v):
                    placeholders = ", ".join(["?"] * len(v))
                    clauses.append(f'"{col_name}" IN ({placeholders})')
                    params.extend(v)
            elif v is None:
                clauses.append(f'"{col_name}" IS NULL')
            else:
                clauses.append(f'"{col_name}" = ?')
                params.append(v)
        return (f" WHERE {' AND '.join(clauses)}", params) if clauses else ("", [])
    raise ValueError(f"Unsupported where clause type: {type(where)}")


def load_filtered_universe(
    path: str | Path,
    where: str | list[str] | dict[str, Any] | None = None,
    columns: list[str] | None = None,
    add_derived_cols: bool = False,
    today: dt.date | None = None,
) -> pd.DataFrame:
    """Load stock universe from Parquet using DuckDB predicate pushdown.

    Parameters
    ----------
    path : str or Path
        Path to the Parquet file.
    where : str, list of str, dict, or None
        SQL WHERE predicate(s) or dictionary of column constraints for filtering.
    columns : list of str, optional
        Specific columns to select. If None, selects all columns (*).
    add_derived_cols : bool, default False
        If True, runs add_derived() on the loaded filtered DataFrame.
    today : dt.date, optional
        Reference date passed to add_derived() if add_derived_cols is True.

    Returns
    -------
    pd.DataFrame
        Filtered stock universe DataFrame with 100% numerical parity with Pandas loader.
    """
    path_str = str(Path(path).resolve())
    cols_sql = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    where_sql, params = _build_where_clause(where)

    query = f"SELECT {cols_sql} FROM read_parquet(?){where_sql}"
    all_params = [path_str] + params

    with duckdb.connect() as con:
        df = con.execute(query, all_params).df()

    if add_derived_cols:
        df = add_derived(df, today=today)

    return df


def _pct_from(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a / b - 1) * 100, NaN where b is missing or not positive."""
    return (a / b.where(b > 0) - 1) * 100


def _relative_ratio(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns or "industry" not in df.columns or "sector" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    val = df[col].where(df[col] > 0)
    ind = df["industry"]
    sec = df["sector"]

    ind_cnt = val.groupby(ind).transform("count")
    ind_med = val.groupby(ind).transform("median")

    sec_cnt = val.groupby(sec).transform("count")
    sec_med = val.groupby(sec).transform("median")

    med = ind_med.where(ind_cnt >= 5, sec_med.where(sec_cnt >= 5))
    res = (val / med - 1) * 100.0
    return res.astype(float)


def _relative_diff(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns or "sector" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    val = df[col]
    sec = df["sector"]

    sec_cnt = val.groupby(sec).transform("count")
    sec_med = val.groupby(sec).transform("median")

    res = val - sec_med.where(sec_cnt >= 5)
    return res.astype(float)


def relative_to_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pe_vs_industry": _relative_ratio(df, "pe"),
        "ps_vs_industry": _relative_ratio(df, "ps"),
        "sales_growth_vs_sector": _relative_diff(df, "sales_growth"),
    }


def _with(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """df with `cols` set (replacing columns of the same name, like df[k] = v), in one concat."""
    return pd.concat([df.drop(columns=list(cols), errors="ignore"), pd.DataFrame(cols, index=df.index)], axis=1)


def add_derived(raw: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    df = raw.copy()
    # Snapshots saved by an older version lack newer fields: add them as missing (NaN never passes a filter).
    from screener.fields import provider_columns
    missing = [k for k in provider_columns() if k not in df.columns]
    if missing:
        missing_dict = {k: None if COLUMNS[k].kind == "text" else np.nan for k in missing}
        if "earnings_date" in missing:
            missing_dict["earnings_date"] = pd.to_datetime(pd.Series(np.nan, index=df.index))
        df = _with(df, missing_dict)

    today = today or dt.date.today()
    p = df["price"]

    der = {}
    der["adj_price"] = p     # today's adjusted price is the last price; adjustment only rescales past prices
    der["dollar_volume"] = df["avg_volume"] * p
    der["fwd_pe"] = p / df["eps_next_fy"].where(df["eps_next_fy"] > 0)
    for n in (20, 50, 150, 200):
        der[f"sma{n}_pct"] = _pct_from(p, df[f"sma{n}"])
    der["sma50w_pct"] = _pct_from(df["close_w"], df["sma50_w"])
    der["high52_pct"] = _pct_from(p, df["high52"])
    der["low52_pct"] = _pct_from(p, df["low52"])
    span = (df["high52"] - df["low52"]).where(lambda s: s > 0)
    der["range52"] = ((p - df["low52"]) / span).clip(0, 1)
    der["range_1m_pct"] = _pct_from(df["high_1m"], df["low_1m"])
    der["range_5d_pct"] = _pct_from(df["high_5d"], df["low_5d"])
    der["target_upside"] = _pct_from(df["target"], p)

    days = (pd.to_datetime(df.get("earnings_date"), errors="coerce") - pd.Timestamp(today)).dt.days
    der["days_to_earnings"] = days.where(days >= 0).astype(float)

    der.update(relative_to_groups(df))

    df = _with(df, der).replace([np.inf, -np.inf], np.nan)
    signals, n_signals = summarize(df)
    df = _with(df, {"signals": signals, "n_signals": n_signals,
                    "new": ""})     # "new" is filled by the app when a previous snapshot exists (history.py)

    return add_scores(df)
