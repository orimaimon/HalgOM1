"""
Side-by-side stock comparison engine.

Pure functions comparing 2-4 stocks per metric, formatting values,
and determining best metric values and summary lines.
"""
from __future__ import annotations

from collections import Counter
import pandas as pd

from screener import fields, i18n
from screener.explain import format_value

METRICS: list[tuple[str, str]] = [
    ("market_cap", ""),
    ("price", ""),
    ("change", ""),
    ("pe", "low"),
    ("fwd_pe", "low"),
    ("ps", "low"),
    ("peg", "low"),
    ("sales_growth", "high"),
    ("eps_growth", "high"),
    ("gross_margin", "high"),
    ("net_margin", "high"),
    ("roe", "high"),
    ("debt_eq", "low"),
    ("div_yield", "high"),
    ("perf_y", "high"),
    ("sma200_pct", ""),
    ("rsi", ""),
    ("beta", ""),
    ("score", "high"),
]

POSITIVE_ONLY_METRICS = {"pe", "fwd_pe", "ps", "peg"}


def compare(
    universe: pd.DataFrame,
    tickers: list[str],
    metrics: list[tuple[str, str]] | None = None,
    lang: str = i18n.HE,
) -> pd.DataFrame:
    """Compare 2-4 tickers side-by-side across metrics.

    Returns a DataFrame with rows = metrics and columns = ["metric"] + present tickers.
    df.attrs["best"]: {metric_key: winning_ticker}
    df.attrs["missing"]: list of requested tickers not present in the universe
    """
    if metrics is None:
        metrics = METRICS

    # Clean and deduplicate tickers case-insensitively, preserving order
    clean_tickers: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        if not isinstance(t, str):
            continue
        t_upper = t.strip().upper()
        if not t_upper:
            continue
        if t_upper not in seen:
            seen.add(t_upper)
            clean_tickers.append(t_upper)

    if len(clean_tickers) > 4:
        raise ValueError("ניתן להשוות עד 4 מניות בלבד")

    # Match tickers in universe
    ticker_col = universe["ticker"] if "ticker" in universe.columns else pd.Series(dtype=str)
    univ_ticker_map = {}
    if not ticker_col.empty:
        for idx, val in ticker_col.items():
            if pd.notna(val):
                univ_ticker_map[str(val).strip().upper()] = idx

    present_tickers: list[str] = []
    missing_tickers: list[str] = []
    ticker_rows: dict[str, pd.Series] = {}

    for t in clean_tickers:
        if t in univ_ticker_map:
            present_tickers.append(t)
            row_idx = univ_ticker_map[t]
            ticker_rows[t] = universe.loc[row_idx]
        else:
            missing_tickers.append(t)

    # Build matrix
    data: dict[str, list[str]] = {"metric": []}
    for t in present_tickers:
        data[t] = []

    best: dict[str, str] = {}

    for key, direction in metrics:
        col_def = fields.COLUMNS.get(key)
        label = col_def.label if col_def else key
        metric_name = i18n.column(key, label, lang)
        data["metric"].append(metric_name)

        competing_vals: list[tuple[str, float]] = []

        for t in present_tickers:
            row = ticker_rows[t]
            val = row.get(key) if key in row.index else None
            formatted = format_value(key, val, lang=lang)
            data[t].append(formatted)

            # Check if value qualifies for 'best'
            if direction in ("high", "low") and val is not None and not pd.isna(val) and val is not pd.NaT:
                try:
                    num_val = float(val)
                    if key in POSITIVE_ONLY_METRICS:
                        if num_val > 0:
                            competing_vals.append((t, num_val))
                    else:
                        competing_vals.append((t, num_val))
                except (ValueError, TypeError):
                    pass

        # Calculate best if >= 2 competing values
        if direction in ("high", "low") and len(competing_vals) >= 2:
            if direction == "high":
                target_val = max(v for _, v in competing_vals)
            else:
                target_val = min(v for _, v in competing_vals)

            winners = [t for t, v in competing_vals if v == target_val]
            if len(winners) == 1:
                best[key] = winners[0]

    metric_keys = [k for k, _ in metrics]
    columns = ["metric"] + present_tickers
    res_df = pd.DataFrame(data, index=metric_keys, columns=columns)
    res_df.attrs["best"] = best
    res_df.attrs["missing"] = missing_tickers

    return res_df


def summary_line(
    universe: pd.DataFrame,
    tickers: list[str],
    metrics: list[tuple[str, str]] | None = None,
    lang: str = i18n.HE,
) -> str:
    """Return a single Hebrew sentence summarizing who leads in how many metrics.

    e.g. 'MSFT מובילה ב-6 מדדים, AAPL ב-4, GOOGL ב-3'
    Returns an empty string when nothing competes.
    """
    df = compare(universe, tickers, metrics=metrics, lang=lang)
    best = df.attrs.get("best", {})
    if not best:
        return ""

    counts = Counter(best.values())
    if not counts:
        return ""

    # Sort tickers by win count descending, preserving present ticker order for ties
    present_tickers = [c for c in df.columns if c != "metric"]
    sorted_tickers = sorted(counts.keys(), key=lambda t: (-counts[t], present_tickers.index(t) if t in present_tickers else 999))

    parts = []
    for idx, t in enumerate(sorted_tickers):
        cnt = counts[t]
        if idx == 0:
            unit = "מדד" if cnt == 1 else "מדדים"
            parts.append(f"{t} מובילה ב-{cnt} {unit}")
        else:
            parts.append(f"{t} ב-{cnt}")

    return ", ".join(parts)
