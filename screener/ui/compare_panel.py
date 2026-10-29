"""Side-by-side stock comparison panel component.

Allows comparing 2-5 stocks side-by-side across fundamental and technical metrics,
highlighting metric winners and dimension leaders.
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from screener import compare, theme
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def render(universe: pd.DataFrame, candidates: list[str]) -> None:
    """Render side-by-side stock comparison expander panel."""
    try:
        with st.expander(t("Stock comparison"), expanded=False):
            # Build options: candidates first (preserving order), followed by rest of universe tickers
            options: list[str] = []
            seen: set[str] = set()

            if candidates:
                for cand in candidates:
                    if cand and isinstance(cand, str):
                        c_upper = cand.strip().upper()
                        if c_upper and c_upper not in seen:
                            seen.add(c_upper)
                            options.append(c_upper)

            if universe is not None and not universe.empty and "ticker" in universe.columns:
                for val in universe["ticker"].dropna():
                    t_str = str(val).strip().upper()
                    if t_str and t_str not in seen:
                        seen.add(t_str)
                        options.append(t_str)

            selected_tickers = st.multiselect(
                t("Stock comparison"),
                options=options,
                default=[],
                max_selections=4,
                key="cmp_tickers",
                label_visibility="collapsed",
            )

            if len(selected_tickers) < 2:
                hebrew_note(t("Select 2 to 4 stocks"))
                return

            try:
                res_df = compare.compare(universe, selected_tickers, lang=L())
            except Exception:
                log.exception("compare.compare failed")
                hebrew_note("שגיאה בחישוב השוואת המניות")
                return

            best_map = res_df.attrs.get("best", {})

            def _style_best(df: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for metric_key, winning_ticker in best_map.items():
                    if metric_key in styles.index and winning_ticker in styles.columns:
                        styles.loc[metric_key, winning_ticker] = f"font-weight: bold; color: {theme.GAIN}"
                return styles

            styled_df = res_df.style.apply(_style_best, axis=None)
            st.dataframe(styled_df, use_container_width=True)

            missing = res_df.attrs.get("missing", [])
            if missing:
                missing_str = ", ".join(missing)
                hebrew_note(f"לא נמצאו בסקרינר: {missing_str}")

            summary = compare.summary_line(universe, selected_tickers, lang=L())
            if summary:
                st.caption(summary)

    except Exception:
        log.exception("compare_panel render failed")
        hebrew_note("אירעה שגיאה בטעינת פאנל השוואת מניות")


def _find_leader(
    ticker_rows: dict[str, pd.Series],
    primary_key: str,
    fallback_key: str,
    primary_high: bool = True,
    fallback_high: bool = True,
) -> str:
    """Find winning ticker(s) for a dimension key among ticker rows."""
    def _eval_key(key: str, high_better: bool) -> tuple[list[str], float | None]:
        vals: list[tuple[str, float]] = []
        is_pos_only = key in compare.POSITIVE_ONLY_METRICS
        is_non_neg = key in compare.NON_NEGATIVE_ONLY_METRICS

        for t_name, row in ticker_rows.items():
            val = row.get(key) if key in row.index else None
            if val is not None and not pd.isna(val) and val is not pd.NaT:
                try:
                    num = float(val)
                    if is_pos_only and num <= 0:
                        continue
                    if is_non_neg and num < 0:
                        continue
                    vals.append((t_name, num))
                except (ValueError, TypeError):
                    pass
        if not vals:
            return [], None
        target = max(v for _, v in vals) if high_better else min(v for _, v in vals)
        winners = [t_name for t_name, v in vals if v == target]
        return winners, target

    winners, _ = _eval_key(primary_key, primary_high)
    if not winners and fallback_key:
        winners, _ = _eval_key(fallback_key, fallback_high)

    if not winners:
        return "—"
    return ", ".join(winners)


def _render_dimension_badges(universe: pd.DataFrame, tickers: list[str]) -> None:
    """Render winner/leader highlight badges for dimensions: ערך, צמיחה, איכות, מומנטום."""
    ticker_col = (
        universe["ticker"]
        if (universe is not None and not universe.empty and "ticker" in universe.columns)
        else pd.Series(dtype=str)
    )
    univ_ticker_map: dict[str, int | str] = {}
    if not ticker_col.empty:
        for idx, val in ticker_col.items():
            if pd.notna(val):
                univ_ticker_map[str(val).strip().upper()] = idx

    ticker_rows: dict[str, pd.Series] = {}
    for t_name in tickers:
        t_upper = t_name.strip().upper()
        if t_upper in univ_ticker_map:
            ticker_rows[t_upper] = universe.loc[univ_ticker_map[t_upper]]

    dimensions = [
        ("ערך", "score_value", "pe", True, False),
        ("צמיחה", "sales_growth", "eps_growth", True, True),
        ("איכות", "score_quality", "roe", True, True),
        ("מומנטום", "score_momentum", "perf_y", True, True),
    ]

    cols = st.columns(4)
    for idx, (label, p_key, f_key, p_high, f_high) in enumerate(dimensions):
        leader = _find_leader(ticker_rows, p_key, f_key, p_high, f_high)
        with cols[idx]:
            st.metric(label=label, value=leader)


def render_stock_comparison(
    universe: pd.DataFrame, default_tickers: list[str] | None = None
) -> None:
    """Render side-by-side stock comparison component with dimension leader badges.

    Parameters
    ----------
    universe : pd.DataFrame
        Stock universe dataframe containing ticker and metric columns.
    default_tickers : list[str] | None
        Optional initial tickers to pre-select (up to 5).
    """
    try:
        options: list[str] = []
        seen: set[str] = set()

        if universe is not None and not universe.empty and "ticker" in universe.columns:
            for val in universe["ticker"].dropna():
                t_str = str(val).strip().upper()
                if t_str and t_str not in seen:
                    seen.add(t_str)
                    options.append(t_str)

        defaults: list[str] = []
        if default_tickers:
            for dt in default_tickers:
                if dt and isinstance(dt, str):
                    dt_upper = dt.strip().upper()
                    if dt_upper in options and dt_upper not in defaults:
                        defaults.append(dt_upper)
            defaults = defaults[:5]

        selected_tickers = st.multiselect(
            t("Stock comparison"),
            options=options,
            default=defaults,
            max_selections=5,
            key="cmp_stock_multi",
        )

        if len(selected_tickers) < 2 or len(selected_tickers) > 5:
            hebrew_note("בחרו 2 עד 5 מניות להשוואה")
            return

        try:
            res_df = compare.compare(universe, selected_tickers, lang=L())
        except Exception:
            log.exception("compare.compare failed in render_stock_comparison")
            hebrew_note("שגיאה בחישוב השוואת המניות")
            return

        _render_dimension_badges(universe, selected_tickers)

        best_map = res_df.attrs.get("best", {})

        def _style_best(df: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for metric_key, winning_ticker in best_map.items():
                if metric_key in styles.index and winning_ticker in styles.columns:
                    styles.loc[metric_key, winning_ticker] = f"font-weight: bold; color: {theme.GAIN}"
            return styles

        styled_df = res_df.style.apply(_style_best, axis=None)
        st.dataframe(styled_df, use_container_width=True)

        missing = res_df.attrs.get("missing", [])
        if missing:
            missing_str = ", ".join(missing)
            hebrew_note(f"לא נמצאו בסקרינר: {missing_str}")

        summary = compare.summary_line(universe, selected_tickers, lang=L())
        if summary:
            st.caption(summary)

    except Exception:
        log.exception("render_stock_comparison failed")
        hebrew_note("אירעה שגיאה בטעינת השוואת המניות")
