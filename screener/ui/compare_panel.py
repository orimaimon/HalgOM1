"""Side-by-side stock comparison panel component.

Allows comparing 2-4 stocks side-by-side across fundamental and technical metrics,
highlighting the best ticker per metric and displaying a summary line.
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
            except Exception as comp_err:
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
