"""
Sector Rotation Quadrant Analysis UI Component.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import i18n, sector_rotation
from screener.theme import GAIN, LOSS
from screener.ui.common import hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def translate_sector(sector: str) -> str:
    """Translates an English sector name to Hebrew using i18n."""
    return i18n.value("sector", str(sector), i18n.HE)


if not hasattr(i18n, "translate_sector"):
    setattr(i18n, "translate_sector", translate_sector)


def _ensure_universe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures required columns exist for group_performance calculation in rotation_table."""
    d = df.copy()
    if "price" not in d.columns:
        d["price"] = np.nan
    if "sma50" not in d.columns:
        d["sma50"] = np.nan
    if "market_cap" not in d.columns:
        d["market_cap"] = np.nan
    return d


def render_sector_rotation(sector_df: pd.DataFrame) -> None:
    """Renders sector rotation leaderboard table, rank change arrows, and top/bottom sector cards & chart."""
    try:
        if sector_df is None or sector_df.empty:
            hebrew_note(t("No sector data"))
            return

        if "rs_trend" in sector_df.columns and "quadrant" in sector_df.columns:
            rot = sector_df.copy()
        else:
            weight_val = ss.get("sr_weight", "Median")
            weight_mode = (
                "cap" if weight_val == "Market cap" else "equal" if weight_val == "Equal" else "median"
            )
            prep_df = _ensure_universe_columns(sector_df)
            rot = sector_rotation.rotation_table(prep_df, weight=weight_mode)

        if rot.empty:
            hebrew_note(t("No sector data"))
            return

        rot = rot.sort_values("rs_trend", ascending=False).reset_index(drop=True)
        rot["rank"] = rot.index + 1

        if "rank_change" in rot.columns and rot["rank_change"].notna().any():
            rot["rank_change_num"] = rot["rank_change"]
        elif "prev_rank" in rot.columns and rot["prev_rank"].notna().any():
            rot["rank_change_num"] = rot["prev_rank"] - rot["rank"]
        elif "perf_m" in rot.columns and "perf_q" in rot.columns and len(rot) > 1:
            m_ranks = rot["perf_m"].rank(ascending=False, method="min")
            rot["rank_change_num"] = (rot.index + 1) - m_ranks
        else:
            rot["rank_change_num"] = 0

        def format_arrow(val):
            if pd.isna(val) or val == 0:
                return "—"
            val = int(val)
            if val > 0:
                return f"▲ +{val}"
            return f"▼ {val}"

        rot["rank_arrow"] = rot["rank_change_num"].map(format_arrow)
        rot["sector_he"] = rot["sector"].map(lambda s: i18n.translate_sector(s))
        rot["quadrant_he_str"] = rot["quadrant"].map(
            lambda q: sector_rotation.QUADRANTS_HE.get(q, q) if pd.notna(q) else "—"
        )

        st.subheader("מובילים ומפגרים בסקטורים")

        col_top, col_bot = st.columns(2)

        top_3 = rot.head(3)
        bot_3 = rot.tail(3).iloc[::-1]

        with col_top:
            st.markdown("##### 🚀 סקטורים מובילים (Top 3)")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for idx, (_, row) in enumerate(top_3.iterrows()):
                val_str = f"{row['rs_trend']:+.2f}%" if pd.notna(row.get("rs_trend")) else "—"
                cols[idx % 3].metric(
                    label=row["sector_he"],
                    value=val_str,
                    delta=f"דירוג #{row['rank']} ({row['rank_arrow']})",
                )

        with col_bot:
            st.markdown("##### 📉 סקטורים מפגרים (Bottom 3)")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for idx, (_, row) in enumerate(bot_3.iterrows()):
                val_str = f"{row['rs_trend']:+.2f}%" if pd.notna(row.get("rs_trend")) else "—"
                cols[idx % 3].metric(
                    label=row["sector_he"],
                    value=val_str,
                    delta=f"דירוג #{row['rank']} ({row['rank_arrow']})",
                    delta_color="inverse",
                )

        chart_df = pd.concat([top_3, bot_3.iloc[::-1]]).drop_duplicates(subset=["sector"])
        if not chart_df.empty:
            fig = go.Figure()
            colors = [GAIN if v >= 0 else LOSS for v in chart_df["rs_trend"]]
            fig.add_trace(
                go.Bar(
                    x=chart_df["sector_he"],
                    y=chart_df["rs_trend"],
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in chart_df["rs_trend"]],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="מגמה יחסית (3M) - מובילים ומפגרים",
                height=300,
                margin=dict(t=30, l=10, r=10, b=10),
                yaxis_title="מגמה יחסית (%)",
                xaxis_title="סקטור",
            )
            st.plotly_chart(fig, use_container_width=True, key="sr_top_bot_chart")

        st.subheader("לוח מובילי סקטורים (Leaderboard)")

        disp_cols = [
            "rank",
            "sector_he",
            "rank_arrow",
            "rs_trend",
            "rs_momentum",
            "perf_m",
            "perf_q",
            "quadrant_he_str",
        ]
        avail_cols = [c for c in disp_cols if c in rot.columns]
        disp_df = rot[avail_cols].copy()

        cc = st.column_config
        config = {
            "rank": cc.NumberColumn("דירוג", format="%d"),
            "sector_he": cc.TextColumn("סקטור"),
            "rank_arrow": cc.TextColumn("שינוי בדירוג"),
            "rs_trend": cc.NumberColumn("מגמה יחסית (3M)", format="%.2f%%"),
            "rs_momentum": cc.NumberColumn("מומנטום יחסי (1M)", format="%.2f%%"),
            "perf_m": cc.NumberColumn("תשואה חודשית", format="%.2f%%"),
            "perf_q": cc.NumberColumn("תשואה רבעונית", format="%.2f%%"),
            "quadrant_he_str": cc.TextColumn("רביע"),
        }

        st.dataframe(
            disp_df,
            hide_index=True,
            column_config=config,
            use_container_width=True,
            key="sr_leaderboard_table",
        )
    except Exception:
        log.exception("error displaying sector rotation visual leaderboard")
        hebrew_note(t("Error displaying sector rotation"))


def render(universe: pd.DataFrame) -> None:
    """Renders sector rotation segmented control, scatter chart, and visual leaderboard."""
    try:
        ss.setdefault("sr_weight", "Median")
        st.segmented_control(
            t("Averages"),
            ["Median", "Equal", "Market cap"],
            key="sr_weight",
            format_func=t,
        )

        weight_val = ss.get("sr_weight", "Median")
        weight_mode = (
            "cap" if weight_val == "Market cap" else "equal" if weight_val == "Equal" else "median"
        )

        prep_df = _ensure_universe_columns(universe) if universe is not None and not universe.empty else universe
        rot = sector_rotation.rotation_table(prep_df, weight=weight_mode)
        if rot.empty:
            hebrew_note(t("No sector data"))
            return

        st.plotly_chart(
            sector_rotation.rotation_figure(rot, L()),
            use_container_width=True,
            key="sr_chart",
        )

        render_sector_rotation(rot)
    except Exception:
        log.exception("error displaying sector rotation")
        hebrew_note(t("Error displaying sector rotation"))
