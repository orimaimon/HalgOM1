"""
Sector Rotation Quadrant Analysis UI Component.
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from screener import i18n, sector_rotation
from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def render(universe: pd.DataFrame) -> None:
    """Renders sector rotation segmented control, scatter chart, and summary table."""
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

        rot = sector_rotation.rotation_table(universe, weight=weight_mode)
        if rot.empty:
            hebrew_note(t("No sector data"))
            return

        st.plotly_chart(
            sector_rotation.rotation_figure(rot, L()),
            use_container_width=True,
            key="sr_chart",
        )

        disp = rot.copy()
        disp["sector"] = [i18n.value("sector", str(s), L()) for s in disp["sector"]]
        if L() == i18n.HE:
            disp["quadrant"] = disp["quadrant_he"]

        cols_to_show = ["sector", "quadrant", "rs_trend", "rs_momentum", "perf_m", "perf_q"]
        disp_table = disp[cols_to_show]

        cc = st.column_config
        config = {
            "sector": cc.TextColumn(col_label("sector")),
            "quadrant": cc.TextColumn(t("Quadrant")),
            "rs_trend": cc.NumberColumn(t("RS trend (3M)"), format="%.2f%%"),
            "rs_momentum": cc.NumberColumn(t("RS momentum (1M)"), format="%.2f%%"),
            "perf_m": cc.NumberColumn(col_label("perf_m"), format="%.2f%%"),
            "perf_q": cc.NumberColumn(col_label("perf_q"), format="%.2f%%"),
        }

        st.dataframe(
            disp_table,
            hide_index=True,
            column_config=config,
            use_container_width=True,
            key="sr_table",
        )
    except Exception:
        log.exception("error displaying sector rotation")
        hebrew_note(t("Error displaying sector rotation"))
