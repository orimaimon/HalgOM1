"""
HalgOM Screener — Streamlit UI.

Run:  run_screener.bat   (or: streamlit run screener/app.py from the HalgOM folder)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import datetime as dt  # noqa: E402
from screener import i18n as I, theme  # noqa: E402
from screener.data.store import session_date  # noqa: E402
from screener.fields import DEFAULT_SORT  # noqa: E402
from screener.filters import ANY, FILTERS, apply_filters, hide_junk, search  # noqa: E402
from screener.ui import alerts_view, home_page, maintenance_view  # noqa: E402
from screener.ui.common import (L, PAGES, _market, setup_logging, start_watchdog, all_presets, get_data, get_previous,  # noqa: E402
    get_store, goodbye, header, hebrew_note, keep_widget_state, nav, new_column, preset_modified, selections, ss, t)
from screener.ui.groups_page import groups_page, heatmap_page  # noqa: E402
from screener.ui.guide_page import guide_page  # noqa: E402
from screener.ui.portfolios_page import portfolios_page  # noqa: E402
from screener.ui.screener_page import filter_panel, results, toolbar  # noqa: E402
from screener.ui.track_record import track_record_expander  # noqa: E402


def main() -> None:
    setup_logging()
    ss.setdefault("lang", I.HE)
    ss.setdefault("large_text", False)
    css = theme.CSS + (theme.RTL_CSS if L() == I.HE else "") + (theme.LARGE_TEXT_CSS if ss.get("large_text") else "")
    st.html(css)
    if ss.get("exiting"):
        goodbye()
        return
    start_watchdog()
    keep_widget_state()
    ss.setdefault("page", PAGES[0])
    ss.setdefault("view", "Overview")
    ss.setdefault("sort_col", DEFAULT_SORT[0])
    ss.setdefault("sort_asc", DEFAULT_SORT[1])
    ss.setdefault("hide_junk", True)
    for f in FILTERS:
        ss.setdefault(f"f_{f.key}", ANY)

    snap, df = get_data()
    maintenance_view.show_whatsnew()
    header(snap, len(df))
    page = nav()
    if page == "Screener":
        presets = all_presets()
        toolbar(presets)
        preset_key = ss.get("preset")
        if preset_key in presets and not preset_modified(presets):
            track_record_expander(preset_key)
        filter_panel(df)

    sel = selections()
    prev = get_previous(snap)
    pipeline_key = (snap.fetched_at, tuple(sorted(sel.items())), ss.get("query", ""),
                    ss.get("hide_junk", True), ss.get("sort_col"), ss.get("sort_asc"),
                    ss.get("only_new", False))
    if ss.get("pipeline_key") == pipeline_key and "pipeline_res" in ss and "pipeline_universe" in ss:
        universe = ss["pipeline_universe"]
        res = ss["pipeline_res"]
    else:
        universe = hide_junk(df, ss.get("hide_junk"))
        res = search(apply_filters(universe, sel), ss.get("query", ""))
        res = res.assign(new=new_column(res, prev, sel))
        res = res.sort_values(ss.get("sort_col"), ascending=ss.get("sort_asc"), na_position="last")
        ss["pipeline_key"] = pipeline_key
        ss["pipeline_universe"] = universe
        ss["pipeline_res"] = res

    if page == "Home":
        m, regime = _market()
        now_date = session_date(dt.datetime.now())
        fired = alerts_view.fired_today(universe, now_date)
        home_page.home_page(universe, get_store(), regime, fired)
    elif page == "Guide":
        guide_page()
    elif page == "Groups":
        groups_page(universe, res)
    elif page == "Heatmap":
        heatmap_page(universe, res)
    elif page == "Portfolios":
        portfolios_page(df)
    else:
        results(df, universe, res, snap, prev)
    hebrew_note(t("Data: TradingView (delayed ~15 min). For research purposes only, not investment advice."))


# Streamlit (`streamlit run`, AppTest) runs this file as "__main__". A plain `import screener.app` (tests that use
# its helpers) must not run the app: that fetched the universe from TradingView and wrote into screener/_data.
if __name__ == "__main__":
    main()
