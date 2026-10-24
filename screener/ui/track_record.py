"""Track record ("תעודת ביצועים") panel on the Screener page and comparison tab on the Groups page.

Moved verbatim from screener/app.py; behavior unchanged."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from screener import i18n as I, presets as P, track
from screener.data import Store

from screener.ui.common import L, USER_PREFIX, _store, hebrew_note, log, t


def _snapshot_state_key(store: Store) -> tuple:
    dates = store.snapshot_dates()
    stamps = []
    for d in dates:
        p = store.snapshot_path(d)
        if p.exists():
            stamps.append((d.isoformat(), p.stat().st_mtime_ns))
    return tuple(stamps)


def _presets_state_key() -> str:
    user_p = P.load_user()
    return hashlib.sha256(json.dumps({k: v.filters for k, v in user_p.items()}, sort_keys=True).encode()).hexdigest()[:12]


# Keyed by the snapshot files (day, mtime) and the saved screens; the result is language-independent.
# A few entries only: each holds the entries table of every screen.
@st.cache_data(show_spinner="מחשב תעודת ביצועים…", max_entries=3)
def _get_track_all_screens(snapshot_key: tuple, presets_key: str) -> dict[str, dict]:
    return track.track_all_screens(_store())


def get_all_track_records() -> dict[str, dict] | None:
    try:
        store = _store()
        return _get_track_all_screens(_snapshot_state_key(store), _presets_state_key())
    except Exception as e:
        log.exception("track_all_screens failed: %s", e)
        return None


TRACK_ERROR_NOTE = "לא ניתן לחשב תעודת ביצועים כרגע."


def track_record_expander(preset_key: str) -> None:
    with st.expander(t("Track record"), expanded=False):
        try:
            _track_record_body(preset_key)
        except Exception as e:          # the Screener page must never break because of this panel
            log.exception("track record panel failed: %s", e)
            hebrew_note(TRACK_ERROR_NOTE)


def _track_record_body(preset_key: str) -> None:
    all_tracks = get_all_track_records()
    if all_tracks is None or preset_key not in all_tracks:
        hebrew_note(TRACK_ERROR_NOTE)
        return

    summary = all_tracks[preset_key]
    sessions = summary.get("sessions", 0)
    few_sessions = summary.get("few_sessions", False)
    horizons = summary.get("horizons", {})

    if few_sessions:
        st.warning(f"⚠️ {t('Still few data points — numbers will stabilize over time')} ({sessions} ימי מסחר)")

    for h in track.HORIZONS:
        hz = horizons.get(h, {})
        h_entries = hz.get("entries", 0)
        pending = hz.get("pending", 0)
        median = hz.get("median")
        pct_pos = hz.get("pct_positive")
        excess = hz.get("excess")
        bench = hz.get("benchmark")

        bench_text = t("Over market median") if bench == track.UNIVERSE else t("Over SPY")
        med_str = f"{median:+.2f}%" if median is not None else "—"
        pos_str = f"{pct_pos:.1f}%" if pct_pos is not None else "—"
        exc_str = f"{excess:+.2f}% {bench_text}" if excess is not None else "—"
        pending_str = f" (ממתין: {pending})" if pending > 0 else ""

        st.markdown(f"**אחרי {h} ימי מסחר:** {h_entries} כניסות{pending_str} · "
                    f"חציון: **{med_str}** · "
                    f"% חיוביות: **{pos_str}** · "
                    f"עודף: **{exc_str}**")

    st.caption(t("Past performance only. Does not guarantee future results and is not investment advice."))

    ent_table = summary.get("table")
    if isinstance(ent_table, pd.DataFrame) and not ent_table.empty:
        recent = ent_table.sort_values("day", ascending=False).head(20).copy()
        shown_cols = ["day", "ticker"]
        col_renames = {"day": t("Date"), "ticker": t("Ticker")}
        for h in track.HORIZONS:
            ret_col = f"ret_{h}"
            col_renames[ret_col] = f"{h} ימים"
            shown_cols.append(ret_col)
            if ret_col in recent.columns:
                recent[ret_col] = recent.apply(
                    lambda r, h=h: t("Pending") if r.get(f"pending_{h}") else ("—" if pd.isna(r.get(f"ret_{h}")) or r.get(f"ret_{h}") is None else f"{r.get(f'ret_{h}'):+.2f}%"),
                    axis=1
                )
        recent_disp = recent[shown_cols].rename(columns=col_renames)
        st.dataframe(recent_disp, hide_index=True, use_container_width=True)


def groups_track_record() -> None:
    try:
        _groups_track_record_body()
    except Exception as e:              # the Groups page must never break because of this tab
        log.exception("track record comparison failed: %s", e)
        hebrew_note(TRACK_ERROR_NOTE)


def _groups_track_record_body() -> None:
    all_tracks = get_all_track_records()
    if all_tracks is None:
        hebrew_note(TRACK_ERROR_NOTE)
        return

    rows = []
    for key, summary in all_tracks.items():
        if key in P.BUILTIN:
            display_name = I.preset(key, L())
        elif key.startswith(USER_PREFIX):
            display_name = key.removeprefix(USER_PREFIX)
        else:
            display_name = key

        entries_count = summary.get("entries", 0)
        hz20 = summary.get("horizons", {}).get(20, {})
        med20 = hz20.get("median")
        pct_pos20 = hz20.get("pct_positive")
        excess20 = hz20.get("excess")

        rows.append({
            "name": display_name,
            "entries": entries_count,
            "median_20": med20,
            "pct_positive_20": pct_pos20,
            "excess_20": excess20,
        })

    rows.sort(key=lambda r: (r["excess_20"] is None, -(r["excess_20"] or 0)))

    def _fmt_pct(v):
        if v is None or pd.isna(v):
            return "—"
        return f"{v:+.2f}%"

    def _fmt_pos(v):
        if v is None or pd.isna(v):
            return "—"
        return f"{v:.1f}%"

    df_track_disp = pd.DataFrame({
        t("Screen"): [r["name"] for r in rows],
        t("Entries"): [r["entries"] for r in rows],
        t("20-day median"): [_fmt_pct(r["median_20"]) for r in rows],
        t("% positive"): [_fmt_pos(r["pct_positive_20"]) for r in rows],
        t("20-day excess"): [_fmt_pct(r["excess_20"]) for r in rows],
    })

    st.dataframe(df_track_disp, hide_index=True, use_container_width=True, key="groups_track_table")
    hebrew_note(t("Past performance only. Does not guarantee future results and is not investment advice."))
