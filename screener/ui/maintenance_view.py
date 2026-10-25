"""Maintenance panel UI component (problem report, backup & restore, version)
and What's New dialog.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from screener import backup, support, whatsnew
from screener.data import duck_engine, store
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")

FILE_LABELS: dict[str, str] = {
    "presets.json": "סריקות שמורות",
    "portfolios.json": "תיקים ורשימות מעקב",
    "alerts.json": "התראות",
}


@st.dialog(t("What's new"))
def _whatsnew_dialog(unseen_entries: list[dict]) -> None:
    for entry in unseen_entries:
        st.markdown(f"**{entry.get('title', '')}**")
        st.write(entry.get("text", ""))
    if st.button(t("Understood"), key="mt_whatsnew_ack"):
        whatsnew.mark_seen()
        ss["mt_whatsnew_shown"] = True
        st.rerun()


def show_whatsnew() -> None:
    if ss.get("mt_whatsnew_shown"):
        return
    try:
        unseen_entries = whatsnew.unseen()
        if unseen_entries:
            _whatsnew_dialog(unseen_entries)
        else:
            ss["mt_whatsnew_shown"] = True
    except Exception:
        log.exception("failed to display whatsnew dialog")


def render() -> None:
    try:
        _render_body()
    except Exception:
        log.exception("maintenance panel render failed")
        hebrew_note("אירעה שגיאה בטעינת פאנל התחזוקה")


def _render_body() -> None:
    # 1. שליחת דוח תקלה
    st.subheader(t("Send problem report"))
    if st.button(t("Send problem report"), key="mt_report"):
        try:
            ver = whatsnew.version_label()
            rep_path = support.make_report(ver)
            st.success(f"נשמר: {rep_path}")
            st.info(t("Send this file to Ori (email or WhatsApp)"))
        except Exception:
            log.exception("failed to create problem report")
            hebrew_note("אירעה שגיאה ביצירת דוח התקלה")

    # 2. גיבוי ושחזור
    st.subheader(t("Backup and restore"))
    col_bak, col_res = st.columns(2)

    with col_bak:
        if st.button(t("Backup"), key="mt_backup"):
            try:
                desktops = support.desktop_dirs()
                target_dir = desktops[0] if (desktops and desktops[0].exists()) else (store.DATA_DIR / "reports")
                bak_path = backup.make_backup(target_dir)
                st.success(f"גיבוי נשמר: {bak_path}")
            except Exception:
                log.exception("failed to create backup")
                hebrew_note("אירעה שגיאה ביצירת הגיבוי")

    with col_res:
        uploaded_file = st.file_uploader(t("Upload backup file (.zip)"), type=["zip"], key="mt_restore_file")
        if uploaded_file is not None:
            try:
                with tempfile.NamedTemporaryFile("wb", suffix=".zip", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    inspection = backup.inspect(tmp_path)
                    has_error = any(not v.get("ok", False) for v in inspection.values()) or ("_zip" in inspection)

                    if has_error:
                        problems = []
                        for v in inspection.values():
                            if not v.get("ok", False):
                                p = v.get("problem")
                                if p:
                                    problems.append(p)
                        err_msg = ", ".join(problems) if problems else backup.MESSAGES["not_a_backup"]
                        hebrew_note(err_msg)
                    else:
                        details = []
                        for name, info in inspection.items():
                            cnt = info.get("items", 0)
                            lbl = FILE_LABELS.get(name, name)
                            details.append(f"{lbl}: {cnt}")
                        st.write("תוכן הגיבוי: " + ", ".join(details))

                        mode_label = st.radio(
                            t("Restore backup"),
                            options=[t("Merge"), t("Replace")],
                            key="mt_restore_mode",
                        )

                        if st.button(t("Confirm restore"), key="mt_restore_go"):
                            mode_val = "merge" if mode_label == t("Merge") else "replace"
                            res = backup.restore(tmp_path, mode=mode_val)
                            if any(not v.get("ok", True) for v in res.values()):
                                errs = [v.get("problem") for v in res.values() if v.get("problem")]
                                hebrew_note(", ".join(filter(None, errs)) or "שגיאה בשחזור")
                            else:
                                summary_parts = []
                                for name, info in res.items():
                                    lbl = FILE_LABELS.get(name, name)
                                    if mode_val == "merge":
                                        summary_parts.append(f"{lbl}: נוספו {info.get('added', 0)}, כבר היו {info.get('skipped', 0)}")
                                    else:
                                        summary_parts.append(f"{lbl}: נוספו {info.get('added', 0)}")
                                st.success("השחזור הושלם בהצלחה! " + " | ".join(summary_parts))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            except Exception:
                log.exception("failed during backup restore")
                hebrew_note("אירעה שגיאה בשחזור הגיבוי")

    # 3. בדיקת תקינות מאגר נתונים
    st.subheader("בדיקת תקינות מאגר נתונים")
    if st.button("בדיקת תקינות מאגר נתונים", key="mt_health_check"):
        try:
            res = duck_engine.health_check(store.DATA_DIR)
            status = res.get("status")
            if status == "ok":
                total = res.get("total_files", 0)
                st.success(f"כל קובצי הנתונים תקינים ({total} קבצים)")
            elif status == "degraded":
                corrupted = res.get("corrupted_files", 0)
                files = res.get("files", [])
                corrupted_details = []
                for f in files:
                    if not f.get("valid", True):
                        p = f.get("path", "")
                        err = f.get("error") or "קובץ פגום"
                        corrupted_details.append(f"{p}: {err}")
                details_str = "\n".join(corrupted_details) if corrupted_details else "נמצאו קבצים פגומים"
                st.warning(f"נמצאו {corrupted} קבצים פגומים:\n{details_str}")
            else:
                err = res.get("error") or "אירעה שגיאה בבדיקת תקינות מאגר הנתונים"
                hebrew_note(err)
        except Exception:
            log.exception("failed duckdb storage health check")
            hebrew_note("אירעה שגיאה בבדיקת תקינות מאגר הנתונים")

    # 4. גירסה
    st.caption(whatsnew.version_label())
