"""
User profile and password management UI component for HalgOM Screener.
Allows changing passwords and adding new authorized users.
"""
from __future__ import annotations

import logging

import streamlit as st

from screener.auth import change_password, create_user
from screener.ui.common import hebrew_note

log = logging.getLogger("screener.app")


def render_user_profile(user_id: str) -> None:
    try:
        _render_body(user_id)
    except Exception:
        log.exception("user profile panel render failed")
        hebrew_note("אירעה שגיאה בטעינת פרופיל המשתמש")


def _render_body(user_id: str) -> None:
    st.subheader("שינוי סיסמה")
    with st.form("up_change_pass_form", clear_on_submit=False):
        old_pass = st.text_input("סיסמה נוכחית", type="password", key="up_old_pass")
        new_pass = st.text_input("סיסמה חדשה", type="password", key="up_new_pass")
        confirm_pass = st.text_input("אימות סיסמה חדשה", type="password", key="up_confirm_pass")
        submitted_pass = st.form_submit_button("שנה סיסמה", key="up_change_pass_btn")

        if submitted_pass:
            if not old_pass or not new_pass or not confirm_pass:
                st.error("יש למלא את כל השדות")
            elif new_pass != confirm_pass:
                st.error("הסיסמאות אינן תואמות")
            else:
                success = change_password(user_id, old_pass, new_pass)
                if success:
                    st.success("הסיסמה שונתה בהצלחה")
                else:
                    st.error("הסיסמה הנוכחית אינה נכונה")

    st.subheader("הוספת משתמש חדש")
    with st.form("up_add_user_form", clear_on_submit=False):
        new_name = st.text_input("שם משתמש", key="up_new_name")
        new_pass_user = st.text_input("סיסמה", type="password", key="up_new_user_pass")
        confirm_pass_user = st.text_input("אימות סיסמה", type="password", key="up_confirm_user_pass")
        submitted_user = st.form_submit_button("צור משתמש", key="up_add_user_btn")

        if submitted_user:
            cleaned_name = new_name.strip()
            if not cleaned_name:
                st.error("יש להזין שם משתמש")
            elif not new_pass_user:
                st.error("יש להזין סיסמה")
            elif new_pass_user != confirm_pass_user:
                st.error("הסיסמאות אינן תואמות")
            else:
                try:
                    create_user(cleaned_name, new_pass_user)
                    st.success(f"המשתמש {cleaned_name} נוצר בהצלחה")
                except Exception as e:
                    log.exception("Failed to create user in profile view: %s", e)
                    st.error(f"שגיאה ביצירת משתמש: {e}")
