"""
Login screen and user badge UI component for HalgOM Screener.
Provides password-protected authentication screen and logged-in user header badge.
"""
from __future__ import annotations

import logging
import streamlit as st

from screener.auth import create_user, get_user, list_users, verify_password

log = logging.getLogger(__name__)


def render_login_screen() -> str | None:
    if st.session_state.get("logged_in") and st.session_state.get("user_id"):
        return str(st.session_state["user_id"])

    st.header("🔒 התחברות למערכת HalgOM")

    users = list_users()
    if not users:
        st.error("לא נמצאו משתמשים במערכת")
        return None

    user_ids = [u.id for u in users]
    user_map = {u.id: u for u in users}

    selected_id = st.selectbox(
        "בחר משתמש",
        options=user_ids,
        format_func=lambda uid: user_map[uid].display_name if uid in user_map else uid,
        key="login_user_id",
    )

    password = st.text_input("סיסמה", type="password", key="login_password")

    if st.button("התחבר", type="primary", key="login_btn"):
        if selected_id and verify_password(selected_id, password):
            user = user_map.get(selected_id) or get_user(selected_id)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user.id
                st.session_state["user_name"] = user.display_name
                st.rerun()
                return user.id
        else:
            st.error("סיסמה שגויה, נסה שוב")

    with st.expander("משתמש חדש"):
        with st.form("reg_form", clear_on_submit=False):
            new_name = st.text_input("שם משתמש", key="reg_name")
            new_pass = st.text_input("סיסמה", type="password", key="reg_pass")
            confirm_pass = st.text_input("אימות סיסמה", type="password", key="reg_confirm")
            submitted = st.form_submit_button("צור משתמש והתחבר", key="reg_submit")

            if submitted:
                cleaned_name = new_name.strip()
                if not cleaned_name:
                    st.error("יש להזין שם משתמש")
                elif not new_pass:
                    st.error("יש להזין סיסמה")
                elif new_pass != confirm_pass:
                    st.error("הסיסמאות אינן תואמות")
                else:
                    try:
                        user = create_user(cleaned_name, new_pass)
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"] = user.id
                        st.session_state["user_name"] = user.display_name
                        st.rerun()
                        return user.id
                    except Exception as e:
                        log.exception("Failed to create user: %s", e)
                        st.error(f"שגיאה ביצירת משתמש: {e}")

    return None


def render_user_badge() -> None:
    if not st.session_state.get("logged_in"):
        return
    user_name = st.session_state.get("user_name", "")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"מחובר: **{user_name}**")
    with col2:
        if st.button("התנתק", key="logout_btn"):
            st.session_state.pop("logged_in", None)
            st.session_state.pop("user_id", None)
            st.session_state.pop("user_name", None)
            st.rerun()
