"""
Login screen and user badge UI component for HalgOM Screener.
Provides password-protected authentication screen, 30-day session persistence, and logged-in user header badge.
"""
from __future__ import annotations

import logging
import os
import streamlit as st

from screener.auth import (
    create_session,
    get_user,
    list_users,
    revoke_session,
    validate_session,
    verify_password,
)

log = logging.getLogger(__name__)


def render_login_screen() -> str | None:
    # 0. Test / Environment bypass (for synthetic UI test runner)
    if os.environ.get("HALGOM_NO_AUTH"):
        st.session_state["logged_in"] = True
        st.session_state["user_id"] = "ori"
        st.session_state["user_name"] = "אורי"
        return "ori"

    # 1. Check in-memory session
    if st.session_state.get("logged_in") and st.session_state.get("user_id"):
        return str(st.session_state["user_id"])

    # 2. Check 30-day persistent cookie session
    token = None
    if hasattr(st, "context") and hasattr(st.context, "cookies"):
        try:
            token = st.context.cookies.get("halgom_session")
        except Exception:
            token = None

    if token:
        user_id = validate_session(token)
        if user_id:
            user = get_user(user_id)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user.id
                st.session_state["user_name"] = user.display_name
                st.session_state["session_token"] = token
                return user.id

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
    remember = st.checkbox("זכור אותי ל-30 יום", value=True, key="login_remember")

    if st.button("התחבר", type="primary", key="login_btn"):
        if selected_id and verify_password(selected_id, password):
            user = user_map.get(selected_id) or get_user(selected_id)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user.id
                st.session_state["user_name"] = user.display_name
                if remember:
                    token = create_session(user.id, days=30)
                    st.session_state["session_token"] = token
                    st.html(
                        f"<script>document.cookie = 'halgom_session={token}; max-age=2592000; path=/; SameSite=Lax';</script>",
                        unsafe_allow_javascript=True,
                    )
                st.rerun()
                return user.id
        else:
            st.error("סיסמה שגויה, נסה שוב")

    return None


def render_user_badge() -> None:
    if not st.session_state.get("logged_in"):
        return
    user_name = st.session_state.get("user_name", "")
    col1, col2 = st.columns([3, 1], vertical_alignment="center")
    with col1:
        st.write(f"מחובר: **{user_name}**")
    with col2:
        if st.button("התנתק", key="logout_btn", use_container_width=True):
            token = st.session_state.pop("session_token", None)
            if not token and hasattr(st, "context") and hasattr(st.context, "cookies"):
                try:
                    token = st.context.cookies.get("halgom_session")
                except Exception:
                    token = None
            if token:
                revoke_session(token)
            st.session_state.pop("logged_in", None)
            st.session_state.pop("user_id", None)
            st.session_state.pop("user_name", None)
            st.html(
                "<script>document.cookie = 'halgom_session=; max-age=0; path=/; SameSite=Lax';</script>",
                unsafe_allow_javascript=True,
            )
            st.rerun()
