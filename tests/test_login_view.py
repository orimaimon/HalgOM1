"""
Unit tests for screener.ui.login_view module.
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from screener.data import store


def script_render_login(data_dir: str):
    from pathlib import Path
    import screener.data.store as store_mod
    import screener.ui.login_view as lv

    store_mod.DATA_DIR = Path(data_dir)
    lv.render_login_screen()


def script_render_badge(data_dir: str):
    from pathlib import Path
    import streamlit as st
    import screener.data.store as store_mod
    import screener.ui.login_view as lv

    store_mod.DATA_DIR = Path(data_dir)
    if "init_done" not in st.session_state:
        st.session_state["init_done"] = True
        st.session_state["logged_in"] = True
        st.session_state["user_id"] = "ori"
        st.session_state["user_name"] = "אורי"
    lv.render_user_badge()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_login_screen_initial(isolated_data_dir):
    at = AppTest.from_function(script_render_login, args=(str(isolated_data_dir),)).run()
    assert not at.exception
    assert any("התחברות למערכת HalgOM" in h.value for h in at.header)
    assert len(at.selectbox) > 0
    assert len(at.text_input) >= 1
    assert len(at.button) >= 1


def test_login_failure(isolated_data_dir):
    at = AppTest.from_function(script_render_login, args=(str(isolated_data_dir),)).run()
    assert not at.exception

    # Enter wrong password and click login
    at.text_input(key="login_password").input("wrongpassword")
    at.button(key="login_btn").click().run()

    assert not at.exception
    assert any("סיסמה שגויה, נסה שוב" in err.value for err in at.error)
    assert not at.session_state.get("logged_in")


def test_login_success(isolated_data_dir):
    at = AppTest.from_function(script_render_login, args=(str(isolated_data_dir),)).run()
    assert not at.exception

    at.text_input(key="login_password").input("halgom123")
    at.button(key="login_btn").click().run()

    assert not at.exception
    assert at.session_state.get("logged_in") is True
    assert at.session_state.get("user_id") == "ori"
    assert at.session_state.get("user_name") == "אורי"


def test_register_new_user(isolated_data_dir):
    at = AppTest.from_function(script_render_login, args=(str(isolated_data_dir),)).run()
    assert not at.exception

    at.text_input(key="reg_name").input("דני")
    at.text_input(key="reg_pass").input("mypass123")
    at.text_input(key="reg_confirm").input("mypass123")
    at.button(key="reg_submit").click().run()

    assert not at.exception
    assert at.session_state.get("logged_in") is True
    assert at.session_state.get("user_id") is not None
    assert at.session_state.get("user_name") == "דני"


def test_register_validation_mismatch(isolated_data_dir):
    at = AppTest.from_function(script_render_login, args=(str(isolated_data_dir),)).run()
    assert not at.exception

    at.text_input(key="reg_name").input("דני")
    at.text_input(key="reg_pass").input("mypass123")
    at.text_input(key="reg_confirm").input("differentpass")
    at.button(key="reg_submit").click().run()

    assert not at.exception
    assert any("הסיסמאות אינן תואמות" in err.value for err in at.error)
    assert not at.session_state.get("logged_in")


def test_render_user_badge_logged_in_and_logout(isolated_data_dir):
    at = AppTest.from_function(script_render_badge, args=(str(isolated_data_dir),)).run()
    assert not at.exception
    assert any("מחובר" in markdown.value for markdown in at.markdown)

    # Click logout button
    at.button(key="logout_btn").click().run()
    assert not at.exception
    assert "logged_in" not in at.session_state
    assert "user_id" not in at.session_state
    assert "user_name" not in at.session_state
