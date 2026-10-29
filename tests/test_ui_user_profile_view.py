"""
Unit tests for screener.ui.user_profile_view module.
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from screener.auth import list_users, verify_password
from screener.data import store


def script_render_user_profile(data_dir: str, user_id: str = "ori"):
    from pathlib import Path
    import screener.data.store as store_mod
    import screener.ui.user_profile_view as upv

    store_mod.DATA_DIR = Path(data_dir)
    upv.render_user_profile(user_id)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_user_profile_initial_render(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir),)).run()
    assert not at.exception
    headers = [s.value for s in at.subheader]
    assert "שינוי סיסמה" in headers
    assert "הוספת משתמש חדש" in headers


def test_change_password_success(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_old_pass").input("halgom123")
    at.text_input(key="up_new_pass").input("newsecret123")
    at.text_input(key="up_confirm_pass").input("newsecret123")
    at.button(key="up_change_pass_btn").click().run()

    assert not at.exception
    assert any("הסיסמה שונתה בהצלחה" in s.value for s in at.success)
    assert verify_password("ori", "newsecret123")
    assert not verify_password("ori", "halgom123")


def test_change_password_mismatch(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_old_pass").input("halgom123")
    at.text_input(key="up_new_pass").input("newsecret123")
    at.text_input(key="up_confirm_pass").input("different123")
    at.button(key="up_change_pass_btn").click().run()

    assert not at.exception
    assert any("הסיסמאות אינן תואמות" in err.value for err in at.error)
    assert verify_password("ori", "halgom123")


def test_change_password_incorrect_old(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_old_pass").input("wrongpassword")
    at.text_input(key="up_new_pass").input("newsecret123")
    at.text_input(key="up_confirm_pass").input("newsecret123")
    at.button(key="up_change_pass_btn").click().run()

    assert not at.exception
    assert any("הסיסמה הנוכחית אינה נכונה" in err.value for err in at.error)
    assert verify_password("ori", "halgom123")


def test_create_user_success(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_new_name").input("דני")
    at.text_input(key="up_new_user_pass").input("daniPass123")
    at.text_input(key="up_confirm_user_pass").input("daniPass123")
    at.button(key="up_add_user_btn").click().run()

    assert not at.exception
    assert any("המשתמש דני נוצר בהצלחה" in s.value for s in at.success)

    users = list_users()
    created = [u for u in users if u.display_name == "דני"]
    assert len(created) == 1
    user = created[0]
    assert verify_password(user.id, "daniPass123")


def test_create_user_validation_mismatch(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_new_name").input("יוסי")
    at.text_input(key="up_new_user_pass").input("pass123")
    at.text_input(key="up_confirm_user_pass").input("pass456")
    at.button(key="up_add_user_btn").click().run()

    assert not at.exception
    assert any("הסיסמאות אינן תואמות" in err.value for err in at.error)
    users = list_users()
    assert not any(u.display_name == "יוסי" for u in users)


def test_create_user_empty_name(isolated_data_dir):
    at = AppTest.from_function(script_render_user_profile, args=(str(isolated_data_dir), "ori")).run()
    assert not at.exception

    at.text_input(key="up_new_name").input("   ")
    at.text_input(key="up_new_user_pass").input("pass123")
    at.text_input(key="up_confirm_user_pass").input("pass123")
    at.button(key="up_add_user_btn").click().run()

    assert not at.exception
    assert any("יש להזין שם משתמש" in err.value for err in at.error)
