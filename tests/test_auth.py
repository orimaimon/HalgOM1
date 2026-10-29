"""
Unit tests for screener.auth module.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from screener.data import store
from screener.auth import (
    User,
    list_users,
    get_user,
    create_user,
    verify_password,
    change_password,
    get_user_data_dir,
    create_session,
    validate_session,
    revoke_session,
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_default_users_initialization(isolated_data_dir):
    users_file = isolated_data_dir / "users.json"
    assert not users_file.exists()

    users = list_users()
    assert users_file.exists()
    assert len(users) == 2

    user_ids = {u.id for u in users}
    assert "ori" in user_ids
    assert "haim" in user_ids

    ori = get_user("ori")
    assert ori is not None
    assert ori.display_name == "אורי"

    haim = get_user("haim")
    assert haim is not None
    assert haim.display_name == "חיים"

    assert verify_password("ori", "halgom123") is True
    assert verify_password("haim", "halgom123") is True


def test_verify_password(isolated_data_dir):
    assert verify_password("ori", "halgom123") is True
    assert verify_password("ori", "wrongpass") is False
    assert verify_password("nonexistent", "halgom123") is False


def test_create_user(isolated_data_dir):
    user = create_user("דן", "secret123")
    assert isinstance(user, User)
    assert user.display_name == "דן"
    assert verify_password(user.id, "secret123") is True

    fetched = get_user(user.id)
    assert fetched is not None
    assert fetched.display_name == "דן"

    # Creating another user with duplicate or non-ascii display name handles unique IDs
    user2 = create_user("דן", "another123")
    assert user2.id != user.id
    assert verify_password(user2.id, "another123") is True


def test_create_user_validation(isolated_data_dir):
    with pytest.raises(ValueError, match="Display name cannot be empty"):
        create_user("   ", "pass123")

    with pytest.raises(ValueError, match="Password cannot be empty"):
        create_user("User", "")


def test_change_password(isolated_data_dir):
    assert change_password("ori", "halgom123", "newpass456") is True
    assert verify_password("ori", "newpass456") is True
    assert verify_password("ori", "halgom123") is False

    # Changing with wrong old password fails
    assert change_password("ori", "wrongpass", "anotherpass") is False
    assert verify_password("ori", "newpass456") is True


def test_get_user_data_dir(isolated_data_dir):
    user_dir = get_user_data_dir("ori")
    expected = isolated_data_dir / "users" / "ori"
    assert user_dir == expected
    assert user_dir.exists()
    assert user_dir.is_dir()


def test_session_lifecycle(isolated_data_dir):
    token = create_session("ori", days=30)
    assert token is not None
    assert validate_session(token) == "ori"

    revoke_session(token)
    assert validate_session(token) is None


def test_session_expiration(isolated_data_dir):
    token = create_session("ori", days=-1)
    assert validate_session(token) is None
