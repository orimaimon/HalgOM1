from pathlib import Path
import pytest
from screener.data import store
from screener.userfiles import get_user_file_path, migrate_legacy_file


def test_get_user_file_path_with_user_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "portfolios.json"
    user_id = "ori"

    path = get_user_file_path(filename, user_id=user_id)

    expected = tmp_path / "users" / user_id / filename
    assert path == expected
    assert path.parent.exists()
    assert path.parent.is_dir()


def test_get_user_file_path_with_none(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "portfolios.json"

    path = get_user_file_path(filename, user_id=None)

    expected = tmp_path / filename
    assert path == expected
    assert path.parent.exists()


def test_get_user_file_path_with_empty_string(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "portfolios.json"

    path = get_user_file_path(filename, user_id="")

    expected = tmp_path / filename
    assert path == expected
    assert path.parent.exists()


def test_migrate_legacy_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "presets.json"
    user_id = "haim"

    legacy_file = tmp_path / filename
    legacy_content = '{"preset1": {"name": "preset1"}}'
    legacy_file.write_text(legacy_content, encoding="utf-8")

    user_file = tmp_path / "users" / user_id / filename
    assert not user_file.exists()

    # Migration copies legacy file and returns True
    migrated = migrate_legacy_file(filename, user_id)
    assert migrated is True
    assert user_file.exists()
    assert user_file.read_text(encoding="utf-8") == legacy_content

    # Second migration attempt returns False because user file already exists
    migrated_again = migrate_legacy_file(filename, user_id)
    assert migrated_again is False


def test_migrate_legacy_file_no_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "alerts.json"
    user_id = "haim"

    migrated = migrate_legacy_file(filename, user_id)
    assert migrated is False
    user_file = tmp_path / "users" / user_id / filename
    assert not user_file.exists()


def test_migrate_legacy_file_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    filename = "portfolios.json"
    user_id = "ori"

    legacy_file = tmp_path / filename
    legacy_file.write_text("legacy_content", encoding="utf-8")

    user_file = tmp_path / "users" / user_id / filename
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text("user_content", encoding="utf-8")

    migrated = migrate_legacy_file(filename, user_id)
    assert migrated is False
    assert user_file.read_text(encoding="utf-8") == "user_content"
