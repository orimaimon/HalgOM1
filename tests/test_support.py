"""
Tests for screener.support (make_report, data_usage, log_daily_usage, git sha resolution).
"""
import zipfile
from pathlib import Path
import pytest
from screener import support
from screener.data import store


def test_get_git_sha(tmp_path: Path):
    # Case 1: No .git -> "—"
    assert support.get_git_sha(tmp_path) == "—"

    # Case 2: .git/HEAD -> ref file
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("1234567890abcdef1234567890abcdef12345678\n", encoding="utf-8")

    assert support.get_git_sha(tmp_path) == "1234567890abcdef1234567890abcdef12345678"

    # Case 3: .git/HEAD -> ref in packed-refs
    (refs_dir / "main").unlink()
    packed_refs = (
        "# pack-refs with: peeled fully-peeled sorted \n"
        "9876543210fedcba9876543210fedcba98765432 refs/heads/main\n"
    )
    (git_dir / "packed-refs").write_text(packed_refs, encoding="utf-8")

    assert support.get_git_sha(tmp_path) == "9876543210fedcba9876543210fedcba98765432"

    # Case 4: Detached HEAD (SHA directly in HEAD)
    (git_dir / "HEAD").write_text("abcdef1234567890abcdef1234567890abcdef12\n", encoding="utf-8")
    assert support.get_git_sha(tmp_path) == "abcdef1234567890abcdef1234567890abcdef12"


def test_desktop_dirs_fallback(tmp_path: Path, monkeypatch):
    # Mock USERPROFILE
    fake_home = tmp_path / "user"
    fake_home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    data_dir = tmp_path / "_data"
    monkeypatch.setattr(support, "get_data_dir", lambda: data_dir)

    # When no desktop folder exists, make_report falls back to _data/reports/
    report_zip = support.make_report("1.0.0")
    assert report_zip.parent == data_dir / "reports"
    assert report_zip.exists()

    # Create the first desktop directory
    desktop1 = fake_home / "Desktop"
    desktop1.mkdir()

    report_zip2 = support.make_report("1.0.0")
    assert report_zip2.parent == desktop1
    assert report_zip2.exists()


def test_make_report_contents_and_secrets_exclusion(tmp_path: Path, monkeypatch):
    # Setup directories
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    data_dir = repo_root / "screener" / "_data"
    logs_dir = data_dir / "logs"
    snaps_dir = data_dir / "snapshots"
    logs_dir.mkdir(parents=True)
    snaps_dir.mkdir(parents=True)

    monkeypatch.setattr(support, "get_repo_root", lambda: repo_root)
    monkeypatch.setattr(support, "get_data_dir", lambda: data_dir)

    # Create logs
    (logs_dir / "screener.log").write_text("screener log content\n", encoding="utf-8")
    (logs_dir / "update.log").write_text("update log content\n", encoding="utf-8")
    (logs_dir / "other.log").write_text("other log content\n", encoding="utf-8")

    # Create snapshots (test newest snapshot selection)
    (snaps_dir / "2026-01-01.json").write_text('{"snapshot": "old"}', encoding="utf-8")
    (snaps_dir / "2026-01-02.json").write_text('{"snapshot": "newest"}', encoding="utf-8")

    # Create secrets and sensitive files outside or inside data
    SECRET_MARKER = "SUPER_SECRET_KEY_12345"

    # Deploy key next to repo
    (repo_root / "deploy_key").write_text(SECRET_MARKER, encoding="utf-8")
    (repo_root / "deploy_key.txt").write_text(SECRET_MARKER, encoding="utf-8")

    # Secret in .halgom
    halgom_dir = fake_home / ".halgom"
    halgom_dir.mkdir()
    (halgom_dir / "deploy_key").write_text(SECRET_MARKER, encoding="utf-8")

    # User files in _data with secret marker
    (data_dir / "portfolios.json").write_text(f'{{"secret": "{SECRET_MARKER}"}}', encoding="utf-8")
    (data_dir / "presets.json").write_text(f'{{"secret": "{SECRET_MARKER}"}}', encoding="utf-8")
    (data_dir / "alerts.json").write_text(f'{{"secret": "{SECRET_MARKER}"}}', encoding="utf-8")

    dest = tmp_path / "output"
    zip_path = support.make_report("1.2.3", dest=dest)

    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_names = set(zf.namelist())

        # Verify expected files exist in zip
        assert "logs/screener.log" in file_names
        assert "logs/update.log" in file_names
        assert "logs/other.log" not in file_names  # only screener.log* and update.log*
        assert "info.json" in file_names
        assert "files.txt" in file_names
        assert "snapshot.json" in file_names

        # Verify snapshot.json content is from newest snapshot
        snap_content = zf.read("snapshot.json").decode("utf-8")
        assert '{"snapshot": "newest"}' in snap_content

        # Verify info.json
        info_content = zf.read("info.json").decode("utf-8")
        assert '"version": "1.2.3"' in info_content
        assert '"git_sha": "—"' in info_content

        # Verify files.txt contains relative paths and sizes, but not content
        files_txt = zf.read("files.txt").decode("utf-8")
        assert "portfolios.json" in files_txt
        assert "presets.json" in files_txt
        assert "alerts.json" in files_txt
        assert "logs/screener.log" in files_txt

        # Strictly check that secret marker string NEVER appears in ANY zip file content or filename
        for name in zf.namelist():
            assert "deploy_key" not in name
            assert ".halgom" not in name
            content_bytes = zf.read(name)
            assert SECRET_MARKER.encode("utf-8") not in content_bytes


def test_data_usage_and_log_daily_usage(tmp_path: Path, monkeypatch, caplog):
    caplog.set_level("INFO")
    data_dir = tmp_path / "_data"
    monkeypatch.setattr(support, "get_data_dir", lambda: data_dir)

    # Create dummy data folders
    snaps = data_dir / "snapshots"
    snaps.mkdir(parents=True)
    (snaps / "s1.bin").write_bytes(b"x" * 1024 * 1024)  # 1MB

    hist = data_dir / "history"
    hist.mkdir(parents=True)
    (hist / "h1.bin").write_bytes(b"y" * 512 * 1024)   # 0.5MB

    # Top-level file in _data
    (data_dir / "root.txt").write_bytes(b"z" * 100)

    usage = support.data_usage()
    assert usage["snapshots"]["bytes"] == 1024 * 1024
    assert usage["snapshots"]["files"] == 1
    assert usage["history"]["bytes"] == 512 * 1024
    assert usage["history"]["files"] == 1
    assert usage["total"]["files"] == 3
    assert usage["total"]["bytes"] == 1024 * 1024 + 512 * 1024 + 100

    # Test log_daily_usage
    caplog.clear()
    assert support.log_daily_usage() is True
    assert "data usage:" in caplog.text
    assert "snapshots 1.0 MB (1 files)" in caplog.text

    # Call again same day -> returns False and does not log
    caplog.clear()
    assert support.log_daily_usage() is False
    assert caplog.text == ""

    # Simulate next day by editing _data/usage_logged.txt
    (data_dir / "usage_logged.txt").write_text("1999-01-01", encoding="utf-8")
    assert support.log_daily_usage() is True
    assert "data usage:" in caplog.text
