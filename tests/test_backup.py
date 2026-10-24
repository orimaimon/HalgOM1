"""
Tests for screener.backup module.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import screener.alerts
import screener.backup
import screener.portfolio
import screener.presets


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    p_presets = tmp_path / "presets.json"
    p_portfolios = tmp_path / "portfolios.json"
    p_alerts = tmp_path / "alerts.json"

    monkeypatch.setattr(screener.presets, "PRESETS_PATH", p_presets)
    monkeypatch.setattr(screener.portfolio, "PORTFOLIOS_PATH", p_portfolios)
    monkeypatch.setattr(screener.alerts, "ALERTS_PATH", p_alerts)

    return {
        "presets": p_presets,
        "portfolios": p_portfolios,
        "alerts": p_alerts,
    }


def test_round_trip(tmp_path, patch_paths):
    # Setup sample user data
    p1 = screener.presets.Preset(filters={"pe": "Under 15"}, view="Valuation", note="Test Preset")
    screener.presets.save_user("Preset1", p1)

    pl1 = screener.portfolio.PList("List1", "watchlist", [{"ticker": "AAPL", "date": "2025-01-01", "price": 150.0}])
    screener.portfolio.save(pl1)

    a1 = screener.alerts.Alert(id="a1", ticker="AAPL", kind="price_below", value=100.0, note="Alert 1")
    screener.alerts.add(a1)

    # Make backup
    backup_dir = tmp_path / "backups"
    zip_path = screener.backup.make_backup(backup_dir)

    assert zip_path.exists()

    # Verify inspection
    inspection = screener.backup.inspect(zip_path)
    assert inspection["presets.json"]["ok"] is True
    assert inspection["presets.json"]["items"] == 1
    assert inspection["portfolios.json"]["ok"] is True
    assert inspection["portfolios.json"]["items"] == 1
    assert inspection["alerts.json"]["ok"] is True
    assert inspection["alerts.json"]["items"] == 1

    # Delete local files
    patch_paths["presets"].unlink()
    patch_paths["portfolios"].unlink()
    patch_paths["alerts"].unlink()

    # Restore backup
    res = screener.backup.restore(zip_path, mode="merge")
    assert res["presets.json"]["added"] == 1
    assert res["portfolios.json"]["added"] == 1
    assert res["alerts.json"]["added"] == 1

    # Verify restored data
    loaded_presets = screener.presets.load_user()
    assert "Preset1" in loaded_presets
    assert loaded_presets["Preset1"].filters == {"pe": "Under 15"}

    loaded_portfolios = screener.portfolio.load_all()
    assert "List1" in loaded_portfolios
    assert loaded_portfolios["List1"].rows[0]["ticker"] == "AAPL"

    loaded_alerts = screener.alerts.load()
    assert "a1" in loaded_alerts
    assert loaded_alerts["a1"].ticker == "AAPL"


def test_merge_mode(tmp_path, patch_paths):
    # Setup initial preset
    p1 = screener.presets.Preset(filters={"pe": "Under 15"}, note="Original")
    screener.presets.save_user("Preset1", p1)

    backup_dir = tmp_path / "backups"
    zip_path = screener.backup.make_backup(backup_dir)

    # Modify existing preset and add a local-only preset
    p1_mod = screener.presets.Preset(filters={"pe": "Under 20"}, note="Local Modified")
    screener.presets.save_user("Preset1", p1_mod)

    p2 = screener.presets.Preset(filters={"roe": "Over 10%"}, note="Local New")
    screener.presets.save_user("Preset2", p2)

    # Restore in merge mode (should keep existing Preset1 & Preset2)
    res = screener.backup.restore(zip_path, mode="merge")
    assert res["presets.json"]["added"] == 0
    assert res["presets.json"]["skipped"] == 1

    loaded = screener.presets.load_user()
    assert loaded["Preset1"].note == "Local Modified"
    assert "Preset2" in loaded


def test_replace_mode(tmp_path, patch_paths):
    p1 = screener.presets.Preset(filters={"pe": "Under 15"}, note="Original")
    screener.presets.save_user("Preset1", p1)

    backup_dir = tmp_path / "backups"
    zip_path = screener.backup.make_backup(backup_dir)

    p2 = screener.presets.Preset(filters={"roe": "Over 10%"}, note="New Local")
    screener.presets.save_user("Preset2", p2)

    # Restore in replace mode
    res = screener.backup.restore(zip_path, mode="replace")
    assert res["presets.json"]["replaced"] is True
    assert res["presets.json"]["added"] == 1

    loaded = screener.presets.load_user()
    assert "Preset1" in loaded
    assert "Preset2" not in loaded

    # Verify .before-restore-* copy exists
    bak_files = list(patch_paths["presets"].parent.glob("presets.json.before-restore-*"))
    assert len(bak_files) == 1


def test_damaged_json(tmp_path, patch_paths):
    p1 = screener.presets.Preset(filters={"pe": "Under 15"})
    screener.presets.save_user("Preset1", p1)

    zip_path = tmp_path / "damaged.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"app": "HalgOM Screener", "format": 1}))
        zf.writestr("presets.json", "{ invalid json ...")

    inspection = screener.backup.inspect(zip_path)
    assert inspection["presets.json"]["ok"] is False
    assert inspection["presets.json"]["problem"] is not None

    # Restore must refuse changes
    res = screener.backup.restore(zip_path)
    assert res["presets.json"]["ok"] is False

    # Original preset should remain unchanged
    loaded = screener.presets.load_user()
    assert "Preset1" in loaded


def test_missing_manifest_or_wrong_format_or_evil_zip(tmp_path, patch_paths):
    # No manifest
    z1 = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(z1, "w") as zf:
        zf.writestr("presets.json", "{}")

    insp1 = screener.backup.inspect(z1)
    assert insp1["_zip"]["ok"] is False

    # Wrong format
    z2 = tmp_path / "wrong_format.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"app": "HalgOM Screener", "format": 99}))

    insp2 = screener.backup.inspect(z2)
    assert insp2["_zip"]["ok"] is False

    # Evil path traversal (zip slip)
    z3 = tmp_path / "evil.zip"
    with zipfile.ZipFile(z3, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"app": "HalgOM Screener", "format": 1}))
        zf.writestr("../evil.json", "{}")

    insp3 = screener.backup.inspect(z3)
    assert insp3["_zip"]["ok"] is False


def test_missing_source_file(tmp_path, patch_paths):
    # Save only presets
    p1 = screener.presets.Preset(filters={"pe": "Under 15"})
    screener.presets.save_user("Preset1", p1)

    zip_path = screener.backup.make_backup(tmp_path / "backups")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "presets.json" in names
        assert "portfolios.json" not in names
        assert "alerts.json" not in names

    # Create alerts locally
    a1 = screener.alerts.Alert(id="a1", ticker="AAPL")
    screener.alerts.add(a1)

    # Restore from zip that lacks alerts.json -> local alerts must remain untouched
    screener.backup.restore(zip_path, mode="replace")
    assert "a1" in screener.alerts.load()


def test_zip_contents_strict(tmp_path, patch_paths):
    screener.presets.save_user("Preset1", screener.presets.Preset(filters={}))
    zip_path = screener.backup.make_backup(tmp_path / "backups")

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert set(namelist) == {"manifest.json", "presets.json"}


def test_restore_changes_nothing_when_the_undo_copy_fails(tmp_path, patch_paths, monkeypatch):
    import json
    import shutil
    from screener import backup
    patch_paths["presets"].write_text(json.dumps({"Mine": {"filters": {"price": "Over $5"}}}), encoding="utf-8")
    zip_path = backup.make_backup(tmp_path / "out")
    patch_paths["presets"].write_text(json.dumps({"Newer": {"filters": {"price": "Over $10"}}}), encoding="utf-8")
    before = patch_paths["presets"].read_text(encoding="utf-8")

    def broken_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", broken_copy)
    res = backup.restore(zip_path, mode="replace")
    assert res["presets.json"]["ok"] is False and "לא שוחזר" in res["presets.json"]["problem"]
    assert patch_paths["presets"].read_text(encoding="utf-8") == before
