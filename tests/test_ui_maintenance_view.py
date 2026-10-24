"""Tests for screener.ui.maintenance_view module."""
from __future__ import annotations

import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

import screener.alerts
import screener.backup
import screener.portfolio
import screener.presets
import screener.support
import screener.whatsnew


def test_report_button_success(tmp_path, monkeypatch):
    rep_zip = tmp_path / "HalgOM-report-test.zip"
    rep_zip.write_text("dummy zip content", encoding="utf-8")

    def script_func(out_dir: str):
        from pathlib import Path
        import screener.support as support
        import screener.ui.maintenance_view as mv

        monkeypatch_path = Path(out_dir) / "HalgOM-report-test.zip"
        support.make_report = lambda ver: monkeypatch_path
        mv.render()

    at = AppTest.from_function(script_func, args=(str(tmp_path),)).run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Click report button
    report_btn = at.button(key="mt_report")
    assert report_btn is not None
    report_btn.click().run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert any("HalgOM-report-test.zip" in s.value for s in at.success)


def test_report_button_error(tmp_path):
    def script_func():
        import screener.support as support
        import screener.ui.maintenance_view as mv

        def raise_err(ver):
            raise RuntimeError("Report error")

        support.make_report = raise_err
        mv.render()

    at = AppTest.from_function(script_func).run()
    if at.exception:
        raise at.exception[0]

    report_btn = at.button(key="mt_report")
    report_btn.click().run()

    if at.exception:
        raise at.exception[0]
    # Check error note rendered
    assert any("אירעה שגיאה" in html.value for html in at.html)


def test_backup_and_restore(tmp_path, monkeypatch):
    p_presets = tmp_path / "presets.json"
    p_portfolios = tmp_path / "portfolios.json"
    p_alerts = tmp_path / "alerts.json"

    p_presets.write_text(json.dumps({"P1": {"filters": {"pe": "Under 15"}}}), encoding="utf-8")
    p_portfolios.write_text(json.dumps({}), encoding="utf-8")
    p_alerts.write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(screener.presets, "PRESETS_PATH", p_presets)
    monkeypatch.setattr(screener.portfolio, "PORTFOLIOS_PATH", p_portfolios)
    monkeypatch.setattr(screener.alerts, "ALERTS_PATH", p_alerts)

    target_dir = tmp_path / "backups"
    zip_path = screener.backup.make_backup(target_dir)

    def script_func(presets_p: str, zip_p: str, out_d: str):
        from pathlib import Path
        import screener.alerts as alerts
        import screener.backup as backup
        import screener.portfolio as portfolio
        import screener.presets as presets
        import screener.support as support
        import screener.ui.maintenance_view as mv

        presets.PRESETS_PATH = Path(presets_p)
        portfolio.PORTFOLIOS_PATH = Path(presets_p).parent / "portfolios.json"
        alerts.ALERTS_PATH = Path(presets_p).parent / "alerts.json"
        support.desktop_dirs = lambda: [Path(out_d)]

        mv.render()

    # Initial render
    at = AppTest.from_function(script_func, args=(str(p_presets), str(zip_path), str(target_dir))).run()
    if at.exception:
        raise at.exception[0]

    # Click backup button
    backup_btn = at.button(key="mt_backup")
    assert backup_btn is not None
    backup_btn.click().run()
    if at.exception:
        raise at.exception[0]
    assert any("גיבוי נשמר" in s.value for s in at.success)

    # Upload backup file
    file_uploader = at.file_uploader(key="mt_restore_file")
    assert file_uploader is not None
    # Upload zip bytes
    zip_bytes = zip_path.read_bytes()
    file_uploader.upload(zip_path.name, zip_bytes).run()

    if at.exception:
        raise at.exception[0]

    # Confirm restore button exists and radio exists
    radio = at.radio(key="mt_restore_mode")
    assert radio is not None

    confirm_btn = at.button(key="mt_restore_go")
    assert confirm_btn is not None

    # Clicking confirm restores backup
    confirm_btn.click().run()
    if at.exception:
        raise at.exception[0]
    assert any("השחזור הושלם בהצלחה" in s.value for s in at.success)


def test_show_whatsnew(tmp_path):
    seen_file = tmp_path / "whatsnew_seen.json"
    whatsnew_file = tmp_path / "whatsnew.json"

    whatsnew_file.write_text(json.dumps({
        "version": "1.0",
        "entries": [{"id": "v1.0-e1", "title": "כותרת חדשה", "text": "פירוט שינוי", "date": "2026-10-24"}]
    }), encoding="utf-8")

    def script_func(wn_path: str, s_path: str):
        from pathlib import Path
        import screener.whatsnew as whatsnew
        import screener.ui.maintenance_view as mv

        orig_wn = whatsnew._resolve_whatsnew_path
        orig_seen = whatsnew._resolve_seen_path
        whatsnew._resolve_whatsnew_path = lambda path=None: Path(wn_path) if path is None else orig_wn(path)
        whatsnew._resolve_seen_path = lambda path=None: Path(s_path) if path is None else orig_seen(path)
        try:
            mv.show_whatsnew()
        finally:
            whatsnew._resolve_whatsnew_path = orig_wn
            whatsnew._resolve_seen_path = orig_seen

    # Create seen_file formatted as json dict {"seen": []}
    seen_file.write_text(json.dumps({"seen": []}), encoding="utf-8")

    at = AppTest.from_function(script_func, args=(str(whatsnew_file), str(seen_file))).run()
    if at.exception:
        raise at.exception[0]

    # ACK button inside dialog
    ack_btn = at.button(key="mt_whatsnew_ack")
    assert ack_btn is not None
    ack_btn.click().run()

    if at.exception:
        raise at.exception[0]

    # Verify mark_seen was called and written
    seen_data = json.loads(seen_file.read_text(encoding="utf-8"))
    assert "v1.0-e1" in seen_data.get("seen", [])

    # A second run shows nothing and doesn't crash
    at2 = AppTest.from_function(script_func, args=(str(whatsnew_file), str(seen_file))).run()
    if at2.exception:
        raise at2.exception[0]
    # No button key="mt_whatsnew_ack" because ss["mt_whatsnew_shown"] is True
    assert len([b for b in at2.button if b.key == "mt_whatsnew_ack"]) == 0


def test_render_error_recovery():
    def script_func():
        import screener.whatsnew as whatsnew
        import screener.ui.maintenance_view as mv

        def bad_version(*a, **k):
            raise RuntimeError("Version crash")

        orig_version = whatsnew.version_label
        whatsnew.version_label = bad_version
        try:
            mv.render()
        finally:
            whatsnew.version_label = orig_version

    at = AppTest.from_function(script_func).run()
    # render() must handle the top-level exception and display hebrew_note
    assert not at.exception
    assert any("אירעה שגיאה בטעינת פאנל התחזוקה" in html.value for html in at.html)
