"""Tests verifying direct download buttons in maintenance view."""
from __future__ import annotations

import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

import screener.alerts
import screener.backup
import screener.portfolio
import screener.presets
import screener.support


def test_report_download_button(tmp_path, monkeypatch):
    rep_zip = tmp_path / "HalgOM-report-test.zip"
    dummy_bytes = b"dummy report zip content"
    rep_zip.write_bytes(dummy_bytes)

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

    # Before clicking report generation, download button should not exist
    dl_btns = [b for b in at.download_button if b.key == "mt_dl_report"]
    assert len(dl_btns) == 0

    # Click report button
    report_btn = at.button(key="mt_report")
    assert report_btn is not None
    report_btn.click().run()

    if at.exception:
        raise at.exception[0]

    # After clicking, download button key="mt_dl_report" should exist
    dl_btns = [b for b in at.download_button if b.key == "mt_dl_report"]
    assert len(dl_btns) == 1
    dl_btn = dl_btns[0]
    assert dl_btn.label == "הורד דוח תקלה למחשב"


def test_backup_download_button(tmp_path, monkeypatch):
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

    at = AppTest.from_function(script_func, args=(str(p_presets), str(zip_path), str(target_dir))).run()
    if at.exception:
        raise at.exception[0]

    # Before clicking backup, download button should not exist
    dl_btns = [b for b in at.download_button if b.key == "mt_dl_backup"]
    assert len(dl_btns) == 0

    # Click backup button
    backup_btn = at.button(key="mt_backup")
    assert backup_btn is not None
    backup_btn.click().run()

    if at.exception:
        raise at.exception[0]

    # After clicking, download button key="mt_dl_backup" should exist
    dl_btns = [b for b in at.download_button if b.key == "mt_dl_backup"]
    assert len(dl_btns) == 1
    dl_btn = dl_btns[0]
    assert dl_btn.label == "הורד קובץ גיבוי למחשב"
