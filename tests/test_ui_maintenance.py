"""Tests for DuckDB storage health check in maintenance_view."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_health_check_ok():
    def script_func():
        import screener.data.duck_engine as duck_engine
        import screener.ui.maintenance_view as mv

        duck_engine.health_check = lambda d: {
            "status": "ok",
            "duckdb_version": "2.0.0",
            "total_files": 5,
            "valid_files": 5,
            "corrupted_files": 0,
            "files": [],
            "error": None,
        }
        mv.render()

    at = AppTest.from_function(script_func).run()
    if at.exception:
        raise at.exception[0]

    btn = at.button(key="mt_health_check")
    assert btn is not None
    btn.click().run()

    if at.exception:
        raise at.exception[0]

    assert any("כל קובצי הנתונים תקינים" in s.value for s in at.success)
    assert any("5 קבצים" in s.value for s in at.success)


def test_health_check_degraded():
    def script_func():
        import screener.data.duck_engine as duck_engine
        import screener.ui.maintenance_view as mv

        duck_engine.health_check = lambda d: {
            "status": "degraded",
            "duckdb_version": "2.0.0",
            "total_files": 5,
            "valid_files": 4,
            "corrupted_files": 1,
            "files": [
                {"valid": True, "path": "file1.parquet", "rows": 10, "columns": {}, "error": None},
                {"valid": False, "path": "corrupted_data.parquet", "rows": 0, "columns": {}, "error": "Corrupted header"},
            ],
            "error": None,
        }
        mv.render()

    at = AppTest.from_function(script_func).run()
    if at.exception:
        raise at.exception[0]

    btn = at.button(key="mt_health_check")
    assert btn is not None
    btn.click().run()

    if at.exception:
        raise at.exception[0]

    assert any("corrupted_data.parquet" in w.value for w in at.warning)
    assert any("Corrupted header" in w.value for w in at.warning)


def test_health_check_exception():
    def script_func():
        import screener.data.duck_engine as duck_engine
        import screener.ui.maintenance_view as mv

        def bad_health_check(d):
            raise RuntimeError("Health check unexpected error")

        duck_engine.health_check = bad_health_check
        mv.render()

    at = AppTest.from_function(script_func).run()
    if at.exception:
        raise at.exception[0]

    btn = at.button(key="mt_health_check")
    assert btn is not None
    btn.click().run()

    if at.exception:
        raise at.exception[0]

    assert any("אירעה שגיאה" in html.value for html in at.html)
