"""
Tests for screener/userfiles.py storage helper: safe reading, entry isolation,
damaged file renaming, OSError protection, backup creation, process locking, and UI toast handling.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from screener import presets as P
from screener import portfolio as PF
from screener import app as APP


def test_unknown_keys_and_bad_entries_isolated(tmp_path):
    preset_file = tmp_path / "presets.json"
    raw_presets = {
        "מסך 1": {"filters": {"pe": "Under 15"}, "view": "Valuation"},
        "מסך 2": {"filters": {"roe": "Over 10%"}, "view": "Financial", "future_unknown_field": "test"},
        "מסך זבל": "not a dict",
    }
    preset_file.write_text(json.dumps(raw_presets, ensure_ascii=False), encoding="utf-8")

    loaded = P.load_user(preset_file)
    assert set(loaded.keys()) == {"מסך 1", "מסך 2"}
    assert loaded["מסך 1"].filters == {"pe": "Under 15"}
    assert loaded["מסך 2"].filters == {"roe": "Over 10%"}

    new_p = P.Preset(filters={"rsi": "Oversold (40)"})
    P.save_user("מסך 3", new_p, path=preset_file)

    reloaded = P.load_user(preset_file)
    assert set(reloaded.keys()) == {"מסך 1", "מסך 2", "מסך 3"}


def test_portfolio_extra_field_preserved(tmp_path):
    pf_file = tmp_path / "portfolios.json"
    raw_portfolios = {
        "תיק 1": {
            "name": "תיק 1",
            "kind": "watchlist",
            "rows": [{"ticker": "AAPL", "date": "2026-10-19", "price": 250.0}],
            "extra_field": "extra_value",
        }
    }
    pf_file.write_text(json.dumps(raw_portfolios, ensure_ascii=False), encoding="utf-8")

    loaded = PF.load_all(pf_file)
    assert "תיק 1" in loaded
    assert loaded["תיק 1"].rows[0]["ticker"] == "AAPL"

    w2 = PF.PList("מעקב 2", PF.WATCHLIST, [{"ticker": "MSFT"}])
    PF.save(w2, path=pf_file)

    reloaded = PF.load_all(pf_file)
    assert set(reloaded.keys()) == {"תיק 1", "מעקב 2"}


def test_damaged_file_renamed(tmp_path):
    preset_file = tmp_path / "presets.json"
    preset_file.write_text("{invalid json", encoding="utf-8")

    loaded = P.load_user(preset_file)
    assert loaded == {}
    assert not preset_file.exists()

    bad_files = list(tmp_path.glob("presets.json.bad-*"))
    assert len(bad_files) == 1
    assert bad_files[0].read_text(encoding="utf-8") == "{invalid json"


def test_non_dict_root_json_renamed(tmp_path):
    pf_file = tmp_path / "portfolios.json"
    pf_file.write_text("[1, 2, 3]", encoding="utf-8")

    loaded = PF.load_all(pf_file)
    assert loaded == {}
    assert not pf_file.exists()

    bad_files = list(tmp_path.glob("portfolios.json.bad-*"))
    assert len(bad_files) == 1
    assert bad_files[0].read_text(encoding="utf-8") == "[1, 2, 3]"


def test_oserror_on_read_prevents_overwrite(tmp_path, monkeypatch):
    preset_file = tmp_path / "presets.json"
    original_data = {"מסך 1": {"filters": {"pe": "Under 15"}, "view": "Valuation"}}
    preset_file.write_text(json.dumps(original_data, ensure_ascii=False), encoding="utf-8")

    def mock_read_text(self, *args, **kwargs):
        if self.name == "presets.json":
            raise PermissionError("File locked by antivirus")
        return original_read(self, *args, **kwargs)

    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with pytest.raises(PermissionError):
        P.save_user("מסך 2", P.Preset(filters={}), path=preset_file)

    monkeypatch.undo()
    reloaded = P.load_user(preset_file)
    assert set(reloaded.keys()) == {"מסך 1"}


def test_bak_file_created_on_save(tmp_path):
    preset_file = tmp_path / "presets.json"
    p1 = P.Preset(filters={"pe": "Under 15"})
    P.save_user("מסך 1", p1, path=preset_file)

    bak_file = tmp_path / "presets.json.bak"
    assert not bak_file.exists()

    p2 = P.Preset(filters={"roe": "Over 10%"})
    P.save_user("מסך 2", p2, path=preset_file)

    assert bak_file.exists()
    bak_data = json.loads(bak_file.read_text(encoding="utf-8"))
    assert set(bak_data.keys()) == {"מסך 1"}

    current_data = json.loads(preset_file.read_text(encoding="utf-8"))
    assert set(current_data.keys()) == {"מסך 1", "מסך 2"}


def test_concurrent_writes(tmp_path):
    preset_file = tmp_path / "presets.json"
    errors = []

    def worker(i: int):
        try:
            p = P.Preset(filters={"rsi": f"Option_{i}"})
            P.save_user(f"מסך_{i}", p, path=preset_file)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    loaded = P.load_user(preset_file)
    assert len(loaded) == 10


def test_app_ui_toast_on_save_error(tmp_path, monkeypatch):
    preset_file = tmp_path / "presets.json"

    def mock_save(*args, **kwargs):
        raise PermissionError("Write locked")

    monkeypatch.setattr(P, "save_user", mock_save)

    toasts = []
    monkeypatch.setattr(st, "toast", lambda msg, icon=None: toasts.append((msg, icon)))

    monkeypatch.setattr(APP, "ss", {"save_name": "מסך ניסיון", "view": "Overview",
                                   "sort_col": "pe", "sort_asc": True})

    APP.on_save()

    assert len(toasts) == 1
    assert toasts[0][0] == "לא ניתן לשמור כרגע, נסו שוב בעוד רגע"
