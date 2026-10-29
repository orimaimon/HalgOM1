from __future__ import annotations

import pytest

import screener.presets as presets


@pytest.fixture(autouse=True)
def _setup_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "DATA_DIR", tmp_path)


def test_user_preset_isolation():
    preset_a = presets.Preset(name="Custom A", filters={"pe": "Under 15"}, note="User A preset")
    preset_b = presets.Preset(name="Custom B", filters={"roe": "Over 10%"}, note="User B preset")

    # User A saves a preset
    presets.save(preset_a, user_id="userA")

    # User A sees Custom A
    presets_a = presets.load_all(user_id="userA")
    assert "Custom A" in presets_a
    assert presets_a["Custom A"].filters == {"pe": "Under 15"}

    # User B does NOT see Custom A
    presets_b = presets.load_all(user_id="userB")
    assert "Custom A" not in presets_b

    # User B saves Custom B
    presets.save(preset_b, user_id="userB")

    # User A does NOT see Custom B
    presets_a_updated = presets.load_all(user_id="userA")
    assert "Custom B" not in presets_a_updated

    # User B sees Custom B
    presets_b_updated = presets.load_all(user_id="userB")
    assert "Custom B" in presets_b_updated


def test_builtin_presets_visible_to_all_users():
    presets_a = presets.load_all(user_id="userA")
    presets_b = presets.load_all(user_id="userB")
    presets_global = presets.load_all(user_id=None)

    for builtin_name in presets.BUILTIN_PRESETS:
        assert builtin_name in presets_a
        assert builtin_name in presets_b
        assert builtin_name in presets_global


def test_delete_user_preset_isolation():
    preset_a = presets.Preset(name="Shared Name", filters={"pe": "Under 15"})
    preset_b = presets.Preset(name="Shared Name", filters={"pe": "Under 20"})

    presets.save(preset_a, user_id="userA")
    presets.save(preset_b, user_id="userB")

    # User A deletes "Shared Name"
    presets.delete("Shared Name", user_id="userA")

    # User A no longer has "Shared Name"
    presets_a = presets.load_all(user_id="userA")
    assert "Shared Name" not in presets_a

    # User B still has "Shared Name"
    presets_b = presets.load_all(user_id="userB")
    assert "Shared Name" in presets_b
    assert presets_b["Shared Name"].filters == {"pe": "Under 20"}
