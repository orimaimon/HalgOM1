"""
Backup and restore module for user data (saved screens, watchlists/portfolios, alerts).

Creates single-file ZIP archives containing presets.json, portfolios.json, alerts.json,
and manifest.json. Supports inspection, merge restore, and replace restore with automatic
pre-restore backups.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import screener.alerts
import screener.portfolio
import screener.presets
import screener.userfiles

log = logging.getLogger(__name__)

MESSAGES: dict[str, str] = {
    "not_a_backup": "הקובץ אינו גיבוי של HalgOM",
    "zip_slip": "קובץ הגיבוי מכיל נתיבים לא חוקיים",
    "corrupt_file": "הקובץ {name} לא תקין או מכיל נתונים פגומים",
    "unknown_file": "הקובץ {name} אינו מוכר",
}

ALLOWED_FILES = ("presets.json", "portfolios.json", "alerts.json")


def _get_target_specs() -> dict[str, tuple[Path, type]]:
    """Resolve target file paths and their dataclasses dynamically at call time."""
    return {
        "presets.json": (screener.presets.PRESETS_PATH, screener.presets.Preset),
        "portfolios.json": (screener.portfolio.PORTFOLIOS_PATH, screener.portfolio.PList),
        "alerts.json": (screener.alerts.ALERTS_PATH, screener.alerts.Alert),
    }


def make_backup(dest_dir: Path | str) -> Path:
    """
    Write a backup ZIP file HalgOM-backup-YYYYMMDD-HHMM.zip into dest_dir.
    Includes existing user data files and manifest.json. Returns path to zip.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now()
    zip_name = f"HalgOM-backup-{now.strftime('%Y%m%d-%H%M')}.zip"
    zip_path = dest_path / zip_name

    specs = _get_target_specs()
    files_manifest: dict[str, dict[str, int]] = {}
    files_to_write: dict[str, Path] = {}

    for name, (path, cls) in specs.items():
        if path.exists():
            items = screener.userfiles.load_user_file(path, cls)
            raw_bytes = path.read_bytes()
            files_manifest[name] = {"bytes": len(raw_bytes), "items": len(items)}
            files_to_write[name] = path

    manifest = {
        "app": "HalgOM Screener",
        "format": 1,
        "created": now.isoformat(),
        "files": files_manifest,
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, path in files_to_write.items():
            zf.write(path, arcname=name)

    return zip_path


def inspect(zip_path: Path | str) -> dict[str, dict[str, Any]]:
    """
    Read backup manifest and validate each file without extracting to disk.
    Returns {name: {"ok": bool, "items": n, "problem": str | None}}.
    A zip without manifest or wrong format returns {"_zip": {"ok": False, "items": 0, "problem": ...}}.
    """
    path = Path(zip_path)
    if not path.exists():
        return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            valid_names = {"manifest.json"}.union(ALLOWED_FILES)

            # Zip slip & path validation
            for name in namelist:
                if name not in valid_names or ".." in name or name.startswith("/") or "\\" in name:
                    return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

            if "manifest.json" not in namelist:
                return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

            try:
                manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
            except Exception:
                return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

            if not isinstance(manifest_data, dict):
                return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

            if manifest_data.get("app") != "HalgOM Screener" or manifest_data.get("format") != 1:
                return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}

            results: dict[str, dict[str, Any]] = {}
            specs = _get_target_specs()

            for name in ALLOWED_FILES:
                if name in namelist:
                    raw_data = zf.read(name)
                    try:
                        parsed_json = json.loads(raw_data.decode("utf-8"))
                        if not isinstance(parsed_json, dict):
                            raise ValueError("Root is not dict")
                    except Exception:
                        results[name] = {"ok": False, "items": 0, "problem": MESSAGES["corrupt_file"].format(name=name)}
                        continue

                    # Validate entries using loader
                    target_path, cls = specs[name]
                    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
                        tmp.write(raw_data)
                        tmp_name = Path(tmp.name)

                    try:
                        loaded = screener.userfiles.load_user_file(tmp_name, cls)
                    finally:
                        if tmp_name.exists():
                            tmp_name.unlink()

                    if len(loaded) != len(parsed_json) or any(not isinstance(v, dict) for v in parsed_json.values()):
                        results[name] = {"ok": False, "items": 0, "problem": MESSAGES["corrupt_file"].format(name=name)}
                    else:
                        results[name] = {"ok": True, "items": len(loaded), "problem": None}

            return results
    except Exception:
        return {"_zip": {"ok": False, "items": 0, "problem": MESSAGES["not_a_backup"]}}


def restore(zip_path: Path | str, mode: str = "merge") -> dict[str, dict[str, Any]]:
    """
    Restore user files from backup ZIP.
    Validates first; makes <name>.before-restore-YYYYmmdd-HHMMSS copies of existing files.
    mode="merge": add non-existing entries. mode="replace": replace file content.
    Returns {name: {"added": n, "skipped": n, "replaced": bool}}.
    """
    inspection = inspect(zip_path)
    if any(not item.get("ok", False) for item in inspection.values()):
        return inspection

    path = Path(zip_path)
    specs = _get_target_specs()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results: dict[str, dict[str, Any]] = {}

    with zipfile.ZipFile(path, "r") as zf:
        namelist = zf.namelist()

        # Step 1: Create before-restore backups for existing target files present in ZIP
        for name in ALLOWED_FILES:
            if name in namelist:
                target_path, _ = specs[name]
                if target_path.exists():
                    bak_path = target_path.with_name(f"{target_path.name}.before-restore-{timestamp}")
                    try:
                        shutil.copy2(target_path, bak_path)
                    except OSError as e:
                        log.warning("Failed to create before-restore backup %s: %s", bak_path, e)

        # Step 2: Perform restore via save_user_file
        for name in ALLOWED_FILES:
            if name in namelist:
                target_path, cls = specs[name]
                raw_data = zf.read(name)

                with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
                    tmp.write(raw_data)
                    tmp_name = Path(tmp.name)

                try:
                    backup_items = screener.userfiles.load_user_file(tmp_name, cls)
                finally:
                    if tmp_name.exists():
                        tmp_name.unlink()

                added = 0
                skipped = 0
                replaced = False

                if mode == "replace":
                    def modify_replace(current: dict[str, Any]) -> None:
                        nonlocal added, skipped, replaced
                        added = len(backup_items)
                        skipped = 0
                        replaced = True
                        current.clear()
                        current.update(backup_items)

                    screener.userfiles.save_user_file(target_path, cls, modify_replace)
                else:  # merge
                    def modify_merge(current: dict[str, Any]) -> None:
                        nonlocal added, skipped, replaced
                        replaced = False
                        for k, v in backup_items.items():
                            if k in current:
                                skipped += 1
                            else:
                                added += 1
                                current[k] = v

                    screener.userfiles.save_user_file(target_path, cls, modify_merge)

                results[name] = {"added": added, "skipped": skipped, "replaced": replaced}

    return results
