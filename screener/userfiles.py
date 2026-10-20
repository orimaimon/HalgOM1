"""
Shared helper for user data files (presets.json, portfolios.json).
Provides safe loading (per-entry parsing, ignoring unknown keys, renaming damaged files)
and atomic saving (process-wide lock, unique temp file + fsync, .bak backup).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

_LOCK = threading.Lock()

T = TypeVar("T")


def load_user_file(path: Path, item_cls: type[T]) -> dict[str, T]:
    """
    Load a JSON dict of dataclasses per-entry safely.
    - Missing file -> returns {}
    - Damaged file (invalid JSON or not a dict) -> renames to <name>.bad-YYYYmmdd-HHMMSS, logs warning, returns {}
    - Entry errors / unknown keys -> ignores unknown keys, skips bad entries with a log warning
    - Temporary read error (OSError) -> raises OSError (never returns empty dict)
    """
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("cannot read user file %s: %s", path, e)
        raise

    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("Root JSON element is not a dict")
    except Exception as e:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bad_path = path.with_name(f"{path.name}.bad-{timestamp}")
        log.warning("Damaged user file %s (%s); renaming to %s and starting empty", path, e, bad_path.name)
        try:
            os.replace(path, bad_path)
        except OSError as rename_err:
            log.error("Failed to rename damaged file %s to %s: %s", path, bad_path, rename_err)
        return {}

    valid_fields = {f.name for f in dataclasses.fields(item_cls)}
    results = {}
    for name, item_dict in raw.items():
        if not isinstance(item_dict, dict):
            log.warning("Skipping bad entry %r in %s: not a dict", name, path)
            continue
        try:
            kwargs = {k: v for k, v in item_dict.items() if k in valid_fields}
            if "name" in valid_fields and "name" not in kwargs:
                kwargs["name"] = name
            results[name] = item_cls(**kwargs)
        except Exception as e:
            log.warning("Skipping bad entry %r in %s: %s", name, path, e)
            continue

    return results


def save_user_file(
    path: Path,
    item_cls: type[T],
    modify_fn: Callable[[dict[str, T]], None],
) -> dict[str, T]:
    """
    Safely modify and write a user file with process-wide locking, .bak backup, and fsync.
    `modify_fn` accepts the loaded dict and modifies it in-place.
    Returns the updated items dict.
    """
    with _LOCK:
        items = load_user_file(path, item_cls)
        modify_fn(items)
        _write_user_file(path, items)
        return items


def _write_user_file(path: Path, items: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: dataclasses.asdict(v) for k, v in items.items()}
    text = json.dumps(raw, ensure_ascii=False, indent=2)

    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        if path.exists():
            bak_path = path.with_name(f"{path.name}.bak")
            try:
                shutil.copy2(path, bak_path)
            except OSError as e:
                log.warning("Failed to create backup copy %s: %s", bak_path, e)

        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
