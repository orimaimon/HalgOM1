"""
"What's new" release notes and version label module.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from screener.data import store

log = logging.getLogger(__name__)

WHATSNEW_PATH = Path(__file__).with_name("whatsnew.json")
ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_whatsnew_path(path: Path | None = None) -> Path:
    return path or WHATSNEW_PATH


def _resolve_seen_path(path: Path | None = None) -> Path:
    return path or (store.DATA_DIR / "whatsnew_seen.json")      # resolved at call time (tests)


def _resolve_root_dir(root: Path | None = None) -> Path:
    return root or ROOT_DIR


def entries(path: Path | None = None) -> list[dict]:
    """Return the list of entries from whatsnew.json.
    A damaged or missing file logs a warning and returns [].
    """
    file_path = _resolve_whatsnew_path(path)
    try:
        if not file_path.exists():
            log.warning("whatsnew file missing: %s", file_path)
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("whatsnew content is not a dict: %s", file_path)
            return []
        items = data.get("entries")
        if not isinstance(items, list):
            log.warning("whatsnew 'entries' is not a list: %s", file_path)
            return []
        out = []
        for item in items:
            if isinstance(item, dict) and "id" in item:
                out.append(item)
            else:
                log.warning("invalid entry in whatsnew: %s", item)
        return out
    except Exception as e:
        log.warning("failed to load whatsnew from %s: %s", file_path, e)
        return []


def unseen(whatsnew_path: Path | None = None, seen_path: Path | None = None) -> list[dict]:
    """Return unseen entries, newest first.
    First run (no seen file): mark every current entry as seen and return [].
    A damaged seen file is treated as empty, never as first run.
    """
    s_path = _resolve_seen_path(seen_path)
    all_entries = entries(whatsnew_path)

    if not s_path.exists():
        # First run: mark current entries seen and return []
        mark_seen(ids=None, whatsnew_path=whatsnew_path, seen_path=s_path)
        return []

    # Seen file exists: try loading seen IDs
    seen_ids: set[str] = set()
    try:
        with open(s_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            seen_ids = set(data)
        elif isinstance(data, dict) and isinstance(data.get("seen"), list):
            seen_ids = set(data["seen"])
        else:
            log.warning("whatsnew seen file has invalid structure: %s", s_path)
    except Exception as e:
        log.warning("failed to load whatsnew seen file %s: %s", s_path, e)
        # Damaged seen file treated as empty (seen_ids remains set())

    return [e for e in all_entries if e.get("id") not in seen_ids]


def mark_seen(
    ids: list[str] | None = None,
    whatsnew_path: Path | None = None,
    seen_path: Path | None = None,
) -> None:
    """Mark specified IDs (or all current entries if ids is None) as seen. Atomic write."""
    s_path = _resolve_seen_path(seen_path)

    if ids is None:
        target_ids = [e["id"] for e in entries(whatsnew_path) if "id" in e]
    else:
        target_ids = list(ids)

    existing_seen: set[str] = set()
    if s_path.exists():
        try:
            with open(s_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing_seen = set(data)
            elif isinstance(data, dict) and isinstance(data.get("seen"), list):
                existing_seen = set(data["seen"])
        except Exception:
            pass

    existing_seen.update(target_ids)
    sorted_seen = sorted(existing_seen)

    try:
        s_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = s_path.with_name(f"{s_path.name}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"seen": sorted_seen}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, s_path)
    except Exception as e:
        log.warning("failed to write whatsnew seen file %s: %s", s_path, e)


def version_label(whatsnew_path: Path | None = None, root: Path | None = None) -> str:
    """Return version label string, e.g. 'גרסה 1.0 · 23/10/2026 · ab12cd3'.
    Uses dulwich to get the checked-out commit date and short SHA.
    If no .git or dulwich fails, uses newest entry date and '—'.
    """
    file_path = _resolve_whatsnew_path(whatsnew_path)
    root_path = _resolve_root_dir(root)

    version = "1.0"
    try:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "version" in data:
                version = str(data["version"])
    except Exception:
        pass

    all_entries = entries(file_path)
    newest_date_formatted = "—"
    if all_entries:
        raw_date = all_entries[0].get("date", "")
        if raw_date:
            try:
                parts = raw_date.split("-")
                if len(parts) == 3:
                    newest_date_formatted = f"{parts[2]}/{parts[1]}/{parts[0]}"
                else:
                    newest_date_formatted = raw_date
            except Exception:
                newest_date_formatted = raw_date

    try:
        from dulwich.repo import Repo

        git_dir = root_path / ".git"
        if not git_dir.exists():
            raise FileNotFoundError("no .git directory")

        with Repo(str(root_path)) as repo:
            head = repo.head()
            commit = repo.get_object(head)
            dt = datetime.fromtimestamp(commit.commit_time)      # local date, as the user sees it
            commit_date_str = dt.strftime("%d/%m/%Y")
            commit_sha_str = head.decode("ascii")[:7]
            return f"גרסה {version} · {commit_date_str} · {commit_sha_str}"
    except Exception as e:
        log.debug("dulwich repo read failed: %s", e)
        return f"גרסה {version} · {newest_date_formatted} · —"
