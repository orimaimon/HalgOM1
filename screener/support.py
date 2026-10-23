"""
Problem report package generator and daily data usage logger.

make_report(version: str, dest: Path | None = None) -> Path
    Generates a zip report on the user's desktop or _data/reports/.

data_usage() -> dict
    Calculates total bytes and file count in _data/ and its top-level folders.

log_daily_usage() -> bool
    Logs a summary of data usage at most once per day to _data/usage_logged.txt.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import logging
import os
import platform
import uuid
import zipfile
from pathlib import Path

from screener.data import store

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_repo_root() -> Path:
    return REPO_ROOT


def get_data_dir() -> Path:
    return store.DATA_DIR


def _shell_desktop() -> Path | None:
    """The Desktop folder Windows actually shows, from the registry. With OneDrive backup it is
    %USERPROFILE%\\OneDrive\\Desktop (or a localized name) while an empty %USERPROFILE%\\Desktop still
    exists, so guessing would put the report where the user never sees it."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
        return Path(os.path.expandvars(value))
    except Exception:            # not Windows, or no such value
        return None


def desktop_dirs() -> list[Path]:
    userprofile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    base = Path(userprofile)
    dirs = [
        base / "Desktop",
        base / "OneDrive" / "Desktop",
        base / "OneDrive" / "שולחן העבודה",
    ]
    shell = _shell_desktop()
    # Only a desktop inside this USERPROFILE (tests point USERPROFILE at a temp folder).
    if shell is not None and base in shell.parents:
        dirs.insert(0, shell)
    return dirs


def get_git_sha(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = get_repo_root()

    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return "—"

    if git_dir.is_file():
        try:
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = (repo_root / content[7:].strip()).resolve()
            else:
                return "—"
        except Exception:
            return "—"

    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return "—"

    try:
        head_content = head_file.read_text(encoding="utf-8").strip()
    except Exception:
        return "—"

    if not head_content:
        return "—"

    if head_content.startswith("ref:"):
        ref_path = head_content[4:].strip()
        target_ref = git_dir / ref_path
        if target_ref.exists():
            try:
                sha = target_ref.read_text(encoding="utf-8").strip()
                if sha:
                    return sha
            except Exception:
                pass

        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            try:
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == ref_path:
                        return parts[0]
            except Exception:
                pass
        return "—"
    else:
        return head_content


def get_package_versions() -> dict[str, str]:
    pkgs = [
        "streamlit",
        "pandas",
        "numpy",
        "pyarrow",
        "yfinance",
        "requests",
        "dulwich",
        "plotly",
        "openpyxl",
    ]
    res = {}
    for pkg in pkgs:
        try:
            res[pkg] = importlib.metadata.version(pkg)
        except Exception:
            res[pkg] = "—"
    return res


def make_report(version: str, dest: Path | None = None) -> Path:
    data_dir = get_data_dir()
    repo_root = get_repo_root()

    if dest is None:
        for d in desktop_dirs():
            if d.is_dir() and os.access(d, os.W_OK):
                dest = d
                break
        if dest is None:
            dest = data_dir / "reports"

    dest.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    zip_path = dest / f"HalgOM-report-{timestamp}.zip"

    skipped: list[str] = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. logs/
        logs_dir = data_dir / "logs"
        if logs_dir.exists():
            try:
                for p in sorted(logs_dir.iterdir()):
                    if p.is_file() and (
                        p.name.startswith("screener.log")
                        or p.name.startswith("update.log")
                    ):
                        try:
                            zf.write(p, arcname=f"logs/{p.name}")
                        except Exception as e:
                            skipped.append(f"logs/{p.name}: {e}")
            except Exception as e:
                skipped.append(f"logs: {e}")

        # 2. snapshot.json
        snap_dir = data_dir / "snapshots"
        snap_added = False
        if snap_dir.exists():
            try:
                snap_jsons = sorted(
                    snap_dir.glob("*.json"),
                    key=lambda p: (p.name, p.stat().st_mtime),
                )
                if snap_jsons:
                    latest_snap = snap_jsons[-1]
                    try:
                        content = latest_snap.read_text(encoding="utf-8")
                        zf.writestr("snapshot.json", content)
                        snap_added = True
                    except Exception as e:
                        skipped.append(f"snapshot.json ({latest_snap.name}): {e}")
            except Exception as e:
                skipped.append(f"snapshot.json: {e}")

        if not snap_added and not any("snapshot.json" in s for s in skipped):
            skipped.append("snapshot.json: no snapshot metadata found")

        # 3. files.txt
        files_lines: list[str] = []
        if data_dir.exists():
            try:
                for root, _, files in os.walk(data_dir):
                    for f in sorted(files):
                        p = Path(root) / f
                        try:
                            rel = p.relative_to(data_dir).as_posix()
                            st = p.stat()
                            mtime_str = dt.datetime.fromtimestamp(
                                st.st_mtime, tz=dt.timezone.utc
                            ).isoformat()
                            files_lines.append(f"{rel}\t{st.st_size}\t{mtime_str}")
                        except Exception as e:
                            skipped.append(f"files.txt entry {f}: {e}")
            except Exception as e:
                skipped.append(f"files.txt: {e}")
        zf.writestr("files.txt", "\n".join(files_lines))

        # 4. info.json
        info_data = {
            "version": version,
            "git_sha": get_git_sha(repo_root),
            "python": platform.python_version(),
            "os": platform.platform(),
            "packages": get_package_versions(),
            "time": dt.datetime.now().astimezone().isoformat(),
            "skipped": skipped,
        }
        zf.writestr(
            "info.json",
            json.dumps(info_data, indent=2, ensure_ascii=False),
        )

    return zip_path


def data_usage() -> dict:
    data_dir = get_data_dir()
    if not data_dir.exists():
        return {"total": {"bytes": 0, "files": 0}}

    total_bytes = 0
    total_files = 0
    res: dict[str, dict[str, int]] = {}

    try:
        items = sorted(data_dir.iterdir())
    except OSError:
        return {"total": {"bytes": 0, "files": 0}}

    for item in items:
        if item.is_dir():
            folder_bytes = 0
            folder_files = 0
            for root, _, files in os.walk(item):
                for f in files:
                    fp = Path(root) / f
                    try:
                        st = fp.stat()
                        folder_bytes += st.st_size
                        folder_files += 1
                    except OSError:
                        pass
            res[item.name] = {"bytes": folder_bytes, "files": folder_files}
            total_bytes += folder_bytes
            total_files += folder_files
        elif item.is_file():
            try:
                st = item.stat()
                total_bytes += st.st_size
                total_files += 1
            except OSError:
                pass

    res["total"] = {"bytes": total_bytes, "files": total_files}
    return res


def log_daily_usage() -> bool:
    data_dir = get_data_dir()
    usage_file = data_dir / "usage_logged.txt"
    today = dt.date.today().isoformat()

    if usage_file.exists():
        try:
            if usage_file.read_text(encoding="utf-8").strip() == today:
                return False
        except OSError:
            pass

    usage = data_usage()
    total_bytes = usage.get("total", {}).get("bytes", 0)
    total_mb = total_bytes / (1024 * 1024)

    parts = [f"data usage: {total_mb:.1f} MB total"]

    folders = [
        (k, v) for k, v in usage.items() if k != "total" and isinstance(v, dict)
    ]
    folders.sort(key=lambda x: x[1].get("bytes", 0), reverse=True)

    for folder_name, info in folders:
        mb = info.get("bytes", 0) / (1024 * 1024)
        parts.append(f"{folder_name} {mb:.1f} MB ({info.get('files', 0)} files)")

    msg = "; ".join(parts)
    log.info(msg)

    _atomic_write_text(usage_file, today)
    return True


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
