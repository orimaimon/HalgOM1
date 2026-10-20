"""
Self-update without git: run_screener.bat runs `python -m screener.update` before the screener starts.

  exit 3   files changed: the launcher restarts itself (run_screener.bat may be one of them)
  exit 0   nothing to do: up to date, skipped, or failed. An update never stops the screener.

  - A folder without .git (a ZIP or a copied folder) becomes a checkout of main.
  - A checkout on main with no local edits, behind origin/main, moves to origin/main.
  - Anything else (another branch, local edits, local commits) is left alone, like `git pull --ff-only`.

Remote: GitHub over SSH with a read-only deploy key when %USERPROFILE%\\.halgom\\deploy_key exists
(needed once the repo is private), otherwise anonymous HTTPS. SSH tries port 22, then GitHub's port 443,
then HTTPS. A `deploy_key` (or `deploy_key.txt`, as Notepad saves it) put next to run_screener.bat is
moved there on the first run, so the key does not stay in a shared folder. The key never goes into
the repo (.gitignore).

A checkout writes .git/HALGOM_CHECKOUT (the target commit) first and removes it last: if it is cut
off (window closed, power loss, a file locked by antivirus or OneDrive), the next run finishes it
instead of taking the half-written files for local edits and skipping every later update.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = b"refs/heads/main"
ORIGIN_MAIN = b"refs/remotes/origin/main"
HTTPS_URL = "https://github.com/orimaimon/HalgOM1.git"
SSH_HOST, SSH_USER, SSH_PATH = "github.com", "git", "orimaimon/HalgOM1.git"
SSH_443 = "ssh.github.com"            # GitHub's SSH over port 443, for networks that block port 22
KEY_NAME = "deploy_key"
KEY_NAMES = (KEY_NAME, KEY_NAME + ".txt")   # Notepad adds .txt
TIMEOUT = 30
DEADLINE = 45                    # seconds for the whole fetch: a stalled network must not hold the start
PENDING = "HALGOM_CHECKOUT"         # in .git: the commit a checkout in progress is writing
UPDATED, NOTHING = 3, 0

# GitHub's published SSH host keys (fingerprints checked against docs.github.com, 2026-10-19).
# The connection is refused if github.com presents any other key.
GITHUB_HOST_KEYS = {
    "ssh-ed25519": "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl",
    "ecdsa-sha2-nistp256": "AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMj"
                           "NuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=",
    "ssh-rsa": "AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PK"
               "CUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JH"
               "nsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6Xy"
               "vxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj"
               "4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=",
}

log = logging.getLogger("screener.update")


def key_path() -> Path:
    return Path.home() / ".halgom" / KEY_NAME


def install_key(root: Path, dest: Path) -> None:
    """Move a deploy key dropped next to run_screener.bat into the user's profile."""
    src = next((root / n for n in KEY_NAMES if (root / n).is_file()), None)
    if src is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    if os.name != "nt":
        dest.chmod(0o600)
    log.info("deploy key moved to %s", dest)


# ── transport ────────────────────────────────────────────────────────────────

class _Channel:
    """The read/write/can_read/close object dulwich expects from an SSH vendor, over a paramiko channel."""

    def __init__(self, client, channel):
        self.client, self.channel = client, channel
        self.channel.setblocking(True)

    @property
    def stderr(self):
        return self.channel.makefile_stderr("rb")

    def can_read(self) -> bool:
        return self.channel.recv_ready()

    def write(self, data: bytes) -> None:
        self.channel.sendall(data)

    def read(self, n: int | None = None) -> bytes:
        if n is None:                                   # read to the end
            chunks = []
            while chunk := self.channel.recv(65536):
                chunks.append(chunk)
            return b"".join(chunks)
        out = b""
        while len(out) < n:
            chunk = self.channel.recv(n - len(out))
            if not chunk:
                break
            out += chunk
        return out

    def close(self, timeout: int | None = None) -> None:
        self.channel.close()
        self.client.close()


def ssh_vendor(key_file: Path):
    """dulwich SSH vendor on paramiko: no ssh.exe needed, GitHub's host key pinned, only our key used."""
    import paramiko
    from dulwich.client import SSHVendor

    class Vendor(SSHVendor):
        def run_command(self, host, command, username=None, port=None, password=None, key_filename=None,
                        ssh_command=None, protocol_version=None):
            client = paramiko.SSHClient()
            for kind, b64 in GITHUB_HOST_KEYS.items():        # ssh.github.com:443 has the same keys
                for name in (SSH_HOST, f"[{SSH_443}]:443"):
                    client.get_host_keys().add(name, kind, paramiko.PKey.from_type_string(kind, _b64(b64)))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(host, port=port or 22, username=username, key_filename=str(key_file),
                           look_for_keys=False, allow_agent=False, timeout=TIMEOUT, banner_timeout=TIMEOUT,
                           auth_timeout=TIMEOUT)
            channel = client.get_transport().open_session(timeout=TIMEOUT)
            if protocol_version and protocol_version > 0:
                channel.set_environment_variable("GIT_PROTOCOL", f"version={protocol_version}")
            channel.settimeout(TIMEOUT)
            channel.exec_command(command if isinstance(command, bytes) else command.encode())
            return _Channel(client, channel)

    return Vendor()


def _b64(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def remotes(url: str | None, key_file: Path | None) -> list[tuple[str, object, str]]:
    """(name, dulwich client, path) to try in order: the given url; else GitHub over SSH with the key
    (port 22, then port 443), then anonymous HTTPS (works while the repo is public)."""
    from dulwich.client import SSHGitClient, default_urllib3_manager, get_transport_and_path
    if url:
        return [("url", *get_transport_and_path(url))]
    out = []
    if key_file is not None and key_file.is_file():
        vendor = ssh_vendor(key_file)
        out += [("ssh", SSHGitClient(SSH_HOST, username=SSH_USER, vendor=vendor), SSH_PATH),
                ("ssh-443", SSHGitClient(SSH_443, port=443, username=SSH_USER, vendor=vendor), SSH_PATH)]
    import urllib3
    pool = default_urllib3_manager(None, timeout=TIMEOUT)
    pool.connection_pool_kw["retries"] = urllib3.Retry(total=None, connect=0, read=0, status=0, other=0,
                                                       redirect=5)   # urllib3 retries 3x by default
    out.append(("https", *get_transport_and_path(HTTPS_URL, pool_manager=pool)))
    return out


def remote(url: str | None, key_file: Path | None):
    """The first (dulwich client, path) that fetch_main() tries."""
    return remotes(url, key_file)[0][1:]


def _with_deadline(fn, seconds: float):
    """fn() in a worker thread; TimeoutError if it is not done in time (the thread is left behind)."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:               # handed to the caller
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no answer from the server in {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def fetch_main(repo, url: str | None, key_file: Path | None) -> bytes:
    def wants(refs, depth=None):
        sha = refs.get(MAIN)
        return [sha] if sha and sha not in repo.object_store else []

    end = time.monotonic() + DEADLINE
    errors = []
    for name, client, path in remotes(url, key_file):
        left = end - time.monotonic()
        if left <= 0:
            break
        try:
            result = _with_deadline(lambda c=client, p=path: c.fetch(p, repo, determine_wants=wants), left)
        except Exception as e:
            log.info("fetch over %s failed: %s: %s", name, type(e).__name__, e)
            errors.append(e)
            continue
        sha = result.refs.get(MAIN)
        if not sha:
            raise RuntimeError("remote has no main branch")
        if errors:
            log.info("fetched over %s", name)
        repo.refs[ORIGIN_MAIN] = sha
        return sha
    raise errors[-1] if errors else TimeoutError(f"no answer from the server in {DEADLINE} s")


# ── update ───────────────────────────────────────────────────────────────────

def _is_ancestor(repo, old: bytes, new: bytes) -> bool:
    return any(entry.commit.id == old for entry in repo.get_walker(include=[new]))


def _tracked(repo, commit: bytes):
    from dulwich.object_store import iter_tree_contents
    return iter_tree_contents(repo.object_store, repo[commit].tree)


def _same(data: bytes, blob: bytes) -> bool:
    return data == blob or data.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")


def _local_changes(repo) -> list[str]:
    """Tracked files whose content differs from HEAD (line endings ignored).

    By content, not through the index: dulwich's index after a checkout on Windows (Git for Windows'
    core.autocrlf, run_screener.bat's eol=crlf) can disagree with git's, and then every later update
    would be skipped. A missing file is not an edit: the next checkout restores it."""
    root = Path(repo.path)
    changed = []
    for entry in _tracked(repo, repo.refs[MAIN]):
        path = root / os.fsdecode(entry.path)
        if not path.is_file():
            continue
        if not _same(path.read_bytes(), repo[entry.sha].data):
            changed.append(os.fsdecode(entry.path))
    return changed


def _staged_new(repo) -> list[str]:
    """Files added to the index (git add) but not committed: a hard reset would delete them."""
    tracked = {entry.path for entry in _tracked(repo, repo.refs[MAIN])}
    return [os.fsdecode(p) for p in repo.open_index() if p not in tracked]


def _in_the_way(repo, old: bytes, new: bytes) -> list[str]:
    """Untracked files of this folder at paths the new commit adds, with other content: the update
    would overwrite them."""
    root = Path(repo.path)
    before = {entry.path for entry in _tracked(repo, old)}
    return [os.fsdecode(e.path) for e in _tracked(repo, new)
            if e.path not in before and (root / os.fsdecode(e.path)).is_file()
            and not _same((root / os.fsdecode(e.path)).read_bytes(), repo[e.sha].data)]


def _missing(repo) -> list[str]:
    """Tracked files that are gone (e.g. quarantined by antivirus)."""
    root = Path(repo.path)
    return [os.fsdecode(e.path) for e in _tracked(repo, repo.refs[MAIN])
            if not (root / os.fsdecode(e.path)).exists()]


def _pending(repo) -> bytes | None:
    """Target commit of a checkout that was cut off, if any."""
    try:
        sha = (Path(repo.controldir()) / PENDING).read_text(encoding="ascii").strip().encode()
    except OSError:
        return None
    return sha if sha in repo.object_store else None


def _checkout(repo, sha: bytes) -> None:
    from dulwich import porcelain
    from dulwich.index import index_entry_from_stat
    marker = Path(repo.controldir()) / PENDING
    marker.write_text(sha.decode(), encoding="ascii")      # removed only when everything is written
    repo.refs[MAIN] = sha
    repo.refs.set_symbolic_ref(b"HEAD", MAIN)
    porcelain.reset(repo, "hard", sha)
    # Index = the commit's blobs + the files' stat, as `git checkout` writes it, so git (Ori's
    # machine) and later updates see a clean checkout whatever dulwich hashed on Windows.
    root, index = Path(repo.path), repo.open_index()
    for entry in _tracked(repo, sha):
        path = root / os.fsdecode(entry.path)
        if path.is_file():
            index[entry.path] = index_entry_from_stat(os.lstat(path), entry.sha, mode=entry.mode)
    index.write()
    marker.unlink()


def update(root: Path = ROOT, url: str | None = None, key_file: Path | None = None) -> int:
    from dulwich.repo import Repo

    if not (root / ".git").exists():
        with Repo.init(str(root)) as repo:
            config = repo.get_config()               # so `git pull` also works here, for a developer
            config.set((b"remote", b"origin"), b"url", HTTPS_URL.encode())
            config.set((b"remote", b"origin"), b"fetch", b"+refs/heads/*:refs/remotes/origin/*")
            config.set((b"branch", b"main"), b"remote", b"origin")
            config.set((b"branch", b"main"), b"merge", MAIN)
            config.write_to_path()
            sha = fetch_main(repo, url, key_file)
            _checkout(repo, sha)
        print("        updated - this folder now updates itself")
        log.info("converted %s to a checkout of main at %s", root, sha.decode()[:7])
        return UPDATED

    with Repo(str(root)) as repo:
        pending = _pending(repo)
        if pending:                                  # a checkout was cut off: finish it first
            _checkout(repo, pending)
            log.info("finished an interrupted update to %s", pending.decode()[:7])
            print("        finished an interrupted update")
            return UPDATED
        if MAIN not in repo.refs:
            if any(r.startswith(b"refs/heads/") for r in repo.refs.allkeys()):
                log.info("skipped: no local main branch")    # e.g. a developer's clone of another branch
                return NOTHING
            sha = fetch_main(repo, url, key_file)    # an earlier conversion stopped before its checkout
            _checkout(repo, sha)
            log.info("finished converting %s at %s", root, sha.decode()[:7])
            return UPDATED
        head = repo.refs.get_symrefs().get(b"HEAD")
        if head != MAIN:
            log.info("skipped: HEAD is %s, not main", head or b"detached")
            return NOTHING
        changed = _local_changes(repo) + _staged_new(repo)
        if changed:
            log.info("skipped: local changes in %s", ", ".join(changed[:5]))
            print("        update skipped - files in this folder were changed")
            return NOTHING
        local = repo.refs[MAIN]
        sha = fetch_main(repo, url, key_file)
        if sha == local:
            missing = _missing(repo)
            if missing:
                _checkout(repo, local)
                log.info("restored %s", ", ".join(missing[:5]))
                return UPDATED
            return NOTHING
        if not _is_ancestor(repo, local, sha):
            log.info("skipped: local main %s is not behind origin/main %s", local.decode()[:7], sha.decode()[:7])
            return NOTHING
        blocking = _in_the_way(repo, local, sha)
        if blocking:
            log.info("skipped: the update would overwrite %s", ", ".join(blocking[:5]))
            print("        update skipped - files in this folder were changed")
            return NOTHING
        _checkout(repo, sha)
        print("        updated to the latest version")
        log.info("updated %s -> %s", local.decode()[:7], sha.decode()[:7])
        return UPDATED


def _setup_log(root: Path) -> None:
    logs = root / "screener" / "_data" / "logs"
    try:
        logs.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logs / "update.log", encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--url", help="remote to update from (tests); default: GitHub")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    _setup_log(root)
    try:
        key = key_path()
        install_key(root, key)
        cloud = cloud_folder(root)
        if cloud:
            print(f"        note: this folder is inside {cloud} - copy it to a local folder, e.g. C:\\HalgOM")
        return update(root, args.url, key)
    except Exception as e:                            # never stop the screener from starting
        log.exception("update failed")
        print(f"        update skipped - {type(e).__name__}")
        # Cut off in the middle of a checkout: run_screener.bat may already be the new file, so the
        # launcher must restart (the next run finishes the checkout).
        return UPDATED if (root / ".git" / PENDING).exists() else NOTHING


def cloud_folder(root: Path) -> str | None:
    """A synced cloud folder (it locks files while uploading and syncs the large .venv), if root is in one."""
    path = str(root).lower()
    onedrive = [os.environ[v].lower() for v in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
                if os.environ.get(v)]
    if "onedrive" in path or any(path.startswith(d) for d in onedrive):
        return "OneDrive"
    if "google drive" in path or "my drive" in path or "googledrive" in path:
        return "Google Drive"
    return None


if __name__ == "__main__":
    sys.exit(main())
