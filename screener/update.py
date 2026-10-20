"""
Self-update without git: run_screener.bat runs `python -m screener.update` before the screener starts.

  exit 3   files changed: the launcher restarts itself (run_screener.bat may be one of them)
  exit 0   nothing to do: up to date, skipped, or failed. An update never stops the screener.

  - A folder without .git (a ZIP or a copied folder) becomes a checkout of main.
  - A checkout on main with no local edits, behind origin/main, moves to origin/main.
  - Anything else (another branch, local edits, local commits) is left alone, like `git pull --ff-only`.

Remote: GitHub over SSH with a read-only deploy key when %USERPROFILE%\\.halgom\\deploy_key exists
(needed once the repo is private), otherwise anonymous HTTPS. A `deploy_key` file put next to
run_screener.bat is moved there on the first run, so the key does not stay in a shared folder.
The key never goes into the repo (.gitignore).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = b"refs/heads/main"
ORIGIN_MAIN = b"refs/remotes/origin/main"
HTTPS_URL = "https://github.com/orimaimon/HalgOM1.git"
SSH_HOST, SSH_USER, SSH_PATH = "github.com", "git", "orimaimon/HalgOM1.git"
KEY_NAME = "deploy_key"
TIMEOUT = 30
FETCH_DEADLINE = 45
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
    src = root / KEY_NAME
    if not src.is_file():
        src = root / f"{KEY_NAME}.txt"
        if not src.is_file():
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
            for kind, b64 in GITHUB_HOST_KEYS.items():
                key_obj = paramiko.PKey.from_type_string(kind, _b64(b64))
                client.get_host_keys().add(host, kind, key_obj)
                client.get_host_keys().add(SSH_HOST, kind, key_obj)
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


def remote(url: str | None, key_file: Path | None):
    """(dulwich client, path) for the given url, else for GitHub (SSH with the key, else HTTPS)."""
    from dulwich.client import SSHGitClient, default_urllib3_manager, get_transport_and_path
    if url:
        return get_transport_and_path(url)
    if key_file is not None and key_file.is_file():
        return SSHGitClient(SSH_HOST, username=SSH_USER, vendor=ssh_vendor(key_file)), SSH_PATH
    mgr = default_urllib3_manager(None, timeout=TIMEOUT)
    mgr.connection_pool_kw["retries"] = False
    return get_transport_and_path(HTTPS_URL, pool_manager=mgr)


def fetch_main(repo, url: str | None, key_file: Path | None) -> bytes:
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch_main_worker, repo, url, key_file)
    try:
        return future.result(timeout=FETCH_DEADLINE)
    except concurrent.futures.TimeoutError:
        log.warning("fetch timed out after %s seconds", FETCH_DEADLINE)
        raise TimeoutError(f"fetch timed out after {FETCH_DEADLINE}s")
    finally:
        executor.shutdown(wait=False)


def _fetch_main_worker(repo, url: str | None, key_file: Path | None) -> bytes:
    from dulwich.client import default_urllib3_manager, get_transport_and_path
    if url:
        client, path = get_transport_and_path(url)
        return _fetch_with_client(client, path, repo, "custom URL")

    if key_file is not None and key_file.is_file():
        from dulwich.client import SSHGitClient
        try:
            client = SSHGitClient("github.com", username=SSH_USER, port=22, vendor=ssh_vendor(key_file))
            return _fetch_with_client(client, SSH_PATH, repo, "SSH github.com:22")
        except Exception as e:
            log.warning("SSH github.com:22 failed (%s), trying ssh.github.com:443", e)

        try:
            client = SSHGitClient("ssh.github.com", username=SSH_USER, port=443, vendor=ssh_vendor(key_file))
            return _fetch_with_client(client, SSH_PATH, repo, "SSH ssh.github.com:443")
        except Exception as e:
            log.warning("SSH ssh.github.com:443 failed (%s), falling back to anonymous HTTPS", e)

    mgr = default_urllib3_manager(None, timeout=TIMEOUT)
    mgr.connection_pool_kw["retries"] = False
    client, path = get_transport_and_path(HTTPS_URL, pool_manager=mgr)
    return _fetch_with_client(client, path, repo, "anonymous HTTPS")


def _fetch_with_client(client, path, repo, desc: str) -> bytes:
    def wants(refs, depth=None):
        sha = refs.get(MAIN)
        return [sha] if sha and sha not in repo.object_store else []

    result = client.fetch(path, repo, determine_wants=wants)
    sha = result.refs.get(MAIN)
    if not sha:
        raise RuntimeError("remote has no main branch")
    repo.refs[ORIGIN_MAIN] = sha
    log.info("fetched main from %s", desc)
    return sha


# ── update ───────────────────────────────────────────────────────────────────

def _is_ancestor(repo, old: bytes, new: bytes) -> bool:
    return any(entry.commit.id == old for entry in repo.get_walker(include=[new]))


def _tracked(repo, commit: bytes):
    from dulwich.object_store import iter_tree_contents
    return iter_tree_contents(repo.object_store, repo[commit].tree)


def _staged_changes(repo) -> list[str]:
    """Index entries that differ from HEAD (staged additions, modifications, or deletions)."""
    try:
        index = repo.open_index()
    except Exception:
        return []
    head_entries = {entry.path: (entry.mode, entry.sha) for entry in _tracked(repo, repo.refs[MAIN])}
    staged = []
    index_paths = set(index)
    head_paths = set(head_entries.keys())

    for p in index_paths - head_paths:
        staged.append(os.fsdecode(p))

    for p in head_paths - index_paths:
        staged.append(os.fsdecode(p))

    for p in index_paths & head_paths:
        idx_sha = index[p].sha
        head_sha = head_entries[p][1]
        if idx_sha != head_sha and idx_sha in repo.object_store:
            idx_data = repo[idx_sha].data
            head_data = repo[head_sha].data
            if idx_data != head_data and idx_data.replace(b"\r\n", b"\n") != head_data.replace(b"\r\n", b"\n"):
                staged.append(os.fsdecode(p))

    return staged


def _untracked_conflicts(repo, incoming_sha: bytes) -> list[str]:
    """Untracked files on disk at paths added by incoming_sha with different content."""
    root = Path(repo.path)
    current_paths = {entry.path for entry in _tracked(repo, repo.refs[MAIN])}
    conflicts = []
    for entry in _tracked(repo, incoming_sha):
        if entry.path not in current_paths:
            p = root / os.fsdecode(entry.path)
            if p.is_file():
                data = p.read_bytes()
                blob = repo[entry.sha].data
                if data != blob and data.replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
                    conflicts.append(os.fsdecode(entry.path))
    return conflicts


def _missing_tracked(repo) -> list[str]:
    """Tracked files in HEAD that do not exist on disk."""
    root = Path(repo.path)
    missing = []
    for entry in _tracked(repo, repo.refs[MAIN]):
        path = root / os.fsdecode(entry.path)
        if not path.is_file():
            missing.append(os.fsdecode(entry.path))
    return missing


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
        data, blob = path.read_bytes(), repo[entry.sha].data
        if data != blob and data.replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            changed.append(os.fsdecode(entry.path))
    return changed


from dulwich import porcelain


def _checkout(repo, sha: bytes) -> None:
    from dulwich.index import index_entry_from_stat
    root = Path(repo.path)
    marker = root / ".git" / "HALGOM_CHECKOUT"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(sha)
    except OSError:
        pass
    repo.refs[MAIN] = sha
    repo.refs.set_symbolic_ref(b"HEAD", MAIN)
    porcelain.reset(repo, "hard", sha)
    # Index = the commit's blobs + the files' stat, as `git checkout` writes it, so git (Ori's
    # machine) and later updates see a clean checkout whatever dulwich hashed on Windows.
    index = repo.open_index()
    for entry in _tracked(repo, sha):
        path = root / os.fsdecode(entry.path)
        if path.is_file():
            index[entry.path] = index_entry_from_stat(os.lstat(path), entry.sha, mode=entry.mode)
    index.write()
    marker.unlink(missing_ok=True)


def update(root: Path = ROOT, url: str | None = None, key_file: Path | None = None) -> int:
    from dulwich.repo import Repo

    git_dir = root / ".git"
    marker = git_dir / "HALGOM_CHECKOUT"
    if marker.is_file():
        target_sha = marker.read_bytes().strip()
        log.info("resuming interrupted checkout to %s", target_sha.decode()[:7])
        if git_dir.exists():
            with Repo(str(root)) as repo:
                if target_sha in repo.object_store:
                    _checkout(repo, target_sha)
                    print("        updated to the latest version")
                    return UPDATED

    if not (root / ".git").exists():
        with Repo.init(str(root)) as repo:
            config = repo.get_config()               # so `git pull` also works here, for a developer
            config.set((b"remote", b"origin"), b"url", HTTPS_URL.encode())
            config.set((b"remote", b"origin"), b"fetch", b"+refs/heads/*:refs/remotes/origin/*")
            config.set((b"branch", b"main"), b"remote", b"origin")
            config.set((b"branch", b"main"), b"merge", MAIN)
            config.write_to_path()
            try:
                sha = fetch_main(repo, url, key_file)
            except TimeoutError:
                print("        update skipped - connection timed out")
                return NOTHING
            _checkout(repo, sha)
        print("        updated - this folder now updates itself")
        log.info("converted %s to a checkout of main at %s", root, sha.decode()[:7])
        return UPDATED

    with Repo(str(root)) as repo:
        if MAIN not in repo.refs:                    # an earlier conversion stopped half-way
            branch_refs = [r for r in repo.refs.keys() if r.startswith(b"refs/heads/")]
            if branch_refs and not marker.is_file():
                log.info("skipped: main branch missing and other branches exist")
                return NOTHING
            try:
                sha = fetch_main(repo, url, key_file)
            except TimeoutError:
                print("        update skipped - connection timed out")
                return NOTHING
            _checkout(repo, sha)
            log.info("finished converting %s at %s", root, sha.decode()[:7])
            return UPDATED
        head = repo.refs.get_symrefs().get(b"HEAD")
        if head != MAIN:
            log.info("skipped: HEAD is %s, not main", head or b"detached")
            return NOTHING
        staged = _staged_changes(repo)
        if staged:
            log.info("skipped: staged changes in %s", ", ".join(staged[:5]))
            print("        update skipped - files in this folder were changed")
            return NOTHING
        changed = _local_changes(repo)
        if changed:
            log.info("skipped: local changes in %s", ", ".join(changed[:5]))
            print("        update skipped - files in this folder were changed")
            return NOTHING
        missing = _missing_tracked(repo)
        if missing:
            log.info("restoring missing tracked files: %s", ", ".join(missing[:5]))
            _checkout(repo, repo.refs[MAIN])
            print("        restored missing files")
            return UPDATED
        local = repo.refs[MAIN]
        try:
            sha = fetch_main(repo, url, key_file)
        except TimeoutError as e:
            log.warning("update skipped due to timeout: %s", e)
            print("        update skipped - connection timed out")
            return NOTHING
        if sha == local:
            return NOTHING
        untracked = _untracked_conflicts(repo, sha)
        if untracked:
            log.info("skipped: untracked file collision in %s", ", ".join(untracked[:5]))
            print("        update skipped - files in this folder were changed")
            return NOTHING
        if not _is_ancestor(repo, local, sha):
            log.info("skipped: local main %s is not behind origin/main %s", local.decode()[:7], sha.decode()[:7])
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
        root_str = str(root).lower()
        onedrive_env = os.environ.get("OneDrive") or os.environ.get("ONEDRIVE")
        in_onedrive = "onedrive" in root_str or (onedrive_env is not None and (Path(onedrive_env) == root or Path(onedrive_env) in root.parents))
        if "google drive" in root_str or "my drive" in root_str or in_onedrive:
            print("        note: this folder is inside a cloud folder - copy it to a local folder, e.g. C:\\HalgOM")
        return update(root, args.url, key)
    except Exception as e:                            # never stop the screener from starting
        log.exception("update failed")
        print(f"        update skipped - {type(e).__name__}")
        if (root / ".git" / "HALGOM_CHECKOUT").is_file():
            return UPDATED
        return NOTHING


if __name__ == "__main__":
    sys.exit(main())
