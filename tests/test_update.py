"""screener/update.py: the self-update that replaces `git pull` (no git on the end user's PC).
A local dulwich repo stands in for GitHub; no network."""
from __future__ import annotations

from pathlib import Path

import pytest
from dulwich import porcelain
from dulwich.repo import Repo

from screener import update as U

BAT = "@echo off\nrem launcher\necho hi\n"


def write(root: Path, files: dict[str, str | None]) -> None:
    for name, text in files.items():
        p = root / name
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode())


def commit(remote: Path, files: dict[str, str | None], msg: str) -> bytes:
    write(remote, files)
    with Repo(str(remote)) as r:
        porcelain.add(r, paths=[str(remote / n) for n, t in files.items() if t is not None])
        removed = [n for n, t in files.items() if t is None]
        if removed:
            porcelain.remove(r, paths=[str(remote / n) for n in removed], cached=True)
        return porcelain.commit(r, message=msg.encode(), author=b"t <t@t>", committer=b"t <t@t>")


@pytest.fixture(autouse=True)
def no_user_git_config(tmp_path, monkeypatch):
    """dulwich reads ~/.gitconfig (commit signing, ...): keep the developer's settings out of the tests."""
    home = tmp_path / "home"
    home.mkdir()
    for var in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "remote"
    path.mkdir()
    with Repo.init(str(path)) as r:
        r.refs.set_symbolic_ref(b"HEAD", U.MAIN)
    commit(path, {".gitattributes": "*.bat text eol=crlf\n", "run.bat": BAT, "a.txt": "v1\n", "b.txt": "b\n"}, "v1")
    return path


@pytest.fixture
def zip_copy(tmp_path):
    """The folder as the end user got it: files of an older version, no .git, plus local data."""
    path = tmp_path / "HalgOM"
    write(path, {"a.txt": "v0\n", "run.bat": BAT.replace("\n", "\r\n"), "screener/_data/presets.json": "{}"})
    return path


def head(root: Path) -> bytes:
    with Repo(str(root)) as r:
        return r.refs[U.MAIN]


def test_zip_folder_becomes_a_checkout(remote, zip_copy):
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED
    assert (zip_copy / ".git").is_dir()
    assert (zip_copy / "a.txt").read_text() == "v1\n" and (zip_copy / "b.txt").exists()
    assert (zip_copy / "run.bat").read_bytes() == BAT.replace("\n", "\r\n").encode()   # .gitattributes eol=crlf
    assert (zip_copy / "screener/_data/presets.json").read_text() == "{}"              # local data kept
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING                             # now up to date
    with Repo(str(zip_copy)) as r:                                  # plain `git pull` works there too
        config = r.get_config()
        assert config.get((b"branch", b"main"), b"remote") == b"origin"
        assert config.get((b"branch", b"main"), b"merge") == U.MAIN


def test_fast_forward_updates_changes_and_deletes(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    new = commit(remote, {"a.txt": "v2\n", "b.txt": None, "c.txt": "c\n", "run.bat": BAT + "echo v2\n"}, "v2")
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED
    assert head(zip_copy) == new
    assert (zip_copy / "a.txt").read_text() == "v2\n" and (zip_copy / "c.txt").exists()
    assert not (zip_copy / "b.txt").exists()
    assert b"\r\necho v2\r\n" in (zip_copy / "run.bat").read_bytes()


def test_staged_changes_are_never_overwritten(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    before = head(zip_copy)
    write(zip_copy, {"staged.txt": "staged\n"})
    with Repo(str(zip_copy)) as r:
        porcelain.add(r, paths=[str(zip_copy / "staged.txt")])
    commit(remote, {"a.txt": "v2\n"}, "v2")
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING
    assert (zip_copy / "staged.txt").read_text() == "staged\n" and head(zip_copy) == before


def test_untracked_colliding_files_are_never_overwritten(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    before = head(zip_copy)
    write(zip_copy, {"new_incoming.txt": "local untracked\n"})
    commit(remote, {"new_incoming.txt": "remote content\n"}, "v2")
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING
    assert (zip_copy / "new_incoming.txt").read_text() == "local untracked\n" and head(zip_copy) == before


def test_missing_tracked_file_is_restored(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    (zip_copy / "a.txt").unlink()
    assert not (zip_copy / "a.txt").exists()
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED
    assert (zip_copy / "a.txt").read_text() == "v1\n"


def test_local_edits_are_never_overwritten(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    before = head(zip_copy)
    write(zip_copy, {"a.txt": "my edit\n"})
    commit(remote, {"a.txt": "v2\n"}, "v2")
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING
    assert (zip_copy / "a.txt").read_text() == "my edit\n" and head(zip_copy) == before


def test_other_branch_or_local_commits_are_left_alone(remote, zip_copy):
    U.update(zip_copy, url=str(remote))
    commit(zip_copy, {"mine.txt": "x\n"}, "local work")          # main is now ahead of origin/main
    commit(remote, {"a.txt": "v2\n"}, "v2")                       # ... and behind it: diverged
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING
    assert (zip_copy / "a.txt").read_text() == "v1\n"
    with Repo(str(zip_copy)) as r:
        r.refs[b"refs/heads/dev"] = r.refs[U.MAIN]
        r.refs.set_symbolic_ref(b"HEAD", b"refs/heads/dev")
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING


def test_index_matches_the_commit_and_index_quirks_do_not_block_updates(remote, zip_copy):
    import dataclasses
    from dulwich.object_store import iter_tree_contents
    U.update(zip_copy, url=str(remote))
    with Repo(str(zip_copy)) as r:
        tree = {e.path: e.sha for e in iter_tree_contents(r.object_store, r[r.refs[U.MAIN]].tree)}
        index = r.open_index()
        assert {p: index[p].sha for p in tree} == tree                  # as `git checkout` writes it
        # what CI saw on Windows: the index holds the CRLF file's hash, so git/dulwich call it modified
        index[b"run.bat"] = dataclasses.replace(index[b"run.bat"], sha=b"0" * 40)
        index.write()
    new = commit(remote, {"a.txt": "v2\n"}, "v2")
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED and head(zip_copy) == new


def test_half_done_conversion_is_finished(remote, zip_copy):
    Repo.init(str(zip_copy)).close()                              # .git exists, no main yet
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED
    assert (zip_copy / "a.txt").read_text() == "v1\n"


def test_onedrive_cloud_folder_warning(tmp_path, monkeypatch, capsys):
    od_folder = tmp_path / "OneDrive" / "HalgOM"
    od_folder.mkdir(parents=True)
    U.main(["--root", str(od_folder), "--url", str(tmp_path / "missing")])
    out = capsys.readouterr().out
    assert "cloud folder" in out or "Google Drive" in out

    plain_folder = tmp_path / "Plain" / "HalgOM"
    plain_folder.mkdir(parents=True)
    monkeypatch.setenv("OneDrive", str(tmp_path / "Plain"))
    U.main(["--root", str(plain_folder), "--url", str(tmp_path / "missing")])
    out2 = capsys.readouterr().out
    assert "cloud folder" in out2 or "Google Drive" in out2


def test_failure_never_stops_the_launcher(tmp_path, zip_copy, monkeypatch):
    monkeypatch.setattr(U, "key_path", lambda: tmp_path / "no-key")
    assert U.main(["--root", str(zip_copy), "--url", str(tmp_path / "missing")]) == U.NOTHING
    assert "update failed" in (zip_copy / "screener/_data/logs/update.log").read_text()


def test_deploy_key_moves_out_of_the_shared_folder(tmp_path):
    root, dest = tmp_path / "HalgOM", tmp_path / "home" / ".halgom" / U.KEY_NAME
    write(root, {U.KEY_NAME: "-----BEGIN OPENSSH PRIVATE KEY-----\n"})
    U.install_key(root, dest)
    assert dest.is_file() and not (root / U.KEY_NAME).exists()
    U.install_key(root, dest)                                     # nothing to move: no error
    assert dest.is_file()

    # Test deploy_key.txt
    dest.unlink()
    write(root, {f"{U.KEY_NAME}.txt": "-----BEGIN OPENSSH PRIVATE KEY-----\n"})
    U.install_key(root, dest)
    assert dest.is_file() and not (root / f"{U.KEY_NAME}.txt").exists()


def test_deploy_key_is_never_committed():
    ignored = (Path(U.ROOT) / ".gitignore").read_text().split()
    assert "deploy_key*" in ignored or U.KEY_NAME in ignored


def test_ssh_fallback_to_443_and_https(tmp_path, zip_copy, remote, monkeypatch):
    key = tmp_path / "deploy_key"
    key.write_text("dummy key")

    calls = []
    orig_fetch_with_client = U._fetch_with_client

    def mock_fetch_with_client(client, path, repo, desc):
        calls.append(desc)
        if desc in ("SSH github.com:22", "SSH ssh.github.com:443"):
            raise ConnectionError(f"{desc} blocked")
        from dulwich.client import get_transport_and_path
        c, p = get_transport_and_path(str(remote))
        return orig_fetch_with_client(c, p, repo, desc)

    monkeypatch.setattr(U, "_fetch_with_client", mock_fetch_with_client)
    U.update(zip_copy, key_file=key)

    assert calls == ["SSH github.com:22", "SSH ssh.github.com:443", "anonymous HTTPS"]


def test_remote_choice(tmp_path):
    from dulwich.client import SSHGitClient
    client, path = U.remote(None, tmp_path / "missing-key")
    assert type(client).__name__.endswith("HttpGitClient") and "orimaimon/HalgOM1" in path
    key = tmp_path / U.KEY_NAME
    key.write_text("x")
    client, path = U.remote(None, key)
    assert isinstance(client, SSHGitClient) and path == U.SSH_PATH


def test_github_host_keys_match_published_fingerprints():
    import base64
    import hashlib
    published = {"ssh-ed25519": "+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU",
                 "ecdsa-sha2-nistp256": "p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM",
                 "ssh-rsa": "uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s"}
    for kind, b64 in U.GITHUB_HOST_KEYS.items():
        digest = base64.b64encode(hashlib.sha256(base64.b64decode(b64)).digest()).decode().rstrip("=")
        assert digest == published[kind], kind


def test_repo_without_main_and_with_other_branches_is_skipped(remote, tmp_path):
    feature_path = tmp_path / "feature_clone"
    feature_path.mkdir()
    with Repo.init(str(feature_path)) as r:
        write(feature_path, {"feat.txt": "work\n"})
        porcelain.add(r, paths=[str(feature_path / "feat.txt")])
        c = porcelain.commit(r, message=b"feature commit", author=b"t <t@t>", committer=b"t <t@t>")
        r.refs[b"refs/heads/feature"] = c
        r.refs.set_symbolic_ref(b"HEAD", b"refs/heads/feature")
        del r.refs[U.MAIN]

    assert U.update(feature_path, url=str(remote)) == U.NOTHING
    assert (feature_path / "feat.txt").read_text() == "work\n"
    with Repo(str(feature_path)) as r:
        assert U.MAIN not in r.refs
        assert r.refs.get_symrefs().get(b"HEAD") == b"refs/heads/feature"


def test_fetch_timeout_skips_update(remote, zip_copy, monkeypatch):
    import time
    U.update(zip_copy, url=str(remote))
    commit(remote, {"a.txt": "v2\n"}, "v2")

    monkeypatch.setattr(U, "FETCH_DEADLINE", 0.1)

    def slow_worker(*args, **kwargs):
        time.sleep(1.0)
        return b"0" * 20

    monkeypatch.setattr(U, "_fetch_main_worker", slow_worker)
    assert U.update(zip_copy, url=str(remote)) == U.NOTHING


def test_interrupted_checkout_recovery(remote, zip_copy, monkeypatch):
    U.update(zip_copy, url=str(remote))
    new = commit(remote, {"a.txt": "v2\n"}, "v2")

    orig_reset = U.porcelain.reset
    def bad_reset(repo, mode, sha):
        orig_reset(repo, mode, sha)
        raise PermissionError("file locked by antivirus")

    monkeypatch.setattr(U.porcelain, "reset", bad_reset)
    res = U.main(["--root", str(zip_copy), "--url", str(remote)])
    assert res == U.UPDATED
    assert (zip_copy / ".git" / "HALGOM_CHECKOUT").is_file()

    monkeypatch.undo()
    res2 = U.main(["--root", str(zip_copy), "--url", str(remote)])
    assert res2 == U.UPDATED
    assert not (zip_copy / ".git" / "HALGOM_CHECKOUT").exists()
    assert (zip_copy / "a.txt").read_text() == "v2\n"

    v3 = commit(remote, {"a.txt": "v3\n"}, "v3")
    assert U.update(zip_copy, url=str(remote)) == U.UPDATED
    assert head(zip_copy) == v3
    assert (zip_copy / "a.txt").read_text() == "v3\n"


def test_ssh_channel_reads_exact_lengths():
    class FakeChannel:
        def __init__(self, data):
            self.data = data
        def setblocking(self, flag):
            pass
        def recv(self, n):
            out, self.data = self.data[:min(n, 3)], self.data[min(n, 3):]   # short reads, like a socket
            return out
    ch = U._Channel(None, FakeChannel(b"0123456789"))
    assert ch.read(5) == b"01234" and ch.read() == b"56789" and ch.read(4) == b""
