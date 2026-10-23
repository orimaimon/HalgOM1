import json
from datetime import datetime
from pathlib import Path

import pytest
from dulwich.repo import Repo

from screener.whatsnew import WHATSNEW_PATH, entries, mark_seen, unseen, version_label


def test_first_run_and_unseen_lifecycle(tmp_path):
    whatsnew_file = tmp_path / "whatsnew.json"
    seen_file = tmp_path / "whatsnew_seen.json"

    initial_content = {
        "version": "1.0",
        "entries": [
            {"id": "entry1", "date": "2026-10-23", "title": "כותרת 1", "text": "טקסט 1"}
        ]
    }
    whatsnew_file.write_text(json.dumps(initial_content), encoding="utf-8")

    # First run: seen file doesn't exist. unseen() should return [] and create seen_file
    assert not seen_file.exists()
    result = unseen(whatsnew_path=whatsnew_file, seen_path=seen_file)
    assert result == []
    assert seen_file.exists()

    # Add a new entry to whatsnew.json
    updated_content = {
        "version": "1.1",
        "entries": [
            {"id": "entry2", "date": "2026-10-24", "title": "כותרת 2", "text": "טקסט 2"},
            {"id": "entry1", "date": "2026-10-23", "title": "כותרת 1", "text": "טקסט 1"},
        ]
    }
    whatsnew_file.write_text(json.dumps(updated_content), encoding="utf-8")

    # unseen() should now return only entry2
    new_unseen = unseen(whatsnew_path=whatsnew_file, seen_path=seen_file)
    assert len(new_unseen) == 1
    assert new_unseen[0]["id"] == "entry2"

    # mark_seen() for entry2 (or all) should hide it
    mark_seen(ids=["entry2"], whatsnew_path=whatsnew_file, seen_path=seen_file)
    assert unseen(whatsnew_path=whatsnew_file, seen_path=seen_file) == []


def test_damaged_whatsnew_file(tmp_path):
    whatsnew_file = tmp_path / "whatsnew.json"
    whatsnew_file.write_text("invalid json content {{{", encoding="utf-8")

    assert entries(path=whatsnew_file) == []


def test_damaged_seen_file(tmp_path):
    whatsnew_file = tmp_path / "whatsnew.json"
    seen_file = tmp_path / "whatsnew_seen.json"

    content = {
        "version": "1.0",
        "entries": [
            {"id": "e1", "date": "2026-10-23", "title": "כותרת", "text": "תוכן"}
        ]
    }
    whatsnew_file.write_text(json.dumps(content), encoding="utf-8")
    seen_file.write_text("corrupted json file", encoding="utf-8")

    # Damaged seen file -> all entries are unseen (treated as empty, never as first run)
    result = unseen(whatsnew_path=whatsnew_file, seen_path=seen_file)
    assert len(result) == 1
    assert result[0]["id"] == "e1"


def test_version_label_with_and_without_git(tmp_path):
    whatsnew_file = tmp_path / "whatsnew.json"
    content = {
        "version": "2.5",
        "entries": [
            {"id": "e1", "date": "2026-10-23", "title": "כותרת", "text": "תוכן"}
        ]
    }
    whatsnew_file.write_text(json.dumps(content), encoding="utf-8")

    # 1. Test without .git directory
    no_git_dir = tmp_path / "no_git"
    no_git_dir.mkdir()
    label_no_git = version_label(whatsnew_path=whatsnew_file, root=no_git_dir)
    assert label_no_git == "גרסה 2.5 · 23/10/2026 · —"

    # 2. Test with a tiny dulwich repo in tmp_path (one commit)
    git_dir = tmp_path / "repo"
    git_dir.mkdir()
    with Repo.init(str(git_dir)) as repo:
        from dulwich.objects import Commit, Tree
        tree = Tree()
        repo.object_store.add_object(tree)
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test Author <test@example.com>"
        commit.commit_time = commit.author_time = 1792752000  # 2026-10-23 10:40:00 UTC
        commit.author_timezone = commit.commit_timezone = 0
        commit.message = b"Initial commit"
        repo.object_store.add_object(commit)
        repo.refs[b"refs/heads/main"] = commit.id
        repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")

        commit_sha = commit.id.decode("ascii")[:7]

    label_git = version_label(whatsnew_path=whatsnew_file, root=git_dir)
    assert label_git == f"גרסה 2.5 · 23/10/2026 · {commit_sha}"


def test_real_whatsnew_json_validity():
    assert WHATSNEW_PATH.exists()
    with open(WHATSNEW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "version" in data
    entries_list = data.get("entries", [])
    assert len(entries_list) >= 4

    ids = set()
    dates = []
    for entry in entries_list:
        eid = entry.get("id")
        assert eid, "entry missing id"
        assert eid not in ids, f"duplicate id: {eid}"
        ids.add(eid)

        date_str = entry.get("date")
        assert date_str, "entry missing date"
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        dates.append(parsed_date)

        title = entry.get("title", "")
        text = entry.get("text", "")
        assert title, "entry missing title"
        assert text, "entry missing text"

        # Every title and text must contain Hebrew letters (\u0590-\u05FF)
        assert any("\u0590" <= c <= "\u05ff" for c in title), f"title missing Hebrew: {title}"
        assert any("\u0590" <= c <= "\u05ff" for c in text), f"text missing Hebrew: {text}"

    # Verify newest first (dates non-increasing)
    for i in range(len(dates) - 1):
        assert dates[i] >= dates[i + 1], f"entries not ordered newest first: {dates[i]} < {dates[i+1]}"
