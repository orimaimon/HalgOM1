"""install_screener.bat: the one-file installer (the real run is the installer job in tests.yml)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = (ROOT / "install_screener.bat").read_bytes()
TEXT = BAT.decode("ascii")                        # cmd breaks on non-ASCII


def test_installer_is_ascii_crlf():
    assert b"\n" not in BAT.replace(b"\r\n", b"")


def test_installer_labels_and_comments():
    labels = set(re.findall(r"^:(\w+)", TEXT, re.M))
    targets = set(re.findall(r"(?:goto|call) :(\w+)", TEXT)) - {"eof"}
    assert targets <= labels, targets - labels
    for line in TEXT.splitlines():               # "rem-" (no space) is not a comment in cmd
        assert not re.match(r"\s*rem\S", line, re.I), line


def test_installer_has_no_blocks():
    """Paths (%EXISTING%, %HALGOM_DIR%) may contain ")" or "&": no parenthesized blocks, paths quoted."""
    for line in TEXT.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(("rem", "for ", "set ", "powershell")):
            continue
        assert not stripped.endswith("(") and stripped != ")", line
    for var in ("%EXISTING%", "%HALGOM_DIR%"):
        for m in re.finditer(re.escape(var), TEXT):
            assert TEXT[m.start() - 1] == '"', TEXT[m.start() - 20:m.end() + 5]


def test_installer_downloads_main_and_copies_the_launcher_last():
    assert "https://github.com/orimaimon/HalgOM1/archive/refs/heads/main.zip" in TEXT
    from screener import update
    assert update.HTTPS_URL == "https://github.com/orimaimon/HalgOM1.git"     # same repo as the updater
    assert 'set "HALGOM_DIR=%LOCALAPPDATA%\\HalgOM"' in TEXT
    # run_screener.bat marks a finished install: everything else is copied before it
    others = TEXT.index("Where-Object { $_.Name -ne 'run_screener.bat' }")
    assert others < TEXT.index("Copy-Item -LiteralPath (Join-Path $src 'run_screener.bat')")
    assert TEXT.index('if exist "%HALGOM_DIR%\\run_screener.bat" goto :start') < others
    # the PowerShell script reads paths from environment variables: a quote in a user name cannot break it
    ps = "".join(m[1] for ln in TEXT.splitlines() if (m := re.fullmatch(r'set "PS=(?:%PS%)?(.*)"', ln)))
    assert '"' not in ps and "%" not in ps and "$env:HALGOM_TMP" in ps and "$env:HALGOM_DIR" in ps


def test_installer_starts_an_existing_install_first():
    """An install from a ZIP elsewhere (its desktop shortcut) is started, not installed a second time."""
    shortcut = TEXT.index("'HalgOM Screener.lnk'")
    assert TEXT.index("if defined HALGOM_DIR goto :folder") < shortcut < TEXT.index('call "%EXISTING%"')
    launcher = (ROOT / "run_screener.bat").read_text(encoding="ascii")
    assert "'HalgOM Screener.lnk'" in launcher and "$s.TargetPath = '%~f0'" in launcher


def test_installer_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "install_screener.bat" in readme
    assert "releases/download/installer/install_screener.bat" in readme
    workflow = (ROOT / ".github" / "workflows" / "installer-release.yml").read_text(encoding="utf-8")
    assert "install_screener.bat" in workflow and "gh release upload installer" in workflow
