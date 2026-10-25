"""E2E Smoke Test for Main Screener Dashboard using Playwright."""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

playwright = pytest.importorskip("playwright")
from playwright.sync_api import Page, expect

from screener.data.store import Store, session_date
from screener.data.tradingview import clean
from tests.test_screener import ROOT, make_raw


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def streamlit_app(tmp_path_factory):
    """Launches Streamlit app on a free port with pre-populated synthetic cache/snapshot."""
    data_dir = tmp_path_factory.mktemp("e2e_data")

    raw = clean(make_raw(300).assign(is_primary=True, currency="USD"))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    store = Store(data_dir)
    fetched_at = dt.datetime.now().astimezone()
    store.save(raw, fetched_at)

    port = get_free_port()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    entry_script = data_dir / "run_app.py"
    entry_script.write_text(f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{ROOT}")

import screener.data.store as store_mod
store_mod.DATA_DIR = Path(r"{data_dir}")

import screener.presets as P
import screener.portfolio as PF
import screener.export as export

P.PRESETS_PATH = Path(r"{data_dir}") / "presets.json"
PF.PORTFOLIOS_PATH = Path(r"{data_dir}") / "portfolios.json"
export.EXPORT_DIR = Path(r"{data_dir}") / "exports"

import screener.lifecycle as lifecycle
lifecycle.ensure_idle_watchdog_started = lambda: None

import screener.data.tradingview as tv
market = __import__("pandas").DataFrame({{"close": [500.0, 400, 80, 15], "change": 0.5, "sma20": [490.0, 390, 79, 16],
                       "sma50": 480.0, "sma200": [450.0, 350, 78, 17], "adx": [30.0, 20, 15, 10],
                       "di_plus": 25.0, "di_minus": 15.0}}, index=["SPY", "QQQ", "HYG", "VIX"])

tv.fetch_universe = lambda: sys.exit("Network call attempted in E2E test!")
tv.fetch_market = lambda: market

from screener.app import main
main()
""", encoding="utf-8")

    python_bin = sys.executable
    cmd = [
        python_bin, "-m", "streamlit", "run", str(entry_script),
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://127.0.0.1:{port}"
    start_time = time.time()
    server_ready = False
    while time.time() - start_time < 30:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                server_ready = True
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)

    if not server_ready:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Streamlit failed to start on port {port}:\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_main_screener_dashboard(streamlit_app: str, page: Page):
    page_errors: list[Exception] = []
    page.on("pageerror", lambda err: page_errors.append(err))

    page.goto(streamlit_app, wait_until="networkidle")

    # Assert main header renders
    brand = page.locator(".hs-brand")
    expect(brand).to_be_visible(timeout=15000)

    # Streamlit segmented control for page selection contains "סורק" (Screener)
    page.get_by_text("סורק").first.click()

    # Assert data table renders with visible rows
    table_container = page.locator(".stDataFrame, [data-testid='stDataFrame']")
    expect(table_container.first).to_be_visible(timeout=15000)

    # Ensure results count header is visible
    results_text = page.locator(".hs-count")
    expect(results_text).to_be_visible()

    # Assert 0 uncaught JS errors
    assert len(page_errors) == 0, f"Uncaught JS errors found on page: {page_errors}"
