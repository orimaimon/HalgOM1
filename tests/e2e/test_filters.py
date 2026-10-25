"""E2E Playwright tests for screener filtering and ticker search."""
import os
import socket
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect

from screener.data.store import Store
from screener.data.tradingview import clean
from tests.test_screener import make_raw


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def streamlit_app(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("e2e_data")
    store = Store(data_dir)
    raw = clean(make_raw(300, seed=42).assign(is_primary=True, currency="USD"))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    now = dt.datetime.now().astimezone()
    store.save(raw, now)

    snap_dir = data_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    port = get_free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)

    runner_code = f"""
import sys, os
from pathlib import Path
import screener.data.store as store_mod
import screener.presets as P
import screener.portfolio as PF
import screener.export as export
import screener.data.tradingview as tv
import pandas as pd

data_dir = Path(r"{data_dir}")
store_mod.DATA_DIR = data_dir
P.PRESETS_PATH = data_dir / 'presets.json'
PF.PORTFOLIOS_PATH = data_dir / 'portfolios.json'
export.EXPORT_DIR = data_dir / 'exports'

import streamlit.web.cli as stcli
sys.argv = ['streamlit', 'run', 'screener/app.py', '--server.port={port}', '--server.headless=true', '--browser.gatherUsageStats=false']
stcli.main()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", runner_code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )

    url = f"http://127.0.0.1:{port}"
    start = time.time()
    server_ready = False
    while time.time() - start < 30:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                server_ready = True
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)

    if not server_ready:
        proc.kill()
        out, err = proc.communicate()
        raise RuntimeError(f"Streamlit server failed to start on port {port}:\nSTDOUT: {out.decode()}\nSTDERR: {err.decode()}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_screener_filter_interaction_and_search(page: Page, streamlit_app: str):
    url = streamlit_app
    page.goto(url)

    # Navigate to Screener page via segmented control
    page.get_by_text("סורק", exact=True).click()

    # Verify results count element is present
    results_locator = page.locator(".hs-count")
    expect(results_locator).to_be_visible(timeout=15000)

    initial_text = results_locator.inner_text()
    # e.g., "207 תוצאות" (due to default hide_junk=True filtering 300 rows)
    assert "תוצאות" in initial_text
    initial_count = int(initial_text.split()[0].replace(",", ""))
    assert initial_count > 0

    # 1. Search test
    search_input = page.get_by_placeholder("סימול / חברה, או רשימה: AAPL, MSFT")
    search_input.fill("T001")
    search_input.press("Enter")

    # Verify table updates and result count changes to exact match (1 result)
    expect(results_locator).to_contain_text("1 תוצאות", timeout=10000)

    # 2. Clear search to test filter interaction
    search_input.fill("")
    search_input.press("Enter")
    expect(results_locator).to_contain_text(f"{initial_count:,} תוצאות", timeout=10000)

    # 3. Filter test: open filter expander and select Market Cap option (in default Descriptive tab)
    filter_expander = page.locator("[data-testid='stExpander'] summary").get_by_text("פילטרים", exact=False)
    filter_expander.click()

    # Find Market Cap selectbox ("שווי שוק")
    mcap_select = page.get_by_label("שווי שוק", exact=True)
    mcap_select.click()

    # Select "ענק ($200B+)"
    page.locator("[role='option']").filter(has_text="ענק ($200B+)").click()

    # Assert row count changes and table updates
    expect(results_locator).not_to_contain_text(f"{initial_count:,} תוצאות", timeout=10000)
    filtered_text = results_locator.inner_text()
    filtered_count = int(filtered_text.split()[0].replace(",", ""))
    assert filtered_count < initial_count

    # 4. Verify filter reset works
    reset_btn = page.get_by_role("button", name="איפוס")
    reset_btn.click()

    # Verify search input is cleared, filter option reset, and count back to initial count
    expect(results_locator).to_contain_text(f"{initial_count:,} תוצאות", timeout=10000)
    expect(search_input).to_have_value("")
