"""Playwright E2E test verifying TradingView Chart widget/iframe loads correctly."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page  # noqa: E402


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def streamlit_server():
    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "screener/app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(cmd)
    url = f"http://127.0.0.1:{port}"
    health_url = f"{url}/_stcore/health"

    start_time = time.time()
    server_ready = False
    while time.time() - start_time < 30:
        try:
            req = urllib.request.urlopen(health_url, timeout=1)
            if req.status == 200:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not server_ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Streamlit server failed to start within 30 seconds")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_tradingview_chart_iframe_render(page: Page, streamlit_server: str) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(f"pageerror: {err}"))
    page.on("crash", lambda p: errors.append("page crashed"))

    # Navigate to app
    page.goto(streamlit_server)

    # Wait for Streamlit app to load
    page.wait_for_selector("text=HalgOM", timeout=15000)
    page.wait_for_timeout(1000)

    # Navigate to Screener tab
    screener_tab = page.get_by_text("סורק", exact=True)
    screener_tab.wait_for(state="visible", timeout=15000)
    screener_tab.click()

    # Wait for Screener results indicator
    results_indicator = page.get_by_text("תוצאות").or_(page.get_by_text("results")).first
    results_indicator.wait_for(state="visible", timeout=15000)

    # Expand "Fast chart review" / "דפדוף מהיר בגרפים"
    expander = page.get_by_text("דפדוף מהיר בגרפים").or_(page.get_by_text("Fast chart review")).first
    expander.wait_for(state="visible", timeout=15000)
    expander.click()

    # Locate outer st.iframe
    outer_iframe = page.locator("iframe").first
    outer_iframe.wait_for(state="visible", timeout=15000)

    outer_box = outer_iframe.bounding_box()
    assert outer_box is not None, "Outer iframe bounding_box is None"
    assert outer_box["width"] > 0, f"Outer iframe width non-positive: {outer_box['width']}"
    assert outer_box["height"] > 0, f"Outer iframe height non-positive: {outer_box['height']}"

    # Locate inner TradingView widget iframe embedded inside outer iframe
    inner_iframe = page.frame_locator("iframe").locator("iframe").first
    inner_iframe.wait_for(state="attached", timeout=15000)

    inner_box = inner_iframe.bounding_box()
    assert inner_box is not None, "Inner TradingView iframe bounding_box is None"
    assert inner_box["width"] > 0, f"Inner TradingView iframe width non-positive: {inner_box['width']}"
    assert inner_box["height"] > 0, f"Inner TradingView iframe height non-positive: {inner_box['height']}"

    # Verify no cross-origin frame crash errors occurred
    assert len(errors) == 0, f"Encountered errors: {errors}"
