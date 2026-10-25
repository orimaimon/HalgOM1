import datetime as dt
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from screener.data.store import Store
from screener.data.tradingview import clean
from tests.test_screener import make_raw


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_app(tmp_path_factory):
    """Start Streamlit server in headless mode with offline synthetic data."""
    data_dir = tmp_path_factory.mktemp("e2e_data")

    # Populate synthetic snapshot so app loads offline immediately
    raw = clean(make_raw(300).assign(is_primary=True, currency="USD"))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    store = Store(data_dir)
    store.save(raw, dt.datetime.now().astimezone())

    port = find_free_port()
    env = os.environ.copy()
    env["HALGOM_DATA_DIR"] = str(data_dir)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["SCREENER_NO_BROWSER"] = "1"
    env["SCREENER_NO_UPDATE"] = "1"

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

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://127.0.0.1:{port}"
    health_url = f"{url}/_stcore/health"
    start_time = time.time()
    up = False
    while time.time() - start_time < 30:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.read().decode("utf-8").strip() == "ok":
                    up = True
                    break
        except Exception:
            pass
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"Streamlit process exited prematurely:\n{stderr.decode('utf-8')}")
        time.sleep(0.5)

    if not up:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Streamlit server did not become healthy in 30s:\n{stderr.decode('utf-8')}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
