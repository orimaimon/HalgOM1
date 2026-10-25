import os
import socket
import subprocess
import sys
import time
import urllib.request
import pytest


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_app():
    port = find_free_port()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    app_path = os.path.join(repo_root, "screener", "app.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.headless=true",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=repo_root,
    )

    url = f"http://127.0.0.1:{port}"
    timeout = 30
    start_time = time.time()
    started = False

    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"Streamlit process exited prematurely with code {proc.returncode}.\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    started = True
                    break
        except Exception:
            time.sleep(0.5)

    if not started:
        proc.kill()
        proc.wait()
        raise TimeoutError(f"Streamlit failed to boot on {url} within {timeout} seconds.")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
