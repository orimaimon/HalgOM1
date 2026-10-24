import ipaddress
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _is_local(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    if host in ("localhost", "") or not isinstance(host, str):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# Tests never touch the network (CLAUDE.md). Installed when this file loads, so it also covers imports at
# collection time, before any fixture runs (`from screener import app` at the top of a test file once sent a
# real TradingView request on every run).
_real_connect = socket.socket.connect


def _offline_connect(self, address):
    if self.family == getattr(socket, "AF_UNIX", None) or _is_local(address):
        return _real_connect(self, address)
    raise OSError(f"tests must not use the network (tried {address!r})")


socket.socket.connect = _offline_connect


@pytest.fixture(autouse=True)
def _private_data(tmp_path_factory, monkeypatch):
    """Tests never write into the real screener/_data folder.

    The app tests render the track record panel, which asks prices.ensure() for SPY/QQQ: without this they
    wrote into screener/_data/history. A test that needs its own data folder still monkeypatches it (the
    later monkeypatch wins)."""
    from screener.data import prices, store
    data = tmp_path_factory.mktemp("data")
    monkeypatch.setattr(store, "DATA_DIR", data)
    monkeypatch.setattr(prices, "DATA_DIR", data / "history")
    monkeypatch.setattr(prices, "STATE_FILE", data / "history" / "update_state.json")
