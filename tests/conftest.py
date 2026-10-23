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


@pytest.fixture(autouse=True)
def _offline_and_private_data(tmp_path_factory, monkeypatch):
    """Tests never touch the network or the real _data folder (CLAUDE.md).

    The app tests render the track record panel, which asks prices.ensure() for SPY/QQQ: without this they
    downloaded from Yahoo on every run and wrote into screener/_data/history. A test that needs its own data
    folder still monkeypatches it (the later monkeypatch wins)."""
    from screener.data import prices, store
    data = tmp_path_factory.mktemp("data")
    monkeypatch.setattr(store, "DATA_DIR", data)
    monkeypatch.setattr(prices, "DATA_DIR", data / "history")
    monkeypatch.setattr(prices, "STATE_FILE", data / "history" / "update_state.json")

    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None) or _is_local(address):
            return real_connect(self, address)
        raise OSError(f"tests must not use the network (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", connect)
