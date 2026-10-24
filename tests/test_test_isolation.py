"""The test suite itself must stay offline and away from the real data folder (CLAUDE.md)."""
import importlib
import socket

import pytest


def test_importing_the_app_does_not_run_it(monkeypatch):
    import screener.data.tradingview as tv

    def must_not_run(*args, **kwargs):
        raise AssertionError("importing screener.app ran the app (fetched the universe)")

    monkeypatch.setattr(tv, "fetch_universe", must_not_run)
    monkeypatch.setattr(tv, "fetch_market", must_not_run)
    import screener.app as app
    importlib.reload(app)          # runs the module's top level again
    assert callable(app.main)


def test_network_is_blocked_in_tests():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="must not use the network"):
            s.connect(("93.184.216.34", 443))
    finally:
        s.close()
