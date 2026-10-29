from __future__ import annotations

import concurrent.futures
import time

from screener.data.fetcher import fetch_parallel


def test_successful_concurrent_execution():
    def fetch_market():
        return {"data": "market_ok"}

    def fetch_boi():
        return {"data": "boi_ok"}

    tasks = {
        "market": fetch_market,
        "boi": fetch_boi,
    }

    results = fetch_parallel(tasks)
    assert results == {
        "market": {"data": "market_ok"},
        "boi": {"data": "boi_ok"},
    }


def test_exception_isolation():
    def fetch_market():
        return "market_ok"

    def fetch_failing():
        raise ValueError("Provider connection failed")

    tasks = {
        "market": fetch_market,
        "failing": fetch_failing,
    }

    results = fetch_parallel(tasks)
    assert results["market"] == "market_ok"
    assert isinstance(results["failing"], ValueError)
    assert str(results["failing"]) == "Provider connection failed"


def test_timeout_handling():
    def fetch_fast():
        return "fast_ok"

    def fetch_slow():
        time.sleep(1.0)
        return "slow_ok"

    tasks = {
        "fast": fetch_fast,
        "slow": fetch_slow,
    }

    t0 = time.time()
    results = fetch_parallel(tasks, timeout=0.1)
    elapsed = time.time() - t0

    assert elapsed < 0.5
    assert results["fast"] == "fast_ok"
    assert isinstance(results["slow"], concurrent.futures.TimeoutError)


def test_timing_benchmark():
    def slow_task_1():
        time.sleep(0.2)
        return "res1"

    def slow_task_2():
        time.sleep(0.2)
        return "res2"

    tasks = {
        "t1": slow_task_1,
        "t2": slow_task_2,
    }

    t0 = time.time()
    results = fetch_parallel(tasks, max_workers=2, timeout=1.0)
    elapsed = time.time() - t0

    assert results == {"t1": "res1", "t2": "res2"}
    assert elapsed < 0.35


def test_empty_tasks():
    results = fetch_parallel({})
    assert results == {}
