from __future__ import annotations

import concurrent.futures
from typing import Any, Callable


def fetch_parallel(
    tasks: dict[str, Callable[[], Any]],
    max_workers: int = 4,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Executes a dictionary of named callable fetch functions concurrently in a ThreadPoolExecutor.

    Returns a dict mapping task names to their result or caught exception (including TimeoutError).
    """
    if not tasks:
        return {}

    results: dict[str, Any] = {}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_name = {
            executor.submit(fn): name for name, fn in tasks.items()
        }
        done, not_done = concurrent.futures.wait(
            future_to_name.keys(), timeout=timeout
        )

        for future in done:
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = exc

        for future in not_done:
            name = future_to_name[future]
            results[name] = concurrent.futures.TimeoutError(
                f"Task '{name}' timed out after {timeout} seconds"
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results
