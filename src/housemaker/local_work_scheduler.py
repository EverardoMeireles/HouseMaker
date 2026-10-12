# ### Imports ###
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

# ### Constants ###
# Three local stages preserve useful throughput for multi-object batches while
# preventing an arbitrary number of 2K/4K texture conversions from competing
# for memory at once.
MAX_CONCURRENT_LOCAL_STAGES = 3
DEFAULT_SLOT_POLL_SECONDS = 0.05


# ### Shared state ###
_LOCAL_WORK_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_LOCAL_STAGES)


# ### Exceptions ###
class LocalWorkCancelled(RuntimeError):
    """Raised when queued local processing is cancelled before it starts."""


# ### Scheduling helpers ###
_Result = TypeVar("_Result")


def run_bounded_local_work(
    operation: Callable[[], _Result],
    *,
    cancel_requested: Callable[[], bool] | None = None,
    poll_seconds: float = DEFAULT_SLOT_POLL_SECONDS,
) -> _Result:
    """Run memory-heavy local work without oversubscribing CPU and RAM.

    Provider/network requests remain independently concurrent. Only the local
    decoding, mesh processing, and texture-variant stages share this small
    process-wide lane.
    """

    cancellation_check = cancel_requested or (lambda: False)
    acquired = False
    while not acquired:
        if cancellation_check():
            raise LocalWorkCancelled
        acquired = _LOCAL_WORK_SLOTS.acquire(timeout=max(0.001, poll_seconds))
    try:
        if cancellation_check():
            raise LocalWorkCancelled
        return operation()
    finally:
        _LOCAL_WORK_SLOTS.release()
