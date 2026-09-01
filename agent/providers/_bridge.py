"""Shared streaming bridge for provider worker threads.

Both providers (OpenAI-compatible, Anthropic) run their sync SDK/HTTP
streaming on a daemon worker thread and bridge parsed chunks to the async
generator through a queue. This module encapsulates that machinery — the
queue, cancel flag, thread start, and the blocking poll loop — so
cancellation and shutdown semantics live in one place instead of being
copy-pasted across providers.
"""

from __future__ import annotations

import asyncio
import queue
import threading


class StreamBridge:
    """Bridges a sync worker thread's (kind, payload) chunks to the consumer.

    The worker calls :meth:`put`; the consumer awaits :meth:`next`, which
    returns a ``(kind, payload)`` tuple. ``kind`` is one of:

    - a provider-specific chunk kind (``"chunk"``/``"event"``) to process;
    - ``"end"`` — the worker finished (or the executor shut down);
    - ``"error"`` — the worker enqueued an exception;
    - ``"aborted"`` — the caller's signal was cancelled (sets the cancel flag).
    """

    def __init__(self, poll: float = 0.2) -> None:
        self.chunks: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_flag = threading.Event()
        self.poll = poll

    def start(self, target, args: tuple = ()) -> None:
        """Run ``target(*args)`` on a daemon worker thread."""
        threading.Thread(target=target, args=args, daemon=True,
                         name="llm-stream").start()

    def put(self, kind: str, payload: object) -> None:
        self.chunks.put((kind, payload))

    async def next(self, signal) -> tuple[str, object]:
        """Next (kind, payload), blocking the async side without the loop.

        On cancellation it sets the cancel flag (so the worker stops) and
        returns ``("aborted", None)``. On executor shutdown it returns
        ``("end", None)`` so the consumer stops cleanly.
        """
        while True:
            try:
                return await asyncio.to_thread(self.chunks.get, True, self.poll)
            except queue.Empty:
                if signal and signal.is_cancelled():
                    self.cancel_flag.set()
                    return ("aborted", None)
                continue
            except Exception:  # noqa: BLE001 — executor shutdown mid-stream
                return ("end", None)
