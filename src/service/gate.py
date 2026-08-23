"""Bounded inference concurrency: one active slot, finite queue, honest busy.

Exactly one inference executes at a time. An arriving request is admitted
into the bounded waiting queue only while free capacity remains; once the
active slot is busy and the queue is exhausted, new arrivals fail fast with
the frozen ``service_busy`` 503 decision and a ``Retry-After`` header.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .errors import ServiceError

__all__ = ["InferenceGate"]


class InferenceGate:
    """Async single-slot gate with an operator-tunable queue depth.

    Production runs exactly one event loop (uvicorn workers=1). For test
    transports that serialize requests through successive event loops, the
    gate adopts a new loop only while fully quiescent; sharing it across
    concurrent loops is rejected instead of silently corrupting state.
    """

    def __init__(self, queue_depth: int = 0, retry_after_seconds: int = 5) -> None:
        if queue_depth < 0:
            raise ValueError("queue_depth must be >= 0")
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be >= 0")
        self.queue_depth = queue_depth
        self.retry_after_seconds = retry_after_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._condition = asyncio.Condition()
        self._active = False
        self._waiting = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire the single inference slot or fail fast with ``service_busy``."""
        await self._admit()
        try:
            yield
        finally:
            await self._release()

    def _adopt_running_loop_if_quiescent(self) -> None:
        running = asyncio.get_running_loop()
        if self._loop is running:
            return
        if self._active or self._waiting > 0:
            raise ServiceError(
                "inference gate shared across concurrent event loops",
                code="internal_error",
            )
        self._loop = running
        self._condition = asyncio.Condition()

    async def _admit(self) -> None:
        self._adopt_running_loop_if_quiescent()
        async with self._condition:
            queued = False
            if self._active or self._waiting > 0:
                if self._waiting >= self.queue_depth:
                    raise ServiceError(
                        "the service is busy processing another request",
                        code="service_busy",
                        headers={"Retry-After": str(self.retry_after_seconds)},
                    )
                self._waiting += 1
                queued = True
            try:
                await self._condition.wait_for(lambda: not self._active)
            finally:
                if queued:
                    self._waiting -= 1
            self._active = True

    async def _release(self) -> None:
        async with self._condition:
            self._active = False
            self._condition.notify_all()
