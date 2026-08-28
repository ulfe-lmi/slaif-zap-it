"""Single-process explicit model lifecycle control.

The controller is deliberately independent of HTTP and model implementation
details.  It owns the public lifecycle state, serializes management work onto
one control executor, and coordinates the asynchronous inference gate with
the synchronous registry load/unload operations.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Lock
from typing import Any, Callable

from .errors import ServiceError
from .gate import InferenceGate

__all__ = ["LifecycleState", "ModelLifecycleController"]


class LifecycleState(str, Enum):
    """Stable and transition states exposed by the management subset."""

    UNAVAILABLE = "UNAVAILABLE"
    LOADING = "LOADING"
    READY = "READY"
    UNLOADING = "UNLOADING"


class ModelLifecycleController:
    """Coordinate one fixed registry with one inference admission authority."""

    def __init__(
        self,
        *,
        registry: Any,
        gate: InferenceGate,
        control_executor: ThreadPoolExecutor,
        operation_timeout_seconds: float = 600.0,
        drain_timeout_seconds: float = 120.0,
        metrics: Any | None = None,
    ) -> None:
        if operation_timeout_seconds <= 0 or drain_timeout_seconds <= 0:
            raise ValueError("model-control timeouts must be positive")
        self.registry = registry
        self.gate = gate
        self.control_executor = control_executor
        self.operation_timeout_seconds = float(operation_timeout_seconds)
        self.drain_timeout_seconds = float(drain_timeout_seconds)
        self.metrics = metrics
        self._lock = Lock()
        self._state = LifecycleState.READY if bool(registry.ready) else LifecycleState.UNAVAILABLE
        self._reason = (
            "model is ready" if self._state is LifecycleState.READY else "model is unavailable"
        )
        self._last_error = False
        self._shutting_down = False
        self._control_future: Any | None = None
        self._set_metrics_state(self._state)

    @property
    def state(self) -> LifecycleState:
        with self._lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        return self.state is LifecycleState.READY and bool(self.registry.ready)

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def snapshot(
        self, *, ready_only: bool = False, model_name: str = "zap-it-1"
    ) -> list[dict[str, str]]:
        """Return the bounded repository-index representation."""
        state = self.state
        if ready_only and state is not LifecycleState.READY:
            return []
        reason = self.reason
        return [{"name": model_name, "state": state.value, "reason": reason}]

    def readiness_detail(self) -> str:
        state = self.state
        if state is LifecycleState.READY and self.registry.ready:
            return "fixed model is ready"
        return self.reason

    def _set_metrics_state(self, state: LifecycleState) -> None:
        if self.metrics is not None:
            self.metrics.set_model_lifecycle_state(state.value)
            self.metrics.model_loaded.set(1 if state is LifecycleState.READY else 0)

    def _refresh_gpu_metrics(self) -> None:
        if self.metrics is not None:
            self.metrics.update_gpu()

    def _set_state(self, state: LifecycleState, reason: str, *, error: bool = False) -> None:
        with self._lock:
            self._state = state
            self._reason = reason
            self._last_error = error
        self._set_metrics_state(state)

    def _begin(self, operation: str, transition: LifecycleState) -> bool:
        with self._lock:
            if self._shutting_down:
                raise ServiceError("model control is shutting down", code="model_control_busy")
            if operation == "load" and self._state is LifecycleState.READY:
                return False
            if operation == "unload" and self._state is LifecycleState.UNAVAILABLE:
                return False
            if self._state in {LifecycleState.LOADING, LifecycleState.UNLOADING}:
                raise ServiceError(
                    "another model-control operation is in progress", code="model_control_busy"
                )
            self._state = transition
            self._reason = "model is loading" if operation == "load" else "model is unloading"
            self._last_error = False
        self._set_metrics_state(transition)
        return True

    def _observe_operation(self, operation: str, outcome: str, started: float) -> None:
        if self.metrics is not None:
            self.metrics.observe_model_control_operation(
                operation, outcome, time.monotonic() - started
            )

    async def _run_control(self, call: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.control_executor, call)
        with self._lock:
            self._control_future = future
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.operation_timeout_seconds
            )
        except asyncio.TimeoutError:
            # A thread cannot be safely cancelled.  Finish it before settling
            # the public state so a request timeout never leaves LOADING or
            # UNLOADING behind.
            try:
                await asyncio.shield(future)
            except BaseException:
                pass
            raise
        except asyncio.CancelledError:
            try:
                await asyncio.shield(future)
            except BaseException:
                pass
            raise
        finally:
            with self._lock:
                if self._control_future is future:
                    self._control_future = None

    async def _drain(self) -> bool:
        """Drain without allowing client cancellation to abandon the gate."""
        task = asyncio.create_task(self.gate.pause_and_drain("model is unloading"))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.drain_timeout_seconds)
            return False
        except asyncio.TimeoutError:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
        except asyncio.CancelledError:
            await asyncio.shield(task)
            return True

    async def load(self) -> None:
        """Load every pinned holder, returning only in a stable state."""
        started = time.monotonic()
        if not self._begin("load", LifecycleState.LOADING):
            self._observe_operation("load", "success", started)
            return
        self.gate.pause_now("model is loading")
        try:
            await self._run_control(self.registry.load)
            if not self.registry.ready:
                self._set_state(
                    LifecycleState.UNAVAILABLE,
                    "model load failed",
                    error=True,
                )
                self._observe_operation("load", "failure", started)
                raise ServiceError("model load failed", code="model_control_failure")
            self._set_state(LifecycleState.READY, "fixed model is ready")
            await self.gate.resume()
            self._refresh_gpu_metrics()
            self._observe_operation("load", "success", started)
        except asyncio.TimeoutError as exc:
            # Registry.load is allowed to finish in the control thread.  If it
            # did become ready, release those holders before exposing cold
            # state; never leave a timed-out request with hidden residency.
            if self.registry.ready:
                try:
                    await self._run_control(self.registry.unload)
                except BaseException:
                    pass
            self._set_state(LifecycleState.UNAVAILABLE, "model load timed out", error=True)
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("load", "timeout", started)
            raise ServiceError("model load operation timed out", code="timeout") from exc
        except asyncio.CancelledError:
            if self.registry.ready:
                try:
                    await self._run_control(self.registry.unload)
                except BaseException:
                    pass
            self._set_state(LifecycleState.UNAVAILABLE, "model load cancelled", error=True)
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("load", "cancelled", started)
            raise
        except ServiceError:
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            raise
        except Exception as exc:
            self._set_state(LifecycleState.UNAVAILABLE, "model load failed", error=True)
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("load", "failure", started)
            raise ServiceError("model load failed", code="model_control_failure") from exc

    async def unload(self) -> None:
        """Pause admission, drain active inference, then prove cold memory."""
        started = time.monotonic()
        if not self._begin("unload", LifecycleState.UNLOADING):
            self._observe_operation("unload", "success", started)
            return
        drain_started = time.monotonic()
        cancelled_by_client = False
        try:
            cancelled_by_client = await self._drain()
            if self.metrics is not None:
                self.metrics.observe_model_control_drain(time.monotonic() - drain_started)
        except asyncio.TimeoutError as exc:
            self._set_state(LifecycleState.READY, "fixed model is ready")
            await self.gate.resume()
            self._observe_operation("unload", "drain_timeout", started)
            raise ServiceError("in-flight inference did not drain", code="timeout") from exc

        try:
            await self._run_control(self.registry.unload)
            if self.registry.ready:
                self._set_state(LifecycleState.READY, "fixed model is ready")
                await self.gate.resume()
                self._observe_operation("unload", "failure", started)
                raise ServiceError(
                    "model unload did not reach cold state", code="model_control_failure"
                )
            self._set_state(LifecycleState.UNAVAILABLE, "model is unavailable")
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            if cancelled_by_client:
                self._set_state(LifecycleState.UNAVAILABLE, "model unload cancelled", error=True)
                self._observe_operation("unload", "cancelled", started)
                raise asyncio.CancelledError
            self._observe_operation("unload", "success", started)
        except asyncio.TimeoutError as exc:
            # The control thread is finished by _run_control before this path;
            # preserve its measured registry verdict and never claim success.
            if self.registry.ready:
                self._set_state(LifecycleState.READY, "fixed model is ready")
                await self.gate.resume()
            else:
                self._set_state(LifecycleState.UNAVAILABLE, "model unload timed out", error=True)
                self.gate.pause_now("model is not ready")
            self._observe_operation("unload", "timeout", started)
            raise ServiceError("model unload operation timed out", code="timeout") from exc
        except asyncio.CancelledError:
            if self.registry.ready:
                self._set_state(LifecycleState.READY, "fixed model is ready")
                await self.gate.resume()
            else:
                self._set_state(LifecycleState.UNAVAILABLE, "model unload cancelled", error=True)
                self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("unload", "cancelled", started)
            raise
        except ServiceError:
            self._set_state(LifecycleState.UNAVAILABLE, "model unload failed", error=True)
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("unload", "failure", started)
            raise
        except Exception as exc:
            self._set_state(LifecycleState.UNAVAILABLE, "model unload failed", error=True)
            self.gate.pause_now("model is not ready")
            self._refresh_gpu_metrics()
            self._observe_operation("unload", "failure", started)
            raise ServiceError("model unload failed", code="model_control_failure") from exc

    async def shutdown(self) -> None:
        """Stop admissions and finish a safe unload during process shutdown."""
        with self._lock:
            self._shutting_down = True
            control_future = self._control_future
        if control_future is not None:
            try:
                await asyncio.shield(control_future)
            except BaseException:
                pass
        # The in-flight control call has settled.  Temporarily allow the
        # controller's own safe unload path while no new management work can
        # enter during lifespan teardown.
        if self.registry.ready:
            with self._lock:
                self._shutting_down = False
            try:
                await self.unload()
            except BaseException:
                pass
        elif self.state in {LifecycleState.LOADING, LifecycleState.UNLOADING}:
            self._set_state(LifecycleState.UNAVAILABLE, "model is unavailable", error=True)
        self.gate.pause_now("model is not ready")
        with self._lock:
            self._shutting_down = True
