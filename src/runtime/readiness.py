"""Readiness adapter joining the pinned device guard and runtime policy."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .device import DeviceGuardError, inspect_visible_device, require_launch_environment
from .strategy import RuntimePolicy, RuntimeReadiness

__all__ = ["make_readiness_provider"]


def make_readiness_provider(
    policy: RuntimePolicy,
    *,
    torch_module: Any | None = None,
    uuid_provider: Callable[[], str | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Callable[[], RuntimeReadiness]:
    """Build a service-compatible provider with fail-closed GPU checks."""

    def provider() -> RuntimeReadiness:
        if policy.strict_gpu:
            try:
                require_launch_environment(
                    environ,
                    physical_gpu_index=policy.physical_gpu_index,
                )
            except DeviceGuardError as exc:
                return RuntimeReadiness(False, str(exc))
        module = torch_module
        if module is None:
            try:
                import torch as module  # type: ignore[no-redef]
            except ImportError:
                return RuntimeReadiness(False, "PyTorch is unavailable")
        try:
            report = inspect_visible_device(
                module,
                expected_uuid=policy.expected_gpu_uuid,
                strict=policy.strict_gpu,
                uuid_provider=uuid_provider,
            )
        except DeviceGuardError as exc:
            return RuntimeReadiness(False, str(exc))
        return policy.readiness(report)

    return provider
