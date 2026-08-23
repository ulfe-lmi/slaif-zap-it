"""Fail-closed device checks for the physically masked GPU1 runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "DeviceGuardError",
    "DeviceReport",
    "inspect_visible_device",
    "launch_environment",
    "require_launch_environment",
]


class DeviceGuardError(RuntimeError):
    """Raised when strict GPU invariants cannot be proven."""


@dataclass(frozen=True)
class DeviceReport:
    """Sanitized visible-device facts suitable for readiness and evidence."""

    mode: str
    available: bool
    visible_count: int
    logical_index: int | None
    name: str | None
    uuid: str | None
    total_memory_mib: int | None


def launch_environment(physical_gpu_index: int = 1) -> dict[str, str]:
    """Return the exact environment required before importing CUDA libraries."""
    if physical_gpu_index < 0:
        raise ValueError("physical_gpu_index must be non-negative")
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(physical_gpu_index),
    }


def require_launch_environment(
    environ: Mapping[str, str] | None = None,
    *,
    physical_gpu_index: int = 1,
) -> None:
    """Refuse strict operation unless the process inherited the GPU1 mask."""
    env = os.environ if environ is None else environ
    expected = launch_environment(physical_gpu_index)
    for key, value in expected.items():
        if env.get(key) != value:
            raise DeviceGuardError(f"{key} must be {value!r} for strict GPU mode")


def _normalize_uuid(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def _device_uuid(cuda: Any, properties: Any, uuid_provider: Any) -> str | None:
    if uuid_provider is not None:
        value = uuid_provider()
        return _normalize_uuid(value)
    value = getattr(properties, "uuid", None)
    if value:
        return _normalize_uuid(value)
    getter = getattr(cuda, "get_device_uuid", None)
    if getter is not None:
        value = getter(0)
        return _normalize_uuid(value)
    return None


def inspect_visible_device(
    torch_module: Any,
    *,
    expected_uuid: str | None,
    strict: bool = True,
    uuid_provider: Any = None,
) -> DeviceReport:
    """Inspect the already-masked PyTorch view and fail closed when required.

    ``torch_module`` is injectable so wrong-UUID, wrong-count and CPU-mode
    behavior can be tested without importing real torch.  In production the
    optional ``uuid_provider`` is backed by a masked ``nvidia-smi`` probe because
    UUID exposure on ``torch.cuda.DeviceProperties`` varies by torch release.
    """
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        if strict:
            raise DeviceGuardError("PyTorch CUDA metadata is unavailable")
        return DeviceReport("cpu", False, 0, None, None, None, None)

    available = bool(cuda.is_available())
    count = int(cuda.device_count())
    if not strict and (not available or count == 0):
        return DeviceReport("cpu", available, count, None, None, None, None)
    if not available:
        raise DeviceGuardError("CUDA is unavailable in strict GPU mode")
    if count != 1:
        raise DeviceGuardError(f"strict GPU mode requires exactly one visible device, got {count}")
    if strict and not expected_uuid:
        raise DeviceGuardError("strict GPU mode requires a pinned expected GPU UUID")

    try:
        name = str(cuda.get_device_name(0))
        properties = cuda.get_device_properties(0)
    except Exception as exc:  # pragma: no cover - real torch error surface
        raise DeviceGuardError("visible CUDA device metadata could not be read") from exc

    uuid = _device_uuid(cuda, properties, uuid_provider)
    if strict and not uuid:
        raise DeviceGuardError("strict GPU mode could not prove the visible GPU UUID")
    normalized_expected = _normalize_uuid(expected_uuid)
    if normalized_expected and uuid != normalized_expected:
        raise DeviceGuardError("visible GPU UUID does not match the pinned target")

    total_bytes = getattr(properties, "total_memory", None)
    total_mib = int(round(int(total_bytes) / (1024 * 1024))) if total_bytes else None
    return DeviceReport("gpu", True, count, 0, name, uuid, total_mib)
