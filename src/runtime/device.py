"""Fail-closed device checks for the operator-masked GPU runtime."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "DeviceGuardError",
    "DeviceReport",
    "PhysicalGpuEvidence",
    "inspect_physical_gpu",
    "require_physical_gpu_match",
    "inspect_visible_device",
    "launch_environment",
    "parse_physical_gpu_index",
    "require_launch_environment",
]


class DeviceGuardError(RuntimeError):
    """Raised when strict GPU invariants cannot be proven."""


def parse_physical_gpu_index(value: str | None, *, default: int = 1) -> int:
    """Parse an operator GPU index as an ASCII non-negative decimal integer.

    The default is retained for Python compatibility with the historical
    physical-GPU1 deployment.  The strict shell launcher requires the
    environment variable explicitly before it invokes this module.
    """
    raw = str(default) if value is None else str(value).strip()
    if not raw or any(char < "0" or char > "9" for char in raw):
        raise ValueError("physical GPU index must be a non-negative decimal integer")
    return int(raw, 10)


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


@dataclass(frozen=True)
class PhysicalGpuEvidence:
    """Sanitized physical-card observation from a targeted nvidia-smi query."""

    physical_index: int
    uuid: str
    pci_bus_id: str
    name: str
    total_memory_mib: int
    used_memory_mib: int
    free_memory_mib: int
    compute_pids: tuple[int, ...] = ()


def _csv_fields(stdout: str) -> list[str]:
    line = next((item.strip() for item in stdout.splitlines() if item.strip()), "")
    return [item.strip() for item in line.split(",")]


def _run_smi(command: list[str], nvidia_smi: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run([nvidia_smi, *command], check=False, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeviceGuardError("nvidia-smi device evidence is unavailable") from exc


def inspect_physical_gpu(
    physical_index: int = 1,
    *,
    expected_uuid: str | None,
    nvidia_smi: str = "nvidia-smi",
    require_idle: bool = True,
) -> PhysicalGpuEvidence:
    """Observe exactly one physical GPU and fail closed on ambiguity/occupancy."""
    if physical_index < 0:
        raise ValueError("physical GPU index must be non-negative")
    result = _run_smi(
        [
            "--id",
            str(physical_index),
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        nvidia_smi,
    )
    if result.returncode != 0:
        raise DeviceGuardError("targeted physical GPU capacity evidence failed")
    fields = _csv_fields(result.stdout)
    if len(fields) != 7:
        raise DeviceGuardError("targeted physical GPU capacity evidence was ambiguous")
    try:
        index = int(fields[0])
        total = int(float(fields[4]))
        used = int(float(fields[5]))
        free = int(float(fields[6]))
    except (TypeError, ValueError) as exc:
        raise DeviceGuardError("targeted physical GPU capacity evidence was invalid") from exc
    uuid = _normalize_uuid(fields[1])
    if uuid is None or index != physical_index:
        raise DeviceGuardError("targeted physical GPU index/UUID evidence disagrees")
    normalized_expected = _normalize_uuid(expected_uuid)
    if normalized_expected and uuid != normalized_expected:
        raise DeviceGuardError("physical GPU UUID does not match the operator pin")
    if total <= 0 or used < 0 or free < 0:
        raise DeviceGuardError("targeted physical GPU memory evidence was invalid")

    processes = _run_smi(
        [
            "--id",
            str(physical_index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        nvidia_smi,
    )
    if processes.returncode != 0:
        raise DeviceGuardError("targeted physical GPU process evidence failed")
    pids: list[int] = []
    for raw in processes.stdout.splitlines():
        raw = raw.strip()
        if not raw or raw.lower().startswith("no running"):
            continue
        try:
            pids.append(int(raw.split(")[", 1)[0].strip()))
        except ValueError as exc:
            raise DeviceGuardError("targeted physical GPU process evidence was invalid") from exc
    if require_idle and pids:
        raise DeviceGuardError("target physical GPU has an unrelated compute process")
    return PhysicalGpuEvidence(
        physical_index=index,
        uuid=uuid,
        pci_bus_id=fields[2],
        name=fields[3],
        total_memory_mib=total,
        used_memory_mib=used,
        free_memory_mib=free,
        compute_pids=tuple(sorted(set(pids))),
    )


def require_physical_gpu_match(visible: DeviceReport, physical: PhysicalGpuEvidence) -> None:
    """Cross-check masked Torch facts against the targeted physical observation."""
    if visible.mode != "gpu" or visible.uuid != physical.uuid:
        raise DeviceGuardError("masked CUDA UUID does not match physical GPU evidence")
    physical_name = getattr(physical, "name", None)
    if visible.name and physical_name and visible.name != physical_name:
        raise DeviceGuardError("masked CUDA model does not match physical GPU evidence")
    if visible.total_memory_mib is None:
        raise DeviceGuardError("masked CUDA capacity is unavailable")
    # CUDA reports usable device memory after reserving a small driver/runtime
    # region, while nvidia-smi reports the marketed physical capacity.  The
    # strategy selector deliberately uses the latter, so allow only that
    # bounded downward delta and still fail closed on over-capacity or a
    # materially inconsistent device mapping.
    capacity_delta = physical.total_memory_mib - visible.total_memory_mib
    allowed_delta = max(1024, int(round(physical.total_memory_mib * 0.05)))
    if capacity_delta < 0 or capacity_delta > allowed_delta:
        raise DeviceGuardError("masked CUDA capacity does not match physical GPU evidence")


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
