"""Content-free admission checks for host RAM and the configured shm root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.core import (
    RAW_MAXIMUM_CONTACT_SHEETS,
    raw_sam2_debug_rgb_bytes,
)
from src.core.config import CoreConfig

from .errors import ServiceError
from .settings import ServiceSettings

__all__ = [
    "host_available_bytes",
    "shm_free_bytes",
    "visualization_raw_bytes",
    "raw_sam2_debug_artifact_count",
    "raw_sam2_debug_bytes",
    "check_visualization_raw_budget",
    "check_request_resources",
]


def host_available_bytes() -> int:
    """Return Linux ``MemAvailable`` without exposing host details."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def shm_free_bytes(root: str) -> int:
    """Return available bytes on the filesystem containing ``root``."""
    try:
        candidate = Path(root)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        stats = os.statvfs(candidate)
        return int(stats.f_bavail * stats.f_frsize)
    except OSError:
        return 0


def _visualization_stream_count(vis_cfg: Mapping[str, Any]) -> int:
    return sum(
        len(entries)
        for stage_name in ("sam2", "clip", "blip3")
        for entries in (vis_cfg.get(stage_name, []),)
        if isinstance(entries, (list, tuple))
    )


def visualization_raw_bytes(config: CoreConfig, *, height: int, width: int) -> int:
    """Return reserved uint8 RGB bytes for configured annotated streams."""
    stream_count = _visualization_stream_count(config.vis_cfg)
    return stream_count * max(int(height), 0) * max(int(width), 0) * 3


def raw_sam2_debug_artifact_count() -> int:
    """Return the fixed maximum artifact count for one SAM2 debug request."""

    return RAW_MAXIMUM_CONTACT_SHEETS + 3


def raw_sam2_debug_bytes(*, height: int, width: int) -> int:
    """Return the renderer's exact worst-case RGB reservation."""

    return raw_sam2_debug_rgb_bytes(int(width), int(height))


def check_visualization_raw_budget(
    config: CoreConfig,
    settings: ServiceSettings,
    *,
    height: int,
    width: int,
) -> int:
    """Return a diagnostic estimate without rejecting inference.

    Optional visualization bytes are admitted after rendering by the shared
    response ledger.  The helper remains as a compatibility seam for callers
    that want an estimate, but it is deliberately not an admission gate.
    """
    stream_count = _visualization_stream_count(config.vis_cfg)
    reserved_streams = stream_count * max(int(height), 0) * max(int(width), 0) * 3
    del settings
    return reserved_streams


def check_request_resources(settings: ServiceSettings) -> None:
    """Fail closed before decoding/allocating when operator floors are unmet."""
    if host_available_bytes() < settings.min_host_available_bytes:
        raise ServiceError(
            "host memory is below the configured admission floor",
            code="insufficient_memory",
        )
    if shm_free_bytes(settings.tmp_root) < settings.min_shm_free_bytes:
        raise ServiceError(
            "shared-memory capacity is below the configured admission floor",
            code="insufficient_shm",
        )
