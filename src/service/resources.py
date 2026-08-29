"""Content-free admission checks for host RAM and the configured shm root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.core import (
    RAW_CONTACT_SHEET_HEIGHT,
    RAW_CONTACT_SHEET_WIDTH,
    RAW_MAXIMUM_CONTACT_SHEETS,
    diagnostic_dimensions,
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
    """Reject L3 rendering that cannot fit before the engine allocates arrays."""
    stream_count = _visualization_stream_count(config.vis_cfg)
    reserved_streams = stream_count * max(int(height), 0) * max(int(width), 0) * 3
    per_stream = max(int(height), 0) * max(int(width), 0) * 3
    if stream_count and per_stream > settings.max_single_artifact_bytes:
        raise ServiceError(
            "annotated visualization exceeds the configured raw artifact limit",
            code="response_too_large",
        )
    if reserved_streams > settings.max_total_raw_artifact_bytes:
        raise ServiceError(
            "annotated visualizations exceed the configured raw artifact budget",
            code="response_too_large",
        )

    if config.sam2_cfg.get("debug", False):
        sheet_bytes = RAW_CONTACT_SHEET_WIDTH * RAW_CONTACT_SHEET_HEIGHT * 3
        diagnostic_width, diagnostic_height = diagnostic_dimensions(int(width), int(height))
        diagnostic_bytes = diagnostic_width * diagnostic_height * 3
        maximum_single = max(sheet_bytes, diagnostic_bytes)
        if maximum_single > settings.max_single_artifact_bytes:
            raise ServiceError(
                "raw SAM2 debug artifact exceeds the configured per-artifact limit",
                code="response_too_large",
            )
        debug_bytes = raw_sam2_debug_bytes(height=height, width=width)
        reserved = reserved_streams + debug_bytes
        if reserved > settings.max_total_raw_artifact_bytes:
            raise ServiceError(
                "raw SAM2 debug artifacts exceed the configured total byte limit",
                code="response_too_large",
            )
        debug_artifacts = raw_sam2_debug_artifact_count()
        if settings.max_debug_artifacts < debug_artifacts:
            raise ServiceError(
                "raw SAM2 debug artifacts exceed the configured artifact count",
                code="response_too_large",
            )
        response_artifacts = 1 + stream_count + debug_artifacts
        if response_artifacts > settings.max_response_artifacts:
            raise ServiceError(
                "raw SAM2 debug response exceeds the configured artifact count",
                code="response_too_large",
            )
        # The response admission includes the uint16 identity source canvas,
        # all configured RGB streams and the fixed raw-debug arrays.  The
        # small fixed margin covers PNG/container and typed-envelope framing;
        # final encoded-size checks remain authoritative after rendering.
        identity_bytes = max(int(height), 0) * max(int(width), 0) * 2
        response_source_bytes = identity_bytes + reserved
        response_upper_bound = 4 * ((response_source_bytes + 2) // 3)
        response_upper_bound += 64 * 1024 + response_artifacts * 1024
        if response_upper_bound > settings.max_response_bytes:
            raise ServiceError(
                "raw SAM2 debug response exceeds the configured response limit",
                code="response_too_large",
            )
        # Configured visualization arrays are produced outside the artifact
        # sink, so only their reservation is deducted from the sink budget.
        # The debug reservation is an admission ceiling for arrays that the
        # sink will actually receive; deducting it here would reject the
        # exact accepted boundary twice.
        return reserved_streams

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
