"""Content-free admission checks for host RAM and the configured shm root."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ServiceError
from .settings import ServiceSettings

__all__ = ["host_available_bytes", "shm_free_bytes", "check_request_resources"]


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
