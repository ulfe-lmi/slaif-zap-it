"""Operator-only resource profiles and readiness policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ALL_RESIDENT_RESIDENCY_MODE",
    "PROFILE_NAMES",
    "SEQUENTIAL_RESIDENCY_MODE",
    "SUPPORTED_RESIDENT_PROFILES",
    "SUPPORTED_RESIDENT_STRATEGY",
    "select_residency_mode",
    "RuntimePolicy",
    "RuntimeReadiness",
    "UnsupportedProfileError",
]

PROFILE_NAMES = (
    "sam2",
    "sam2_clip",
    "sam2_blip3",
    "sam2_clip_blip3",
)

SEQUENTIAL_RESIDENCY_MODE = "sam2_clip_gpu_blip3_cpu_swap"
ALL_RESIDENT_RESIDENCY_MODE = "sam2_clip_blip3_gpu_resident"
# Compatibility name retained for callers that import the old constant.  It
# now describes the selected operator mode, never a BLIP3 rejection policy.
SUPPORTED_RESIDENT_STRATEGY = SEQUENTIAL_RESIDENCY_MODE
SUPPORTED_RESIDENT_PROFILES = ("sam2", "sam2_clip", "sam2_blip3", "sam2_clip_blip3")
RESIDENCY_BOUNDARY_MIB = 24 * 1024


class UnsupportedProfileError(ValueError):
    """Raised before inference when the operator profile rejects a request."""


@dataclass(frozen=True)
class RuntimeReadiness:
    """Duck-typed readiness result consumed by the service API."""

    ready: bool
    detail: str


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def select_residency_mode(total_memory_mib: int) -> str:
    """Select residency from physical total capacity, not current free memory."""
    total = int(total_memory_mib)
    if total <= 0:
        raise ValueError("physical GPU total memory must be positive")
    return (
        SEQUENTIAL_RESIDENCY_MODE if total < RESIDENCY_BOUNDARY_MIB else ALL_RESIDENT_RESIDENCY_MODE
    )


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable operator policy; uploaded YAML cannot change these fields."""

    strategy: str = SUPPORTED_RESIDENT_STRATEGY
    supported_profiles: tuple[str, ...] = SUPPORTED_RESIDENT_PROFILES
    expected_gpu_uuid: str | None = None
    physical_gpu_index: int = 1
    strict_gpu: bool = True
    model_registry_ready: bool = False

    def __post_init__(self) -> None:
        if not self.strategy.strip():
            raise ValueError("strategy must be non-empty")
        unknown = set(self.supported_profiles).difference(PROFILE_NAMES)
        if unknown:
            raise ValueError(f"unknown runtime profile(s): {sorted(unknown)!r}")
        if self.physical_gpu_index < 0:
            raise ValueError("physical_gpu_index must be non-negative")
        if self.strict_gpu and not self.expected_gpu_uuid:
            raise ValueError("strict GPU policy requires expected_gpu_uuid")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        expected_gpu_uuid: str | None = None,
        model_registry_ready: bool = False,
    ) -> "RuntimePolicy":
        env = os.environ if environ is None else environ
        # Residency is derived from live physical capacity.  Deliberately do
        # not read the legacy resource-strategy/profile override variables:
        # they are operator policy, not a client or environment selector.
        strategy = SEQUENTIAL_RESIDENCY_MODE
        profiles_raw = ",".join(SUPPORTED_RESIDENT_PROFILES)
        strict_raw = env.get("SLAIF_ZAP_IT_STRICT_GPU", "1").strip().lower()
        strict = strict_raw not in {"0", "false", "no", "off"}
        uuid = expected_gpu_uuid or env.get("SLAIF_ZAP_IT_EXPECTED_GPU_UUID") or None
        physical = int(env.get("SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX", "1"))
        return cls(
            strategy=strategy,
            supported_profiles=_csv(profiles_raw),
            expected_gpu_uuid=uuid,
            physical_gpu_index=physical,
            strict_gpu=strict,
            model_registry_ready=model_registry_ready,
        )

    @classmethod
    def for_capacity(
        cls,
        total_memory_mib: int,
        *,
        expected_gpu_uuid: str | None,
        physical_gpu_index: int = 1,
        strict_gpu: bool = True,
        model_registry_ready: bool = False,
    ) -> "RuntimePolicy":
        """Build the immutable policy from fresh physical-device evidence."""
        return cls(
            strategy=select_residency_mode(total_memory_mib),
            supported_profiles=SUPPORTED_RESIDENT_PROFILES,
            expected_gpu_uuid=expected_gpu_uuid,
            physical_gpu_index=physical_gpu_index,
            strict_gpu=strict_gpu,
            model_registry_ready=model_registry_ready,
        )

    def profile_for_config(self, config: Any) -> str:
        """Map normalized core config to the actual model profile it requests."""
        clip = bool(getattr(config, "clip_cfg", None))
        blip3 = bool(getattr(config, "blip3_cfg", None))
        if blip3 and clip:
            return "sam2_clip_blip3"
        if blip3:
            return "sam2_blip3"
        if clip:
            return "sam2_clip"
        return "sam2"

    def validate_config(self, config: Any) -> str:
        """Reject unsupported model combinations before the expensive stage."""
        profile = self.profile_for_config(config)
        if profile not in self.supported_profiles:
            raise UnsupportedProfileError(
                f"runtime profile {profile!r} is not supported by the operator strategy"
            )
        return profile

    def readiness(self, device_report: Any | None = None) -> RuntimeReadiness:
        """Return honest readiness without silently falling back to another GPU."""
        if self.strict_gpu:
            if device_report is None:
                return RuntimeReadiness(False, "strict GPU device evidence is not available")
            if getattr(device_report, "mode", None) != "gpu":
                return RuntimeReadiness(False, "strict GPU device evidence is not a GPU")
            visible_uuid = getattr(device_report, "uuid", None)
            expected_uuid = self.expected_gpu_uuid
            if expected_uuid and not expected_uuid.startswith("GPU-"):
                expected_uuid = f"GPU-{expected_uuid}"
            if visible_uuid != expected_uuid:
                return RuntimeReadiness(False, "visible GPU UUID does not match the operator pin")
        elif device_report is None:
            return RuntimeReadiness(False, "runtime device evidence is not available")
        if not self.model_registry_ready:
            return RuntimeReadiness(False, "qualified model registry is not ready")
        return RuntimeReadiness(True, f"runtime strategy {self.strategy} is ready")

    def with_model_registry_ready(self, ready: bool = True) -> "RuntimePolicy":
        return RuntimePolicy(
            strategy=self.strategy,
            supported_profiles=self.supported_profiles,
            expected_gpu_uuid=self.expected_gpu_uuid,
            physical_gpu_index=self.physical_gpu_index,
            strict_gpu=self.strict_gpu,
            model_registry_ready=ready,
        )
