"""Operator-only resource profiles and readiness policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "PROFILE_NAMES",
    "RuntimePolicy",
    "RuntimeReadiness",
    "UnsupportedProfileError",
]

PROFILE_NAMES = (
    "sam2",
    "clip",
    "blip3",
    "sam2_clip",
    "sam2_blip3",
    "sam2_clip_blip3",
)


class UnsupportedProfileError(ValueError):
    """Raised before inference when the operator profile rejects a request."""


@dataclass(frozen=True)
class RuntimeReadiness:
    """Duck-typed readiness result consumed by the service API."""

    ready: bool
    detail: str


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable operator policy; uploaded YAML cannot change these fields."""

    strategy: str = "sam2_clip_resident_blip3_rejected"
    supported_profiles: tuple[str, ...] = ("sam2", "clip", "sam2_clip")
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
        strategy = env.get(
            "SLAIF_ZAP_IT_RESOURCE_STRATEGY",
            "sam2_clip_resident_blip3_rejected",
        )
        profiles_raw = env.get("SLAIF_ZAP_IT_SUPPORTED_PROFILES", "sam2,clip,sam2_clip")
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
            if getattr(device_report, "uuid", None) != self.expected_gpu_uuid:
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
