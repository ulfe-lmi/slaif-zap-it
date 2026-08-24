"""Operator-controlled GPU/runtime qualification helpers.

The runtime package deliberately has no import-time dependency on PyTorch,
CUDA, Hugging Face or detectron2.  CPU tests can therefore exercise the policy
and fail-closed paths without downloading models or initializing an accelerator.
"""

from .device import (
    DeviceGuardError,
    DeviceReport,
    PhysicalGpuEvidence,
    inspect_physical_gpu,
    inspect_visible_device,
    launch_environment,
    require_launch_environment,
    require_physical_gpu_match,
)
from .models import APPROVED_MODEL_SPECS, ModelSpec
from .ports import PortCheck, select_candidate_port, verify_port_unused
from .readiness import make_readiness_provider
from .shm import ShmError, ShmWorkspace, ensure_shm_root, shm_free_bytes
from .strategy import (
    PROFILE_NAMES,
    ALL_RESIDENT_RESIDENCY_MODE,
    SEQUENTIAL_RESIDENCY_MODE,
    RuntimePolicy,
    RuntimeReadiness,
    SUPPORTED_RESIDENT_PROFILES,
    SUPPORTED_RESIDENT_STRATEGY,
    UnsupportedProfileError,
    select_residency_mode,
)

__all__ = [
    "APPROVED_MODEL_SPECS",
    "DeviceGuardError",
    "DeviceReport",
    "PhysicalGpuEvidence",
    "ModelSpec",
    "PROFILE_NAMES",
    "ALL_RESIDENT_RESIDENCY_MODE",
    "PortCheck",
    "RuntimePolicy",
    "RuntimeReadiness",
    "SEQUENTIAL_RESIDENCY_MODE",
    "SUPPORTED_RESIDENT_PROFILES",
    "SUPPORTED_RESIDENT_STRATEGY",
    "ShmError",
    "ShmWorkspace",
    "UnsupportedProfileError",
    "ensure_shm_root",
    "inspect_visible_device",
    "inspect_physical_gpu",
    "make_readiness_provider",
    "launch_environment",
    "require_launch_environment",
    "require_physical_gpu_match",
    "select_residency_mode",
    "select_candidate_port",
    "shm_free_bytes",
    "verify_port_unused",
]
