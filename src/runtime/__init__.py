"""Operator-controlled GPU/runtime qualification helpers.

The runtime package deliberately has no import-time dependency on PyTorch,
CUDA, Hugging Face or detectron2.  CPU tests can therefore exercise the policy
and fail-closed paths without downloading models or initializing an accelerator.
"""

from .device import (
    DeviceGuardError,
    DeviceReport,
    inspect_visible_device,
    launch_environment,
    require_launch_environment,
)
from .models import APPROVED_MODEL_SPECS, ModelSpec
from .ports import PortCheck, select_candidate_port, verify_port_unused
from .readiness import make_readiness_provider
from .shm import ShmError, ShmWorkspace, ensure_shm_root, shm_free_bytes
from .strategy import (
    PROFILE_NAMES,
    RuntimePolicy,
    RuntimeReadiness,
    SUPPORTED_RESIDENT_PROFILES,
    SUPPORTED_RESIDENT_STRATEGY,
    UnsupportedProfileError,
)

__all__ = [
    "APPROVED_MODEL_SPECS",
    "DeviceGuardError",
    "DeviceReport",
    "ModelSpec",
    "PROFILE_NAMES",
    "PortCheck",
    "RuntimePolicy",
    "RuntimeReadiness",
    "SUPPORTED_RESIDENT_PROFILES",
    "SUPPORTED_RESIDENT_STRATEGY",
    "ShmError",
    "ShmWorkspace",
    "UnsupportedProfileError",
    "ensure_shm_root",
    "inspect_visible_device",
    "make_readiness_provider",
    "launch_environment",
    "require_launch_environment",
    "select_candidate_port",
    "shm_free_bytes",
    "verify_port_unused",
]
