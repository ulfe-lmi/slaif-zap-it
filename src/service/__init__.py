"""HTTP service contract for ZAP-IT.

The service exposes ``POST /v1/completions`` with strict multipart parsing,
hostile-input validation, monotonic verbosity levels, JSON/ZIP responses,
stable sanitized errors, bounded concurrency, optional API-key auth and
health/readiness endpoints. Explicit model management is a fixed-model
KServe/Triton repository-extension subset, not V2 tensor inference.
"""

from .app import ReadyState, create_app, create_default_app
from .capabilities import CapabilitiesResponse
from .errors import ERROR_STATUS_CODES, ServiceError
from .fake_engine import FakeEngine
from .gate import InferenceGate
from .model_control import LifecycleState, ModelLifecycleController
from .settings import SERVICE_MODEL_ID, ServiceSettings
from .rle import MaskRLEError, decode_mask_rle, encode_mask_rle

__all__ = [
    "create_app",
    "create_default_app",
    "CapabilitiesResponse",
    "ReadyState",
    "FakeEngine",
    "InferenceGate",
    "LifecycleState",
    "ModelLifecycleController",
    "ServiceError",
    "ERROR_STATUS_CODES",
    "ServiceSettings",
    "SERVICE_MODEL_ID",
    "MaskRLEError",
    "encode_mask_rle",
    "decode_mask_rle",
]
