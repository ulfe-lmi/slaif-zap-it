"""HTTP service contract for ZAP-IT (objective 002: CPU/fake-engine only).

The service exposes ``POST /v1/completions`` with strict multipart parsing,
hostile-input validation, monotonic verbosity levels, JSON/ZIP responses,
stable sanitized errors, bounded concurrency, optional API-key auth and
health/readiness endpoints. Inference is delegated to an injected engine
callable; no live GPU/model path exists yet.
"""

from .app import ReadyState, create_app, create_default_app
from .errors import ERROR_STATUS_CODES, ServiceError
from .fake_engine import FakeEngine
from .gate import InferenceGate
from .settings import SERVICE_MODEL_ID, ServiceSettings
from .rle import MaskRLEError, decode_mask_rle, encode_mask_rle

__all__ = [
    "create_app",
    "create_default_app",
    "ReadyState",
    "FakeEngine",
    "InferenceGate",
    "ServiceError",
    "ERROR_STATUS_CODES",
    "ServiceSettings",
    "SERVICE_MODEL_ID",
    "MaskRLEError",
    "encode_mask_rle",
    "decode_mask_rle",
]
