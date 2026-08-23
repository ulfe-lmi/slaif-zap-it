"""Stable error taxonomy and the frozen sanitized error envelope.

Every client-visible failure maps to exactly one stable snake_case code with a
fixed HTTP status. Messages are sanitized: no stack traces, raw YAML or image
bytes, host paths, credentials, environment data or model internals.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "ERROR_STATUS_CODES",
    "RETRYABLE_ERROR_CODES",
    "ServiceError",
    "error_envelope",
    "error_status_for",
]

#: Frozen mapping of stable error code -> HTTP status code.
ERROR_STATUS_CODES: Mapping[str, int] = {
    "invalid_multipart": 400,
    "missing_part": 400,
    "duplicate_part": 400,
    "invalid_image": 400,
    "invalid_config": 400,
    "unsafe_config": 400,
    "unsupported_field": 400,
    "unsupported_verbosity": 400,
    "unsupported_format": 400,
    "unsupported_model": 400,
    "unsupported_profile": 400,
    "stream_unsupported": 400,
    "unauthorized": 401,
    "payload_too_large": 413,
    "image_too_large": 413,
    "response_too_large": 413,
    "cancelled": 499,
    "inference_failure": 500,
    "internal_error": 500,
    "service_busy": 503,
    "not_ready": 503,
    "insufficient_memory": 507,
    "timeout": 504,
}

#: Error codes that justify advertising a retry to well-behaved clients.
RETRYABLE_ERROR_CODES = frozenset({"service_busy", "not_ready", "timeout", "insufficient_memory"})


def error_status_for(code: str) -> int:
    """Return the HTTP status registered for ``code`` (500 when unknown)."""
    return ERROR_STATUS_CODES.get(code, 500)


class ServiceError(Exception):
    """A typed, sanitized API failure carrying its stable code and status."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code or error_status_for(code)
        self.headers = headers

    def envelope(self, request_id: str) -> Dict[str, Any]:
        return error_envelope(self.code, self.message, request_id)


def error_envelope(code: str, message: str, request_id: str) -> Dict[str, Any]:
    """Build the frozen error envelope shape."""
    return {"error": {"code": code, "message": message, "request_id": request_id}}
