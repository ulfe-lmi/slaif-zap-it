"""Optional operator-controlled API-key authentication.

Strict-loopback deployments default to NO key. Setting the operator
environment variable ``SLAIF_ZAP_IT_API_KEY`` (before process start) enables
mandatory ``Authorization: Bearer <key>`` checks with a constant-time
comparison. Keys are never logged or echoed.
"""

from __future__ import annotations

import secrets
from typing import Optional, Tuple

from .errors import ServiceError

__all__ = ["verify_bearer_key", "authorization_failure"]


def verify_bearer_key(provided: Optional[str], expected: str) -> None:
    """Raise ``unauthorized`` unless ``provided`` matches ``expected``."""
    if provided is None or not provided.startswith("Bearer "):
        raise authorization_failure()
    candidate = provided[len("Bearer ") :]
    if not secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
        raise authorization_failure()


def parse_authorization_header(value: Optional[str]) -> Tuple[bool, str]:
    """Return ``(is_bearer, token)`` without logging the value."""
    if value is None:
        return False, ""
    if value.startswith("Bearer "):
        return True, value[len("Bearer ") :]
    return False, ""


def authorization_failure() -> ServiceError:
    return ServiceError(
        "missing or invalid API credentials",
        code="unauthorized",
    )
