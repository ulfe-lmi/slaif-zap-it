"""Operator-controlled service settings.

Limits are fixed at startup (environment variables or explicit construction)
and are never client-overridable per request. The documented production
defaults follow the frozen objective 002 decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "API_KEY_ENV_VAR",
    "SETTINGS_ENV_PREFIX",
    "TMP_ROOT_ENV_VAR",
    "DEFAULT_TMP_ROOT",
    "SERVICE_MODEL_ID",
    "ServiceSettings",
]

API_KEY_ENV_VAR = "SLAIF_ZAP_IT_API_KEY"
SETTINGS_ENV_PREFIX = "SLAIF_ZAP_IT_"
TMP_ROOT_ENV_VAR = "SLAIF_ZAP_IT_TMP_ROOT"
DEFAULT_TMP_ROOT = "/dev/shm/slaif-zap-it"

#: Fixed public service/model identifier for this contract version.
SERVICE_MODEL_ID = "zap-it-1"

_MIB = 1024 * 1024


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable startup configuration for the HTTP service."""

    max_image_upload_bytes: int = 20 * _MIB
    max_config_upload_bytes: int = 256 * 1024
    max_decoded_pixels: int = 64_000_000
    max_response_bytes: int = 256 * _MIB
    request_deadline_seconds: float = 120.0
    queue_depth: int = 0
    retry_after_seconds: int = 5
    api_key: Optional[str] = None
    model_id: str = SERVICE_MODEL_ID
    tmp_root: str = DEFAULT_TMP_ROOT

    def __post_init__(self) -> None:
        if self.max_image_upload_bytes <= 0:
            raise ValueError("max_image_upload_bytes must be positive")
        if self.max_config_upload_bytes <= 0:
            raise ValueError("max_config_upload_bytes must be positive")
        if self.max_decoded_pixels <= 0:
            raise ValueError("max_decoded_pixels must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if self.request_deadline_seconds <= 0:
            raise ValueError("request_deadline_seconds must be positive")
        if self.queue_depth < 0:
            raise ValueError("queue_depth must be >= 0")
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be >= 0")
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty identifier")

    @property
    def max_request_bytes(self) -> int:
        """Total encoded multipart body cap with slack for small fields."""
        return self.max_image_upload_bytes + self.max_config_upload_bytes + 64 * 1024

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "ServiceSettings":
        """Read operator overrides from ``SLAIF_ZAP_IT_*`` variables."""
        env = os.environ if environ is None else environ
        kwargs = {}
        simple_ints = {
            "SLAIF_ZAP_IT_MAX_IMAGE_UPLOAD_BYTES": "max_image_upload_bytes",
            "SLAIF_ZAP_IT_MAX_CONFIG_UPLOAD_BYTES": "max_config_upload_bytes",
            "SLAIF_ZAP_IT_MAX_DECODED_PIXELS": "max_decoded_pixels",
            "SLAIF_ZAP_IT_MAX_RESPONSE_BYTES": "max_response_bytes",
            "SLAIF_ZAP_IT_QUEUE_DEPTH": "queue_depth",
            "SLAIF_ZAP_IT_RETRY_AFTER_SECONDS": "retry_after_seconds",
        }
        for env_name, field_name in simple_ints.items():
            raw = env.get(env_name)
            if raw is not None and raw != "":
                kwargs[field_name] = _positive_int(raw, env_name)
        raw_deadline = env.get("SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS")
        if raw_deadline:
            deadline = float(raw_deadline)
            if deadline <= 0:
                raise ValueError("SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS must be positive")
            kwargs["request_deadline_seconds"] = deadline
        api_key = env.get(API_KEY_ENV_VAR)
        if api_key == "":
            api_key = None
        kwargs["api_key"] = api_key
        tmp_root = env.get(TMP_ROOT_ENV_VAR)
        if tmp_root:
            kwargs["tmp_root"] = tmp_root
        return cls(**kwargs)
