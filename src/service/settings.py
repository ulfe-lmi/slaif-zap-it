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


def _nonnegative_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return parsed


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable startup configuration for the HTTP service."""

    max_image_upload_bytes: int = 20 * _MIB
    max_config_upload_bytes: int = 256 * 1024
    max_decoded_pixels: int = 64_000_000
    max_response_bytes: int = 256 * _MIB
    max_image_width: int = 8192
    max_image_height: int = 8192
    max_objects: int = 256
    max_visualization_streams: int = 8
    max_response_artifacts: int = 64
    max_debug_artifacts: int = 48
    max_single_artifact_bytes: int = 32 * _MIB
    max_total_raw_artifact_bytes: int = 128 * _MIB
    max_mask_rle_runs_per_object: int = 250_000
    max_mask_rle_runs_total: int = 1_000_000
    min_host_available_bytes: int = 2 * 1024 * _MIB
    min_shm_free_bytes: int = 64 * _MIB
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
        for field_name in (
            "max_image_width",
            "max_image_height",
            "max_objects",
            "max_visualization_streams",
            "max_response_artifacts",
            "max_debug_artifacts",
            "max_single_artifact_bytes",
            "max_total_raw_artifact_bytes",
            "max_mask_rle_runs_per_object",
            "max_mask_rle_runs_total",
            "min_host_available_bytes",
            "min_shm_free_bytes",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
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
            "SLAIF_ZAP_IT_MAX_IMAGE_WIDTH": "max_image_width",
            "SLAIF_ZAP_IT_MAX_IMAGE_HEIGHT": "max_image_height",
            "SLAIF_ZAP_IT_MAX_OBJECTS": "max_objects",
            "SLAIF_ZAP_IT_MAX_VISUALIZATION_STREAMS": "max_visualization_streams",
            "SLAIF_ZAP_IT_MAX_RESPONSE_ARTIFACTS": "max_response_artifacts",
            "SLAIF_ZAP_IT_MAX_DEBUG_ARTIFACTS": "max_debug_artifacts",
            "SLAIF_ZAP_IT_MAX_SINGLE_ARTIFACT_BYTES": "max_single_artifact_bytes",
            "SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES": "max_total_raw_artifact_bytes",
            "SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT": "max_mask_rle_runs_per_object",
            "SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL": "max_mask_rle_runs_total",
            "SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES": "min_host_available_bytes",
            "SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES": "min_shm_free_bytes",
        }
        for env_name, field_name in simple_ints.items():
            raw = env.get(env_name)
            if raw is not None and raw != "":
                kwargs[field_name] = _positive_int(raw, env_name)
        raw_queue = env.get("SLAIF_ZAP_IT_QUEUE_DEPTH")
        if raw_queue is not None and raw_queue != "":
            kwargs["queue_depth"] = _nonnegative_int(raw_queue, "SLAIF_ZAP_IT_QUEUE_DEPTH")
        raw_retry = env.get("SLAIF_ZAP_IT_RETRY_AFTER_SECONDS")
        if raw_retry is not None and raw_retry != "":
            kwargs["retry_after_seconds"] = _nonnegative_int(
                raw_retry, "SLAIF_ZAP_IT_RETRY_AFTER_SECONDS"
            )
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
