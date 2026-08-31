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
    "MODEL_CONTROL_API_KEY_ENV_VAR",
    "MODEL_CONTROL_MODE_ENV_VAR",
    "MODEL_CONTROL_DRAIN_SECONDS_ENV_VAR",
    "MODEL_CONTROL_OPERATION_SECONDS_ENV_VAR",
    "SETTINGS_ENV_PREFIX",
    "TMP_ROOT_ENV_VAR",
    "DEFAULT_TMP_ROOT",
    "SERVICE_MODEL_ID",
    "BLIP3_MAX_QUESTIONS_ENV_VAR",
    "SAM2_LIMIT_ENV_VARS",
    "ServiceSettings",
]

API_KEY_ENV_VAR = "SLAIF_ZAP_IT_API_KEY"
MODEL_CONTROL_API_KEY_ENV_VAR = "SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY"
MODEL_CONTROL_MODE_ENV_VAR = "SLAIF_ZAP_IT_MODEL_CONTROL_MODE"
MODEL_CONTROL_DRAIN_SECONDS_ENV_VAR = "SLAIF_ZAP_IT_MODEL_CONTROL_DRAIN_SECONDS"
MODEL_CONTROL_OPERATION_SECONDS_ENV_VAR = "SLAIF_ZAP_IT_MODEL_CONTROL_OPERATION_SECONDS"
SETTINGS_ENV_PREFIX = "SLAIF_ZAP_IT_"
TMP_ROOT_ENV_VAR = "SLAIF_ZAP_IT_TMP_ROOT"
DEFAULT_TMP_ROOT = "/dev/shm/slaif-zap-it"

#: Fixed public service/model identifier for this contract version.
SERVICE_MODEL_ID = "zap-it-1"
BLIP3_MAX_QUESTIONS_ENV_VAR = "SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS"

SAM2_LIMIT_ENV_VARS = {
    "SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_SIDE": "sam2_max_points_per_side",
    "SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_BATCH": "sam2_max_points_per_batch",
    "SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS": "sam2_max_crop_n_layers",
    "SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_PROMPTS": "sam2_max_estimated_prompts",
    "SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_MASK_PREDICTIONS": ("sam2_max_estimated_mask_predictions"),
    "SLAIF_ZAP_IT_SAM2_MAX_MIN_MASK_REGION_AREA": "sam2_max_min_mask_region_area",
}

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
    test_serialization_delay_seconds: float = 0.0
    api_key: Optional[str] = None
    model_control_mode: str = "none"
    model_control_api_key: Optional[str] = None
    model_control_drain_seconds: float = 120.0
    model_control_operation_seconds: float = 600.0
    model_id: str = SERVICE_MODEL_ID
    tmp_root: str = DEFAULT_TMP_ROOT
    sam2_max_points_per_side: int = 64
    sam2_max_points_per_batch: int = 64
    sam2_max_crop_n_layers: int = 2
    sam2_max_estimated_prompts: int = 8192
    sam2_max_estimated_mask_predictions: int = 24576
    sam2_max_min_mask_region_area: int = 1_000_000
    blip3_max_questions: int = 256

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
        if self.test_serialization_delay_seconds < 0:
            raise ValueError("test_serialization_delay_seconds must be >= 0")
        if self.model_control_mode not in {"none", "explicit"}:
            raise ValueError("model_control_mode must be 'none' or 'explicit'")
        if not 0 < self.model_control_drain_seconds <= 3600:
            raise ValueError("model_control_drain_seconds must be in (0, 3600]")
        if not 0 < self.model_control_operation_seconds <= 3600:
            raise ValueError("model_control_operation_seconds must be in (0, 3600]")
        if self.model_control_mode == "explicit":
            if not self.model_control_api_key:
                raise ValueError(
                    "explicit model control requires SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY"
                )
            if self.api_key and self.model_control_api_key == self.api_key:
                raise ValueError("model control and inference credentials must be different")
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty identifier")
        intrinsic_maxima = {
            "sam2_max_points_per_side": 1024,
            "sam2_max_points_per_batch": 1024,
            "sam2_max_crop_n_layers": 8,
            "sam2_max_min_mask_region_area": 64_000_000,
        }
        for field_name, intrinsic_maximum in intrinsic_maxima.items():
            value = getattr(self, field_name)
            if type(value) is not int or value < 0 or value > intrinsic_maximum:
                raise ValueError(f"{field_name} must be an integer from 0 to {intrinsic_maximum}")
        for field_name in (
            "sam2_max_points_per_side",
            "sam2_max_points_per_batch",
            "sam2_max_estimated_prompts",
            "sam2_max_estimated_mask_predictions",
            "sam2_max_min_mask_region_area",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.sam2_max_crop_n_layers < 0:
            raise ValueError("sam2_max_crop_n_layers must be non-negative")
        if self.sam2_max_min_mask_region_area < 0:
            raise ValueError("sam2_max_min_mask_region_area must be non-negative")
        if type(self.blip3_max_questions) is not int or not 1 <= self.blip3_max_questions <= 256:
            raise ValueError("blip3_max_questions must be an integer from 1 to 256")

    @property
    def sam2_operator_caps(self) -> dict[str, int]:
        """Return the immutable public SAM2 admission caps."""
        return {
            "points_per_side": self.sam2_max_points_per_side,
            "points_per_batch": self.sam2_max_points_per_batch,
            "crop_n_layers": self.sam2_max_crop_n_layers,
            "estimated_prompt_count": self.sam2_max_estimated_prompts,
            "estimated_mask_prediction_count": self.sam2_max_estimated_mask_predictions,
            "min_mask_region_area": self.sam2_max_min_mask_region_area,
        }

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
        for env_name, field_name in SAM2_LIMIT_ENV_VARS.items():
            raw = env.get(env_name)
            if raw is not None and raw != "":
                if field_name == "sam2_max_crop_n_layers":
                    kwargs[field_name] = _nonnegative_int(raw, env_name)
                else:
                    kwargs[field_name] = _positive_int(raw, env_name)
        raw_blip3_questions = env.get(BLIP3_MAX_QUESTIONS_ENV_VAR)
        if raw_blip3_questions is not None:
            if raw_blip3_questions == "":
                raise ValueError(f"{BLIP3_MAX_QUESTIONS_ENV_VAR} must be an integer from 1 to 256")
            try:
                kwargs["blip3_max_questions"] = int(raw_blip3_questions)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{BLIP3_MAX_QUESTIONS_ENV_VAR} must be an integer from 1 to 256"
                ) from exc
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
        raw_serialization_delay = env.get("SLAIF_ZAP_IT_TEST_SERIALIZATION_DELAY_SECONDS")
        if raw_serialization_delay:
            serialization_delay = float(raw_serialization_delay)
            if serialization_delay < 0:
                raise ValueError(
                    "SLAIF_ZAP_IT_TEST_SERIALIZATION_DELAY_SECONDS must be non-negative"
                )
            kwargs["test_serialization_delay_seconds"] = serialization_delay
        mode = env.get(MODEL_CONTROL_MODE_ENV_VAR, "none").strip().lower()
        if mode == "":
            mode = "none"
        kwargs["model_control_mode"] = mode
        control_key = env.get(MODEL_CONTROL_API_KEY_ENV_VAR)
        kwargs["model_control_api_key"] = control_key or None
        for env_name, field_name in (
            (
                MODEL_CONTROL_DRAIN_SECONDS_ENV_VAR,
                "model_control_drain_seconds",
            ),
            (
                MODEL_CONTROL_OPERATION_SECONDS_ENV_VAR,
                "model_control_operation_seconds",
            ),
        ):
            raw_seconds = env.get(env_name)
            if raw_seconds:
                seconds = float(raw_seconds)
                if seconds <= 0:
                    raise ValueError(f"{env_name} must be positive")
                kwargs[field_name] = seconds
        api_key = env.get(API_KEY_ENV_VAR)
        if api_key == "":
            api_key = None
        kwargs["api_key"] = api_key
        tmp_root = env.get(TMP_ROOT_ENV_VAR)
        if tmp_root:
            kwargs["tmp_root"] = tmp_root
        return cls(**kwargs)
