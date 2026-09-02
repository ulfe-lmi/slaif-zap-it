"""Narrow, stateless OpenAI Responses-compatible transport helpers.

The parser deliberately accepts only the inline single-image/single-YAML
subset documented by Objective 024.  It never fetches URLs, consults file
IDs, writes uploads, or imports the OpenAI SDK.  The SDK is a development-only
qualification dependency; this module owns the server-side adapter.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from modules.segmenter.sam2 import SAM2_DEFAULTS, SAM2_GENERATOR_FIELDS, estimated_prompt_count
from modules.visualizer import render_annotated_labelled

from .envelope import build_object_record, encode_png
from .errors import ServiceError
from .schemas import (
    OpenAIErrorEnvelope,
    PublicProjection,
    ResponsesResponse,
)
from .settings import ServiceSettings

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - Pillow is a required dependency
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]

__all__ = [
    "PUBLIC_SCHEMA_VERSION",
    "RESPONSES_REQUEST_ENVELOPE_BYTES",
    "ResponsesAuthenticationError",
    "ParsedResponsesRequest",
    "responses_request_body_limit",
    "parse_responses_request",
    "build_public_projection",
    "serialize_public_projection",
    "build_responses_response",
    "responses_error_body",
]


PUBLIC_SCHEMA_VERSION = "zap-it.public.v1"
RESPONSES_REQUEST_ENVELOPE_BYTES = 64 * 1024
_MAX_FILENAME_BYTES = 128
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:yaml|yml)$")
_DATA_URL = re.compile(r"^data:([a-z0-9!#$&^_.+/\-]+);base64,([A-Za-z0-9+/]+={0,2})$")
_IMAGE_MIME_TO_FORMAT = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}
_YAML_MIME_TYPES = frozenset(
    {"application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml", "text/plain"}
)
_PARAM = re.compile(r"^(?:model|input(?:\[[0-9]+\])?(?:\.[A-Za-z0-9_]+|\[[0-9]+\])*)$")


class ResponsesAuthenticationError(Exception):
    """Authentication failure that must use the Responses error envelope."""


@dataclass(frozen=True)
class ParsedResponsesRequest:
    """Already decoded, request-local input for the shared inference seam."""

    image_bytes: bytes
    config_bytes: bytes
    model: str
    image_generation: bool


def _encoded_size(decoded_bytes: int) -> int:
    return 4 * ((max(int(decoded_bytes), 0) + 2) // 3)


def responses_request_body_limit(settings: ServiceSettings) -> int:
    """Return the exact decoded-upload-derived JSON/base64 body cap."""
    return (
        _encoded_size(settings.max_image_upload_bytes)
        + _encoded_size(settings.max_config_upload_bytes)
        + RESPONSES_REQUEST_ENVELOPE_BYTES
    )


def _invalid(
    message: str, *, code: str = "invalid_config", param: str | None = None
) -> ServiceError:
    details = {"param": param} if param is not None else None
    return ServiceError(message, code=code, details=details)


def _require_mapping(value: Any, param: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _invalid("request field must be an object", param=param)
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], param: str) -> None:
    if set(value) != expected:
        raise _invalid(
            "request contains an unsupported or missing field",
            code="unsupported_field",
            param=param,
        )


def _decode_data_url(
    value: Any,
    *,
    allowed_mimes: set[str] | frozenset[str],
    max_decoded_bytes: int,
    param: str,
) -> tuple[str, bytes]:
    if type(value) is not str:
        raise _invalid(
            "inline data must be a string data URL", code="responses_invalid_data_url", param=param
        )
    match = _DATA_URL.fullmatch(value)
    if match is None:
        if value.startswith("data:"):
            raise _invalid(
                "inline data URL is malformed or not strict base64",
                code="responses_invalid_data_url",
                param=param,
            )
        raise _invalid(
            "only inline base64 data URLs are supported",
            code="responses_unsupported_source",
            param=param,
        )
    mime = match.group(1)
    encoded = match.group(2)
    if mime not in allowed_mimes:
        raise _invalid(
            "inline data URL MIME type is unsupported", code="unsupported_media_type", param=param
        )
    if len(encoded) % 4 or len(encoded) > _encoded_size(max_decoded_bytes):
        code = (
            "payload_too_large"
            if len(encoded) > _encoded_size(max_decoded_bytes)
            else "responses_invalid_data_url"
        )
        raise _invalid(
            "inline data URL exceeds its decoded upload limit"
            if code == "payload_too_large"
            else "inline data URL is malformed or not strict base64",
            code=code,
            param=param,
        )
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise _invalid(
            "inline data URL is malformed or not strict base64",
            code="responses_invalid_data_url",
            param=param,
        ) from exc
    if not decoded:
        raise _invalid(
            "inline data URL payload must not be empty",
            code="responses_invalid_data_url",
            param=param,
        )
    if len(decoded) > max_decoded_bytes:
        raise _invalid(
            "decoded upload exceeds its configured limit", code="payload_too_large", param=param
        )
    return mime, decoded


def _validate_image(mime: str, payload: bytes, param: str) -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to decode uploaded images")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            actual_format = (opened.format or "").upper()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise _invalid(
            "uploaded inline image is corrupt or invalid", code="invalid_image", param=param
        ) from exc
    if actual_format != _IMAGE_MIME_TO_FORMAT[mime]:
        raise _invalid(
            "declared image MIME does not match its decoded format",
            code="responses_mime_mismatch",
            param=param,
        )


def _safe_filename(value: Any, param: str) -> str:
    if type(value) is not str or len(value.encode("ascii", errors="ignore")) != len(value):
        raise _invalid(
            "filename must be a bounded ASCII YAML basename",
            code="responses_unsafe_filename",
            param=param,
        )
    if (
        len(value) > _MAX_FILENAME_BYTES
        or "/" in value
        or "\\" in value
        or not _SAFE_FILENAME.fullmatch(value)
    ):
        raise _invalid(
            "filename must be a safe .yaml or .yml basename",
            code="responses_unsafe_filename",
            param=param,
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _invalid(
            "filename contains forbidden control characters",
            code="responses_unsafe_filename",
            param=param,
        )
    return value


def parse_responses_request(value: Any, settings: ServiceSettings) -> ParsedResponsesRequest:
    """Parse the exact supported request subset without allocating inference state."""
    if not isinstance(value, dict):
        raise _invalid("request body must be a JSON object", param=None)
    allowed_top = {"model", "input", "tools", "store", "stream", "background"}
    if set(value).difference(allowed_top):
        raise _invalid(
            "request contains an unsupported top-level field", code="unsupported_field", param=None
        )
    if "model" not in value or type(value["model"]) is not str:
        raise _invalid("model is required and must be a string", param="model")
    if value["model"] != settings.model_id:
        raise _invalid("requested model is not supported", code="unsupported_model", param="model")
    if "input" not in value or type(value["input"]) is not list or len(value["input"]) != 1:
        raise _invalid("input must contain exactly one user message", param="input")

    message = _require_mapping(value["input"][0], "input[0]")
    message_keys = set(message)
    if message_keys not in ({"role", "content"}, {"type", "role", "content"}):
        raise _invalid(
            "user message contains an unsupported or missing field",
            code="unsupported_field",
            param="input[0]",
        )
    if "type" in message and message["type"] != "message":
        raise _invalid(
            "message type must be 'message'", code="unsupported_field", param="input[0].type"
        )
    if message.get("role") != "user":
        raise _invalid("input message role must be 'user'", param="input[0].role")
    content = message.get("content")
    if type(content) is not list or len(content) != 2:
        raise _invalid(
            "user content must contain exactly one image and one YAML file",
            param="input[0].content",
        )

    image_part: Mapping[str, Any] | None = None
    file_part: Mapping[str, Any] | None = None
    for index, raw_part in enumerate(content):
        part = _require_mapping(raw_part, f"input[0].content[{index}]")
        part_type = part.get("type")
        if part_type == "input_image":
            if image_part is not None:
                raise _invalid(
                    "exactly one input image is supported",
                    code="duplicate_image",
                    param="input[0].content",
                )
            _require_exact_keys(part, {"type", "detail", "image_url"}, f"input[0].content[{index}]")
            if part.get("detail") != "auto":
                raise _invalid(
                    "image detail must be 'auto'", param=f"input[0].content[{index}].detail"
                )
            image_part = part
        elif part_type == "input_file":
            if file_part is not None:
                raise _invalid(
                    "exactly one input YAML file is supported",
                    code="duplicate_config",
                    param="input[0].content",
                )
            _require_exact_keys(
                part, {"type", "filename", "file_data"}, f"input[0].content[{index}]"
            )
            file_part = part
        else:
            raise _invalid(
                "only input_image and input_file content parts are supported",
                code="unsupported_field",
                param=f"input[0].content[{index}].type",
            )
    if image_part is None:
        raise _invalid(
            "one input image is required", code="missing_image", param="input[0].content"
        )
    if file_part is None:
        raise _invalid(
            "one input YAML file is required", code="missing_config", param="input[0].content"
        )

    image_mime, image_bytes = _decode_data_url(
        image_part.get("image_url"),
        allowed_mimes=set(_IMAGE_MIME_TO_FORMAT),
        max_decoded_bytes=settings.max_image_upload_bytes,
        param="input[0].content.image_url",
    )
    _validate_image(image_mime, image_bytes, "input[0].content.image_url")
    _safe_filename(file_part.get("filename"), "input[0].content.filename")
    _config_mime, config_bytes = _decode_data_url(
        file_part.get("file_data"),
        allowed_mimes=_YAML_MIME_TYPES,
        max_decoded_bytes=settings.max_config_upload_bytes,
        param="input[0].content.file_data",
    )
    try:
        config_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid(
            "inline YAML file must be UTF-8",
            code="invalid_config",
            param="input[0].content.file_data",
        ) from exc

    image_generation = False
    if "tools" in value:
        tools = value["tools"]
        if type(tools) is not list:
            raise _invalid("tools must be an array", param="tools")
        if len(tools) > 1:
            raise _invalid(
                "only one tool declaration is supported", code="unsupported_field", param="tools"
            )
        if tools:
            tool = _require_mapping(tools[0], "tools[0]")
            _require_exact_keys(tool, {"type"}, "tools[0]")
            if tool.get("type") != "image_generation":
                raise _invalid(
                    "only the image_generation tool is supported",
                    code="unsupported_tool",
                    param="tools[0].type",
                )
            image_generation = True

    for field_name in ("store", "stream", "background"):
        if field_name in value:
            if type(value[field_name]) is not bool:
                raise _invalid(f"{field_name} must be a boolean", param=field_name)
            if value[field_name] is True:
                code = "stream_unsupported" if field_name == "stream" else "unsupported_field"
                raise _invalid(f"{field_name}=true is unsupported", code=code, param=field_name)

    return ParsedResponsesRequest(
        image_bytes=image_bytes,
        config_bytes=config_bytes,
        model=settings.model_id,
        image_generation=image_generation,
    )


def _bounded_warning(value: Any) -> str:
    text = str(value)
    text = " ".join(character if ord(character) >= 32 else " " for character in text)
    return text[:256]


def build_public_projection(
    outcome: Any,
    *,
    model_id: str,
    config_digest: str,
    class_mapping: Mapping[str, int],
    candidate_views: Mapping[str, Any],
    clip_routing: Mapping[str, Any],
    clip_prompt_metadata: Mapping[str, Any] | None = None,
    config_warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and validate the deterministic public projection."""
    result = outcome.result
    metadata = dict(getattr(result, "sam2_metadata", {}) or {})
    effective = dict(SAM2_DEFAULTS)
    effective.update(metadata.get("effective", {}))
    sources = {field: "default" for field in SAM2_GENERATOR_FIELDS}
    sources.update(metadata.get("sources", {}))
    prompts = int(
        metadata.get(
            "estimated_prompt_count",
            estimated_prompt_count(
                effective["points_per_side"],
                effective["crop_n_layers"],
                effective["crop_n_points_downscale_factor"],
            ),
        )
    )
    predictions = int(
        metadata.get(
            "estimated_mask_prediction_count",
            prompts * (3 if effective["multimask_output"] else 1),
        )
    )
    public_objects = []
    for obj in result.objects:
        record = build_object_record(
            obj,
            result.image_width,
            result.image_height,
            mask_rle=None,
        )
        record.pop("mask_rle", None)
        public_objects.append(record)
    effective_clip_prompt_metadata = (
        getattr(result, "clip_prompt_metadata", None) or clip_prompt_metadata or None
    )
    projection = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "model": model_id,
        "config_digest": config_digest,
        "image": {"width": int(result.image_width), "height": int(result.image_height)},
        "class_mapping": {str(key): int(value) for key, value in class_mapping.items()},
        "sam2": {
            "requested": dict(metadata.get("requested", {})),
            "effective": effective,
            "sources": sources,
            "selected_profile": metadata.get("selected_profile"),
            "estimated_prompt_count": prompts,
            "estimated_mask_prediction_count": predictions,
            "actual_candidate_count": int(
                metadata.get(
                    "actual_candidate_count", result.candidate_counts.get("sam2_candidates", 0)
                )
            ),
            "resource_warnings": [
                _bounded_warning(item) for item in metadata.get("resource_warnings", [])
            ][:32],
        },
        "candidate_counts": {
            str(key): int(value) for key, value in sorted(result.candidate_counts.items())
        },
        "candidate_views": {str(key): dict(value) for key, value in candidate_views.items()},
        "clip_routing": dict(clip_routing),
        "clip_prompts": effective_clip_prompt_metadata,
        "objects": public_objects,
        "warnings": [
            _bounded_warning(item) for item in (*getattr(result, "warnings", ()), *config_warnings)
        ][:32],
    }
    validated = PublicProjection.model_validate(projection)
    return validated.model_dump(mode="python", exclude_none=True)


def serialize_public_projection(projection: Mapping[str, Any]) -> str:
    """Serialize exactly the public projection bytes placed in output_text."""
    return json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


def build_responses_response(
    projection: Mapping[str, Any],
    *,
    model_id: str,
    image_rgb: np.ndarray,
    objects: Sequence[Any],
    image_generation: bool,
    settings: ServiceSettings,
) -> tuple[dict[str, Any], bytes | None, str]:
    """Build the official-SDK-shaped envelope and optional canonical PNG."""
    output_text = serialize_public_projection(projection)
    response_id = _new_id("resp")
    message_id = _new_id("msg")
    output: list[dict[str, Any]] = [
        {
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        }
    ]
    png_bytes: bytes | None = None
    if image_generation:
        if len(objects) > settings.max_objects:
            raise ServiceError(
                "object count exceeds the configured limit", code="response_too_large"
            )
        expected_raw_bytes = int(image_rgb.shape[0]) * int(image_rgb.shape[1]) * 3
        if expected_raw_bytes > settings.max_total_raw_artifact_bytes:
            raise ServiceError(
                "canonical annotated image exceeds the raw artifact budget",
                code="response_too_large",
            )
        rendered = render_annotated_labelled(
            image_rgb,
            objects,
            alpha=0.5,
            show_confidence=False,
        )
        png_bytes = encode_png(rendered)
        if len(png_bytes) > settings.max_single_artifact_bytes:
            raise ServiceError(
                "canonical annotated image exceeds the artifact size limit",
                code="response_too_large",
            )
        output.append(
            {
                "id": _new_id("ig"),
                "type": "image_generation_call",
                "status": "completed",
                "result": base64.b64encode(png_bytes).decode("ascii"),
            }
        )
    now = max(float(time.time()), 0.0)
    response = {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": model_id,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }
    ResponsesResponse.model_validate(response)
    encoded_size = len(
        json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    if encoded_size > settings.max_response_bytes:
        raise ServiceError(
            "assembled Responses response exceeds the maximum response size",
            code="response_too_large",
        )
    return response, png_bytes, output_text


_ERROR_MESSAGES = {
    "unauthorized": "authentication failed",
    "responses_invalid_json": "request body is malformed JSON",
    "invalid_config": "request configuration is invalid",
    "unsupported_field": "request contains an unsupported field",
    "unsupported_tool": "requested tool is unsupported",
    "unsupported_model": "requested model is unsupported",
    "unsupported_media_type": "request media type is unsupported",
    "responses_unsupported_source": "only inline data URLs are supported",
    "responses_invalid_data_url": "inline data URL is invalid",
    "responses_unsafe_filename": "filename is unsafe",
    "responses_mime_mismatch": "declared MIME does not match the decoded image",
    "invalid_image": "uploaded image is invalid",
    "payload_too_large": "request payload exceeds the configured limit",
    "image_too_large": "decoded image exceeds the configured limit",
    "response_too_large": "response exceeds the configured limit",
    "resource_limit": "request exceeds the configured resource limit",
    "service_busy": "service is busy",
    "not_ready": "service is not ready",
    "timeout": "request timed out",
    "cancelled": "request was cancelled",
    "inference_failure": "inference failed",
    "internal_error": "internal server error",
}
_SERVER_CODES = frozenset(
    {"service_busy", "not_ready", "timeout", "cancelled", "inference_failure", "internal_error"}
)


def responses_error_body(exc: ServiceError, request_id: str) -> dict[str, Any]:
    """Adapt a typed service failure to the bounded OpenAI error envelope."""
    del request_id  # Request IDs belong in x-request-id for this surface.
    error_type = (
        "authentication_error"
        if exc.code == "unauthorized"
        else (
            "server_error"
            if exc.code in _SERVER_CODES or exc.status_code >= 500
            else "invalid_request_error"
        )
    )
    raw_param = exc.details.get("param") if exc.details else None
    param = (
        raw_param
        if isinstance(raw_param, str) and len(raw_param) <= 128 and _PARAM.fullmatch(raw_param)
        else None
    )
    body = {
        "error": {
            "message": _ERROR_MESSAGES.get(exc.code, "request could not be processed")[:256],
            "type": error_type,
            "param": param,
            "code": str(exc.code)[:64],
        }
    }
    OpenAIErrorEnvelope.model_validate(body)
    return body
