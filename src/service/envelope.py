"""Completion-envelope construction: monotonic levels, JSON and ZIP parity.

The public completion envelope carries ``id``, ``object=text_completion``,
``created``, ``model``, exactly one choice whose ``text`` holds normalized
YOLO lines, ``usage=null`` (no invented token counts) and a ``service``
metadata block gated by the monotonic verbosity levels:

- L0: YOLO text plus minimal safe metadata;
- L1: adds the lossless uint16 identity-mask PNG artifact;
- L2: adds per-object fields actually produced by executed stages;
- L3: adds bounded stage statuses/timings/provenance/warnings and available
  debug/visualization artifacts.

Binary artifacts use the stable object ``{name, media_type, encoding,
sha256, size, data}``. ZIP responses contain ``manifest.json``,
``detections.yolo.txt``, the identity mask when applicable and level-gated
artifacts with deterministic names and matching hashes/sizes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is a required dependency
    Image = None  # type: ignore[assignment]

from src.core.errors import CoreError
from src.core.renderers import render_identity_png, render_yolo
from src.core.results import ObjectResult, SingleImageOutcome
from src.core.sinks import ArtifactSink, ArtifactSinkError, StoredArtifact

from .errors import ServiceError

__all__ = [
    "SCHEMA_VERSION",
    "MAX_RESPONSE_ARTIFACTS",
    "ResponseContext",
    "build_completion_json",
    "build_completion_zip",
    "bound_json_size",
]

SCHEMA_VERSION = "zap-it.v1"
MAX_RESPONSE_ARTIFACTS = 64
_DETERMINISTIC_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ResponseContext:
    """Request-scoped metadata used when building the completion."""

    request_id: str
    model_id: str
    verbosity: int
    response_format: str
    config_digest: str
    class_mapping: Mapping[str, int]
    config_warnings: Sequence[str] = field(default_factory=tuple)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)


def _artifact(name: str, media_type: str, payload: bytes) -> Dict[str, Any]:
    return {
        "name": name,
        "media_type": media_type,
        "encoding": "base64",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _encode_png(array: np.ndarray) -> bytes:
    if Image is None:  # pragma: no cover - broken install guard
        raise RuntimeError("Pillow is required to encode PNG artifacts.")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _object_record(obj: ObjectResult, width: int, height: int) -> Dict[str, Any]:
    cx, cy, bw, bh = obj.normalized_bbox(width, height)
    record: Dict[str, Any] = {
        "instance_id": obj.instance_id,
        "class_id": obj.class_id if obj.class_id is not None else 0,
        "label": obj.label,
        "bbox_xyxy": [int(v) for v in obj.bbox_xyxy],
        "bbox_normalized": [
            round(cx, 6),
            round(cy, 6),
            round(bw, 6),
            round(bh, 6),
        ],
        "area_px": obj.area_px,
        "centroid_rc": [round(obj.centroid_rc[0], 3), round(obj.centroid_rc[1], 3)],
    }
    optional_fields = {
        "predicted_iou": obj.predicted_iou,
        "stability_score": obj.stability_score,
        "clip_score": obj.clip_score,
        "blip3_answer": obj.blip3_answer,
        "geometry": obj.geometry(),
    }
    for key, value in optional_fields.items():
        if value is not None:
            record[key] = value
    if obj.warnings:
        record["warnings"] = list(obj.warnings)
    return record


def _collect_artifacts(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    sink: Optional[ArtifactSink],
    warnings_out: List[str],
) -> List[Dict[str, Any]]:
    result = outcome.result
    artifacts: List[Dict[str, Any]] = []
    if context.verbosity >= 1:
        png = render_identity_png(
            result.objects,
            width=result.image_width,
            height=result.image_height,
            ensure_all_ids=True,
        )
        artifacts.append(_artifact("identity-mask.png", "image/png", png))
    if context.verbosity >= 3:
        for key, array in result.rendered.items():
            safe_key = str(key).replace("\\", "_").strip("/.") or "stream"
            artifacts.append(
                _artifact(f"visualization/{safe_key}.png", "image/png", _encode_png(array))
            )
        if sink is not None:
            for stored in sink.artifacts():
                artifacts.extend(_stored_sink_artifact(stored, warnings_out))
    return artifacts


def _stored_sink_artifact(stored: StoredArtifact, warnings_out: List[str]) -> List[Dict[str, Any]]:
    try:
        if stored.kind == "image-array":
            assert stored.array is not None
            payload = _encode_png(stored.array)
            media_type = "image/png"
        elif stored.kind == "record":
            payload = json.dumps(stored.record, default=str, sort_keys=True).encode("utf-8")
            media_type = "application/json"
        else:
            assert stored.data is not None
            payload = stored.data
            media_type = stored.content_type or "application/octet-stream"
    except (AssertionError, ArtifactSinkError, OSError, ValueError):
        warnings_out.append(f"debug artifact {stored.name!r} could not be encoded")
        return []
    return [_artifact(stored.name, media_type, payload)]


def build_completion_json(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    *,
    sink: Optional[ArtifactSink] = None,
) -> Dict[str, Any]:
    """Build the level-gated JSON completion document."""
    result = outcome.result
    warnings_out: List[str] = list(context.config_warnings)
    try:
        yolo_text = render_yolo(
            result.objects, image_width=result.image_width, image_height=result.image_height
        )
        artifacts = _collect_artifacts(outcome, context, sink, warnings_out)
    except CoreError as exc:
        raise ServiceError(
            "pipeline result exceeds the representable object limit",
            code="inference_failure",
        ) from exc

    if len(artifacts) > MAX_RESPONSE_ARTIFACTS:
        omitted = len(artifacts) - MAX_RESPONSE_ARTIFACTS
        artifacts = artifacts[:MAX_RESPONSE_ARTIFACTS]
        warnings_out.append(f"artifact budget exceeded; {omitted} debug artifacts omitted")

    service_meta: Dict[str, Any] = {
        "request_id": context.request_id,
        "verbosity": context.verbosity,
        "finish_reason": "stop",
        "image": {"width": result.image_width, "height": result.image_height},
        "class_mapping": dict(context.class_mapping),
        "config_digest": context.config_digest,
    }
    if context.verbosity >= 1:
        service_meta["artifacts"] = [
            {
                "name": artifact["name"],
                "media_type": artifact["media_type"],
                "encoding": "base64",
                "sha256": artifact["sha256"],
                "size": artifact["size"],
                "data": artifact["data"],
            }
            for artifact in artifacts
        ]
    if context.verbosity >= 2:
        service_meta["objects"] = [
            _object_record(obj, result.image_width, result.image_height) for obj in result.objects
        ]
    if context.verbosity >= 3:
        service_meta["stage_statuses"] = [status.as_dict() for status in result.stage_statuses]
        service_meta["candidate_counts"] = dict(result.candidate_counts)
        service_meta["timings_ms"] = {
            key: round(value, 3) for key, value in sorted(result.timings.items())
        }
        provenance = result.provenance.as_dict()
        if context.runtime_metadata:
            provenance["runtime"] = dict(context.runtime_metadata)
        service_meta["provenance"] = provenance
        service_meta["warnings"] = list(result.warnings) + warnings_out

    return {
        "id": f"cmpl-{context.request_id}",
        "object": "text_completion",
        "created": int(_now_unix()),
        "model": context.model_id,
        "choices": [{"index": 0, "text": yolo_text, "finish_reason": "stop"}],
        "usage": None,
        "schema_version": SCHEMA_VERSION,
        "service": service_meta,
    }


def _now_unix() -> float:
    return time.time()


def _bound_total_size(artifacts: Sequence[Mapping[str, Any]], max_bytes: int) -> None:
    total = sum(int(item["size"]) for item in artifacts)
    if total > max_bytes:
        raise ServiceError(
            "assembled response would exceed the maximum response size",
            code="response_too_large",
        )


def _zip_entry(buffer: io.BytesIO, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=_DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, payload)


def build_completion_zip(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    *,
    sink: Optional[ArtifactSink] = None,
    max_bytes: int,
) -> bytes:
    """Build the deterministic ZIP response with matching manifest."""
    document = build_completion_json(outcome, context, sink=sink)
    artifacts = document["service"].get("artifacts", [])
    _bound_total_size(artifacts, max_bytes)

    manifest = json.loads(json.dumps(document))
    for entry in manifest["service"].get("artifacts", []):
        entry.pop("data", None)

    buffer = io.BytesIO()
    _zip_entry(buffer, "manifest.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
    _zip_entry(
        buffer,
        "detections.yolo.txt",
        document["choices"][0]["text"].encode("utf-8"),
    )
    for artifact in artifacts:
        _zip_entry(buffer, artifact["name"], base64.b64decode(artifact["data"]))
    payload = buffer.getvalue()
    if len(payload) > max_bytes:
        raise ServiceError(
            "assembled ZIP response exceeds the maximum response size",
            code="response_too_large",
        )
    return payload


def bound_json_size(document: Mapping[str, Any], max_response_bytes: int) -> None:
    """Reject JSON documents whose encoded size exceeds ``max_response_bytes``."""
    encoded = len(json.dumps(document).encode("utf-8"))
    if encoded > max_response_bytes:
        raise ServiceError(
            "assembled JSON response exceeds the maximum response size",
            code="response_too_large",
        )
