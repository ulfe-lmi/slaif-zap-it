"""Bounded, deterministic L0-L3 response preparation for JSON and ZIP."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from modules.segmenter.sam2 import SAM2_DEFAULTS, SAM2_GENERATOR_FIELDS, estimated_prompt_count

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a required dependency
    Image = None  # type: ignore[assignment]

from src.core.errors import CoreError, IdentityMaskProjectionError
from src.core.renderers import render_identity_png, render_yolo
from src.core.results import ObjectResult, SingleImageOutcome
from src.core.sinks import ArtifactSink, ArtifactSinkError, StoredArtifact
from src.version import __version__

from .errors import ServiceError
from .artifacts import ArtifactDeliveryLedger, ArtifactSelection
from .rle import MaskRLEError, SerializationTimeout, encode_mask_rle

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
_DEFAULT_SINGLE_ARTIFACT_BYTES = 32 * 1024 * 1024
_DEFAULT_TOTAL_ARTIFACT_BYTES = 128 * 1024 * 1024
_DEFAULT_RLE_RUNS_PER_OBJECT = 250_000
_DEFAULT_RLE_RUNS_TOTAL = 1_000_000
_DETERMINISTIC_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ResponseContext:
    """Request-scoped metadata and immutable operator response budgets."""

    request_id: str
    model_id: str
    verbosity: int
    response_format: str
    config_digest: str
    class_mapping: Mapping[str, int]
    config_warnings: Sequence[str] = field(default_factory=tuple)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    max_objects: int = 256
    max_response_artifacts: int = MAX_RESPONSE_ARTIFACTS
    max_single_artifact_bytes: int = _DEFAULT_SINGLE_ARTIFACT_BYTES
    max_total_raw_artifact_bytes: int = _DEFAULT_TOTAL_ARTIFACT_BYTES
    max_mask_rle_runs_per_object: int = _DEFAULT_RLE_RUNS_PER_OBJECT
    max_mask_rle_runs_total: int = _DEFAULT_RLE_RUNS_TOTAL
    max_response_bytes: int = 256 * 1024 * 1024
    deadline_monotonic: Optional[float] = None
    candidate_views: Mapping[str, Any] = field(default_factory=dict)
    clip_routing: Mapping[str, Any] = field(default_factory=dict)
    artifact_ledger: Optional[ArtifactDeliveryLedger] = None
    service_safe_artifact_names: bool = False


@dataclass(frozen=True)
class _RawArtifact:
    name: str
    media_type: str
    payload: bytes


@dataclass(frozen=True)
class _PreparedResponse:
    document: Dict[str, Any]
    yolo_text: str
    artifacts: tuple[_RawArtifact, ...]
    ledger: ArtifactDeliveryLedger


def _artifact(name: str, media_type: str, payload: bytes) -> Dict[str, Any]:
    """Build the public JSON artifact descriptor (small compatibility seam)."""
    return {
        "name": name,
        "media_type": media_type,
        "encoding": "base64",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _encode_png(array: np.ndarray) -> bytes:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to encode PNG artifacts.")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _check_absolute_deadline(deadline_monotonic: Optional[float]) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise ServiceError("request serialization exceeded the deadline", code="timeout")


def _check_deadline(context: ResponseContext) -> None:
    _check_absolute_deadline(context.deadline_monotonic)


def _object_record(
    obj: ObjectResult,
    width: int,
    height: int,
    *,
    mask_rle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cx, cy, bw, bh = obj.normalized_bbox(width, height)
    record: Dict[str, Any] = {
        "instance_id": obj.instance_id,
        "source_candidate_id": obj.source_candidate_id,
        "filtered_index": (
            obj.filtered_index if obj.filtered_index is not None else obj.source_index
        ),
        "class_id": obj.class_id if obj.class_id is not None else 0,
        "label": obj.label,
        "bbox_xyxy": [int(v) for v in obj.bbox_xyxy],
        "bbox_normalized": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
        "area_px": obj.area_px,
        "centroid_rc": [round(obj.centroid_rc[0], 3), round(obj.centroid_rc[1], 3)],
    }
    optional_fields = {
        "predicted_iou": obj.predicted_iou,
        "stability_score": obj.stability_score,
        "clip_score": obj.clip_score,
        "blip3_answer": obj.blip3_answer,
        "geometry": obj.geometry(),
        "clip_scores": obj.metadata.get("clip_scores"),
        "clip_routing": obj.metadata.get("clip_routing"),
        "blip3_verification": obj.metadata.get("blip3_verification"),
        "blip3_verifications": obj.metadata.get("blip3_verifications"),
    }
    for key, value in optional_fields.items():
        if value is not None:
            record[key] = value
    if mask_rle is not None:
        record["mask_rle"] = dict(mask_rle)
    if obj.warnings:
        record["warnings"] = list(obj.warnings)
    return record


def _sam2_manifest(result: Any, *, include_raw_visualization: bool = False) -> Dict[str, Any]:
    """Return the complete typed SAM2 manifest, including legacy outcomes."""
    defaults = dict(SAM2_DEFAULTS)
    metadata = dict(getattr(result, "sam2_metadata", {}) or {})
    effective = dict(defaults)
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
    manifest = {
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
        "execution_time_ms": round(
            float(metadata.get("execution_time_ms", result.timings.get("stage.sam2", 0.0))), 3
        ),
        "resource_warnings": list(metadata.get("resource_warnings", [])),
        "operator_limits": dict(metadata.get("operator_limits", {})),
        "field_provenance": dict(metadata.get("field_provenance", {})),
    }
    if include_raw_visualization and metadata.get("raw_visualization") is not None:
        manifest["raw_visualization"] = dict(metadata["raw_visualization"])
    return manifest


def _stored_sink_artifact(stored: StoredArtifact) -> _RawArtifact:
    try:
        if stored.kind == "image-array":
            if stored.array is None:
                raise ValueError("missing image array")
            return _RawArtifact(stored.name, "image/png", _encode_png(stored.array))
        if stored.kind == "record":
            payload = json.dumps(stored.record, default=str, sort_keys=True).encode("utf-8")
            return _RawArtifact(stored.name, "application/json", payload)
        if stored.data is None:
            raise ValueError("missing artifact bytes")
        return _RawArtifact(
            stored.name, stored.content_type or "application/octet-stream", bytes(stored.data)
        )
    except (AssertionError, ArtifactSinkError, OSError, ValueError) as exc:
        raise ServiceError(
            "debug artifact could not be encoded", code="response_too_large"
        ) from exc


def _collect_raw_artifacts(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    sink: Optional[ArtifactSink],
) -> tuple[tuple[_RawArtifact, ...], ArtifactDeliveryLedger]:
    result = outcome.result
    artifacts: List[_RawArtifact] = []
    ledger = context.artifact_ledger or ArtifactDeliveryLedger(
        ArtifactSelection.from_mapping({}, applied=context.verbosity >= 3),
        max_response_artifacts=context.max_response_artifacts,
        max_debug_artifacts=context.max_response_artifacts,
        max_single_artifact_bytes=context.max_single_artifact_bytes,
        max_total_raw_artifact_bytes=context.max_total_raw_artifact_bytes,
        max_response_bytes=context.max_response_bytes,
        verbosity=context.verbosity,
    )
    _check_deadline(context)
    if context.verbosity >= 1:
        try:
            payload = render_identity_png(
                result.objects,
                width=result.image_width,
                height=result.image_height,
                ensure_all_ids=True,
            )
        except IdentityMaskProjectionError as exc:
            raise ServiceError(
                "identity representation cannot preserve a distinct source pixel for every object",
                code="inference_failure",
            ) from exc
        artifacts.append(_RawArtifact("identity-mask.png", "image/png", payload))
        _check_deadline(context)
    if context.verbosity >= 3:
        # Sink insertion order reflects the pipeline stages (SAM2, CLIP,
        # BLIP3, then post-filter debug).  Keep that order ahead of final
        # visualization streams for deterministic pagination.
        if sink is not None:
            for stored in sink.artifacts():
                _check_deadline(context)
                raw_artifact = _stored_sink_artifact(stored)
                ledger.import_delivered(
                    raw_artifact.name,
                    stage=None,
                    estimated_raw_bytes=(
                        int(stored.array.nbytes)
                        if stored.array is not None
                        else len(stored.data or b"")
                    ),
                    payload_size=len(raw_artifact.payload),
                    media_type=raw_artifact.media_type,
                )
                if ledger.status_for(raw_artifact.name) == "stored":
                    artifacts.append(raw_artifact)
                _check_deadline(context)
            if context.artifact_ledger is None:
                for omission in sink.omissions():
                    ledger.import_omission(omission)

        for visualization_index, (key, array) in enumerate(
            sorted(result.rendered.items(), key=lambda item: str(item[0])), start=1
        ):
            _check_deadline(context)
            safe_key = (
                f"stream-{visualization_index:04d}"
                if context.service_safe_artifact_names
                else str(key).replace("\\", "_").strip("/.") or "stream"
            )
            payload = _encode_png(array)
            name = f"visualization/{safe_key}.png"
            status = ledger.offer(
                name,
                stage="visualization",
                estimated_raw_bytes=int(array.nbytes),
                media_type="image/png",
            )
            ledger.mark_payload_size(name, len(payload))
            if status == "stored":
                artifacts.append(_RawArtifact(name, "image/png", payload))
            _check_deadline(context)
    _check_deadline(context)
    return tuple(artifacts), ledger


def _artifact_descriptor(
    artifact: _RawArtifact,
    *,
    include_data: bool,
    json_skeleton: bool = False,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    _check_absolute_deadline(deadline_monotonic)
    descriptor: Dict[str, Any] = {
        "name": artifact.name,
        "media_type": artifact.media_type,
        "encoding": "base64",
        "sha256": hashlib.sha256(artifact.payload).hexdigest(),
        "size": len(artifact.payload),
    }
    if include_data:
        _check_absolute_deadline(deadline_monotonic)
        descriptor["data"] = base64.b64encode(artifact.payload).decode("ascii")
        _check_absolute_deadline(deadline_monotonic)
    elif json_skeleton:
        descriptor["data"] = ""
    return descriptor


def _prepare(
    outcome: SingleImageOutcome, context: ResponseContext, sink: Optional[ArtifactSink]
) -> _PreparedResponse:
    result = outcome.result
    _check_deadline(context)
    if len(result.objects) > context.max_objects:
        raise ServiceError("object count exceeds the configured limit", code="response_too_large")
    try:
        yolo_text = render_yolo(
            result.objects, image_width=result.image_width, image_height=result.image_height
        )
        rle_records: list[Optional[Mapping[str, Any]]] = [None] * len(result.objects)
        if context.verbosity >= 3:
            total_runs = 0
            for index, obj in enumerate(result.objects):
                _check_deadline(context)
                rle = encode_mask_rle(
                    obj.mask,
                    max_runs=context.max_mask_rle_runs_per_object,
                    deadline_monotonic=context.deadline_monotonic,
                )
                total_runs += len(rle["counts"])
                if total_runs > context.max_mask_rle_runs_total:
                    raise ServiceError("mask RLE run limit exceeded", code="response_too_large")
                rle_records[index] = rle
            _check_deadline(context)
        artifacts, ledger = _collect_raw_artifacts(outcome, context, sink)
        _check_deadline(context)
    except SerializationTimeout as exc:
        raise ServiceError("request serialization exceeded the deadline", code="timeout") from exc
    except MaskRLEError as exc:
        raise ServiceError("mask RLE run limit exceeded", code="response_too_large") from exc
    except CoreError as exc:
        raise ServiceError(
            "pipeline result exceeds the representable object limit", code="inference_failure"
        ) from exc

    service_meta: Dict[str, Any] = {
        "request_id": context.request_id,
        "verbosity": context.verbosity,
        "finish_reason": "stop",
        "image": {"width": result.image_width, "height": result.image_height},
        "class_mapping": dict(context.class_mapping),
        "config_digest": context.config_digest,
        "package_version": __version__,
        "sam2": _sam2_manifest(result, include_raw_visualization=context.verbosity >= 3),
        "candidate_views": {
            str(stage): dict(values) for stage, values in context.candidate_views.items()
        },
        "clip_routing": dict(context.clip_routing),
    }
    if context.verbosity >= 3 and (
        getattr(result, "clip_routing_diagnostics", ()) or context.clip_routing
    ):
        service_meta["clip_routing_diagnostics"] = [
            dict(record) for record in result.clip_routing_diagnostics
        ]
    if context.verbosity >= 1:
        service_meta["artifacts"] = [
            _artifact_descriptor(
                artifact,
                include_data=False,
                json_skeleton=True,
                deadline_monotonic=context.deadline_monotonic,
            )
            for artifact in artifacts
        ]
    if context.verbosity >= 2:
        _check_deadline(context)
        service_meta["objects"] = [
            _object_record(
                obj,
                result.image_width,
                result.image_height,
                mask_rle=rle_records[index],
            )
            for index, obj in enumerate(result.objects)
        ]
    if context.verbosity >= 3:
        _check_deadline(context)
        service_meta["stage_statuses"] = [status.as_dict() for status in result.stage_statuses]
        service_meta["candidate_counts"] = dict(result.candidate_counts)
        if result.post_filter_diagnostics:
            service_meta["post_filter_diagnostics"] = dict(result.post_filter_diagnostics)
        service_meta["timings_ms"] = {
            key: round(value, 3) for key, value in sorted(result.timings.items())
        }
        provenance = result.provenance.as_dict()
        if context.runtime_metadata:
            provenance["runtime"] = dict(context.runtime_metadata)
        service_meta["provenance"] = provenance
        service_meta["warnings"] = list(result.warnings) + list(context.config_warnings)
        service_meta["candidate_view_inputs"] = [
            dict(record) for record in result.candidate_view_inputs
        ]
        service_meta["blip3_candidate_views"] = [
            dict(record) for record in result.blip3_candidate_views
        ]
        service_meta["artifact_delivery"] = ledger.document(
            artifacts={artifact.name: artifact.payload for artifact in artifacts}
        )
    _check_deadline(context)
    return _PreparedResponse(
        {
            "id": f"cmpl-{context.request_id}",
            "object": "text_completion",
            "created": int(_now_unix()),
            "model": context.model_id,
            "choices": [{"index": 0, "text": yolo_text, "finish_reason": "stop"}],
            "usage": None,
            "schema_version": SCHEMA_VERSION,
            "service": service_meta,
        },
        yolo_text,
        artifacts,
        ledger,
    )


def _json_size_upper_bound(
    prepared: _PreparedResponse, *, deadline_monotonic: Optional[float] = None
) -> int:
    # Use the same JSON encoder options as the final document so the base64
    # expansion delta is exact before any payload is encoded.
    _check_absolute_deadline(deadline_monotonic)
    skeleton = json.dumps(prepared.document, ensure_ascii=False).encode("utf-8")
    total = len(skeleton)
    for item in prepared.artifacts:
        _check_absolute_deadline(deadline_monotonic)
        # ``_prepare`` uses a two-byte empty data value as its skeleton.  Add
        # only the replacement base64 characters so the hard check does not
        # omit an artifact that actually fits.
        total += max(0, 4 * ((len(item.payload) + 2) // 3) - 2)
    _check_absolute_deadline(deadline_monotonic)
    return total


def _refresh_artifact_statuses(value: Any, ledger: ArtifactDeliveryLedger) -> None:
    """Refresh every nested candidate/debug status after delivery changes."""
    if isinstance(value, dict):
        for name_key, status_key in (
            ("artifact_name", "artifact_status"),
            ("input_artifact_name", "input_artifact_status"),
        ):
            name = value.get(name_key)
            status = ledger.status_for(name) if isinstance(name, str) else None
            if status is not None and status_key in value:
                value[status_key] = status
        for child in value.values():
            _refresh_artifact_statuses(child, ledger)
    elif isinstance(value, list):
        for child in value:
            _refresh_artifact_statuses(child, ledger)


def _refresh_prepared(
    prepared: _PreparedResponse,
    context: ResponseContext,
    *,
    include_data: bool = False,
    json_skeleton: bool = False,
) -> _PreparedResponse:
    """Refresh descriptors and the L3 ledger after optional tail omission."""
    artifacts = tuple(
        artifact
        for artifact in prepared.artifacts
        if (context.verbosity >= 1 and artifact.name == "identity-mask.png")
        or prepared.ledger.status_for(artifact.name) == "stored"
    )
    document = json.loads(json.dumps(prepared.document))
    if context.verbosity >= 1:
        document["service"]["artifacts"] = [
            _artifact_descriptor(
                artifact,
                include_data=include_data,
                json_skeleton=json_skeleton,
                deadline_monotonic=context.deadline_monotonic,
            )
            for artifact in artifacts
        ]
    if context.verbosity >= 3:
        document["service"]["artifact_delivery"] = prepared.ledger.document(
            artifacts={artifact.name: artifact.payload for artifact in artifacts}
        )
        _refresh_artifact_statuses(document["service"], prepared.ledger)
    return dataclass_replace(prepared, document=document, artifacts=artifacts)


def build_completion_json(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    *,
    sink: Optional[ArtifactSink] = None,
) -> Dict[str, Any]:
    """Prepare one bounded JSON response, rejecting before base64 expansion."""
    prepared = _prepare(outcome, context, sink)
    while (
        _json_size_upper_bound(prepared, deadline_monotonic=context.deadline_monotonic)
        > context.max_response_bytes
    ):
        if not prepared.ledger.drop_last_for_response():
            raise ServiceError(
                "assembled JSON response exceeds the maximum response size",
                code="response_too_large",
            )
        prepared = _refresh_prepared(prepared, context, json_skeleton=True)
    _check_deadline(context)
    prepared = _refresh_prepared(prepared, context, include_data=True)
    document = prepared.document
    _check_deadline(context)
    encoded_size = len(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    _check_deadline(context)
    if encoded_size > context.max_response_bytes:
        raise ServiceError(
            "assembled JSON response exceeds the maximum response size",
            code="response_too_large",
        )
    return document


def _now_unix() -> float:
    return time.time()


def _zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=_DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def build_completion_zip(
    outcome: SingleImageOutcome,
    context: ResponseContext,
    *,
    sink: Optional[ArtifactSink] = None,
    max_bytes: int,
) -> bytes:
    """Build ZIP directly from prepared raw bytes, without a base64 duplicate."""
    prepared = _prepare(outcome, context, sink)

    def assemble(current: _PreparedResponse) -> bytes:
        current = _refresh_prepared(current, context)
        manifest_bytes = json.dumps(current.document, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            _check_deadline(context)
            _zip_entry(archive, "manifest.json", manifest_bytes)
            _zip_entry(archive, "detections.yolo.txt", current.yolo_text.encode("utf-8"))
            for artifact in current.artifacts:
                _check_deadline(context)
                _zip_entry(archive, artifact.name, artifact.payload)
        return buffer.getvalue()

    payload = assemble(prepared)
    while len(payload) > max_bytes:
        if not prepared.ledger.drop_last_for_response():
            raise ServiceError(
                "assembled ZIP response exceeds the maximum response size",
                code="response_too_large",
            )
        prepared = _refresh_prepared(prepared, context)
        payload = assemble(prepared)
    _check_deadline(context)
    return payload


def bound_json_size(
    document: Mapping[str, Any],
    max_response_bytes: int,
    *,
    deadline_monotonic: Optional[float] = None,
) -> None:
    """Reject an already-built JSON document over the configured cap."""
    _check_absolute_deadline(deadline_monotonic)
    encoded_size = len(json.dumps(document).encode("utf-8"))
    _check_absolute_deadline(deadline_monotonic)
    if encoded_size > max_response_bytes:
        raise ServiceError(
            "assembled JSON response exceeds the maximum response size",
            code="response_too_large",
        )
