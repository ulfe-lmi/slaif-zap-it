"""Pydantic schemas documenting the frozen wire contract in OpenAPI.

These models describe the response/error shapes for API documentation and
schema snapshots. The runtime builders construct plain dictionaries; the
models are attached through route ``responses`` metadata so validation of
dynamic level-gated fields stays honest without duplicating the logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .settings import SERVICE_MODEL_ID

__all__ = [
    "ArtifactDescriptor",
    "ObjectRecord",
    "ServiceMetadata",
    "Choice",
    "CompletionResponse",
    "ErrorBody",
    "ErrorEnvelope",
    "HealthStatus",
]


class ArtifactDescriptor(BaseModel):
    name: str = Field(description="Logical artifact name; never a filesystem path")
    media_type: str
    encoding: str = Field(default="base64", description="Always base64 in JSON responses")
    sha256: str = Field(description="SHA-256 hex digest of the artifact bytes")
    size: int = Field(ge=0)
    data: str = Field(description="Base64-encoded payload")


class ObjectRecord(BaseModel):
    instance_id: int = Field(ge=1)
    class_id: int = Field(ge=0)
    label: Optional[str] = None
    bbox_xyxy: List[int]
    bbox_normalized: List[float]
    area_px: int = Field(ge=0)
    centroid_rc: List[float]
    predicted_iou: Optional[float] = None
    stability_score: Optional[float] = None
    clip_score: Optional[float] = None
    blip3_answer: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    mask_rle: Optional[Dict[str, Any]] = Field(
        default=None,
        description="L3-only uncompressed column-major COCO-style mask RLE",
    )
    warnings: Optional[List[str]] = None


class ServiceMetadata(BaseModel):
    request_id: str
    verbosity: int = Field(ge=0, le=3)
    finish_reason: str
    image: Dict[str, int]
    class_mapping: Dict[str, int]
    config_digest: str
    artifacts: Optional[List[ArtifactDescriptor]] = None
    objects: Optional[List[ObjectRecord]] = None
    stage_statuses: Optional[List[Dict[str, Any]]] = None
    candidate_counts: Optional[Dict[str, int]] = None
    timings_ms: Optional[Dict[str, float]] = None
    provenance: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


class Choice(BaseModel):
    index: int = 0
    text: str = Field(description="Normalized five-field YOLO lines, one per object")
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = SERVICE_MODEL_ID
    choices: List[Choice]
    usage: None = None
    schema_version: str
    service: ServiceMetadata


class ErrorBody(BaseModel):
    code: str
    message: str = Field(description="Sanitized; never contains raw inputs or internals")
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthStatus(BaseModel):
    status: str
    uptime_s: Optional[str] = None
