"""Pydantic schemas documenting the frozen wire contract in OpenAPI.

These models describe the response/error shapes for API documentation and
schema snapshots. The runtime builders construct plain dictionaries; the
models are attached through route ``responses`` metadata so validation of
dynamic level-gated fields stays honest without duplicating the logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.raw_visualizations import validate_raw_sam2_manifest

from .settings import SERVICE_MODEL_ID

__all__ = [
    "ArtifactDescriptor",
    "ObjectRecord",
    "PostFilterReason",
    "PostFilterLimits",
    "PostFilterRejection",
    "PostFilterDiagnostics",
    "RawVisualizationDimensions",
    "RawVisualizationManifest",
    "Sam2Metadata",
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


PostFilterReason = Literal["maxsize", "empty_mask", "max_w", "max_h"]


class PostFilterLimits(BaseModel):
    maxsize: int = Field(ge=0)
    max_w: int = Field(ge=0)
    max_h: int = Field(ge=0)


class PostFilterRejection(BaseModel):
    source_index: int = Field(ge=0)
    reason: PostFilterReason
    area_px: int = Field(ge=0)
    bbox_width_px: int = Field(ge=0)
    bbox_height_px: int = Field(ge=0)


class PostFilterDiagnostics(BaseModel):
    limits: PostFilterLimits
    evaluated: int = Field(ge=0)
    removed_by_maxsize: int = Field(ge=0)
    removed_empty_mask: int = Field(ge=0)
    removed_by_max_w: int = Field(ge=0)
    removed_by_max_h: int = Field(ge=0)
    retained: int = Field(ge=0)
    reason_precedence: List[PostFilterReason] = Field(min_length=4, max_length=4)
    rejections: List[PostFilterRejection] = Field(max_length=256)
    rejections_truncated: int = Field(ge=0)


class RawVisualizationDimensions(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class RawVisualizationManifest(BaseModel):
    """L3 facts for bounded raw SAM2 candidate visualizations."""

    enabled: bool
    candidate_id_base: int = Field(ge=1)
    raw_candidate_count: int = Field(ge=0)
    visualizable_candidate_count: int = Field(ge=0)
    omitted_empty_candidate_count: int = Field(ge=0)
    represented_candidate_count: int = Field(ge=0, le=96)
    represented_candidate_ids: List[int] = Field(max_length=96)
    truncated_candidate_count: int = Field(ge=0)
    contact_sheet_count: int = Field(ge=0, le=8)
    covered_pixel_count: int = Field(ge=0)
    uncovered_pixel_count: int = Field(ge=0)
    max_overlap_count: int = Field(ge=0)
    overlap_histogram: Dict[str, int]
    overlap_histogram_overflow_pixel_count: int = Field(ge=0)
    overlap_histogram_truncated: bool
    source_dimensions: RawVisualizationDimensions
    diagnostic_dimensions: RawVisualizationDimensions
    artifact_names: List[str] = Field(max_length=11)
    warnings: List[str] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_arithmetic(self) -> "RawVisualizationManifest":
        """Keep the documented raw-rendering arithmetic explicit in the schema."""

        validate_raw_sam2_manifest(self.model_dump(mode="python"))
        return self


class Sam2Metadata(BaseModel):
    """Request-local SAM2 configuration and execution facts."""

    requested: Dict[str, Any]
    effective: Dict[str, Any]
    sources: Dict[str, Literal["explicit", "profile", "default"]]
    selected_profile: Optional[Literal["fast", "balanced", "quality"]] = None
    estimated_prompt_count: int = Field(ge=0)
    estimated_mask_prediction_count: int = Field(ge=0)
    actual_candidate_count: int = Field(ge=0)
    execution_time_ms: float = Field(ge=0)
    resource_warnings: List[str]
    raw_visualization: Optional[RawVisualizationManifest] = None


class ServiceMetadata(BaseModel):
    request_id: str
    verbosity: int = Field(ge=0, le=3)
    finish_reason: str
    image: Dict[str, int]
    class_mapping: Dict[str, int]
    config_digest: str
    sam2: Sam2Metadata
    artifacts: Optional[List[ArtifactDescriptor]] = None
    objects: Optional[List[ObjectRecord]] = None
    stage_statuses: Optional[List[Dict[str, Any]]] = None
    candidate_counts: Optional[Dict[str, int]] = None
    post_filter_diagnostics: Optional[PostFilterDiagnostics] = None
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
