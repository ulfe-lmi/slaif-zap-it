"""Pydantic schemas documenting the frozen wire contract in OpenAPI.

These models describe the response/error shapes for API documentation and
schema snapshots. The runtime builders construct plain dictionaries; the
models are attached through route ``responses`` metadata so validation of
dynamic level-gated fields stays honest without duplicating the logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.raw_visualizations import validate_raw_sam2_manifest

from .settings import SERVICE_MODEL_ID

__all__ = [
    "ArtifactDescriptor",
    "CandidateViewClipConfig",
    "CandidateViewBlip3Config",
    "CandidateViewsMetadata",
    "CandidateViewInputRecord",
    "Blip3CandidateViewRecord",
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


class CandidateViewClipConfig(BaseModel):
    """Effective request-local CLIP view policy."""

    mode: Literal["mask_dilated"]
    context_fraction: float = Field(ge=0.0, le=0.5)
    min_context_pixels: int = Field(ge=0, le=256)
    max_context_pixels: int = Field(ge=0, le=512)
    outside_fill: Literal["zero"]
    context_intensity: float = Field(ge=0.0, le=1.0)
    applied: bool


class CandidateViewBlip3Config(BaseModel):
    """Effective request-local single-image BLIP3 view policy."""

    mode: Literal["single_dilated_blur"]
    context_fraction: float = Field(ge=0.0, le=0.5)
    min_context_pixels: int = Field(ge=0, le=256)
    max_context_pixels: int = Field(ge=0, le=512)
    crop_extent_multiplier: float = Field(ge=1.0, le=2.0)
    blur_sigma_fraction: float = Field(ge=0.0, le=0.5)
    contour_enabled: bool
    contour_fraction: float = Field(ge=0.0, le=0.25)
    contour_min_pixels: int = Field(ge=1, le=3)
    contour_max_pixels: int = Field(ge=1, le=3)
    contour_rgb: List[int] = Field(min_length=3, max_length=3)
    applied: bool

    @model_validator(mode="after")
    def validate_bounds(self) -> "CandidateViewBlip3Config":
        if self.min_context_pixels > self.max_context_pixels:
            raise ValueError("min_context_pixels must not exceed max_context_pixels")
        if self.contour_min_pixels > self.contour_max_pixels:
            raise ValueError("contour_min_pixels must not exceed contour_max_pixels")
        if any(type(channel) is not int or not 0 <= channel <= 255 for channel in self.contour_rgb):
            raise ValueError("contour_rgb must contain strict integers from 0 to 255")
        return self


class CandidateViewsMetadata(BaseModel):
    """Effective candidate-view values and stage application status."""

    clip: CandidateViewClipConfig
    blip3: CandidateViewBlip3Config


class CandidateViewInputRecord(BaseModel):
    """Bounded provenance for one emitted exact model-input debug artifact."""

    stage: Literal["clip", "blip3"]
    source_candidate_id: int = Field(ge=1)
    filtered_index: int = Field(ge=0)
    question_id: Optional[int] = Field(default=None, ge=1)
    artifact_name: str
    target_bbox_xyxy: Optional[List[int]] = None
    context_bbox_xyxy: Optional[List[int]] = None
    effective_radius: Optional[int] = Field(default=None, ge=0, le=512)
    source_dimensions: Optional[Dict[str, int]] = None
    crop_dimensions: Optional[Dict[str, int]] = None
    model_input_dimensions: Dict[str, int]
    raw_mask_bbox_xyxy_inclusive: Optional[List[int]] = None
    support_bbox_xyxy_inclusive: Optional[List[int]] = None
    crop_bbox_xyxy_exclusive: Optional[List[int]] = None
    raw_context_radius: Optional[int] = Field(default=None, ge=0)
    effective_context_radius: Optional[int] = Field(default=None, ge=0, le=512)
    raw_contour_width: Optional[int] = Field(default=None, ge=0)
    effective_contour_width: Optional[int] = Field(default=None, ge=0, le=3)
    effective_blur_sigma: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    source_composite_dimensions: Optional[Dict[str, int]] = None

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> "CandidateViewInputRecord":
        """Require a fixed tokenized name matching this input record."""
        if self.stage == "clip":
            pattern = re.compile(
                r"^(?:[A-Za-z0-9][A-Za-z0-9_.-]*-)?"
                r"clip-candidate-view-CANDIDATE-(\d{4,})\.png$"
            )
            if self.question_id is not None:
                raise ValueError("CLIP candidate-view records cannot have question_id")
            expected_prefix = "clip-candidate-view-CANDIDATE-"
            if (
                self.target_bbox_xyxy is None
                or self.context_bbox_xyxy is None
                or self.effective_radius is None
                or self.source_dimensions is None
                or self.crop_dimensions is None
            ):
                raise ValueError("CLIP candidate-view records require CLIP geometry fields")
        else:
            pattern = re.compile(
                r"^(?:[A-Za-z0-9][A-Za-z0-9_.-]*-)?"
                r"blip3-verification-CANDIDATE-(\d{4,})-QUESTION-(\d{4,})\.png$"
            )
            if self.question_id is None:
                raise ValueError("BLIP3 candidate-view records require question_id")
            expected_prefix = "blip3-verification-CANDIDATE-"
            if (
                self.raw_mask_bbox_xyxy_inclusive is None
                or self.support_bbox_xyxy_inclusive is None
                or self.crop_bbox_xyxy_exclusive is None
                or self.raw_context_radius is None
                or self.effective_context_radius is None
                or self.raw_contour_width is None
                or self.effective_contour_width is None
                or self.effective_blur_sigma is None
                or self.source_composite_dimensions is None
            ):
                raise ValueError(
                    "BLIP3 candidate-view records require single-image geometry fields"
                )

        match = pattern.fullmatch(self.artifact_name)
        if match is None:
            raise ValueError(f"{self.stage} candidate-view artifact_name must be a fixed PNG name")
        if int(match.group(1)) != self.source_candidate_id:
            raise ValueError("candidate-view artifact name has the wrong source candidate ID")
        if self.stage == "blip3" and int(match.group(2)) != self.question_id:
            raise ValueError("candidate-view artifact name has the wrong question ID")
        if expected_prefix not in self.artifact_name:
            raise ValueError("candidate-view artifact name is missing its stage token")
        return self


class Blip3CandidateViewRecord(BaseModel):
    """One bounded L3 composition attempt, independent of debug artifacts."""

    source_candidate_id: int = Field(ge=1)
    filtered_index: int = Field(ge=0)
    status: Literal["rendered", "rejected"]
    reason: Optional[Literal["crop_cannot_contain_support_and_contour"]] = None
    render_mode: Literal["single_dilated_blur"]
    raw_mask_bbox_xyxy_inclusive: List[int]
    support_bbox_xyxy_inclusive: Optional[List[int]] = None
    crop_bbox_xyxy_exclusive: List[int]
    raw_context_radius: int = Field(ge=0)
    effective_context_radius: int = Field(ge=0, le=512)
    raw_contour_width: int = Field(ge=0)
    effective_contour_width: int = Field(ge=0, le=3)
    effective_blur_sigma: float = Field(ge=0.0, le=20.0)
    source_composite_dimensions: Dict[str, int]
    model_input_dimensions: Optional[Dict[str, int]] = None

    @model_validator(mode="after")
    def validate_status_reason(self) -> "Blip3CandidateViewRecord":
        if self.status == "rendered" and self.reason is not None:
            raise ValueError("rendered BLIP3 candidate views cannot have a diagnostic")
        if self.status == "rejected" and self.reason is None:
            raise ValueError("rejected BLIP3 candidate views require a diagnostic")
        return self


class ObjectRecord(BaseModel):
    instance_id: int = Field(ge=1)
    source_candidate_id: int = Field(ge=1)
    filtered_index: int = Field(ge=0)
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
    candidate_views: CandidateViewsMetadata
    artifacts: Optional[List[ArtifactDescriptor]] = None
    objects: Optional[List[ObjectRecord]] = None
    stage_statuses: Optional[List[Dict[str, Any]]] = None
    candidate_counts: Optional[Dict[str, int]] = None
    post_filter_diagnostics: Optional[PostFilterDiagnostics] = None
    timings_ms: Optional[Dict[str, float]] = None
    provenance: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    candidate_view_inputs: Optional[List[CandidateViewInputRecord]] = None
    blip3_candidate_views: Optional[List[Blip3CandidateViewRecord]] = None


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
