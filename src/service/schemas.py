"""Pydantic schemas documenting the frozen wire contract in OpenAPI.

These models describe the response/error shapes for API documentation and
schema snapshots. The runtime builders construct plain dictionaries; the
models are attached through route ``responses`` metadata so validation of
dynamic level-gated fields stays honest without duplicating the logic.
"""

from __future__ import annotations

import re
import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from src.core.raw_visualizations import validate_raw_sam2_manifest

from .settings import SERVICE_MODEL_ID

__all__ = [
    "ArtifactDescriptor",
    "ArtifactOmission",
    "ArtifactSelectionMetadata",
    "ArtifactBudgetMetadata",
    "ArtifactDeliveryMetadata",
    "StageStatus",
    "CandidateCounts",
    "TimingMetadata",
    "ClipRoutingRule",
    "ClipRoutingConfiguration",
    "RuntimeMetadata",
    "ProvenanceMetadata",
    "Sam2ConfigValues",
    "Sam2ResourceAlternative",
    "Sam2ResourceLimitDetails",
    "Blip3ResourceLimitDetails",
    "ClipPromptValidationDetails",
    "ClipPromptMetadata",
    "CandidateViewClipConfig",
    "CandidateViewBlip3Config",
    "CandidateViewsMetadata",
    "CandidateViewInputRecord",
    "Blip3CandidateViewRecord",
    "ClipRoutingDiagnostic",
    "ClipRoutingReason",
    "ClipRoutingCapOutcome",
    "Blip3VerificationRecord",
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


StrictCandidateId = Annotated[int, Field(strict=True, ge=1, le=256)]
FiniteNonNegativeFloat = Annotated[float, Field(ge=0)]


class ArtifactDescriptor(BaseModel):
    name: str = Field(description="Logical artifact name; never a filesystem path")
    media_type: str
    encoding: str = Field(default="base64", description="Always base64 in JSON responses")
    sha256: str = Field(description="SHA-256 hex digest of the artifact bytes")
    size: int = Field(ge=0)
    data: str = Field(description="Base64-encoded payload")


class ArtifactOmission(BaseModel):
    """One bounded optional artifact that was not delivered."""

    name: str = Field(description="Fixed logical artifact name")
    stage: Literal["sam2", "clip", "blip3", "visualization"]
    source_candidate_id: Optional[int] = Field(default=None, ge=1)
    question_id: Optional[int] = Field(default=None, ge=1)
    estimated_raw_bytes: int = Field(ge=0)
    reason: Literal[
        "not_selected_stage",
        "not_selected_candidate",
        "not_selected_page",
        "omitted_count_limit",
        "omitted_single_size_limit",
        "omitted_raw_total_limit",
        "omitted_response_limit",
    ]


class ArtifactSelectionMetadata(BaseModel):
    """Normalized diagnostic-artifact selection in a response."""

    stages: List[Literal["sam2", "clip", "blip3", "visualization"]] = Field(
        min_length=1, max_length=4
    )
    candidate_ids: Optional[List[StrictCandidateId]] = Field(default=None, max_length=256)
    page: int = Field(ge=1, le=65535)
    page_size: int = Field(ge=1, le=48)

    @model_validator(mode="after")
    def validate_selection(self) -> "ArtifactSelectionMetadata":
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("artifact selection stages must be unique")
        if self.candidate_ids is not None:
            if len(set(self.candidate_ids)) != len(self.candidate_ids):
                raise ValueError("artifact selection candidate IDs must be unique")
        return self


class ArtifactBudgetMetadata(BaseModel):
    """Operator-owned optional artifact budgets disclosed without host data."""

    max_response_artifacts: int = Field(ge=1)
    max_debug_artifacts: int = Field(ge=1)
    max_single_artifact_bytes: int = Field(ge=1)
    max_total_raw_artifact_bytes: int = Field(ge=1)
    max_response_bytes: int = Field(ge=1)


class ArtifactDeliveryMetadata(BaseModel):
    """L3 selection, admission, omission, and exact byte accounting."""

    requested: ArtifactSelectionMetadata
    effective: ArtifactSelectionMetadata
    applied: bool
    operator_budgets: ArtifactBudgetMetadata
    eligible_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    delivered_count: int = Field(ge=0)
    selection_excluded_count: int = Field(ge=0)
    budget_omitted_count: int = Field(ge=0)
    unreported_overflow_count: int = Field(ge=0)
    unreported_selection_excluded_count: int = Field(default=0, ge=0)
    unreported_budget_omitted_count: int = Field(default=0, ge=0)
    estimated_raw_bytes: int = Field(ge=0)
    estimated_base64_bytes: int = Field(ge=0)
    estimated_zip_bytes: int = Field(ge=0)
    actual_delivered_raw_bytes: int = Field(ge=0)
    actual_delivered_base64_bytes: int = Field(ge=0)
    actual_delivered_zip_bytes: Optional[int] = Field(default=None, ge=0)
    truncated: bool
    delivered_names: List[str] = Field(max_length=577)
    omitted: List[ArtifactOmission] = Field(max_length=576)
    warnings: List[str] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def validate_accounting(self) -> "ArtifactDeliveryMetadata":
        if (
            self.delivered_count + self.budget_omitted_count + self.selection_excluded_count
            != self.eligible_count
        ):
            raise ValueError("artifact delivery counts do not reconcile")
        if self.selected_count != self.delivered_count + self.budget_omitted_count:
            raise ValueError("artifact selected count does not reconcile")
        if self.truncated != (self.budget_omitted_count > 0):
            raise ValueError("artifact delivery truncation does not reconcile")
        if (
            self.budget_omitted_count
            + self.selection_excluded_count
            - self.unreported_overflow_count
            != len(self.omitted)
        ):
            # Internal overflow is allowed to hide entries, but never undercount
            # a visible omission record.
            if self.unreported_overflow_count == 0:
                raise ValueError("artifact omission ledger does not reconcile")
        return self


class StageStatus(BaseModel):
    """One bounded pipeline-stage status emitted at verbosity 3."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=64)
    status: Literal["executed", "skipped", "not_configured"]
    detail: str = Field(default="", max_length=512)
    duration_ms: Optional[FiniteNonNegativeFloat] = Field(
        default=None,
        description="Finite non-negative stage duration in milliseconds.",
    )
    warnings: List[str] = Field(default_factory=list, max_length=16)


class CandidateCounts(BaseModel):
    """Canonical non-negative counts emitted by the executed pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    sam2_candidates: Optional[int] = Field(default=None, ge=0)
    raw_sam2_generated: Optional[int] = Field(default=None, ge=0)
    non_empty_candidates: Optional[int] = Field(default=None, ge=0)
    after_area_bbox: Optional[int] = Field(default=None, ge=0)
    geometry_evaluated: Optional[int] = Field(default=None, ge=0)
    after_geometry: Optional[int] = Field(default=None, ge=0)
    geometry_rejected: Optional[int] = Field(default=None, ge=0)
    after_clip: Optional[int] = Field(default=None, ge=0)
    clip_scored: Optional[int] = Field(default=None, ge=0)
    initially_routed: Optional[int] = Field(default=None, ge=0)
    routed_after_cap: Optional[int] = Field(default=None, ge=0)
    blip3_verified: Optional[int] = Field(default=None, ge=0)
    final: Optional[int] = Field(default=None, ge=0)
    after_final_label_filter: Optional[int] = Field(default=None, ge=0)


class TimingMetadata(RootModel[Dict[str, FiniteNonNegativeFloat]]):
    """Bounded timing map; keys are dynamic ``stage.<name>`` pipeline timers."""

    root: Dict[str, FiniteNonNegativeFloat] = Field(
        max_length=64,
        description="Dynamic stage.<name> timing keys with finite millisecond values.",
    )

    @model_validator(mode="after")
    def validate_finite(self) -> "TimingMetadata":
        if any(not math.isfinite(value) for value in self.root.values()):
            raise ValueError("timing values must be finite milliseconds")
        return self


class ClipRoutingRule(BaseModel):
    """Effective request-local target rule for CLIP-to-BLIP3 routing."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    labels: List[str] = Field(min_length=1, max_length=32)
    top_k: Optional[int] = Field(default=None, ge=1, le=256, strict=True)
    score_margin_from_best: Optional[FiniteNonNegativeFloat] = Field(default=None, le=2.0)
    minimum_target_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    uncertain_labels: List[str] = Field(default_factory=list, max_length=32)
    max_candidates: Optional[int] = Field(default=None, ge=1, le=256, strict=True)


class ClipRoutingConfiguration(BaseModel):
    """Effective CLIP routing configuration, retaining the legacy map shape."""

    model_config = ConfigDict(extra="forbid")

    route_to_blip3: Optional[ClipRoutingRule] = None


class RuntimeModelIdentity(BaseModel):
    """Sanitized identity for one operator-approved resident model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=256)


class RuntimeDeviceMetadata(BaseModel):
    """Sanitized selected-device facts included in L3 provenance."""

    model_config = ConfigDict(extra="forbid")

    physical_index: Optional[int] = Field(default=None, ge=0)
    logical: str = Field(min_length=1, max_length=32)
    visible_count: Optional[int] = Field(default=None, ge=1)
    uuid: Optional[str] = Field(default=None, min_length=1, max_length=128)
    name: Optional[str] = Field(default=None, min_length=1, max_length=256)
    total_memory_mib: Optional[int] = Field(default=None, ge=1)


class RuntimeResidencyMetadata(BaseModel):
    """Sanitized model-residency policy facts."""

    model_config = ConfigDict(extra="forbid")

    logical_device: Optional[str] = Field(default=None, min_length=1, max_length=32)
    all_models_resident: Optional[bool] = None
    request_transition_policy: str = Field(min_length=1, max_length=64)


class RuntimeModelControlMetadata(BaseModel):
    """Sanitized fixed model-control route disclosure."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(min_length=1, max_length=32)
    paths: List[str] = Field(max_length=3)
    management_subset_only: bool


class RuntimeMetadata(BaseModel):
    """Typed, bounded operator provenance nested under the L3 runtime record."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(min_length=1, max_length=128)
    supported_profiles: Optional[List[str]] = Field(default=None, max_length=8)
    device: RuntimeDeviceMetadata
    models: Dict[str, RuntimeModelIdentity] = Field(min_length=1, max_length=3)
    residency: RuntimeResidencyMetadata
    model_control: Optional[RuntimeModelControlMetadata] = None


class ProvenanceMetadata(BaseModel):
    """Stable core provenance plus sanitized operator runtime facts."""

    model_config = ConfigDict(extra="forbid")

    config_digest: str = Field(min_length=1, max_length=128)
    core_version: str = Field(min_length=1, max_length=128)
    notes: List[str] = Field(default_factory=list, max_length=16)
    runtime: Optional[RuntimeMetadata] = None


class Sam2ConfigValues(BaseModel):
    """Request-safe SAM2 scalar mapping; omitted fields are genuinely absent."""

    model_config = ConfigDict(extra="forbid")

    profile: Optional[Literal["fast", "balanced", "quality"]] = None
    points_per_side: Optional[int] = Field(default=None, ge=1, le=1024)
    points_per_batch: Optional[int] = Field(default=None, ge=1, le=1024)
    pred_iou_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stability_score_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stability_score_offset: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    mask_threshold: Optional[float] = Field(default=None, ge=-32.0, le=32.0)
    box_nms_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    crop_n_layers: Optional[int] = Field(default=None, ge=0, le=8)
    crop_nms_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    crop_overlap_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    crop_n_points_downscale_factor: Optional[int] = Field(default=None, ge=1, le=32)
    min_mask_region_area: Optional[int] = Field(default=None, ge=0, le=64_000_000)
    use_m2m: Optional[bool] = None
    multimask_output: Optional[bool] = None
    debug: Optional[bool] = None


class Sam2ResourceAlternative(BaseModel):
    """One complete request-safe SAM2 alternative."""

    mask_generator: Sam2ConfigValues
    estimated_prompt_count: int = Field(ge=0)
    estimated_mask_prediction_count: int = Field(ge=0)


class Sam2ResourceLimitDetails(BaseModel):
    """Sanitized evidence for a rejected SAM2 operator-cap request."""

    limit_kind: Literal["field", "estimated_prompt_count", "estimated_mask_prediction_count"]
    requested: Sam2ConfigValues
    effective: Sam2ConfigValues
    selected_profile: Optional[Literal["fast", "balanced", "quality"]] = None
    estimated_prompt_count: int = Field(ge=0)
    estimated_mask_prediction_count: int = Field(ge=0)
    operator_limits: Dict[str, int]
    causing_values: Dict[str, Any]
    admissible_alternatives: List[Sam2ResourceAlternative] = Field(min_length=1, max_length=3)
    warning: str = Field(max_length=256)


class Blip3ResourceLimitDetails(BaseModel):
    """Sanitized evidence for a planned BLIP3 workload over its startup cap."""

    model_config = ConfigDict(extra="forbid")

    planned_questions: int = Field(ge=0, le=1_000_000)
    allowed_limit: int = Field(ge=1, le=256)
    controlling_field: Literal["SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS"]
    admissible_alternatives: List[str] = Field(min_length=1, max_length=3)


class ClipPromptValidationDetails(BaseModel):
    """Sanitized detail for one rejected canonical CLIP prompt input."""

    model_config = ConfigDict(extra="forbid")

    reason: Literal[
        "too_many_classes",
        "invalid_container_type",
        "empty_prompt_array",
        "invalid_prompt_type",
        "empty_prompt",
        "character_limit",
        "duplicate_prompt",
        "per_class_count",
        "total_count",
        "token_limit",
    ]
    class_identifier: Optional[str] = Field(default=None, min_length=1, max_length=64)
    prompt_index: Optional[int] = Field(default=None, ge=0, le=63)
    first_prompt_index: Optional[int] = Field(default=None, ge=0, le=63)
    actual_type: Optional[str] = Field(default=None, min_length=1, max_length=32)
    measured_class_count: Optional[int] = Field(default=None, ge=0)
    measured_per_class_count: Optional[int] = Field(default=None, ge=0)
    measured_total_count: Optional[int] = Field(default=None, ge=0)
    measured_character_count: Optional[int] = Field(default=None, ge=0)
    measured_token_count: Optional[int] = Field(default=None, ge=0)
    allowed_limit: int = Field(ge=0)


class ClipPromptMetadata(BaseModel):
    """Bounded L3 accounting for effective canonical CLIP prompts."""

    model_config = ConfigDict(extra="forbid")

    class_prompt_counts: Dict[str, int] = Field(min_length=1, max_length=32)
    total_prompt_count: int = Field(ge=1, le=256)
    tokenizer_limit: Literal[77]
    duplicate_policy: Literal["reject"]


class CandidateViewClipConfig(BaseModel):
    """Effective request-local CLIP view policy."""

    mode: Literal["raw_bbox_crop"]
    context_fraction: float = Field(ge=0.0, le=0.5)
    min_context_pixels: int = Field(ge=0, le=256)
    max_context_pixels: int = Field(ge=0, le=512)
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
    artifact_status: Optional[str] = None
    mask_bbox_xyxy_inclusive: Optional[List[int]] = None
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
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Effective request-local view settings used to build the input",
    )

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
                (self.target_bbox_xyxy is None and self.mask_bbox_xyxy_inclusive is None)
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
    clip_scores: Optional[Dict[str, float]] = Field(default=None, max_length=32)
    clip_routing: Optional["ClipRoutingDiagnostic"] = None
    blip3_answer: Optional[str] = None
    blip3_verification: Optional["Blip3VerificationRecord"] = None
    blip3_verifications: Optional[List["Blip3VerificationRecord"]] = None
    geometry: Optional[Dict[str, Any]] = None
    mask_rle: Optional[Dict[str, Any]] = Field(
        default=None,
        description="L3-only uncompressed column-major COCO-style mask RLE",
    )
    warnings: Optional[List[str]] = None


PostFilterReason = Literal[
    "maxsize",
    "empty_mask",
    "max_w",
    "max_h",
    "min_area",
    "max_area",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "min_aspect_ratio",
    "max_aspect_ratio",
    "border_touching",
]


class PostFilterLimits(BaseModel):
    maxsize: Optional[int] = Field(default=None, ge=0)
    max_w: Optional[int] = Field(default=None, ge=0)
    max_h: Optional[int] = Field(default=None, ge=0)
    min_area: Optional[int] = Field(default=None, ge=0)
    max_area: Optional[int] = Field(default=None, ge=0)
    min_width: Optional[int] = Field(default=None, ge=0)
    max_width: Optional[int] = Field(default=None, ge=0)
    min_height: Optional[int] = Field(default=None, ge=0)
    max_height: Optional[int] = Field(default=None, ge=0)
    min_aspect_ratio: Optional[float] = Field(default=None, ge=0.0)
    max_aspect_ratio: Optional[float] = Field(default=None, ge=0.0)
    allow_border_touching: Optional[bool] = None


class PostFilterRejection(BaseModel):
    source_index: Optional[int] = Field(default=None, ge=0)
    source_candidate_id: Optional[int] = Field(default=None, ge=1)
    filtered_index: Optional[int] = Field(default=None, ge=0)
    reason: PostFilterReason
    area_px: int = Field(ge=0)
    bbox_width_px: int = Field(ge=0)
    bbox_height_px: int = Field(ge=0)
    bbox_xyxy_inclusive: Optional[List[int]] = None
    aspect_ratio: Optional[float] = None
    border_touching: Optional[bool] = None
    configured_limit_field: Optional[str] = None
    configured_limit_value: Any = None


class PostFilterDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limits: PostFilterLimits
    evaluated: int = Field(default=0, ge=0)
    removed_by_maxsize: int = Field(default=0, ge=0)
    removed_empty_mask: int = Field(default=0, ge=0)
    removed_by_max_w: int = Field(default=0, ge=0)
    removed_by_max_h: int = Field(default=0, ge=0)
    removed_by_empty_mask: int = Field(default=0, ge=0)
    removed_by_min_area: int = Field(default=0, ge=0)
    removed_by_max_area: int = Field(default=0, ge=0)
    removed_by_min_width: int = Field(default=0, ge=0)
    removed_by_max_width: int = Field(default=0, ge=0)
    removed_by_min_height: int = Field(default=0, ge=0)
    removed_by_max_height: int = Field(default=0, ge=0)
    removed_by_min_aspect_ratio: int = Field(default=0, ge=0)
    removed_by_max_aspect_ratio: int = Field(default=0, ge=0)
    removed_by_border_touching: int = Field(default=0, ge=0)
    retained: int = Field(default=0, ge=0)
    non_empty: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    reason_precedence: List[PostFilterReason] = Field(min_length=4, max_length=10)
    rejections: List[PostFilterRejection] = Field(max_length=256)
    rejections_truncated: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_aggregate_accounting(self) -> "PostFilterDiagnostics":
        """Reject contradictory generated diagnostics instead of dropping fields."""
        legacy_fields = (
            "removed_by_maxsize",
            "removed_empty_mask",
            "removed_by_max_w",
            "removed_by_max_h",
        )
        canonical_fields = (
            "removed_by_empty_mask",
            "removed_by_min_area",
            "removed_by_max_area",
            "removed_by_min_width",
            "removed_by_max_width",
            "removed_by_min_height",
            "removed_by_max_height",
            "removed_by_min_aspect_ratio",
            "removed_by_max_aspect_ratio",
            "removed_by_border_touching",
        )
        supplied = self.model_fields_set
        if supplied.intersection(canonical_fields):
            removed = sum(getattr(self, field) for field in canonical_fields)
            if self.rejected != removed or self.evaluated != self.retained + removed:
                raise ValueError("canonical post-filter aggregates do not reconcile")
        elif supplied.intersection(legacy_fields):
            removed = sum(getattr(self, field) for field in legacy_fields)
            if self.evaluated != self.retained + removed:
                raise ValueError("legacy post-filter aggregates do not reconcile")
        return self


ClipRoutingReason = Literal[
    "target_top_1",
    "target_in_top_k",
    "target_within_score_margin",
    "target_exceeded_minimum_score",
    "explicitly_uncertain",
    "clear_negative",
    "max_candidate_limit",
]
ClipRoutingCapOutcome = Literal[
    "not_applicable", "not_applied", "retained", "capped_out", "not_routed"
]


class ClipRoutingDiagnostic(BaseModel):
    source_candidate_id: int = Field(ge=1)
    filtered_index: int = Field(ge=0)
    clip_scores: Dict[str, float]
    winning_prompt_indices: Dict[str, int] = Field(default_factory=dict, max_length=32)
    winning_prompt_index: Optional[int] = Field(default=None, ge=0, le=63)
    winning_prompt: Optional[str] = Field(default=None, max_length=512)
    winner: Optional[str] = None
    winning_label: Optional[str] = None
    chosen_target: Optional[str] = None
    target_rank: Optional[int] = Field(default=None, ge=1)
    chosen_target_rank: Optional[int] = Field(default=None, ge=1)
    target_score: Optional[float] = None
    best_score: Optional[float] = None
    best_score_delta: Optional[float] = None
    route_to_blip3: bool
    initially_routed: Optional[bool] = None
    matched_conditions: List[ClipRoutingReason]
    primary_reason: ClipRoutingReason
    primary_reason_before_cap: Optional[ClipRoutingReason] = None
    matched_conditions_before_cap: Optional[List[ClipRoutingReason]] = None
    cap_outcome: ClipRoutingCapOutcome
    crop_metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_cosine_scores(self) -> "ClipRoutingDiagnostic":
        values = list(self.clip_scores.values())
        values.extend(value for value in (self.target_score, self.best_score) if value is not None)
        if any(
            not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0 for value in values
        ):
            raise ValueError("CLIP scores must be finite cosine similarities in [-1, 1]")
        if self.best_score_delta is not None and (
            not math.isfinite(float(self.best_score_delta)) or self.best_score_delta < 0
        ):
            raise ValueError("CLIP best-score delta must be finite and non-negative")
        return self


class Blip3VerificationRecord(BaseModel):
    source_candidate_id: int = Field(ge=1)
    filtered_index: int = Field(ge=0)
    question_id: int = Field(ge=1)
    routing_target_label: Optional[str] = None
    routing_reason: Optional[ClipRoutingReason] = None
    configured_question: str
    effective_question: str
    raw_answer: str
    normalized_answer: str
    normalized_true_result: str
    normalized_false_result: str
    configured_true_result: str
    configured_false_result: str
    configured_true_label: str
    configured_false_label: str
    mapping_outcome: Literal["true_match", "false_match", "unmatched_answer"]
    input_artifact_name: Optional[str] = None
    input_artifact_status: str
    final_label: str


ObjectRecord.model_rebuild()


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

    requested: Sam2ConfigValues
    effective: Sam2ConfigValues
    sources: Dict[str, Literal["explicit", "profile", "default"]]
    selected_profile: Optional[Literal["fast", "balanced", "quality"]] = None
    estimated_prompt_count: int = Field(ge=0)
    estimated_mask_prediction_count: int = Field(ge=0)
    actual_candidate_count: int = Field(ge=0)
    execution_time_ms: float = Field(ge=0)
    resource_warnings: List[str]
    operator_limits: Dict[str, Optional[int]] = {}
    field_provenance: Dict[str, Dict[str, Any]] = {}
    raw_visualization: Optional[RawVisualizationManifest] = None


class ServiceMetadata(BaseModel):
    request_id: str
    verbosity: int = Field(ge=0, le=3)
    finish_reason: str
    image: Dict[str, int]
    class_mapping: Dict[str, int]
    config_digest: str
    package_version: str
    sam2: Sam2Metadata
    candidate_views: CandidateViewsMetadata
    clip_routing: Optional[ClipRoutingConfiguration] = None
    clip_prompts: Optional[ClipPromptMetadata] = None
    artifacts: Optional[List[ArtifactDescriptor]] = None
    artifact_delivery: Optional[ArtifactDeliveryMetadata] = None
    objects: Optional[List[ObjectRecord]] = None
    stage_statuses: Optional[List[StageStatus]] = Field(
        default=None,
        description="Ordered pipeline status records; only present at verbosity 3.",
    )
    candidate_counts: Optional[CandidateCounts] = Field(
        default=None,
        description="Canonical non-negative stage counts; only present at verbosity 3.",
    )
    post_filter_diagnostics: Optional[PostFilterDiagnostics] = None
    timings_ms: Optional[TimingMetadata] = Field(
        default=None,
        description=(
            "Finite non-negative millisecond timings keyed by dynamic stage.<name> names."
        ),
    )
    provenance: Optional[ProvenanceMetadata] = None
    warnings: Optional[List[str]] = None
    candidate_view_inputs: Optional[List[CandidateViewInputRecord]] = None
    blip3_candidate_views: Optional[List[Blip3CandidateViewRecord]] = None
    clip_routing_diagnostics: Optional[List[ClipRoutingDiagnostic]] = None


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
    details: Optional[
        Union[Sam2ResourceLimitDetails, Blip3ResourceLimitDetails, ClipPromptValidationDetails]
    ] = Field(
        default=None,
        description=(
            "Sanitized structured details for SAM2/BLIP3 resource limits or canonical CLIP "
            "prompt validation errors"
        ),
    )


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthStatus(BaseModel):
    status: str
    uptime_s: Optional[str] = None
