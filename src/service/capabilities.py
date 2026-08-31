"""The authenticated, static SAM2 service capability description."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from modules.segmenter.sam2 import SAM2_DEFAULTS, SAM2_PROFILES
from src.core.raw_visualizations import (
    RAW_CANDIDATE_ID_BASE,
    RAW_CANDIDATES_PER_SHEET,
    RAW_CONTACT_SHEET_COLUMNS,
    RAW_CONTACT_SHEET_ROWS,
    RAW_CONTEXT_PADDING_FRACTION,
    RAW_MASK_ALPHA,
    RAW_MAXIMUM_CONTACT_SHEETS,
    RAW_MAXIMUM_REPRESENTED_CANDIDATES,
    RAW_MAX_DIAGNOSTIC_PIXELS,
    RAW_MIN_CONTEXT_PADDING_PIXELS,
    RAW_TILE_CONTENT_HEIGHT,
    RAW_TILE_CONTENT_WIDTH,
    RAW_TILE_LABEL_HEIGHT,
)
from src.core.mask_views import CANDIDATE_VIEW_DEFAULTS
from src.runtime.models import APPROVED_MODEL_SPECS

from .envelope import SCHEMA_VERSION
from .settings import SERVICE_MODEL_ID, ServiceSettings

__all__ = [
    "CapabilityField",
    "CandidateViewCapabilityStage",
    "CandidateViewsCapability",
    "FixedControls",
    "CapabilitiesResponse",
    "RawSam2DebugPolicy",
    "build_capabilities",
]


class CapabilityField(BaseModel):
    """A public field type and bounded intrinsic constraint."""

    type: Literal["integer", "number", "boolean", "string", "array"]
    minimum: int | float | None = None
    maximum: int | float | None = None
    allowed: List[Any] | None = None


class FixedControls(BaseModel):
    """Fixed controls disclosed as policy without sensitive operator values."""

    model: Dict[str, str]
    checkpoint_path: str = Field(description="Operator-managed path; value is not disclosed")
    config_path: str = Field(description="Operator-managed path; value is not disclosed")
    device: str
    gpu: str
    dtype: str
    cache_paths: str = Field(description="Operator-managed paths; values are not disclosed")
    residency: str
    artifact_destinations: str
    point_grids: None = None
    output_mode: Literal["binary_mask"] = "binary_mask"
    arbitrary_kwargs: bool = False


class RawSam2DebugPolicy(BaseModel):
    """Static policy for the bounded L3 raw-candidate diagnostic."""

    trigger: str
    fixed_artifact_names: List[str]
    candidate_id_base: int
    columns: int
    rows: int
    candidates_per_sheet: int
    tile_content_width: int
    tile_content_height: int
    tile_label_height: int
    mask_alpha: float
    context_padding_fraction: float
    minimum_context_padding_pixels: int
    maximum_contact_sheets: int
    maximum_represented_candidates: int
    maximum_diagnostic_pixels: int
    candidate_order: str
    score_format: str
    palette: str
    diagnostics: Dict[str, str]
    truncation: str


class CandidateViewCapabilityStage(BaseModel):
    """Typed public policy for one candidate-view stage."""

    fields: Dict[str, CapabilityField]
    defaults: Dict[str, Any]
    debug_trigger: str
    fixed_artifact_name: str
    notes: Dict[str, str] = Field(default_factory=dict)


class CandidateViewsCapability(BaseModel):
    """Mask boundary, identity bases, and stage-specific candidate-view policy."""

    clip: CandidateViewCapabilityStage
    blip3: CandidateViewCapabilityStage
    dilation_formula: str
    context_rounding: str
    source_candidate_id: str
    filtered_index: str
    question_id: str
    bbox_policy: str
    crop_policy: str
    blur_policy: str
    contour_formula: str
    containment_policy: str
    effective_policy: str


class CapabilitiesResponse(BaseModel):
    """Explicit OpenAPI model for the read-only capabilities endpoint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_id: str
    supported_generator_fields: Dict[str, CapabilityField]
    intrinsic_ranges: Dict[str, List[Any]]
    operator_maxima: Dict[str, int]
    defaults: Dict[str, Any]
    profiles: Dict[str, Dict[str, Any]]
    source_precedence: List[str]
    estimation_formulas: Dict[str, str]
    fixed_controls: FixedControls
    raw_sam2_debug: RawSam2DebugPolicy
    candidate_views: CandidateViewsCapability


def _field_descriptions() -> Dict[str, CapabilityField]:
    numeric_ranges = {
        "points_per_side": ("integer", 1, 1024),
        "points_per_batch": ("integer", 1, 1024),
        "pred_iou_thresh": ("number", 0.0, 1.0),
        "stability_score_thresh": ("number", 0.0, 1.0),
        "stability_score_offset": ("number", 0.0, 10.0),
        "mask_threshold": ("number", -32.0, 32.0),
        "box_nms_thresh": ("number", 0.0, 1.0),
        "crop_n_layers": ("integer", 0, 8),
        "crop_nms_thresh": ("number", 0.0, 1.0),
        "crop_overlap_ratio": ("number", 0.0, 1.0),
        "crop_n_points_downscale_factor": ("integer", 1, 32),
        "min_mask_region_area": ("integer", 0, 64_000_000),
    }
    fields = {
        name: CapabilityField(type=kind, minimum=lower, maximum=upper)
        for name, (kind, lower, upper) in numeric_ranges.items()
    }
    fields.update(
        {
            "use_m2m": CapabilityField(type="boolean", allowed=[False, True]),
            "multimask_output": CapabilityField(type="boolean", allowed=[False, True]),
            "debug": CapabilityField(type="boolean", allowed=[False, True]),
            "profile": CapabilityField(type="string", allowed=list(SAM2_PROFILES)),
        }
    )
    return fields


def _candidate_view_fields(*, include_contour: bool) -> Dict[str, CapabilityField]:
    if include_contour:
        return {
            "mode": CapabilityField(type="string", allowed=["single_dilated_blur"]),
            "context_fraction": CapabilityField(type="number", minimum=0.0, maximum=0.5),
            "min_context_pixels": CapabilityField(type="integer", minimum=0, maximum=256),
            "max_context_pixels": CapabilityField(type="integer", minimum=0, maximum=512),
            "crop_extent_multiplier": CapabilityField(type="number", minimum=1.0, maximum=2.0),
            "blur_sigma_fraction": CapabilityField(type="number", minimum=0.0, maximum=0.5),
            "contour_enabled": CapabilityField(type="boolean", allowed=[False, True]),
            "contour_fraction": CapabilityField(type="number", minimum=0.0, maximum=0.25),
            "contour_min_pixels": CapabilityField(type="integer", minimum=1, maximum=3),
            "contour_max_pixels": CapabilityField(type="integer", minimum=1, maximum=3),
            "contour_rgb": CapabilityField(
                type="array",
            ),
        }
    return {
        "mode": CapabilityField(type="string", allowed=["mask_dilated"]),
        "context_fraction": CapabilityField(type="number", minimum=0.0, maximum=0.5),
        "min_context_pixels": CapabilityField(type="integer", minimum=0, maximum=256),
        "max_context_pixels": CapabilityField(type="integer", minimum=0, maximum=512),
        "outside_fill": CapabilityField(type="string", allowed=["zero"]),
        "context_intensity": CapabilityField(type="number", minimum=0.0, maximum=1.0),
    }


def build_capabilities(settings: ServiceSettings) -> Dict[str, Any]:
    """Build a deterministic capability document with no request state."""
    sam_spec = APPROVED_MODEL_SPECS["sam2"]
    response = CapabilitiesResponse(
        schema_version=SCHEMA_VERSION,
        model_id=SERVICE_MODEL_ID,
        supported_generator_fields=_field_descriptions(),
        intrinsic_ranges={
            name: [field.minimum, field.maximum]
            if field.minimum is not None
            else list(field.allowed or [])
            for name, field in _field_descriptions().items()
        },
        operator_maxima=settings.sam2_operator_caps,
        defaults=dict(SAM2_DEFAULTS),
        profiles={name: dict(values) for name, values in SAM2_PROFILES.items()},
        source_precedence=["explicit", "profile", "default"],
        estimation_formulas={
            "estimated_prompt_count": (
                "sum(4^layer * int(points_per_side / "
                "crop_n_points_downscale_factor^layer)^2 for layer 0..crop_n_layers)"
            ),
            "estimated_mask_prediction_count": (
                "estimated_prompt_count * (3 when multimask_output is true, otherwise 1)"
            ),
        },
        fixed_controls=FixedControls(
            model={"id": sam_spec.model_id, "revision": sam_spec.revision},
            checkpoint_path="operator-managed; not disclosed",
            config_path="operator-managed; not disclosed",
            device="logical cuda:0 only",
            gpu="operator-selected GPU; physical details not disclosed",
            dtype="float16",
            cache_paths="operator-managed local cache; not disclosed",
            residency="pinned model resident; request-local generator",
            artifact_destinations="in-memory response; no client-selected destination",
            point_grids=None,
            output_mode="binary_mask",
            arbitrary_kwargs=False,
        ),
        raw_sam2_debug=RawSam2DebugPolicy(
            trigger="verbosity == 3 and mask_generator.debug == true",
            fixed_artifact_names=[
                *[
                    f"sam2-candidates-page-{page:04d}.png"
                    for page in range(1, RAW_MAXIMUM_CONTACT_SHEETS + 1)
                ],
                "sam2-union-coverage.png",
                "sam2-overlap-heatmap.png",
                "sam2-uncovered-pixels.png",
            ],
            candidate_id_base=RAW_CANDIDATE_ID_BASE,
            columns=RAW_CONTACT_SHEET_COLUMNS,
            rows=RAW_CONTACT_SHEET_ROWS,
            candidates_per_sheet=RAW_CANDIDATES_PER_SHEET,
            tile_content_width=RAW_TILE_CONTENT_WIDTH,
            tile_content_height=RAW_TILE_CONTENT_HEIGHT,
            tile_label_height=RAW_TILE_LABEL_HEIGHT,
            mask_alpha=RAW_MASK_ALPHA,
            context_padding_fraction=RAW_CONTEXT_PADDING_FRACTION,
            minimum_context_padding_pixels=RAW_MIN_CONTEXT_PADDING_PIXELS,
            maximum_contact_sheets=RAW_MAXIMUM_CONTACT_SHEETS,
            maximum_represented_candidates=RAW_MAXIMUM_REPRESENTED_CANDIDATES,
            maximum_diagnostic_pixels=RAW_MAX_DIAGNOSTIC_PIXELS,
            candidate_order="ascending _source_index; candidate_id = _source_index + 1",
            score_format="IoU and stability: three decimals when finite, otherwise n/a",
            palette="fixed RGB arithmetic palette keyed only by candidate_id",
            diagnostics={
                "union": "black uncovered, white covered",
                "overlap": "black zero; blue-to-red fixed ramp scaled by observed maximum",
                "uncovered": "white uncovered, black covered; inverse of union at source resolution",
                "candidate_tiles": "padded crops may be enlarged for readability; aspect-preserving letterbox",
                "downscale": "diagnostics: nearest-neighbor only; never upscale; at most 2,000,000 pixels",
            },
            truncation="first 96 non-empty source-order candidates; one aggregate warning; no ninth sheet",
        ),
        candidate_views=CandidateViewsCapability(
            clip=CandidateViewCapabilityStage(
                fields=_candidate_view_fields(include_contour=False),
                defaults=dict(CANDIDATE_VIEW_DEFAULTS["clip"]),
                debug_trigger="verbosity == 3 and clip.debug == true",
                fixed_artifact_name="clip-candidate-view-CANDIDATE-0008.png",
                notes={
                    "visibility": "exact Euclidean support with legacy zero-fill CLIP behavior",
                },
            ),
            blip3=CandidateViewCapabilityStage(
                fields=_candidate_view_fields(include_contour=True),
                defaults=dict(CANDIDATE_VIEW_DEFAULTS["blip3"]),
                debug_trigger=("verbosity == 3 and an effective BLIP3 rule has debug == true"),
                fixed_artifact_name="blip3-verification-CANDIDATE-0008-QUESTION-0003.png",
                notes={
                    "source_composite": "RGB source crop; D and contour are restored byte-for-byte",
                    "debug_identity": "lossless PNG bytes equal the sole final model-input image",
                    "contour_rgb": "array of exactly three strict integers, each from 0 to 255",
                },
            ),
            dilation_formula=(
                "L = max(inclusive raw-mask width, inclusive raw-mask height); "
                "raw_context_radius = ceil(context_fraction * L); "
                "effective_context_radius = min(max(raw_context_radius, "
                "min_context_pixels), max_context_pixels)"
            ),
            context_rounding="CLIP only: floor(source_channel * context_intensity); BLIP3 uses Pillow GaussianBlur",
            source_candidate_id="one-based: _source_index + 1",
            filtered_index="zero-based: post-SAM2-filter retained source order",
            question_id="one-based: question index + 1",
            bbox_policy=(
                "raw-mask and support bboxes are inclusive xyxy; crop bbox is half-open xyxy"
            ),
            crop_policy=(
                "ceil(multiplier * raw bbox dimensions), center placed with floor on half-pixel ties, "
                "endpoints independently clamped without shifting"
            ),
            blur_policy=(
                "Pillow ImageFilter.GaussianBlur; sigma=min(max(blur_sigma_fraction * L, 2), 20)"
            ),
            contour_formula=(
                "contour = exact_euclidean_dilate(D, effective_contour_width) & ~D; "
                "disabled contour is empty"
            ),
            containment_policy=(
                "support plus contour must be wholly inside the clamped crop; otherwise "
                "candidate-local rejection crop_cannot_contain_support_and_contour"
            ),
            effective_policy=(
                "L0-L3 expose only the stage field set plus applied; detailed BLIP3 records "
                "and debug artifacts are L3-only"
            ),
        ),
    )
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=True)
    return response.dict(exclude_none=True)
