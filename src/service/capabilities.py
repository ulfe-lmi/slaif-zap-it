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
from .yaml_input import service_config_leaf_paths

__all__ = [
    "CapabilityField",
    "CapabilitySection",
    "ConfigurationCapabilities",
    "DiagnosticArtifactCapability",
    "ResponseEvidenceCapability",
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
    min_items: int | None = Field(default=None, exclude_if=lambda value: value is None)
    max_items: int | None = Field(default=None, exclude_if=lambda value: value is None)
    item_type: Literal["integer", "number", "boolean", "string"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    item_minimum: int | float | None = Field(default=None, exclude_if=lambda value: value is None)
    item_maximum: int | float | None = Field(default=None, exclude_if=lambda value: value is None)
    nullable: bool = False
    default: Any = None
    units: str | None = Field(default=None, exclude_if=lambda value: value is None)
    stage: str | None = Field(default=None, exclude_if=lambda value: value is None)
    description: str | None = Field(default=None, exclude_if=lambda value: value is None)
    profile_interaction: str | None = Field(default=None, exclude_if=lambda value: value is None)
    operator_limit: str | None = Field(default=None, exclude_if=lambda value: value is None)


class CapabilitySection(BaseModel):
    """Typed inventory for one fixed configuration surface."""

    fields: Dict[str, CapabilityField]
    description: str


class ConfigurationCapabilities(BaseModel):
    """Canonical machine-readable inventory of accepted YAML leaf paths."""

    fields: Dict[str, CapabilityField]
    sections: Dict[str, CapabilitySection]
    dynamic_fields: Dict[str, CapabilityField]


class DiagnosticArtifactCapability(BaseModel):
    """Typed capability surface for request-local L3 artifact selection."""

    fields: Dict[str, CapabilityField]
    default_selection: Dict[str, Any]
    semantics: Dict[str, str]


class ResponseEvidenceCapability(BaseModel):
    """Named response evidence surfaces disclosed by the API."""

    levels: Dict[str, str]
    artifact_delivery: Dict[str, str]
    error_details: Dict[str, str]


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
    clip_labels: Dict[str, CapabilityField]
    clip_routing: Dict[str, Any]
    geometry: Dict[str, CapabilityField]
    blip3_rules: Dict[str, CapabilityField]


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
    configuration: ConfigurationCapabilities
    diagnostic_artifacts: DiagnosticArtifactCapability
    response_evidence: ResponseEvidenceCapability


def _cap(
    kind: Literal["integer", "number", "boolean", "string", "array"],
    *,
    description: str,
    default: Any = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    allowed: List[Any] | None = None,
    stage: str | None = None,
    nullable: bool = False,
    units: str | None = None,
    profile_interaction: str | None = None,
    operator_limit: str | None = None,
    **kwargs: Any,
) -> CapabilityField:
    return CapabilityField(
        type=kind,
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
        allowed=allowed,
        stage=stage,
        nullable=nullable,
        units=units,
        profile_interaction=profile_interaction,
        operator_limit=operator_limit,
        **kwargs,
    )


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
    operator_fields = {
        "points_per_side",
        "points_per_batch",
        "crop_n_layers",
        "min_mask_region_area",
    }
    fields = {
        name: _cap(
            kind,
            minimum=lower,
            maximum=upper,
            default=SAM2_DEFAULTS[name],
            description=f"Safe SAM2 {name} value.",
            stage="sam2",
            units="pixels" if "area" in name else "count" if kind == "integer" else None,
            profile_interaction="explicit value overrides profile, then default",
            operator_limit=name if name in operator_fields else None,
        )
        for name, (kind, lower, upper) in numeric_ranges.items()
    }
    fields.update(
        {
            "use_m2m": _cap(
                "boolean",
                allowed=[False, True],
                default=False,
                stage="sam2",
                description="Enable SAM2 mask-to-mask refinement.",
            ),
            "multimask_output": _cap(
                "boolean",
                allowed=[False, True],
                default=True,
                stage="sam2",
                description="Emit multiple predictions per prompt.",
            ),
            "debug": _cap(
                "boolean",
                allowed=[False, True],
                default=False,
                stage="sam2",
                description="Enable raw SAM2 diagnostics at L3 only.",
            ),
            "profile": _cap(
                "string",
                allowed=list(SAM2_PROFILES),
                nullable=True,
                stage="sam2",
                description="Optional preset applied before explicit values.",
            ),
        }
    )
    return fields


def _decorate_candidate_fields(
    fields: Dict[str, CapabilityField], *, stage: str
) -> Dict[str, CapabilityField]:
    return {
        name: field.model_copy(
            update={
                "default": CANDIDATE_VIEW_DEFAULTS[stage].get(name),
                "stage": stage,
                "description": f"Request-local {stage} candidate-view {name} policy.",
                "profile_interaction": "independent from SAM2 profile",
            }
        )
        for name, field in fields.items()
    }


def _candidate_view_fields(*, include_contour: bool) -> Dict[str, CapabilityField]:
    if include_contour:
        return _decorate_candidate_fields(
            {
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
                    min_items=3,
                    max_items=3,
                    item_type="integer",
                    item_minimum=0,
                    item_maximum=255,
                ),
            },
            stage="blip3",
        )
    return _decorate_candidate_fields(
        {
            "mode": CapabilityField(type="string", allowed=["raw_bbox_crop"]),
            "context_fraction": CapabilityField(type="number", minimum=0.0, maximum=0.5),
            "min_context_pixels": CapabilityField(type="integer", minimum=0, maximum=256),
            "max_context_pixels": CapabilityField(type="integer", minimum=0, maximum=512),
        },
        stage="clip",
    )


def _configuration_capabilities() -> ConfigurationCapabilities:
    """Return the single canonical inventory used by validation and docs."""
    sam2 = _field_descriptions()
    preprocessing = {
        "roi": _cap(
            "string",
            nullable=True,
            stage="preprocessing",
            description="Optional source ROI string; false disables ROI.",
        ),
        "resize": _cap(
            "number",
            nullable=True,
            minimum=0.000001,
            stage="preprocessing",
            description="Positive resize factor.",
            units="factor",
        ),
        "debug": _cap(
            "boolean",
            default=False,
            stage="preprocessing",
            description="Emit the trusted pipeline ROI debug artifact at L3.",
        ),
    }
    geometry = {
        name: _cap(
            "boolean" if name in {"allow_border_touching", "debug"} else "number",
            minimum=None if name in {"allow_border_touching", "debug"} else 0,
            maximum=None
            if name in {"allow_border_touching", "debug"}
            else (1000 if "aspect" in name else 64_000_000 if "area" in name else 32_768),
            allowed=[False, True] if name in {"allow_border_touching", "debug"} else None,
            nullable=True,
            default=(
                True if name == "allow_border_touching" else False if name == "debug" else None
            ),
            stage="postsam2processing",
            description=f"Canonical post-SAM2 {name} constraint.",
            operator_limit="min_mask_region_area" if name == "min_area" else None,
        )
        for name in (
            "min_area",
            "max_area",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_aspect_ratio",
            "max_aspect_ratio",
            "allow_border_touching",
            "debug",
        )
    }
    geometry.update(
        {
            alias: _cap(
                "integer",
                minimum=0,
                stage="postsam2processing",
                description=f"Legacy alias for postsam2processing.{canonical}.",
            )
            for alias, canonical in {
                "maxsize": "max_area",
                "max_w": "max_width",
                "max_h": "max_height",
            }.items()
        }
    )
    clip_fields = {
        "debug": _cap(
            "boolean",
            default=False,
            stage="clip",
            description="Enable CLIP input PNG diagnostics at L3.",
        ),
        "labels.<identifier>": _cap(
            "string",
            minimum=1,
            maximum=512,
            stage="clip",
            description="Dynamic CLIP label prompt.",
            units="characters",
        ),
    }
    routing_fields = {
        f"route_to_blip3.{name}": _cap(
            "array"
            if name in {"labels", "uncertain_labels"}
            else "integer"
            if name in {"top_k", "max_candidates"}
            else "number",
            minimum=(
                1
                if name in {"top_k", "max_candidates"}
                else 0.0
                if name == "score_margin_from_best"
                else -1.0
                if name == "minimum_target_score"
                else None
            ),
            maximum=(
                256
                if name == "max_candidates"
                else 2.0
                if name == "score_margin_from_best"
                else 1.0
                if name == "minimum_target_score"
                else None
            ),
            nullable=name
            in {"top_k", "score_margin_from_best", "minimum_target_score", "max_candidates"},
            stage="clip_routing",
            description=f"CLIP routing {name} policy.",
            units="cosine" if "score" in name else None,
        )
        for name in (
            "labels",
            "top_k",
            "score_margin_from_best",
            "minimum_target_score",
            "uncertain_labels",
            "max_candidates",
        )
    }
    blip_fields = {
        f"<routing_label>.{name}": _cap(
            "boolean" if name == "debug" else "string",
            minimum=None if name == "debug" else 1,
            maximum=None if name == "debug" else 2048,
            allowed=[False, True] if name == "debug" else None,
            stage="blip3",
            description=f"Dynamic BLIP3 rule {name}.",
        )
        for name in (
            "question",
            "trueresult",
            "falseresult",
            "newcategory",
            "falsecategory",
            "debug",
        )
    }
    candidate_fields = {
        f"clip.{name}": field
        for name, field in _candidate_view_fields(include_contour=False).items()
    }
    candidate_fields.update(
        {
            f"blip3.{name}": field
            for name, field in _candidate_view_fields(include_contour=True).items()
        }
    )
    for path, field in tuple(candidate_fields.items()):
        stage, name = path.split(".", 1)
        candidate_fields[path] = field.model_copy(
            update={
                "default": CANDIDATE_VIEW_DEFAULTS[stage].get(name),
                "stage": stage,
                "description": f"Request-local {stage} candidate-view {name} policy.",
                "profile_interaction": "independent from SAM2 profile",
            }
        )
    visualization = {
        "alpha": _cap(
            "number",
            minimum=0.0,
            maximum=1.0,
            default=0.6,
            stage="visualization",
            description="Default overlay alpha.",
        ),
        "labels": _cap(
            "array",
            item_type="string",
            stage="label_filter",
            description="Terminal labels retained for output.",
        ),
    }
    for stage in ("sam2", "clip", "blip3"):
        for field_name, field in {
            "id": _cap(
                "string",
                minimum=1,
                maximum=64,
                stage="visualization",
                description="Safe visualization identifier.",
            ),
            "renderer": _cap(
                "string",
                allowed=["annotated", "alpha-overlay", "annotated-labelled"],
                stage="visualization",
                description="Supported fixed renderer enum.",
            ),
            "alpha": _cap(
                "number",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
                stage="visualization",
                description="Optional stream alpha.",
            ),
            "show_confidence": _cap(
                "boolean",
                default=False,
                nullable=True,
                stage="visualization",
                description="Show bounded CLIP confidence on labelled output.",
            ),
        }.items():
            visualization[f"{stage}.<index>.{field_name}"] = field
    diagnostic = {
        "stages": _cap(
            "array",
            allowed=list(("sam2", "clip", "blip3", "visualization")),
            stage="artifact_delivery",
            description="Optional stages to deliver.",
        ),
        "candidate_ids": _cap(
            "array",
            nullable=True,
            minimum=1,
            maximum=256,
            item_type="integer",
            item_minimum=1,
            item_maximum=256,
            stage="artifact_delivery",
            description="Optional one-based source candidate filter.",
        ),
        "page": _cap(
            "integer",
            minimum=1,
            maximum=65535,
            default=1,
            stage="artifact_delivery",
            description="One-based selected artifact page.",
        ),
        "page_size": _cap(
            "integer",
            minimum=1,
            maximum=48,
            default=48,
            stage="artifact_delivery",
            description="Selected artifacts per page.",
        ),
    }
    sections = {
        "root": CapabilitySection(
            fields={
                "alpha": _cap(
                    "number",
                    minimum=0.0,
                    maximum=1.0,
                    default=0.6,
                    stage="pipeline",
                    description="Global mask blend alpha.",
                )
            },
            description="Top-level algorithm control.",
        ),
        "preprocessing": CapabilitySection(
            fields=preprocessing, description="ROI and resize controls."
        ),
        "mask_generator": CapabilitySection(
            fields=sam2, description="Request-safe SAM2 generator scalars."
        ),
        "postsam2processing": CapabilitySection(
            fields=geometry, description="Canonical geometry and legacy aliases."
        ),
        "clip": CapabilitySection(fields=clip_fields, description="CLIP labels and debug control."),
        "clip_routing": CapabilitySection(
            fields=routing_fields, description="Complete CLIP-to-BLIP3 routing policy."
        ),
        "blip3": CapabilitySection(
            fields=blip_fields, description="Dynamic BLIP3 verification rules."
        ),
        "candidate_views": CapabilitySection(
            fields=candidate_fields, description="CLIP and BLIP3 candidate-view policies."
        ),
        "visualization": CapabilitySection(
            fields=visualization, description="Safe visualization entries and renderers."
        ),
        "diagnostic_artifacts": CapabilitySection(
            fields=diagnostic, description="Request-local optional artifact selection."
        ),
    }
    flat: Dict[str, CapabilityField] = {}
    for section_name, section in sections.items():
        prefix = "" if section_name == "root" else section_name + "."
        for path, field in section.fields.items():
            flat[prefix + path] = field
    if set(flat) != set(service_config_leaf_paths()):
        raise RuntimeError("service capability inventory drifted from the YAML validator")
    dynamic = {
        path: field
        for path, field in flat.items()
        if "<" in path or "identifier" in path or "routing_label" in path
    }
    return ConfigurationCapabilities(fields=flat, sections=sections, dynamic_fields=dynamic)


def build_capabilities(settings: ServiceSettings) -> Dict[str, Any]:
    """Build a deterministic capability document with no request state."""
    sam_spec = APPROVED_MODEL_SPECS["sam2"]
    configuration = _configuration_capabilities()
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
                    "visibility": "complete rectangular source crop; no mask-derived pixel alteration",
                    "legacy": "trusted CLI may explicitly opt into mask_dilated; API never accepts it",
                },
            ),
            blip3=CandidateViewCapabilityStage(
                fields=_candidate_view_fields(include_contour=True),
                defaults=dict(CANDIDATE_VIEW_DEFAULTS["blip3"]),
                debug_trigger=("verbosity == 3 and an effective BLIP3 rule has debug == true"),
                fixed_artifact_name="blip3-verification-CANDIDATE-0008-QUESTION-0003.png",
                notes={
                    "source_composite": (
                        "RGB source crop; support D pixels are restored from source bytes and "
                        "exterior contour pixels are painted with configured RGB"
                    ),
                    "debug_identity": (
                        "decoded lossless PNG RGB pixels equal the sole final model-input array"
                    ),
                    "contour_rgb": "array of exactly three strict integers, each from 0 to 255",
                },
            ),
            dilation_formula=(
                "L = max(inclusive raw-mask width, inclusive raw-mask height); "
                "CLIP raw_context_radius = floor(context_fraction * L + 0.5); BLIP3 retains its reviewed ceil formula; "
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
                "nominal size = ceil(multiplier * raw inclusive bbox dimensions); start = "
                "floor(inclusive pixel-center - (nominal size - 1) / 2); endpoints independently "
                "clamped without shifting"
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
            clip_labels={
                "identifier": CapabilityField(
                    type="string", allowed=["^[A-Za-z][A-Za-z0-9_-]{0,63}$"]
                ),
                "prompt": CapabilityField(type="string", minimum=1, maximum=512),
            },
            clip_routing={
                "execution_stage": "after complete CLIP vectors and before BLIP3",
                "fields": {
                    "labels": "non-empty unique list of configured label identifiers",
                    "top_k": "null or integer 1..number of CLIP labels",
                    "score_margin_from_best": "null or finite number 0..2",
                    "minimum_target_score": "null or finite cosine score -1..1",
                    "uncertain_labels": "disjoint configured label identifiers",
                    "max_candidates": "null or integer 1..256",
                },
                "logic": "OR of top-1, top-k, margin, minimum-score and uncertain-label conditions",
                "score_units": "cosine similarity in [-1, 1]",
                "reason_precedence": [
                    "target_top_1",
                    "target_in_top_k",
                    "target_within_score_margin",
                    "target_exceeded_minimum_score",
                    "explicitly_uncertain",
                    "clear_negative",
                ],
            },
            geometry={
                name: CapabilityField(
                    type=("boolean" if name in {"allow_border_touching", "debug"} else "number"),
                    minimum=None if name in {"allow_border_touching", "debug"} else 0,
                    maximum=None
                    if name in {"allow_border_touching", "debug"}
                    else (1000 if "aspect" in name else (64_000_000 if "area" in name else 32_768)),
                    allowed=[False, True] if name in {"allow_border_touching", "debug"} else None,
                )
                for name in (
                    "min_area",
                    "max_area",
                    "min_width",
                    "max_width",
                    "min_height",
                    "max_height",
                    "min_aspect_ratio",
                    "max_aspect_ratio",
                    "allow_border_touching",
                    "debug",
                )
            },
            blip3_rules={
                name: CapabilityField(type="string", minimum=1, maximum=2048)
                for name in (
                    "question",
                    "trueresult",
                    "falseresult",
                    "newcategory",
                    "falsecategory",
                )
            },
        ),
        configuration=configuration,
        diagnostic_artifacts=DiagnosticArtifactCapability(
            fields=configuration.sections["diagnostic_artifacts"].fields,
            default_selection={
                "stages": ["sam2", "clip", "blip3", "visualization"],
                "candidate_ids": None,
                "page": 1,
                "page_size": 48,
            },
            semantics={
                "stage_selection": "Only narrows eligible L3 artifacts; it never enables debug flags.",
                "candidate_selection": "Filters only CLIP and BLIP3 candidate-specific PNGs by one-based source_candidate_id.",
                "pagination": "Applied after stage and candidate selection in deterministic pipeline/name order.",
                "overflow": "Budget omission is non-fatal for optional artifacts; essential response overflow remains response_too_large.",
            },
        ),
        response_evidence=ResponseEvidenceCapability(
            levels={
                "0": "YOLO text and minimum envelope metadata.",
                "1": "L0 plus the uint16 identity mask.",
                "2": "L1 plus produced per-object algorithm evidence.",
                "3": "L2 plus bounded stage, provenance, timing, and optional artifact ledger.",
            },
            artifact_delivery={
                "truncated": "True only when an eligible selected optional artifact is omitted by an operator budget.",
                "omitted": "Bounded typed entries identify fixed name, stage, source/question IDs, estimate, and reason.",
                "hashes": "JSON descriptors and ZIP manifests identify exact delivered bytes by SHA-256 and size.",
            },
            error_details={
                "resource_limit": "SAM2 capacity rejections include sanitized estimates, causes, limits, and alternatives.",
                "compatibility": "Other error envelopes retain code, message, and request_id only.",
            },
        ),
    )
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=True)
    return response.dict(exclude_none=True)
