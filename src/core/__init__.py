"""Typed, reusable, in-memory single-image ZAP-IT core (objective 001-a).

Public surface:

- :func:`.engine.run_single_image` — the canonical single-image entry point;
- :class:`.config.CoreConfig` — normalized algorithmic configuration boundary;
- :class:`.results.ObjectResult` / :class:`.results.PipelineResult` — typed
  deterministic results in original-image coordinates;
- :func:`.renderers.render_yolo` / :func:`.renderers.render_identity_png` —
  pure deterministic renderers sharing one final-object ordering;
- :mod:`.sinks` — logical artifact sinks (memory for service use, filesystem
  adapter for legacy CLI compatibility);
- :class:`.errors.CoreError` — typed error base.
"""

from .config import (
    ALGORITHMIC_TOP_LEVEL_FIELDS,
    BATCH_ONLY_TOP_LEVEL_FIELDS,
    ConfigClassification,
    CoreConfig,
    classify_config_fields,
    config_digest,
)
from .engine import StageFunctions, default_stage_functions, run_single_image
from .errors import CoreError, IdentityMaskOverflowError, IdentityMaskProjectionError
from .mask_views import (
    CANDIDATE_VIEW_DEFAULTS,
    CANDIDATE_VIEW_LIMITS,
    CandidateViewConfig,
    MaskViewResult,
    RawClipCropResult,
    build_candidate_views,
    build_raw_clip_crop,
    build_mask_views,
    default_candidate_view_configs,
    effective_candidate_view_configs,
    exact_euclidean_dilate,
)
from .routing import (
    ROUTING_PRIMARY_REASONS,
    ClipRoutingDecision,
    apply_clip_routing,
    route_clip_candidate,
)
from .ordering import mask_centroid_rc, order_final_objects, ordering_key
from .renderers import (
    MAX_IDENTITY_OBJECTS,
    YOLO_DECIMALS,
    format_yolo_line,
    render_identity_png,
    render_yolo,
)
from .raw_visualizations import (
    RAW_CANDIDATE_ID_BASE,
    RAW_CANDIDATES_PER_SHEET,
    RAW_CONTACT_SHEET_COLUMNS,
    RAW_CONTACT_SHEET_HEIGHT,
    RAW_CONTACT_SHEET_ROWS,
    RAW_CONTACT_SHEET_WIDTH,
    RAW_CONTEXT_PADDING_FRACTION,
    RAW_FIXED_ARTIFACT_NAMES,
    RAW_MASK_ALPHA,
    RAW_MAXIMUM_CONTACT_SHEETS,
    RAW_MAXIMUM_REPRESENTED_CANDIDATES,
    RAW_MAX_DIAGNOSTIC_PIXELS,
    RAW_MIN_CONTEXT_PADDING_PIXELS,
    RAW_TILE_CONTENT_HEIGHT,
    RAW_TILE_CONTENT_WIDTH,
    RAW_TILE_LABEL_HEIGHT,
    RawSam2Visualization,
    candidate_color,
    diagnostic_dimensions,
    finalize_raw_sam2_visualization,
    raw_sam2_debug_rgb_bytes,
    render_raw_sam2_visualizations,
    validate_raw_sam2_manifest,
)
from .results import (
    ObjectResult,
    PipelineResult,
    Provenance,
    SingleImageOutcome,
    StageStatus,
)
from .sinks import (
    ArtifactSink,
    ArtifactSinkError,
    ArtifactBudget,
    BoundedMemoryArtifactSink,
    FilesystemArtifactSink,
    MemoryArtifactSink,
    StoredArtifact,
)

__all__ = [
    "ALGORITHMIC_TOP_LEVEL_FIELDS",
    "BATCH_ONLY_TOP_LEVEL_FIELDS",
    "ArtifactSink",
    "ArtifactSinkError",
    "ArtifactBudget",
    "BoundedMemoryArtifactSink",
    "ConfigClassification",
    "CoreConfig",
    "CoreError",
    "CANDIDATE_VIEW_DEFAULTS",
    "CANDIDATE_VIEW_LIMITS",
    "CandidateViewConfig",
    "FilesystemArtifactSink",
    "IdentityMaskOverflowError",
    "IdentityMaskProjectionError",
    "MaskViewResult",
    "RawClipCropResult",
    "MAX_IDENTITY_OBJECTS",
    "MemoryArtifactSink",
    "ObjectResult",
    "PipelineResult",
    "Provenance",
    "SingleImageOutcome",
    "StageFunctions",
    "StageStatus",
    "StoredArtifact",
    "YOLO_DECIMALS",
    "classify_config_fields",
    "config_digest",
    "build_candidate_views",
    "build_mask_views",
    "build_raw_clip_crop",
    "default_candidate_view_configs",
    "effective_candidate_view_configs",
    "exact_euclidean_dilate",
    "ROUTING_PRIMARY_REASONS",
    "ClipRoutingDecision",
    "apply_clip_routing",
    "route_clip_candidate",
    "default_stage_functions",
    "format_yolo_line",
    "mask_centroid_rc",
    "order_final_objects",
    "ordering_key",
    "render_identity_png",
    "RAW_CANDIDATE_ID_BASE",
    "RAW_CANDIDATES_PER_SHEET",
    "RAW_CONTACT_SHEET_COLUMNS",
    "RAW_CONTACT_SHEET_HEIGHT",
    "RAW_CONTACT_SHEET_ROWS",
    "RAW_CONTACT_SHEET_WIDTH",
    "RAW_CONTEXT_PADDING_FRACTION",
    "RAW_FIXED_ARTIFACT_NAMES",
    "RAW_MASK_ALPHA",
    "RAW_MAXIMUM_CONTACT_SHEETS",
    "RAW_MAXIMUM_REPRESENTED_CANDIDATES",
    "RAW_MAX_DIAGNOSTIC_PIXELS",
    "RAW_MIN_CONTEXT_PADDING_PIXELS",
    "RAW_TILE_CONTENT_HEIGHT",
    "RAW_TILE_CONTENT_WIDTH",
    "RAW_TILE_LABEL_HEIGHT",
    "RawSam2Visualization",
    "candidate_color",
    "diagnostic_dimensions",
    "finalize_raw_sam2_visualization",
    "raw_sam2_debug_rgb_bytes",
    "render_raw_sam2_visualizations",
    "validate_raw_sam2_manifest",
    "render_yolo",
    "run_single_image",
]
