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
from .ordering import mask_centroid_rc, order_final_objects, ordering_key
from .renderers import (
    MAX_IDENTITY_OBJECTS,
    YOLO_DECIMALS,
    format_yolo_line,
    render_identity_png,
    render_yolo,
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
    "FilesystemArtifactSink",
    "IdentityMaskOverflowError",
    "IdentityMaskProjectionError",
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
    "default_stage_functions",
    "format_yolo_line",
    "mask_centroid_rc",
    "order_final_objects",
    "ordering_key",
    "render_identity_png",
    "render_yolo",
    "run_single_image",
]
