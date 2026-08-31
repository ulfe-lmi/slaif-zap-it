"""Normalized configuration boundary for the in-memory core.

The legacy CLI loads a trusted YAML file into a plain ``dict``. The core needs
a typed, normalized view of only the *algorithmic* fields, plus an explicit
classification of which top-level keys are batch/deployment concerns that the
core must ignore. The service's separate hostile-upload validator applies its
untrusted request policy before this trusted core boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from .mask_views import (
    CandidateViewConfig,
    default_candidate_view_configs,
    effective_candidate_view_configs,
)

__all__ = [
    "ALGORITHMIC_TOP_LEVEL_FIELDS",
    "BATCH_ONLY_TOP_LEVEL_FIELDS",
    "CoreConfig",
    "ConfigClassification",
    "classify_config_fields",
    "config_digest",
]


#: Top-level configuration keys consumed by the single-image core.
ALGORITHMIC_TOP_LEVEL_FIELDS = frozenset(
    {
        "alpha",
        "preprocessing",
        "mask_generator",
        "postsam2processing",
        "clip",
        "clip_routing",
        "blip3",
        "candidate_views",
        "visualization",
    }
)

#: Top-level keys that belong to batch orchestration, dataset export or output
#: deployment. The core never reads them; adapters may.
BATCH_ONLY_TOP_LEVEL_FIELDS = frozenset(
    {
        "images",
        "video",
        "export_yolo_det",
    }
)


@dataclass(frozen=True)
class ConfigClassification:
    """Partition of a raw config mapping's top-level keys."""

    algorithm_fields: Tuple[str, ...]
    batch_only_fields: Tuple[str, ...]
    unrecognized_fields: Tuple[str, ...]

    def as_dict(self) -> Dict[str, List[str]]:
        """Return JSON-serializable classification with sorted field lists."""
        return {
            "algorithm": sorted(self.algorithm_fields),
            "batch_only": sorted(self.batch_only_fields),
            "unrecognized": sorted(self.unrecognized_fields),
        }


def classify_config_fields(config: Mapping[str, Any]) -> ConfigClassification:
    """Classify the top-level keys of ``config``.

    Unrecognized keys are reported honestly; the core does not silently adopt
    them as algorithm inputs and does not treat them as errors at this layer
    (legacy configs carry deployment fields by design).
    """
    algorithm: List[str] = []
    batch_only: List[str] = []
    unrecognized: List[str] = []
    for key in config.keys():
        if key in ALGORITHMIC_TOP_LEVEL_FIELDS:
            algorithm.append(key)
        elif key in BATCH_ONLY_TOP_LEVEL_FIELDS:
            batch_only.append(key)
        else:
            unrecognized.append(key)
    return ConfigClassification(
        algorithm_fields=tuple(sorted(algorithm)),
        batch_only_fields=tuple(sorted(batch_only)),
        unrecognized_fields=tuple(sorted(unrecognized)),
    )


@dataclass(frozen=True)
class CoreConfig:
    """Normalized algorithmic configuration for one single-image run."""

    alpha: float
    roi_val: Any | None
    resize_val: Any | None
    prep_debug: bool
    clip_cfg: Mapping[str, Any] = field(default_factory=dict)
    clip_routing_cfg: Mapping[str, Any] = field(default_factory=dict)
    blip3_cfg: Mapping[str, Any] = field(default_factory=dict)
    sam2_cfg: Mapping[str, Any] = field(default_factory=dict)
    postsam2_cfg: Mapping[str, Any] = field(default_factory=dict)
    vis_cfg: Mapping[str, Any] = field(default_factory=dict)
    keep_labels: Tuple[str, ...] = ()
    post_maxsize: int = 999_999_999
    max_w: int = 999_999_999
    max_h: int = 999_999_999
    # Service-only SAM2 provenance is kept separate from constructor scalars so
    # the core can carry it without forwarding metadata to a model adapter.
    sam2_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_views: Mapping[str, CandidateViewConfig] = field(
        default_factory=default_candidate_view_configs
    )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CoreConfig":
        """Build a normalized config from a legacy parsed YAML mapping.

        Normalization matches the historical behavior of the batch pipeline:
        ``visualization.alpha`` is hoisted to the effective ``alpha`` value and
        ``preprocessing.roi: false`` behaves like no ROI.
        """
        prep = config.get("preprocessing", {}) or {}
        clip_cfg = config.get("clip", {}) or {}
        clip_routing_cfg = config.get("clip_routing", {}) or {}
        blip3_cfg = config.get("blip3", {}) or {}
        raw_sam2_cfg = config.get("mask_generator", {}) or {}
        sam2_cfg = (
            {key: value for key, value in raw_sam2_cfg.items() if not str(key).startswith("_")}
            if isinstance(raw_sam2_cfg, Mapping)
            else raw_sam2_cfg
        )
        postsam2_cfg = config.get("postsam2processing", {}) or {}
        vis_cfg = config.get("visualization", {}) or {}
        candidate_views = effective_candidate_view_configs(config.get("candidate_views"))

        post_maxsize = postsam2_cfg.get("maxsize", 999_999_999)
        max_w = postsam2_cfg.get("max_w", 999_999_999)
        max_h = postsam2_cfg.get("max_h", 999_999_999)

        labels_cfg = vis_cfg.get("labels", [])
        if isinstance(labels_cfg, str):
            keep_labels: Tuple[str, ...] = tuple(
                s.strip() for s in labels_cfg.split(",") if s.strip()
            )
        elif isinstance(labels_cfg, (list, tuple, set)):
            keep_labels = tuple(str(item).strip() for item in labels_cfg if str(item).strip())
        else:
            keep_labels = ()

        roi_val = prep.get("roi", None)
        if roi_val is False:
            roi_val = None

        return cls(
            alpha=config["alpha"],
            roi_val=roi_val,
            resize_val=prep.get("resize", None),
            prep_debug=bool(prep.get("debug", False)),
            clip_cfg=clip_cfg,
            clip_routing_cfg=clip_routing_cfg,
            blip3_cfg=blip3_cfg,
            sam2_cfg=sam2_cfg,
            postsam2_cfg=postsam2_cfg,
            vis_cfg=vis_cfg,
            keep_labels=keep_labels,
            post_maxsize=post_maxsize,
            max_w=max_w,
            max_h=max_h,
            candidate_views=candidate_views,
        )

    @property
    def candidate_views_cfg(self) -> Mapping[str, CandidateViewConfig]:
        """Compatibility alias for callers naming the section as a config."""
        return self.candidate_views

    def candidate_view_config(self, stage: str) -> CandidateViewConfig:
        """Return one normalized stage policy, including constructor defaults."""
        value = self.candidate_views.get(stage)
        if isinstance(value, CandidateViewConfig):
            return value
        return CandidateViewConfig.from_mapping(value, stage=stage)

    def debug_artifacts_requested(self) -> bool:
        """Return whether any stage requested filesystem-style debug artifacts."""
        return bool(
            self.prep_debug
            or self.sam2_cfg.get("debug", False)
            or self.postsam2_cfg.get("debug", False)
            or self.clip_cfg.get("debug", False)
            or any(
                isinstance(rule, dict) and rule.get("debug", False)
                for rule in self.blip3_cfg.values()
            )
        )


def config_digest(config: Any) -> str:
    """Stable SHA-256 digest of a normalized config (provenance hook).

    The digest is deterministic across processes for equal normalized values;
    it intentionally excludes wall-clock time or host information.
    """
    if isinstance(config, CoreConfig):
        payload = dataclasses.asdict(config)
        payload["keep_labels"] = list(payload["keep_labels"])
    else:
        payload = dict(config)
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
