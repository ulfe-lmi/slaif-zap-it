"""Typed result contracts for the in-memory single-image core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "ObjectResult",
    "StageStatus",
    "Provenance",
    "PipelineResult",
    "SingleImageOutcome",
]


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rr, cc = np.nonzero(mask)
    y_min, y_max = int(rr.min()), int(rr.max())
    x_min, x_max = int(cc.min()), int(cc.max())
    return (x_min, y_min, x_max, y_max)


def _normalized_bbox(
    bbox_xyxy: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    cx = (x_min + x_max) / 2.0 / image_width
    cy = (y_min + y_max) / 2.0 / image_height
    bw = (x_max - x_min + 1) / image_width
    bh = (y_max - y_min + 1) / image_height
    return (cx, cy, bw, bh)


@dataclass(frozen=True)
class ObjectResult:
    """One final object in original-image coordinates.

    ``mask`` keeps the complete source segmentation (including disconnected
    components and overlap with other objects) so overlap truth is never lost;
    renderers project it deliberately. ``metadata`` carries the scalar fields
    produced by the stages for this object.
    """

    instance_id: int
    source_index: int
    mask: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    class_id: Optional[int] = None
    class_id_source: Optional[str] = None
    filtered_index: Optional[int] = None

    @property
    def source_candidate_id(self) -> int:
        """Public one-based identity derived from the internal source index."""
        return self.source_index + 1

    @property
    def label(self) -> Optional[str]:
        return self.metadata.get("clip_label")

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        return _bbox_from_mask(self.mask)

    def normalized_bbox(
        self, image_width: int, image_height: int
    ) -> Tuple[float, float, float, float]:
        return _normalized_bbox(self.bbox_xyxy, image_width, image_height)

    @property
    def centroid_rc(self) -> Tuple[float, float]:
        rr, cc = np.nonzero(self.mask)
        return (float(rr.mean()), float(cc.mean()))

    @property
    def predicted_iou(self) -> Optional[float]:
        value = self.metadata.get("predicted_iou")
        return None if value is None else float(value)

    @property
    def stability_score(self) -> Optional[float]:
        value = self.metadata.get("stability_score")
        return None if value is None else float(value)

    @property
    def clip_score(self) -> Optional[float]:
        value = self.metadata.get("clip_score")
        return None if value is None else float(value)

    @property
    def blip3_answer(self) -> Optional[str]:
        value = self.metadata.get("blip3_answer")
        return None if value is None else str(value)

    def geometry(self) -> Optional[Mapping[str, Any]]:
        """Geometry hook: the canonical path never executes the stage."""
        return self.metadata.get("geometry")

    def serialized_metadata(self) -> Dict[str, Any]:
        """Deterministic JSON-friendly view of ``metadata``.

        NumPy arrays are skipped; NumPy scalars become builtins; insertion
        order follows the underlying metadata dict so repeated runs of the
        same inputs serialize byte-identically.
        """
        serialized: Dict[str, Any] = {}
        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                continue
            if isinstance(value, (np.integer,)):
                serialized[key] = int(value)
            elif isinstance(value, (np.floating,)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        serialized.setdefault("source_candidate_id", self.source_candidate_id)
        if self.filtered_index is not None:
            serialized.setdefault("filtered_index", self.filtered_index)
        return serialized


@dataclass(frozen=True)
class StageStatus:
    """Bounded status record for one pipeline stage."""

    name: str
    status: str  # "executed" | "skipped" | "not_configured"
    detail: str = ""
    duration_ms: Optional[float] = None
    warnings: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Provenance:
    """Deterministic provenance hooks attached to a result."""

    config_digest: str
    core_version: str = "001-a"
    notes: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_digest": self.config_digest,
            "core_version": self.core_version,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PipelineResult:
    """Typed outcome of one single-image core run."""

    image_height: int
    image_width: int
    roi_box: Tuple[int, int, int, int]
    resize_info: Mapping[str, Any]
    objects: Tuple[ObjectResult, ...]
    stage_statuses: Tuple[StageStatus, ...]
    candidate_counts: Mapping[str, int]
    rendered: Mapping[str, np.ndarray]
    warnings: Tuple[str, ...]
    timings: Mapping[str, float]
    provenance: Provenance
    post_filter_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    sam2_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_view_inputs: Tuple[Mapping[str, Any], ...] = ()
    blip3_candidate_views: Tuple[Mapping[str, Any], ...] = ()
    clip_routing_diagnostics: Tuple[Mapping[str, Any], ...] = ()

    def object_by_id(self, instance_id: int) -> ObjectResult:
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        raise KeyError(f"no object with instance id {instance_id}")

    def stage_status(self, name: str) -> Optional[StageStatus]:
        for status in self.stage_statuses:
            if status.name == name:
                return status
        return None

    def serialized_records(self) -> List[Dict[str, Any]]:
        """Legacy-compatible per-object serialized metadata records."""
        return [obj.serialized_metadata() for obj in self.objects]


@dataclass(frozen=True)
class SingleImageOutcome:
    """Core run result plus the reusable model states to thread forward."""

    result: PipelineResult
    segmenter_state: Optional[dict]
    clip_state: Optional[dict]
    blip3_state: Optional[dict]
