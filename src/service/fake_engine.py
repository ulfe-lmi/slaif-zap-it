"""Injectable engine seam and the deterministic CPU fake engine.

The service depends on an engine callable with the same signature as
:func:`src.core.engine.run_single_image`. Production wiring arrives with a
later GPU objective; the HTTP contract (this objective) is exercised through
:class:`FakeEngine`, which produces deterministic, honest outcomes without
CUDA, models or network access. Fake state never leaks across calls: every
invocation builds fresh request-local objects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.config import CoreConfig, config_digest
from src.core.raw_visualizations import render_raw_sam2_visualizations
from src.core.results import (
    ObjectResult,
    PipelineResult,
    Provenance,
    SingleImageOutcome,
    StageStatus,
)
from src.postprocessing import filter_by_area_bbox

from .settings import SERVICE_MODEL_ID

__all__ = ["EngineCallable", "FakeEngine"]

EngineCallable = Callable[..., SingleImageOutcome]


def _rect_mask(
    height: int, width: int, top: float, bottom: float, left: float, right: float
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r0 = int(top * height)
    r1 = max(int(bottom * height), r0 + 1)
    c0 = int(left * width)
    c1 = max(int(right * width), c0 + 1)
    mask[r0 : min(r1, height), c0 : min(c1, width)] = True
    return mask


@dataclass
class FakeEngine:
    """Deterministic in-memory outcome factory for API contract tests.

    The fake derives two non-overlapping rectangular objects from the decoded
    image geometry (larger first), or zero objects when the effective config
    keeps no labels. Optional hooks let tests simulate latency, inference
    failures, timeouts and cancellation deterministically.
    """

    delay_seconds: float = 0.0
    fail: bool = False
    hang_seconds: float = 0.0
    object_count: int = 2
    calls: List[Dict[str, Any]] = field(default_factory=list)
    active_calls: int = 0
    max_observed_active: int = 0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def __call__(
        self,
        image_rgb: np.ndarray,
        config: CoreConfig,
        *,
        frame_id: str = "image",
        segmenter_state: Optional[dict] = None,
        clip_state: Optional[dict] = None,
        blip3_state: Optional[dict] = None,
        dryrun: bool = False,
        verbosity: int = 1,
        device: Optional[Any] = None,
        log_print_func=None,
        artifact_sink=None,
        stages=None,
        class_labels=(),
        render_visualizations=None,
        service_safe_artifact_names=False,
    ) -> SingleImageOutcome:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim < 2:
            raise ValueError("image_rgb must be a decoded numpy array")
        with self._lock:
            self.active_calls += 1
            self.max_observed_active = max(self.max_observed_active, self.active_calls)
        try:
            return self._execute(
                image_rgb,
                config,
                frame_id=frame_id,
                verbosity=verbosity,
                artifact_sink=artifact_sink,
                class_labels=class_labels,
                service_safe_artifact_names=service_safe_artifact_names,
            )
        finally:
            with self._lock:
                self.active_calls -= 1

    def _execute(
        self,
        image_rgb: np.ndarray,
        config: CoreConfig,
        *,
        frame_id: str,
        verbosity: int,
        artifact_sink=None,
        class_labels=(),
        service_safe_artifact_names=False,
    ) -> SingleImageOutcome:
        self.calls.append(
            {
                "frame_id": frame_id,
                "verbosity": verbosity,
                "class_labels": tuple(class_labels),
                "config_digest": config_digest(config),
                "shape": tuple(image_rgb.shape),
                "has_sink": artifact_sink is not None,
            }
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.hang_seconds:
            deadline = time.monotonic() + self.hang_seconds
            while time.monotonic() < deadline:  # pragma: no branch - timing loop
                time.sleep(0.001)
        if self.fail:
            raise RuntimeError("fake engine inference failure")

        height, width = image_rgb.shape[:2]
        specs: List[Tuple[float, float, float, float]] = []
        if self.object_count >= 1 and not config.keep_labels:
            specs.append((0.05, 0.45, 0.05, 0.55))
        if self.object_count >= 2 and not config.keep_labels:
            specs.append((0.50, 0.80, 0.60, 0.90))

        candidates = []
        for source_index, (top, bottom, left, right) in enumerate(specs):
            mask = _rect_mask(height, width, top, bottom, left, right)
            candidates.append(
                {
                    "segmentation": mask,
                    "area": int(np.count_nonzero(mask)),
                    "predicted_iou": round(0.90 + 0.01 * (source_index + 1), 4),
                    "stability_score": round(0.85 + 0.01 * (source_index + 1), 4),
                    "_source_index": source_index,
                }
            )
        result_warnings = ["fake engine output; not a model prediction"]
        sam2_metadata = {
            **dict(config.sam2_metadata or {}),
            "actual_candidate_count": len(specs),
            "execution_time_ms": 0.5,
        }
        if service_safe_artifact_names and verbosity >= 3 and config.sam2_cfg.get("debug", False):
            if artifact_sink is None:
                raise ValueError("SAM2 debug requires an artifact sink")
            raw_rendered = render_raw_sam2_visualizations(image_rgb, candidates)
            for artifact_name, artifact_array in raw_rendered.artifacts:
                artifact_sink.store_image(artifact_name, artifact_array, fmt="png")
            raw_summary = dict(raw_rendered.summary)
            raw_summary.update(
                {
                    "raw_candidate_count": len(specs),
                    "omitted_empty_candidate_count": 0,
                }
            )
            sam2_metadata["raw_visualization"] = raw_summary
            result_warnings.extend(raw_summary.get("warnings", []))
        post_filter_diagnostics: Dict[str, Any] = {}
        filtered = filter_by_area_bbox(
            candidates,
            config.post_maxsize,
            config.max_w,
            config.max_h,
            diagnostics=post_filter_diagnostics,
            collect_rejections=verbosity >= 3,
        )
        ordered = sorted(
            filtered,
            key=lambda mask: int(np.count_nonzero(mask["segmentation"])),
            reverse=True,
        )

        labels_cycle = list(class_labels) or ["object"]
        objects: List[ObjectResult] = []
        for instance_id, mask_record in enumerate(ordered, start=1):
            mask = mask_record["segmentation"]
            label = labels_cycle[(instance_id - 1) % len(labels_cycle)]
            metadata = {
                "clip_label": str(label),
                "predicted_iou": mask_record.get("predicted_iou"),
                "stability_score": mask_record.get("stability_score"),
                "clip_score": round(0.80 + 0.01 * instance_id, 4),
                "area": int(np.count_nonzero(mask)),
            }
            objects.append(
                ObjectResult(
                    instance_id=instance_id,
                    source_index=instance_id,
                    mask=mask,
                    metadata=metadata,
                    class_id=(instance_id - 1) % max(len(class_labels), 1),
                    class_id_source="mapping" if class_labels else "fallback",
                )
            )

        stage_statuses = (
            StageStatus(name="preprocessing", status="executed", detail="fake"),
            StageStatus(name="sam2", status="executed", detail=f"{len(specs)} candidates"),
            StageStatus(
                name="clip",
                status="executed" if config.clip_cfg else "not_configured",
                detail="fake",
            ),
            StageStatus(
                name="blip3",
                status="executed" if config.blip3_cfg else "not_configured",
                detail="fake",
            ),
        )
        result = PipelineResult(
            image_height=int(height),
            image_width=int(width),
            roi_box=(0, 0, int(width), int(height)),
            resize_info={"mode": "native"},
            objects=tuple(objects),
            stage_statuses=stage_statuses,
            candidate_counts={
                "sam2_candidates": len(specs),
                "after_area_bbox": len(filtered),
                "after_clip": len(filtered),
                "final": len(objects),
            },
            rendered={},
            warnings=tuple(result_warnings),
            timings={"stage.sam2": 0.5},
            provenance=Provenance(
                config_digest=config_digest(config),
                core_version="002-a-fake",
                notes=("deterministic fake engine",),
            ),
            post_filter_diagnostics=post_filter_diagnostics,
            sam2_metadata=sam2_metadata,
        )
        return SingleImageOutcome(
            result=result,
            segmenter_state={"engine": SERVICE_MODEL_ID},
            clip_state=None,
            blip3_state=None,
        )
