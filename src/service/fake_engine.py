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
from src.core.clip_prompts import summarize_canonical_labels
from src.core.raw_visualizations import render_raw_sam2_visualizations
from src.core.results import (
    ObjectResult,
    PipelineResult,
    Provenance,
    SingleImageOutcome,
    StageStatus,
)
from src.postprocessing import filter_by_area_bbox
from src.postprocessing import filter_by_geometry
from src.core.routing import apply_clip_routing
from modules.verifier.blip3 import compose_verification_query, normalize_blip3_token

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
                "sam2_debug": bool(config.sam2_cfg.get("debug", False)),
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
        if self.object_count >= 1 and (not config.keep_labels or config.clip_routing_cfg):
            specs.append((0.05, 0.45, 0.05, 0.55))
        if self.object_count >= 2 and (not config.keep_labels or config.clip_routing_cfg):
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
            raw_rendered = render_raw_sam2_visualizations(
                image_rgb,
                candidates,
                raw_candidate_count=len(specs),
                omitted_empty_candidate_count=0,
            )
            for artifact_name, artifact_array in raw_rendered.artifacts:
                artifact_sink.store_image(artifact_name, artifact_array, fmt="png")
            raw_summary = dict(raw_rendered.summary)
            sam2_metadata["raw_visualization"] = raw_summary
            result_warnings.extend(raw_summary.get("warnings", []))
        post_filter_diagnostics: Dict[str, Any] = {}
        canonical_geometry = any(
            field in config.postsam2_cfg
            for field in (
                "min_area",
                "max_area",
                "min_width",
                "max_width",
                "min_height",
                "max_height",
                "min_aspect_ratio",
                "max_aspect_ratio",
                "allow_border_touching",
            )
        )
        if canonical_geometry:
            filtered = filter_by_geometry(
                candidates,
                config.postsam2_cfg,
                diagnostics=post_filter_diagnostics,
                collect_rejections=verbosity >= 3,
            )
        else:
            filtered = filter_by_area_bbox(
                candidates,
                config.post_maxsize,
                config.max_w,
                config.max_h,
                diagnostics=post_filter_diagnostics,
                collect_rejections=verbosity >= 3,
            )
        for filtered_index, mask_record in enumerate(filtered):
            mask_record["_filtered_index"] = filtered_index
        clip_labels = list((config.clip_cfg.get("labels", {}) or {}).keys())
        labels_cycle = clip_labels or list(class_labels) or ["object"]
        routing_diagnostics = []
        routed = filtered
        route_counts = {}
        if clip_labels:
            for index, mask_record in enumerate(filtered):
                scores = {}
                prompt_indices = {}
                for label_index, label in enumerate(labels_cycle):
                    configured_prompts = (config.clip_cfg.get("labels", {}) or {}).get(label, "")
                    prompts = (
                        [configured_prompts]
                        if isinstance(configured_prompts, str)
                        else configured_prompts
                        if isinstance(configured_prompts, (list, tuple))
                        else [""]
                    )
                    prompt_scores = [
                        round(0.80 - 0.03 * label_index - 0.001 * prompt_index - 0.01 * index, 4)
                        for prompt_index, _prompt in enumerate(prompts)
                    ]
                    score = max(prompt_scores) if prompt_scores else 0.0
                    scores[str(label)] = score
                    prompt_indices[str(label)] = (
                        min(
                            prompt_index
                            for prompt_index, prompt_score in enumerate(prompt_scores)
                            if prompt_score == score
                        )
                        if prompt_scores
                        else 0
                    )
                mask_record["clip_scores"] = scores
                winner = max(scores, key=lambda label: (scores[label], -list(scores).index(label)))
                mask_record["clip_label"] = winner
                mask_record["clip_score"] = scores[winner]
                mask_record["_clip_winning_prompt_indices"] = prompt_indices
                mask_record["_clip_winning_prompt_index"] = prompt_indices[winner]
                prompts = (config.clip_cfg.get("labels", {}) or {}).get(winner, "")
                prompts = [prompts] if isinstance(prompts, str) else prompts
                mask_record["_clip_winning_prompt"] = (
                    prompts[prompt_indices[winner]] if prompts else ""
                )
        if config.clip_routing_cfg:
            routed, routing_diagnostics, route_counts = apply_clip_routing(
                filtered, config.clip_routing_cfg
            )
            for mask_record, diagnostic in zip(filtered, routing_diagnostics):
                prompt_indices = mask_record.get("_clip_winning_prompt_indices")
                if isinstance(prompt_indices, dict):
                    diagnostic["winning_prompt_indices"] = dict(prompt_indices)
                    diagnostic["winning_prompt_index"] = mask_record.get(
                        "_clip_winning_prompt_index"
                    )
                    diagnostic["winning_prompt"] = mask_record.get("_clip_winning_prompt")
        clip_scored_count = len(filtered)
        objects: List[ObjectResult] = []
        for instance_id, mask_record in enumerate(
            sorted(
                routed, key=lambda mask: int(np.count_nonzero(mask["segmentation"])), reverse=True
            ),
            start=1,
        ):
            mask = mask_record["segmentation"]
            label = mask_record.get(
                "clip_label", labels_cycle[(instance_id - 1) % len(labels_cycle)]
            )
            if config.clip_routing_cfg and mask_record.get("clip_routing"):
                target = mask_record["clip_routing"]["chosen_target"]
                rule = config.blip3_cfg.get(target, {})
                label = str(rule.get("newcategory", label))
                mask_record["blip3_answer"] = str(rule.get("trueresult", "Yes"))
                mask_record["blip3_verification"] = {
                    "source_candidate_id": int(mask_record.get("_source_index", instance_id - 1))
                    + 1,
                    "filtered_index": int(mask_record.get("_filtered_index", instance_id - 1)),
                    "question_id": instance_id,
                    "routing_target_label": target,
                    "routing_reason": mask_record["clip_routing"].get("primary_reason"),
                    "configured_question": str(rule.get("question", "")),
                    "effective_question": compose_verification_query(str(rule.get("question", ""))),
                    "raw_answer": mask_record["blip3_answer"],
                    "normalized_answer": normalize_blip3_token(str(rule.get("trueresult", ""))),
                    "normalized_true_result": normalize_blip3_token(
                        str(rule.get("trueresult", ""))
                    ),
                    "normalized_false_result": normalize_blip3_token(
                        str(rule.get("falseresult", ""))
                    ),
                    "configured_true_result": str(rule.get("trueresult", "")),
                    "configured_false_result": str(rule.get("falseresult", "")),
                    "configured_true_label": label,
                    "configured_false_label": str(rule.get("falsecategory", "negative")),
                    "mapping_outcome": "true_match",
                    "input_artifact_name": None,
                    "input_artifact_status": "not_requested",
                    "final_label": label,
                }
            metadata = {
                "clip_label": str(label),
                "predicted_iou": mask_record.get("predicted_iou"),
                "stability_score": mask_record.get("stability_score"),
                "clip_score": mask_record.get("clip_score", round(0.80 + 0.01 * instance_id, 4)),
                "area": int(np.count_nonzero(mask)),
            }
            if "clip_scores" in mask_record:
                metadata["clip_scores"] = dict(mask_record["clip_scores"])
            if "clip_routing" in mask_record:
                metadata["clip_routing"] = dict(mask_record["clip_routing"])
            for key in ("blip3_answer", "blip3_verification"):
                if key in mask_record:
                    metadata[key] = mask_record[key]
            objects.append(
                ObjectResult(
                    instance_id=instance_id,
                    source_index=int(mask_record.get("_source_index", instance_id - 1)),
                    mask=mask,
                    metadata=metadata,
                    class_id=(instance_id - 1) % max(len(class_labels), 1),
                    class_id_source="mapping" if class_labels else "fallback",
                    filtered_index=int(mask_record.get("_filtered_index", instance_id - 1)),
                )
            )

        stage_statuses = (
            StageStatus(name="preprocessing", status="executed", detail="fake"),
            StageStatus(name="sam2", status="executed", detail=f"{len(specs)} candidates"),
            StageStatus(
                name="geometry",
                status="executed",
                detail=f"{len(candidates)} evaluated -> {len(filtered)} retained",
                duration_ms=0.0,
            ),
            StageStatus(
                name="clip",
                status="executed" if config.clip_cfg else "not_configured",
                detail=(
                    f"{len(filtered)} -> {clip_scored_count}"
                    if config.clip_cfg
                    else "no clip configuration"
                ),
            ),
            StageStatus(
                name="clip_routing",
                status="executed" if config.clip_routing_cfg else "not_configured",
                detail="fake",
                duration_ms=0.0 if config.clip_routing_cfg else None,
            ),
            StageStatus(
                name="blip3",
                status="executed" if config.blip3_cfg else "not_configured",
                detail="fake",
            ),
        )
        clip_prompt_metadata = dict(config.clip_prompt_metadata)
        if not clip_prompt_metadata and isinstance(config.clip_cfg, dict):
            labels = config.clip_cfg.get("labels", {})
            if isinstance(labels, dict):
                summary = summarize_canonical_labels(labels)
                if summary.total_prompt_count:
                    clip_prompt_metadata = summary.as_dict()

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
                **(
                    {
                        "raw_sam2_generated": len(specs),
                        "non_empty_candidates": len(specs),
                        "geometry_evaluated": int(post_filter_diagnostics.get("evaluated", 0)),
                        "after_geometry": len(filtered),
                        "geometry_rejected": int(post_filter_diagnostics.get("rejected", 0)),
                        "clip_scored": clip_scored_count,
                        **route_counts,
                        "blip3_verified": len(routed) if config.clip_routing_cfg else 0,
                        "after_final_label_filter": len(objects),
                    }
                    if canonical_geometry or config.clip_routing_cfg
                    else {}
                ),
            },
            rendered={},
            warnings=tuple(result_warnings),
            timings={
                "stage.sam2": 0.5,
                "stage.geometry": 0.0 if canonical_geometry else 0.0,
                "stage.clip_crop": 0.0,
                "stage.clip_scoring": 0.0,
                "stage.clip_routing": 0.0,
                "stage.blip3_composition": 0.0,
                "stage.blip3_verification": 0.0,
                "stage.final_filter": 0.0,
                "stage.ordering": 0.0,
            },
            provenance=Provenance(
                config_digest=config_digest(config),
                core_version="002-a-fake",
                notes=("deterministic fake engine",),
            ),
            post_filter_diagnostics=post_filter_diagnostics,
            sam2_metadata=sam2_metadata,
            clip_routing_diagnostics=tuple(routing_diagnostics),
            clip_prompt_metadata=clip_prompt_metadata,
        )
        return SingleImageOutcome(
            result=result,
            segmenter_state={"engine": SERVICE_MODEL_ID},
            clip_state=None,
            blip3_state=None,
        )
