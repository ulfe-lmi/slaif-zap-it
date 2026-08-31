"""In-memory single-image ZAP-IT engine.

The engine consumes an already-decoded RGB image plus a normalized
:class:`~src.core.config.CoreConfig`, executes the canonical stage chain
(ROI/resize -> SAM2 -> post-filtering -> optional CLIP/BLIP3 -> label filter)
and returns a typed :class:`~src.core.results.PipelineResult` together with
the reusable model states. It performs no filesystem access of its own;
configured debug artifacts are routed through an
:class:`~src.core.sinks.ArtifactSink`.

Stage callables are injectable (:class:`StageFunctions`) so tests can drive
the full orchestration with fakes and the legacy batch adapter can forward its
module-level names for backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from contextlib import nullcontext
from contextlib import AbstractContextManager
from inspect import Parameter, signature
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from modules.visualizer import generate_visualizations as _generate_visualizations
from .config import CoreConfig, config_digest
from .clip_prompts import summarize_canonical_labels
from .errors import CoreError
from .mask_views import build_mask_views, build_raw_clip_crop
from .ordering import order_final_objects
from .routing import apply_clip_routing
from ..postprocessing import filter_by_area_bbox as _canonical_filter_by_area_bbox
from .raw_visualizations import render_raw_sam2_visualizations
from .results import ObjectResult, PipelineResult, Provenance, SingleImageOutcome, StageStatus
from .sinks import ArtifactSink

__all__ = ["StageFunctions", "default_stage_functions", "run_single_image"]


@dataclass(frozen=True)
class StageFunctions:
    """Injectable stage boundaries around the reusable algorithm modules."""

    apply_roi: Callable[..., Any]
    resize_image: Callable[..., Any]
    run_sam2: Callable[..., Any]
    filter_by_area_bbox: Callable[..., Any]
    run_clip: Callable[..., Any]
    run_blip3: Callable[..., Any]
    generate_visualizations: Callable[..., Any] = _generate_visualizations


def default_stage_functions() -> StageFunctions:
    """Build the production stage set from the algorithm modules."""
    from modules.classifier import run_clip
    from modules.input.images import apply_roi, resize_image
    from modules.segmenter import run_sam2
    from modules.verifier import run_blip3
    from src.postprocessing import filter_by_area_bbox

    return StageFunctions(
        apply_roi=apply_roi,
        resize_image=resize_image,
        run_sam2=run_sam2,
        filter_by_area_bbox=filter_by_area_bbox,
        run_clip=run_clip,
        run_blip3=run_blip3,
    )


def _accepts_keyword(callable_obj: Callable[..., Any], name: str) -> bool:
    """Keep injected legacy stage callables compatible with new diagnostics."""
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _resolve_device(preferred: Optional[Any] = None) -> Any:
    """Resolve the torch device exactly like the legacy batch path."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional in CPU environments
        return preferred if preferred is not None else "cpu"
    if preferred is not None:
        return preferred
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _remap_mask_to_original(
    segmentation: np.ndarray,
    roi_box: Tuple[int, int, int, int],
    resized_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Project a resized-space mask back to original-image coordinates.

    Historical note (objective 001-a): the previous implementation iterated
    over source pixels and applied ``int(rpos * scaleY)`` forward nearest
    neighbor mapping. With downscaled inference images this leaves global ROI
    rows unreachable (e.g. ROI height 11 resized to 5 rows reaches at most
    ``y + int(4 * 11/5) = y + 8``), silently shrinking masks, areas and bboxes
    near ROI edges. This implementation uses the exact inverse mapping instead:
    every destination pixel samples its nearest resized pixel, guaranteeing
    full coverage, monotonic coordinates and in-bounds writes.
    """
    x, y, x2, y2 = roi_box
    h_res, w_res = resized_shape
    h_orig, w_orig = original_shape
    roi_h = max(int(y2 - y), 0)
    roi_w = max(int(x2 - x), 0)

    global_mask = np.zeros((h_orig, w_orig), dtype=bool)
    if roi_h == 0 or roi_w == 0:
        return global_mask

    src_rows = np.floor(np.arange(roi_h, dtype=np.float64) * (h_res / float(roi_h)))
    src_cols = np.floor(np.arange(roi_w, dtype=np.float64) * (w_res / float(roi_w)))
    src_rows = np.clip(src_rows.astype(np.int64), 0, h_res - 1)
    src_cols = np.clip(src_cols.astype(np.int64), 0, w_res - 1)
    global_mask[y : y + roi_h, x : x + roi_w] = segmentation[np.ix_(src_rows, src_cols)]
    return global_mask


def _build_class_mapping(class_labels: Sequence[str]) -> Mapping[str, int]:
    mapping: dict[str, int] = {}
    for index, label in enumerate(class_labels):
        mapping.setdefault(str(label).strip(), index)
    return mapping


def _candidate_view_debug_capacity(
    image_rgb: np.ndarray,
    masks: Sequence[Mapping[str, Any]],
    config: CoreConfig,
    *,
    stage: str,
) -> tuple[int, int, list[int]]:
    """Count the exact candidate-view artifacts for one model seam.

    CLIP admission runs before CLIP values exist, so it only considers CLIP
    debug artifacts.  BLIP3 admission runs after CLIP and can therefore use
    the actual labels and scores to count applicable debug rules.
    """
    if stage not in {"clip", "blip3"}:
        raise ValueError(f"unsupported candidate-view admission stage: {stage}")

    clip_cfg = config.clip_cfg
    clip_debug = (
        stage == "clip"
        and clip_cfg.get("debug") is True
        and bool(
            (
                isinstance(clip_cfg.get("labels"), Mapping)
                and any(isinstance(value, str) for value in clip_cfg["labels"].values())
            )
            or any(
                type(key) is str and key.lower().startswith("label ") and isinstance(value, str)
                for key, value in clip_cfg.items()
            )
        )
    )
    blip_rules = config.blip3_cfg if isinstance(config.blip3_cfg, Mapping) else {}
    blip_debug_rules: list[tuple[float | None, str | None, Mapping[str, Any]]] = []
    if stage == "blip3":
        for rule_name, rule in blip_rules.items():
            if not isinstance(rule_name, str) or not isinstance(rule, Mapping):
                continue
            if rule_name.startswith("any,"):
                try:
                    threshold = float(rule_name.split(",", 1)[1])
                except ValueError:
                    continue
                blip_debug_rules.append((threshold, None, rule))
            else:
                blip_debug_rules.append((None, rule_name, rule))

    artifact_count = 0
    raw_bytes = 0
    artifact_sizes: list[int] = []
    for ordinal, mask in enumerate(masks):
        source_index = mask.get("_source_index")
        source_id = int(source_index) + 1 if type(source_index) is int else ordinal + 1
        if clip_debug:
            clip_view_config = config.candidate_view_config("clip")
            if clip_view_config.mode == "raw_bbox_crop":
                view = build_raw_clip_crop(
                    image_rgb,
                    mask["segmentation"],
                    source_id,
                    clip_view_config,
                    filtered_index=int(mask.get("_filtered_index", ordinal)),
                    debug=True,
                )
                size = int(view.rgb.nbytes)
            else:
                view = build_mask_views(
                    image_rgb,
                    mask["segmentation"],
                    source_id,
                    clip_view_config,
                    stage="clip",
                )
                size = int(view.context_rgb.nbytes)
            artifact_count += 1
            raw_bytes += size
            artifact_sizes.append(size)

        if blip_debug_rules:
            score = float(mask.get("clip_score", 0.0))
            label = mask.get("clip_label")
            debug_questions = 0
            for threshold, rule_label, rule in blip_debug_rules:
                if config.clip_routing_cfg:
                    applies = bool(mask.get("_route_to_blip3")) and rule_label == mask.get(
                        "clip_routing", {}
                    ).get("chosen_target")
                else:
                    applies = score <= threshold if threshold is not None else label == rule_label
                if applies and rule.get("debug") is True:
                    debug_questions += 1
            if debug_questions:
                from modules.verifier.blip3 import (
                    Blip3CandidateViewRejected,
                    single_blip3_view_model_input_nbytes,
                )

                try:
                    model_size = single_blip3_view_model_input_nbytes(
                        image_rgb.shape,
                        mask["segmentation"],
                        source_id,
                        config.candidate_view_config("blip3"),
                    )
                except Blip3CandidateViewRejected:
                    # Candidate-local containment rejection produces no debug
                    # artifact and must not consume the shared budget.
                    model_size = None
                if model_size is not None:
                    artifact_count += debug_questions
                    raw_bytes += debug_questions * model_size
                    artifact_sizes.extend([model_size] * debug_questions)
    return artifact_count, raw_bytes, artifact_sizes


def run_single_image(
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
    log_print_func: Optional[Callable[[str, int, int], None]] = None,
    artifact_sink: Optional[ArtifactSink] = None,
    stages: Optional[StageFunctions] = None,
    class_labels: Sequence[str] = (),
    render_visualizations: Optional[bool] = None,
    service_safe_artifact_names: bool = False,
    blip3_stage_context: Optional[Callable[[], AbstractContextManager[Any]]] = None,
) -> SingleImageOutcome:
    """Execute the single-image pipeline fully in memory.

    Returns a typed result plus the updated reusable model states. Request
    state (masks, answers, artifacts, warnings) is created fresh per call;
    only the caller-provided model holder state dicts are threaded through.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim < 2:
        raise ValueError("image_rgb must be a decoded numpy array with at least 2 dimensions")
    if stages is None:
        stages = default_stage_functions()
    log = log_print_func or (lambda *a, **k: None)

    warnings_out: List[str] = []
    timings: dict[str, float] = {}

    def timed(name: str, func):
        start = time.perf_counter()
        try:
            return func()
        finally:
            timings[name] = (time.perf_counter() - start) * 1000.0

    if segmenter_state is None:
        segmenter_state = {}
    if clip_state is None and config.clip_cfg:
        clip_state = {}
    if blip3_state is None and config.blip3_cfg:
        blip3_state = {}

    h_orig, w_orig = image_rgb.shape[:2]
    log(f" => Original shape = {w_orig}x{h_orig}", 1, verbosity)

    def _require_sink(debug_owner: str) -> ArtifactSink:
        if artifact_sink is None:
            raise CoreError(
                f"{debug_owner} debug artifacts are configured but no artifact sink "
                "was provided; pass a sink or disable the debug flags"
            )
        return artifact_sink

    # -- preprocessing -------------------------------------------------------
    partial_np, (x, y, x2, y2) = timed(
        "preprocess.roi", lambda: stages.apply_roi(image_rgb, config.roi_val)
    )
    if config.roi_val:
        log(
            f" => ROI=({config.roi_val}) => partial shape={partial_np.shape[1]}x{partial_np.shape[0]}",
            1,
            verbosity,
        )

    if config.prep_debug and config.roi_val:
        sink = _require_sink("preprocessing.debug")
        roi_name = (
            "preprocessing-roi.png" if service_safe_artifact_names else f"{frame_id}-roi01.jpg"
        )
        sink.store_image(roi_name, partial_np, fmt="png" if service_safe_artifact_names else "jpeg")
        log(f" => captured ROI debug => {roi_name}", 1, verbosity)

    resized_np, resize_info = timed(
        "preprocess.resize", lambda: stages.resize_image(partial_np, config.resize_val)
    )
    if resize_info["mode"] == "native":
        log(" => Single pass @native", 1, verbosity)
    else:
        new_w, new_h = resize_info["size"]
        log(
            f" => {resize_info['mode']} => {new_w}x{new_h} (factor={resize_info['factor']:.2f})",
            1,
            verbosity,
        )

    # -- segmentation --------------------------------------------------------
    segmenter_params = {
        "alpha": config.alpha,
        "dryrun": dryrun,
        "mask_generator_config": dict(config.sam2_cfg),
    }
    if "mask_generator" in segmenter_state:
        segmenter_params["mask_generator"] = segmenter_state["mask_generator"]

    def _run_segmenter():
        return stages.run_sam2(
            segmenter_state,
            segmenter_params,
            resized_np,
            verbosity=verbosity,
            log_print_func=log,
        )

    staged_segmenter_state, partial_masks, _sam2_meta = timed("stage.sam2", _run_segmenter)
    if staged_segmenter_state is not None:
        segmenter_state = staged_segmenter_state

    candidate_counts: dict[str, int] = {}
    candidate_view_inputs: list[Mapping[str, Any]] = []
    blip3_candidate_views: list[Mapping[str, Any]] = []

    # ``partial_masks`` is the raw automatic-generator result.  Keep that
    # count separate from the historical L3 candidate count, which counts only
    # non-empty masks after remapping to original-image coordinates.
    sam2_metadata = dict(config.sam2_metadata or {})
    raw_candidate_count = len(partial_masks)
    if isinstance(_sam2_meta, Mapping):
        raw_candidate_count = int(_sam2_meta.get("num_masks", raw_candidate_count))
    sam2_metadata["actual_candidate_count"] = raw_candidate_count
    sam2_metadata["execution_time_ms"] = round(timings.get("stage.sam2", 0.0), 3)
    sam2_metadata.setdefault("resource_warnings", [])

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
    all_masks_pre: List[dict] = []
    non_empty_masks: List[dict] = []
    for candidate_index, mask in enumerate(partial_masks):
        seg_rs = mask["segmentation"]
        if not np.any(seg_rs) and not canonical_geometry:
            continue
        seg_global = _remap_mask_to_original(seg_rs, (x, y, x2, y2), seg_rs.shape, (h_orig, w_orig))
        record = {
            "segmentation": seg_global,
            "area": int(seg_global.sum()),
            "predicted_iou": mask.get("predicted_iou", None),
            "stability_score": mask.get("stability_score", None),
            "_source_index": candidate_index,
        }
        all_masks_pre.append(record)
        if np.any(seg_global):
            non_empty_masks.append(record)
    candidate_counts["sam2_candidates"] = len(non_empty_masks)
    if canonical_geometry or config.clip_routing_cfg:
        candidate_counts.update(
            {
                "raw_sam2_generated": raw_candidate_count,
                "non_empty_candidates": len(non_empty_masks),
            }
        )

    if config.sam2_cfg.get("debug", False):
        if service_safe_artifact_names:
            if verbosity >= 3:
                sink = _require_sink("mask_generator.debug")
                omitted_empty = raw_candidate_count - len(non_empty_masks)
                if omitted_empty < 0:
                    raise CoreError("SAM2 raw candidate accounting is inconsistent")
                raw_rendered = render_raw_sam2_visualizations(
                    image_rgb,
                    non_empty_masks,
                    raw_candidate_count=raw_candidate_count,
                    omitted_empty_candidate_count=omitted_empty,
                )
                for artifact_name, artifact_array in raw_rendered.artifacts:
                    sink.store_image(artifact_name, artifact_array, fmt="png")
                raw_summary = dict(raw_rendered.summary)
                warnings_out.extend(raw_summary.get("warnings", []))
                sam2_metadata["raw_visualization"] = raw_summary
        else:
            sink = _require_sink("mask_generator.debug")
            log("[mask_generator debug] => capturing raw SAM2 patches...", 1, verbosity)
            for idx, mm in enumerate(non_empty_masks):
                seg = mm["segmentation"]
                rr, cc = np.nonzero(seg)
                if len(rr) == 0:
                    continue
                patch = image_rgb[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1, :]
                sink.store_image(f"{frame_id}_sam2-patch{idx:04d}.jpg", patch)
                log(f"  => captured {frame_id}_sam2-patch{idx:04d}.jpg", 2, verbosity)

    post_filter_diagnostics: dict[str, Any] = {}

    def _run_post_filter():
        filter_func = stages.filter_by_area_bbox
        filter_kwargs: dict[str, Any] = {
            "verbosity": verbosity,
            "log_print_func": log,
        }
        if _accepts_keyword(filter_func, "diagnostics"):
            filter_kwargs["diagnostics"] = post_filter_diagnostics
        if _accepts_keyword(filter_func, "collect_rejections"):
            filter_kwargs["collect_rejections"] = verbosity >= 3
        if canonical_geometry and _accepts_keyword(filter_func, "geometry_config"):
            filter_kwargs["geometry_config"] = config.postsam2_cfg
            filtered = filter_func(all_masks_pre, **filter_kwargs)
        else:
            filtered = filter_func(
                all_masks_pre,
                config.post_maxsize,
                config.max_w,
                config.max_h,
                **filter_kwargs,
            )
        if not post_filter_diagnostics:
            # Old injected fakes may not know about the sidecar. Derive an honest
            # standard-filter view without changing their legacy list result.
            if canonical_geometry:
                from ..postprocessing import filter_by_geometry

                filter_by_geometry(
                    all_masks_pre,
                    config.postsam2_cfg,
                    diagnostics=post_filter_diagnostics,
                    collect_rejections=verbosity >= 3,
                )
            else:
                _canonical_filter_by_area_bbox(
                    all_masks_pre,
                    config.post_maxsize,
                    config.max_w,
                    config.max_h,
                    diagnostics=post_filter_diagnostics,
                    collect_rejections=verbosity >= 3,
                )
        return filtered

    filtered_for_clip = timed("stage.postsam2_filter", _run_post_filter)
    timings["stage.geometry"] = timings["stage.postsam2_filter"]
    # This index is assigned once, immediately after the SAM2 area/bbox filter,
    # and is never renumbered when a later stage rejects a candidate.
    for filtered_index, mask in enumerate(filtered_for_clip):
        mask["_filtered_index"] = filtered_index
    candidate_counts["after_area_bbox"] = len(filtered_for_clip)
    if canonical_geometry or config.clip_routing_cfg:
        candidate_counts.update(
            {
                "geometry_evaluated": int(post_filter_diagnostics.get("evaluated", 0)),
                "after_geometry": len(filtered_for_clip),
                "geometry_rejected": int(
                    post_filter_diagnostics.get(
                        "rejected",
                        post_filter_diagnostics.get("evaluated", 0)
                        - post_filter_diagnostics.get("retained", 0),
                    )
                ),
            }
        )

    # -- classification ------------------------------------------------------
    resolved_device = device if device is not None else _resolve_device(None)
    if config.clip_cfg:
        log(f"[clip] => classifying {len(filtered_for_clip)} bounding boxes...", 1, verbosity)
        clip_params = {
            "config": config.clip_cfg,
            "device": resolved_device,
            "masks": filtered_for_clip,
            "fname_stem": frame_id,
            "dryrun": dryrun,
            "candidate_view_config": config.candidate_view_config("clip"),
            "candidate_view_inputs": candidate_view_inputs,
            "canonical_labels": bool(config.clip_routing_cfg)
            or config.candidate_view_config("clip").mode == "raw_bbox_crop",
        }
        if config.clip_cfg.get("debug", False):
            clip_params["artifact_sink"] = _require_sink("clip.debug")
        clip_params["safe_artifact_names"] = service_safe_artifact_names

        def _run_clip():
            return stages.run_clip(
                clip_state,
                clip_params,
                image_rgb,
                verbosity=verbosity,
                log_print_func=log,
            )

        staged_clip_state, masked_after_clip, _clip_meta = timed("stage.clip", _run_clip)
        if staged_clip_state is not None:
            clip_state = staged_clip_state
        if isinstance(_clip_meta, Mapping):
            for key in ("crop_time_ms", "scoring_time_ms"):
                if key in _clip_meta:
                    timings[f"stage.clip_{key.removesuffix('_time_ms')}"] = max(
                        0.0, float(_clip_meta[key])
                    )
        log("[clip] => classification done, now final label filter...", 1, verbosity)
        # Capture this before routing or BLIP3 can replace the working list.
        # It is the number of candidates returned from the actual CLIP stage.
        clip_scored_count = len(masked_after_clip)
    else:
        masked_after_clip = filtered_for_clip
        clip_scored_count = len(masked_after_clip)
    candidate_counts["after_clip"] = clip_scored_count
    if canonical_geometry or config.clip_routing_cfg:
        candidate_counts["clip_scored"] = clip_scored_count

    clip_only_masks = [dict(m) for m in masked_after_clip]
    clip_routing_diagnostics: list[Mapping[str, Any]] = []
    routed_for_blip3 = masked_after_clip
    if config.clip_routing_cfg:
        routed_for_blip3, clip_routing_diagnostics, route_counts = timed(
            "stage.clip_routing",
            lambda: apply_clip_routing(masked_after_clip, config.clip_routing_cfg),
        )
        for mask, diagnostic in zip(masked_after_clip, clip_routing_diagnostics):
            prompt_indices = mask.get("_clip_winning_prompt_indices")
            if isinstance(prompt_indices, Mapping):
                diagnostic["winning_prompt_indices"] = {
                    str(label): int(index) for label, index in prompt_indices.items()
                }
                diagnostic["winning_prompt_index"] = (
                    int(mask["_clip_winning_prompt_index"])
                    if mask.get("_clip_winning_prompt_index") is not None
                    else None
                )
                diagnostic["winning_prompt"] = (
                    str(mask["_clip_winning_prompt"])
                    if mask.get("_clip_winning_prompt") is not None
                    else None
                )
        candidate_counts.update(route_counts)
    if config.clip_cfg:
        timings.setdefault("stage.clip_crop", 0.0)
        timings.setdefault("stage.clip_scoring", timings.get("stage.clip", 0.0))
    if config.clip_routing_cfg:
        timings.setdefault("stage.clip_routing", 0.0)

    if config.blip3_cfg:
        log("[blip3] => verifying masks...", 1, verbosity)
        blip3_params = {
            "config": config.blip3_cfg,
            "device": resolved_device,
            "masks": routed_for_blip3,
            "fname_stem": frame_id,
            "dryrun": dryrun,
            "service_safe_artifact_names": service_safe_artifact_names,
            "candidate_view_config": config.candidate_view_config("blip3"),
            "candidate_view_inputs": candidate_view_inputs,
            "candidate_view_records": blip3_candidate_views,
        }
        if any(
            isinstance(rule, Mapping) and rule.get("debug", False)
            for rule in config.blip3_cfg.values()
        ):
            blip3_params["artifact_sink"] = _require_sink("blip3 rule debug")
        if isinstance(blip3_state, dict):
            # These bounds are fixed by the service registry, never by upload.
            if "max_questions" in blip3_state:
                blip3_params["max_questions"] = int(blip3_state["max_questions"])
            if "max_new_tokens" in blip3_state:
                blip3_params["max_new_tokens"] = int(blip3_state["max_new_tokens"])

        def _run_blip3():
            stage_context = blip3_stage_context() if blip3_stage_context else nullcontext()
            with stage_context:
                return stages.run_blip3(
                    blip3_state,
                    blip3_params,
                    image_rgb,
                    verbosity=verbosity,
                    log_print_func=log,
                )

        staged_blip3_state, masked_after_clip, _blip3_meta = timed("stage.blip3", _run_blip3)
        if staged_blip3_state is not None:
            blip3_state = staged_blip3_state
        if canonical_geometry or config.clip_routing_cfg:
            candidate_counts["blip3_verified"] = int(
                (_blip3_meta or {}).get("verified_count", len(masked_after_clip))
            )
            if isinstance(_blip3_meta, Mapping):
                for key in ("composition_time_ms", "verification_time_ms"):
                    if key in _blip3_meta:
                        timings[f"stage.blip3_{key.removesuffix('_time_ms')}"] = max(
                            0.0, float(_blip3_meta[key])
                        )
            timings.setdefault("stage.blip3_composition", 0.0)
            timings.setdefault("stage.blip3_verification", timings.get("stage.blip3", 0.0))

    # -- final label filter --------------------------------------------------
    pre_filter_count = len(masked_after_clip)

    def _run_final_filter():
        return [
            mm
            for mm in masked_after_clip
            if not (canonical_geometry or config.clip_routing_cfg) or not mm.get("_blip3_rejected")
            if not (config.keep_labels and mm.get("clip_label", None) not in config.keep_labels)
        ]

    final_masks = timed("stage.final_filter", _run_final_filter)
    candidate_counts["final"] = len(final_masks)
    if canonical_geometry or config.clip_routing_cfg:
        candidate_counts["after_final_label_filter"] = len(final_masks)

    if config.postsam2_cfg.get("debug", False):
        sink = _require_sink("postsam2processing.debug")
        log(
            "[postsam2processing debug] => capturing final patches after classification...",
            1,
            verbosity,
        )
        for idx, mm in enumerate(final_masks):
            seg = mm["segmentation"]
            rr, cc = np.nonzero(seg)
            if len(rr) == 0:
                continue
            patch = image_rgb[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1, :]
            patch_name = (
                f"postsam2-filtered-patch-{idx + 1:04d}.png"
                if service_safe_artifact_names
                else f"{frame_id}_sam2-filtered-patch{idx:04d}.jpg"
            )
            sink.store_image(
                patch_name,
                patch,
                fmt="png" if service_safe_artifact_names else "jpeg",
            )
            log(
                f"  => captured final patch => {patch_name}",
                2,
                verbosity,
            )

    # -- deterministic identity/ordering -------------------------------------
    ordered_masks = timed("stage.ordering", lambda: order_final_objects(final_masks))
    objects: List[ObjectResult] = []
    class_mapping = _build_class_mapping(class_labels)
    for instance_id, mask in enumerate(ordered_masks, start=1):
        metadata = {
            key: value
            for key, value in mask.items()
            if key not in {"segmentation", "_source_index", "_filtered_index"}
            and not str(key).startswith("_")
        }
        object_warnings: List[str] = []
        label = metadata.get("clip_label")
        if label is not None and str(label) not in class_mapping:
            object_warnings.append(
                f"label {label!r} is absent from the effective class mapping; "
                "falling back to YOLO class id 0"
            )
        mapped = class_mapping.get(str(label)) if label is not None else None
        objects.append(
            ObjectResult(
                instance_id=instance_id,
                source_index=int(mask["_source_index"]),
                mask=mask["segmentation"],
                metadata=metadata,
                warnings=tuple(object_warnings),
                class_id=mapped if mapped is not None else 0,
                class_id_source="mapping" if mapped is not None else "fallback",
                filtered_index=(
                    int(mask["_filtered_index"]) if "_filtered_index" in mask else None
                ),
            )
        )

    # -- visualization (arrays only; writers stay outside the core) ----------
    # ``None`` preserves the legacy CLI behavior.  The service passes an
    # explicit value so L0-L2 cannot spend work merely enriching a response.
    # Labelled streams are the only visualization that receives final objects;
    # legacy mask-only streams continue to see their original stage inputs.
    should_render_visualizations = (
        bool(render_visualizations) if render_visualizations is not None else True
    )
    stage_masks = {
        "sam2": all_masks_pre,
        "clip": clip_only_masks,
        "blip3": final_masks,
    }
    labelled_requested = any(
        isinstance(entries, list)
        and any(
            isinstance(entry, Mapping)
            and str(entry.get("renderer", "")).lower() == "annotated-labelled"
            for entry in entries
        )
        for stage_name in ("sam2", "clip", "blip3")
        for entries in (config.vis_cfg.get(stage_name, []),)
    )

    def _render_visualizations() -> Any:
        kwargs = {
            "default_alpha": config.alpha,
            "verbosity": verbosity,
            "log_print_func": log,
        }
        if labelled_requested:
            kwargs["final_objects"] = tuple(objects)
        return stages.generate_visualizations(image_rgb, stage_masks, config.vis_cfg, **kwargs)

    rendered = (
        timed("stage.visualization", _render_visualizations) or {}
        if should_render_visualizations
        else {}
    )

    stage_statuses = (
        StageStatus(
            name="preprocessing",
            status="executed",
            detail=f"roi={config.roi_val!r} resize={resize_info.get('mode')}",
            duration_ms=timings.get("preprocess.roi", 0.0) + timings.get("preprocess.resize", 0.0),
        ),
        StageStatus(
            name="sam2",
            status="executed",
            detail=f"{len(non_empty_masks)} non-empty of {raw_candidate_count} generated",
            duration_ms=timings.get("stage.sam2"),
        ),
        StageStatus(
            name="postsam2_filter",
            status="executed",
            detail=f"{len(all_masks_pre)} -> {len(filtered_for_clip)}",
            duration_ms=timings.get("stage.postsam2_filter"),
        ),
        StageStatus(
            name="geometry",
            status="executed",
            detail=f"{len(all_masks_pre)} evaluated -> {len(filtered_for_clip)} retained",
            duration_ms=timings.get("stage.geometry"),
        ),
        StageStatus(
            name="clip",
            status="executed" if config.clip_cfg else "not_configured",
            detail=(
                f"{len(filtered_for_clip)} -> {clip_scored_count}"
                if config.clip_cfg
                else "no clip configuration"
            ),
            duration_ms=timings.get("stage.clip") if config.clip_cfg else None,
        ),
        StageStatus(
            name="clip_routing",
            status="executed" if config.clip_routing_cfg else "not_configured",
            detail=(
                f"{candidate_counts.get('initially_routed', 0)} initially -> "
                f"{candidate_counts.get('routed_after_cap', 0)} routed"
                if config.clip_routing_cfg
                else "no routing configuration"
            ),
            duration_ms=timings.get("stage.clip_routing") if config.clip_routing_cfg else None,
        ),
        StageStatus(
            name="blip3",
            status="executed" if config.blip3_cfg else "not_configured",
            detail="verification applied" if config.blip3_cfg else "no blip3 configuration",
            duration_ms=timings.get("stage.blip3") if config.blip3_cfg else None,
        ),
        StageStatus(
            name="label_filter",
            status="executed",
            detail=f"{pre_filter_count} -> {len(final_masks)}",
            duration_ms=timings.get("stage.final_filter"),
        ),
        StageStatus(
            name="ordering",
            status="executed",
            detail=f"{len(objects)} final object(s) assigned ids 1..{len(objects)}",
            duration_ms=timings.get("stage.ordering"),
        ),
        StageStatus(
            name="visualization",
            status=(
                "executed"
                if rendered
                else ("skipped" if not should_render_visualizations else "not_configured")
            ),
            detail=(
                f"{len(rendered)} stream(s)"
                if should_render_visualizations
                else "rendering disabled for this service verbosity"
            ),
            duration_ms=timings.get("stage.visualization") if rendered else None,
        ),
    )

    aggregate_warnings = list(warnings_out)
    for obj in objects:
        for warning in obj.warnings:
            aggregate_warnings.append(f"object {obj.instance_id}: {warning}")

    provenance = Provenance(
        config_digest=config_digest(config),
        notes=("mask remap: inverse nearest-neighbor to original coordinates",),
    )
    clip_prompt_metadata = dict(config.clip_prompt_metadata)
    if not clip_prompt_metadata and isinstance(config.clip_cfg, Mapping):
        labels = config.clip_cfg.get("labels", {})
        if isinstance(labels, Mapping):
            summary = summarize_canonical_labels(labels)
            if summary.total_prompt_count:
                clip_prompt_metadata = summary.as_dict()

    result = PipelineResult(
        image_height=int(h_orig),
        image_width=int(w_orig),
        roi_box=(int(x), int(y), int(x2), int(y2)),
        resize_info=dict(resize_info),
        objects=tuple(objects),
        stage_statuses=stage_statuses,
        candidate_counts=candidate_counts,
        rendered=rendered,
        warnings=tuple(aggregate_warnings),
        timings=dict(timings),
        provenance=provenance,
        post_filter_diagnostics=post_filter_diagnostics,
        sam2_metadata=sam2_metadata,
        candidate_view_inputs=tuple(dict(record) for record in candidate_view_inputs),
        blip3_candidate_views=tuple(dict(record) for record in blip3_candidate_views),
        clip_routing_diagnostics=tuple(dict(record) for record in clip_routing_diagnostics),
        clip_prompt_metadata=clip_prompt_metadata,
    )
    return SingleImageOutcome(
        result=result,
        segmenter_state=segmenter_state,
        clip_state=clip_state,
        blip3_state=blip3_state,
    )
