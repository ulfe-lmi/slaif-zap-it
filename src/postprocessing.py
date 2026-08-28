"""Post-processing helpers for segmentation masks."""

from __future__ import annotations

from collections.abc import MutableMapping
from numbers import Integral
from typing import Any

import numpy as np

__all__ = [
    "MAX_POST_FILTER_REJECTION_RECORDS",
    "POST_FILTER_REASON_PRECEDENCE",
    "filter_by_area_bbox",
]

MAX_POST_FILTER_REJECTION_RECORDS = 256
POST_FILTER_REASON_PRECEDENCE = ("maxsize", "empty_mask", "max_w", "max_h")


def _public_number(value: Any) -> Any:
    """Convert NumPy numeric scalars without changing ordinary numeric values."""
    return value.item() if isinstance(value, np.generic) else value


def _source_index(mask: dict, ordinal: int) -> int:
    value = mask.get("_source_index")
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return int(value)
    return ordinal


def _diagnostic_record(
    mask: dict,
    ordinal: int,
    reason: str,
    area_px: int,
    bbox_width_px: int,
    bbox_height_px: int,
) -> dict[str, Any]:
    """Return the intentionally numeric-only public rejection record."""
    return {
        "source_index": _source_index(mask, ordinal),
        "reason": reason,
        "area_px": area_px,
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
    }


def _evaluate_candidate(
    mask: dict,
    post_maxsize: Any,
    max_w: Any,
    max_h: Any,
) -> tuple[str, int, int, int]:
    """Evaluate one candidate using the canonical mutually exclusive order.

    The terminal maxsize branch deliberately returns zero bbox dimensions because
    segmentation and bbox dimensions have not been evaluated yet.
    """
    area_value = mask["area"]
    area_px = int(area_value)
    if area_value > post_maxsize:
        return "maxsize", area_px, 0, 0

    segmentation = mask["segmentation"]
    rr, cc = np.where(segmentation)
    if len(rr) == 0:
        bbox_width_px = bbox_height_px = 0
    else:
        bbox_width_px = int(cc.max() - cc.min() + 1)
        bbox_height_px = int(rr.max() - rr.min() + 1)
    if len(rr) == 0:
        return "empty_mask", area_px, 0, 0
    if bbox_width_px > max_w:
        return "max_w", area_px, bbox_width_px, bbox_height_px
    if bbox_height_px > max_h:
        return "max_h", area_px, bbox_width_px, bbox_height_px
    return "retained", area_px, bbox_width_px, bbox_height_px


def _diagnostic_template(post_maxsize: Any, max_w: Any, max_h: Any) -> dict[str, Any]:
    return {
        "limits": {
            "maxsize": _public_number(post_maxsize),
            "max_w": _public_number(max_w),
            "max_h": _public_number(max_h),
        },
        "evaluated": 0,
        "removed_by_maxsize": 0,
        "removed_empty_mask": 0,
        "removed_by_max_w": 0,
        "removed_by_max_h": 0,
        "retained": 0,
        "reason_precedence": list(POST_FILTER_REASON_PRECEDENCE),
        "rejections": [],
        "rejections_truncated": 0,
    }


def filter_by_area_bbox(
    masks,
    post_maxsize,
    max_w,
    max_h,
    verbosity: int = 1,
    log_print_func=None,
    diagnostics: MutableMapping[str, Any] | None = None,
    collect_rejections: bool = True,
):
    """Filter masks whose area or bounding-box exceeds configured thresholds.

    Args:
        masks: Iterable of dictionaries containing a ``"segmentation"`` boolean array
            and precomputed ``"area"`` values.
        post_maxsize: Maximum allowed mask area.
        max_w: Maximum allowed bounding-box width.
        max_h: Maximum allowed bounding-box height.
        verbosity: Logging verbosity level (0..3).
        log_print_func: Optional logging callback compatible with ``log_print``.
        diagnostics: Optional mapping to populate with deterministic aggregate and
            bounded rejection diagnostics. The list return value is unchanged.
        collect_rejections: Whether to retain the bounded rejection records. Counts
            are always collected when ``diagnostics`` is supplied.

    Returns:
        List of mask dictionaries that satisfy the configured limits.
    """

    mask_list = list(masks)
    kept = []
    diagnostic = _diagnostic_template(post_maxsize, max_w, max_h)
    rejected_total = 0
    for ordinal, mm in enumerate(mask_list):
        reason, area_px, bbox_width_px, bbox_height_px = _evaluate_candidate(
            mm, post_maxsize, max_w, max_h
        )
        diagnostic["evaluated"] += 1
        if reason == "retained":
            kept.append(mm)
            diagnostic["retained"] += 1
            continue

        rejected_total += 1
        diagnostic[
            {
                "maxsize": "removed_by_maxsize",
                "empty_mask": "removed_empty_mask",
                "max_w": "removed_by_max_w",
                "max_h": "removed_by_max_h",
            }[reason]
        ] += 1
        if collect_rejections and len(diagnostic["rejections"]) < MAX_POST_FILTER_REJECTION_RECORDS:
            diagnostic["rejections"].append(
                _diagnostic_record(mm, ordinal, reason, area_px, bbox_width_px, bbox_height_px)
            )

    diagnostic["rejections_truncated"] = rejected_total - len(diagnostic["rejections"])
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(diagnostic)

    if log_print_func and verbosity >= 1:
        log_print_func(
            f"[postsam2processing] => from {len(mask_list)} => {len(kept)} remain by area/box",
            1,
            verbosity,
        )

    return kept
