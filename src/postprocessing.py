"""Post-processing helpers for segmentation masks."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from numbers import Integral
from typing import Any

import numpy as np

__all__ = [
    "MAX_POST_FILTER_REJECTION_RECORDS",
    "POST_FILTER_REASON_PRECEDENCE",
    "GEOMETRY_REASON_PRECEDENCE",
    "filter_by_area_bbox",
    "filter_by_geometry",
]

MAX_POST_FILTER_REJECTION_RECORDS = 256
POST_FILTER_REASON_PRECEDENCE = ("maxsize", "empty_mask", "max_w", "max_h")
GEOMETRY_REASON_PRECEDENCE = (
    "empty_mask",
    "min_area",
    "max_area",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "min_aspect_ratio",
    "max_aspect_ratio",
    "border_touching",
)


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
    post_maxsize=999_999_999,
    max_w=999_999_999,
    max_h=999_999_999,
    verbosity: int = 1,
    log_print_func=None,
    diagnostics: MutableMapping[str, Any] | None = None,
    collect_rejections: bool = True,
    geometry_config: MutableMapping[str, Any] | None = None,
    **geometry_kwargs: Any,
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

    canonical = geometry_config
    if canonical is None and geometry_kwargs:
        canonical = geometry_kwargs
    if canonical is not None:
        return filter_by_geometry(
            masks,
            canonical,
            verbosity=verbosity,
            log_print_func=log_print_func,
            diagnostics=diagnostics,
            collect_rejections=collect_rejections,
        )

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


def _geometry_limit_value(config: Mapping[str, Any], field_name: str) -> Any:
    value = config.get(field_name)
    return _public_number(value)


def _geometry_record(
    mask: Mapping[str, Any],
    ordinal: int,
    *,
    reason: str,
    area_px: int,
    bbox: tuple[int, int, int, int] | None,
    bbox_width_px: int,
    bbox_height_px: int,
    aspect_ratio: float | None,
    border_touching: bool | None,
    limit_field: str | None,
    limit_value: Any,
) -> dict[str, Any]:
    source_index = _source_index(dict(mask), ordinal)
    return {
        "source_candidate_id": source_index + 1,
        "source_index": source_index,
        "filtered_index": int(mask.get("_filtered_index", ordinal)),
        "reason": reason,
        "area_px": area_px,
        "bbox_xyxy_inclusive": None if bbox is None else list(bbox),
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "aspect_ratio": aspect_ratio,
        "border_touching": border_touching,
        "configured_limit_field": limit_field,
        "configured_limit_value": _public_number(limit_value),
    }


def filter_by_geometry(
    masks,
    config: Mapping[str, Any] | None = None,
    *,
    verbosity: int = 1,
    log_print_func=None,
    diagnostics: MutableMapping[str, Any] | None = None,
    collect_rejections: bool = True,
    **geometry_kwargs: Any,
):
    """Apply optional geometry impossibility rules without hiding candidates.

    Every source candidate is evaluated far enough to report its inclusive
    bounding box.  A bounded L3 rejection record is retained for each rejected
    candidate, while the returned list contains only retained non-empty masks.
    """
    config = {**dict(config or {}), **geometry_kwargs}
    limits = {
        name: _geometry_limit_value(config, name)
        for name in (
            "min_area",
            "max_area",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_aspect_ratio",
            "max_aspect_ratio",
        )
    }
    allow_border_touching = config.get("allow_border_touching", True)
    diagnostic: dict[str, Any] = {
        "limits": {**limits, "allow_border_touching": bool(allow_border_touching)},
        "evaluated": 0,
        "non_empty": 0,
        "retained": 0,
        "rejected": 0,
        "rejections": [],
        "rejections_truncated": 0,
        "reason_precedence": list(GEOMETRY_REASON_PRECEDENCE),
    }
    for reason in GEOMETRY_REASON_PRECEDENCE:
        diagnostic[f"removed_by_{reason}"] = 0
    kept: list[dict[str, Any]] = []
    rejected_total = 0
    mask_list = list(masks)
    for ordinal, mask in enumerate(mask_list):
        diagnostic["evaluated"] += 1
        segmentation = np.asarray(mask.get("segmentation"), dtype=bool)
        rows, cols = np.nonzero(segmentation)
        area_px = int(np.count_nonzero(segmentation))
        if rows.size == 0:
            bbox = None
            bbox_width_px = bbox_height_px = 0
            aspect_ratio = None
            border_touching = None
            reason = "empty_mask"
            limit_field = None
            limit_value = None
        else:
            diagnostic["non_empty"] += 1
            bbox = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
            bbox_width_px = bbox[2] - bbox[0] + 1
            bbox_height_px = bbox[3] - bbox[1] + 1
            aspect_ratio = bbox_width_px / float(bbox_height_px)
            height, width = segmentation.shape
            border_touching = (
                bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1
            )
            reason = "retained"
            limit_field = None
            limit_value = None
            checks = (
                (
                    "min_area",
                    limits["min_area"],
                    area_px < limits["min_area"] if limits["min_area"] is not None else False,
                ),
                (
                    "max_area",
                    limits["max_area"],
                    area_px > limits["max_area"] if limits["max_area"] is not None else False,
                ),
                (
                    "min_width",
                    limits["min_width"],
                    bbox_width_px < limits["min_width"]
                    if limits["min_width"] is not None
                    else False,
                ),
                (
                    "max_width",
                    limits["max_width"],
                    bbox_width_px > limits["max_width"]
                    if limits["max_width"] is not None
                    else False,
                ),
                (
                    "min_height",
                    limits["min_height"],
                    bbox_height_px < limits["min_height"]
                    if limits["min_height"] is not None
                    else False,
                ),
                (
                    "max_height",
                    limits["max_height"],
                    bbox_height_px > limits["max_height"]
                    if limits["max_height"] is not None
                    else False,
                ),
                (
                    "min_aspect_ratio",
                    limits["min_aspect_ratio"],
                    aspect_ratio < limits["min_aspect_ratio"]
                    if limits["min_aspect_ratio"] is not None
                    else False,
                ),
                (
                    "max_aspect_ratio",
                    limits["max_aspect_ratio"],
                    aspect_ratio > limits["max_aspect_ratio"]
                    if limits["max_aspect_ratio"] is not None
                    else False,
                ),
                ("border_touching", False, border_touching and allow_border_touching is False),
            )
            for field_name, value, failed in checks:
                if failed:
                    reason = field_name
                    limit_field = (
                        field_name if field_name != "border_touching" else "allow_border_touching"
                    )
                    limit_value = value if field_name != "border_touching" else False
                    break

        if reason == "retained":
            mask["geometry"] = {
                "area_px": area_px,
                "bbox_xyxy_inclusive": list(bbox) if bbox is not None else None,
                "bbox_width_px": bbox_width_px,
                "bbox_height_px": bbox_height_px,
                "aspect_ratio": aspect_ratio,
                "border_touching": border_touching,
            }
            kept.append(mask)
            diagnostic["retained"] += 1
            continue
        rejected_total += 1
        diagnostic["rejected"] += 1
        diagnostic[f"removed_by_{reason}"] += 1
        if collect_rejections and len(diagnostic["rejections"]) < MAX_POST_FILTER_REJECTION_RECORDS:
            diagnostic["rejections"].append(
                _geometry_record(
                    mask,
                    ordinal,
                    reason=reason,
                    area_px=area_px,
                    bbox=bbox,
                    bbox_width_px=bbox_width_px,
                    bbox_height_px=bbox_height_px,
                    aspect_ratio=aspect_ratio,
                    border_touching=border_touching,
                    limit_field=limit_field,
                    limit_value=limit_value,
                )
            )
    diagnostic["rejections_truncated"] = rejected_total - len(diagnostic["rejections"])
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(diagnostic)
    if log_print_func and verbosity >= 1:
        log_print_func(
            f"[postsam2processing] => from {len(mask_list)} => {len(kept)} remain by geometry",
            1,
            verbosity,
        )
    return kept
