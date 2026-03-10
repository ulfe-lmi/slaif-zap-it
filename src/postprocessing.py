"""Post-processing helpers for segmentation masks."""

from __future__ import annotations

import numpy as np


def filter_by_area_bbox(
    masks,
    post_maxsize,
    max_w,
    max_h,
    verbosity: int = 1,
    log_print_func=None,
):
    """Filter masks whose area or bounding-box exceeds configured thresholds.

    Args:
        masks: Iterable of dictionaries containing a ``"segmentation"`` boolean array
            and precomputed ``"area"`` values.
        post_maxsize: Maximum allowed mask area.
        max_w: Maximum allowed bounding-box width.
        max_h: Maximum allowed bounding-box height.
        verbosity: Logging verbosity level (0..2).
        log_print_func: Optional logging callback compatible with ``log_print``.

    Returns:
        List of mask dictionaries that satisfy the configured limits.
    """

    kept = []
    for mm in masks:
        if mm["area"] > post_maxsize:
            continue
        seg_bool = mm["segmentation"]
        rr, cc = np.where(seg_bool)
        if len(rr) == 0:
            continue
        y_min, y_max = rr.min(), rr.max()
        x_min, x_max = cc.min(), cc.max()
        w_box = x_max - x_min + 1
        h_box = y_max - y_min + 1
        if w_box <= max_w and h_box <= max_h:
            kept.append(mm)

    if log_print_func and verbosity >= 1:
        log_print_func(
            f"[postsam2processing] => from {len(masks)} => {len(kept)} remain by area/box",
            1,
            verbosity,
        )

    return kept
