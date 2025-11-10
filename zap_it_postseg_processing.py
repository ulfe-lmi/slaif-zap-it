"""
zap-it-postseg-processing.py

Handles post-SAM2 filters for area, bounding box size, etc.
We can also place other post-seg filtering logic here if desired.
"""

import numpy as np


def _normalize_limit(value):
    """Return ``None`` when a size limit should be ignored."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value <= 0 else value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid limit value: {value!r}") from None
    return None if numeric <= 0 else numeric


def filter_by_area_bbox(masks, post_maxsize, max_w, max_h, verbosity=1, log_print_func=None):
    """
    Given a list of 'masks' (each with 'segmentation' & 'area'),
    discard any whose area > post_maxsize or whose bounding box w/h exceed max_w / max_h.

    :param masks: list of dict => {"segmentation", "area", ...}
    :param post_maxsize: maximum mask area
    :param max_w: maximum bounding box width
    :param max_h: maximum bounding box height
    :param verbosity: 0..2
    :param log_print_func: optional log printing function
    :return: a filtered list of the same dict objects
    """
    area_limit = _normalize_limit(post_maxsize)
    max_w_limit = _normalize_limit(max_w)
    max_h_limit = _normalize_limit(max_h)

    kept = []
    for mm in masks:
        if area_limit is not None and mm["area"] > area_limit:
            continue
        seg_bool = mm["segmentation"]
        rr, cc = np.where(seg_bool)
        if len(rr) == 0:
            continue
        y_min, y_max = rr.min(), rr.max()
        x_min, x_max = cc.min(), cc.max()
        w_box = x_max - x_min + 1
        h_box = y_max - y_min + 1
        if (
            (max_w_limit is None or w_box <= max_w_limit)
            and (max_h_limit is None or h_box <= max_h_limit)
        ):
            kept.append(mm)

    if log_print_func and verbosity >= 1:
        log_print_func(f"[postsam2processing] => from {len(masks)} => {len(kept)} remain by area/box", 1, verbosity)

    return kept
