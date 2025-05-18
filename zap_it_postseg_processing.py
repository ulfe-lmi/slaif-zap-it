"""
zap-it-postseg-processing.py

Handles post-SAM2 filters for area, bounding box size, etc.
We can also place other post-seg filtering logic here if desired.
"""

import numpy as np

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
        log_print_func(f"[postsam2processing] => from {len(masks)} => {len(kept)} remain by area/box", 1, verbosity)

    return kept
