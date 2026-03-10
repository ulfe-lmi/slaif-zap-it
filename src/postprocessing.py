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


def krippendorff_alpha_ordinal(ratings, categories: int = 3) -> float:
    """Compute Krippendorff's alpha for ordinal annotations.

    Args:
        ratings: 2D array-like of shape (units, coders). Missing ratings can be
            represented by ``None`` or ``np.nan``.
        categories: Number of ordinal categories. Defaults to 3.

    Returns:
        Krippendorff's alpha value.
    """

    if categories < 2:
        raise ValueError("categories must be >= 2")

    arr = np.asarray(ratings, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2D array-like")

    coincidence = np.zeros((categories, categories), dtype=float)

    for unit in arr:
        valid = unit[~np.isnan(unit)].astype(int)
        if valid.size < 2:
            continue

        if np.any((valid < 0) | (valid >= categories)):
            raise ValueError("ratings contain values outside the category range")

        counts = np.bincount(valid, minlength=categories).astype(float)
        n_u = counts.sum()

        for i in range(categories):
            for j in range(categories):
                if i == j:
                    coincidence[i, j] += counts[i] * (counts[i] - 1.0) / (n_u - 1.0)
                else:
                    coincidence[i, j] += counts[i] * counts[j] / (n_u - 1.0)

    total = coincidence.sum()
    if total <= 0:
        raise ValueError("ratings must contain at least one unit with >=2 valid coders")

    delta = np.zeros_like(coincidence)
    denom = float(categories - 1)
    for i in range(categories):
        for j in range(categories):
            delta[i, j] = ((i - j) / denom) ** 2

    observed_disagreement = (coincidence * delta).sum() / total

    marginals = coincidence.sum(axis=1)
    expected = np.zeros_like(coincidence)
    for i in range(categories):
        for j in range(categories):
            if i == j:
                expected[i, j] = marginals[i] * (marginals[i] - 1.0) / (total - 1.0)
            else:
                expected[i, j] = marginals[i] * marginals[j] / (total - 1.0)

    expected_disagreement = (expected * delta).sum() / total

    if expected_disagreement <= 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def krippendorf_alfa(ratings) -> float:
    """Backward-compatible alias for Krippendorff alpha on 3 ordinal categories."""

    return krippendorff_alpha_ordinal(ratings, categories=3)
