"""Tests for post-segmentation processing helpers."""

import numpy as np

from zap_it_postseg_processing import filter_by_area_bbox


def _make_mask(width, height):
    seg = np.ones((height, width), dtype=bool)
    return {"segmentation": seg, "area": int(seg.sum())}


def test_filter_accepts_none_limits():
    masks = [_make_mask(2, 2), _make_mask(4, 4)]

    kept = filter_by_area_bbox(masks, None, None, None, verbosity=0)

    assert kept == masks


def test_filter_accepts_zero_limits():
    masks = [_make_mask(2, 2), _make_mask(4, 4)]

    kept = filter_by_area_bbox(masks, 0, 0, 0, verbosity=0)

    assert kept == masks


def test_filter_still_applies_positive_limits():
    masks = [_make_mask(2, 2), _make_mask(4, 4)]

    kept = filter_by_area_bbox(masks, 5, 3, 3, verbosity=0)

    assert kept == [masks[0]]
