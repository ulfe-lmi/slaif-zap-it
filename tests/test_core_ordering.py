"""Tests for the deterministic final-object ordering definition."""

import numpy as np

from src.core import order_final_objects, ordering_key


def mask(rows_cols, area=None):
    seg = np.zeros((10, 10), dtype=bool)
    for r, c in rows_cols:
        seg[r, c] = True
    entry = {"segmentation": seg}
    if area is not None:
        entry["area"] = area
    return entry


def test_ordering_prefers_descending_area():
    small = mask([(0, 0)])
    large = mask([(5, 5), (6, 6), (7, 7)])
    ordered = order_final_objects([small, large])
    assert ordered[0] is large
    assert ordered[1] is small


def test_area_tie_breaks_by_centroid_row_then_col():
    left = mask([(2, 1), (8, 1)])  # centroid row 5.0, col 1.0
    right = mask([(2, 3), (8, 3)])  # centroid row 5.0, col 3.0
    upper = mask([(0, 5), (4, 5)])  # centroid row 2.0, col 5.0
    ordered = order_final_objects([right, upper, left])
    assert [id(m) for m in ordered] == [id(upper), id(left), id(right)]


def test_full_tie_resolves_to_earliest_candidate_index():
    same_a = mask([(0, 0)])
    same_b = mask([(0, 0)])
    key_a = ordering_key(0, same_a)
    key_b = ordering_key(1, same_b)
    assert key_a < key_b
    ordered = order_final_objects([same_b, same_a])
    assert ordered[0] is same_b  # earliest candidate index wins exact ties


def test_precomputed_area_is_honored_over_pixel_count():
    tiny = mask([(0, 0)], area=999)
    big_pixels = mask([(5, 5), (6, 6), (7, 7)], area=1)
    ordered = order_final_objects([big_pixels, tiny])
    assert ordered[0] is tiny


def test_input_dicts_are_not_copied():
    entry = mask([(0, 0)])
    ordered = order_final_objects([entry])
    assert ordered[0] is entry
