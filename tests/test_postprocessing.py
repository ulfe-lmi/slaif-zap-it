import numpy as np

from src.postprocessing import filter_by_area_bbox


def _make_mask(area, bbox, fill=True):
    seg = np.zeros((10, 10), dtype=bool)
    if fill:
        x0, y0, x1, y1 = bbox
        seg[y0:y1, x0:x1] = True
    return {"area": area, "segmentation": seg}


def test_filter_by_area_bbox_rejects_large_area():
    masks = [
        _make_mask(5, (0, 0, 2, 2)),
        _make_mask(500, (0, 0, 3, 3)),
    ]
    kept = filter_by_area_bbox(masks, post_maxsize=100, max_w=10, max_h=10)
    assert len(kept) == 1
    assert kept[0]["area"] == 5


def test_filter_by_area_bbox_enforces_bbox_and_logs():
    messages = []

    def logger(msg, level, verbosity):
        messages.append((msg, level, verbosity))

    masks = [
        _make_mask(10, (0, 0, 5, 6)),
        _make_mask(10, (0, 0, 9, 9)),
    ]
    kept = filter_by_area_bbox(masks, post_maxsize=50, max_w=5, max_h=6, verbosity=2, log_print_func=logger)
    assert len(kept) == 1
    assert kept[0]["segmentation"].sum() == 30
    assert any("postsam2processing" in msg for msg, *_ in messages)


def test_filter_by_area_bbox_skips_empty_masks():
    mask = _make_mask(1, (0, 0, 1, 1), fill=False)
    kept = filter_by_area_bbox([mask], post_maxsize=10, max_w=10, max_h=10)
    assert kept == []
