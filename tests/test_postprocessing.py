import numpy as np

from src.postprocessing import filter_by_area_bbox


def _make_mask(area, bbox, fill=True, shape=(10, 10)):
    seg = np.zeros(shape, dtype=bool)
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


def test_filter_diagnostics_maxsize_short_circuits_segmentation_access():
    class SegmentationAccessRaises(dict):
        def __getitem__(self, key):
            if key == "segmentation":
                raise AssertionError("maxsize must be decided before segmentation access")
            return super().__getitem__(key)

    candidate = SegmentationAccessRaises(area=101)
    diagnostics = {}

    assert filter_by_area_bbox([candidate], 100, 10, 10, diagnostics=diagnostics) == []
    assert diagnostics["evaluated"] == 1
    assert diagnostics["removed_by_maxsize"] == 1
    assert diagnostics["rejections"] == [
        {
            "source_index": 0,
            "reason": "maxsize",
            "area_px": 101,
            "bbox_width_px": 0,
            "bbox_height_px": 0,
        }
    ]


def test_filter_by_area_bbox_enforces_bbox_and_logs():
    messages = []

    def logger(msg, level, verbosity):
        messages.append((msg, level, verbosity))

    masks = [
        _make_mask(10, (0, 0, 5, 6)),
        _make_mask(10, (0, 0, 9, 9)),
    ]
    kept = filter_by_area_bbox(
        masks, post_maxsize=50, max_w=5, max_h=6, verbosity=2, log_print_func=logger
    )
    assert len(kept) == 1
    assert kept[0]["segmentation"].sum() == 30
    assert any("postsam2processing" in msg for msg, *_ in messages)


def test_filter_by_area_bbox_skips_empty_masks():
    mask = _make_mask(1, (0, 0, 1, 1), fill=False)
    kept = filter_by_area_bbox([mask], post_maxsize=10, max_w=10, max_h=10)
    assert kept == []


def test_filter_diagnostics_use_one_precedence_and_preserve_legacy_result():
    retained = _make_mask(10, (0, 0, 2, 2))
    too_large = _make_mask(101, (0, 0, 2, 2))
    empty = _make_mask(0, (0, 0, 1, 1), fill=False)
    too_wide = _make_mask(10, (0, 0, 7, 2))
    too_tall = _make_mask(10, (0, 0, 2, 7))
    both_dimensions = _make_mask(10, (0, 0, 7, 7))
    area_and_width = _make_mask(101, (0, 0, 7, 2))
    masks = [retained, too_large, empty, too_wide, too_tall, both_dimensions, area_and_width]
    diagnostics = {}

    kept = filter_by_area_bbox(
        masks,
        post_maxsize=100,
        max_w=5,
        max_h=5,
        diagnostics=diagnostics,
    )

    assert kept == [retained]
    assert kept[0] is retained
    assert diagnostics == {
        "limits": {"maxsize": 100, "max_w": 5, "max_h": 5},
        "evaluated": 7,
        "removed_by_maxsize": 2,
        "removed_empty_mask": 1,
        "removed_by_max_w": 2,
        "removed_by_max_h": 1,
        "retained": 1,
        "reason_precedence": ["maxsize", "empty_mask", "max_w", "max_h"],
        "rejections": [
            {
                "source_index": index,
                "reason": reason,
                "area_px": area,
                "bbox_width_px": width,
                "bbox_height_px": height,
            }
            for index, reason, area, width, height in (
                (1, "maxsize", 101, 0, 0),
                (2, "empty_mask", 0, 0, 0),
                (3, "max_w", 10, 7, 2),
                (4, "max_h", 10, 2, 7),
                (5, "max_w", 10, 7, 7),
                (6, "maxsize", 101, 0, 0),
            )
        ],
        "rejections_truncated": 0,
    }
    assert diagnostics["evaluated"] == diagnostics["retained"] + sum(
        diagnostics[key]
        for key in (
            "removed_by_maxsize",
            "removed_empty_mask",
            "removed_by_max_w",
            "removed_by_max_h",
        )
    )


def test_filter_diagnostics_retain_exact_thresholds_and_content_free_logging():
    mask = _make_mask(4, (0, 0, 2, 2))
    mask.update({"_source_index": 17, "prompt": "do not expose"})
    messages = []
    diagnostics = {}

    kept = filter_by_area_bbox(
        [mask],
        post_maxsize=4,
        max_w=2,
        max_h=2,
        verbosity=2,
        log_print_func=lambda *args: messages.append(args),
        diagnostics=diagnostics,
    )

    assert kept == [mask]
    assert diagnostics["retained"] == 1
    assert diagnostics["rejections"] == []
    assert "do not expose" not in repr(messages)


def test_filter_diagnostics_roof_wide_candidates_are_numeric_filter_evidence():
    masks = []
    for source_index, start in ((7, 10), (9, 70)):
        mask = _make_mask(121 * 12, (start, 0, start + 121, 12), shape=(12, 200))
        mask["_source_index"] = source_index
        mask["label"] = "roof panel"
        masks.append(mask)
    diagnostics = {}

    kept = filter_by_area_bbox(
        masks,
        post_maxsize=10_000,
        max_w=100,
        max_h=12,
        diagnostics=diagnostics,
    )

    assert kept == []
    assert diagnostics["evaluated"] == 2
    assert diagnostics["removed_by_max_w"] == 2
    assert [record["source_index"] for record in diagnostics["rejections"]] == [7, 9]
    assert [record["bbox_width_px"] for record in diagnostics["rejections"]] == [121, 121]
    assert all(
        set(record) == {"source_index", "reason", "area_px", "bbox_width_px", "bbox_height_px"}
        for record in diagnostics["rejections"]
    )


def test_filter_diagnostics_use_inclusive_bbox_and_fallback_source_ordinal():
    segmentation = np.zeros((4, 5), dtype=bool)
    segmentation[0, 0] = True
    segmentation[0, 4] = True
    segmentation[3, 2] = True
    candidate = {"segmentation": segmentation, "area": 3, "answer": "hidden"}
    diagnostics = {}

    kept = filter_by_area_bbox(
        [candidate], post_maxsize=3, max_w=4, max_h=4, diagnostics=diagnostics
    )

    assert kept == []
    record = diagnostics["rejections"][0]
    assert record == {
        "source_index": 0,
        "reason": "max_w",
        "area_px": 3,
        "bbox_width_px": 5,
        "bbox_height_px": 4,
    }
    assert "answer" not in record


def test_filter_diagnostics_are_bounded_and_repeatable():
    masks = [_make_mask(1, (0, 0, 1, 1)) for _ in range(257)]
    first = {}
    second = {}

    assert filter_by_area_bbox(masks, 100, 0, 100, diagnostics=first) == []
    assert filter_by_area_bbox(masks, 100, 0, 100, diagnostics=second) == []
    assert first == second
    assert first["evaluated"] == 257
    assert first["removed_by_max_w"] == 257
    assert len(first["rejections"]) == 256
    assert first["rejections"][0]["source_index"] == 0
    assert first["rejections"][-1]["source_index"] == 255
    assert first["rejections_truncated"] == 1
