"""Tests for the pure YOLO and uint16 identity-mask renderers."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from src.core import (
    MAX_IDENTITY_OBJECTS,
    ObjectResult,
    IdentityMaskOverflowError,
    format_yolo_line,
    render_identity_png,
    render_yolo,
)


def obj(instance_id, rows_cols, shape=(10, 10), label="cat", class_id=0, **kwargs):
    mask = np.zeros(shape, dtype=bool)
    for r, c in rows_cols:
        mask[r, c] = True
    return ObjectResult(
        instance_id=instance_id,
        source_index=instance_id - 1,
        mask=mask,
        metadata={"clip_label": label},
        class_id=class_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# YOLO renderer
# ---------------------------------------------------------------------------


def test_yolo_line_format_is_five_fields_fixed_precision():
    line = format_yolo_line(3, 0.5, 0.25, 0.125, 0.0625)
    assert line == "3 0.500000 0.250000 0.125000 0.062500"


def test_yolo_coordinates_normalized_to_original_dimensions():
    # bbox pixel extents x1,y1,x2,y2 = (1,1,2,2) on a 4x4 image
    o = obj(1, [(1, 1), (1, 2), (2, 1), (2, 2)], shape=(4, 4), class_id=2)
    text = render_yolo([o], image_width=4, image_height=4)
    assert text == "2 0.375000 0.375000 0.500000 0.500000\n"


def test_yolo_empty_detections_produce_empty_text():
    assert render_yolo([], image_width=4, image_height=4) == ""


def test_yolo_emits_one_line_per_object_in_final_order():
    first = obj(1, [(0, 0)], class_id=5)
    second = obj(2, [(9, 9)], label=None, class_id=0)
    text = render_yolo([first, second], image_width=10, image_height=10)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("5 ")
    assert lines[1].startswith("0 ")
    assert len(lines[0].split(" ")) == 5


def test_yolo_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        render_yolo([], image_width=0, image_height=10)


# ---------------------------------------------------------------------------
# Identity mask renderer
# ---------------------------------------------------------------------------


def test_identity_png_dims_dtype_background_and_ids():
    a = obj(1, [(0, 0), (0, 1)], shape=(8, 10))
    b = obj(2, [(5, 5)], shape=(8, 10))
    png_bytes = render_identity_png([a, b], width=10, height=8)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert decoded.shape == (8, 10)
    assert decoded.dtype == np.uint16
    assert int(decoded.sum()) == (1 * 2) + (2 * 1)
    assert decoded[3, 3] == 0
    assert decoded[0, 0] == 1
    assert decoded[0, 1] == 1
    assert decoded[5, 5] == 2


def test_disconnected_components_share_one_instance_id():
    blob = obj(7, [(0, 0), (9, 9)])  # two disjoint components, one object
    png_bytes = render_identity_png([blob], width=10, height=10)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert set(np.unique(decoded)) == {0, 7}


def test_larger_area_object_wins_contested_pixels():
    small = obj(1, [(0, 0)], shape=(2, 2))  # area 1
    large = obj(2, [(0, 0), (0, 1), (1, 0)], shape=(2, 2))  # area 3, overlaps at (0,0)
    png_bytes = render_identity_png([small, large], width=2, height=2)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert decoded[0, 0] == 2  # larger area wins
    assert decoded[0, 1] == 2


def test_area_tie_wins_by_smaller_instance_id():
    one = obj(1, [(0, 0)], shape=(1, 1))
    two = obj(2, [(0, 0)], shape=(1, 1))
    png_bytes = render_identity_png([one, two], width=1, height=1)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert decoded[0, 0] == 1


def test_identity_png_bytes_are_deterministic():
    def build():
        return [
            obj(1, [(0, 0)], shape=(8, 8)),
            obj(2, [(3, 3), (3, 4)], shape=(8, 8)),
        ]

    first = render_identity_png(build(), width=8, height=8)
    second = render_identity_png(build(), width=8, height=8)
    assert first == second


def test_identity_overflow_guard_raises_before_allocation():
    class _HugeSequence:
        def __len__(self):
            return MAX_IDENTITY_OBJECTS + 1

        def __iter__(self):  # pragma: no cover - must never be reached
            raise AssertionError("renderer must guard before touching elements")

    with pytest.raises(IdentityMaskOverflowError):
        render_identity_png(_HugeSequence(), width=4, height=4)


def test_serialized_metadata_is_json_safe_and_deterministic():
    o = ObjectResult(
        instance_id=1,
        source_index=0,
        mask=np.zeros((2, 2), dtype=bool),
        metadata={
            "clip_label": "cat",
            "clip_score": np.float64(0.5),
            "count": np.int32(3),
            "segmentation_like": np.zeros((2, 2), dtype=bool),
        },
    )
    payload = o.serialized_metadata()
    assert "segmentation_like" not in payload
    assert payload["count"] == 3
    assert isinstance(payload["clip_score"], float)
    assert json.dumps(payload) == json.dumps(o.serialized_metadata())
