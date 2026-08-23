from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.input.images import apply_roi


def test_apply_roi_clamps_negative_coordinates():
    image = np.arange(100).reshape(10, 10)
    image = np.stack([image, image, image], axis=-1)  # fake RGB

    cropped, (x, y, x2, y2) = apply_roi(image, "-2,-2,8,8")

    # Coordinates should be clamped to the image bounds instead of wrapping
    assert (x, y) == (0, 0)
    assert (x2, y2) == (6, 6)
    assert cropped.shape == (6, 6, 3)


def test_apply_roi_skips_falsey_values():
    image = np.arange(100).reshape(10, 10)
    image = np.stack([image, image, image], axis=-1)  # fake RGB

    for roi in ("", "False", "false", "None", None):
        cropped, bounds = apply_roi(image, roi)
        assert cropped is image
        assert bounds == (0, 0, image.shape[1], image.shape[0])


def test_apply_roi_raises_for_invalid_format():
    image = np.arange(100).reshape(10, 10, 1)

    for roi in ("1,2,3", "1,2,three,4", "1;2;3;4"):
        with pytest.raises(ValueError):
            apply_roi(image, roi)


def test_apply_roi_with_valid_string():
    image = np.arange(100).reshape(10, 10)
    image = np.stack([image, image, image], axis=-1)  # fake RGB

    cropped, bounds = apply_roi(image, "1,2,3,4")

    assert bounds == (1, 2, 4, 6)
    assert cropped.shape == (4, 3, 3)
