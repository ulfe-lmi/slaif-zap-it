from pathlib import Path
import sys

import numpy as np

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


