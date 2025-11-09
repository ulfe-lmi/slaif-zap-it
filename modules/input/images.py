"""Utilities for working with image inputs."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps


_SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg",)


def list_images(base_dir: str, extensions: Sequence[str] = _SUPPORTED_EXTENSIONS) -> list[str]:
    """Return a sorted list of image filenames in ``base_dir`` matching ``extensions``."""
    lowered_exts = tuple(ext.lower() for ext in extensions)
    images = [
        name
        for name in os.listdir(base_dir)
        if name.lower().endswith(lowered_exts)
        and os.path.isfile(os.path.join(base_dir, name))
    ]
    images.sort()
    return images


def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    """Load ``path`` into a PIL image and RGB NumPy array with EXIF orientation handled."""
    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image, np.array(image)


def apply_roi(
    image_np: np.ndarray, roi_value: Optional[str]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop ``image_np`` to ``roi_value`` if provided."""
    height, width = image_np.shape[:2]
    if not roi_value:
        return image_np, (0, 0, width, height)

    x, y, w, h = [int(v) for v in str(roi_value).split(",")]
    x2 = min(x + w, width)
    y2 = min(y + h, height)
    return image_np[y:y2, x:x2, :], (x, y, x2, y2)


def resize_image(
    image_np: np.ndarray, resize_value: Optional[str]
) -> tuple[np.ndarray, dict[str, object]]:
    """Resize ``image_np`` according to ``resize_value``."""
    if resize_value is None:
        return image_np, {"mode": "native"}

    factor = float(resize_value)
    if abs(factor - 1.0) < 1e-7:
        return image_np, {"mode": "native"}

    new_w = int(image_np.shape[1] * factor)
    new_h = int(image_np.shape[0] * factor)
    resized = np.array(
        Image.fromarray(image_np).resize((new_w, new_h), Image.Resampling.LANCZOS)
    )

    mode = "downscale" if factor < 1.0 else "upscale"
    return resized, {"mode": mode, "factor": factor, "size": (new_w, new_h)}


def save_roi_debug(image_np: np.ndarray, path: str) -> None:
    """Persist ``image_np`` to ``path`` for debugging."""
    Image.fromarray(image_np).save(path, "JPEG")
