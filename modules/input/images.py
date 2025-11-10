"""Utilities for working with image inputs."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - pillow is optional in some environments
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


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
    if Image is None or ImageOps is None:  # pragma: no cover - dependent on pillow being missing
        raise RuntimeError("Pillow is required to load images.")

    image = Image.open(path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image, np.array(image)


def apply_roi(
    image_np: np.ndarray, roi_value: Optional[str]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop ``image_np`` to ``roi_value`` if provided."""
    height, width = image_np.shape[:2]
    if roi_value is None:
        return image_np, (0, 0, width, height)

    roi_str = str(roi_value).strip()
    if not roi_str or roi_str.lower() in {"false", "none"}:
        return image_np, (0, 0, width, height)

    parts = [part.strip() for part in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "ROI must contain exactly four comma-separated integers; got:"
            f" {roi_value!r}"
        )

    try:
        x, y, w, h = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "ROI must contain exactly four comma-separated integers; got:"
            f" {roi_value!r}"
        ) from exc

    # Clamp the coordinates so negative inputs do not wrap around the image by
    # computing the intersection between the ROI and the image bounds.
    x1 = max(0, min(x, width))
    y1 = max(0, min(y, height))
    x2 = min(width, max(x + w, x1))
    y2 = min(height, max(y + h, y1))

    return image_np[y1:y2, x1:x2, :], (x1, y1, x2, y2)


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
    if Image is None:  # pragma: no cover - dependent on pillow being missing
        raise RuntimeError("Pillow is required to resize images.")

    resized = np.array(
        Image.fromarray(image_np).resize((new_w, new_h), Image.Resampling.LANCZOS)
    )

    mode = "downscale" if factor < 1.0 else "upscale"
    return resized, {"mode": mode, "factor": factor, "size": (new_w, new_h)}


def save_roi_debug(image_np: np.ndarray, path: str) -> None:
    """Persist ``image_np`` to ``path`` for debugging."""
    if Image is None:  # pragma: no cover - dependent on pillow being missing
        raise RuntimeError("Pillow is required to save ROI debug images.")

    Image.fromarray(image_np).save(path, "JPEG")
