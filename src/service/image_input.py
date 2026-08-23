"""Bounded, hostile-safe image decoding.

Uploads are decoded only after the encoded byte size has already been
enforced. Decoded dimensions are read from the image header before any pixel
buffer is allocated so decompression bombs are rejected cheaply. Only the
frozen safe media set (JPEG/PNG/WebP) is accepted; everything else —
including truncated or malformed codecs — is a stable ``invalid_image``
error and never leaks codec internals.
"""

from __future__ import annotations

import io

import numpy as np

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - pillow is a required dependency
    Image = None  # type: ignore[assignment]

from .errors import ServiceError

__all__ = ["ALLOWED_IMAGE_FORMATS", "decode_image_safely"]

ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def decode_image_safely(
    data: bytes,
    *,
    max_decoded_pixels: int,
    max_width: int = 8192,
    max_height: int = 8192,
) -> np.ndarray:
    """Decode ``data`` into an RGB ``uint8`` array under strict limits."""
    if not data:
        raise ServiceError("image part is empty", code="invalid_image")
    if Image is None:  # pragma: no cover - broken install guard
        raise RuntimeError("Pillow is required to decode uploaded images.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            fmt = (opened.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise ServiceError(
                    "unsupported image media type; allowed: JPEG, PNG, WebP",
                    code="invalid_image",
                )
            width, height = opened.size
            if width < 1 or height < 1:
                raise ServiceError("image has invalid dimensions", code="invalid_image")
            if width > max_width or height > max_height:
                raise ServiceError(
                    "decoded image dimensions exceed the maximum allowed size",
                    code="image_too_large",
                )
            if width * height > max_decoded_pixels:
                raise ServiceError(
                    "decoded image exceeds the maximum allowed pixel count",
                    code="image_too_large",
                )
            rgb = opened.convert("RGB")
            # PIL may expose a read-only view.  Own a writable request-local
            # buffer so downstream tensor adapters do not emit warnings that
            # disclose package paths into the operator log.
            return np.array(rgb, dtype=np.uint8, copy=True)
    except ServiceError:
        raise
    except UnidentifiedImageError as exc:
        raise ServiceError(
            "uploaded image could not be identified as JPEG, PNG or WebP",
            code="invalid_image",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ServiceError(
            "uploaded image is corrupt or malformed",
            code="invalid_image",
        ) from exc
