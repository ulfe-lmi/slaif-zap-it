"""Pure deterministic renderers: YOLO text and the uint16 identity mask.

Both renderers consume the ordered :class:`~src.core.results.ObjectResult`
sequence produced by the engine, so object records, YOLO lines and identity
values share one ordering definition by construction.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is a required dependency
    Image = None  # type: ignore[assignment]

from .errors import CoreError, IdentityMaskOverflowError
from .results import ObjectResult

__all__ = [
    "YOLO_DECIMALS",
    "MAX_IDENTITY_OBJECTS",
    "format_yolo_line",
    "render_yolo",
    "render_identity_png",
]

#: Fixed decimal places per YOLO coordinate field (exporter-compatible).
YOLO_DECIMALS = 6

#: Highest representable instance id in a uint16 identity mask (0=background).
MAX_IDENTITY_OBJECTS = 65535


def format_yolo_line(class_id: int, cx: float, cy: float, bw: float, bh: float) -> str:
    """Format one five-field YOLO line with fixed precision."""
    return (
        f"{class_id} "
        f"{cx:.{YOLO_DECIMALS}f} {cy:.{YOLO_DECIMALS}f} "
        f"{bw:.{YOLO_DECIMALS}f} {bh:.{YOLO_DECIMALS}f}"
    )


def render_yolo(objects: Sequence[ObjectResult], *, image_width: int, image_height: int) -> str:
    """Render deterministic YOLO text normalized to ORIGINAL image dimensions.

    Every final object receives exactly one line in final-object order; empty
    detections produce an empty string. Class ids must already be assigned on
    the objects (the engine does this from the effective class mapping);
    objects without a mapping fall back to class id ``0`` at assignment time,
    never here.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    chunks = []
    for obj in objects:
        cx, cy, bw, bh = obj.normalized_bbox(image_width, image_height)
        class_id = obj.class_id if obj.class_id is not None else 0
        chunks.append(format_yolo_line(class_id, cx, cy, bw, bh))
        chunks.append("\n")
    return "".join(chunks)


def render_identity_png(
    objects: Sequence[ObjectResult],
    *,
    width: int,
    height: int,
    ensure_all_ids: bool = False,
) -> bytes:
    """Render the lossless uint16 identity mask PNG for ``objects``.

    Contract:

    - output dimensions are exactly ``(height, width)`` of the original image;
    - dtype is unsigned 16-bit grayscale; background pixels are ``0``;
    - each object paints its complete source mask with its ``instance_id``
      (disconnected components therefore share one ID);
    - contested (overlapping) pixels are won by the larger-area object; exact
      area ties are won by the smaller instance ID;
    - when ``ensure_all_ids`` is true, a deterministic row-major source pixel
      is reserved for any otherwise fully occluded object, preserving the
      service contract that IDs 1..N remain bijective with the object list;
    - more than :data:`MAX_IDENTITY_OBJECTS` objects raise
      :class:`IdentityMaskOverflowError` before any pixel buffer allocation;
    - encoding is deterministic for equal inputs within one environment.

    The single-valued projection intentionally loses overlap facts; callers
    needing overlap truth read ``ObjectResult.mask`` directly.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    count = len(objects)
    if count > MAX_IDENTITY_OBJECTS:
        raise IdentityMaskOverflowError(
            f"{count} objects exceed the uint16 identity-mask limit of "
            f"{MAX_IDENTITY_OBJECTS} instance ids"
        )
    canvas = np.zeros((height, width), dtype=np.uint16)
    # Paint small areas first so larger areas overwrite them; ties paint the
    # higher instance id first so the lower id wins contested pixels.
    paint_order = sorted(objects, key=lambda obj: (obj.area_px, -obj.instance_id))
    for obj in paint_order:
        canvas[obj.mask] = obj.instance_id
    if ensure_all_ids and objects:
        # The baseline winner policy can make a smaller object disappear when
        # it is fully covered by a larger mask.  Preserve the source masks and
        # reserve one distinct source pixel for each missing ID.  Do not steal
        # the last visible pixel of another object; if no such projection
        # exists, fail closed instead of returning a misleading artifact.
        visible_counts = {
            int(instance_id): int(np.count_nonzero(canvas == instance_id))
            for instance_id in (obj.instance_id for obj in objects)
        }
        reserved_pixels: set[tuple[int, int]] = set()
        for obj in sorted(objects, key=lambda item: item.instance_id):
            if visible_counts.get(obj.instance_id, 0) > 0:
                continue
            rows, cols = np.nonzero(obj.mask)
            selected: tuple[int, int] | None = None
            for row, col in zip(rows.tolist(), cols.tolist()):
                pixel = (int(row), int(col))
                if pixel in reserved_pixels:
                    continue
                owner = int(canvas[pixel])
                if owner and visible_counts.get(owner, 0) <= 1:
                    continue
                selected = pixel
                break
            if selected is None:
                raise CoreError(
                    "identity mask cannot preserve a distinct pixel for every object ID"
                )
            owner = int(canvas[selected])
            if owner:
                visible_counts[owner] -= 1
            canvas[selected] = obj.instance_id
            visible_counts[obj.instance_id] = visible_counts.get(obj.instance_id, 0) + 1
            reserved_pixels.add(selected)
    if Image is None:  # pragma: no cover - broken install guard
        raise RuntimeError("Pillow is required to encode PNG artifacts.")
    buffer = io.BytesIO()
    # A 2-D uint16 array selects the lossless ``I;16`` mode automatically;
    # passing mode= explicitly is deprecated in Pillow >= 12.
    Image.fromarray(canvas).save(buffer, format="PNG")
    return buffer.getvalue()
