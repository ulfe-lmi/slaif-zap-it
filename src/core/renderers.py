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

from .errors import IdentityMaskOverflowError, IdentityMaskProjectionError
from .results import ObjectResult

__all__ = [
    "YOLO_DECIMALS",
    "MAX_IDENTITY_OBJECTS",
    "IDENTITY_CANDIDATE_CHUNK_SIZE",
    "format_yolo_line",
    "render_yolo",
    "render_identity_png",
]

#: Fixed decimal places per YOLO coordinate field (exporter-compatible).
YOLO_DECIMALS = 6

#: Highest representable instance id in a uint16 identity mask (0=background).
MAX_IDENTITY_OBJECTS = 65535

#: Maximum number of candidate pixels materialized by one matching scan.
#: This is deliberately independent of image dimensions and mask area.
IDENTITY_CANDIDATE_CHUNK_SIZE = 65536


def _candidate_chunk(
    mask_flat: np.ndarray,
    canvas_flat: np.ndarray,
    instance_id: int,
    chunk_start: int,
    *,
    prefer_visible: bool,
) -> tuple[np.ndarray, int]:
    """Return one bounded, row-major candidate-index chunk.

    The returned values are indices relative to ``chunk_start``.  Keeping the
    offset relative avoids allocating a second array when translating a
    candidate to a canvas position.  No call materializes more than
    :data:`IDENTITY_CANDIDATE_CHUNK_SIZE` indices.
    """
    chunk_end = min(chunk_start + IDENTITY_CANDIDATE_CHUNK_SIZE, mask_flat.size)
    mask_values = mask_flat[chunk_start:chunk_end]
    visible = canvas_flat[chunk_start:chunk_end] == instance_id
    selected = mask_values & visible if prefer_visible else mask_values & ~visible
    return np.flatnonzero(selected), chunk_end


def _visible_representatives(
    ordered: Sequence[ObjectResult], canvas: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Find one row-major baseline pixel for each visible object ID.

    The scan is over the already-rendered canvas, not over source-mask
    candidates.  It uses fixed-size chunks and returns the current matching
    state plus whether every object was already visible.  The latter enables
    the no-matching fast path required for the common case.
    """
    object_ids = np.asarray([int(obj.instance_id) for obj in ordered], dtype=np.int64)
    assigned = np.full(len(ordered), -1, dtype=np.int64)
    canvas_flat = canvas.reshape(-1)
    for chunk_start in range(0, canvas_flat.size, IDENTITY_CANDIDATE_CHUNK_SIZE):
        chunk_end = min(chunk_start + IDENTITY_CANDIDATE_CHUNK_SIZE, canvas_flat.size)
        values = canvas_flat[chunk_start:chunk_end]
        positions = np.flatnonzero(np.isin(values, object_ids))
        for position in positions:
            object_index = int(np.searchsorted(object_ids, values[position]))
            if assigned[object_index] < 0:
                assigned[object_index] = chunk_start + int(position)
    return assigned, bool(np.all(assigned >= 0))


def _owner_index(assigned: np.ndarray, pixel_index: int) -> int:
    """Return the object currently holding ``pixel_index``, or ``-1``."""
    # At most one entry can match because assignments are injective.  A short
    # object-sized scan avoids a dict keyed by every source pixel.
    for object_index, assigned_pixel in enumerate(assigned):
        if int(assigned_pixel) == pixel_index:
            return object_index
    return -1


def _next_candidate(
    object_index: int,
    ordered: Sequence[ObjectResult],
    canvas_flat: np.ndarray,
    phases: np.ndarray,
    chunk_starts: np.ndarray,
    candidate_offsets: np.ndarray,
) -> int | None:
    """Advance one object's bounded deterministic candidate cursor."""
    mask_flat = ordered[object_index].mask.reshape(-1)
    instance_id = int(ordered[object_index].instance_id)
    while int(phases[object_index]) < 2:
        phase = int(phases[object_index])
        chunk_start = int(chunk_starts[object_index])
        if chunk_start >= mask_flat.size:
            if phase == 0:
                phases[object_index] = 1
                chunk_starts[object_index] = 0
                candidate_offsets[object_index] = 0
                continue
            phases[object_index] = 2
            return None
        candidates, chunk_end = _candidate_chunk(
            mask_flat,
            canvas_flat,
            instance_id,
            chunk_start,
            prefer_visible=phase == 0,
        )
        candidate_offset = int(candidate_offsets[object_index])
        if candidate_offset < candidates.size:
            candidate_offsets[object_index] = candidate_offset + 1
            return chunk_start + int(candidates[candidate_offset])
        chunk_starts[object_index] = chunk_end
        candidate_offsets[object_index] = 0
    return None


def _augment_assignment(
    ordered: Sequence[ObjectResult], canvas: np.ndarray, assigned: np.ndarray
) -> None:
    """Complete ``assigned`` with iterative deterministic augmenting paths.

    Only one object-sized assignment/cursor state and one bounded NumPy
    candidate chunk are retained.  The stack contains object indices, never
    source pixels or edge records.  Standard augmenting-path completeness then
    supplies a matching whenever one exists.
    """
    object_count = len(ordered)
    canvas_flat = canvas.reshape(-1)
    visited = np.zeros(object_count, dtype=np.int64)
    phases = np.zeros(object_count, dtype=np.uint8)
    chunk_starts = np.zeros(object_count, dtype=np.int64)
    candidate_offsets = np.zeros(object_count, dtype=np.int64)
    path_pixels = np.full(object_count, -1, dtype=np.int64)
    search_id = 0

    for root in range(object_count):
        if assigned[root] >= 0:
            continue
        search_id += 1
        stack = [root]
        visited[root] = search_id
        phases[root] = 0
        chunk_starts[root] = 0
        candidate_offsets[root] = 0
        while stack:
            current = stack[-1]
            pixel = _next_candidate(
                current,
                ordered,
                canvas_flat,
                phases,
                chunk_starts,
                candidate_offsets,
            )
            if pixel is None:
                stack.pop()
                continue
            owner = _owner_index(assigned, pixel)
            if owner == current or (owner >= 0 and visited[owner] == search_id):
                continue
            path_pixels[current] = pixel
            if owner < 0:
                assigned[stack[-1]] = pixel
                for parent_position in range(len(stack) - 2, -1, -1):
                    parent = stack[parent_position]
                    assigned[parent] = path_pixels[parent]
                break
            visited[owner] = search_id
            phases[owner] = 0
            chunk_starts[owner] = 0
            candidate_offsets[owner] = 0
            stack.append(owner)
        else:
            raise IdentityMaskProjectionError(
                "identity mask cannot preserve a distinct source pixel for every object ID"
            )


def _representative_assignment(
    objects: Sequence[ObjectResult], canvas: np.ndarray
) -> dict[int, tuple[int, int]]:
    """Return a deterministic complete matching for service IDs.

    Existing baseline representatives are seeded first.  Missing IDs are
    completed with iterative augmenting paths; each object's candidates are
    scanned in baseline-visible then row-major order through fixed chunks.
    This is complete without materializing per-pixel edge state, and its only
    Python state scales with the object count and augmenting-path stack.
    """
    ordered = sorted(objects, key=lambda item: item.instance_id)
    object_count = len(ordered)
    assigned, complete_baseline = _visible_representatives(ordered, canvas)
    if not complete_baseline:
        _augment_assignment(ordered, canvas, assigned)
    assignment: dict[int, tuple[int, int]] = {}
    width = canvas.shape[1]
    for object_index, pixel in enumerate(assigned):
        if int(pixel) < 0:
            raise IdentityMaskProjectionError(
                "identity mask cannot preserve a distinct source pixel for every object ID"
            )
        row, column = divmod(int(pixel), width)
        assignment[int(ordered[object_index].instance_id)] = (row, column)
    if len(assignment) != object_count:
        raise IdentityMaskProjectionError(
            "identity mask cannot preserve a distinct source pixel for every object ID"
        )
    return assignment


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
    - when ``ensure_all_ids`` is true, baseline-visible representatives are
      retained where possible and missing IDs are completed by deterministic
      augmenting paths, preserving the service contract that IDs 1..N remain
      bijective with the object list;
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
        # The baseline winner policy can make objects disappear when they are
        # fully covered. Match every object to one distinct true source pixel;
        # applying the selected representatives is deterministic and never
        # mutates source masks. Candidate scans are bounded independently of
        # image dimensions and mask area.
        assignment = _representative_assignment(objects, canvas)
        for obj in objects:
            canvas[assignment[int(obj.instance_id)]] = obj.instance_id
    if Image is None:  # pragma: no cover - broken install guard
        raise RuntimeError("Pillow is required to encode PNG artifacts.")
    buffer = io.BytesIO()
    # A 2-D uint16 array selects the lossless ``I;16`` mode automatically;
    # passing mode= explicitly is deprecated in Pillow >= 12.
    Image.fromarray(canvas).save(buffer, format="PNG")
    return buffer.getvalue()
