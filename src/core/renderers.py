"""Pure deterministic renderers: YOLO text and the uint16 identity mask.

Both renderers consume the ordered :class:`~src.core.results.ObjectResult`
sequence produced by the engine, so object records, YOLO lines and identity
values share one ordering definition by construction.
"""

from __future__ import annotations

import io
from collections import deque
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
    "format_yolo_line",
    "render_yolo",
    "render_identity_png",
]

#: Fixed decimal places per YOLO coordinate field (exporter-compatible).
YOLO_DECIMALS = 6

#: Highest representable instance id in a uint16 identity mask (0=background).
MAX_IDENTITY_OBJECTS = 65535


def _representative_assignment(
    objects: Sequence[ObjectResult], canvas: np.ndarray
) -> dict[int, tuple[int, int]]:
    """Return a deterministic minimum-override matching for service IDs.

    Every object is connected to each true source pixel in its retained mask.
    A zero-cost edge is an already visible winner pixel; all other edges are
    overrides.  Successive shortest augmenting paths therefore find the
    minimum number of canvas changes, with candidate order breaking ties by
    visible-winner preference and then row-major position.
    """
    ordered = sorted(objects, key=lambda item: item.instance_id)
    candidates_by_object: list[list[tuple[int, int]]] = []
    all_pixels: set[tuple[int, int]] = set()
    for obj in ordered:
        rows, cols = np.nonzero(obj.mask)
        candidates = [(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())]
        if not candidates:
            raise IdentityMaskProjectionError(
                "identity mask cannot preserve a distinct source pixel for every object ID"
            )
        candidates_by_object.append(candidates)
        all_pixels.update(candidates)

    pixels = sorted(all_pixels)
    pixel_index = {pixel: index for index, pixel in enumerate(pixels)}
    object_count = len(ordered)
    source = 0
    object_offset = 1
    pixel_offset = object_offset + object_count
    sink = pixel_offset + len(pixels)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, cost: int) -> list[int]:
        forward = [end, len(graph[end]), 1, cost]
        reverse = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    # The rank sum is bounded by object_count * len(pixels). Making one
    # override more expensive than that entire secondary range guarantees
    # that override minimization is always the primary objective.
    override_cost = object_count * max(1, len(pixels)) + 1
    edge_refs: list[tuple[int, tuple[int, int], list[int]]] = []
    for object_index, (obj, candidates) in enumerate(zip(ordered, candidates_by_object)):
        object_node = object_offset + object_index
        add_edge(source, object_node, 0)
        preferred = [pixel for pixel in candidates if int(canvas[pixel]) == int(obj.instance_id)]
        preferred_set = set(preferred)
        candidate_order = preferred + [pixel for pixel in candidates if pixel not in preferred_set]
        for rank, pixel in enumerate(candidate_order):
            owner = int(canvas[pixel])
            cost = rank if owner == int(obj.instance_id) else override_cost + rank
            edge = add_edge(object_node, pixel_offset + pixel_index[pixel], cost)
            edge_refs.append((object_index, pixel, edge))
    for pixel_index_value in range(len(pixels)):
        add_edge(pixel_offset + pixel_index_value, sink, 0)

    # Successive shortest augmenting paths are complete for this bounded
    # bipartite assignment. SPFA handles the negative reverse edges while
    # retaining stable graph insertion order for deterministic equal-cost ties.
    for _ in range(object_count):
        distances: list[int | None] = [None] * len(graph)
        previous: list[tuple[int, list[int]] | None] = [None] * len(graph)
        distances[source] = 0
        queue: deque[int] = deque([source])
        queued = {source}
        while queue:
            node = queue.popleft()
            queued.discard(node)
            distance = distances[node]
            assert distance is not None
            for edge in graph[node]:
                if edge[2] <= 0:
                    continue
                candidate_distance = distance + edge[3]
                if distances[edge[0]] is None or candidate_distance < distances[edge[0]]:
                    distances[edge[0]] = candidate_distance
                    previous[edge[0]] = (node, edge)
                    if edge[0] not in queued:
                        queue.append(edge[0])
                        queued.add(edge[0])
        if distances[sink] is None:
            raise IdentityMaskProjectionError(
                "identity mask cannot preserve a distinct source pixel for every object ID"
            )
        node = sink
        while node != source:
            prior = previous[node]
            assert prior is not None
            prior_node, edge = prior
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = prior_node

    assignment: dict[int, tuple[int, int]] = {}
    for object_index, pixel, edge in edge_refs:
        if edge[2] == 0:
            assignment[int(ordered[object_index].instance_id)] = pixel
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
        # The baseline winner policy can make objects disappear when they are
        # fully covered. Match every object to one distinct true source pixel;
        # applying the selected representatives is the minimum deterministic
        # set of projection overrides and never mutates source masks.
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
