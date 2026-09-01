"""Deterministic centroid-radial support for BLIP3 candidate views.

The implementation in this module is deliberately independent of RGB data,
model code, filesystem state, and service configuration.  It consumes one
complete boolean mask and produces crop-local radial support when requested by
the BLIP3 compositor.

Boundary samples are source-pixel centres.  Components use 8-connectivity and
the background uses 4-connectivity when deciding which boundary is external.
The ordered walk is clockwise, starts at the lexicographically smallest
``(y, x)`` sample, and reverses its direction only when that makes the next
sample lexicographically smaller.  Lines use a dominant-axis DDA with
half-up rounding of the minor coordinate; this is the fixed tie rule for the
all-octant Bresenham-equivalent rasterizer.  Inclusive quadrilaterals are
rasterized by Pillow's deterministic integer polygon primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .errors import CoreError

_FIXED_SCALE = 1_000_000
_EIGHT_NEIGHBOURS = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)
_DIRECTION_INDEX = {direction: index for index, direction in enumerate(_EIGHT_NEIGHBOURS)}


def _require_mask(
    mask: np.ndarray, source_shape: tuple[int, int], source_candidate_id: int
) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.dtype(bool)
        or mask.ndim != 2
        or mask.shape != source_shape
        or not np.any(mask)
    ):
        raise CoreError("centroid-radial geometry requires a non-empty source-shaped boolean mask")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise CoreError("source candidate ID must be a positive integer")


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise CoreError("centroid-radial geometry requires a non-empty mask")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _component_pixels(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components as global ``(x, y)`` pixel points."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []
    for flat_index in np.flatnonzero(mask):
        y0, x0 = divmod(int(flat_index), width)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(x0, y0)]
        points: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            points.append((x, y))
            for dx, dy in _EIGHT_NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        components.append(points)
    return components


def _outside_background(component: np.ndarray) -> np.ndarray:
    """Mark background reachable from the padded local border by 4-connectivity."""
    padded = np.pad(component, 1, mode="constant", constant_values=False)
    outside = np.zeros_like(padded, dtype=bool)
    outside[0, :] = outside[-1, :] = True
    outside[:, 0] = outside[:, -1] = True
    outside &= ~padded
    frontier = outside.copy()
    while np.any(frontier):
        expanded = np.zeros_like(frontier, dtype=bool)
        expanded[1:, :] |= frontier[:-1, :]
        expanded[:-1, :] |= frontier[1:, :]
        expanded[:, 1:] |= frontier[:, :-1]
        expanded[:, :-1] |= frontier[:, 1:]
        frontier = expanded & ~padded & ~outside
        outside |= frontier
    return outside[1:-1, 1:-1]


def _external_pixels(
    component_points: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return a component's external boundary mask and global local origin."""
    xs = np.fromiter((point[0] for point in component_points), dtype=np.intp)
    ys = np.fromiter((point[1] for point in component_points), dtype=np.intp)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    component = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    component[ys - y0, xs - x0] = True
    outside = _outside_background(component)
    padded_outside = np.pad(outside, 1, mode="constant", constant_values=True)
    boundary = component & (
        padded_outside[:-2, :-2]
        | padded_outside[:-2, 1:-1]
        | padded_outside[:-2, 2:]
        | padded_outside[1:-1, :-2]
        | padded_outside[1:-1, 2:]
        | padded_outside[2:, :-2]
        | padded_outside[2:, 1:-1]
        | padded_outside[2:, 2:]
    )
    return component, boundary, (x0, y0)


def _moore_walk(component: np.ndarray, boundary: np.ndarray) -> list[tuple[int, int]]:
    """Trace the external Moore walk, repairing self-touching branches by DFS."""
    boundary_points = [(int(x), int(y)) for y, x in zip(*np.nonzero(boundary))]
    if not boundary_points:
        return []
    start = min(boundary_points, key=lambda point: (point[1], point[0]))
    if len(boundary_points) == 1:
        return [start]

    def neighbour_index(point: tuple[int, int]) -> int:
        dx = point[0] - current[0]
        dy = point[1] - current[1]
        return _DIRECTION_INDEX.get((dx, dy), 6)

    current = start
    backtrack = (start[0] - 1, start[1])
    walk = [start]
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    max_steps = max(16, len(boundary_points) * 16)
    for _ in range(max_steps):
        state = (current, backtrack)
        if state in seen_states:
            break
        seen_states.add(state)
        index = neighbour_index(backtrack)
        next_point = None
        next_backtrack = None
        for offset in range(1, 9):
            candidate_index = (index + offset) % 8
            dx, dy = _EIGHT_NEIGHBOURS[candidate_index]
            candidate = (current[0] + dx, current[1] + dy)
            if (
                0 <= candidate[0] < component.shape[1]
                and 0 <= candidate[1] < component.shape[0]
                and component[candidate[1], candidate[0]]
            ):
                prev_dx, prev_dy = _EIGHT_NEIGHBOURS[(candidate_index - 1) % 8]
                next_point = candidate
                next_backtrack = (current[0] + prev_dx, current[1] + prev_dy)
                break
        if next_point is None or next_backtrack is None:
            break
        if next_point == start and len(walk) > 1:
            break
        walk.append(next_point)
        current, backtrack = next_point, next_backtrack

    # A thin or self-touching digital component can have external pixels that
    # the ordinary Moore stop pair does not visit.  A deterministic DFS tour of
    # the external-pixel graph adds only adjacent samples and keeps repetitions.
    boundary_set = set(boundary_points)
    if set(walk) != boundary_set:
        walk = []
        visited: set[tuple[int, int]] = set()

        def visit(point: tuple[int, int]) -> None:
            visited.add(point)
            walk.append(point)
            neighbours = []
            for dx, dy in _EIGHT_NEIGHBOURS:
                candidate = (point[0] + dx, point[1] + dy)
                if candidate in boundary_set:
                    neighbours.append(candidate)
            neighbours.sort(
                key=lambda item: (
                    _DIRECTION_INDEX[(item[0] - point[0], item[1] - point[1])],
                    item[1],
                    item[0],
                )
            )
            for candidate in neighbours:
                if candidate not in visited:
                    visit(candidate)
                    walk.append(point)

        visit(start)
        for point in sorted(boundary_set.difference(visited), key=lambda item: (item[1], item[0])):
            walk.append(point)
            visit(point)
    return walk


def _normalize_contour(walk: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    samples = list(walk)
    normalized: list[tuple[int, int]] = []
    for point in samples:
        if not normalized or point != normalized[-1]:
            normalized.append(point)
    if len(normalized) > 1 and normalized[-1] == normalized[0]:
        normalized.pop()
    if not normalized:
        return ()
    first_value = min((point[1], point[0]) for point in normalized)
    first_index = next(
        index for index, point in enumerate(normalized) if (point[1], point[0]) == first_value
    )
    rotated = normalized[first_index:] + normalized[:first_index]
    if len(rotated) > 2:
        reverse = [rotated[0], *reversed(rotated[1:])]
        if (reverse[1][1], reverse[1][0]) < (rotated[1][1], rotated[1][0]):
            rotated = reverse
    return tuple(rotated)


def _contours(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], ...]:
    contours: list[tuple[tuple[int, int], ...]] = []
    for component_points in _component_pixels(mask):
        component, boundary, (x0, y0) = _external_pixels(component_points)
        walk = _moore_walk(component, boundary)
        contours.append(_normalize_contour((x + x0, y + y0) for x, y in walk))
    contours.sort(
        key=lambda contour: (contour[0][1], contour[0][0]) if contour else (math.inf, math.inf)
    )
    return tuple(contour for contour in contours if contour)


def _half_up_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.intp)


def _rasterize_lines(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize all inclusive lines in one vectorized bounded batch."""
    starts = np.asarray(starts, dtype=np.intp)
    ends = np.asarray(ends, dtype=np.intp)
    if starts.size == 0:
        empty = np.empty((0, 0), dtype=np.intp)
        return empty, empty, np.empty((0, 0), dtype=bool)
    dx = ends[:, 0] - starts[:, 0]
    dy = ends[:, 1] - starts[:, 1]
    dominant_x = np.abs(dx) >= np.abs(dy)
    lengths = np.maximum(np.abs(dx), np.abs(dy))
    step_count = int(lengths.max()) + 1
    progress = np.arange(step_count, dtype=np.intp)[None, :]
    valid = progress <= lengths[:, None]
    safe_lengths = np.maximum(lengths, 1)
    x_progress = np.minimum(progress, np.abs(dx)[:, None])
    y_progress = np.minimum(progress, np.abs(dy)[:, None])
    x_major_minor = (2 * np.abs(dy)[:, None] * x_progress + safe_lengths[:, None]) // (
        2 * safe_lengths[:, None]
    )
    y_major_minor = (2 * np.abs(dx)[:, None] * y_progress + safe_lengths[:, None]) // (
        2 * safe_lengths[:, None]
    )
    x_values = np.where(
        dominant_x[:, None],
        starts[:, 0, None] + np.sign(dx)[:, None] * progress,
        starts[:, 0, None] + np.sign(dx)[:, None] * y_major_minor,
    )
    y_values = np.where(
        dominant_x[:, None],
        starts[:, 1, None] + np.sign(dy)[:, None] * x_major_minor,
        starts[:, 1, None] + np.sign(dy)[:, None] * progress,
    )
    return x_values, y_values, valid


def _line_positive_counts(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    xs, ys, valid = _rasterize_lines(starts, ends)
    if xs.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    safe_xs = np.clip(xs, 0, mask.shape[1] - 1)
    safe_ys = np.clip(ys, 0, mask.shape[0] - 1)
    return np.sum(mask[safe_ys, safe_xs] & valid, axis=1, dtype=np.intp)


@dataclass(frozen=True)
class CentroidRadialGeometry:
    """Prepared source-space rays and crop-local support builder."""

    source_shape_hw: tuple[int, int]
    source_candidate_id: int
    raw_bbox_xyxy_inclusive: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    contours: tuple[tuple[tuple[int, int], ...], ...]
    boundary_points: np.ndarray
    contour_ranges: tuple[tuple[int, int], ...]
    outward_units: np.ndarray
    raw_distances: np.ndarray
    bounded_distances: np.ndarray
    window_bbox_xyxy_exclusive: tuple[int, int, int, int]
    raw_mask_window: np.ndarray

    @property
    def external_boundary_pixel_count(self) -> int:
        return int(self.boundary_points.shape[0])

    def distances_for_scale(self, scale_q: int) -> np.ndarray:
        if type(scale_q) is not int or not 0 <= scale_q <= _FIXED_SCALE:
            raise CoreError("radial scale must be an integer millionth from 0 to 1000000")
        return (scale_q * self.bounded_distances // _FIXED_SCALE).astype(np.intp)

    def support_for_scale(self, scale_q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(support_window, endpoints, effective_distances)``."""
        effective = self.distances_for_scale(scale_q)
        points = self.boundary_points
        height, width = self.source_shape_hw
        endpoint_float = np.empty((points.shape[0], 2), dtype=np.float64)
        endpoint_float[:] = points
        endpoint_float += self.outward_units * effective[:, None]
        endpoint_float[:, 0] = np.clip(endpoint_float[:, 0], 0.0, float(width - 1))
        endpoint_float[:, 1] = np.clip(endpoint_float[:, 1], 0.0, float(height - 1))
        endpoints = _half_up_nonnegative(endpoint_float)
        endpoints[:, 0] = np.clip(endpoints[:, 0], 0, width - 1)
        endpoints[:, 1] = np.clip(endpoints[:, 1], 0, height - 1)

        wx0, wy0, wx1, wy1 = self.window_bbox_xyxy_exclusive
        support = self.raw_mask_window.copy()
        local_points = points - np.asarray((wx0, wy0), dtype=np.intp)
        local_endpoints = endpoints - np.asarray((wx0, wy0), dtype=np.intp)
        xs, ys, valid = _rasterize_lines(local_points, local_endpoints)
        if xs.shape[0]:
            safe_xs = np.clip(xs, 0, wx1 - wx0 - 1)
            safe_ys = np.clip(ys, 0, wy1 - wy0 - 1)
            support[safe_ys[valid], safe_xs[valid]] = True

        # Pillow's integer polygon fill is inclusive and handles concave
        # quadrilateral degeneracies consistently.  Every contour is closed
        # independently; no component-to-component bridge is introduced.
        canvas = Image.fromarray(support.astype(np.uint8), mode="L")
        draw = ImageDraw.Draw(canvas)
        for start, end in self.contour_ranges:
            count = end - start
            if count < 2:
                continue
            for index in range(start, end):
                next_index = start if index + 1 == end else index + 1
                polygon = [
                    tuple(int(value) for value in local_points[index]),
                    tuple(int(value) for value in local_points[next_index]),
                    tuple(int(value) for value in local_endpoints[next_index]),
                    tuple(int(value) for value in local_endpoints[index]),
                ]
                draw.polygon(polygon, fill=1)
        return np.asarray(canvas, dtype=bool), endpoints, effective


def build_centroid_radial_geometry(
    segmentation_mask: np.ndarray,
    source_shape_hw: tuple[int, int],
    source_candidate_id: int,
    config: Any,
) -> CentroidRadialGeometry:
    """Prepare the exact centroid-radial mask-chord geometry.

    The only source data inspected is the complete boolean mask.  Chord counts
    include every positive rasterized pixel, including positive runs after a
    zero-valued gap.  The returned arrays are immutable and all support work is
    restricted to the raw bbox expanded by the configured maximum context and
    contour margin.
    """
    source_shape = (int(source_shape_hw[0]), int(source_shape_hw[1]))
    _require_mask(segmentation_mask, source_shape, source_candidate_id)
    raw_bbox = _tight_bbox(segmentation_mask)
    raw_x0, raw_y0, raw_x1, raw_y1 = raw_bbox
    rows, cols = np.nonzero(segmentation_mask)
    count = float(rows.size)
    centroid = (
        float(np.sum(cols, dtype=np.float64) / count),
        float(np.sum(rows, dtype=np.float64) / count),
    )
    contours = _contours(segmentation_mask)
    points_list: list[tuple[int, int]] = []
    ranges: list[tuple[int, int]] = []
    for contour in contours:
        start = len(points_list)
        points_list.extend(contour)
        ranges.append((start, len(points_list)))
    points = np.asarray(points_list, dtype=np.intp).reshape((-1, 2))

    inward_ends = np.empty_like(points)
    outward_units = np.zeros((points.shape[0], 2), dtype=np.float64)
    cx, cy = centroid
    bbox_edges = np.asarray((raw_x0, raw_y0, raw_x1, raw_y1), dtype=np.float64)
    for index, (px, py) in enumerate(points):
        vector = np.asarray((cx - px, cy - py), dtype=np.float64)
        if vector[0] == 0.0 and vector[1] == 0.0:
            inward_ends[index] = (px, py)
            outward_units[index] = (1.0, 0.0)
            continue
        candidates: list[float] = []
        for coordinate, velocity, lower, upper in (
            (float(px), vector[0], bbox_edges[0], bbox_edges[2]),
            (float(py), vector[1], bbox_edges[1], bbox_edges[3]),
        ):
            if velocity > 0.0:
                candidates.append((upper - coordinate) / velocity)
            elif velocity < 0.0:
                candidates.append((lower - coordinate) / velocity)
        positive = [value for value in candidates if value >= 1.0]
        ray_length = min(positive) if positive else 1.0
        continuous = np.asarray((px, py), dtype=np.float64) + ray_length * vector
        continuous = np.clip(continuous, (raw_x0, raw_y0), (raw_x1, raw_y1))
        inward_ends[index] = _half_up_nonnegative(continuous)
        difference = np.asarray((float(px) - cx, float(py) - cy), dtype=np.float64)
        norm = float(np.linalg.norm(difference))
        outward_units[index] = difference / norm

    positive_counts = _line_positive_counts(segmentation_mask, points, inward_ends)
    raw_distances = np.asarray(
        [int(math.ceil(float(config.context_fraction) * int(value))) for value in positive_counts],
        dtype=np.intp,
    )
    bounded_distances = np.clip(
        raw_distances,
        int(config.min_context_pixels),
        int(config.max_context_pixels),
    ).astype(np.intp)
    maximum = int(np.max(bounded_distances, initial=0))
    contour_margin = int(getattr(config, "contour_max_pixels", 0) or 0)
    height, width = source_shape
    window_bbox = (
        max(0, raw_x0 - maximum - contour_margin - 1),
        max(0, raw_y0 - maximum - contour_margin - 1),
        min(width, raw_x1 + maximum + contour_margin + 2),
        min(height, raw_y1 + maximum + contour_margin + 2),
    )
    wx0, wy0, wx1, wy1 = window_bbox
    raw_window = np.ascontiguousarray(segmentation_mask[wy0:wy1, wx0:wx1].copy())
    for array in (points, outward_units, raw_distances, bounded_distances, raw_window):
        array.setflags(write=False)
    return CentroidRadialGeometry(
        source_shape_hw=source_shape,
        source_candidate_id=source_candidate_id,
        raw_bbox_xyxy_inclusive=raw_bbox,
        centroid_xy=centroid,
        contours=contours,
        boundary_points=points,
        contour_ranges=tuple(ranges),
        outward_units=outward_units,
        raw_distances=raw_distances,
        bounded_distances=bounded_distances,
        window_bbox_xyxy_exclusive=window_bbox,
        raw_mask_window=raw_window,
    )


__all__ = ["CentroidRadialGeometry", "build_centroid_radial_geometry"]
