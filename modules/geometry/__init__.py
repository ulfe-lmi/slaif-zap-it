"""Geometry package exposing optional Canny/Hough helpers."""

from .geometry import (
    GeometryIntersection,
    GeometryLine,
    apply_geometry_on_mask,
    draw_geometry_on_image,
    is_between,
    line_intersection,
)

__all__ = [
    "GeometryIntersection",
    "GeometryLine",
    "apply_geometry_on_mask",
    "draw_geometry_on_image",
    "is_between",
    "line_intersection",
]
