"""Geometry helpers for post-processing mask outputs.

This module provides optional Canny/Hough-based line detection utilities for
ZAP-IT runs that enable geometry analysis. The helpers may be imported directly
from :mod:`modules.geometry.geometry` or via the package namespace.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

GeometryLine = Tuple[int, int, int, int]
GeometryIntersection = Tuple[float, float]


def apply_geometry_on_mask(
    mask_bool: np.ndarray,
    geometry_cfg: dict,
    mask_index: int,
    out_dir: str,
    base_name: str,
    orig_shape: Sequence[int],
    verbosity: int = 1,
) -> Tuple[List[GeometryLine], List[GeometryIntersection]]:
    """Run the geometry pipeline over a single binary mask."""

    del orig_shape  # Currently unused but retained for API compatibility.

    height, width = mask_bool.shape[:2]

    debug = bool(geometry_cfg.get("debug", False))
    thr1 = geometry_cfg.get("canny_threshold1", 50)
    thr2 = geometry_cfg.get("canny_threshold2", 150)
    aperture = int(geometry_cfg.get("canny_aperture", 3))

    rho = float(geometry_cfg.get("hough_rho", 1.0))
    theta_deg = float(geometry_cfg.get("hough_theta", 1.0))
    hough_thr = int(geometry_cfg.get("hough_threshold", 30))
    h_min_len = int(geometry_cfg.get("hough_min_line_length", 20))
    h_max_gap = int(geometry_cfg.get("hough_max_line_gap", 10))

    mask_u8 = mask_bool.astype(np.uint8) * 255

    if debug and verbosity >= 1:
        print(
            "[geometry] => applying canny on mask %s, shape=(%s,%s), thr=(%s,%s), "
            "aperture=%s" % (mask_index, height, width, thr1, thr2, aperture)
        )

    edges = cv2.Canny(mask_u8, threshold1=thr1, threshold2=thr2, apertureSize=aperture)
    edge_nonzero = int(np.count_nonzero(edges))

    if debug and verbosity >= 2:
        canny_file = f"{base_name}_mask{mask_index}_canny.png"
        canny_path = os.path.join(out_dir, canny_file)
        cv2.imwrite(canny_path, edges)
        print(f"[geometry debug] => wrote canny image => {canny_path} (nonzero={edge_nonzero})")

    theta_rad = np.deg2rad(theta_deg)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta_rad,
        threshold=hough_thr,
        minLineLength=h_min_len,
        maxLineGap=h_max_gap,
    )

    lines_data: List[GeometryLine] = []
    if lines_p is not None:
        for ln in lines_p:
            x1, y1, x2, y2 = ln[0]
            lines_data.append((int(x1), int(y1), int(x2), int(y2)))

    if debug and verbosity >= 2:
        if lines_data:
            print(f"[geometry debug] => found {len(lines_data)} lines for mask={mask_index}")
        else:
            print(f"[geometry debug] => no lines found for mask={mask_index}")

    intersections: List[GeometryIntersection] = []
    for i in range(len(lines_data)):
        x1a, y1a, x2a, y2a = lines_data[i]
        for j in range(i + 1, len(lines_data)):
            x1b, y1b, x2b, y2b = lines_data[j]
            point = line_intersection(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
            if point is not None:
                ix, iy = point
                if 0 <= ix < width and 0 <= iy < height:
                    intersections.append((float(ix), float(iy)))

    lines_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_lines.tsv")
    with open(lines_tsv, "w", encoding="utf-8") as line_file:
        line_file.write("x1\ty1\tx2\ty2\n")
        for xa, ya, xb, yb in lines_data:
            line_file.write(f"{xa}\t{ya}\t{xb}\t{yb}\n")

    inters_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_intersections.tsv")
    with open(inters_tsv, "w", encoding="utf-8") as intersection_file:
        intersection_file.write("ix\tiy\n")
        for ix, iy in intersections:
            intersection_file.write(f"{ix}\t{iy}\n")

    return lines_data, intersections


def draw_geometry_on_image(
    image_arr: np.ndarray,
    lines_data: Iterable[GeometryLine],
    intersections: Iterable[GeometryIntersection],
    geometry_cfg: dict,
    circle_radius_frac: float = 0.01,
) -> np.ndarray:
    """Overlay detected geometry on an RGB image array."""

    del geometry_cfg  # Reserved for future use; kept for API compatibility.

    height, width = image_arr.shape[:2]
    diag_len = float(np.sqrt(height * height + width * width))
    base_radius = int(diag_len * circle_radius_frac)
    if base_radius < 2:
        base_radius = 2

    bgr = image_arr[..., ::-1].copy()

    for x1, y1, x2, y2 in lines_data:
        cv2.line(
            bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    for ix, iy in intersections:
        center = (int(ix), int(iy))
        cv2.circle(bgr, center, base_radius + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, center, base_radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        if base_radius >= 2:
            cv2.circle(bgr, center, base_radius - 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    image_arr[...] = bgr[..., ::-1]
    return image_arr


def line_intersection(
    x1a: float,
    y1a: float,
    x2a: float,
    y2a: float,
    x1b: float,
    y1b: float,
    x2b: float,
    y2b: float,
) -> GeometryIntersection | None:
    """Return the intersection point of two line segments, if it exists."""

    a1 = float(y2a - y1a)
    b1 = float(x1a - x2a)
    c1 = a1 * x1a + b1 * y1a

    a2 = float(y2b - y1b)
    b2 = float(x1b - x2b)
    c2 = a2 * x1b + b2 * y1b

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None

    ix = (b2 * c1 - b1 * c2) / det
    iy = (a1 * c2 - a2 * c1) / det

    if not is_between(ix, x1a, x2a) or not is_between(iy, y1a, y2a):
        return None
    if not is_between(ix, x1b, x2b) or not is_between(iy, y1b, y2b):
        return None

    return ix, iy


def is_between(val: float, end1: float, end2: float) -> bool:
    """Return ``True`` if ``val`` lies within the closed interval defined by ``end1`` and ``end2``."""

    lower = min(end1, end2) - 1e-9
    upper = max(end1, end2) + 1e-9
    return lower <= val <= upper
