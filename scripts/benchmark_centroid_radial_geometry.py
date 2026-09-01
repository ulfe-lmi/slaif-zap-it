#!/usr/bin/env python3
"""Manual benchmark for the production centroid-radial geometry.

This is intentionally separate from the normal test suite: wall-clock timing
is reported, but ordinary CI does not depend on the host's CPU load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path
import sys
import time

import numpy as np
from PIL import __version__ as pillow_version

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import CandidateViewConfig, build_centroid_radial_geometry
from src.core.radial_geometry import _RAY_BATCH_SIZE


def _workload() -> list[np.ndarray]:
    """Build a deterministic 122-candidate mixed-shape qualification corpus."""
    masks: list[np.ndarray] = []
    yy, xx = np.indices((199, 199))

    def origin(center: int, extent: int) -> int:
        return max(0, min(199 - extent, center - extent // 2))

    for index in range(122):
        mask = np.zeros((199, 199), dtype=bool)
        center_x = 99 + (index % 5 - 2) * 3
        center_y = 99 + ((index // 5) % 5 - 2) * 3
        if index in (6, 67):
            kind = 6
        elif index in (7, 68):
            kind = 7
        else:
            kind = index % 6
        if kind == 0:  # horizontal elongated
            width, height = 80 + index % 5 * 8, 5 + index % 4
            x0, y0 = origin(center_x, width), origin(center_y, height)
            mask[y0 : y0 + height, x0 : x0 + width] = True
        elif kind == 1:  # vertical elongated
            width, height = 5 + index % 4, 80 + index % 5 * 8
            x0, y0 = origin(center_x, width), origin(center_y, height)
            mask[y0 : y0 + height, x0 : x0 + width] = True
        elif kind == 2:  # rotated elongated
            angle = np.deg2rad(17.0 + index % 5 * 11.0)
            dx, dy = xx - center_x, yy - center_y
            rotated_x = dx * np.cos(angle) + dy * np.sin(angle)
            rotated_y = -dx * np.sin(angle) + dy * np.cos(angle)
            mask = (rotated_x / 45.0) ** 2 + (rotated_y / 8.0) ** 2 <= 1.0
        elif kind == 3:  # concave L
            x0, y0 = origin(center_x, 54), origin(center_y, 54)
            mask[y0 : y0 + 54, x0 : x0 + 9] = True
            mask[y0 : y0 + 9, x0 : x0 + 54] = True
        elif kind == 4:  # disconnected fragmented blocks
            x0, y0 = origin(center_x, 112), origin(center_y, 82)
            for block in range(5):
                bx = x0 + block * 23
                by = y0 + (block % 2) * 41
                mask[by : by + 13, bx : bx + 15] = True
        elif kind == 5:  # centroid-in-gap pair
            x0, y0 = origin(center_x, 150), origin(center_y, 24)
            mask[y0 : y0 + 24, x0 : x0 + 30] = True
            mask[y0 : y0 + 24, x0 + 120 : x0 + 150] = True
        elif kind == 6:  # high-boundary, many local components
            x0, y0 = origin(center_x, 180), origin(center_y, 180)
            for row in range(7):
                for column in range(7):
                    by, bx = y0 + row * 30, x0 + column * 30
                    mask[by : by + 6, bx : bx + 6] = True
        else:  # a large mask with an interior hole
            x0, y0 = origin(center_x, 118), origin(center_y, 118)
            mask[y0 : y0 + 118, x0 : x0 + 118] = True
            mask[y0 + 35 : y0 + 83, x0 + 35 : x0 + 83] = False
        masks.append(mask)
    return masks


def _run(
    masks: list[np.ndarray], config: CandidateViewConfig
) -> tuple[float, list[float], str, int, int]:
    started = time.perf_counter_ns()
    candidate_times: list[float] = []
    digest = hashlib.sha256()
    boundaries = 0
    support_pixels = 0
    for source_id, mask in enumerate(masks, start=1):
        candidate_started = time.perf_counter_ns()
        geometry = build_centroid_radial_geometry(mask, mask.shape, source_id, config)
        support, endpoints, distances = geometry.support_for_scale(1_000_000)
        elapsed_ms = (time.perf_counter_ns() - candidate_started) / 1_000_000.0
        candidate_times.append(elapsed_ms)
        boundaries += geometry.external_boundary_pixel_count
        support_pixels += int(np.count_nonzero(support))
        digest.update(support.tobytes())
        digest.update(endpoints.tobytes())
        digest.update(distances.tobytes())
    total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return total_ms, candidate_times, digest.hexdigest(), boundaries, support_pixels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threshold-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.threshold_seconds <= 0:
        parser.error("repeat must be positive, warmup non-negative, threshold positive")

    masks = _workload()
    config = CandidateViewConfig.from_mapping(
        {
            "context_fraction": 0.20,
            "min_context_pixels": 0,
            "max_context_pixels": 64,
            "crop_extent_multiplier": 2.0,
            "blur_sigma_fraction": 0.15,
            "contour_enabled": True,
            "contour_fraction": 0.02,
            "contour_min_pixels": 1,
            "contour_max_pixels": 3,
            "infeasible_geometry_policy": "centroid_radial_mask_chord",
        },
        stage="blip3",
    )
    for _ in range(args.warmup):
        _run(masks, config)
    measurements = [_run(masks, config) for _ in range(args.repeat)]
    digests = {result[2] for result in measurements}
    totals = [result[0] for result in measurements]
    candidate_times = measurements[0][1]
    threshold_ms = args.threshold_seconds * 1000.0
    median_total = statistics.median(totals)
    status = "PASSED" if len(digests) == 1 and median_total < threshold_ms else "FAILED"
    result = {
        "status": status,
        "candidate_count": len(masks),
        "source_shape": [199, 199],
        "warmup_count": args.warmup,
        "repeat_count": args.repeat,
        "threshold_seconds": args.threshold_seconds,
        "total_geometry_ms": {
            "minimum": min(totals),
            "median": median_total,
            "maximum": max(totals),
        },
        "repeated_total_geometry_ms": totals,
        "per_candidate_ms": {
            "minimum": min(candidate_times),
            "median": statistics.median(candidate_times),
            "maximum": max(candidate_times),
        },
        "boundary_samples_total": measurements[0][3],
        "support_pixels_total": measurements[0][4],
        "result_digest": next(iter(digests)) if len(digests) == 1 else sorted(digests),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "cpu": platform.processor() or platform.machine(),
        "core_count": os.cpu_count(),
        "platform": platform.platform(),
        "host": platform.node(),
        "ray_batch_size": _RAY_BATCH_SIZE,
        "qualification_judgement": "median total below threshold; maximum disclosed",
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
