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
from pathlib import Path
import sys
import statistics
import time

import numpy as np
from PIL import __version__ as pillow_version

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import CandidateViewConfig, build_centroid_radial_geometry


def _workload() -> list[np.ndarray]:
    """Build 122 bounded, deterministic non-empty candidate masks."""
    masks: list[np.ndarray] = []
    for index in range(122):
        mask = np.zeros((199, 199), dtype=bool)
        x0 = 8 + (index * 17) % 150
        y0 = 8 + (index * 23) % 156
        width = 12 + (index % 20)
        height = 10 + ((index * 7) % 20)
        if index % 3 == 0:
            mask[y0 : y0 + height, x0 : x0 + width] = True
        elif index % 3 == 1:
            mask[y0 : y0 + height // 2, x0 : x0 + width] = True
            mask[y0 + height // 2 : y0 + height, x0 : x0 + width // 3] = True
        else:
            mask[y0 : y0 + height // 3, x0 : x0 + width] = True
            mask[y0 + height // 3 : y0 + height, x0 + width // 2 : x0 + width] = True
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
    status = "PASSED" if len(digests) == 1 and min(totals) < threshold_ms else "FAILED"
    result = {
        "status": status,
        "candidate_count": len(masks),
        "source_shape": [199, 199],
        "warmup_count": args.warmup,
        "repeat_count": args.repeat,
        "threshold_seconds": args.threshold_seconds,
        "total_geometry_ms": {
            "minimum": min(totals),
            "median": statistics.median(totals),
            "maximum": max(totals),
        },
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
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
