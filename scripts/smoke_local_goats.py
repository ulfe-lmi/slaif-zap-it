#!/usr/bin/env python3
"""Opt-in local-only goat regression harness.

This command is deliberately separate from CI. It reads operator-held,
ignored academic inputs, derives an API-safe YAML mapping in memory, crops the
image in memory, and sends only the crop to a loopback service. stdout contains
sanitized aliases, digests, dimensions, statuses, timings and counts; it never
prints source YAML, labels, prompts, response bodies or fixture bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from PIL import Image

from smoke_local_service import (
    post_completion,
    run_level_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_A = REPO_ROOT / "demos/goats/goats1.jpg"
DEFAULT_IMAGE_B = REPO_ROOT / "demos/goats/goats2.jpg"
DEFAULT_CONFIG = REPO_ROOT / "configs/goats2.yaml"
ALLOWED_TOP_LEVEL = frozenset(
    {
        "alpha",
        "preprocessing",
        "mask_generator",
        "postsam2processing",
        "clip",
        "blip3",
        "visualization",
    }
)
DENIED_KEY_WORDS = frozenset(
    {
        "model",
        "revision",
        "panoptic",
        "device",
        "path",
        "input",
        "output",
        "export",
        "cache",
        "url",
        "network",
        "command",
        "code",
        "import",
        "service",
        "resource",
        "dtype",
        "tokenizer",
        "processor",
        "remote",
        "download",
        "credential",
        "secret",
    }
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_path(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{label} is absent, not a regular in-root file, or unsafe") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return resolved


def central_crop(image_bytes: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return a central 50% PNG without writing a derivative to disk."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        width, height = source.size
        box = (width // 4, height // 4, (3 * width) // 4, (3 * height) // 4)
        crop = source.convert("RGB").crop(box)
    output = io.BytesIO()
    crop.save(output, format="PNG")
    return output.getvalue(), (width, height), (box[2] - box[0], box[3] - box[1])


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if any(marker in value for marker in ("/", "\\", "://")) or value.startswith("~"):
            return None
        return value
    return None


def _safe_value(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(word in lowered for word in DENIED_KEY_WORDS):
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_name = str(child_key)
            sanitized = _safe_value(child_value, child_name)
            if sanitized is not None:
                result[child_name] = sanitized
        return result
    if isinstance(value, (list, tuple)):
        result = [_safe_value(item, key) for item in value]
        return [item for item in result if item is not None]
    return _safe_scalar(value)


def derive_api_safe_config(raw_yaml: bytes) -> tuple[bytes, int]:
    """Allowlist algorithm sections and strip operator/model controls."""
    loaded = yaml.safe_load(raw_yaml)
    if not isinstance(loaded, Mapping):
        raise ValueError("academic config must be a YAML mapping")
    derived: dict[str, Any] = {}
    stripped = 0
    for key, value in loaded.items():
        name = str(key)
        if name not in ALLOWED_TOP_LEVEL:
            stripped += 1
            continue
        if name == "mask_generator":
            # The qualified resident GPU profile fixes SAM2 generator
            # parameters. Keep the algorithm section present but empty so the
            # live adapter cannot mistake academic tuning for a request-level
            # runtime override.
            stripped += len(value) if isinstance(value, Mapping) else 1
            derived[name] = {}
            continue
        sanitized = _safe_value(value, name)
        if name == "visualization" and isinstance(sanitized, dict):
            # Academic panoptic visualization is not an API-safe renderer.
            sanitized.pop("blip3", None)
        if sanitized is not None:
            derived[name] = sanitized
        elif value is not None:
            stripped += 1
    # Keep this derivation useful even when the legacy config carries only
    # batch/visualization controls. The service applies its own safe defaults.
    if "alpha" not in derived:
        derived["alpha"] = 0.6
    encoded = yaml.safe_dump(derived, sort_keys=True, allow_unicode=True).encode("utf-8")
    return encoded, stripped


def _workspace_snapshot(root: Path) -> tuple[int, int]:
    if not root.exists():
        return (0, 0)
    count = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return count, total


def _run_sequence(
    host: str,
    port: int,
    image_bytes: bytes,
    config_bytes: bytes,
    api_key: str | None,
    alias: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for verbosity, response_format in ((2, "json"), (3, "json"), (3, "zip")):
        summary = run_level_case(
            host,
            port,
            verbosity=verbosity,
            response_format=response_format,
            fixture_png=image_bytes,
            config_bytes=config_bytes,
            api_key=api_key,
        )
        cases.append(
            {
                "alias": alias,
                "case": summary["case"],
                "passed": bool(summary["passed"]),
                "status": summary["http_status"],
                "latency_ms": summary["latency_ms"],
                "response_bytes": summary["response_bytes"],
                "object_count": summary.get("objects", summary.get("yolo_lines", 0)),
                "artifact_count": len(summary.get("identity_mask", {}))
                if isinstance(summary.get("identity_mask"), dict)
                else summary.get("zip_members", 0),
            }
        )
        if summary.get("detail"):
            cases[-1]["error_code"] = str(summary["detail"]).removeprefix("error code=")
    return cases


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def run_request_summary(
    host: str,
    port: int,
    *,
    image_bytes: bytes,
    config_bytes: bytes,
    api_key: str | None,
) -> tuple[int, int, dict[str, Any], dict[str, str]]:
    """Send one L3 JSON request and retain only non-content numeric summaries."""
    status, payload, meta = post_completion(
        host,
        port,
        image_bytes=image_bytes,
        config_bytes=config_bytes,
        verbosity=3,
        response_format="json",
        api_key=api_key,
        timeout_s=240.0,
    )
    if status != 200 or not isinstance(payload, dict):
        return status, 0, {}, meta
    service = payload.get("service") if isinstance(payload.get("service"), dict) else {}
    timings = service.get("timings_ms") if isinstance(service.get("timings_ms"), dict) else {}
    safe_timings = {
        str(key): float(value)
        for key, value in timings.items()
        if str(key).startswith("stage.") and isinstance(value, (int, float))
    }
    objects = service.get("objects")
    return status, len(objects) if isinstance(objects, list) else 0, safe_timings, meta


def run_benchmark(
    host: str,
    port: int,
    crop_a_bytes: bytes,
    crop_b_bytes: bytes,
    config_bytes: bytes,
    api_key: str | None,
) -> dict[str, Any]:
    """Run the exact ten-request A/B sequence with sanitized summaries only."""
    samples: list[dict[str, Any]] = []
    for index in range(10):
        alias = "A" if index % 2 == 0 else "B"
        image_bytes = crop_a_bytes if alias == "A" else crop_b_bytes
        status, object_count, timings, meta = run_request_summary(
            host,
            port,
            image_bytes=image_bytes,
            config_bytes=config_bytes,
            api_key=api_key,
        )
        samples.append(
            {
                "index": index + 1,
                "image": alias,
                "status": status,
                "latency_ms": float(meta.get("latency_ms", 0)),
                "objects": object_count,
                "stage_timings_ms": timings,
            }
        )
    by_image: dict[str, list[float]] = {"A": [], "B": []}
    for sample in samples:
        if sample["status"] == 200:
            by_image[sample["image"]].append(sample["latency_ms"])

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "first": round(values[0], 1),
            "minimum": round(min(values), 1),
            "median": round(ordered[len(ordered) // 2], 1),
            "p95_nearest_rank": round(_nearest_rank_p95(values), 1),
            "maximum": round(max(values), 1),
        }

    return {
        "status": "PASSED" if all(item["status"] == 200 for item in samples) else "FAILED",
        "request_order": [item["image"] for item in samples],
        "samples": samples,
        "statistics": {key: stats(value) for key, value in by_image.items()},
        "aggregate": stats([item["latency_ms"] for item in samples if item["status"] == 200]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--image-a", type=Path, default=DEFAULT_IMAGE_A)
    parser.add_argument("--image-b", type=Path, default=DEFAULT_IMAGE_B)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixture-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--api-key", default=os.environ.get("SLAIF_ZAP_IT_API_KEY"))
    parser.add_argument("--tmp-root", type=Path, default=Path("/dev/shm/slaif-zap-it"))
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run exactly ten real BLIP3-enabled central-crop requests A,B,A,B,A,B,A,B,A,B",
    )
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("the local academic harness only permits loopback")
    try:
        image_a_path = _safe_path(args.image_a, args.fixture_root, "image_a")
        image_b_path = _safe_path(args.image_b, args.fixture_root, "image_b")
        config_path = _safe_path(args.config, args.fixture_root, "config")
        image_a_bytes = image_a_path.read_bytes()
        image_b_bytes = image_b_path.read_bytes()
        raw_config = config_path.read_bytes()
        crop_a_bytes, original_a_size, crop_a_size = central_crop(image_a_bytes)
        crop_b_bytes, original_b_size, crop_b_size = central_crop(image_b_bytes)
        safe_config, stripped_count = derive_api_safe_config(raw_config)
        before = _workspace_snapshot(args.tmp_root)
        results = []
        benchmark = None
        if args.benchmark:
            benchmark = run_benchmark(
                args.host,
                args.port,
                crop_a_bytes,
                crop_b_bytes,
                safe_config,
                args.api_key,
            )
        else:
            # A/B/A: the two independently decoded academic crops, then crop A
            # again. Each state is exercised at L2 JSON, L3 JSON and L3 ZIP so
            # request state cannot be hidden by one response format.
            for alias, payload in (
                ("a1", crop_a_bytes),
                ("b", crop_b_bytes),
                ("a2", crop_a_bytes),
            ):
                results.extend(
                    _run_sequence(args.host, args.port, payload, safe_config, args.api_key, alias)
                )
        after = _workspace_snapshot(args.tmp_root)
    except (OSError, ValueError, yaml.YAMLError, AssertionError) as exc:
        print(json.dumps({"status": "FAILED", "error": type(exc).__name__}))
        return 1
    passed = (
        all(bool(item["passed"]) for item in results)
        and (benchmark is None or benchmark["status"] == "PASSED")
        and before == after
    )
    output = {
        "status": "PASSED" if passed else "FAILED",
        "fixture_aliases": ["a1", "b", "a2"],
        "image_a": {
            "original_dimensions": list(original_a_size),
            "crop_dimensions": list(crop_a_size),
            "image_sha256": _digest(image_a_bytes),
            "crop_sha256": _digest(crop_a_bytes),
        },
        "image_b": {
            "original_dimensions": list(original_b_size),
            "crop_dimensions": list(crop_b_size),
            "image_sha256": _digest(image_b_bytes),
            "crop_sha256": _digest(crop_b_bytes),
        },
        "config_sha256": _digest(safe_config),
        "stripped_field_count": stripped_count,
        "zero_persistence": before == after,
        "workspace_file_count": after[0],
        "workspace_bytes": after[1],
        "cases": results,
    }
    if benchmark is not None:
        output["benchmark"] = benchmark
        output["status"] = benchmark["status"] if passed else "FAILED"
    print(json.dumps(output, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
