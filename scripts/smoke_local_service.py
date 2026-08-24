#!/usr/bin/env python3
"""Bounded E2E smoke tool for the loopback ZAP-IT service.

Runs real HTTP requests against a running local service and checks CPU-side
response invariants only (status codes, envelope shape, YOLO text grammar,
uint16 identity-mask properties, object/ID bijection, ZIP members). It uses a
synthetic in-memory fixture by default; a custom image path may be supplied.
The tool never prints raw image/config/response payloads, only sanitized
facts (statuses, sizes, counts, hashes, timings).

Examples:
    python scripts/smoke_local_service.py --port 17891
    python scripts/smoke_local_service.py --port 17891 --levels 0 1 2 3 \
        --formats json zip
    python scripts/smoke_local_service.py --port 17891 --busy
    python scripts/smoke_local_service.py --port 17891 --repeat 5
    python scripts/smoke_local_service.py --port 17891 --invalid
    python scripts/smoke_local_service.py --port 17891 --failure
    python scripts/smoke_local_service.py --port 17891 --timeout
    python scripts/smoke_local_service.py --port 17891 --cancel
    python scripts/smoke_local_service.py --port 17891 --response-too-large

Exit code 0 means every requested case passed its invariants honestly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
import uuid as uuid_module
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

FIXTURE_SIZE = 128


def make_fixture_png(size: int = FIXTURE_SIZE) -> bytes:
    """Deterministic small RGB fixture with three solid color regions."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    half = size // 2
    quarter_a, quarter_b = size // 4, (size * 3) // 4
    for y in range(size):
        for x in range(size):
            if y < half and x < half:
                pixels[x, y] = (220, 40, 30)
            elif y >= half and x >= half:
                pixels[x, y] = (30, 180, 50)
            elif quarter_a <= y < quarter_b and quarter_a <= x < quarter_b:
                pixels[x, y] = (30, 50, 220)
            else:
                pixels[x, y] = (24, 24, 24)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


SAFE_CONFIG_YAML = b"""
alpha: 0.6
preprocessing:
  # resize is a multiplicative factor; keep the 128x128 fixture at native size.
  resize: 1.0
mask_generator: {}
postsam2processing:
  maxsize: 100000
clip:
  labels:
    red: "a red object"
    green: "a green object"
visualization:
  alpha: 0.6
"""

BLIP3_CONFIG_YAML = b"""
blip3:
  object:
    question: "What is in this region?"
    trueresult: "yes"
    falseresult: "no"
"""

INVALID_CONFIG_YAML = b"[unterminated\n"


def build_multipart(
    *,
    image_bytes: bytes,
    config_bytes: bytes,
    verbosity: int,
    response_format: str,
    boundary: str,
) -> tuple[bytes, str]:
    parts: list[bytes] = []

    def field(name: str, value: bytes, filename: str | None, content_type: str) -> None:
        headers = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            headers += f'; filename="{filename}"'
        headers += f"\r\nContent-Type: {content_type}\r\n\r\n"
        parts.append(f"--{boundary}\r\n{headers}".encode() + value + b"\r\n")

    field("image", image_bytes, "fixture.png", "image/png")
    field("config", config_bytes, "config.yaml", "application/yaml")
    field("verbosity", str(verbosity).encode(), None, "text/plain")
    field("response_format", response_format.encode(), None, "text/plain")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def post_completion(
    host: str,
    port: int,
    *,
    image_bytes: bytes,
    config_bytes: bytes,
    verbosity: int,
    response_format: str,
    timeout_s: float = 150.0,
    api_key: str | None = None,
) -> tuple[int, dict[str, Any] | bytes, dict[str, str]]:
    boundary = uuid_module.uuid4().hex
    body, content_type = build_multipart(
        image_bytes=image_bytes,
        config_bytes=config_bytes,
        verbosity=verbosity,
        response_format=response_format,
        boundary=boundary,
    )
    headers = {"Content-Type": content_type}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            if response.headers.get("Content-Type", "").startswith("application/json"):
                document = json.loads(payload.decode("utf-8"))
            else:
                document = payload
            return (
                response.status,
                document,
                {
                    "latency_ms": str(elapsed_ms),
                    "bytes": str(len(payload)),
                    **{
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower().startswith("retry-after")
                    },
                },
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = {"raw_error_bytes": len(payload)}
        return (
            exc.code,
            document,
            {
                "latency_ms": str(elapsed_ms),
                "bytes": str(len(payload)),
                **{
                    key.lower(): value
                    for key, value in exc.headers.items()
                    if key.lower().startswith("retry-after")
                },
            },
        )


def parse_yolo_lines(text: str) -> list[tuple[int, float, float, float, float]]:
    parsed: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        assert len(fields) == 5, f"YOLO line must have 5 fields, got {len(fields)}"
        class_id = int(fields[0])
        values = [float(item) for item in fields[1:]]
        for value in values:
            assert 0.0 <= value <= 1.0, "normalized coordinates must be within [0, 1]"
        parsed.append((class_id, *values))
    return parsed


def check_identity_png(
    png_bytes: bytes, width: int, height: int, expected_ids: int
) -> dict[str, Any]:
    import numpy as np

    array = np.array(Image.open(io.BytesIO(png_bytes)))
    assert array.dtype == np.uint16, f"identity mask must be uint16, got {array.dtype}"
    assert array.shape == (height, width), (
        f"identity mask dims {array.shape} != image {(height, width)}"
    )
    unique_ids = sorted(int(value) for value in np.unique(array) if value != 0)
    assert unique_ids == list(range(1, expected_ids + 1)), (
        f"IDs {unique_ids[:8]}... not bijective with 1..{expected_ids}"
    )
    return {"dtype": str(array.dtype), "dims": list(array.shape), "object_count": expected_ids}


def extract_identity_png(document: dict[str, Any]) -> bytes:
    artifacts = (document.get("service") or {}).get("artifacts") or []
    entry = next(
        (item for item in artifacts if item.get("name") == "identity-mask.png"),
        None,
    )
    assert entry is not None, "L>=1 responses must include identity-mask.png"
    assert entry.get("media_type") == "image/png"
    assert entry.get("encoding") == "base64"
    assert isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64
    try:
        payload = base64.b64decode(entry["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise AssertionError("identity mask is not valid base64") from exc
    assert len(payload) == int(entry.get("size", -1)), "identity mask size hash is inconsistent"
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"], (
        "identity mask hash is inconsistent"
    )
    return payload


def error_code(payload: object) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str):
                return code
    return "?"


def case_summary(
    case: str,
    status: int,
    meta: dict[str, str],
    extra: dict[str, Any] | None = None,
    *,
    passed: bool,
    detail: str = "",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case": case,
        "http_status": status,
        "passed": bool(passed),
        "latency_ms": float(meta.get("latency_ms", 0)),
        "response_bytes": int(meta.get("bytes", 0)),
    }
    if extra:
        summary.update(extra)
    if detail:
        summary["detail"] = detail
    return summary


def run_level_case(
    host: str,
    port: int,
    *,
    verbosity: int,
    response_format: str,
    fixture_png: bytes,
    config_bytes: bytes = SAFE_CONFIG_YAML,
    api_key: str | None = None,
) -> dict[str, Any]:
    case = f"L{verbosity}_{response_format}"
    status, payload, meta = post_completion(
        host,
        port,
        image_bytes=fixture_png,
        config_bytes=config_bytes,
        verbosity=verbosity,
        response_format=response_format,
        api_key=api_key,
    )
    if status != 200:
        code = payload.get("error", {}).get("code") if isinstance(payload, dict) else "?"
        return case_summary(case, status, meta, passed=False, detail=f"error code={code}")
    width, height = Image.open(io.BytesIO(fixture_png)).size
    if response_format == "zip":
        assert isinstance(payload, bytes)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names and "detections.yolo.txt" in names, (
                f"ZIP missing core members: {sorted(names)[:6]}"
            )
            manifest = json.loads(archive.read("manifest.json"))
            yolo_text = archive.read("detections.yolo.txt").decode("utf-8")
            png_members = [name for name in names if name.endswith(".png")]
            identity_name = next(
                (name for name in png_members if "identity" in Path(name).name),
                None,
            )
            png_info: dict[str, Any] | None = None
            if verbosity >= 1:
                assert identity_name is not None, "ZIP must carry identity mask at L>=1"
                png_info = check_identity_png(
                    archive.read(identity_name),
                    width,
                    height,
                    len(parse_yolo_lines(yolo_text)),
                )
            else:
                assert not png_members, "L0 ZIP must not carry image artifacts"
            manifest_artifacts = (manifest.get("service") or {}).get("artifacts") or []
            assert all("data" not in item for item in manifest_artifacts)
        yolo_hash = hashlib.sha256(yolo_text.encode()).hexdigest()[:12]
        return case_summary(
            case,
            status,
            meta,
            {
                "zip_members": len(names),
                "yolo_lines": len(parse_yolo_lines(yolo_text)),
                "yolo_sha256_12": yolo_hash,
                "manifest_keys": sorted(manifest.keys())[:8],
                "identity_mask": png_info,
            },
            passed=True,
        )

    document = payload
    assert isinstance(document, dict)
    choice = document["choices"][0]
    service_meta = document.get("service") or {}
    yolo_text = choice["text"]
    yolo_lines = parse_yolo_lines(yolo_text)
    extra: dict[str, Any] = {
        "model": document.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "yolo_lines": len(yolo_lines),
        "yolo_sha256_12": hashlib.sha256(yolo_text.encode()).hexdigest()[:12],
    }
    if verbosity >= 1:
        png_bytes = extract_identity_png(document)
        extra["identity_mask"] = check_identity_png(png_bytes, width, height, len(yolo_lines))
    else:
        assert "artifacts" not in service_meta, "L0 JSON must not carry artifacts"
    if verbosity >= 2:
        objects = service_meta.get("objects") or []
        ids = [item.get("instance_id") for item in objects]
        assert ids == list(range(1, len(ids) + 1)), f"object IDs not bijective: {ids[:8]}"
        assert len(objects) == len(yolo_lines), "objects and YOLO lines must agree"
        extra["objects"] = len(objects)
        sample_fields = sorted(objects[0].keys()) if objects else []
        extra["object_fields"] = sample_fields[:12]
    if verbosity >= 3:
        stages = service_meta.get("stage_statuses") or []
        extra["stages_reported"] = len(stages)
        timings = service_meta.get("timings_ms")
        extra["has_timings"] = isinstance(timings, dict)
        provenance = service_meta.get("provenance") or {}
        runtime = provenance.get("runtime") or {}
        if runtime:
            assert runtime.get("device", {}).get("logical") == "cuda:0"
            assert runtime.get("strategy") in {
                "sam2_clip_gpu_blip3_cpu_swap",
                "sam2_clip_blip3_gpu_resident",
            }
    return case_summary(case, status, meta, extra, passed=True)


def run_blip3_rejection_case(
    host: str, port: int, fixture_png: bytes, api_key: str | None = None
) -> dict[str, Any]:
    status, payload, meta = post_completion(
        host,
        port,
        image_bytes=fixture_png,
        config_bytes=BLIP3_CONFIG_YAML,
        verbosity=2,
        response_format="json",
        api_key=api_key,
    )
    code = error_code(payload)
    passed = status == 200 and isinstance(payload, dict)
    return case_summary(
        "blip3_supported",
        status,
        meta,
        {"error_code": code},
        passed=passed,
    )


def run_busy_case(host: str, port: int, *, fixture_png: bytes, attempts: int = 3) -> dict[str, Any]:
    """Fire one large slow request plus an immediate follower.

    With queue_depth=0 the follower must receive 503 service_busy while the
    slot is held. A wide fixture keeps the occupied window comfortably long.
    """
    import threading

    wide_png = make_fixture_png(size=1024)
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        results: dict[str, tuple[int, Any, dict[str, str]]] = {}

        def slow_call() -> None:
            results["slow"] = post_completion(
                host,
                port,
                image_bytes=wide_png,
                config_bytes=SAFE_CONFIG_YAML,
                verbosity=0,
                response_format="json",
                timeout_s=240.0,
            )

        thread = threading.Thread(target=slow_call, daemon=True)
        thread.start()
        time.sleep(0.75)
        follower_status, follower_payload, follower_meta = post_completion(
            host,
            port,
            image_bytes=fixture_png,
            config_bytes=SAFE_CONFIG_YAML,
            verbosity=0,
            response_format="json",
        )
        thread.join(timeout=300)
        slow_status = results.get("slow", (None, None, {}))[0]
        follower_code = error_code(follower_payload)
        busy = follower_status == 503 and follower_code == "service_busy"
        last = {
            "attempt": attempt + 1,
            "slow_status": slow_status,
            "follower_status": follower_status,
            "follower_error_code": follower_code,
            "retry_after": follower_meta.get("retry-after"),
            "overlap_proven": bool(busy and slow_status == 200),
        }
        if busy:
            break
    return {
        "case": "busy_overlap",
        "passed": bool(last.get("overlap_proven")),
        **last,
    }


def run_invalid_case(host: str, port: int, fixture_png: bytes) -> dict[str, Any]:
    """Exercise sanitized invalid-image and invalid-YAML rejection paths."""
    image_status, image_payload, image_meta = post_completion(
        host,
        port,
        image_bytes=b"not-an-image",
        config_bytes=SAFE_CONFIG_YAML,
        verbosity=0,
        response_format="json",
    )
    config_status, config_payload, config_meta = post_completion(
        host,
        port,
        image_bytes=fixture_png,
        config_bytes=INVALID_CONFIG_YAML,
        verbosity=0,
        response_format="json",
    )
    image_code = error_code(image_payload)
    config_code = error_code(config_payload)
    return {
        "case": "invalid_input",
        "passed": image_status == 400
        and image_code == "invalid_image"
        and config_status == 400
        and config_code == "invalid_config",
        "image_status": image_status,
        "image_error_code": image_code,
        "image_response_bytes": int(image_meta.get("bytes", 0)),
        "config_status": config_status,
        "config_error_code": config_code,
        "config_response_bytes": int(config_meta.get("bytes", 0)),
    }


def run_expected_error_case(
    host: str,
    port: int,
    *,
    case: str,
    expected_status: int,
    expected_code: str,
    fixture_png: bytes,
    verbosity: int = 0,
    timeout_s: float = 150.0,
) -> dict[str, Any]:
    status, payload, meta = post_completion(
        host,
        port,
        image_bytes=fixture_png,
        config_bytes=SAFE_CONFIG_YAML,
        verbosity=verbosity,
        response_format="json",
        timeout_s=timeout_s,
    )
    code = error_code(payload)
    return case_summary(
        case,
        status,
        meta,
        {"error_code": code},
        passed=status == expected_status and code == expected_code,
    )


def run_cancel_case(
    host: str,
    port: int,
    *,
    fixture_png: bytes,
    settle_seconds: float = 3.0,
) -> dict[str, Any]:
    """Close a real request socket, then prove a later request still works."""
    body, content_type = build_multipart(
        image_bytes=fixture_png,
        config_bytes=SAFE_CONFIG_YAML,
        verbosity=0,
        response_format="json",
        boundary=uuid_module.uuid4().hex,
    )
    connection = http.client.HTTPConnection(host, port, timeout=10)
    connection.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": content_type, "Connection": "close"},
    )
    connection.close()
    time.sleep(max(settle_seconds, 0.0))
    recovery_status, recovery_payload, recovery_meta = post_completion(
        host,
        port,
        image_bytes=fixture_png,
        config_bytes=SAFE_CONFIG_YAML,
        verbosity=0,
        response_format="json",
    )
    recovery_code = error_code(recovery_payload)
    return {
        "case": "client_cancel_recovery",
        "passed": recovery_status == 200,
        "client_socket_closed": True,
        "recovery_status": recovery_status,
        "recovery_error_code": recovery_code if recovery_status != 200 else None,
        "recovery_latency_ms": float(recovery_meta.get("latency_ms", 0)),
    }


def run_repeat_case(host: str, port: int, *, fixture_png: bytes, repeats: int) -> dict[str, Any]:
    latencies: list[float] = []
    yolo_hashes: set[str] = set()
    statuses: set[int] = set()
    for _ in range(repeats):
        status, payload, meta = post_completion(
            host,
            port,
            image_bytes=fixture_png,
            config_bytes=SAFE_CONFIG_YAML,
            verbosity=0,
            response_format="json",
        )
        statuses.add(status)
        assert status == 200, "repeat case requires success"
        text = payload["choices"][0]["text"] if isinstance(payload, dict) else ""
        yolo_hashes.add(hashlib.sha256(text.encode()).hexdigest()[:12])
        latencies.append(float(meta.get("latency_ms", 0)))
    return {
        "case": "repeat_stability",
        "repeats": repeats,
        "statuses": sorted(statuses),
        "distinct_yolo_outputs": len(yolo_hashes),
        "latency_ms_min": round(min(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
        "passed": statuses == {200} and len(yolo_hashes) <= 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3], choices=[0, 1, 2, 3])
    parser.add_argument("--formats", nargs="*", default=["json"], choices=["json", "zip"])
    parser.add_argument("--skip-blip3-check", action="store_true")
    parser.add_argument(
        "--busy", action="store_true", help="prove overlapping-request busy behavior"
    )
    parser.add_argument("--repeat", type=int, default=0, help="sequential repeat-stability calls")
    parser.add_argument("--invalid", action="store_true", help="exercise invalid image/YAML errors")
    parser.add_argument(
        "--failure", action="store_true", help="expect the operator-injected inference failure"
    )
    parser.add_argument(
        "--timeout", action="store_true", help="expect the operator-injected deadline timeout"
    )
    parser.add_argument(
        "--cancel", action="store_true", help="close a request and prove subsequent recovery"
    )
    parser.add_argument(
        "--response-too-large",
        action="store_true",
        help="expect the operator-configured response-size rejection",
    )
    parser.add_argument("--cancel-settle-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    fixture_png = make_fixture_png()
    results: list[dict[str, Any]] = []
    for verbosity, response_format in itertools.product(args.levels, args.formats):
        results.append(
            run_level_case(
                args.host,
                args.port,
                verbosity=verbosity,
                response_format=response_format,
                fixture_png=fixture_png,
            )
        )
    if not args.skip_blip3_check:
        results.append(run_blip3_rejection_case(args.host, args.port, fixture_png))
    if args.repeat > 0:
        results.append(
            run_repeat_case(args.host, args.port, fixture_png=fixture_png, repeats=args.repeat)
        )
    if args.busy:
        results.append(run_busy_case(args.host, args.port, fixture_png=fixture_png))
    if args.invalid:
        results.append(run_invalid_case(args.host, args.port, fixture_png))
    if args.failure:
        results.append(
            run_expected_error_case(
                args.host,
                args.port,
                case="inference_failure",
                expected_status=500,
                expected_code="inference_failure",
                fixture_png=fixture_png,
            )
        )
    if args.timeout:
        results.append(
            run_expected_error_case(
                args.host,
                args.port,
                case="deadline_timeout",
                expected_status=504,
                expected_code="timeout",
                fixture_png=fixture_png,
                timeout_s=60.0,
            )
        )
    if args.cancel:
        results.append(
            run_cancel_case(
                args.host,
                args.port,
                fixture_png=fixture_png,
                settle_seconds=args.cancel_settle_seconds,
            )
        )
    if args.response_too_large:
        results.append(
            run_expected_error_case(
                args.host,
                args.port,
                case="response_too_large",
                expected_status=413,
                expected_code="response_too_large",
                fixture_png=fixture_png,
                verbosity=3,
            )
        )

    json.dump(results, sys.stdout, indent=2)
    print()
    failed = [item for item in results if not item.get("passed")]
    if failed:
        print(f"smoke: {len(failed)} case(s) FAILED", file=sys.stderr)
        return 1
    print(f"smoke: {len(results)} case(s) PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
