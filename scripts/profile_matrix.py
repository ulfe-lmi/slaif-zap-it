#!/usr/bin/env python3
"""Authenticated, content-free four-profile live service matrix.

The harness deliberately keeps transport, response, and resource validation
separate from the operator's real model process.  It sends only a generated
128x128 RGB fixture and API-safe in-memory YAML, and prints sanitized facts
instead of response bodies, prompts, answers, or request identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from smoke_local_service import make_fixture_png, post_completion  # noqa: E402

try:
    from src.runtime.models import APPROVED_MODEL_SPECS
except ImportError:  # pragma: no cover - operator script is run from the checkout
    APPROVED_MODEL_SPECS = {}


PROFILE_SEQUENCE = (
    "sam2",
    "sam2_clip",
    "sam2_blip3",
    "sam2_clip_blip3",
    "sam2_clip_blip3",
    "sam2_blip3",
    "sam2_clip",
    "sam2",
)
SUPPORTED_PROFILES = tuple(dict.fromkeys(PROFILE_SEQUENCE))
ALL_RESIDENT_STRATEGY = "sam2_clip_blip3_gpu_resident"
GPU_MEMORY_CEILING_MIB = 24_576 * 0.9
EXPECTED_STAGE_NAMES = (
    "preprocessing",
    "sam2",
    "postsam2_filter",
    "clip",
    "blip3",
    "label_filter",
    "visualization",
    "ordering",
)


class MatrixValidationError(AssertionError):
    """Raised when the live matrix cannot prove an invariant."""


@dataclass(frozen=True)
class ResourceSample:
    """Content-free per-request resource evidence."""

    current_allocated_mib: float
    peak_allocated_mib: float
    current_reserved_mib: float
    peak_reserved_mib: float
    free_mib: float
    host_rss_mib: float
    transition_count: int
    registry_initializations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "torch_current_allocated_mib": round(self.current_allocated_mib, 1),
            "torch_peak_allocated_mib": round(self.peak_allocated_mib, 1),
            "torch_current_reserved_mib": round(self.current_reserved_mib, 1),
            "torch_peak_reserved_mib": round(self.peak_reserved_mib, 1),
            "torch_free_mib": round(self.free_mib, 1),
            "host_rss_mib": round(self.host_rss_mib, 1),
            "transition_count": self.transition_count,
            "registry_initializations": self.registry_initializations,
        }


def profile_config(profile: str) -> bytes:
    """Return one API-safe YAML configuration for a supported profile."""
    if profile not in SUPPORTED_PROFILES:
        raise MatrixValidationError(f"unsupported matrix profile: {profile}")
    lines = [
        "alpha: 0.6",
        "preprocessing:",
        "  resize: 1.0",
        "mask_generator: {}",
        "postsam2processing:",
        "  maxsize: 100000",
    ]
    if "clip" in profile:
        lines.extend(
            [
                "clip:",
                "  labels:",
                '    red: "a red object"',
                '    green: "a green object"',
            ]
        )
    if "blip3" in profile:
        lines.extend(
            [
                "blip3:",
                "  any,1.0:",
                '    question: "What is in this region?"',
                '    trueresult: "yes"',
                '    falseresult: "no"',
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _number(metrics: Mapping[str, float], name: str) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixValidationError(f"missing metric: {name}") from exc
    if value < 0:
        raise MatrixValidationError(f"negative metric: {name}")
    return value


def _parse_prometheus_metrics(body: bytes) -> dict[str, float]:
    """Parse only fixed, unlabeled gauges/counters used by this harness."""
    wanted = {
        "zap_it_torch_gpu_allocated_bytes",
        "zap_it_torch_gpu_peak_allocated_bytes",
        "zap_it_torch_gpu_reserved_bytes",
        "zap_it_torch_gpu_peak_reserved_bytes",
        "zap_it_torch_gpu_free_bytes",
        "zap_it_host_rss_max_bytes",
        "zap_it_residency_transition_count",
    }
    values: dict[str, float] = {}
    registry_initializations: float | None = None
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MatrixValidationError("metrics response is not UTF-8") from exc
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(" ", 1)
        if len(fields) != 2:
            continue
        name, raw_value = fields
        if name.startswith("zap_it_model_initializations_total{"):
            if 'component="registry"' in name and 'outcome="success"' in name:
                try:
                    registry_initializations = float(raw_value)
                except ValueError as exc:
                    raise MatrixValidationError(
                        "registry initialization metric is malformed"
                    ) from exc
            continue
        if name not in wanted:
            continue
        try:
            values[name] = float(raw_value)
        except ValueError as exc:
            raise MatrixValidationError(f"metric {name} is malformed") from exc
    if registry_initializations is None:
        raise MatrixValidationError("missing registry initialization metric")
    values["zap_it_model_initializations_total.registry.success"] = registry_initializations
    return values


def fetch_metrics(host: str, port: int, *, api_key: str) -> ResourceSample:
    """Fetch authenticated fixed-label resource metrics without retaining raw text."""
    request = urllib.request.Request(f"http://{host}:{port}/metrics", method="GET")
    request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise MatrixValidationError(f"metrics HTTP status {response.status}")
            metrics = _parse_prometheus_metrics(response.read())
    except urllib.error.HTTPError as exc:
        raise MatrixValidationError(f"metrics HTTP status {exc.code}") from exc
    scale = 1024 * 1024
    return ResourceSample(
        current_allocated_mib=_number(metrics, "zap_it_torch_gpu_allocated_bytes") / scale,
        peak_allocated_mib=_number(metrics, "zap_it_torch_gpu_peak_allocated_bytes") / scale,
        current_reserved_mib=_number(metrics, "zap_it_torch_gpu_reserved_bytes") / scale,
        peak_reserved_mib=_number(metrics, "zap_it_torch_gpu_peak_reserved_bytes") / scale,
        free_mib=_number(metrics, "zap_it_torch_gpu_free_bytes") / scale,
        host_rss_mib=_number(metrics, "zap_it_host_rss_max_bytes") / scale,
        transition_count=int(_number(metrics, "zap_it_residency_transition_count")),
        registry_initializations=int(
            _number(metrics, "zap_it_model_initializations_total.registry.success")
        ),
    )


def check_shm_residue(root: str | os.PathLike[str]) -> int:
    """Reject request residue while allowing the launcher's runtime files."""
    path = Path(root)
    if not path.is_dir() or path.is_symlink():
        raise MatrixValidationError("shared-memory root is unavailable")
    unexpected = []
    for child in path.iterdir():
        if child.name == "runtime":
            continue
        unexpected.append(child.name)
    if unexpected:
        raise MatrixValidationError("request residue exists in shared memory")
    return sum(1 for _ in path.iterdir())


def _stage_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    service = payload.get("service")
    if not isinstance(service, Mapping):
        raise MatrixValidationError("response service metadata is malformed")
    stages = service.get("stage_statuses")
    if not isinstance(stages, list):
        raise MatrixValidationError("L3 response has no stage status list")
    result: dict[str, Mapping[str, Any]] = {}
    for item in stages:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise MatrixValidationError("stage status is malformed")
        if item["name"] in result:
            raise MatrixValidationError("duplicate stage status")
        result[item["name"]] = item
    if tuple(result) != EXPECTED_STAGE_NAMES:
        raise MatrixValidationError("stage status shape is unexpected")
    return result


def _semantic_digest(payload: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]]) -> str:
    service = payload["service"]
    objects = service.get("objects") or []
    if not isinstance(objects, list):
        raise MatrixValidationError("object list is malformed")
    artifacts = service.get("artifacts") or []
    if not isinstance(artifacts, list):
        raise MatrixValidationError("artifact list is malformed")
    shape = {
        "stages": [(name, stages[name].get("status")) for name in EXPECTED_STAGE_NAMES],
        "candidate_count_keys": sorted((service.get("candidate_counts") or {}).keys()),
        "object_count": len(objects),
        "object_field_shapes": sorted(
            [sorted(str(key) for key in item.keys()) for item in objects]
        ),
        "answer_count": sum(
            1 for item in objects if isinstance(item, Mapping) and item.get("blip3_answer")
        ),
        "artifact_names": sorted(
            str(item.get("name")) for item in artifacts if isinstance(item, Mapping)
        ),
        "yolo_line_count": len(
            [
                line
                for line in str(payload.get("choices", [{}])[0].get("text", "")).splitlines()
                if line.strip()
            ]
        ),
    }
    encoded = json.dumps(shape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_profile_response(
    profile: str,
    status: int,
    payload: Any,
    meta: Mapping[str, str],
    resources: ResourceSample,
    *,
    residue_count: int,
) -> dict[str, Any]:
    """Fail closed on the complete sanitized contract for one matrix call."""
    if profile not in SUPPORTED_PROFILES:
        raise MatrixValidationError("unsupported profile")
    if status != 200 or not isinstance(payload, Mapping):
        raise MatrixValidationError("matrix call did not return HTTP 200 JSON")
    service = payload.get("service")
    if not isinstance(service, Mapping) or service.get("verbosity") != 3:
        raise MatrixValidationError("matrix response is not authenticated L3 metadata")
    provenance = service.get("provenance")
    runtime = provenance.get("runtime") if isinstance(provenance, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise MatrixValidationError("runtime provenance is missing")
    if runtime.get("strategy") != ALL_RESIDENT_STRATEGY:
        raise MatrixValidationError("wrong runtime strategy")
    device = runtime.get("device")
    if not isinstance(device, Mapping) or device.get("logical") != "cuda:0":
        raise MatrixValidationError("wrong logical device")
    models = runtime.get("models")
    expected_models = {
        name: {"id": spec.model_id, "revision": spec.revision}
        for name, spec in APPROVED_MODEL_SPECS.items()
        if name in {"sam2", "clip", "blip3"}
    }
    if not isinstance(models, Mapping) or dict(models) != expected_models:
        raise MatrixValidationError("wrong pinned runtime model identities")
    if set(models) != {"sam2", "clip", "blip3"}:
        raise MatrixValidationError("wrong runtime model count")
    residency = runtime.get("residency")
    if not isinstance(residency, Mapping) or residency.get("request_transition_policy") != "none":
        raise MatrixValidationError("runtime residency policy is not all-resident")
    stages = _stage_map(payload)
    expected = {
        "sam2": {"sam2": "executed", "clip": "not_configured", "blip3": "not_configured"},
        "sam2_clip": {"sam2": "executed", "clip": "executed", "blip3": "not_configured"},
        "sam2_blip3": {"sam2": "executed", "clip": "not_configured", "blip3": "executed"},
        "sam2_clip_blip3": {"sam2": "executed", "clip": "executed", "blip3": "executed"},
    }[profile]
    for name, expected_status in expected.items():
        if stages[name].get("status") != expected_status:
            raise MatrixValidationError(f"wrong {name} stage status for {profile}")
    objects = service.get("objects")
    if not isinstance(objects, list):
        raise MatrixValidationError("L3 object list is malformed")
    answer_count = sum(
        1
        for item in objects
        if isinstance(item, Mapping)
        and isinstance(item.get("blip3_answer"), str)
        and item["blip3_answer"]
    )
    if "blip3" in profile and answer_count < 1:
        raise MatrixValidationError("BLIP3 profile produced no bounded answer")
    if "blip3" not in profile and answer_count != 0:
        raise MatrixValidationError("non-BLIP3 profile produced an answer")
    if resources.transition_count != 0:
        raise MatrixValidationError("residency transition count is nonzero")
    if resources.peak_reserved_mib >= GPU_MEMORY_CEILING_MIB:
        raise MatrixValidationError("peak reserved memory breaches the 90% ceiling")
    return {
        "http_status": status,
        "latency_ms": round(float(meta.get("latency_ms", 0.0)), 1),
        "profile": profile,
        "object_count": len(objects),
        "answer_count": answer_count,
        "stage_count": len(stages),
        "stage_statuses": {name: stages[name]["status"] for name in ("sam2", "clip", "blip3")},
        "semantic_digest": _semantic_digest(payload, stages),
        "request_residue_entries": residue_count,
        **resources.as_dict(),
    }


def validate_resource_trajectory(records: list[Mapping[str, Any]]) -> None:
    """Reject a monotonic GPU/host growth trajectory in sanitized records."""
    if not records:
        raise MatrixValidationError("matrix produced no resource samples")
    for key in ("torch_current_reserved_mib", "torch_current_allocated_mib", "host_rss_mib"):
        values = [float(item[key]) for item in records]
        if all(left <= right for left, right in zip(values, values[1:])) and values[-1] > values[0]:
            raise MatrixValidationError(f"monotonic resource growth detected: {key}")


def _stat(values: list[float]) -> dict[str, float]:
    return {
        "first": round(values[0], 1),
        "minimum": round(min(values), 1),
        "maximum": round(max(values), 1),
        "median": round(statistics.median(values), 1),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build report-ready per-profile summaries without response content."""
    summaries: dict[str, Any] = {}
    for profile in SUPPORTED_PROFILES:
        items = [item for item in records if item["profile"] == profile]
        if len(items) != 2:
            raise MatrixValidationError(f"profile {profile} did not occur twice")
        if items[0]["semantic_digest"] != items[1]["semantic_digest"]:
            raise MatrixValidationError(f"profile {profile} semantic shape is not repeatable")
        summaries[profile] = {
            "calls": 2,
            "timing_ms": _stat([float(item["latency_ms"]) for item in items]),
            "objects": _stat([float(item["object_count"]) for item in items]),
            "answers": _stat([float(item["answer_count"]) for item in items]),
            "stage_count": _stat([float(item["stage_count"]) for item in items]),
            "semantic_digest": items[0]["semantic_digest"],
        }
    return summaries


def run_matrix(
    host: str,
    port: int,
    *,
    api_key: str,
    shm_root: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Run the exact interleaved eight-call matrix."""
    fixture = make_fixture_png()
    records: list[dict[str, Any]] = []
    for profile in PROFILE_SEQUENCE:
        started = time.perf_counter()
        status, payload, meta = post_completion(
            host,
            port,
            image_bytes=fixture,
            config_bytes=profile_config(profile),
            verbosity=3,
            response_format="json",
            timeout_s=timeout_s,
            api_key=api_key,
        )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        resources = fetch_metrics(host, port, api_key=api_key)
        residue_count = check_shm_residue(shm_root)
        meta = {**meta, "latency_ms": str(elapsed)}
        records.append(
            validate_profile_response(
                profile,
                status,
                payload,
                meta,
                resources,
                residue_count=residue_count,
            )
        )
    validate_resource_trajectory(records)
    if records[-1]["registry_initializations"] != 1:
        raise MatrixValidationError("model registry initialized more than once")
    return {
        "sequence": list(PROFILE_SEQUENCE),
        "physical_capacity_mib": 24_576,
        "peak_reserved_ceiling_mib": GPU_MEMORY_CEILING_MIB,
        "peak_reserved_mib": max(item["torch_peak_reserved_mib"] for item in records),
        "calls": records,
        "profiles": summarize(records),
        "matrix_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", default=os.environ.get("SLAIF_ZAP_IT_API_KEY"))
    parser.add_argument(
        "--shm-root", default=os.environ.get("SLAIF_ZAP_IT_TMP_ROOT", "/dev/shm/slaif-zap-it")
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    if not args.api_key:
        print("profile-matrix: authenticated API key is required", file=sys.stderr)
        return 2
    try:
        result = run_matrix(
            args.host,
            args.port,
            api_key=args.api_key,
            shm_root=args.shm_root,
            timeout_s=args.timeout,
        )
    except (MatrixValidationError, AssertionError, OSError, urllib.error.URLError) as exc:
        print(f"profile-matrix: FAILED ({type(exc).__name__})", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    print("profile-matrix: PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
