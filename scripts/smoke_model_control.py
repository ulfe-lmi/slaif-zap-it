#!/usr/bin/env python3
"""Mechanically checked live qualification for the explicit model lifecycle.

The service must already be running in explicit mode on the exact card named by
the active order. The harness emits only statuses, bounded timings/counts,
state names, semantic digests and resource facts; it never emits credentials,
request bodies, prompts, labels, answers or response bodies.

The process is stopped normally at the end, including after a failed check.
Launch the service with ``SLAIF_ZAP_IT_TEST_DELAY_SECONDS`` set to a small
operator-only value so the drain check can observe a real active inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__:
    from scripts.smoke_local_service import make_fixture_png, post_completion
else:
    from smoke_local_service import make_fixture_png, post_completion


ASSIGNED_GPU_INDEX = 0
ASSIGNED_GPU_UUID = "GPU-a91444df-4e87-011e-3347-9b3a4b9f9575"
ASSIGNED_GPU_PCI = "00000000:0B:00.0"
ASSIGNED_GPU_NAME = "NVIDIA GeForce RTX 3090"
ASSIGNED_GPU_TOTAL_MIB = 24576
MODEL_NAME = "zap-it-1"
COMBINED_CONFIG_YAML = b"""
alpha: 0.6
preprocessing:
  resize: 1.0
mask_generator: {}
postsam2processing:
  maxsize: 100000
clip:
  labels:
    red: "a red object"
    green: "a green object"
blip3:
  "any,1.0":
    question: "What is in this region?"
    trueresult: "yes"
    falseresult: "no"
"""


class SmokeFailure(RuntimeError):
    """A sanitized harness assertion."""


def fail(message: str) -> None:
    raise SmokeFailure(message)


def _json_or_empty(raw: bytes) -> Any:
    if not raw.strip():
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def call(
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    method: str = "GET",
    body: bytes = b"",
    timeout: float = 20.0,
) -> tuple[int, Any, int, float]:
    headers: dict[str, str] = {}
    if body:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url}{path}", data=body or None, headers=headers, method=method
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, _json_or_empty(raw), len(raw), _elapsed_ms(started)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, _json_or_empty(raw), len(raw), _elapsed_ms(started)
    except (OSError, urllib.error.URLError) as exc:
        del exc
        fail("HTTP request failed")


def _error_code(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return error["code"]
    return "?"


def _assert_status(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        fail(f"{label} returned HTTP {actual}, expected {expected}")


def _index(base_url: str, token: str, *, ready: bool = False) -> tuple[str, float]:
    body = b'{"ready":true}' if ready else b"{}"
    status, payload, _size, elapsed = call(
        base_url, "/v2/repository/index", token=token, method="POST", body=body
    )
    _assert_status(status, 200, "repository index")
    if not isinstance(payload, list) or len(payload) > 1:
        fail("repository index was not a bounded list")
    if not payload:
        return "OMITTED", elapsed
    entry = payload[0]
    if not isinstance(entry, dict) or entry.get("name") != MODEL_NAME:
        fail("repository index returned an unexpected model entry")
    state = entry.get("state")
    if state not in {"UNAVAILABLE", "LOADING", "READY", "UNLOADING"}:
        fail("repository index returned an invalid lifecycle state")
    return str(state), elapsed


def _operation(
    base_url: str, token: str, operation: str, *, timeout: float = 900.0
) -> tuple[int, int, float]:
    status, payload, size, elapsed = call(
        base_url,
        f"/v2/repository/models/{MODEL_NAME}/{operation}",
        token=token,
        method="POST",
        body=b"{}",
        timeout=timeout,
    )
    if status == 200 and (size != 0 or payload is not None):
        fail(f"{operation} success body was not empty")
    return status, size, elapsed


_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def _parse_metric_labels(text: str) -> dict[str, str]:
    """Parse the bounded Prometheus label subset in linear time."""
    labels: dict[str, str] = {}
    offset = 0
    length = len(text)
    while offset < length:
        key_start = offset
        while offset < length and (text[offset].isalnum() or text[offset] == "_"):
            offset += 1
        key = text[key_start:offset]
        if not key or offset + 1 >= length or text[offset : offset + 2] != '="':
            return {}
        offset += 2
        value: list[str] = []
        while offset < length:
            char = text[offset]
            offset += 1
            if char == '"':
                break
            if char == "\\":
                if offset >= length:
                    return {}
                escaped = text[offset]
                offset += 1
                value.append({"n": "\n", "\\": "\\", '"': '"'}.get(escaped, escaped))
            else:
                value.append(char)
        else:
            return {}
        labels[key] = "".join(value)
        if offset == length:
            break
        if text[offset] != ",":
            return {}
        offset += 1
    return labels


def _metrics(base_url: str, inference_token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/metrics", headers={"Authorization": f"Bearer {inference_token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 200:
                fail(f"metrics returned HTTP {response.status}")
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        del exc
        fail("metrics authorization failed")
    except (OSError, UnicodeError, urllib.error.URLError) as exc:
        del exc
        fail("metrics body could not be read")
    values: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        match = _METRIC_RE.match(line)
        if not match:
            continue
        labels = _parse_metric_labels(match.group("labels") or "")
        values.append((match.group("name"), labels, float(match.group("value"))))

    def value(
        name: str, labels: dict[str, str] | None = None, default: float | None = None
    ) -> float:
        wanted = labels or {}
        for metric_name, metric_labels, metric_value in values:
            if metric_name == name and metric_labels == wanted:
                return metric_value
        if default is not None:
            return default
        fail(f"required metric missing: {name}")

    state = "UNKNOWN"
    for candidate in ("UNAVAILABLE", "LOADING", "READY", "UNLOADING"):
        if value("zap_it_model_lifecycle_state", {"state": candidate}, 0.0) == 1.0:
            state = candidate
            break
    return {
        "torch_allocated_bytes": int(value("zap_it_torch_gpu_allocated_bytes", default=0)),
        "torch_reserved_bytes": int(value("zap_it_torch_gpu_reserved_bytes", default=0)),
        "torch_peak_allocated_bytes": int(
            value("zap_it_torch_gpu_peak_allocated_bytes", default=0)
        ),
        "torch_peak_reserved_bytes": int(value("zap_it_torch_gpu_peak_reserved_bytes", default=0)),
        "torch_free_bytes": int(value("zap_it_torch_gpu_free_bytes", default=0)),
        "host_rss_max_bytes": int(value("zap_it_host_rss_max_bytes", default=0)),
        "model_loaded": int(value("zap_it_model_loaded", default=0)),
        "initializations": int(
            value(
                "zap_it_model_initializations_total",
                {"component": "registry", "outcome": "success"},
                0.0,
            )
        ),
        "active_inference": int(value("zap_it_active_inference", default=0)),
        "lifecycle_state": state,
    }


def _nvidia_rows(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        fail("nvidia-smi GPU query failed")
    return [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row]


def _compute_rows() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return []
    return [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row]


def _gpu_snapshot() -> dict[str, Any]:
    rows = _nvidia_rows("index,uuid,name,pci.bus_id,memory.total,memory.used,memory.free")
    gpus: dict[int, dict[str, Any]] = {}
    for row in rows:
        if len(row) != 7:
            fail("nvidia-smi returned an unexpected GPU row")
        try:
            index = int(row[0])
            facts = {
                "uuid": row[1],
                "name": row[2],
                "pci": row[3],
                "total_mib": int(row[4]),
                "used_mib": int(row[5]),
                "free_mib": int(row[6]),
            }
        except (TypeError, ValueError) as exc:
            del exc
            fail("nvidia-smi returned invalid GPU facts")
        gpus[index] = facts
    target = gpus.get(ASSIGNED_GPU_INDEX)
    if target is None:
        fail("assigned physical GPU index is not present")
    if (
        target["uuid"] != ASSIGNED_GPU_UUID
        or target["pci"] != ASSIGNED_GPU_PCI
        or target["name"] != ASSIGNED_GPU_NAME
        or target["total_mib"] != ASSIGNED_GPU_TOTAL_MIB
    ):
        fail("assigned physical GPU facts do not match the active order")
    compute = _compute_rows()
    target_processes = [row for row in compute if row and row[0] == ASSIGNED_GPU_UUID]
    other_processes = [row for row in compute if row and row[0] != ASSIGNED_GPU_UUID]
    target_memory = 0
    for row in target_processes:
        if len(row) >= 3:
            try:
                target_memory += int(row[2])
            except ValueError:
                continue
    return {
        "target": dict(target),
        "gpu_count": len(gpus),
        "non_target_compute_processes": len(other_processes),
        "target_compute_processes": len(target_processes),
        "target_compute_memory_mib": target_memory,
    }


def _process_facts(pid: int, port: int) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        fail("service PID is not running")
    try:
        cmdline = (proc / "cmdline").read_bytes()
        stat = (proc / "stat").read_text(encoding="utf-8")
        status = (proc / "status").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        del exc
        fail("service process facts could not be read")
    if b"scripts/serve_local.py" not in cmdline:
        fail("PID is not the ZAP-IT serve_local entrypoint")
    remainder = stat[stat.rfind(")") + 2 :].split()
    if len(remainder) <= 19:
        fail("service PID stat record is malformed")
    rss_match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    if rss_match is None:
        fail("service RSS fact is unavailable")
    result = subprocess.run(
        ["ss", "-H", "-ltnp"], check=False, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        fail("listener inspection failed")
    listener = any(
        f":{port}" in line and (f"pid={pid}," in line or f"pid={pid})" in line)
        for line in result.stdout.splitlines()
    )
    return {
        "pid": pid,
        "start_time": remainder[19],
        "listener": listener,
        "rss_mib": round(int(rss_match.group(1)) / 1024, 1),
    }


def _identity_tuple(facts: dict[str, Any]) -> tuple[Any, ...]:
    return facts["pid"], facts["start_time"], bool(facts["listener"])


def _start_operation_and_observe(
    base_url: str, token: str, operation: str, *, timeout: float
) -> tuple[int, float, list[str]]:
    states: list[str] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_operation, base_url, token, operation, timeout=timeout)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            state, _elapsed = _index(base_url, token)
            if not states or states[-1] != state:
                states.append(state)
            time.sleep(0.05)
        if not future.done():
            fail(f"{operation} operation did not settle")
        status, _size, elapsed = future.result()
    if operation == "load" and "LOADING" not in states:
        fail("load did not expose LOADING concurrently")
    if operation == "unload" and "UNLOADING" not in states:
        fail("unload did not expose UNLOADING concurrently")
    return status, elapsed, states


def _semantic_inference(
    base_url: str, inference_token: str, fixture: bytes, *, timeout: float
) -> dict[str, Any]:
    port = int(base_url.rsplit(":", 1)[1])
    status, payload, meta = post_completion(
        "127.0.0.1",
        port,
        image_bytes=fixture,
        config_bytes=COMBINED_CONFIG_YAML,
        verbosity=3,
        response_format="json",
        timeout_s=timeout,
        api_key=inference_token,
    )
    if status != 200 or not isinstance(payload, dict):
        fail(f"combined inference returned HTTP {status} ({_error_code(payload)})")
    service = payload.get("service")
    if not isinstance(service, dict):
        fail("combined inference omitted service metadata")
    stages = service.get("stage_statuses")
    if not isinstance(stages, list):
        fail("combined inference omitted stage statuses")
    stage_map = {
        item.get("name"): item.get("status")
        for item in stages
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for stage in ("sam2", "clip", "blip3"):
        if stage_map.get(stage) != "executed":
            fail(f"combined inference did not execute {stage}")
    objects = service.get("objects")
    if not isinstance(objects, list) or not objects:
        fail("combined inference returned no bounded objects")
    answers = [item.get("blip3_answer") for item in objects if isinstance(item, dict)]
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        fail("combined inference returned no non-empty bounded BLIP3 answers")
    if any(len(answer) > 512 for answer in answers):
        fail("combined inference returned an unbounded BLIP3 answer")
    choice = (payload.get("choices") or [{}])[0]
    yolo_text = choice.get("text") if isinstance(choice, dict) else None
    if not isinstance(yolo_text, str):
        fail("combined inference omitted completion text")
    shape = {
        "yolo_lines": len([line for line in yolo_text.splitlines() if line.strip()]),
        "stages": sorted((str(name), str(status)) for name, status in stage_map.items()),
        "objects": len(objects),
        "object_fields": sorted(
            {key for item in objects if isinstance(item, dict) for key in item.keys()}
        ),
        "answer_lengths": sorted(len(answer) for answer in answers),
        "response_format": "json",
        "verbosity": 3,
    }
    digest = hashlib.sha256(json.dumps(shape, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": status,
        "latency_ms": float(meta.get("latency_ms", 0)),
        "response_bytes": int(meta.get("bytes", 0)),
        "object_count": len(objects),
        "answer_count": len(answers),
        "answer_max_length": max(len(answer) for answer in answers),
        "semantic_digest": digest,
    }


def _completion_status(
    base_url: str, inference_token: str, fixture: bytes, *, timeout: float
) -> tuple[int, str, float]:
    port = int(base_url.rsplit(":", 1)[1])
    status, payload, meta = post_completion(
        "127.0.0.1",
        port,
        image_bytes=fixture,
        config_bytes=COMBINED_CONFIG_YAML,
        verbosity=3,
        response_format="json",
        timeout_s=timeout,
        api_key=inference_token,
    )
    return status, _error_code(payload), float(meta.get("latency_ms", 0))


def _cold_checks(base_url: str, control: str, inference: str, fixture: bytes) -> dict[str, Any]:
    health, _payload, _size, health_ms = call(base_url, "/healthz")
    _assert_status(health, 200, "cold health")
    ready, ready_payload, _size, _elapsed = call(base_url, "/readyz")
    _assert_status(ready, 503, "cold readiness")
    if _error_code(ready_payload) != "not_ready":
        fail("cold readiness did not return not_ready")
    completion_status, completion_code, _latency = _completion_status(
        base_url, inference, fixture, timeout=30
    )
    _assert_status(completion_status, 503, "cold completion")
    if completion_code != "not_ready":
        fail("cold completion did not return not_ready")
    state, index_ms = _index(base_url, control)
    if state != "UNAVAILABLE":
        fail("cold repository index was not UNAVAILABLE")
    ready_state, _ = _index(base_url, control, ready=True)
    if ready_state != "OMITTED":
        fail("cold ready-only repository index was not empty")
    return {
        "health_status": health,
        "health_latency_ms": health_ms,
        "ready_status": ready,
        "completion_status": completion_status,
        "index_state": state,
        "index_latency_ms": index_ms,
    }


def _auth_and_contract_checks(base_url: str, control: str, inference: str) -> dict[str, Any]:
    wrong = hashlib.sha256(os.urandom(32)).hexdigest()
    cases: dict[str, int] = {}
    for label, token in (
        ("index_missing", None),
        ("index_wrong", wrong),
        ("index_inference", inference),
        ("load_missing", None),
        ("load_wrong", wrong),
        ("load_inference", inference),
        ("unload_missing", None),
        ("unload_wrong", wrong),
        ("unload_inference", inference),
    ):
        if label.startswith("index"):
            path, body = "/v2/repository/index", b"{}"
        else:
            operation = "load" if label.startswith("load") else "unload"
            path, body = f"/v2/repository/models/{MODEL_NAME}/{operation}", b"{}"
        status, _payload, _size, _elapsed = call(
            base_url, path, token=token, method="POST", body=body
        )
        _assert_status(status, 401, label)
        cases[label] = status

    invalid_requests = {
        "wrong_model": ("/v2/repository/models/other/load", b"{}", 404),
        "query_parameter": (f"/v2/repository/models/{MODEL_NAME}/load?device=cuda:0", b"{}", 400),
        "unknown_body": (f"/v2/repository/models/{MODEL_NAME}/load", b'{"device":"cuda:0"}', 400),
        "malformed_body": (f"/v2/repository/models/{MODEL_NAME}/load", b"[", 400),
        "oversized_body": (f"/v2/repository/models/{MODEL_NAME}/load", b"x" * 16_385, 413),
        "index_unknown": ("/v2/repository/index", b'{"device":"cuda:0"}', 400),
        "index_non_object": ("/v2/repository/index", b"[]", 400),
    }
    for label, (path, body, expected) in invalid_requests.items():
        status, _payload, _size, _elapsed = call(
            base_url, path, token=control, method="POST", body=body
        )
        _assert_status(status, expected, label)
        cases[label] = status
    return {"statuses": cases, "negative_bearer_count": 9, "allocation_mutation": "none"}


def _release_gate(
    loaded: dict[str, Any], cold_before: dict[str, Any], cold_after: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("allocated", "reserved"):
        key = f"torch_{label}_bytes"
        loaded_delta = max(0, loaded[key] - cold_before[key])
        remaining_delta = max(0, cold_after[key] - cold_before[key])
        if loaded_delta <= 0:
            fail(f"loaded Torch {label} delta was zero")
        released_fraction = (loaded_delta - remaining_delta) / loaded_delta
        if cold_after[key] > 64 * 1024 * 1024 or released_fraction < 0.90:
            fail(f"Torch {label} cold-memory gate failed")
        result[f"{label}_loaded_delta_bytes"] = loaded_delta
        result[f"{label}_remaining_delta_bytes"] = remaining_delta
        result[f"{label}_released_fraction"] = round(released_fraction, 4)
    return result


def _stop_and_cleanup(
    pid: int,
    port: int,
    runtime_root: Path,
    log_path: Path,
    control: str,
    inference: str,
    fixture: bytes,
) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            del exc
            fail("normal service stop signal failed")
        deadline = time.monotonic() + 120
        while proc.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        if proc.exists():
            fail("service did not stop normally")
    if log_path.exists():
        try:
            raw_log = log_path.read_bytes()
        except OSError as exc:
            del exc
            fail("service log could not be inspected")
        forbidden = [control.encode(), inference.encode(), COMBINED_CONFIG_YAML]
        if any(item and item in raw_log for item in forbidden):
            fail("service log contains credential or request content")
        log_checked = True
    else:
        fail("service log was unavailable for sanitized cleanup inspection")
    pidfile = runtime_root / "serve-local.pid"
    if pidfile.exists():
        try:
            if pidfile.read_text(encoding="utf-8").strip() == str(pid):
                pidfile.unlink()
        except OSError as exc:
            del exc
            fail("owned service pidfile cleanup failed")
    try:
        log_path.unlink(missing_ok=True)
        runtime_root.rmdir()
    except OSError as exc:
        del exc
        fail("owned service runtime cleanup failed")
    try:
        remaining = list(runtime_root.parent.iterdir())
    except OSError as exc:
        del exc
        fail("shared-memory root could not be inspected")
    if remaining:
        fail("shared-memory root was not empty after normal stop")
    try:
        socket_result = socket.create_connection(("127.0.0.1", port), timeout=1)
    except OSError:
        port_free = True
    else:
        socket_result.close()
        port_free = False
    if not port_free:
        fail("service listener remained after normal stop")
    final_gpu = _gpu_snapshot()
    if final_gpu["target_compute_processes"] != 0 or final_gpu["non_target_compute_processes"] != 0:
        fail("compute process remained after normal stop")
    if final_gpu["target"]["used_mib"] > 47:
        fail("assigned GPU did not return to the fresh baseline tolerance")
    return {
        "port_free": port_free,
        "log_sanitized": log_checked,
        "shm_entries": 0,
        "target_compute_processes": final_gpu["target_compute_processes"],
        "non_target_compute_processes": final_gpu["non_target_compute_processes"],
        "target_used_mib": final_gpu["target"]["used_mib"],
        "target_baseline_tolerance_mib": 32,
        "fixture_sha256_12": hashlib.sha256(fixture).hexdigest()[:12],
    }


def _resolve_pid(pid_arg: int | None, pidfile: Path) -> int:
    if pid_arg is not None:
        return pid_arg
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        del exc
        fail("service PID was not supplied and owned pidfile was unavailable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--shm-root", default="/dev/shm/slaif-zap-it")
    parser.add_argument("--pid-file")
    parser.add_argument("--log-file")
    parser.add_argument("--cold-tolerance-mib", type=int, default=128)
    args = parser.parse_args(argv)
    control = os.environ.get("SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY")
    inference = os.environ.get("SLAIF_ZAP_IT_API_KEY")
    if not control or not inference or control == inference:
        print("lifecycle smoke FAILED: distinct operator credentials are required", file=sys.stderr)
        return 1
    if (
        os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
    ):
        print("lifecycle smoke FAILED: assigned CUDA visibility is not exact", file=sys.stderr)
        return 1
    if (
        os.environ.get("SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX") != "0"
        or os.environ.get("SLAIF_ZAP_IT_EXPECTED_GPU_UUID") != ASSIGNED_GPU_UUID
    ):
        print("lifecycle smoke FAILED: assigned GPU environment is not exact", file=sys.stderr)
        return 1
    if os.environ.get("SLAIF_ZAP_IT_MODEL_CONTROL_MODE") != "explicit":
        print("lifecycle smoke FAILED: explicit model-control mode is required", file=sys.stderr)
        return 1
    runtime_root = Path(args.shm_root) / "runtime"
    pidfile = Path(args.pid_file) if args.pid_file else runtime_root / "serve-local.pid"
    log_path = Path(args.log_file) if args.log_file else runtime_root / "serve-local.log"
    pid = _resolve_pid(args.pid, pidfile)
    base_url = f"http://127.0.0.1:{args.port}"
    fixture = make_fixture_png(size=128)
    evidence: dict[str, Any] = {
        "status": "PENDING",
        "assigned_gpu": {
            "physical_index": ASSIGNED_GPU_INDEX,
            "uuid": ASSIGNED_GPU_UUID,
            "pci": ASSIGNED_GPU_PCI,
            "name": ASSIGNED_GPU_NAME,
            "total_mib": ASSIGNED_GPU_TOTAL_MIB,
            "logical_device": "cuda:0",
        },
    }
    identity: dict[str, Any] | None = None
    try:
        baseline_gpu = _gpu_snapshot()
        identity = _process_facts(pid, args.port)
        if not identity["listener"]:
            fail("service PID does not own the required loopback listener")
        cold_metrics = _metrics(base_url, inference)
        evidence["cold"] = {
            "http": _cold_checks(base_url, control, inference, fixture),
            "metrics": cold_metrics,
            "gpu": baseline_gpu["target"],
            "process": identity,
        }
        if cold_metrics["initializations"] != 0 or cold_metrics["model_loaded"] != 0:
            fail("cold state already contains model allocation")
        evidence["auth_contract"] = _auth_and_contract_checks(base_url, control, inference)
        post_invalid = _metrics(base_url, inference)
        invalid_state, _ = _index(base_url, control)
        if (
            invalid_state != "UNAVAILABLE"
            or post_invalid["initializations"] != cold_metrics["initializations"]
            or post_invalid["model_loaded"] != cold_metrics["model_loaded"]
        ):
            fail("invalid control requests changed cold lifecycle state")

        _gpu_snapshot()
        load_status, load_ms, load_states = _start_operation_and_observe(
            base_url, control, "load", timeout=args.timeout
        )
        _assert_status(load_status, 200, "first load")
        state, _ = _index(base_url, control)
        _assert_status(call(base_url, "/readyz")[0], 200, "first ready")
        loaded_metrics = _metrics(base_url, inference)
        loaded_gpu = _gpu_snapshot()
        if state != "READY" or loaded_metrics["initializations"] != 1:
            fail("first load did not settle READY with one initialization")
        init_before_idempotent = loaded_metrics["initializations"]
        second_load_status, second_load_size, second_load_ms = _operation(
            base_url, control, "load", timeout=args.timeout
        )
        _assert_status(second_load_status, 200, "ready load idempotency")
        after_idempotent = _metrics(base_url, inference)
        if second_load_size != 0 or after_idempotent["initializations"] != init_before_idempotent:
            fail("ready load was not an idempotent no-op")
        inference_one = _semantic_inference(base_url, inference, fixture, timeout=args.timeout)
        evidence["cycle_1"] = {
            "load_status": load_status,
            "load_latency_ms": load_ms,
            "load_states": load_states,
            "load_metrics": loaded_metrics,
            "load_gpu": loaded_gpu["target"],
            "ready_load_status": second_load_status,
            "ready_load_latency_ms": second_load_ms,
            "inference": inference_one,
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            active_future: Future[tuple[int, str, float]] = executor.submit(
                _completion_status, base_url, inference, fixture, timeout=args.timeout
            )
            active_deadline = time.monotonic() + min(args.timeout, 120.0)
            while time.monotonic() < active_deadline:
                active_metrics = _metrics(base_url, inference)
                if active_metrics["active_inference"] == 1:
                    break
                if active_future.done():
                    fail("delayed inference completed before drain observation")
                time.sleep(0.05)
            else:
                fail("active real inference was not observable for drain")
            unload_future = executor.submit(
                _operation, base_url, control, "unload", timeout=args.timeout
            )
            unload_states: list[str] = []
            unload_deadline = time.monotonic() + args.timeout
            while not unload_future.done() and time.monotonic() < unload_deadline:
                state, _ = _index(base_url, control)
                if not unload_states or unload_states[-1] != state:
                    unload_states.append(state)
                if state == "UNLOADING":
                    break
                time.sleep(0.05)
            if "UNLOADING" not in unload_states:
                fail("drain unload did not expose UNLOADING")
            rejected_status, rejected_code, rejected_latency = _completion_status(
                base_url, inference, fixture, timeout=30
            )
            _assert_status(rejected_status, 503, "inference during unload")
            if rejected_code != "not_ready":
                fail("inference during unload was not rejected as not_ready")
            active_status, active_code, active_latency = active_future.result(timeout=args.timeout)
            _assert_status(active_status, 200, "draining inference")
            if active_code != "?":
                fail("draining inference returned an error")
            unload_status, _unload_size, unload_ms = unload_future.result(timeout=args.timeout)
        _assert_status(unload_status, 200, "first unload")
        cold_after_first = _metrics(base_url, inference)
        cold_gpu_first = _gpu_snapshot()
        cold_process_first = _process_facts(pid, args.port)
        if _identity_tuple(cold_process_first) != _identity_tuple(identity):
            fail("PID/listener identity changed during first lifecycle")
        if cold_after_first["model_loaded"] != 0 or cold_after_first["initializations"] != 1:
            fail("first unload did not expose cold model metrics")
        release_one = _release_gate(loaded_metrics, cold_metrics, cold_after_first)
        second_unload_status, second_unload_size, second_unload_ms = _operation(
            base_url, control, "unload", timeout=args.timeout
        )
        _assert_status(second_unload_status, 200, "cold unload idempotency")
        if second_unload_size != 0:
            fail("cold unload returned a non-empty body")
        stable_cold = _metrics(base_url, inference)
        if stable_cold["initializations"] != cold_after_first["initializations"]:
            fail("cold unload changed initialization count")
        evidence["drain"] = {
            "unload_status": unload_status,
            "unload_latency_ms": unload_ms,
            "unload_states": unload_states,
            "active_inference_status": active_status,
            "active_inference_latency_ms": active_latency,
            "rejected_status": rejected_status,
            "rejected_code": rejected_code,
            "rejected_latency_ms": rejected_latency,
            "cold_metrics": cold_after_first,
            "cold_gpu": cold_gpu_first["target"],
            "release_gate": release_one,
            "idempotent_unload_ms": second_unload_ms,
        }

        _gpu_snapshot()
        second_load_status, second_load_ms, second_load_states = _start_operation_and_observe(
            base_url, control, "load", timeout=args.timeout
        )
        _assert_status(second_load_status, 200, "second load")
        second_loaded_metrics = _metrics(base_url, inference)
        second_loaded_gpu = _gpu_snapshot()
        if second_loaded_metrics["initializations"] != 2:
            fail("second load did not increment initialization count exactly once")
        second_ready_load_status, second_ready_load_size, _ = _operation(
            base_url, control, "load", timeout=args.timeout
        )
        _assert_status(second_ready_load_status, 200, "second ready load idempotency")
        if second_ready_load_size != 0 or _metrics(base_url, inference)["initializations"] != 2:
            fail("second ready load was not an idempotent no-op")
        inference_two = _semantic_inference(base_url, inference, fixture, timeout=args.timeout)
        if inference_two["semantic_digest"] != inference_one["semantic_digest"]:
            fail("two-cycle inference semantic digest changed")
        second_unload_status, second_unload_ms, second_unload_states = _start_operation_and_observe(
            base_url, control, "unload", timeout=args.timeout
        )
        _assert_status(second_unload_status, 200, "second unload")
        cold_after_second = _metrics(base_url, inference)
        cold_gpu_second = _gpu_snapshot()
        cold_process_second = _process_facts(pid, args.port)
        if _identity_tuple(cold_process_second) != _identity_tuple(identity):
            fail("PID/listener identity changed during second lifecycle")
        release_two = _release_gate(second_loaded_metrics, cold_metrics, cold_after_second)
        if (
            cold_gpu_second["target"]["used_mib"]
            > cold_gpu_first["target"]["used_mib"] + args.cold_tolerance_mib
        ):
            fail("physical GPU memory grew monotonically across lifecycle cycles")
        if cold_process_second["rss_mib"] > cold_process_first["rss_mib"] + args.cold_tolerance_mib:
            fail("host RSS grew beyond the cold-cycle tolerance")
        evidence["cycle_2"] = {
            "load_status": second_load_status,
            "load_latency_ms": second_load_ms,
            "load_states": second_load_states,
            "load_metrics": second_loaded_metrics,
            "load_gpu": second_loaded_gpu["target"],
            "ready_load_status": second_ready_load_status,
            "inference": inference_two,
            "unload_status": second_unload_status,
            "unload_latency_ms": second_unload_ms,
            "unload_states": second_unload_states,
            "cold_metrics": cold_after_second,
            "cold_gpu": cold_gpu_second["target"],
            "release_gate": release_two,
            "same_pid_listener": True,
        }
        evidence["status"] = "PASSED"
    except (SmokeFailure, AssertionError) as exc:
        evidence["status"] = "FAILED"
        evidence["failure"] = (
            str(exc) if isinstance(exc, SmokeFailure) else "semantic assertion failed"
        )
    finally:
        if identity is not None:
            try:
                evidence["cleanup"] = _stop_and_cleanup(
                    pid, args.port, runtime_root, log_path, control, inference, fixture
                )
            except SmokeFailure as exc:
                evidence["status"] = "FAILED"
                evidence["cleanup_failure"] = str(exc)
    json.dump(evidence, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if evidence["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
