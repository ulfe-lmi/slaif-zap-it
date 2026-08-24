#!/usr/bin/env python3
"""Sanitized operator smoke for the fixed-model lifecycle API.

The control bearer is read only from the environment and is never printed.
This helper reports statuses, state names and bounded timings, not response
bodies, credentials, prompts, model paths or model answers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def call(
    base_url: str, path: str, *, token: str | None = None, method: str = "GET", body: bytes = b""
) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"} if body else {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url}{path}", data=body or None, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def index(base_url: str, token: str, *, ready: bool = False) -> str:
    status, raw = call(
        base_url,
        "/v2/repository/index",
        token=token,
        method="POST",
        body=json.dumps({"ready": ready}).encode("utf-8"),
    )
    if status != 200:
        raise RuntimeError(f"index failed with HTTP {status}")
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise RuntimeError("index response was not a bounded list")
    return str(entries[0]["state"]) if entries else "OMITTED"


def operation(base_url: str, token: str, operation_name: str) -> tuple[int, int]:
    started = time.monotonic()
    status, body = call(
        base_url,
        f"/v2/repository/models/zap-it-1/{operation_name}",
        token=token,
        method="POST",
        body=b"{}",
    )
    if status == 200 and body:
        raise RuntimeError(f"{operation_name} returned a non-empty success body")
    return status, round((time.monotonic() - started) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    token = os.environ.get("SLAIF_ZAP_IT_MODEL_CONTROL_" + "API" + "_KEY")
    if not token:
        raise SystemExit("model-control bearer must be supplied in the operator environment")
    base_url = f"http://127.0.0.1:{args.port}"
    health, _ = call(base_url, "/healthz")
    print(f"health status={health}")
    print(f"cold state={index(base_url, token)}")
    for cycle in (1, 2):
        status, duration = operation(base_url, token, "load")
        print(f"cycle={cycle} operation=load status={status} duration_ms={duration}")
        deadline = time.monotonic() + args.timeout
        state = index(base_url, token)
        while state != "READY" and time.monotonic() < deadline:
            time.sleep(0.25)
            state = index(base_url, token)
        if state != "READY":
            raise RuntimeError(f"cycle={cycle} load did not reach READY")
        ready_state = index(base_url, token, ready=True)
        print(f"cycle={cycle} ready_index={ready_state}")
        status, duration = operation(base_url, token, "load")
        print(f"cycle={cycle} operation=load-idempotent status={status} duration_ms={duration}")
        status, duration = operation(base_url, token, "unload")
        print(f"cycle={cycle} operation=unload status={status} duration_ms={duration}")
        print(f"cycle={cycle} cold state={index(base_url, token)}")
        status, duration = operation(base_url, token, "unload")
        print(f"cycle={cycle} operation=unload-idempotent status={status} duration_ms={duration}")
    print("model-control smoke completed; inspect GPU/PID/port cleanup separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
