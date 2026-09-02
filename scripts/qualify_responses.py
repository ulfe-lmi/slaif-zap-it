#!/usr/bin/env python3
"""Bounded operator qualification for ``POST /v1/responses``.

This script intentionally uses the official ``openai==3.7.0`` client for the
HTTP request and response parsing.  It prints only content-free status,
counts, sizes, hashes and timing; the bearer, request body and model output
text never enter stdout or the retained summary.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import stat
import tempfile
import time
from pathlib import Path

from openai import OpenAI
from openai.types.responses import Response as SDKResponse
from PIL import Image


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _small_png() -> bytes:
    image = Image.new("RGB", (32, 24), (80, 120, 180))
    pixels = image.load()
    for row in range(6, 18):
        for column in range(8, 24):
            pixels[column, row] = (220, 80, 40)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _safe_evidence_dir(root: str) -> Path:
    base = Path(root).resolve()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    base.chmod(0o700)
    path = Path(tempfile.mkdtemp(prefix="responses-", dir=base))
    path.chmod(0o700)
    return path


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor != -1:
            os.close(descriptor)


def qualify(host: str, port: int, evidence_root: str, output_png: str | None) -> dict[str, object]:
    key = os.environ.get("SLAIF_ZAP_IT_API_KEY")
    if not key:
        raise RuntimeError("SLAIF_ZAP_IT_API_KEY is required in the operator environment")
    image_bytes = _small_png()
    yaml_bytes = b"alpha: 0.5\nclip:\n  labels:\n    object: 'an object'\n"
    started = time.monotonic()
    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key=key,
    )
    response = client.responses.create(
        model="zap-it-1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": _data_url("image/png", image_bytes),
                    },
                    {
                        "type": "input_file",
                        "filename": "qualification.yaml",
                        "file_data": _data_url("application/yaml", yaml_bytes),
                    },
                ],
            }
        ],
        tools=[{"type": "image_generation"}],
    )
    elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
    if not isinstance(response, SDKResponse):
        raise RuntimeError("official SDK returned an unexpected response type")
    projection = json.loads(response.output_text)
    if projection.get("schema_version") != "zap-it.public.v1":
        raise RuntimeError("public projection version mismatch")
    if response.tool_choice != "auto":
        raise RuntimeError("effective tool-choice metadata mismatch")
    response_tools = list(response.tools)
    if len(response_tools) != 1 or response_tools[0].type != "image_generation":
        raise RuntimeError("echoed image-generation tool metadata mismatch")
    image_calls = [item for item in response.output if item.type == "image_generation_call"]
    if len(image_calls) != 1 or image_calls[0].status != "completed" or not image_calls[0].result:
        raise RuntimeError("expected exactly one completed image-generation output item")
    png_bytes = base64.b64decode(image_calls[0].result, validate=True)
    with Image.open(io.BytesIO(png_bytes)) as decoded:
        if decoded.format != "PNG" or decoded.size != (32, 24):
            raise RuntimeError("annotated PNG dimensions or format mismatch")
    summary: dict[str, object] = {
        "status": "PASSED",
        "sdk_version": "3.7.0",
        "response_object": response.object,
        "response_status": response.status,
        "tool_choice": response.tool_choice,
        "response_tool_count": len(response_tools),
        "response_tool_types": [tool.type for tool in response_tools],
        "object_count": len(projection.get("objects", [])),
        "image_call_count": len(image_calls),
        "projection_bytes": len(response.output_text.encode("utf-8")),
        "projection_sha256": hashlib.sha256(response.output_text.encode("utf-8")).hexdigest(),
        "png_bytes": len(png_bytes),
        "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "elapsed_ms": elapsed_ms,
    }
    evidence_dir = _safe_evidence_dir(evidence_root)
    _write_private(
        evidence_dir / "summary.json", json.dumps(summary, sort_keys=True).encode("utf-8")
    )
    if output_png is not None:
        _write_private(Path(output_png), png_bytes)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SLAIF_ZAP_IT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLAIF_ZAP_IT_PORT", "0")))
    parser.add_argument(
        "--evidence-root",
        default=os.environ.get("SLAIF_ZAP_IT_TMP_ROOT", "/dev/shm/slaif-zap-it"),
    )
    parser.add_argument("--output-png", default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be a valid TCP port")
    qualify(args.host, args.port, args.evidence_root, args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
