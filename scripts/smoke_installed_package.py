#!/usr/bin/env python3
"""Prove the installed wheel exposes the service without checkout imports."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from importlib import import_module
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import src
from src.service import ReadyState, create_app
from src.service.fake_engine import FakeEngine


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), (32, 64, 96)).save(output, format="PNG")
    return output.getvalue()


def _files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "image": ("frame.png", _png_bytes(), "image/png"),
        "config": (
            "config.yaml",
            b"alpha: 0.5\nclip:\n  labels:\n    synthetic: 'a synthetic object'\n",
            "application/yaml",
        ),
    }


def main() -> int:
    live_service = import_module("src.runtime.live_service")
    module_path = Path(live_service.__file__).resolve()
    checkout_root = Path(__file__).resolve().parents[1]
    if checkout_root in module_path.parents:
        raise RuntimeError("installed smoke unexpectedly imported the checkout")
    if src.__version__ != "0.1.0":
        raise RuntimeError(f"unexpected package version: {src.__version__!r}")
    if shutil.which("zap-it-service") is None:
        raise RuntimeError("zap-it-service console script is not installed")

    app = create_app(
        engine=FakeEngine(),
        readiness_provider=lambda: ReadyState(True, "installed fake engine ready"),
    )
    evidence: dict[str, object] = {
        "package_version": src.__version__,
        "live_service_module": "site-packages",
        "console_script": True,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        json_response = client.post("/v1/completions", files=_files(), data={"verbosity": "2"})
        if json_response.status_code != 200:
            raise RuntimeError(f"installed JSON smoke failed: {json_response.status_code}")
        json_document = json_response.json()
        if json_document["service"]["package_version"] != "0.1.0":
            raise RuntimeError("installed JSON provenance is not 0.1.0")

        zip_response = client.post(
            "/v1/completions",
            files=_files(),
            data={"verbosity": "2", "response_format": "zip"},
        )
        if zip_response.status_code != 200:
            raise RuntimeError(f"installed ZIP smoke failed: {zip_response.status_code}")
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        if manifest["service"]["package_version"] != "0.1.0":
            raise RuntimeError("installed ZIP provenance is not 0.1.0")

    evidence["json_package_version"] = json_document["service"]["package_version"]
    evidence["zip_package_version"] = manifest["service"]["package_version"]
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
