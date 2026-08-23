"""Objective 005 CPU checks for parity, budgets, RLE and metrics."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.core import ArtifactBudget, BoundedMemoryArtifactSink
from src.core.config import CoreConfig
from src.core.engine import StageFunctions, run_single_image
from src.service import ReadyState, ServiceSettings, create_app
from src.service.errors import ServiceError
from src.service.fake_engine import FakeEngine
from src.service.rle import MaskRLEError, decode_mask_rle, encode_mask_rle
from src.service import resources
from src.service.yaml_input import parse_hostile_config


def _png_bytes(width: int = 8, height: int = 6) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _files(config: bytes = b"alpha: 0.5\n"):
    return {
        "image": ("image.png", _png_bytes(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 3), dtype=bool),
        np.ones((2, 3), dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    ],
)
def test_mask_rle_round_trip_and_column_major(mask):
    encoded = encode_mask_rle(mask, max_runs=100)
    assert encoded["size"] == list(mask.shape)
    assert encoded["order"] == "column-major"
    assert np.array_equal(mask, decode_mask_rle(encoded))


def test_mask_rle_run_limit_fails_before_unbounded_growth():
    checkerboard = np.indices((8, 8)).sum(axis=0) % 2 == 0
    with pytest.raises(MaskRLEError):
        encode_mask_rle(checkerboard, max_runs=4)


def test_visualization_policy_rejects_panoptic_and_unsafe_ids():
    with pytest.raises(ServiceError) as panoptic:
        parse_hostile_config(
            b"visualization:\n  sam2:\n    - id: view\n      renderer: panoptic\n",
            verbosity=3,
        )
    assert panoptic.value.code == "unsupported_field"
    with pytest.raises(ServiceError) as unsafe:
        parse_hostile_config(
            b"visualization:\n  sam2:\n    - id: ../escape\n      renderer: annotated\n",
            verbosity=3,
        )
    assert unsafe.value.code == "unsafe_config"


def test_visualization_policy_is_bounded():
    entries = "\n".join(f"    - id: v{i}\n      renderer: annotated" for i in range(9))
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            ("visualization:\n  sam2:\n" + entries + "\n").encode(),
            verbosity=3,
        )
    assert excinfo.value.code == "response_too_large"


def test_bounded_sink_rejects_count_and_raw_byte_limits_without_residue():
    sink = BoundedMemoryArtifactSink(
        ArtifactBudget(max_artifacts=1, max_single_bytes=2, max_total_bytes=3)
    )
    sink.store_bytes("one.bin", b"12")
    with pytest.raises(ValueError):
        sink.store_bytes("two.bin", b"3")
    assert sink.names() == ("one.bin",)
    with pytest.raises(ValueError):
        sink.store_bytes("one.bin", b"123")
    assert sink.get("one.bin").data == b"12"


def test_service_l3_has_exact_rle_and_zip_manifest_parity():
    app = create_app(engine=FakeEngine(), readiness_provider=lambda: ReadyState(True, "ready"))
    with TestClient(app) as client:
        json_response = client.post("/v1/completions", files=_files(), data={"verbosity": "3"})
        zip_response = client.post(
            "/v1/completions",
            files=_files(),
            data={"verbosity": "3", "response_format": "zip"},
        )
    assert json_response.status_code == zip_response.status_code == 200
    document = json_response.json()
    object_rle = {item["instance_id"]: item["mask_rle"] for item in document["service"]["objects"]}
    for item in document["service"]["objects"]:
        decoded = decode_mask_rle(item["mask_rle"])
        assert int(decoded.sum()) == item["area_px"]
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert {
        item["instance_id"]: item["mask_rle"] for item in manifest["service"]["objects"]
    } == object_rle


def test_service_rejects_image_dimensions_before_pixel_allocation():
    app = create_app(
        engine=FakeEngine(),
        settings=ServiceSettings(max_image_width=4),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    with TestClient(app) as client:
        response = client.post("/v1/completions", files=_files())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_l0_l2_disable_core_visualization_execution():
    called = []

    def apply_roi(image, _roi):
        return image, (0, 0, image.shape[1], image.shape[0])

    def resize(image, _resize):
        return image, {"mode": "native"}

    def sam(state, _params, _image, **_kwargs):
        return state, [{"segmentation": np.eye(3, dtype=bool), "area": 3}], {}

    def filtered(masks, *_args, **_kwargs):
        return masks

    def visual(*_args, **_kwargs):
        called.append(True)
        return {"view": np.zeros((3, 3, 3), dtype=np.uint8)}

    stages = StageFunctions(
        apply_roi=apply_roi,
        resize_image=resize,
        run_sam2=sam,
        filter_by_area_bbox=filtered,
        run_clip=lambda state, params, image, **kwargs: (state, params["masks"], {}),
        run_blip3=lambda state, params, image, **kwargs: (state, params["masks"], {}),
        generate_visualizations=visual,
    )
    config = CoreConfig(
        alpha=0.5, roi_val=None, resize_val=None, prep_debug=False, vis_cfg={"sam2": []}
    )
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    run_single_image(image, config, stages=stages, verbosity=2, render_visualizations=False)
    assert called == []
    run_single_image(image, config, stages=stages, verbosity=3, render_visualizations=True)
    assert called == [True]


def test_metrics_are_custom_and_content_free():
    app = create_app(engine=FakeEngine(), readiness_provider=lambda: ReadyState(True, "ready"))
    with TestClient(app) as client:
        response = client.post("/v1/completions", files=_files(), data={"verbosity": "1"})
        metrics = client.get("/metrics")
    assert response.status_code == 200
    assert metrics.status_code == 200
    assert 'zap_it_requests_total{outcome="success"}' in metrics.text
    assert "python_info" not in metrics.text
    assert "image.png" not in metrics.text
    assert "config.yaml" not in metrics.text


@pytest.mark.parametrize(
    "resource_name, code",
    [("host_available_bytes", "insufficient_memory"), ("shm_free_bytes", "insufficient_shm")],
)
def test_resource_admission_fails_before_engine_and_recovers(monkeypatch, resource_name, code):
    monkeypatch.setattr(resources, resource_name, lambda *_args: 0)
    engine = FakeEngine()
    app = create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "ready"))
    with TestClient(app) as client:
        response = client.post("/v1/completions", files=_files())
    assert response.status_code == 507
    assert response.json()["error"]["code"] == code
    assert engine.calls == []
