"""Objective 005 CPU checks for parity, budgets, RLE and metrics."""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.core import ArtifactBudget, BoundedMemoryArtifactSink
from src.core.config import CoreConfig
from src.core.engine import StageFunctions, run_single_image
from src.service import ReadyState, ServiceSettings, create_app
from src.service.envelope import ResponseContext, build_completion_json
from src.service.errors import ServiceError
from src.service.fake_engine import FakeEngine
from src.service.rle import MaskRLEError, decode_mask_rle, encode_mask_rle
from src.service import rle as rle_module
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


def test_mask_rle_chunked_uniform_and_checkerboard_are_deterministic(monkeypatch):
    monkeypatch.setattr(rle_module, "RLE_CHUNK_ELEMENTS", 17)
    uniform = np.zeros((41, 53), dtype=bool)
    uniform_encoded = encode_mask_rle(uniform, max_runs=10)
    assert uniform_encoded["counts"] == [41 * 53]
    assert np.array_equal(uniform, decode_mask_rle(uniform_encoded))

    checkerboard = np.indices((41, 53)).sum(axis=0) % 2 == 0
    first = encode_mask_rle(checkerboard, max_runs=10_000)
    second = encode_mask_rle(checkerboard, max_runs=10_000)
    assert first == second
    assert len(first["counts"]) > 41
    assert np.array_equal(checkerboard, decode_mask_rle(first))


def test_mask_rle_large_uniform_has_bounded_transition_work():
    mask = np.zeros((1024, 1024), dtype=bool)
    encoded = encode_mask_rle(mask, max_runs=4)
    assert encoded["counts"] == [mask.size]


def test_serialization_deadline_rejects_then_successfully_reuses_result():
    engine = FakeEngine()
    outcome = engine(
        np.zeros((6, 8, 3), dtype=np.uint8),
        CoreConfig(alpha=0.5, roi_val=None, resize_val=None, prep_debug=False),
    )
    expired = ResponseContext(
        request_id="deadline",
        model_id="zap-it-1",
        verbosity=3,
        response_format="json",
        config_digest="digest",
        class_mapping={},
        deadline_monotonic=time.monotonic() - 1,
    )
    with pytest.raises(ServiceError) as excinfo:
        build_completion_json(outcome, expired)
    assert excinfo.value.code == "timeout"

    recovered = ResponseContext(
        request_id="recovered",
        model_id="zap-it-1",
        verbosity=3,
        response_format="json",
        config_digest="digest",
        class_mapping={},
    )
    document = build_completion_json(outcome, recovered)
    assert document["service"]["objects"]


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


def test_visualization_raw_budget_rejects_before_engine_and_accepts_boundary():
    config = b"""alpha: 0.5
visualization:
  sam2:
    - id: view-a
      renderer: annotated
    - id: view-b
      renderer: alpha-overlay
"""
    raw_bytes = 8 * 6 * 3 * 2
    rejected_engine = FakeEngine()
    rejected_app = create_app(
        engine=rejected_engine,
        settings=ServiceSettings(
            max_single_artifact_bytes=raw_bytes,
            max_total_raw_artifact_bytes=raw_bytes - 1,
        ),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    with TestClient(rejected_app) as client:
        rejected = client.post("/v1/completions", files=_files(config), data={"verbosity": "3"})
    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "response_too_large"
    assert rejected_engine.calls == []

    observed_budgets = []

    class _BudgetSpy(FakeEngine):
        def __call__(self, *args, **kwargs):
            observed_budgets.append(kwargs["artifact_sink"].budget.max_total_bytes)
            return super().__call__(*args, **kwargs)

    accepted_app = create_app(
        engine=_BudgetSpy(),
        settings=ServiceSettings(
            max_single_artifact_bytes=raw_bytes,
            max_total_raw_artifact_bytes=raw_bytes,
        ),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    with TestClient(accepted_app) as client:
        accepted = client.post("/v1/completions", files=_files(config), data={"verbosity": "3"})
    assert accepted.status_code == 200
    assert observed_budgets == [0]


def test_l3_zero_visualization_streams_skip_hypothetical_raw_budget():
    raw_stream_bytes = 8 * 6 * 3
    settings = ServiceSettings(
        max_single_artifact_bytes=raw_stream_bytes - 1,
        max_total_raw_artifact_bytes=raw_stream_bytes - 1,
    )

    no_stream_engine = FakeEngine()
    no_stream_app = create_app(
        engine=no_stream_engine,
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    with TestClient(no_stream_app) as client:
        no_stream = client.post("/v1/completions", files=_files(), data={"verbosity": "3"})
    assert no_stream.status_code == 200
    assert len(no_stream_engine.calls) == 1

    configured_engine = FakeEngine()
    configured_app = create_app(
        engine=configured_engine,
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    config = b"""alpha: 0.5
visualization:
  sam2:
    - id: view
      renderer: annotated
"""
    with TestClient(configured_app) as client:
        configured = client.post(
            "/v1/completions",
            files=_files(config),
            data={"verbosity": "3"},
        )
    assert configured.status_code == 413
    assert configured.json()["error"]["code"] == "response_too_large"
    assert configured_engine.calls == []


@pytest.mark.parametrize("response_format", ["json", "zip"])
def test_serialization_timeout_metrics_are_exclusive_and_recover(response_format):
    settings = ServiceSettings(
        request_deadline_seconds=0.5,
        test_serialization_delay_seconds=0.75,
    )
    app = create_app(
        engine=FakeEngine(),
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "ready"),
    )

    def metric_value(text, name, labels=None):
        for line in text.splitlines():
            if not line.startswith(name + ("{" if labels else " ")):
                continue
            if labels and not all(label in line for label in labels.split(",")):
                continue
            if " " in line:
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    with TestClient(app) as client:
        timed_out = client.post(
            "/v1/completions",
            files=_files(),
            data={"verbosity": "2", "response_format": response_format},
        )
        assert timed_out.status_code == 504
        timed_out_metrics = client.get("/metrics").text
        assert metric_value(timed_out_metrics, "zap_it_timeout_total") == 1
        assert metric_value(timed_out_metrics, "zap_it_requests_total", 'outcome="success"') == 0
        assert (
            metric_value(
                timed_out_metrics,
                "zap_it_completions_total",
                f'verbosity="2",response_format="{response_format}"',
            )
            == 0
        )
        assert metric_value(timed_out_metrics, "zap_it_response_bytes_count") == 0
        assert metric_value(timed_out_metrics, "zap_it_object_count_count") == 0
        assert metric_value(timed_out_metrics, "zap_it_artifact_count_count") == 0
        assert metric_value(timed_out_metrics, "zap_it_serialization_duration_seconds_count") == 1

        object.__setattr__(settings, "test_serialization_delay_seconds", 0.0)
        recovered = client.post(
            "/v1/completions",
            files=_files(),
            data={"verbosity": "2", "response_format": response_format},
        )
        assert recovered.status_code == 200
        recovered_metrics = client.get("/metrics").text
        assert metric_value(recovered_metrics, "zap_it_timeout_total") == 1
        assert metric_value(recovered_metrics, "zap_it_requests_total", 'outcome="success"') == 1
        assert (
            metric_value(
                recovered_metrics,
                "zap_it_completions_total",
                f'verbosity="2",response_format="{response_format}"',
            )
            == 1
        )
        assert metric_value(recovered_metrics, "zap_it_response_bytes_count") == 1
        assert metric_value(recovered_metrics, "zap_it_object_count_count") == 1
        assert metric_value(recovered_metrics, "zap_it_artifact_count_count") == 1
        assert metric_value(recovered_metrics, "zap_it_serialization_duration_seconds_count") == 2


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_l0_l2_visualization_config_does_not_trigger_raw_preflight(verbosity):
    config = b"""alpha: 0.5
visualization:
  sam2:
    - id: view
      renderer: annotated
"""
    engine = FakeEngine()
    app = create_app(
        engine=engine,
        settings=ServiceSettings(max_single_artifact_bytes=128),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions", files=_files(config), data={"verbosity": str(verbosity)}
        )
    assert response.status_code == 200
    assert len(engine.calls) == 1


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


def test_resource_admission_recovers_without_resident_engine_leak(monkeypatch):
    engine = FakeEngine()
    app = create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "ready"))
    monkeypatch.setattr(resources, "host_available_bytes", lambda: 0)
    with TestClient(app) as client:
        rejected = client.post("/v1/completions", files=_files())
        assert rejected.status_code == 507
        assert rejected.json()["error"]["code"] == "insufficient_memory"

        monkeypatch.setattr(resources, "host_available_bytes", lambda: 1 << 60)
        recovered = client.post("/v1/completions", files=_files())
    assert recovered.status_code == 200
    assert len(engine.calls) == 1
