"""CPU/API coverage for the request-local SAM2 configuration contract."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from modules.segmenter import sam2
from src.service import FakeEngine, ReadyState, ServiceError, ServiceSettings, create_app
from src.service.yaml_input import parse_hostile_config


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 30, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _config(mask_generator: str = "") -> bytes:
    return ("alpha: 0.5\n" + mask_generator).encode()


def test_request_generator_forwards_all_safe_scalars_and_fixed_controls():
    model = object()
    calls = []

    def factory(received_model, **kwargs):
        calls.append((received_model, kwargs))
        return type("Generator", (), {"generate": lambda self, image: []})()

    values = dict(sam2.SAM2_DEFAULTS)
    values.update(
        {
            "points_per_side": 32,
            "points_per_batch": 31,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.8,
            "stability_score_offset": 2.0,
            "mask_threshold": -1.5,
            "box_nms_thresh": 0.6,
            "crop_n_layers": 1,
            "crop_nms_thresh": 0.65,
            "crop_overlap_ratio": 0.25,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 50,
            "use_m2m": True,
            "multimask_output": False,
            "profile": "quality",
            "debug": True,
            "point_grids": "must-not-forward",
            "output_mode": "must-not-forward",
        }
    )
    generator = sam2.build_request_generator(model, values, factory=factory)
    assert generator is not None
    assert calls == [
        (
            model,
            {
                **{key: values[key] for key in sam2.SAM2_GENERATOR_FIELDS},
                "point_grids": None,
                "output_mode": "binary_mask",
            },
        )
    ]


def test_model_only_run_creates_fresh_generators_without_state_writeback():
    model = object()
    generated = []

    class Generator:
        def __init__(self, values):
            self.values = values

        def generate(self, image):
            generated.append(self.values["points_per_side"])
            return []

    def factory(received_model, **kwargs):
        assert received_model is model
        return Generator(kwargs)

    state = {"model": model, "generator_factory": factory}
    first_state, _, first_meta = sam2.run(
        state,
        {
            "mask_generator_config": {"points_per_side": 8},
            "dryrun": False,
        },
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    second_state, _, second_meta = sam2.run(
        state,
        {
            "mask_generator_config": {"points_per_side": 32},
            "dryrun": False,
        },
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    assert first_state is second_state is state
    assert "mask_generator" not in state
    assert generated == [8, 32]
    assert first_meta["num_masks"] == second_meta["num_masks"] == 0


def test_exact_prompt_formula():
    assert sam2.estimated_prompt_count(8, 0, 1) == 64
    assert sam2.estimated_prompt_count(32, 1, 2) == 2048
    assert sam2.estimated_prompt_count(16, 2, 2) == 768


def test_profile_resolution_sources_and_metadata():
    result = parse_hostile_config(
        _config(
            "mask_generator:\n"
            "  profile: quality\n"
            "  points_per_side: 32\n"
            "  multimask_output: false\n"
        ),
        verbosity=0,
        settings=ServiceSettings(),
    )
    metadata = result.sam2_metadata
    assert metadata["requested"] == {
        "profile": "quality",
        "points_per_side": 32,
        "multimask_output": False,
    }
    assert metadata["effective"]["points_per_batch"] == 32
    assert metadata["sources"]["points_per_side"] == "explicit"
    assert metadata["sources"]["points_per_batch"] == "profile"
    assert metadata["sources"]["mask_threshold"] == "default"
    assert metadata["estimated_prompt_count"] == 2048
    assert metadata["estimated_mask_prediction_count"] == 2048


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("points_per_side", "true"),
        ("points_per_batch", "'8'"),
        ("pred_iou_thresh", "NaN"),
        ("stability_score_offset", "null"),
        ("use_m2m", "1"),
        ("multimask_output", "0"),
        ("debug", "1"),
    ],
)
def test_sam2_strict_types_are_rejected(field, value):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            _config(f"mask_generator:\n  {field}: {value}\n"),
            verbosity=3,
        )
    assert excinfo.value.code == "invalid_config"
    assert field in str(excinfo.value)


def test_sam2_unknown_and_unsafe_controls_have_distinct_errors():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_config("mask_generator:\n  point_grids: []\n"), verbosity=0)
    assert excinfo.value.code == "unsupported_field"

    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_config("mask_generator:\n  checkpoint: /model\n"), verbosity=0)
    assert excinfo.value.code == "unsafe_config"


def test_operator_caps_reject_without_clamping_and_warn_at_eighty_percent():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            _config("mask_generator:\n  points_per_side: 65\n"),
            verbosity=0,
            settings=ServiceSettings(sam2_max_points_per_side=64),
        )
    assert excinfo.value.code == "resource_limit"
    assert excinfo.value.status_code == 413

    result = parse_hostile_config(
        _config("mask_generator:\n  points_per_side: 8\n"),
        verbosity=0,
        settings=ServiceSettings(sam2_max_estimated_prompts=80),
    )
    assert result.sam2_metadata["estimated_prompt_count"] == 64
    assert result.sam2_metadata["resource_warnings"] == [
        "estimated_prompt_count is at least 80% of its operator cap"
    ]


def test_capabilities_are_authenticated_static_and_explicit():
    calls = []

    def not_ready():
        calls.append("readiness")
        return ReadyState(False, "not ready")

    app = create_app(
        engine=FakeEngine(),
        settings=ServiceSettings(api_key="capability-test-key"),
        readiness_provider=not_ready,
    )
    client = TestClient(app)
    assert client.get("/v1/capabilities").status_code == 401
    response = client.get(
        "/v1/capabilities", headers={"Authorization": "Bearer capability-test-key"}
    )
    assert response.status_code == 200
    body = response.json()
    assert calls == []
    assert body["schema_version"] == "zap-it.v1"
    assert body["model_id"] == "zap-it-1"
    assert body["defaults"] == sam2.SAM2_DEFAULTS
    assert set(body["profiles"]) == {"fast", "balanced", "quality"}
    assert body["operator_maxima"]["points_per_side"] == 64
    assert body["fixed_controls"]["output_mode"] == "binary_mask"
    assert body["fixed_controls"]["point_grids"] is None
    assert body["fixed_controls"]["arbitrary_kwargs"] is False
    serialized = json.dumps(body)
    assert "SLAIF_ZAP_IT_API_KEY" not in serialized
    assert "GPU-a914" not in serialized
    assert client.get("/openapi.json").status_code == 200
    assert "CapabilitiesResponse" in json.dumps(client.get("/openapi.json").json())


def test_capabilities_require_a_configured_inference_bearer():
    client = TestClient(
        create_app(
            engine=FakeEngine(),
            readiness_provider=lambda: ReadyState(True, "ready"),
        )
    )
    assert client.get("/v1/capabilities").status_code == 401


def test_manifest_is_present_at_l0_and_matches_zip():
    app = create_app(
        engine=FakeEngine(),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    client = TestClient(app)
    files = {
        "image": ("frame.png", _png(), "image/png"),
        "config": (
            "config.yaml",
            _config("mask_generator:\n  profile: quality\n"),
            "application/yaml",
        ),
    }
    json_response = client.post("/v1/completions", files=files, data={"verbosity": "0"})
    zip_response = client.post(
        "/v1/completions",
        files=files,
        data={"verbosity": "0", "response_format": "zip"},
    )
    assert json_response.status_code == zip_response.status_code == 200
    json_body = json_response.json()
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert json_body["service"]["sam2"] == manifest["service"]["sam2"]
    assert json_body["service"]["sam2"]["actual_candidate_count"] == 2
    assert json_body["service"]["sam2"]["execution_time_ms"] == 0.5
