"""CPU/API coverage for the request-local SAM2 configuration contract."""

from __future__ import annotations

from collections import Counter
import io
import json
import math
import zipfile

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from modules.segmenter import sam2
from src.core import StageFunctions, run_single_image
from src.postprocessing import filter_by_area_bbox
from src.runtime.models import APPROVED_MODEL_SPECS
from src.service import FakeEngine, ReadyState, ServiceError, ServiceSettings, create_app
from src.service.capabilities import CapabilitiesResponse
from src.service.gate import InferenceGate
from src.service.settings import SAM2_LIMIT_ENV_VARS
from src.service.yaml_input import parse_hostile_config


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 30, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _config(mask_generator: str = "") -> bytes:
    return ("alpha: 0.5\n" + mask_generator).encode()


SAM2_FIELDS = (
    "points_per_side",
    "points_per_batch",
    "pred_iou_thresh",
    "stability_score_thresh",
    "stability_score_offset",
    "mask_threshold",
    "box_nms_thresh",
    "crop_n_layers",
    "crop_nms_thresh",
    "crop_overlap_ratio",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
    "use_m2m",
    "multimask_output",
)
SAM2_INTEGER_FIELDS = (
    "points_per_side",
    "points_per_batch",
    "crop_n_layers",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
)
SAM2_NUMBER_FIELDS = (
    "pred_iou_thresh",
    "stability_score_thresh",
    "stability_score_offset",
    "mask_threshold",
    "box_nms_thresh",
    "crop_nms_thresh",
    "crop_overlap_ratio",
)
SAM2_BOOLEAN_FIELDS = ("use_m2m", "multimask_output")
SAM2_NONFINITE_YAML_SCALARS = (".nan", ".inf", "-.inf")


def _yaml_literal(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _sam2_config(field: str, value) -> bytes:
    return _config(f"mask_generator:\n  {field}: {_yaml_literal(value)}\n")


def _sam2_yaml_scalar_config(field: str, scalar: str) -> bytes:
    return _config(f"mask_generator:\n  {field}: {scalar}\n")


def _unrestricted_sam2_settings() -> ServiceSettings:
    return ServiceSettings(
        sam2_max_points_per_side=1024,
        sam2_max_points_per_batch=1024,
        sam2_max_crop_n_layers=8,
        sam2_max_estimated_prompts=20_000_000,
        sam2_max_estimated_mask_predictions=60_000_000,
        sam2_max_min_mask_region_area=64_000_000,
    )


def test_request_generator_forwards_all_safe_scalars_and_fixed_controls():
    assert sam2.SAM2_GENERATOR_FIELDS == SAM2_FIELDS
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
            "model": "must-not-forward",
            "path": "must-not-forward",
            "device": "must-not-forward",
            "dtype": "must-not-forward",
            "cache_dir": "must-not-forward",
            "arbitrary_kwargs": "must-not-forward",
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


def test_counted_ab_a_lifecycle_reuses_one_model_and_isolates_generators():
    activity = Counter()
    loaded_models = []

    class Model:
        def to(self, *_args, **_kwargs):
            activity["to"] += 1
            return self

        def half(self):
            activity["half"] += 1
            return self

    def load_model():
        activity["model_load"] += 1
        model = Model()
        loaded_models.append(model)
        return model

    model = load_model()
    generated = []
    proposal_sets = []
    generators = []
    holder = {"model": model}

    class Generator:
        def __init__(self, values):
            self.values = values

        def generate(self, image):
            crop_layer = self.values["crop_n_layers"]
            generated.append((self.values["points_per_side"], crop_layer))
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[: crop_layer + 1, : crop_layer + 1] = True
            proposal_sets.append(mask.copy())
            return [{"segmentation": mask, "crop_layer": crop_layer}]

    def factory(received_model, **kwargs):
        assert received_model is model
        assert "generator" not in holder
        generator = Generator(kwargs)
        generators.append(generator)
        return generator

    holder["generator_factory"] = factory
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    settings_a = dict(sam2.SAM2_DEFAULTS)
    settings_a.update(points_per_side=8, crop_n_layers=0)
    settings_b = dict(sam2.SAM2_DEFAULTS)
    settings_b.update(points_per_side=32, crop_n_layers=1)

    states = []
    for settings in (settings_a, settings_b, settings_a):
        state, masks, metadata = sam2.run(
            holder,
            {"mask_generator_config": settings, "dryrun": False},
            image,
        )
        states.append(state)
        assert len(masks) == metadata["num_masks"] == 1
        assert "generator" not in holder

    assert len(loaded_models) == 1
    assert states == [holder, holder, holder]
    assert len(generators) == 3
    assert len({id(generator) for generator in generators}) == 3
    assert all(generator is not holder.get("generator") for generator in generators)
    assert all(generator.values["points_per_side"] in {8, 32} for generator in generators)
    assert [(g.values["points_per_side"], g.values["crop_n_layers"]) for g in generators] == [
        (8, 0),
        (32, 1),
        (8, 0),
    ]
    assert generated == [(8, 0), (32, 1), (8, 0)]
    assert not np.array_equal(proposal_sets[0], proposal_sets[1])
    assert np.array_equal(proposal_sets[0], proposal_sets[2])
    assert activity == Counter({"model_load": 1})


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
        {"mask_generator_config": {"points_per_side": 8}, "dryrun": False},
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    second_state, _, second_meta = sam2.run(
        state,
        {"mask_generator_config": {"points_per_side": 32}, "dryrun": False},
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


@pytest.mark.parametrize(
    ("field", "lower", "upper"),
    [
        ("points_per_side", 1, 1024),
        ("points_per_batch", 1, 1024),
        ("pred_iou_thresh", 0.0, 1.0),
        ("stability_score_thresh", 0.0, 1.0),
        ("stability_score_offset", 0.0, 10.0),
        ("mask_threshold", -32.0, 32.0),
        ("box_nms_thresh", 0.0, 1.0),
        ("crop_n_layers", 0, 8),
        ("crop_nms_thresh", 0.0, 1.0),
        ("crop_overlap_ratio", 0.0, 1.0),
        ("crop_n_points_downscale_factor", 1, 32),
        ("min_mask_region_area", 0, 64_000_000),
    ],
)
def test_sam2_numeric_boundaries_are_accepted(field, lower, upper):
    settings = _unrestricted_sam2_settings()
    for value in (lower, upper):
        result = parse_hostile_config(_sam2_config(field, value), verbosity=0, settings=settings)
        assert result.sam2_metadata["effective"][field] == value


@pytest.mark.parametrize("field", SAM2_BOOLEAN_FIELDS)
def test_sam2_boolean_boundaries_are_accepted(field):
    for value in (False, True):
        result = parse_hostile_config(_sam2_config(field, value), verbosity=0)
        assert result.sam2_metadata["effective"][field] is value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("points_per_side", 0),
        ("points_per_side", 1025),
        ("points_per_batch", 0),
        ("points_per_batch", 1025),
        ("pred_iou_thresh", -0.01),
        ("pred_iou_thresh", 1.01),
        ("stability_score_thresh", -0.01),
        ("stability_score_thresh", 1.01),
        ("stability_score_offset", -0.01),
        ("stability_score_offset", 10.01),
        ("mask_threshold", -32.01),
        ("mask_threshold", 32.01),
        ("box_nms_thresh", -0.01),
        ("box_nms_thresh", 1.01),
        ("crop_n_layers", -1),
        ("crop_n_layers", 9),
        ("crop_nms_thresh", -0.01),
        ("crop_nms_thresh", 1.01),
        ("crop_overlap_ratio", -0.01),
        ("crop_overlap_ratio", 1.01),
        ("crop_n_points_downscale_factor", 0),
        ("crop_n_points_downscale_factor", 33),
        ("min_mask_region_area", -1),
        ("min_mask_region_area", 64_000_001),
    ],
)
def test_sam2_numeric_intrinsic_range_failures(field, value):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_sam2_config(field, value), verbosity=0)
    assert excinfo.value.code == "invalid_config"


@pytest.mark.parametrize(
    ("field", "value"),
    [(field, True) for field in SAM2_INTEGER_FIELDS + SAM2_NUMBER_FIELDS]
    + [(field, 1) for field in SAM2_BOOLEAN_FIELDS]
    + [(field, "8") for field in SAM2_INTEGER_FIELDS]
    + [(field, "0.5") for field in SAM2_NUMBER_FIELDS]
    + [(field, None) for field in SAM2_FIELDS]
    + [("profile", None), ("debug", None)],
)
def test_sam2_parser_strict_types_are_rejected(field, value):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_sam2_config(field, value), verbosity=0)
    assert excinfo.value.code == "invalid_config"
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", SAM2_NUMBER_FIELDS)
@pytest.mark.parametrize("value", ("NaN", ".inf", "-.inf"))
def test_sam2_quoted_nonfinite_strings_are_rejected_as_strings(field, value):
    raw = _sam2_config(field, value)
    parsed = yaml.safe_load(raw.decode("utf-8"))
    assert type(parsed["mask_generator"][field]) is str

    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=0)
    assert excinfo.value.code == "invalid_config"
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", SAM2_NUMBER_FIELDS)
@pytest.mark.parametrize("scalar", SAM2_NONFINITE_YAML_SCALARS)
def test_sam2_actual_yaml_nonfinite_numbers_are_rejected(field, scalar):
    raw = _sam2_yaml_scalar_config(field, scalar)
    parsed = yaml.safe_load(raw.decode("utf-8"))
    value = parsed["mask_generator"][field]
    assert type(value) is float
    assert not math.isfinite(value)

    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=0)
    assert excinfo.value.code == "invalid_config"
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", SAM2_NUMBER_FIELDS)
@pytest.mark.parametrize("scalar", SAM2_NONFINITE_YAML_SCALARS)
def test_api_rejects_actual_yaml_nonfinite_sam2_numbers_before_inference(field, scalar):
    raw = _sam2_yaml_scalar_config(field, scalar)
    parsed = yaml.safe_load(raw.decode("utf-8"))
    value = parsed["mask_generator"][field]
    assert type(value) is float
    assert not math.isfinite(value)

    engine = FakeEngine()
    readiness_calls = []

    def readiness():
        readiness_calls.append(True)
        return ReadyState(True, "ready")

    client = TestClient(create_app(engine=engine, readiness_provider=readiness))
    response = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", _png(), "image/png"),
            "config": ("config.yaml", raw, "application/yaml"),
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_config"
    assert field in body["error"]["message"]
    assert scalar not in response.text
    assert engine.calls == []
    assert readiness_calls == []


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("profile", "invalid_config"),
        ("typo", "unsupported_field"),
        ("point_grids", "unsupported_field"),
        ("output_mode", "unsupported_field"),
        ("checkpoint", "unsafe_config"),
        ("model_name", "unsafe_config"),
        ("path", "unsafe_config"),
        ("device", "unsafe_config"),
        ("dtype", "unsafe_config"),
        ("cache_dir", "unsafe_config"),
        ("arbitrary_kwargs", "unsupported_field"),
    ],
)
def test_sam2_profile_unknown_and_control_errors_are_distinct(field, code):
    value = "unknown" if field == "profile" else "value"
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_sam2_config(field, value), verbosity=0)
    assert excinfo.value.code == code


def test_sam2_deepest_crop_layer_must_retain_one_point():
    raw = _config(
        "mask_generator:\n"
        "  points_per_side: 1\n"
        "  crop_n_layers: 1\n"
        "  crop_n_points_downscale_factor: 2\n"
    )
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=0)
    assert excinfo.value.code == "invalid_config"
    assert "at least one point" in str(excinfo.value)


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


@pytest.mark.parametrize("profile", ("fast", "balanced", "quality"))
def test_each_profile_resolves_exact_partial_overrides(profile):
    result = parse_hostile_config(
        _config(f"mask_generator:\n  profile: {profile}\n"),
        verbosity=0,
    )
    expected = dict(sam2.SAM2_DEFAULTS)
    expected.update(sam2.SAM2_PROFILES[profile])
    assert result.sam2_metadata["effective"] == expected
    assert result.sam2_metadata["sources"] == {
        field: ("profile" if field in sam2.SAM2_PROFILES[profile] else "default")
        for field in SAM2_FIELDS
    }


def test_all_explicit_values_override_profile_even_when_equal_to_inherited():
    explicit = dict(sam2.SAM2_DEFAULTS)
    explicit["points_per_side"] = sam2.SAM2_PROFILES["quality"]["points_per_side"]
    lines = "mask_generator:\n  profile: quality\n" + "".join(
        f"  {field}: {_yaml_literal(value)}\n" for field, value in explicit.items()
    )
    result = parse_hostile_config(_config(lines), verbosity=0)
    assert result.sam2_metadata["effective"] == explicit
    assert result.sam2_metadata["sources"] == {field: "explicit" for field in SAM2_FIELDS}


@pytest.mark.parametrize(
    ("points", "layers", "downscale", "expected"),
    [(1, 0, 1, 1), (8, 1, 1, 320), (32, 1, 2, 2048), (16, 2, 2, 768)],
)
def test_prompt_formula_representative_crops(points, layers, downscale, expected):
    assert sam2.estimated_prompt_count(points, layers, downscale) == expected


@pytest.mark.parametrize(("multimask", "expected"), [(False, 64), (True, 192)])
def test_prompt_prediction_multiplier(multimask, expected):
    result = parse_hostile_config(
        _config(
            "mask_generator:\n"
            "  points_per_side: 8\n"
            f"  multimask_output: {_yaml_literal(multimask)}\n"
        ),
        verbosity=0,
    )
    assert result.sam2_metadata["estimated_prompt_count"] == 64
    assert result.sam2_metadata["estimated_mask_prediction_count"] == expected


@pytest.mark.parametrize(
    ("field", "requested", "cap"),
    [
        ("points_per_side", 8, "sam2_max_points_per_side"),
        ("points_per_batch", 8, "sam2_max_points_per_batch"),
        ("crop_n_layers", 1, "sam2_max_crop_n_layers"),
        ("min_mask_region_area", 10, "sam2_max_min_mask_region_area"),
    ],
)
def test_field_cap_equality_is_accepted_and_above_is_rejected(field, requested, cap):
    accepted_settings = ServiceSettings(**{cap: requested})
    accepted = parse_hostile_config(
        _sam2_config(field, requested), verbosity=0, settings=accepted_settings
    )
    assert accepted.sam2_metadata["effective"][field] == requested

    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            _sam2_config(field, requested + 1), verbosity=0, settings=accepted_settings
        )
    assert excinfo.value.code == "resource_limit"
    assert excinfo.value.status_code == 413


@pytest.mark.parametrize("estimate", ("estimated_prompt_count", "estimated_mask_prediction_count"))
def test_estimated_work_cap_equality_is_accepted_and_above_is_rejected(estimate):
    cap_field = (
        "sam2_max_estimated_prompts"
        if estimate == "estimated_prompt_count"
        else "sam2_max_estimated_mask_predictions"
    )
    accepted_settings = ServiceSettings(**{cap_field: 64})
    raw = _config("mask_generator:\n  points_per_side: 8\n  multimask_output: false\n")
    accepted = parse_hostile_config(raw, verbosity=0, settings=accepted_settings)
    assert accepted.sam2_metadata[estimate] == 64

    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=0, settings=ServiceSettings(**{cap_field: 63}))
    assert excinfo.value.code == "resource_limit"
    assert excinfo.value.status_code == 413


def test_resource_warnings_are_deterministic_at_eighty_percent_without_clamping():
    settings = ServiceSettings(
        sam2_max_estimated_prompts=80,
        sam2_max_estimated_mask_predictions=240,
    )
    result = parse_hostile_config(
        _config("mask_generator:\n  points_per_side: 8\n"),
        verbosity=0,
        settings=settings,
    )
    assert result.sam2_metadata["effective"]["points_per_side"] == 8
    assert result.sam2_metadata["estimated_prompt_count"] == 64
    assert result.sam2_metadata["estimated_mask_prediction_count"] == 192
    assert result.sam2_metadata["resource_warnings"] == [
        "estimated_prompt_count is at least 80% of its operator cap",
        "estimated_mask_prediction_count is at least 80% of its operator cap",
    ]


@pytest.mark.parametrize(
    ("env_name", "field_name", "value"),
    [
        (env_name, field_name, "2" if field_name == "sam2_max_crop_n_layers" else "9")
        for env_name, field_name in SAM2_LIMIT_ENV_VARS.items()
    ],
)
def test_each_sam2_operator_cap_environment_override_is_loaded(env_name, field_name, value):
    settings = ServiceSettings.from_environment({env_name: value})
    assert getattr(settings, field_name) == int(value)


@pytest.mark.parametrize(
    ("env_name", "value"),
    [(env_name, "not-an-integer") for env_name in SAM2_LIMIT_ENV_VARS]
    + [(env_name, "0") for env_name in SAM2_LIMIT_ENV_VARS if "CROP_N_LAYERS" not in env_name]
    + [("SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS", "-1")],
)
def test_sam2_operator_cap_environment_type_and_range_failures(env_name, value):
    with pytest.raises(ValueError):
        ServiceSettings.from_environment({env_name: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sam2_max_points_per_side", 1025),
        ("sam2_max_points_per_batch", 1025),
        ("sam2_max_crop_n_layers", 9),
        ("sam2_max_min_mask_region_area", 64_000_001),
    ],
)
def test_operator_caps_cannot_exceed_intrinsic_maxima(field, value):
    with pytest.raises(ValueError):
        ServiceSettings(**{field: value})


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


def test_capabilities_are_authenticated_static_deterministic_and_explicit(monkeypatch):
    calls = []
    engine = FakeEngine()

    def not_ready():
        calls.append("readiness")
        return ReadyState(False, "not ready")

    def forbidden_gate_acquisition(_self):
        raise AssertionError("capabilities must not acquire the inference gate")

    monkeypatch.setattr(InferenceGate, "slot", forbidden_gate_acquisition)

    app = create_app(
        engine=engine,
        settings=ServiceSettings(api_key="capability-test-key"),
        readiness_provider=not_ready,
    )
    client = TestClient(app)
    assert client.get("/v1/capabilities").status_code == 401
    wrong = client.get("/v1/capabilities", headers={"Authorization": "Bearer wrong-key"})
    assert wrong.status_code == 401
    response = client.get(
        "/v1/capabilities", headers={"Authorization": "Bearer capability-test-key"}
    )
    repeat = client.get("/v1/capabilities", headers={"Authorization": "Bearer capability-test-key"})
    assert response.status_code == 200
    assert repeat.status_code == 200
    assert response.content == repeat.content
    body = response.json()
    assert calls == []
    assert engine.calls == []
    assert body["schema_version"] == "zap-it.v1"
    assert body["model_id"] == "zap-it-1"
    assert body["defaults"] == sam2.SAM2_DEFAULTS
    assert set(body["profiles"]) == {"fast", "balanced", "quality"}
    assert set(body["supported_generator_fields"]) == {*SAM2_FIELDS, "profile", "debug"}
    expected_ranges = {
        "points_per_side": ("integer", 1, 1024),
        "points_per_batch": ("integer", 1, 1024),
        "pred_iou_thresh": ("number", 0.0, 1.0),
        "stability_score_thresh": ("number", 0.0, 1.0),
        "stability_score_offset": ("number", 0.0, 10.0),
        "mask_threshold": ("number", -32.0, 32.0),
        "box_nms_thresh": ("number", 0.0, 1.0),
        "crop_n_layers": ("integer", 0, 8),
        "crop_nms_thresh": ("number", 0.0, 1.0),
        "crop_overlap_ratio": ("number", 0.0, 1.0),
        "crop_n_points_downscale_factor": ("integer", 1, 32),
        "min_mask_region_area": ("integer", 0, 64_000_000),
    }
    for field, (kind, lower, upper) in expected_ranges.items():
        descriptor = body["supported_generator_fields"][field]
        assert descriptor["type"] == kind
        assert descriptor["minimum"] == lower
        assert descriptor["maximum"] == upper
        assert descriptor["description"]
        assert descriptor["stage"] == "sam2"
        assert body["intrinsic_ranges"][field] == [lower, upper]
    for field in SAM2_BOOLEAN_FIELDS:
        descriptor = body["supported_generator_fields"][field]
        assert descriptor["type"] == "boolean"
        assert descriptor["allowed"] == [False, True]
        assert descriptor["description"]
        assert body["intrinsic_ranges"][field] == [False, True]
    assert body["supported_generator_fields"]["profile"]["allowed"] == [
        "fast",
        "balanced",
        "quality",
    ]
    assert body["supported_generator_fields"]["debug"]["allowed"] == [False, True]
    assert body["operator_maxima"] == ServiceSettings().sam2_operator_caps
    assert body["blip3_question_capacity"]["max_questions"] == 256
    assert body["blip3_question_capacity"]["maximum"] == 256
    assert body["blip3_question_capacity"]["request_configurable"] is False
    assert body["defaults"] == {field: sam2.SAM2_DEFAULTS[field] for field in SAM2_FIELDS}
    assert body["profiles"] == {
        profile: sam2.SAM2_PROFILES[profile] for profile in ("fast", "balanced", "quality")
    }
    assert body["source_precedence"] == ["explicit", "profile", "default"]
    assert "4^layer" in body["estimation_formulas"]["estimated_prompt_count"]
    assert (
        "3 when multimask_output is true"
        in body["estimation_formulas"]["estimated_mask_prediction_count"]
    )
    assert body["fixed_controls"] == {
        "model": {
            "id": APPROVED_MODEL_SPECS["sam2"].model_id,
            "revision": APPROVED_MODEL_SPECS["sam2"].revision,
        },
        "checkpoint_path": "operator-managed; not disclosed",
        "config_path": "operator-managed; not disclosed",
        "device": "logical cuda:0 only",
        "gpu": "operator-selected GPU; physical details not disclosed",
        "dtype": "float16",
        "cache_paths": "operator-managed local cache; not disclosed",
        "residency": "pinned model resident; request-local generator",
        "artifact_destinations": "in-memory response; no client-selected destination",
        "point_grids": None,
        "output_mode": "binary_mask",
        "arbitrary_kwargs": False,
    }
    raw_policy = body["raw_sam2_debug"]
    assert "candidate_views" not in raw_policy
    assert body["candidate_views"]["clip"]["fixed_artifact_name"] == (
        "clip-candidate-view-CANDIDATE-0008.png"
    )
    assert body["candidate_views"]["blip3"]["fixed_artifact_name"] == (
        "blip3-verification-CANDIDATE-0008-QUESTION-0003.png"
    )
    assert raw_policy["trigger"] == "verbosity == 3 and mask_generator.debug == true"
    assert raw_policy["candidate_id_base"] == 1
    assert raw_policy["columns"] == 3
    assert raw_policy["rows"] == 4
    assert raw_policy["candidates_per_sheet"] == 12
    assert raw_policy["maximum_contact_sheets"] == 8
    assert raw_policy["maximum_represented_candidates"] == 96
    assert raw_policy["maximum_diagnostic_pixels"] == 2_000_000
    assert raw_policy["fixed_artifact_names"] == [
        *[f"sam2-candidates-page-{page:04d}.png" for page in range(1, 9)],
        "sam2-union-coverage.png",
        "sam2-overlap-heatmap.png",
        "sam2-uncovered-pixels.png",
    ]
    assert "three decimals" in raw_policy["score_format"]
    assert "may be enlarged" in raw_policy["diagnostics"]["candidate_tiles"]
    assert "first 96" in raw_policy["truncation"]
    serialized = json.dumps(body)
    assert "SLAIF_ZAP_IT_API_KEY" not in serialized
    assert "GPU-" not in serialized
    assert "426972" not in serialized
    assert all(path not in serialized for path in ("/dev/", "/srv/", "/tmp/"))
    assert client.get("/openapi.json").status_code == 200
    openapi = client.get("/openapi.json").json()
    assert "CapabilitiesResponse" in json.dumps(openapi)
    declared = set(CapabilitiesResponse.model_json_schema()["properties"])
    assert declared == set(openapi["components"]["schemas"]["CapabilitiesResponse"]["properties"])
    assert declared == set(body)
    assert set(openapi["components"]["schemas"]["CapabilitiesResponse"]["properties"]) == {
        "schema_version",
        "model_id",
        "supported_generator_fields",
        "intrinsic_ranges",
        "operator_maxima",
        "blip3_question_capacity",
        "defaults",
        "profiles",
        "source_precedence",
        "estimation_formulas",
        "fixed_controls",
        "raw_sam2_debug",
        "candidate_views",
        "configuration",
        "diagnostic_artifacts",
        "response_evidence",
    }


def test_capabilities_require_a_configured_inference_bearer():
    client = TestClient(
        create_app(
            engine=FakeEngine(),
            readiness_provider=lambda: ReadyState(True, "ready"),
        )
    )
    assert client.get("/v1/capabilities").status_code == 401


def test_private_lan_policy_can_disable_docs_and_openapi():
    client = TestClient(create_app(engine=FakeEngine(), enable_docs=False))
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_rejects_sam2_config_before_gate_or_engine(monkeypatch):
    engine = FakeEngine()
    readiness_calls = []

    def readiness():
        readiness_calls.append(True)
        return ReadyState(True, "ready")

    def forbidden_gate_acquisition(_self):
        raise AssertionError("rejected configuration must not acquire inference")

    monkeypatch.setattr(InferenceGate, "slot", forbidden_gate_acquisition)
    client = TestClient(create_app(engine=engine, readiness_provider=readiness))
    cases = (
        ("mask_generator:\n  points_per_side: 'secret-value'\n", 400, "invalid_config"),
        ("mask_generator:\n  points_per_side: 65\n", 413, "resource_limit"),
    )
    for mask_generator, status_code, code in cases:
        response = client.post(
            "/v1/completions",
            files={
                "image": ("frame.png", _png(), "image/png"),
                "config": ("config.yaml", _config(mask_generator), "application/yaml"),
            },
        )
        assert response.status_code == status_code
        body = response.json()
        assert body["error"]["code"] == code
        if code == "resource_limit":
            assert set(body["error"]) == {"code", "message", "request_id", "details"}
            assert body["error"]["details"]["admissible_alternatives"]
        else:
            assert set(body["error"]) == {"code", "message", "request_id"}
        assert mask_generator not in response.text
        assert all(
            secret not in response.text
            for secret in ("/", "credentials", "GPU-", "host", "secret-value")
        )
    assert engine.calls == []
    assert readiness_calls == []


@pytest.mark.parametrize("verbosity", (0, 1, 2, 3))
def test_sam2_manifest_is_complete_and_typed_at_every_verbosity(verbosity):
    client = TestClient(
        create_app(engine=FakeEngine(), readiness_provider=lambda: ReadyState(True, "ready"))
    )
    response = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", _png(), "image/png"),
            "config": (
                "config.yaml",
                _config("mask_generator:\n  profile: quality\n"),
                "application/yaml",
            ),
        },
        data={"verbosity": str(verbosity)},
    )
    assert response.status_code == 200, response.text
    service = response.json()["service"]
    metadata = service["sam2"]
    expected_effective = dict(sam2.SAM2_DEFAULTS)
    expected_effective.update(sam2.SAM2_PROFILES["quality"])
    assert set(metadata["effective"]) == set(SAM2_FIELDS)
    assert metadata["effective"] == expected_effective
    assert set(metadata["sources"]) == set(SAM2_FIELDS)
    assert all(
        metadata["sources"][field]
        == ("profile" if field in sam2.SAM2_PROFILES["quality"] else "default")
        for field in SAM2_FIELDS
    )
    assert metadata["requested"] == {"profile": "quality"}
    assert metadata["selected_profile"] == "quality"
    assert metadata["estimated_prompt_count"] == 2048
    assert metadata["estimated_mask_prediction_count"] == 6144
    assert metadata["actual_candidate_count"] == 2
    assert metadata["execution_time_ms"] == 0.5
    assert metadata["execution_time_ms"] >= 0
    assert round(metadata["execution_time_ms"], 3) == metadata["execution_time_ms"]
    assert metadata["resource_warnings"] == []
    assert service["verbosity"] == verbosity
    if verbosity == 0:
        assert "artifacts" not in service
        assert "objects" not in service
    if verbosity == 1:
        assert "artifacts" in service
        assert "objects" not in service
    if verbosity == 2:
        assert "artifacts" in service
        assert "objects" in service
        assert "candidate_counts" not in service
    if verbosity == 3:
        assert "artifacts" in service
        assert "objects" in service
        assert "candidate_counts" in service
        assert "timings_ms" in service
        assert "warnings" in service


def _service_without_volatile_artifact_data(service):
    normalized = json.loads(json.dumps(service))
    normalized.pop("request_id", None)
    for artifact in normalized.get("artifacts", []):
        artifact.pop("data", None)
    return normalized


@pytest.mark.parametrize("verbosity", (0, 1, 2, 3))
def test_json_and_zip_manifest_metadata_match_at_every_verbosity(verbosity):
    app = create_app(engine=FakeEngine(), readiness_provider=lambda: ReadyState(True, "ready"))
    client = TestClient(app)
    files = {
        "image": ("frame.png", _png(), "image/png"),
        "config": (
            "config.yaml",
            _config("mask_generator:\n  profile: quality\n"),
            "application/yaml",
        ),
    }
    json_response = client.post("/v1/completions", files=files, data={"verbosity": str(verbosity)})
    zip_response = client.post(
        "/v1/completions",
        files=files,
        data={"verbosity": str(verbosity), "response_format": "zip"},
    )
    assert json_response.status_code == zip_response.status_code == 200
    json_document = json_response.json()
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert json_document["choices"][0]["text"] == manifest["choices"][0]["text"]
    assert _service_without_volatile_artifact_data(
        json_document["service"]
    ) == _service_without_volatile_artifact_data(manifest["service"])


def _empty_candidate_core_engine(stage_calls):
    def engine(image_rgb, config, **kwargs):
        def apply_roi(image, _roi):
            height, width = image.shape[:2]
            return image, (0, 0, width, height)

        def resize_image(image, _resize):
            return image, {"mode": "native", "size": (image.shape[1], image.shape[0])}

        def run_sam2(state, _params, image, **_kwargs):
            stage_calls["sam2"] += 1
            nonempty = np.zeros(image.shape[:2], dtype=bool)
            nonempty[0, 0] = True
            empty = np.zeros(image.shape[:2], dtype=bool)
            return state, [{"segmentation": nonempty}, {"segmentation": empty}], {"num_masks": 2}

        def run_clip(state, params, _image, **_kwargs):
            stage_calls["clip"] += 1
            for mask in params["masks"]:
                mask["clip_label"] = "goat"
            return state, params["masks"], {}

        def run_blip3(state, params, _image, **_kwargs):
            stage_calls["blip3"] += 1
            return state, params["masks"], {}

        stages = StageFunctions(
            apply_roi=apply_roi,
            resize_image=resize_image,
            run_sam2=run_sam2,
            filter_by_area_bbox=filter_by_area_bbox,
            run_clip=run_clip,
            run_blip3=run_blip3,
            generate_visualizations=lambda *_args, **_kwargs: {},
        )
        return run_single_image(
            image_rgb,
            config,
            frame_id=kwargs.get("frame_id", "image"),
            segmenter_state=kwargs.get("segmenter_state"),
            clip_state=kwargs.get("clip_state"),
            blip3_state=kwargs.get("blip3_state"),
            dryrun=kwargs.get("dryrun", False),
            verbosity=kwargs.get("verbosity", 1),
            device=kwargs.get("device"),
            log_print_func=kwargs.get("log_print_func"),
            artifact_sink=kwargs.get("artifact_sink"),
            stages=stages,
            class_labels=kwargs.get("class_labels", ()),
            render_visualizations=kwargs.get("render_visualizations"),
            service_safe_artifact_names=kwargs.get("service_safe_artifact_names", False),
        )

    return engine


def test_l3_raw_candidate_count_survives_empty_remap_and_optional_stages():
    stage_calls = Counter()
    client = TestClient(
        create_app(
            engine=_empty_candidate_core_engine(stage_calls),
            readiness_provider=lambda: ReadyState(True, "ready"),
        )
    )
    config = _config(
        "clip:\n  labels:\n    goat: 'a goat'\nblip3:\n  goat:\n    question: 'is this a goat?'\n    trueresult: 'Yes'\n    falseresult: 'No'\n    newcategory: goat\n    falsecategory: negative\nclip_routing:\n  route_to_blip3:\n    labels: [goat]\n    top_k: 1\n    score_margin_from_best: null\n    minimum_target_score: null\n    uncertain_labels: []\n    max_candidates: null\n"
    )
    response = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", _png(), "image/png"),
            "config": ("config.yaml", config, "application/yaml"),
        },
        data={"verbosity": "3"},
    )
    assert response.status_code == 200, response.text
    service = response.json()["service"]
    assert service["sam2"]["actual_candidate_count"] == 2
    assert service["candidate_counts"]["sam2_candidates"] == 1
    assert service["candidate_counts"]["after_area_bbox"] == 1
    assert service["candidate_counts"]["after_clip"] == 1
    assert service["candidate_counts"]["final"] == 1
    assert stage_calls == Counter({"sam2": 1, "clip": 1, "blip3": 1})


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
