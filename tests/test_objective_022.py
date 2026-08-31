"""CPU proofs for canonical CLIP multi-prompt validation and evidence."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from modules.classifier import clip as clip_module
from modules.verifier.blip3 import Blip3ResourceLimitError, _Blip3Filter
from src.core.clip_prompts import ClipPromptValidationError, validate_clip_prompt_tokens
from src.runtime.live_service import ResidentRegistry, live_engine_callable
from src.service import FakeEngine, ReadyState, create_app
from src.service.capabilities import build_capabilities
from src.service.errors import ServiceError
from src.service.schemas import ClipPromptValidationDetails, ErrorEnvelope
from src.service.settings import ServiceSettings
from src.service.yaml_input import parse_hostile_config


REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_TOMATO_CONFIG = REPO_ROOT / "tests" / "fixtures" / "configs" / "ripe-tomato-multiprompt.yaml"


def _config(labels: dict[str, object]) -> bytes:
    return yaml.safe_dump({"alpha": 0.5, "clip": {"labels": labels}}).encode()


def _png_bytes(height: int = 12, width: int = 12) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_canonical_scalar_and_arrays_normalize_without_splitting_or_joining():
    parsed = parse_hostile_config(
        _config(
            {
                "first": "  one, literal\nline  ",
                "second": [" first ", "second", "third"],
                "reuse": ["first"],
            }
        ),
        verbosity=3,
    )
    assert parsed.effective_mapping["clip"]["labels"] == {
        "first": ["one, literal\nline"],
        "second": ["first", "second", "third"],
        "reuse": ["first"],
    }
    assert parsed.clip_prompt_metadata == {
        "class_prompt_counts": {"first": 1, "second": 3, "reuse": 1},
        "total_prompt_count": 5,
        "tokenizer_limit": 77,
        "duplicate_policy": "reject",
    }


@pytest.mark.parametrize(
    ("labels", "reason"),
    [
        ({"thing": []}, "empty_prompt_array"),
        ({"thing": ["ok", 4]}, "invalid_prompt_type"),
        ({"thing": ["same", " same "]}, "duplicate_prompt"),
        ({"thing": " "}, "empty_prompt"),
        ({"thing": "x" * 513}, "character_limit"),
    ],
)
def test_canonical_prompt_structural_errors_are_typed_and_sanitized(labels, reason):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(_config(labels), verbosity=3)
    error = excinfo.value
    assert error.code == "invalid_config"
    assert error.details["reason"] == reason
    details = ErrorEnvelope.model_validate(error.envelope("request")).error.details
    assert isinstance(details, ClipPromptValidationDetails)
    assert "same" not in str(details)


def test_canonical_prompt_count_boundaries_and_total_overflow_are_deterministic():
    at_class_limit = parse_hostile_config(
        _config({"thing": [f"prompt {index}" for index in range(64)]}), verbosity=3
    )
    assert at_class_limit.clip_prompt_metadata["total_prompt_count"] == 64
    with pytest.raises(ServiceError) as too_many_per_class:
        parse_hostile_config(
            _config({"thing": [f"prompt {index}" for index in range(65)]}), verbosity=3
        )
    assert too_many_per_class.value.details == {
        "reason": "per_class_count",
        "allowed_limit": 64,
        "class_identifier": "thing",
        "prompt_index": 64,
        "measured_per_class_count": 65,
    }

    labels = {f"class_{index}": [f"prompt {index}"] for index in range(32)}
    with pytest.raises(ServiceError) as too_many_classes:
        parse_hostile_config(_config({**labels, "overflow": ["prompt"]}), verbosity=3)
    assert too_many_classes.value.details["reason"] == "too_many_classes"
    assert too_many_classes.value.details["measured_class_count"] == 33

    labels = {
        f"class_{index}": [f"prompt {index}_{item}" for item in range(64)] for index in range(4)
    }
    labels["overflow"] = ["first overflow prompt"]
    with pytest.raises(ServiceError) as total:
        parse_hostile_config(_config(labels), verbosity=3)
    assert total.value.details == {
        "reason": "total_count",
        "allowed_limit": 256,
        "class_identifier": "overflow",
        "prompt_index": 0,
        "measured_total_count": 257,
    }


class _TokenCounter:
    model_max_length = 77

    def __call__(self, prompt, **_kwargs):
        count = 78 if prompt == "too long" else 77
        return {"input_ids": list(range(count))}


class _Processor:
    tokenizer = _TokenCounter()


def test_exact_tokenizer_preflight_accepts_77_and_rejects_78_before_engine():
    processor = _Processor()
    assert validate_clip_prompt_tokens(processor, {"thing": ["a", "b"]}) == (
        tuple(range(77)),
        tuple(range(77)),
    )
    with pytest.raises(ClipPromptValidationError) as token_error:
        validate_clip_prompt_tokens(processor, {"thing": ["ok", "too long"]})
    assert token_error.value.details == {
        "reason": "token_limit",
        "allowed_limit": 77,
        "class_identifier": "thing",
        "prompt_index": 1,
        "measured_token_count": 78,
    }

    calls = []

    def validator(clip_config):
        calls.append(clip_config)
        validate_clip_prompt_tokens(processor, clip_config["labels"])

    engine = FakeEngine()
    app = create_app(
        engine=engine,
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
        clip_prompt_validator=validator,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "config": ("config.yaml", _config({"thing": "too long"}), "application/yaml"),
        },
        data={"verbosity": "0"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_config"
    assert body["error"]["details"]["measured_token_count"] == 78
    assert engine.calls == []
    assert len(calls) == 1


def test_live_adapter_maps_defense_prompt_errors_to_400_and_unrelated_errors_to_500():
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})
    registry.load()

    def typed_failure(*_args, **_kwargs):
        raise ClipPromptValidationError(
            "canonical CLIP prompt exceeds the tokenizer context limit",
            {
                "reason": "token_limit",
                "allowed_limit": 77,
                "class_identifier": "thing",
                "prompt_index": 0,
                "measured_token_count": 78,
            },
        )

    app = create_app(
        engine=live_engine_callable(registry, runner=typed_failure),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    response = TestClient(app).post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "config": ("config.yaml", _config({"thing": "ok"}), "application/yaml"),
        },
        data={"verbosity": "0"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "invalid_config",
        "message": "canonical CLIP prompt exceeds the tokenizer context limit",
        "request_id": response.json()["error"]["request_id"],
        "details": {
            "reason": "token_limit",
            "allowed_limit": 77,
            "class_identifier": "thing",
            "prompt_index": 0,
            "measured_token_count": 78,
        },
    }

    def unrelated_failure(*_args, **_kwargs):
        raise RuntimeError("model runtime failure")

    unrelated_app = create_app(
        engine=live_engine_callable(registry, runner=unrelated_failure),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    unrelated = TestClient(unrelated_app).post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "config": ("config.yaml", _config({"thing": "ok"}), "application/yaml"),
        },
        data={"verbosity": "0"},
    )
    assert unrelated.status_code == 500
    assert unrelated.json()["error"]["code"] == "inference_failure"


@pytest.mark.parametrize("value", ["", "0", "-1", "257", "not-an-integer"])
def test_blip3_question_capacity_operator_values_fail_closed(value):
    with pytest.raises(ValueError, match="1 to 256"):
        ServiceSettings.from_environment({"SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS": value})


def test_blip3_question_capacity_accepts_32_and_256_without_model_loading():
    assert ServiceSettings().blip3_max_questions == 256
    assert (
        ServiceSettings.from_environment(
            {"SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS": "32"}
        ).blip3_max_questions
        == 32
    )
    assert (
        ServiceSettings.from_environment(
            {"SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS": "256"}
        ).blip3_max_questions
        == 256
    )
    capabilities = build_capabilities(ServiceSettings(blip3_max_questions=32))
    assert capabilities["blip3_question_capacity"] == {
        "max_questions": 32,
        "default": 256,
        "maximum": 256,
        "units": "questions/request",
        "stage": "BLIP3 planning before generation",
        "controlling_field": "SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS",
        "request_configurable": False,
        "notes": (
            "Canonical routing plans at most one question per routed candidate; "
            "legacy multi-rule scheduling shares this total cap."
        ),
    }


def test_blip3_planning_accepts_256_and_rejects_257_before_model_calls():
    filter_ = _Blip3Filter.__new__(_Blip3Filter)
    filter_.max_questions = 256
    label_rules = {"target": {"question": "q"}}
    any_rules = []
    planned = [
        {"clip_routing": {"chosen_target": "target"}, "_route_to_blip3": True} for _ in range(256)
    ]
    assert filter_._planned_question_count(planned, any_rules, label_rules) == 256
    with pytest.raises(Blip3ResourceLimitError) as error:
        filter_._enforce_question_limit(257)
    assert error.value.details == {
        "planned_questions": 257,
        "allowed_limit": 256,
        "controlling_field": "SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS",
        "admissible_alternatives": [
            "set clip_routing.max_candidates to a lower deterministic cap",
            "tighten the clip_routing predicates to route fewer candidates",
        ],
    }


def test_live_adapter_maps_blip3_over_limit_to_structured_resource_413():
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})
    registry.load()

    def over_limit(*_args, **_kwargs):
        raise Blip3ResourceLimitError(
            "BLIP3 candidate count exceeds the fixed 256-question limit",
            planned_questions=257,
            allowed_limit=256,
        )

    app = create_app(
        engine=live_engine_callable(registry, runner=over_limit),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    response = TestClient(app).post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "config": ("config.yaml", _config({"thing": "ok"}), "application/yaml"),
        },
        data={"verbosity": "0"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "resource_limit"
    assert response.json()["error"]["details"] == {
        "planned_questions": 257,
        "allowed_limit": 256,
        "controlling_field": "SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS",
        "admissible_alternatives": [
            "set clip_routing.max_candidates to a lower deterministic cap",
            "tighten the clip_routing predicates to route fewer candidates",
        ],
    }


def test_classifier_aggregates_individual_prompt_scores_by_class_with_low_index_ties():
    class _Scalar:
        def __init__(self, value):
            self.value = value

        def __float__(self):
            return float(self.value)

    class _Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float64)

        @property
        def T(self):
            return _Tensor(self.data.T)

        def numel(self):
            return int(self.data.size)

        def norm(self, dim=-1, keepdim=False):
            return _Tensor(np.linalg.norm(self.data, axis=dim, keepdims=keepdim))

        def __truediv__(self, other):
            return _Tensor(self.data / other.data)

        def __getitem__(self, index):
            value = self.data[index]
            return _Scalar(value) if np.ndim(value) == 0 else _Tensor(value)

    class _Torch:
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        def no_grad(self):
            return self._NoGrad()

        def is_tensor(self, value):
            return isinstance(value, _Tensor)

        def matmul(self, left, right):
            return _Tensor(np.matmul(left.data, right.data))

    class _Processor:
        def __call__(self, **_kwargs):
            return {}

    class _Model:
        def get_image_features(self, **_kwargs):
            return _Tensor([[1.0, 0.0]])

    classifier = clip_module._ClipFilter.__new__(clip_module._ClipFilter)
    classifier._torch = _Torch()
    classifier.processor = _Processor()
    classifier.model = _Model()
    classifier.model_dtype = None
    classifier.device = "cpu"
    classifier.text_embeds = _Tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    classifier.class_map = {"target": ["weak", "strong"], "other": ["same"]}
    classifier.class_idx = ["target", "target", "other"]
    classifier.class_prompt_idx = [0, 1, 0]
    classifier.all_prompts = ["weak", "strong", "same"]
    classifier.verbosity = 0
    classifier.log_print = lambda *_args, **_kwargs: None

    result = classifier.classify_single_scores_detailed(np.zeros((2, 2, 3), dtype=np.uint8), 0)
    assert result[3] == {"target": 1.0, "other": 1.0}
    assert result[4] == {"target": 1, "other": 0}
    assert result[5] == 1
    assert classifier.classify_single(np.zeros((2, 2, 3), dtype=np.uint8), 0) == (
        "target",
        1.0,
        "strong",
    )


def test_text_encoder_sends_each_prompt_with_bounded_options_and_unchanged_ids():
    class _Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float64)

        def norm(self, dim=-1, keepdim=False):
            return _Tensor(np.linalg.norm(self.data, axis=dim, keepdims=keepdim))

        def __truediv__(self, other):
            return _Tensor(self.data / other.data)

        def numel(self):
            return int(self.data.size)

    class _Torch:
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        def no_grad(self):
            return self._NoGrad()

        def is_tensor(self, value):
            return isinstance(value, _Tensor)

    class _Tokenizer:
        model_max_length = 77

        def __call__(self, prompt, **_kwargs):
            offset = 1 if prompt == "one" else 2
            return {"input_ids": list(range(offset, offset + 77))}

    class _Processor:
        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.calls = []

        def __call__(self, *, text, return_tensors, padding, max_length, truncation):
            self.calls.append(
                {
                    "text": list(text),
                    "return_tensors": return_tensors,
                    "padding": padding,
                    "max_length": max_length,
                    "truncation": truncation,
                }
            )
            return {"input_ids": [self.tokenizer(item)["input_ids"] for item in text]}

    class _Model:
        def get_text_features(self, **_inputs):
            return _Tensor([[1.0], [1.0]])

    processor = _Processor()
    classifier = clip_module._ClipFilter.__new__(clip_module._ClipFilter)
    classifier._torch = _Torch()
    classifier.processor = processor
    classifier.model = _Model()
    classifier.model_dtype = None
    classifier.device = "cpu"
    classifier.class_map = {"thing": ["one", "two"]}
    classifier._rebuild_prompt_index()
    classifier._encode_text_prompts()
    assert processor.calls == [
        {
            "text": ["one", "two"],
            "return_tensors": "pt",
            "padding": True,
            "max_length": 77,
            "truncation": True,
        }
    ]
    assert classifier.text_embeds.numel() == 2


def test_fake_l3_prompt_summary_and_capability_contract_are_bounded():
    labels = {
        "target": [f"target {index}" for index in range(2)],
        "other": "other",
    }
    app = create_app(
        engine=FakeEngine(),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    response = TestClient(app).post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(), "image/png"),
            "config": ("config.yaml", _config(labels), "application/yaml"),
        },
        data={"verbosity": "3"},
    )
    assert response.status_code == 200
    assert response.json()["service"]["clip_prompts"] == {
        "class_prompt_counts": {"target": 2, "other": 1},
        "total_prompt_count": 3,
        "tokenizer_limit": 77,
        "duplicate_policy": "reject",
    }

    capabilities = build_capabilities(ServiceSettings())
    descriptor = capabilities["configuration"]["sections"]["clip"]["fields"]["labels.<identifier>"]
    assert descriptor["type"] == "string_or_array"
    assert descriptor["value_types"] == ["string", "array"]
    assert descriptor["max_items"] == 64
    assert capabilities["candidate_views"]["clip_labels"]["prompt"]["type"] == "string_or_array"
    required = {
        item["path"]: item["required"] for item in capabilities["configuration"]["field_catalog"]
    }
    for field in ("question", "trueresult", "falseresult", "newcategory", "falsecategory"):
        assert required[f"blip3.<routing_label>.{field}"] is True


def test_exact_97_prompt_shape_reaches_fake_engine_with_five_semantic_classes():
    raw = EXACT_TOMATO_CONFIG.read_bytes()
    parsed = parse_hostile_config(raw, verbosity=3)
    labels = parsed.effective_mapping["clip"]["labels"]
    assert parsed.clip_prompt_metadata["class_prompt_counts"] == {
        "ripe_tomato": 32,
        "foliage": 15,
        "stem_or_vine": 15,
        "greenhouse_structure": 20,
        "background": 15,
    }
    assert parsed.clip_prompt_metadata["total_prompt_count"] == 97
    assert list(labels) == [
        "ripe_tomato",
        "foliage",
        "stem_or_vine",
        "greenhouse_structure",
        "background",
    ]
    assert parsed.effective_mapping["clip_routing"]["route_to_blip3"]["labels"] == ["ripe_tomato"]
    assert parsed.effective_mapping["blip3"]["ripe_tomato"]["falsecategory"] == "negative"

    from src.core.config import CoreConfig

    outcome = FakeEngine()(  # type: ignore[call-arg]
        np.zeros((80, 120, 3), dtype=np.uint8),
        CoreConfig.from_mapping(parsed.effective_mapping),
        class_labels=tuple(labels),
        verbosity=3,
    )
    assert outcome.result.clip_prompt_metadata["total_prompt_count"] == 97
    assert all(list(obj.metadata["clip_scores"]) == list(labels) for obj in outcome.result.objects)
    assert all(
        diagnostic["chosen_target"] == "ripe_tomato"
        and set(diagnostic["clip_scores"]) == set(labels)
        for diagnostic in outcome.result.clip_routing_diagnostics
    )

    app = create_app(
        engine=FakeEngine(),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    response = TestClient(app).post(
        "/v1/completions",
        files={
            "image": ("image.png", _png_bytes(80, 120), "image/png"),
            "config": ("ripe-tomato-multiprompt.yaml", raw, "application/yaml"),
        },
        data={"verbosity": "3"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["service"]["clip_prompts"] == {
        "class_prompt_counts": {
            "ripe_tomato": 32,
            "foliage": 15,
            "stem_or_vine": 15,
            "greenhouse_structure": 20,
            "background": 15,
        },
        "total_prompt_count": 97,
        "tokenizer_limit": 77,
        "duplicate_policy": "reject",
    }
    assert all(
        list(obj["clip_scores"]) == list(labels)
        and obj["clip_routing"]["chosen_target"] == "ripe_tomato"
        for obj in body["service"]["objects"]
    )
