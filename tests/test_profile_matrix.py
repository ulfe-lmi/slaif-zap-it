from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _matrix():
    path = REPO_ROOT / "scripts" / "profile_matrix.py"
    spec = importlib.util.spec_from_file_location("profile_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(profile: str):
    module = _matrix()
    stages = {
        "sam2": "executed",
        "clip": "executed" if "clip" in profile else "not_configured",
        "blip3": "executed" if "blip3" in profile else "not_configured",
    }
    stage_statuses = [
        {"name": "preprocessing", "status": "executed"},
        {"name": "sam2", "status": stages["sam2"]},
        {"name": "postsam2_filter", "status": "executed"},
        {"name": "clip", "status": stages["clip"]},
        {"name": "blip3", "status": stages["blip3"]},
        {"name": "label_filter", "status": "executed"},
        {"name": "visualization", "status": "skipped"},
        {"name": "ordering", "status": "executed"},
    ]
    object_record = {"instance_id": 1, "label": "object"}
    if "blip3" in profile:
        object_record["blip3_answer"] = "bounded answer"
    service = {
        "verbosity": 3,
        "stage_statuses": stage_statuses,
        "candidate_counts": {"sam2_candidates": 1, "final": 1},
        "objects": [object_record],
        "artifacts": [{"name": "identity-mask.png"}],
    }
    return {
        "choices": [{"text": "0 0.5 0.5 1.0 1.0\n"}],
        "service": {
            **service,
            "provenance": {
                "runtime": {
                    "strategy": module.ALL_RESIDENT_STRATEGY,
                    "device": {"logical": "cuda:0"},
                    "models": {
                        name: {"id": spec.model_id, "revision": spec.revision}
                        for name, spec in module.APPROVED_MODEL_SPECS.items()
                        if name in {"sam2", "clip", "blip3"}
                    },
                    "residency": {"request_transition_policy": "none"},
                }
            },
        },
    }


def _resources(module):
    return module.ResourceSample(10, 12, 14, 16, 100, 200, 0, 1)


@pytest.mark.parametrize("profile", ["sam2", "sam2_clip", "sam2_blip3", "sam2_clip_blip3"])
def test_all_supported_profiles_have_expected_stage_shapes(profile):
    module = _matrix()
    result = module.validate_profile_response(
        profile,
        200,
        _payload(profile),
        {"latency_ms": "10"},
        _resources(module),
        residue_count=1,
    )
    assert result["stage_count"] == 8
    assert len(result["semantic_digest"]) == 64
    assert result["answer_count"] == (1 if "blip3" in profile else 0)


def test_profile_config_is_api_safe_and_selects_exact_four_shapes():
    module = _matrix()
    assert module.profile_config("sam2") == (
        b"alpha: 0.6\npreprocessing:\n  resize: 1.0\nmask_generator: {}\n"
        b"postsam2processing:\n  maxsize: 100000\n"
    )
    assert b"clip:" not in module.profile_config("sam2_blip3")
    assert b"blip3:" in module.profile_config("sam2_blip3")
    with pytest.raises(module.MatrixValidationError):
        module.profile_config("unknown")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["service"]["provenance"]["runtime"].update(strategy="wrong"),
        lambda payload: payload["service"]["provenance"]["runtime"]["device"].update(logical="cpu"),
        lambda payload: payload["service"]["provenance"]["runtime"]["models"].pop("clip"),
        lambda payload: payload["service"]["stage_statuses"].__setitem__(
            1, {"name": "sam2", "status": "skipped"}
        ),
        lambda payload: payload["service"]["objects"][0].pop("blip3_answer", None),
    ],
)
def test_validator_fails_closed_on_profile_stage_and_model_mismatches(mutator):
    module = _matrix()
    payload = _payload("sam2_clip_blip3")
    mutator(payload)
    with pytest.raises(module.MatrixValidationError):
        module.validate_profile_response(
            "sam2_clip_blip3",
            200,
            payload,
            {"latency_ms": "10"},
            _resources(module),
            residue_count=1,
        )


def test_validator_fails_closed_on_status_transition_memory_and_residue():
    module = _matrix()
    payload = _payload("sam2")
    with pytest.raises(module.MatrixValidationError):
        module.validate_profile_response(
            "sam2", 500, payload, {"latency_ms": "10"}, _resources(module), residue_count=1
        )
    with pytest.raises(module.MatrixValidationError):
        module.validate_profile_response(
            "sam2",
            200,
            payload,
            {"latency_ms": "10"},
            module.ResourceSample(10, 12, 14, 22118.4, 100, 200, 0, 1),
            residue_count=1,
        )
    with pytest.raises(module.MatrixValidationError):
        module.validate_profile_response(
            "sam2",
            200,
            payload,
            {"latency_ms": "10"},
            module.ResourceSample(10, 12, 14, 16, 100, 200, 1, 1),
            residue_count=1,
        )


def test_shared_memory_residue_is_rejected(tmp_path):
    module = _matrix()
    (tmp_path / "runtime").mkdir()
    assert module.check_shm_residue(tmp_path) == 1
    (tmp_path / "request-opaque").mkdir()
    with pytest.raises(module.MatrixValidationError, match="residue"):
        module.check_shm_residue(tmp_path)


def test_resource_trajectory_rejects_monotonic_gpu_or_host_growth():
    module = _matrix()
    records = [
        {"torch_current_reserved_mib": 1, "torch_current_allocated_mib": 1, "host_rss_mib": 2},
        {"torch_current_reserved_mib": 2, "torch_current_allocated_mib": 1, "host_rss_mib": 2},
        {"torch_current_reserved_mib": 3, "torch_current_allocated_mib": 1, "host_rss_mib": 2},
    ]
    with pytest.raises(module.MatrixValidationError):
        module.validate_resource_trajectory(records)
