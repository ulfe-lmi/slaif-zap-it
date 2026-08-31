"""Focused CPU proofs for the Objective 020 semantic seams."""

from pathlib import Path

import numpy as np
import pytest

from modules.verifier.blip3 import _Blip3Filter, normalize_blip3_token
from src.core import CandidateViewConfig, build_raw_clip_crop
from src.core.routing import apply_clip_routing, route_clip_candidate
from src.postprocessing import filter_by_geometry
from src.service.errors import ServiceError
from src.service.settings import ServiceSettings
from src.service.yaml_input import parse_hostile_config


def _raw_config(**overrides):
    values = {
        "mode": "raw_bbox_crop",
        "context_fraction": 0.25,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
    }
    values.update(overrides)
    return CandidateViewConfig.from_mapping(values, stage="clip")


def _route_config(**overrides):
    values = {
        "labels": ["target"],
        "top_k": 2,
        "score_margin_from_best": 0.03,
        "minimum_target_score": None,
        "uncertain_labels": [],
        "max_candidates": None,
    }
    values.update(overrides)
    return {"route_to_blip3": values}


def test_raw_clip_crop_is_source_exact_and_immutable_at_edges_and_holes():
    rows, cols = np.indices((9, 11))
    image = np.stack((rows * 11 + cols, rows + cols * 3, rows * 5 + cols * 7), axis=2).astype(
        np.uint8
    )
    mask = np.zeros((9, 11), dtype=bool)
    mask[1:5, 0:4] = True
    mask[2:4, 1:3] = False
    mask[7, 10] = True
    image_before = image.copy()
    mask_before = mask.copy()
    result = build_raw_clip_crop(image, mask, 8, _raw_config(context_fraction=0.5))
    x0, y0, x1, y1 = result.crop_bbox_xyxy_exclusive
    assert result.raw_context_radius == 6  # the disconnected edge point makes L=11
    assert np.array_equal(result.rgb, image[y0:y1, x0:x1])
    assert result.rgb.flags.c_contiguous and not result.rgb.flags.writeable
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)
    assert result.metadata_dict()["dilation"] is None
    assert result.metadata_dict()["contour"] is None


def test_raw_clip_half_up_rounding_differs_from_bankers_rounding():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    result = build_raw_clip_crop(image, mask, 1, _raw_config(context_fraction=0.5))
    assert result.raw_context_radius == 1


def test_permissive_router_keeps_complete_vectors_and_deterministic_cap():
    masks = [
        {"_source_index": 4, "clip_scores": {"negative": 0.90, "target": 0.87}},
        {"_source_index": 1, "clip_scores": {"negative": 0.20, "target": 0.20}},
        {"_source_index": 7, "clip_scores": {"negative": 0.10, "target": 0.19}},
    ]
    routed, diagnostics, counts = apply_clip_routing(
        masks, _route_config(max_candidates=2, score_margin_from_best=0.0)
    )
    assert counts == {"initially_routed": 3, "routed_after_cap": 2}
    assert [item["source_candidate_id"] for item in diagnostics] == [5, 2, 8]
    assert diagnostics[0]["primary_reason"] == "target_in_top_k"
    assert diagnostics[1]["primary_reason"] == "target_in_top_k"
    assert diagnostics[2]["primary_reason"] == "max_candidate_limit"
    assert [item["_source_index"] + 1 for item in routed] == [5, 2]
    assert diagnostics[0]["clip_scores"] == masks[0]["clip_scores"]


@pytest.mark.parametrize(
    ("scores", "config", "reason"),
    [
        ({"target": 0.8, "negative": 0.2}, _route_config(top_k=None), "target_top_1"),
        ({"negative": 0.9, "target": 0.8}, _route_config(top_k=2), "target_in_top_k"),
        (
            {"negative": 0.9, "target": 0.87},
            _route_config(top_k=None, score_margin_from_best=0.03),
            "target_within_score_margin",
        ),
        (
            {"negative": 0.9, "target": 0.1},
            _route_config(top_k=None, score_margin_from_best=None, minimum_target_score=0.1),
            "target_exceeded_minimum_score",
        ),
    ],
)
def test_router_reason_precedence_and_inclusive_conditions(scores, config, reason):
    decision = route_clip_candidate({"clip_scores": scores}, config)
    assert decision.routed
    assert decision.record["primary_reason"] == reason


def test_geometry_records_empty_and_configured_rejections_without_disappearing():
    empty = {"_source_index": 3, "segmentation": np.zeros((5, 6), dtype=bool)}
    wide = {"_source_index": 8, "segmentation": np.ones((1, 3), dtype=bool)}
    diagnostics = {}
    kept = filter_by_geometry(
        [empty, wide],
        {
            "min_area": 1,
            "max_area": None,
            "min_width": None,
            "max_width": 2,
            "min_height": None,
            "max_height": None,
            "min_aspect_ratio": None,
            "max_aspect_ratio": None,
            "allow_border_touching": True,
        },
        diagnostics=diagnostics,
    )
    assert kept == []
    assert diagnostics["evaluated"] == 2
    assert diagnostics["removed_by_empty_mask"] == 1
    assert diagnostics["removed_by_max_width"] == 1
    assert diagnostics["rejections"][0]["source_candidate_id"] == 4
    assert diagnostics["rejections"][1]["bbox_xyxy_inclusive"] == [0, 0, 2, 0]
    assert diagnostics["rejections"][1]["configured_limit_field"] == "max_width"


def test_blip3_normalization_is_nfkc_exact_and_unmatched_is_not_a_false_match():
    assert normalize_blip3_token("  YES! ") == "yes"
    assert normalize_blip3_token("Ｎｏ；") == "no"

    class QA:
        device = "cpu"

        def answer(self, _image, _query, max_new_tokens):
            assert max_new_tokens == 32
            return "unknown"

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=bool)
    mask[25:55, 25:55] = True
    record = {
        "segmentation": mask,
        "_source_index": 0,
        "_filtered_index": 0,
        "clip_label": "negative",
        "clip_score": 0.8,
        "clip_routing": {"chosen_target": "target", "primary_reason": "target_in_top_1"},
        "_route_to_blip3": True,
    }
    filt = _Blip3Filter.from_qa(
        QA(),
        {
            "target": {
                "question": "Is it the target?",
                "trueresult": "Yes",
                "falseresult": "No",
                "newcategory": "target",
                "falsecategory": "negative",
            }
        },
        max_questions=32,
        max_new_tokens=32,
    )
    filt.filter_masks(
        [record],
        image,
        None,
        "request",
        candidate_view_config={"context_fraction": 0, "max_context_pixels": 0},
    )
    evidence = record["blip3_verification"]
    assert evidence["mapping_outcome"] == "unmatched_answer"
    assert evidence["final_label"] == "negative"
    assert evidence["configured_question"] == "Is it the target?"
    assert "[TARGET QUESTION]" in evidence["effective_question"]


@pytest.mark.parametrize("verbosity", [0, 3])
@pytest.mark.parametrize(
    "config_path", sorted(Path("configs").glob("*.yaml")), ids=lambda p: p.name
)
def test_every_shipped_config_uses_the_service_contract(config_path, verbosity):
    validated = parse_hostile_config(
        config_path.read_bytes(), verbosity=verbosity, settings=ServiceSettings()
    )
    assert validated.effective_mapping["candidate_views"]["clip"]["mode"] == "raw_bbox_crop"
    assert validated.effective_mapping["clip_routing"]["route_to_blip3"]["labels"]


def test_api_rejects_masked_clip_and_missing_or_orphan_routing_rules():
    invalid = (
        "clip:\n  labels:\n    target: 'a target'\n"
        "candidate_views:\n  clip:\n    mode: mask_dilated\n"
    )
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(invalid.encode(), verbosity=3)
    assert excinfo.value.code in {"invalid_config", "unsupported_field"}

    missing = (
        "clip:\n  labels:\n    target: 'a target'\n"
        "blip3:\n  target:\n    question: 'Is it target?'\n    trueresult: 'Yes'\n"
        "    falseresult: 'No'\n    newcategory: target\n    falsecategory: negative\n"
    )
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(missing.encode(), verbosity=3)
    assert "clip_routing" in str(excinfo.value)
