"""CPU contract proofs for Objective 021 optional delivery and evidence."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from src.core import (
    MemoryArtifactSink,
    ObjectResult,
    PipelineResult,
    Provenance,
    SingleImageOutcome,
)
from src.service.artifacts import ArtifactDeliveryLedger, ArtifactSelection
from src.service.envelope import ResponseContext, build_completion_json, build_completion_zip
from src.service.errors import ServiceError
from src.service.schemas import CompletionResponse, ErrorEnvelope
from src.service.settings import ServiceSettings
from src.service.yaml_input import parse_hostile_config, service_config_leaf_paths
from src.service.capabilities import build_capabilities


def _outcome(*, rendered: bool = True) -> SingleImageOutcome:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    obj = ObjectResult(
        instance_id=1,
        source_index=0,
        mask=mask,
        metadata={"clip_label": "thing", "clip_score": 0.75},
        class_id=0,
    )
    result = PipelineResult(
        image_height=8,
        image_width=8,
        roi_box=(0, 0, 8, 8),
        resize_info={"mode": "native"},
        objects=(obj,),
        stage_statuses=(),
        candidate_counts={"sam2_candidates": 1, "after_area_bbox": 1, "final": 1},
        rendered={"client-id-is-not-a-name": np.zeros((8, 8, 3), dtype=np.uint8)}
        if rendered
        else {},
        warnings=("fake evidence",),
        timings={"stage.sam2": 0.1},
        provenance=Provenance(config_digest="digest"),
    )
    return SingleImageOutcome(result, None, None, None)


def _context(ledger: ArtifactDeliveryLedger, response_format: str = "json") -> ResponseContext:
    return ResponseContext(
        request_id="request",
        model_id="zap-it-1",
        verbosity=3,
        response_format=response_format,
        config_digest="digest",
        class_mapping={"thing": 0},
        max_response_bytes=1_000_000,
        candidate_views={
            "clip": {
                "mode": "raw_bbox_crop",
                "context_fraction": 0.1,
                "min_context_pixels": 0,
                "max_context_pixels": 64,
                "applied": False,
            },
            "blip3": {
                "mode": "single_dilated_blur",
                "context_fraction": 0.2,
                "min_context_pixels": 0,
                "max_context_pixels": 64,
                "crop_extent_multiplier": 2.0,
                "blur_sigma_fraction": 0.15,
                "contour_enabled": True,
                "contour_fraction": 0.02,
                "contour_min_pixels": 1,
                "contour_max_pixels": 3,
                "contour_rgb": [255, 224, 0],
                "applied": False,
            },
        },
        artifact_ledger=ledger,
        service_safe_artifact_names=True,
    )


def _ledger(**overrides: int) -> ArtifactDeliveryLedger:
    values = {
        "max_response_artifacts": 64,
        "max_debug_artifacts": 48,
        "max_single_artifact_bytes": 32 * 1024 * 1024,
        "max_total_raw_artifact_bytes": 128 * 1024 * 1024,
        "max_response_bytes": 1_000_000,
        "verbosity": 3,
    }
    values.update(overrides)
    return ArtifactDeliveryLedger(ArtifactSelection.from_mapping({}, applied=True), **values)


def test_optional_raw_and_response_budgets_return_structured_success_for_json_and_zip():
    for response_format in ("json", "zip"):
        ledger = _ledger(max_single_artifact_bytes=1, max_total_raw_artifact_bytes=1)
        sink = MemoryArtifactSink()
        sink.store_image(
            "clip-candidate-view-CANDIDATE-0001.png", np.zeros((8, 8, 3), dtype=np.uint8), fmt="png"
        )
        context = _context(ledger, response_format)
        if response_format == "json":
            document = build_completion_json(_outcome(), context, sink=sink)
            CompletionResponse.model_validate(document)
            delivery = document["service"]["artifact_delivery"]
            assert [item["name"] for item in document["service"]["artifacts"]] == [
                "identity-mask.png"
            ]
        else:
            payload = build_completion_zip(_outcome(), context, sink=sink, max_bytes=1_000_000)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            delivery = manifest["service"]["artifact_delivery"]
            assert [item["name"] for item in manifest["service"]["artifacts"]] == [
                "identity-mask.png"
            ]
        assert delivery["truncated"] is True
        assert delivery["budget_omitted_count"] == 2
        assert delivery["delivered_count"] == 0
        assert delivery["selection_excluded_count"] == 0


def test_selection_is_strict_sorted_and_request_local():
    parsed = parse_hostile_config(
        b"diagnostic_artifacts:\n"
        b"  stages: [clip]\n"
        b"  candidate_ids: [4, 2]\n"
        b"  page: 1\n"
        b"  page_size: 1\n",
        verbosity=3,
    )
    assert parsed.effective_mapping["diagnostic_artifacts"]["candidate_ids"] == [4, 2]
    selection = ArtifactSelection.from_mapping(
        parsed.effective_mapping["diagnostic_artifacts"], applied=True
    )
    assert selection.requested_candidate_ids == (4, 2)
    assert selection.effective_candidate_ids == (2, 4)
    ledger = ArtifactDeliveryLedger(
        selection,
        max_response_artifacts=64,
        max_debug_artifacts=48,
        max_single_artifact_bytes=100,
        max_total_raw_artifact_bytes=1000,
        max_response_bytes=10000,
        verbosity=3,
    )
    assert (
        ledger.offer("clip-candidate-view-CANDIDATE-0001.png", estimated_raw_bytes=1, sink=True)
        == "not_selected_candidate"
    )
    assert (
        ledger.offer("clip-candidate-view-CANDIDATE-0002.png", estimated_raw_bytes=1, sink=True)
        == "stored"
    )
    assert (
        ledger.offer(
            "blip3-verification-CANDIDATE-0002-QUESTION-0001.png", estimated_raw_bytes=1, sink=True
        )
        == "not_selected_stage"
    )
    assert ledger.document()["truncated"] is False


def test_resource_error_is_typed_sanitized_and_has_valid_alternatives():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            b"mask_generator:\n  points_per_side: 65\n", verbosity=3, settings=ServiceSettings()
        )
    error = excinfo.value
    assert error.code == "resource_limit"
    envelope = ErrorEnvelope.model_validate(error.envelope("request"))
    assert envelope.error.details is not None
    details = envelope.error.details
    assert details.limit_kind == "field"
    assert details.admissible_alternatives
    assert all(
        "max_points_per_side" not in item.mask_generator.model_fields_set
        for item in details.admissible_alternatives
    )


def test_capability_inventory_matches_validator_and_examples_are_safe():
    body = build_capabilities(ServiceSettings())
    assert set(body["configuration"]["fields"]) == set(service_config_leaf_paths())
    assert body["diagnostic_artifacts"]["fields"]["page"]["minimum"] == 1
