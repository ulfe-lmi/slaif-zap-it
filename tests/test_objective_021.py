"""CPU contract proofs for Objective 021 optional delivery and evidence."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from src.core import (
    MemoryArtifactSink,
    ObjectResult,
    PipelineResult,
    Provenance,
    SingleImageOutcome,
)
from src.core.sinks import ArtifactSinkError
from fastapi.testclient import TestClient
from src.service.artifacts import ArtifactDeliveryLedger, ArtifactSelection
from src.service.envelope import ResponseContext, build_completion_json, build_completion_zip
from src.service.errors import ServiceError
from src.service.schemas import CompletionResponse, ErrorEnvelope
from src.service import FakeEngine, ReadyState, create_app
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


@pytest.mark.parametrize("response_format", ("json", "zip"))
def test_response_byte_pressure_removes_optional_payload_and_preserves_identity(response_format):
    optional_name = "clip-candidate-view-CANDIDATE-0001.png"
    optional_payload = bytes((index * 73 + (index >> 3)) % 256 for index in range(65_536))

    def sink_with_optional():
        sink = MemoryArtifactSink()
        sink.store_bytes(optional_name, optional_payload, content_type="image/png")
        return sink

    def build(context, sink=None):
        if response_format == "json":
            return build_completion_json(_outcome(rendered=False), context, sink=sink)
        return build_completion_zip(
            _outcome(rendered=False), context, sink=sink, max_bytes=context.max_response_bytes
        )

    full = build(_context(_ledger(), response_format), sink_with_optional())
    essential = build(_context(_ledger(), response_format))
    full_size = (
        len(json.dumps(full, ensure_ascii=False).encode("utf-8"))
        if response_format == "json"
        else len(full)
    )
    essential_size = (
        len(json.dumps(essential, ensure_ascii=False).encode("utf-8"))
        if response_format == "json"
        else len(essential)
    )
    cap = (full_size + essential_size) // 2
    pressured = build(
        replace(_context(_ledger(), response_format), max_response_bytes=cap),
        sink_with_optional(),
    )

    if response_format == "json":
        document = pressured
        CompletionResponse.model_validate(document)
        descriptors = document["service"]["artifacts"]
        delivery = document["service"]["artifact_delivery"]
        assert len(json.dumps(document, ensure_ascii=False).encode("utf-8")) <= cap
        assert [item["name"] for item in descriptors] == ["identity-mask.png"]
        assert document["service"]["objects"]
    else:
        with zipfile.ZipFile(io.BytesIO(pressured)) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("manifest.json"))
            identity_payload = archive.read("identity-mask.png")
        document = {**manifest, "service": {**manifest["service"]}}
        document["service"]["artifacts"] = [
            {**item, "data": ""} for item in document["service"]["artifacts"]
        ]
        CompletionResponse.model_validate(document)
        descriptors = manifest["service"]["artifacts"]
        delivery = manifest["service"]["artifact_delivery"]
        assert len(pressured) <= cap
        assert optional_name not in names
        assert identity_payload
        assert [item["name"] for item in descriptors] == ["identity-mask.png"]

    assert delivery["truncated"] is True
    assert delivery["budget_omitted_count"] == 1
    assert delivery["delivered_count"] == 0
    assert delivery["selection_excluded_count"] == 0
    assert delivery["delivered_names"] == ["identity-mask.png"]
    assert delivery["actual_delivered_raw_bytes"] == 0
    assert delivery["omitted"] == [
        {
            "name": optional_name,
            "stage": "clip",
            "source_candidate_id": 1,
            "question_id": None,
            "estimated_raw_bytes": len(optional_payload),
            "reason": "omitted_response_limit",
        }
    ]


@pytest.mark.parametrize(
    "candidate_ids",
    ([0], [257], [True], [[1]], [{"candidate": 1}], [1, 1]),
)
def test_selector_candidate_ids_are_strict_bounded_and_sanitized(candidate_ids):
    raw = "diagnostic_artifacts:\n  candidate_ids: " + repr(candidate_ids) + "\n"
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw.encode(), verbosity=3)
    assert excinfo.value.code == "invalid_config"


@pytest.mark.parametrize("stages", ("[[clip]]", "[{stage: clip}]", "[clip, clip]"))
def test_selector_stage_members_are_validated_before_uniqueness(stages):
    raw = f"diagnostic_artifacts:\n  stages: {stages}\n"
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw.encode(), verbosity=3)
    assert excinfo.value.code == "invalid_config"


def test_selector_response_schema_is_strict_and_bounded():
    from src.service.schemas import ArtifactSelectionMetadata

    valid = ArtifactSelectionMetadata.model_validate(
        {"stages": ["clip"], "candidate_ids": [1, 256], "page": 1, "page_size": 1}
    )
    assert valid.candidate_ids == [1, 256]
    for candidate_ids in ([0], [257], [True], [[1]], [{"candidate": 1}]):
        with pytest.raises(ValueError):
            ArtifactSelectionMetadata.model_validate(
                {"stages": ["clip"], "candidate_ids": candidate_ids, "page": 1, "page_size": 1}
            )


def test_capability_catalog_and_named_response_refs_close_openapi_contract():
    settings = ServiceSettings(api_key="k" * 32)
    app = create_app(
        engine=FakeEngine(),
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "fake"),
    )
    client = TestClient(app)
    body = build_capabilities(settings)
    catalog = body["configuration"]["field_catalog"]
    paths = list(service_config_leaf_paths())
    assert [record["path"] for record in catalog] == paths
    assert len({record["path"] for record in catalog}) == len(paths)
    for record in catalog:
        descriptor = record["descriptor"]
        assert descriptor["type"] and descriptor["stage"] and descriptor["description"]
        assert {"required", "nullable", "default"} <= set(record)

    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert schema["CapabilityCatalogEntry"]["properties"]["path"]["enum"] == paths
    service_properties = schema["ServiceMetadata"]["properties"]
    assert service_properties["stage_statuses"]["anyOf"][0]["items"]["$ref"].endswith(
        "/StageStatus"
    )
    for field, model in (
        ("candidate_counts", "CandidateCounts"),
        ("timings_ms", "TimingMetadata"),
        ("provenance", "ProvenanceMetadata"),
        ("clip_routing", "ClipRoutingConfiguration"),
    ):
        assert any(
            part.get("$ref", "").endswith("/" + model)
            for part in service_properties[field]["anyOf"]
        )


@pytest.mark.parametrize("response_format", ("json", "zip"))
def test_response_byte_pressure_retains_hard_413_for_essential_document(response_format):
    context = _context(_ledger(), response_format)
    essential = (
        build_completion_json(_outcome(rendered=False), context)
        if response_format == "json"
        else build_completion_zip(
            _outcome(rendered=False), context, max_bytes=context.max_response_bytes
        )
    )
    essential_size = (
        len(json.dumps(essential, ensure_ascii=False).encode("utf-8"))
        if response_format == "json"
        else len(essential)
    )
    with pytest.raises(ServiceError) as excinfo:
        pressured_context = replace(context, max_response_bytes=essential_size - 1)
        if response_format == "json":
            build_completion_json(_outcome(rendered=False), pressured_context)
        else:
            build_completion_zip(
                _outcome(rendered=False), pressured_context, max_bytes=essential_size - 1
            )
    assert excinfo.value.code == "response_too_large"


def test_ledger_mixed_omission_bound_is_exact_and_duplicate_facts_are_rejected():
    selection = ArtifactSelection.from_mapping({"stages": ["clip"]}, applied=True)
    ledger = ArtifactDeliveryLedger(
        selection,
        max_response_artifacts=64,
        max_debug_artifacts=48,
        max_single_artifact_bytes=1,
        max_total_raw_artifact_bytes=1000,
        max_response_bytes=10000,
        verbosity=3,
    )
    for candidate_id in range(1, 601):
        stage_name = "clip-candidate-view" if candidate_id % 2 else "sam2-candidate"
        ledger.offer(
            f"{stage_name}-CANDIDATE-{candidate_id:04d}.png",
            estimated_raw_bytes=2,
            sink=True,
        )
    delivery = ledger.document()
    assert {
        key: delivery[key]
        for key in (
            "eligible_count",
            "selected_count",
            "delivered_count",
            "selection_excluded_count",
            "budget_omitted_count",
            "unreported_overflow_count",
            "unreported_selection_excluded_count",
            "unreported_budget_omitted_count",
        )
    } == {
        "eligible_count": 600,
        "selected_count": 48,
        "delivered_count": 0,
        "selection_excluded_count": 552,
        "budget_omitted_count": 48,
        "unreported_overflow_count": 24,
        "unreported_selection_excluded_count": 24,
        "unreported_budget_omitted_count": 0,
    }
    assert delivery["truncated"] is True
    assert len(delivery["omitted"]) == 576
    assert len(delivery["warnings"]) == 1
    with pytest.raises(ArtifactSinkError, match="contradictory"):
        ledger.offer(
            "clip-candidate-view-CANDIDATE-0001.png",
            estimated_raw_bytes=3,
            sink=True,
        )


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
