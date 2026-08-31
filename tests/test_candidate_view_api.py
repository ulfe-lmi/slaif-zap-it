"""CPU/fake API evidence for the single-image BLIP3 response contract."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from modules.classifier import clip as clip_module
from modules.input.images import apply_roi, resize_image
from modules.verifier import blip3 as blip3_module
from src.core import CoreConfig, StageFunctions, run_single_image
from src.postprocessing import filter_by_area_bbox
from src.service import ReadyState, create_app
from src.service.schemas import (
    Blip3CandidateViewRecord,
    CandidateViewInputRecord,
    CompletionResponse,
)


def _png_bytes(width=32, height=24):
    rows, cols = np.indices((height, width))
    image = np.stack(
        ((rows * 7 + cols * 3) % 251, (rows * 5 + cols * 11) % 251, (rows * 13 + cols * 2) % 251),
        axis=2,
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class _QA:
    device = "cpu"

    def __init__(self):
        self.images = []

    def answer(self, image, _query, max_new_tokens):
        assert max_new_tokens == 32
        self.images.append(image.copy())
        return "Yes"


class _CandidateViewEngine:
    def __init__(self):
        self.clip_inputs = []
        self.qa = _QA()
        clip_filter = object.__new__(clip_module._ClipFilter)

        class _TextEmbeds:
            def numel(self):
                return 1

        clip_filter.text_embeds = _TextEmbeds()
        clip_filter.update_labels = None
        clip_filter.debug = False
        clip_filter.verbosity = 0
        clip_filter.log_print = lambda *_args, **_kwargs: None

        def classify(patch, _index):
            self.clip_inputs.append(patch.copy())
            return "target", 0.1, "a target"

        clip_filter.classify_single = classify
        self.clip_state = {"clip_filter": clip_filter}
        self.blip_state = {"blip3_qa": self.qa, "max_questions": 32, "max_new_tokens": 32}
        self.holder_ids = []
        self.config_snapshots = []
        self.image_inputs = []

    def __call__(self, image_rgb, config: CoreConfig, **kwargs):
        self.holder_ids.append((id(self.clip_state["clip_filter"]), id(self.qa)))
        self.config_snapshots.append(
            {
                "clip": copy.deepcopy(dict(config.clip_cfg)),
                "blip3": copy.deepcopy(dict(config.blip3_cfg)),
                "candidate_views": {
                    stage: config.candidate_view_config(stage).as_dict(stage=stage)
                    for stage in ("clip", "blip3")
                },
            }
        )
        self.image_inputs.append(image_rgb.copy())

        def run_sam2(state, _params, image, **_kwargs):
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[4:20, 8:24] = True
            return state or {}, [{"segmentation": mask, "predicted_iou": 0.9}], {"num_masks": 1}

        stages = StageFunctions(
            apply_roi=apply_roi,
            resize_image=resize_image,
            run_sam2=run_sam2,
            filter_by_area_bbox=filter_by_area_bbox,
            run_clip=clip_module.run,
            run_blip3=blip3_module.run,
            generate_visualizations=lambda *_args, **_kwargs: {},
        )
        return run_single_image(
            image_rgb,
            config,
            frame_id=kwargs.get("frame_id", "image"),
            clip_state=self.clip_state,
            blip3_state=self.blip_state,
            verbosity=kwargs.get("verbosity", 1),
            artifact_sink=kwargs["artifact_sink"],
            stages=stages,
            class_labels=kwargs.get("class_labels", ()),
            render_visualizations=False,
            service_safe_artifact_names=True,
        )


def _files(*, response_format="json", clip_fraction=0.25, context_fraction=0.2, debug=True):
    config = f"""alpha: 0.5
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    falseresult: 'No'
    newcategory: target
    falsecategory: negative
    debug: {str(debug).lower()}
clip_routing:
  route_to_blip3:
    labels: [target]
    top_k: 1
    score_margin_from_best: null
    minimum_target_score: null
    uncertain_labels: []
    max_candidates: null
candidate_views:
  clip:
    context_fraction: {clip_fraction}
    max_context_pixels: 8
  blip3:
    context_fraction: {context_fraction}
    max_context_pixels: 8
    contour_enabled: true
    contour_rgb: [255, 224, 0]
""".encode()
    return {
        "image": ("input.png", _png_bytes(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }, {"verbosity": "3", "response_format": response_format}


def test_effective_policy_and_l0_l3_gating():
    engine = _CandidateViewEngine()
    client = TestClient(
        create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "fake ready"))
    )
    for level in range(4):
        files, data = _files()
        data["verbosity"] = str(level)
        response = client.post("/v1/completions", files=files, data=data)
        assert response.status_code == 200, response.text
        document = response.json()
        CompletionResponse.model_validate(document)
        service = document["service"]
        assert service["candidate_views"]["blip3"] == {
            "mode": "single_dilated_blur",
            "context_fraction": 0.2,
            "min_context_pixels": 0,
            "max_context_pixels": 8,
            "crop_extent_multiplier": 2.0,
            "blur_sigma_fraction": 0.15,
            "contour_enabled": True,
            "contour_fraction": 0.02,
            "contour_min_pixels": 1,
            "contour_max_pixels": 3,
            "contour_rgb": [255, 224, 0],
            "applied": True,
        }
        if level < 3:
            assert "blip3_candidate_views" not in service
            assert "candidate_view_inputs" not in service
        else:
            records = [
                Blip3CandidateViewRecord.model_validate(item)
                for item in service["blip3_candidate_views"]
            ]
            assert len(records) == 1 and records[0].status == "rendered"
            debug_records = [
                CandidateViewInputRecord.model_validate(item)
                for item in service["candidate_view_inputs"]
            ]
            assert [record.stage for record in debug_records] == ["clip", "blip3"]


def test_json_zip_manifest_and_debug_payload_are_identical():
    engine = _CandidateViewEngine()
    client = TestClient(
        create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "fake ready"))
    )
    json_files, json_data = _files()
    json_response = client.post("/v1/completions", files=json_files, data=json_data)
    assert json_response.status_code == 200
    document = json_response.json()
    descriptors = {item["name"]: item for item in document["service"]["artifacts"]}
    json_payloads = {name: base64.b64decode(item["data"]) for name, item in descriptors.items()}

    zip_files, zip_data = _files(response_format="zip")
    zip_response = client.post("/v1/completions", files=zip_files, data=zip_data)
    assert zip_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        zip_payloads = {name: archive.read(name) for name in json_payloads}
    CompletionResponse.model_validate(
        {
            **manifest,
            "service": {
                **manifest["service"],
                "artifacts": [{**item, "data": ""} for item in manifest["service"]["artifacts"]],
            },
        }
    )
    assert zip_payloads == json_payloads
    for name, payload in zip_payloads.items():
        descriptor = next(item for item in manifest["service"]["artifacts"] if item["name"] == name)
        assert hashlib.sha256(payload).hexdigest() == descriptor["sha256"]
        assert len(payload) == descriptor["size"]


def test_request_local_clip_and_blip3_settings_are_a_b_a_isolated_and_holder_stable(
    monkeypatch,
):
    engine = _CandidateViewEngine()
    forbidden_initializations = []

    def fail_clip_initialize(*_args, **_kwargs):
        forbidden_initializations.append("clip")
        raise AssertionError("resident CLIP holder was reinitialized")

    def fail_blip3_holder(*_args, **_kwargs):
        forbidden_initializations.append("blip3")
        raise AssertionError("resident BLIP3 holder was reinitialized")

    monkeypatch.setattr(clip_module, "initialize", fail_clip_initialize)
    monkeypatch.setattr(blip3_module, "_Blip3QA", fail_blip3_holder)
    client = TestClient(
        create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "fake ready"))
    )
    responses = []
    for clip_fraction, blip3_fraction in ((0.1, 0.1), (0.3, 0.3), (0.1, 0.1)):
        files, data = _files(clip_fraction=clip_fraction, context_fraction=blip3_fraction)
        response = client.post("/v1/completions", files=files, data=data)
        assert response.status_code == 200
        responses.append(response.json())
    assert len(engine.holder_ids) == 3
    assert len(set(engine.holder_ids)) == 1
    assert forbidden_initializations == []
    assert np.array_equal(engine.qa.images[0], engine.qa.images[2])
    assert not np.array_equal(engine.qa.images[0], engine.qa.images[1])
    assert np.array_equal(engine.clip_inputs[0], engine.clip_inputs[2])
    assert not np.array_equal(engine.clip_inputs[0], engine.clip_inputs[1])
    assert (
        engine.config_snapshots[0]["candidate_views"]
        == engine.config_snapshots[2]["candidate_views"]
    )
    assert (
        engine.config_snapshots[0]["candidate_views"]
        != engine.config_snapshots[1]["candidate_views"]
    )
    assert responses[0]["service"]["candidate_views"] == responses[2]["service"]["candidate_views"]
    assert responses[0]["service"]["candidate_views"] != responses[1]["service"]["candidate_views"]


class _IdentityFlowEngine:
    """Run source IDs through removal, semantic stages, and final renderers."""

    def __init__(self):
        self.clip_inputs = []
        self.qa = _QA()
        clip_filter = object.__new__(clip_module._ClipFilter)

        class _TextEmbeds:
            def numel(self):
                return 1

        clip_filter.text_embeds = _TextEmbeds()
        clip_filter.update_labels = None
        clip_filter.debug = False
        clip_filter.verbosity = 0
        clip_filter.log_print = lambda *_args, **_kwargs: None

        def classify(patch, _index):
            self.clip_inputs.append(patch.copy())
            return "target", 0.2, "a target"

        clip_filter.classify_single = classify
        self.clip_state = {"clip_filter": clip_filter}
        self.blip_state = {"blip3_qa": self.qa, "max_questions": 32, "max_new_tokens": 32}
        self.final_objects = None

    @staticmethod
    def _mask(shape, top, bottom, left, right):
        mask = np.zeros(shape, dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    def __call__(self, image_rgb, config: CoreConfig, **kwargs):
        def run_sam2(state, _params, image, **_kwargs):
            shape = image.shape[:2]
            return (
                state or {},
                [
                    {
                        "segmentation": self._mask(shape, 2, 3, 1, 8),
                        "predicted_iou": 0.91,
                        "stability_score": 0.81,
                    },
                    {
                        "segmentation": self._mask(shape, 8, 10, 8, 11),
                        "predicted_iou": 0.92,
                        "stability_score": 0.82,
                    },
                    {
                        "segmentation": self._mask(shape, 20, 26, 25, 31),
                        "predicted_iou": 0.93,
                        "stability_score": 0.83,
                    },
                ],
                {"num_masks": 3},
            )

        def capture_visualization(_image, _masks_by_stage, _config, **kwargs):
            self.final_objects = tuple(kwargs["final_objects"])
            return {}

        stages = StageFunctions(
            apply_roi=apply_roi,
            resize_image=resize_image,
            run_sam2=run_sam2,
            filter_by_area_bbox=filter_by_area_bbox,
            run_clip=clip_module.run,
            run_blip3=blip3_module.run,
            generate_visualizations=capture_visualization,
        )
        return run_single_image(
            image_rgb,
            config,
            frame_id=kwargs.get("frame_id", "image"),
            clip_state=self.clip_state,
            blip3_state=self.blip_state,
            verbosity=kwargs.get("verbosity", 3),
            artifact_sink=kwargs["artifact_sink"],
            stages=stages,
            class_labels=kwargs.get("class_labels", ()),
            render_visualizations=kwargs.get("render_visualizations", True),
            service_safe_artifact_names=True,
        )


IDENTITY_FLOW_CONFIG = b"""alpha: 0.5
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    falseresult: 'No'
    newcategory: target
    falsecategory: negative
    debug: true
clip_routing:
  route_to_blip3:
    labels: [target]
    top_k: 1
    score_margin_from_best: null
    minimum_target_score: null
    uncertain_labels: []
    max_candidates: null
postsam2processing:
  max_w: 6
candidate_views:
  clip:
    context_fraction: 0
    max_context_pixels: 0
  blip3:
    context_fraction: 0
    max_context_pixels: 0
    contour_enabled: false
visualization:
  blip3:
    - id: identity-flow
      renderer: annotated-labelled
"""


def _identity_flow_files(response_format="json"):
    return {
        "image": ("generated.png", _png_bytes(width=40, height=32), "image/png"),
        "config": ("config.yaml", IDENTITY_FLOW_CONFIG, "application/yaml"),
    }, {"verbosity": "3", "response_format": response_format}


def _identity_tuples(values):
    return [
        (
            value["instance_id"],
            value["source_candidate_id"],
            value["filtered_index"],
            value["area_px"],
        )
        for value in values
    ]


def _zip_manifest(response):
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {name: archive.read(name) for name in archive.namelist()}
    return manifest, members


def test_source_identity_survives_filter_semantics_order_visualization_json_and_zip():
    engine = _IdentityFlowEngine()
    client = TestClient(
        create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "fake ready"))
    )
    json_files, json_data = _identity_flow_files()
    json_response = client.post("/v1/completions", files=json_files, data=json_data)
    assert json_response.status_code == 200, json_response.text
    document = json_response.json()
    CompletionResponse.model_validate(document)
    service = document["service"]
    assert service["candidate_counts"] == {
        "sam2_candidates": 3,
        "after_area_bbox": 2,
        "after_clip": 2,
        "final": 2,
        "raw_sam2_generated": 3,
        "non_empty_candidates": 3,
        "geometry_evaluated": 3,
        "after_geometry": 2,
        "geometry_rejected": 1,
        "clip_scored": 2,
        "initially_routed": 2,
        "routed_after_cap": 2,
        "blip3_verified": 2,
        "after_final_label_filter": 2,
    }
    expected_objects = [(1, 3, 1, 36), (2, 2, 0, 6)]
    assert _identity_tuples(service["objects"]) == expected_objects
    assert engine.final_objects is not None
    assert [
        (obj.instance_id, obj.source_candidate_id, obj.filtered_index, obj.area_px)
        for obj in engine.final_objects
    ] == expected_objects

    records = [
        CandidateViewInputRecord.model_validate(item) for item in service["candidate_view_inputs"]
    ]
    assert [
        (record.stage, record.source_candidate_id, record.filtered_index, record.artifact_name)
        for record in records
    ] == [
        ("clip", 2, 0, "clip-candidate-view-CANDIDATE-0002.png"),
        ("clip", 3, 1, "clip-candidate-view-CANDIDATE-0003.png"),
        ("blip3", 2, 0, "blip3-verification-CANDIDATE-0002-QUESTION-0001.png"),
        ("blip3", 3, 1, "blip3-verification-CANDIDATE-0003-QUESTION-0002.png"),
    ]
    assert {record.artifact_name for record in records} == {
        item["name"] for item in service["artifacts"] if item["name"] != "identity-mask.png"
    }
    assert all(record.source_candidate_id in {2, 3} for record in records)
    assert [
        (item["source_candidate_id"], item["filtered_index"])
        for item in service["blip3_candidate_views"]
    ] == [(2, 0), (3, 1)]

    zip_files, zip_data = _identity_flow_files("zip")
    zip_response = client.post("/v1/completions", files=zip_files, data=zip_data)
    assert zip_response.status_code == 200, zip_response.text
    manifest, members = _zip_manifest(zip_response)
    schema_manifest = copy.deepcopy(manifest)
    for artifact in schema_manifest["service"]["artifacts"]:
        artifact["data"] = ""
    CompletionResponse.model_validate(schema_manifest)
    assert _identity_tuples(manifest["service"]["objects"]) == expected_objects
    assert manifest["service"]["candidate_view_inputs"] == service["candidate_view_inputs"]
    for descriptor in manifest["service"]["artifacts"]:
        if descriptor["name"] == "identity-mask.png":
            continue
        payload = members[descriptor["name"]]
        assert hashlib.sha256(payload).hexdigest() == descriptor["sha256"]
        assert len(payload) == descriptor["size"]
