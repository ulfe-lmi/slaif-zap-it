"""CPU/fake API evidence for the complete candidate-view response contract."""

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
from src.service.schemas import CandidateViewInputRecord, CompletionResponse


def _png_bytes(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 30, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


class _QA:
    device = "cpu"

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def answer(self, image, _query, max_new_tokens):
        assert max_new_tokens == 32
        self.images.append(np.asarray(image).copy())
        return "Yes"


class _CandidateViewEngine:
    def __init__(self) -> None:
        self.clip_inputs: list[np.ndarray] = []
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
        self.clip_filter = clip_filter
        self.clip_state = {"clip_filter": clip_filter}
        self.blip_state = {
            "blip3_qa": self.qa,
            "max_questions": 32,
            "max_new_tokens": 32,
        }
        self.holder_ids = []
        self.config_snapshots = []
        self.image_inputs = []

    def __call__(self, image_rgb, config: CoreConfig, **kwargs):
        self.holder_ids.append((id(self.clip_filter), id(self.qa)))
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
            mask[5:16, 8:20] = True
            return state or {}, [{"segmentation": mask, "predicted_iou": 0.9}], {}

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


class _IdentityFlowEngine:
    """Run the actual core seams with generated masks and resident fake holders."""

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


def _files(response_format: str = "json"):
    config = b"""alpha: 0.5
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    debug: true
candidate_views:
  clip:
    context_fraction: 0.25
    max_context_pixels: 8
  blip3:
    context_fraction: 0.25
    max_context_pixels: 8
    contour_width: 1
"""
    return {
        "image": ("input.png", _png_bytes(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }, {"verbosity": "3", "response_format": response_format}


def _png_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_json_zip_candidate_view_inputs_and_media_are_one_to_one():
    engine = _CandidateViewEngine()
    client = TestClient(
        create_app(
            engine=engine,
            readiness_provider=lambda: ReadyState(True, "fake ready"),
        )
    )
    json_files, json_data = _files("json")
    json_response = client.post("/v1/completions", files=json_files, data=json_data)
    assert json_response.status_code == 200
    document = json_response.json()
    service = document["service"]
    assert service["candidate_views"]["clip"]["applied"] is True
    assert service["candidate_views"]["blip3"]["applied"] is True
    assert service["objects"]
    records = [
        CandidateViewInputRecord.model_validate(item) for item in service["candidate_view_inputs"]
    ]
    assert [record.stage for record in records] == ["clip", "blip3"]
    assert [record.source_candidate_id for record in records] == [1, 1]
    assert records[0].filtered_index == records[1].filtered_index == 0
    assert records[1].question_id == 1

    descriptors = {
        item["name"]: item for item in service["artifacts"] if item["name"] != "identity-mask.png"
    }
    record_names = {record.artifact_name for record in records}
    assert record_names == set(descriptors)
    assert set(descriptors) == {
        "clip-candidate-view-CANDIDATE-0001.png",
        "blip3-verification-CANDIDATE-0001-QUESTION-0001.png",
    }
    assert all(item["media_type"] == "image/png" for item in descriptors.values())
    assert all(item["size"] > 0 and len(item["sha256"]) == 64 for item in descriptors.values())

    json_payloads = {name: base64.b64decode(item["data"]) for name, item in descriptors.items()}
    assert json_payloads[records[0].artifact_name] == _png_payload(engine.clip_inputs[0])
    assert json_payloads[records[1].artifact_name] == _png_payload(engine.qa.images[0])

    zip_files, zip_data = _files("zip")
    zip_response = client.post("/v1/completions", files=zip_files, data=zip_data)
    assert zip_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        zip_payloads = {name: archive.read(name) for name in descriptors}
    manifest_descriptors = {
        item["name"]: item
        for item in manifest["service"]["artifacts"]
        if item["name"] in descriptors
    }
    assert zip_payloads == json_payloads
    assert {name: item["sha256"] for name, item in manifest_descriptors.items()} == {
        name: item["sha256"] for name, item in descriptors.items()
    }
    assert {name: len(payload) for name, payload in zip_payloads.items()} == {
        name: item["size"] for name, item in descriptors.items()
    }


def _generated_image_bytes(width=48, height=40):
    rows, cols = np.indices((height, width))
    image = np.stack(
        (
            (rows * 7 + cols * 3 + 11) % 251,
            (rows * 5 + cols * 13 + 17) % 251,
            (rows * 19 + cols * 2 + 23) % 251,
        ),
        axis=2,
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue(), image


def _matrix_files(clip_fraction, blip3_fraction, *, stages=True, response_format="json"):
    stage_config = (
        """
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    debug: true
"""
        if stages
        else ""
    )
    config = f"""alpha: 0.5
{stage_config}candidate_views:
  clip:
    context_fraction: {clip_fraction}
    max_context_pixels: 64
  blip3:
    context_fraction: {blip3_fraction}
    max_context_pixels: 64
    contour_width: 2
""".encode()
    image_bytes, _image = _generated_image_bytes()
    return {
        "image": ("generated.png", image_bytes, "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }, {"verbosity": "0", "response_format": response_format}


def _assert_effective_policy(document, clip_fraction, blip3_fraction, *, applied):
    service = document["service"]
    assert service["candidate_views"]["clip"] == {
        "mode": "mask_dilated",
        "context_fraction": clip_fraction,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
        "applied": applied,
    }
    assert service["candidate_views"]["blip3"] == {
        "mode": "mask_dilated",
        "context_fraction": blip3_fraction,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
        "contour_width": 2,
        "applied": applied,
    }


def _zip_manifest(response):
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {name: archive.read(name) for name in archive.namelist()}
    return manifest, members


def test_candidate_view_policy_levels_and_stable_resident_ab_a_isolation(monkeypatch):
    engine = _CandidateViewEngine()
    forbidden_initializations = []

    def fail_clip_initialize(*_args, **_kwargs):
        if engine.clip_state.get("clip_filter") is not None:
            forbidden_initializations.append("clip")
            raise AssertionError("resident CLIP holder was reinitialized")
        raise AssertionError("unexpected CLIP initialization without a resident holder")

    def fail_blip3_holder(*_args, **_kwargs):
        if engine.blip_state.get("blip3_qa") is not None:
            forbidden_initializations.append("blip3")
            raise AssertionError("resident BLIP3 holder was reinitialized")
        raise AssertionError("unexpected BLIP3 holder construction without a resident holder")

    # These are the actual fallback construction seams.  If a request silently
    # discards the supplied holder and constructs a replacement, this test fails.
    monkeypatch.setattr(clip_module, "initialize", fail_clip_initialize)
    monkeypatch.setattr(blip3_module, "_Blip3QA", fail_blip3_holder)
    client = TestClient(
        create_app(
            engine=engine,
            readiness_provider=lambda: ReadyState(True, "fake ready"),
        )
    )
    fractions = {
        "a": (0.10, 0.20),
        "b": (0.40, 0.45),
    }
    configured_a_l3 = None
    for level in range(4):
        before_clip = len(engine.clip_inputs)
        before_qa = len(engine.qa.images)
        responses = []
        for label in ("a", "b", "a"):
            files, data = _matrix_files(*fractions[label], stages=True)
            data["verbosity"] = str(level)
            response = client.post("/v1/completions", files=files, data=data)
            assert response.status_code == 200, response.text
            document = response.json()
            CompletionResponse.model_validate(document)
            _assert_effective_policy(document, *fractions[label], applied=True)
            assert engine.holder_ids[-1] == engine.holder_ids[0]
            responses.append(document)

        clip_inputs = engine.clip_inputs[before_clip:]
        qa_inputs = engine.qa.images[before_qa:]
        assert len(clip_inputs) == len(qa_inputs) == 3
        assert not np.array_equal(clip_inputs[0], clip_inputs[1])
        assert not np.array_equal(qa_inputs[0], qa_inputs[1])
        assert np.array_equal(clip_inputs[0], clip_inputs[2])
        assert np.array_equal(qa_inputs[0], qa_inputs[2])
        assert (
            responses[0]["service"]["candidate_views"] == responses[2]["service"]["candidate_views"]
        )
        assert (
            responses[0]["service"]["candidate_views"] != responses[1]["service"]["candidate_views"]
        )
        if level < 3:
            assert all("candidate_view_inputs" not in item["service"] for item in responses)
            assert all("artifacts" not in item["service"] for item in responses if level == 0)
            if level == 1:
                assert all(
                    [artifact["name"] for artifact in item["service"]["artifacts"]]
                    == ["identity-mask.png"]
                    for item in responses
                )
            if level == 2:
                assert all("objects" in item["service"] for item in responses)
        else:
            for document in responses:
                records = [
                    CandidateViewInputRecord.model_validate(item)
                    for item in document["service"]["candidate_view_inputs"]
                ]
                assert [record.stage for record in records] == ["clip", "blip3"]
                assert all(record.source_candidate_id == 1 for record in records)
            configured_a_l3 = responses[0]

        absent_files, absent_data = _matrix_files(*fractions["a"], stages=False)
        absent_data["verbosity"] = str(level)
        absent = client.post("/v1/completions", files=absent_files, data=absent_data)
        assert absent.status_code == 200, absent.text
        absent_document = absent.json()
        CompletionResponse.model_validate(absent_document)
        _assert_effective_policy(absent_document, *fractions["a"], applied=False)
        if level < 3:
            assert "candidate_view_inputs" not in absent_document["service"]
        else:
            assert absent_document["service"]["candidate_view_inputs"] == []
        if level == 0:
            assert "artifacts" not in absent_document["service"]
        else:
            assert [artifact["name"] for artifact in absent_document["service"]["artifacts"]] == [
                "identity-mask.png"
            ]

    assert forbidden_initializations == []
    assert len(forbidden_initializations) == 0
    assert engine.holder_ids
    assert all(holder_ids == engine.holder_ids[0] for holder_ids in engine.holder_ids)
    assert all(np.array_equal(engine.image_inputs[0], item) for item in engine.image_inputs)
    assert (
        engine.config_snapshots[0]["candidate_views"]
        == engine.config_snapshots[2]["candidate_views"]
    )
    assert (
        engine.config_snapshots[1]["candidate_views"]
        != engine.config_snapshots[0]["candidate_views"]
    )
    assert configured_a_l3 is not None

    zip_files, zip_data = _matrix_files(*fractions["a"], stages=True, response_format="zip")
    zip_data["verbosity"] = "3"
    zip_response = client.post("/v1/completions", files=zip_files, data=zip_data)
    assert zip_response.status_code == 200, zip_response.text
    manifest, members = _zip_manifest(zip_response)
    schema_manifest = copy.deepcopy(manifest)
    for artifact in schema_manifest["service"]["artifacts"]:
        artifact["data"] = ""
    CompletionResponse.model_validate(schema_manifest)
    json_descriptors = {
        item["name"]: item
        for item in configured_a_l3["service"]["artifacts"]
        if item["name"] != "identity-mask.png"
    }
    zip_descriptors = {
        item["name"]: item
        for item in manifest["service"]["artifacts"]
        if item["name"] != "identity-mask.png"
    }
    assert set(json_descriptors) == set(zip_descriptors)
    for name, descriptor in json_descriptors.items():
        payload = base64.b64decode(descriptor["data"])
        assert members[name] == payload
        assert hashlib.sha256(members[name]).hexdigest() == descriptor["sha256"]
        assert len(members[name]) == descriptor["size"] == zip_descriptors[name]["size"]
        assert zip_descriptors[name]["sha256"] == descriptor["sha256"]


IDENTITY_FLOW_CONFIG = b"""alpha: 0.5
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    debug: true
postsam2processing:
  max_w: 6
candidate_views:
  clip:
    context_fraction: 0
    max_context_pixels: 0
  blip3:
    context_fraction: 0
    max_context_pixels: 0
    contour_width: 0
visualization:
  blip3:
    - id: identity-flow
      renderer: annotated-labelled
"""


def _identity_flow_files(response_format="json"):
    image_bytes, _image = _generated_image_bytes(width=40, height=32)
    return {
        "image": ("generated.png", image_bytes, "image/png"),
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


def test_source_identity_survives_filter_semantics_order_visualization_json_and_zip():
    engine = _IdentityFlowEngine()
    client = TestClient(
        create_app(
            engine=engine,
            readiness_provider=lambda: ReadyState(True, "fake ready"),
        )
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
    }

    expected_objects = [(1, 3, 1, 36), (2, 2, 0, 6)]
    assert _identity_tuples(service["objects"]) == expected_objects
    assert engine.final_objects is not None
    assert [
        (obj.instance_id, obj.source_candidate_id, obj.filtered_index, obj.area_px)
        for obj in engine.final_objects
    ] == expected_objects
    assert [(obj.source_candidate_id, obj.filtered_index) for obj in engine.final_objects] == [
        (3, 1),
        (2, 0),
    ]

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
    record_names = {record.artifact_name for record in records}
    descriptor_names = {
        artifact["name"]
        for artifact in service["artifacts"]
        if artifact["name"] != "identity-mask.png"
    }
    assert descriptor_names == record_names
    assert all("CANDIDATE-0001" not in name for name in record_names)
    assert all(record.source_candidate_id in {2, 3} for record in records)

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
    manifest_names = {
        artifact["name"]
        for artifact in manifest["service"]["artifacts"]
        if artifact["name"] != "identity-mask.png"
    }
    assert manifest_names == record_names
    for descriptor in manifest["service"]["artifacts"]:
        if descriptor["name"] == "identity-mask.png":
            continue
        payload = members[descriptor["name"]]
        assert hashlib.sha256(payload).hexdigest() == descriptor["sha256"]
        assert len(payload) == descriptor["size"]
    assert set(members) >= {"manifest.json", "detections.yolo.txt", *record_names}
