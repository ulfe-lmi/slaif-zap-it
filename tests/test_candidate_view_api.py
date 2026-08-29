"""CPU/fake API evidence for the complete candidate-view response contract."""

from __future__ import annotations

import base64
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
from src.service.schemas import CandidateViewInputRecord


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
        clip_filter.debug = True
        clip_filter.verbosity = 0
        clip_filter.log_print = lambda *_args, **_kwargs: None

        def classify(patch, _index):
            self.clip_inputs.append(patch.copy())
            return "target", 0.1, "a target"

        clip_filter.classify_single = classify
        self.clip_state = {"clip_filter": clip_filter}
        self.blip_state = {
            "blip3_filter": blip3_module._Blip3Filter.from_qa(
                self.qa,
                {
                    "target": {
                        "question": "is this the target?",
                        "trueresult": "Yes",
                        "debug": True,
                    }
                },
                max_questions=32,
                max_new_tokens=32,
            ),
            "max_questions": 32,
            "max_new_tokens": 32,
        }

    def __call__(self, image_rgb, config: CoreConfig, **kwargs):
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
