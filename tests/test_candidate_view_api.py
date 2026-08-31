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


def _files(*, response_format="json", context_fraction=0.2, debug=True):
    config = f"""alpha: 0.5
clip:
  debug: true
  labels:
    target: a target
blip3:
  target:
    question: is this the target?
    trueresult: 'Yes'
    debug: {str(debug).lower()}
candidate_views:
  clip:
    context_fraction: 0.25
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


def test_request_local_blip3_settings_are_a_b_a_isolated_and_holder_stable():
    engine = _CandidateViewEngine()
    client = TestClient(
        create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "fake ready"))
    )
    responses = []
    for fraction in (0.1, 0.3, 0.1):
        files, data = _files(context_fraction=fraction)
        response = client.post("/v1/completions", files=files, data=data)
        assert response.status_code == 200
        responses.append(response.json())
    assert len(engine.holder_ids) == 3
    assert len(set(engine.holder_ids)) == 1
    assert np.array_equal(engine.qa.images[0], engine.qa.images[2])
    assert not np.array_equal(engine.qa.images[0], engine.qa.images[1])
    assert responses[0]["service"]["candidate_views"] == responses[2]["service"]["candidate_views"]
    assert responses[0]["service"]["candidate_views"] != responses[1]["service"]["candidate_views"]
