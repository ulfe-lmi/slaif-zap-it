"""Pixel-level and API-policy tests for final-object labelled visualizations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from modules import visualizer
from modules.input.images import apply_roi, resize_image
from src.core import CoreConfig, ObjectResult, StageFunctions, run_single_image
from src.service.envelope import (
    ResponseContext,
    _encode_png,
    build_completion_json,
    build_completion_zip,
)
from src.service.errors import ServiceError
from src.service.schemas import ArtifactDescriptor, ArtifactOmission
from src.service.yaml_input import parse_hostile_config


def _object(
    instance_id: int,
    label: str | None,
    mask: np.ndarray,
    *,
    score: float | None = 0.876,
) -> ObjectResult:
    metadata = {"clip_label": label, "clip_score": score}
    return ObjectResult(
        instance_id=instance_id,
        source_index=instance_id,
        mask=mask,
        metadata=metadata,
        class_id=0,
        class_id_source="mapping",
    )


def _rect(shape: tuple[int, int], top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class _DrawRecorder:
    """Record the renderer's actual Pillow geometry while delegating all drawing."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.rectangles = []
        self.text_calls = []

    def textbbox(self, *args, **kwargs):
        return self.delegate.textbbox(*args, **kwargs)

    def rectangle(self, coordinates, *args, **kwargs):
        self.rectangles.append(tuple(int(value) for value in coordinates))
        return self.delegate.rectangle(coordinates, *args, **kwargs)

    def text(self, coordinates, text, *args, **kwargs):
        font = kwargs.get("font")
        bounds = self.delegate.textbbox(coordinates, text, font=font)
        self.text_calls.append(
            {
                "coordinates": tuple(coordinates),
                "text": text,
                "font": font,
                "bounds": bounds,
                "width": int(bounds[2] - bounds[0]),
                "height": int(bounds[3] - bounds[1]),
            }
        )
        return self.delegate.text(coordinates, text, *args, **kwargs)


def _render_with_draw_recording(
    monkeypatch, image: np.ndarray, objects, *, alpha: float = 0.5, show_confidence: bool = False
):
    original_draw = visualizer.ImageDraw.Draw
    recorder = None

    def recording_draw(canvas):
        nonlocal recorder
        recorder = _DrawRecorder(original_draw(canvas))
        return recorder

    with monkeypatch.context() as patch:
        patch.setattr(visualizer.ImageDraw, "Draw", recording_draw)
        output = visualizer.render_annotated_labelled(
            image, objects, alpha=alpha, show_confidence=show_confidence
        )
    assert recorder is not None
    return output, recorder


def test_label_sanitization_is_nfkc_ascii_bounded_and_deterministic():
    assert (
        visualizer.sanitize_visualization_label("  Ｓｏｌａｒ\u3000Ｐａｎｅｌ  ") == "Solar Panel"
    )
    assert visualizer.sanitize_visualization_label("a///b\\\\c") == "a_b_c"
    assert visualizer.sanitize_visualization_label("a\x00\tb") == "a_b"
    assert visualizer.sanitize_visualization_label("") == "unknown"
    assert len(visualizer.sanitize_visualization_label("x" * 80)) == 48
    assert visualizer.sanitize_visualization_label(
        "x" * 80
    ) == visualizer.sanitize_visualization_label("x" * 80)


def test_labelled_renderer_draws_final_label_and_instance_pixels():
    image = np.full((80, 100, 3), 240, dtype=np.uint8)
    mask = _rect((80, 100), 50, 70, 35, 65)
    obj = _object(3, "solar/panel", mask)

    mask_only = visualizer.render_annotated(
        image, [{"segmentation": mask, "area": int(mask.sum())}], alpha=0.55
    )
    labelled = visualizer.render_annotated_labelled(image, [obj], alpha=0.55, show_confidence=True)

    assert labelled.shape == image.shape
    assert not np.array_equal(labelled, mask_only)
    assert np.any(np.all(labelled == (34, 70, 124), axis=2))
    assert np.any(np.all(labelled >= (249, 249, 249), axis=2))
    assert visualizer._label_text(obj, show_confidence=True) == "solar_panel 3   CLIP 0.88"
    assert obj.label == "solar/panel"
    assert np.array_equal(obj.mask, mask)


def test_labelled_renderer_repeats_exact_bytes_and_is_sensitive_to_final_label_and_id():
    image = np.full((100, 180, 3), 80, dtype=np.uint8)
    mask = _rect((100, 180), 55, 78, 70, 110)
    obj = _object(7, "stable label", mask)
    original_metadata = dict(obj.metadata)
    original_mask = obj.mask.copy()

    first = visualizer.render_annotated_labelled(image, [obj], alpha=0.45, show_confidence=True)
    second = visualizer.render_annotated_labelled(image, [obj], alpha=0.45, show_confidence=True)
    first_png = _encode_png(first)
    second_png = _encode_png(second)

    assert np.array_equal(first, second)
    assert (
        hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(second.tobytes()).hexdigest()
    )
    assert first_png == second_png
    assert hashlib.sha256(first_png).hexdigest() == hashlib.sha256(second_png).hexdigest()

    label_variant = replace(obj, metadata={**obj.metadata, "clip_label": "changed label"})
    id_variant = replace(obj, instance_id=8)
    label_pixels = visualizer.render_annotated_labelled(
        image, [label_variant], alpha=0.45, show_confidence=True
    )
    id_pixels = visualizer.render_annotated_labelled(
        image, [id_variant], alpha=0.45, show_confidence=True
    )
    label_png = _encode_png(label_pixels)
    id_png = _encode_png(id_pixels)

    assert not np.array_equal(first, label_pixels)
    assert not np.array_equal(first, id_pixels)
    assert (
        hashlib.sha256(first.tobytes()).hexdigest()
        != hashlib.sha256(label_pixels.tobytes()).hexdigest()
    )
    assert (
        hashlib.sha256(first.tobytes()).hexdigest()
        != hashlib.sha256(id_pixels.tobytes()).hexdigest()
    )
    assert hashlib.sha256(first_png).hexdigest() != hashlib.sha256(label_png).hexdigest()
    assert hashlib.sha256(first_png).hexdigest() != hashlib.sha256(id_png).hexdigest()
    assert obj.instance_id == 7
    assert obj.label == "stable label"
    assert obj.metadata == original_metadata
    assert np.array_equal(obj.mask, original_mask)


def test_labelled_renderer_uses_final_objects_not_stage_mask_metadata():
    image = np.full((80, 100, 3), 240, dtype=np.uint8)
    mask = _rect((80, 100), 50, 70, 35, 65)
    final_obj = _object(7, "BLIP3 final", mask)
    stage_mask = {"segmentation": mask, "area": int(mask.sum()), "clip_label": "old CLIP"}

    actual = visualizer.generate_visualizations(
        image,
        {"blip3": [stage_mask]},
        {"blip3": [{"id": "result", "renderer": "annotated-labelled"}]},
        default_alpha=0.55,
        final_objects=[final_obj],
    )["result"]
    expected = visualizer.render_annotated_labelled(image, [final_obj], alpha=0.55)
    old = visualizer.render_annotated_labelled(image, [_object(7, "old CLIP", mask)], alpha=0.55)

    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, old)


def test_border_and_tiny_masks_are_bounded(monkeypatch):
    for shape in ((1, 1), (2, 3), (8, 8)):
        mask = np.ones(shape, dtype=bool)
        output, recorder = _render_with_draw_recording(
            monkeypatch,
            np.zeros((*shape, 3), dtype=np.uint8),
            [_object(1, "a very long label that must be shortened", mask)],
        )
        assert output.shape == (*shape, 3)
        assert output.dtype == np.uint8
        assert len(recorder.rectangles) == len(recorder.text_calls) == 1
        left, top, right_inclusive, bottom_inclusive = recorder.rectangles[0]
        right, bottom = right_inclusive + 1, bottom_inclusive + 1
        assert 0 <= left < right <= shape[1]
        assert 0 <= top < bottom <= shape[0]

        # On a one-pixel (and similarly tiny) canvas the bitmap font cannot
        # physically fit even the mandatory instance suffix.  This checks
        # deterministic bounded clipping, not an impossible readable suffix.
        text_call = recorder.text_calls[0]
        suffix = " 1"
        assert text_call["text"].endswith(suffix)
        assert len(text_call["text"]) <= visualizer._LABEL_LIMIT + len(suffix)
        available_width = max(shape[1] - 2 * visualizer._LABEL_PADDING, 1)
        suffix_width = visualizer._text_size(recorder.delegate, suffix, text_call["font"])[0]
        if suffix_width > available_width:
            assert text_call["text"] == (
                visualizer.sanitize_visualization_label("a very long label that must be shortened")
                + suffix
            )

        repeated, repeated_recorder = _render_with_draw_recording(
            monkeypatch,
            np.zeros((*shape, 3), dtype=np.uint8),
            [_object(1, "a very long label that must be shortened", mask)],
        )
        assert np.array_equal(output, repeated)
        assert recorder.rectangles == repeated_recorder.rectangles
        assert [call["text"] for call in recorder.text_calls] == [
            call["text"] for call in repeated_recorder.text_calls
        ]


@pytest.mark.parametrize(
    ("case", "bounds"),
    [
        ("top-edge", (0, 24, 64, 104)),
        ("bottom-edge", (96, 120, 64, 104)),
        ("left-edge", (42, 78, 0, 40)),
        ("right-edge", (42, 78, 128, 160)),
        ("top-left-corner-adjacent", (2, 28, 2, 38)),
        ("bottom-right-corner-adjacent", (92, 118, 122, 158)),
    ],
)
def test_labelled_edge_layout_records_bounded_boxes_and_pillow_text_bounds(
    monkeypatch, case, bounds
):
    del case
    height, width = 120, 160
    top, bottom, left, right = bounds
    mask = _rect((height, width), top, bottom, left, right)
    output, recorder = _render_with_draw_recording(
        monkeypatch,
        np.full((height, width, 3), 80, dtype=np.uint8),
        [_object(11, "edge", mask)],
        show_confidence=True,
    )

    assert output.shape == (height, width, 3)
    assert len(recorder.rectangles) == len(recorder.text_calls) == 1
    rectangle = recorder.rectangles[0]
    box_left, box_top = rectangle[:2]
    box_right, box_bottom = rectangle[2] + 1, rectangle[3] + 1
    assert 0 <= box_left < box_right <= width
    assert 0 <= box_top < box_bottom <= height

    text_left, text_top, text_right, text_bottom = recorder.text_calls[0]["bounds"]
    suffix_width = visualizer._text_size(
        recorder.delegate, " 11   CLIP 0.88", recorder.text_calls[0]["font"]
    )[0]
    assert suffix_width <= width - 2 * visualizer._LABEL_PADDING
    assert 0 <= text_left < text_right <= width
    assert 0 <= text_top < text_bottom <= height
    assert box_left <= text_left < text_right <= box_right
    assert box_top <= text_top < text_bottom <= box_bottom
    assert recorder.text_calls[0]["text"].endswith(" 11   CLIP 0.88")


def test_labelled_layout_shortens_long_label_and_preserves_finite_suffix(monkeypatch):
    height, width = 96, 112
    long_label = "deliberately long safe visualization label for fitting"
    mask = _rect((height, width), 54, 78, 38, 74)
    output, recorder = _render_with_draw_recording(
        monkeypatch,
        np.zeros((height, width, 3), dtype=np.uint8),
        [_object(23, long_label, mask)],
        show_confidence=True,
    )

    assert output.shape == (height, width, 3)
    text_call = recorder.text_calls[0]
    suffix = " 23   CLIP 0.88"
    sanitized = visualizer.sanitize_visualization_label(long_label)
    assert text_call["text"].endswith(suffix)
    assert text_call["text"][: -len(suffix)] != sanitized
    assert len(text_call["text"][: -len(suffix)]) < len(sanitized)
    suffix_width = visualizer._text_size(recorder.delegate, suffix, text_call["font"])[0]
    assert suffix_width <= width - 2 * visualizer._LABEL_PADDING
    assert text_call["width"] <= width - 2 * visualizer._LABEL_PADDING
    assert (
        text_call["width"]
        == visualizer._text_size(recorder.delegate, text_call["text"], text_call["font"])[0]
    )


def test_nearby_label_boxes_are_not_completely_overlapped_when_candidates_allow(monkeypatch):
    image = np.full((90, 120, 3), 240, dtype=np.uint8)
    first_mask = _rect((90, 120), 40, 58, 45, 65)
    second_mask = _rect((90, 120), 40, 58, 67, 87)
    objects = [_object(1, "first", first_mask), _object(2, "second", second_mask)]
    output, recorder = _render_with_draw_recording(monkeypatch, image, objects)
    repeated, repeated_recorder = _render_with_draw_recording(monkeypatch, image, objects)
    assert np.array_equal(output, repeated)
    assert recorder.rectangles == repeated_recorder.rectangles
    assert [call["coordinates"] for call in recorder.text_calls] == [
        call["coordinates"] for call in repeated_recorder.text_calls
    ]

    colors = ((34, 70, 124), (97, 114, 144))
    boxes = []
    for color in colors:
        rows, columns = np.where(np.all(output == color, axis=2))
        assert len(rows) > 0
        boxes.append(
            (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        )
    intersection = visualizer._intersection_area(boxes[0], boxes[1])
    first_area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
    second_area = (boxes[1][2] - boxes[1][0]) * (boxes[1][3] - boxes[1][1])
    assert intersection < min(first_area, second_area)

    repeated_boxes = []
    for color in colors:
        rows, columns = np.where(np.all(repeated == color, axis=2))
        repeated_boxes.append(
            (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        )
    assert boxes == repeated_boxes


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), float("-inf")])
def test_confidence_is_optional_and_nonfinite_scores_are_not_rendered(score):
    obj = _object(2, "thing", np.ones((20, 40), dtype=bool), score=score)
    assert visualizer._label_text(obj, show_confidence=False) == "thing 2"
    assert visualizer._label_text(obj, show_confidence=True) == "thing 2"


def test_confidence_uses_finite_two_decimal_format_and_changes_pixels():
    mask = np.ones((60, 120), dtype=bool)
    obj = _object(2, "thing", mask, score=0.876)
    no_score = visualizer.render_annotated_labelled(
        np.zeros((60, 120, 3), dtype=np.uint8), [obj], show_confidence=False
    )
    with_score = visualizer.render_annotated_labelled(
        np.zeros((60, 120, 3), dtype=np.uint8), [obj], show_confidence=True
    )
    assert visualizer._label_text(obj, show_confidence=True) == "thing 2   CLIP 0.88"
    assert not np.array_equal(no_score, with_score)


def _final_core_outcome():
    shape = (80, 100)
    masks = [
        {"segmentation": _rect(shape, 8, 30, 8, 34), "area": 22 * 26},
        {"segmentation": _rect(shape, 45, 65, 55, 78), "area": 20 * 23},
    ]

    def fake_sam(state, _params, _image, **_kwargs):
        return state, [dict(mask) for mask in masks], {}

    def fake_clip(state, params, _image, **_kwargs):
        for index, mask in enumerate(params["masks"]):
            mask["clip_label"] = f"clip-{index}"
            mask["clip_score"] = 0.4 + index / 10
        return state, params["masks"], {}

    def fake_blip3(state, params, _image, **_kwargs):
        for index, mask in enumerate(params["masks"]):
            mask["clip_label"] = f"final-{index}"
            mask["clip_score"] = 0.876 + index / 100
        return state, params["masks"], {}

    stages = StageFunctions(
        apply_roi=apply_roi,
        resize_image=resize_image,
        run_sam2=fake_sam,
        filter_by_area_bbox=lambda current, *_args, **_kwargs: current,
        run_clip=fake_clip,
        run_blip3=fake_blip3,
    )
    config = CoreConfig(
        alpha=0.55,
        roi_val=None,
        resize_val=None,
        prep_debug=False,
        clip_cfg={"enabled": True},
        blip3_cfg={"thing": {"question": "is this a thing?"}},
        vis_cfg={
            "blip3": [
                {
                    "id": "labelled-result",
                    "renderer": "annotated-labelled",
                    "alpha": 0.55,
                    "show_confidence": True,
                }
            ]
        },
    )
    return run_single_image(
        np.zeros((*shape, 3), dtype=np.uint8),
        config,
        stages=stages,
        class_labels=("final-0", "final-1"),
        verbosity=3,
        render_visualizations=True,
    )


def test_blip3_final_order_drives_manifest_and_json_zip_artifact_parity():
    outcome = _final_core_outcome()
    objects = outcome.result.objects
    assert [obj.instance_id for obj in objects] == [1, 2]
    assert [obj.label for obj in objects] == ["final-0", "final-1"]
    assert set(outcome.result.rendered) == {"labelled-result"}

    context = ResponseContext(
        request_id="request",
        model_id="zap-it-1",
        verbosity=3,
        response_format="json",
        config_digest="digest",
        class_mapping={"final-0": 0, "final-1": 1},
    )
    document = build_completion_json(outcome, context)
    payload = base64.b64decode(document["service"]["artifacts"][1]["data"])
    with zipfile.ZipFile(
        io.BytesIO(build_completion_zip(outcome, context, max_bytes=1 << 20))
    ) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        zip_payload = archive.read("visualization/labelled-result.png")

    assert [item["instance_id"] for item in document["service"]["objects"]] == [1, 2]
    assert [item["label"] for item in document["service"]["objects"]] == ["final-0", "final-1"]
    assert manifest["service"]["artifacts"][1]["sha256"] == hashlib.sha256(zip_payload).hexdigest()
    assert document["service"]["artifacts"][1]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert document["service"]["artifacts"][1]["size"] == len(payload) == len(zip_payload)
    assert payload == zip_payload
    assert manifest["service"]["artifacts"][1]["name"] == "visualization/labelled-result.png"


def test_service_safe_visualizations_keep_fixed_names_and_report_logical_ids():
    outcome = _final_core_outcome()
    context = ResponseContext(
        request_id="request",
        model_id="zap-it-1",
        verbosity=3,
        response_format="json",
        config_digest="digest",
        class_mapping={"final-0": 0, "final-1": 1},
        service_safe_artifact_names=True,
    )

    document = build_completion_json(outcome, context)
    artifacts = document["service"]["artifacts"]
    assert artifacts[0]["name"] == "identity-mask.png"
    assert "visualization_id" not in artifacts[0]
    visualization = artifacts[1]
    assert visualization["name"] == "visualization/stream-0001.png"
    assert visualization["visualization_id"] == "labelled-result"
    ArtifactDescriptor.model_validate(visualization)

    zip_payload = build_completion_zip(outcome, context, max_bytes=1 << 20)
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        member = archive.read("visualization/stream-0001.png")
    manifest_visualization = manifest["service"]["artifacts"][1]
    assert {key: value for key, value in visualization.items() if key != "data"} == (
        manifest_visualization
    )
    assert member == base64.b64decode(visualization["data"])

    trusted = build_completion_json(
        outcome,
        replace(context, service_safe_artifact_names=False),
    )
    assert trusted["service"]["artifacts"][1]["name"] == "visualization/labelled-result.png"
    assert "visualization_id" not in trusted["service"]["artifacts"][1]


def test_service_safe_multiple_visualizations_have_stable_ordinal_names_and_ledger_ids():
    outcome = _final_core_outcome()
    rendered = {
        "path-like-looking-but-validated": np.full((80, 100, 3), 11, dtype=np.uint8),
        "second-stream": np.full((80, 100, 3), 22, dtype=np.uint8),
    }
    outcome = replace(outcome, result=replace(outcome.result, rendered=rendered))
    context = ResponseContext(
        request_id="request",
        model_id="zap-it-1",
        verbosity=3,
        response_format="json",
        config_digest="digest",
        class_mapping={"final-0": 0, "final-1": 1},
        max_response_artifacts=2,
        service_safe_artifact_names=True,
    )
    document = build_completion_json(outcome, context)
    visuals = [
        item
        for item in document["service"]["artifacts"]
        if item["name"].startswith("visualization/")
    ]
    assert [item["name"] for item in visuals] == [
        "visualization/stream-0001.png",
    ]
    assert visuals[0]["visualization_id"] == "path-like-looking-but-validated"
    delivery = document["service"]["artifact_delivery"]
    assert delivery["omitted"] == [
        {
            "name": "visualization/stream-0002.png",
            "stage": "visualization",
            "source_candidate_id": None,
            "question_id": None,
            "estimated_raw_bytes": rendered["second-stream"].nbytes,
            "reason": "omitted_count_limit",
            "visualization_id": "second-stream",
        }
    ]
    ArtifactOmission.model_validate(delivery["omitted"][0])
    with pytest.raises(ValueError):
        ArtifactDescriptor.model_validate({**visuals[0], "visualization_id": "../escape"})

    changed = replace(
        outcome,
        result=replace(
            outcome.result,
            rendered={
                "renamed-stream": rendered["path-like-looking-but-validated"],
                "second-stream": rendered["second-stream"],
            },
        ),
    )
    changed_document = build_completion_json(outcome, replace(context, max_response_artifacts=3))
    renamed_document = build_completion_json(changed, replace(context, max_response_artifacts=3))
    original_visuals = [
        item
        for item in changed_document["service"]["artifacts"]
        if item["name"].startswith("visualization/")
    ]
    renamed_visuals = [
        item
        for item in renamed_document["service"]["artifacts"]
        if item["name"].startswith("visualization/")
    ]
    assert [item["name"] for item in original_visuals] == [item["name"] for item in renamed_visuals]
    assert [item["sha256"] for item in original_visuals] == [
        item["sha256"] for item in renamed_visuals
    ]
    assert [item["visualization_id"] for item in renamed_visuals] == [
        "renamed-stream",
        "second-stream",
    ]


@pytest.mark.parametrize("stage", ["sam2", "clip"])
def test_labelled_renderer_is_only_allowed_at_final_stage(stage):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(
            (
                "visualization:\n"
                f"  {stage}:\n"
                "    - id: labelled\n"
                "      renderer: annotated-labelled\n"
            ).encode(),
            verbosity=3,
        )
    assert excinfo.value.code == "unsupported_field"


def test_labelled_policy_requires_strict_confidence_and_rejects_unsupported_rules():
    valid = parse_hostile_config(
        b"visualization:\n  blip3:\n    - id: labelled\n      renderer: annotated-labelled\n      show_confidence: true\n",
        verbosity=3,
    )
    assert valid.effective_mapping["visualization"]["blip3"][0]["show_confidence"] is True

    with pytest.raises(ServiceError) as non_boolean:
        parse_hostile_config(
            b"visualization:\n  blip3:\n    - id: labelled\n      renderer: annotated-labelled\n      show_confidence: 1\n",
            verbosity=3,
        )
    assert non_boolean.value.code == "invalid_config"

    with pytest.raises(ServiceError) as unknown:
        parse_hostile_config(
            b"visualization:\n  blip3:\n    - id: labelled\n      renderer: annotated-labelled\n      extra: true\n",
            verbosity=3,
        )
    assert unknown.value.code == "unsupported_field"


def test_l0_l2_do_not_execute_labelled_visualization_or_reserve_raw_bytes():
    config = b"visualization:\n  blip3:\n    - id: labelled\n      renderer: annotated-labelled\n"
    from fastapi.testclient import TestClient

    from src.service import ReadyState, ServiceSettings, create_app
    from src.service.fake_engine import FakeEngine

    engine = FakeEngine()
    app = create_app(
        engine=engine,
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(True, "ready"),
    )
    image = io.BytesIO()
    Image.new("RGB", (32, 24), (1, 2, 3)).save(image, format="PNG")
    files = {
        "image": ("image.png", image.getvalue(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }
    with TestClient(app) as client:
        for verbosity in ("0", "1", "2"):
            response = client.post("/v1/completions", files=files, data={"verbosity": verbosity})
            assert response.status_code == 200
    assert len(engine.calls) == 3
