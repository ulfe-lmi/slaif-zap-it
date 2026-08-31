"""BLIP3 adapter tests using deterministic fake QA holders."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from modules.verifier import blip3 as blip_mod
from src.core import BoundedMemoryArtifactSink
from src.core.errors import CoreError


def _image_and_mask():
    rows, cols = np.indices((100, 140))
    image = np.stack(
        ((rows * 7 + cols * 3) % 251, (rows * 5 + cols * 11) % 251, (rows * 13 + cols * 2) % 251),
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((100, 140), dtype=bool)
    mask[35:65, 50:90] = True
    return image, mask


def test_initialize_dryrun_alternates_labels():
    state = blip_mod.initialize({}, dryrun=True, verbosity=0)
    masks = [
        {"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "initial"} for _ in range(3)
    ]
    updated, answers = state["blip3_filter"].filter_masks(
        masks, np.zeros((1, 1, 3), dtype=np.uint8), ".", "frame"
    )
    assert answers[0].startswith("dryrun")
    assert updated[0]["clip_label"] == "negative"
    assert updated[1]["clip_label"] != updated[0]["clip_label"]


def test_run_requires_masks():
    with pytest.raises(ValueError):
        blip_mod.run({}, {"dryrun": True}, np.zeros((1, 1, 3), dtype=np.uint8))


def test_run_with_mock_filter(monkeypatch):
    class FakeFilter:
        def __init__(self, cfg, device="cpu", verbosity=0, log_print_func=None):
            self.cfg = cfg

        def filter_masks(self, masks, image_np, out_dir, fname_stem):
            for mask in masks:
                mask["clip_label"] = "approved"
            return masks, ["answer"]

    monkeypatch.setattr(blip_mod, "_Blip3Filter", FakeFilter)
    masks = [{"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "maybe"}]
    _state, processed, meta = blip_mod.run(
        None,
        {
            "dryrun": False,
            "config": {"label": {}},
            "masks": masks,
            "out_dir": ".",
            "fname_stem": "img",
        },
        np.zeros((1, 1, 3), dtype=np.uint8),
    )
    assert processed[0]["clip_label"] == "approved"
    assert meta["answers"] == ["answer"]


class _QA:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def answer(self, image, query, max_new_tokens):
        self.calls.append((image, query, max_new_tokens))
        return "Yes"


def test_multiple_rules_share_one_final_image_and_preserve_questions():
    image, mask = _image_and_mask()
    qa = _QA()
    records = []
    masks = [
        {
            "segmentation": mask,
            "_source_index": 7,
            "_filtered_index": 2,
            "clip_label": "goat",
            "clip_score": 0.1,
        }
    ]
    filter_ = blip_mod._Blip3Filter.from_qa(
        qa,
        {
            "any,0.5": {"question": "is there an animal?"},
            "goat": {"question": "is this a goat?", "trueresult": "Yes", "newcategory": "animal"},
        },
        max_questions=32,
        max_new_tokens=32,
    )
    filter_.filter_masks(masks, image, None, "request", candidate_view_records=records)
    assert len(qa.calls) == 2
    assert qa.calls[0][0] is qa.calls[1][0]
    assert all(call[2] == 32 for call in qa.calls)
    assert "is there an animal?" in qa.calls[0][1]
    assert "is this a goat?" in qa.calls[1][1]
    assert qa.calls[0][1].endswith(blip_mod.BLIP3_FIXED_INSTRUCTION)
    assert masks[0]["clip_label"] == "animal"
    assert len(records) == 1 and records[0]["status"] == "rendered"


def test_debug_artifact_is_the_exact_single_model_input():
    image, mask = _image_and_mask()
    qa = _QA()
    sink = BoundedMemoryArtifactSink()
    records = []
    filter_ = blip_mod._Blip3Filter.from_qa(
        qa,
        {"target": {"question": "is this safe?", "trueresult": "Yes", "debug": True}},
        max_questions=32,
        max_new_tokens=32,
    )
    filter_.filter_masks(
        [
            {
                "segmentation": mask,
                "_source_index": 7,
                "_filtered_index": 3,
                "clip_label": "target",
                "clip_score": 0.1,
            }
        ],
        image,
        None,
        "request",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_inputs=records,
    )
    assert sink.names() == ("blip3-verification-CANDIDATE-0008-QUESTION-0001.png",)
    assert np.array_equal(sink.artifacts()[0].array, np.asarray(qa.calls[0][0]))
    assert records[0]["raw_mask_bbox_xyxy_inclusive"] == [50, 35, 89, 64]
    assert records[0]["model_input_dimensions"] == {
        "height": qa.calls[0][0].height,
        "width": qa.calls[0][0].width,
    }


def test_composer_rejects_invalid_image_mask_and_empty_mask():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=bool)
    with pytest.raises(CoreError):
        blip_mod.compose_verification_image(image.astype(np.int16), mask)
    with pytest.raises(CoreError):
        blip_mod.compose_verification_image(image, mask.astype(np.uint8))
    with pytest.raises(CoreError):
        blip_mod.compose_verification_image(image, np.zeros((8, 8), dtype=bool))
    with pytest.raises(CoreError, match="non-empty"):
        blip_mod.compose_verification_image(image, mask)


def test_composer_crop_metadata_handles_borders_and_spanning_mask():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    ordinary = np.zeros((300, 400), dtype=bool)
    ordinary[100:120, 150:180] = True
    composed = blip_mod.compose_verification_image(image, ordinary)
    assert composed.crop_bbox_xyxy_exclusive == (135, 90, 195, 130)
    assert composed.crop_shape_hw == (40, 60)
    assert composed.model_input_shape_hw == (256, 384)
    assert not hasattr(composed, "paired")

    for rows, cols in (
        (slice(0, 10), slice(0, 10)),
        (slice(290, 300), slice(390, 400)),
        (slice(0, 10), slice(390, 400)),
        (slice(290, 300), slice(0, 10)),
    ):
        border_mask = np.zeros((300, 400), dtype=bool)
        border_mask[rows, cols] = True
        border_composed = blip_mod.compose_verification_image(image, border_mask)
        x0, y0, x1, y1 = border_composed.crop_bbox_xyxy_exclusive
        assert 0 <= x0 < x1 <= 400
        assert 0 <= y0 < y1 <= 300
        assert border_composed.source_composite.shape[:2] == (y1 - y0, x1 - x0)

    spanning = blip_mod.compose_verification_image(image, np.ones((300, 400), dtype=bool))
    assert spanning.crop_bbox_xyxy_exclusive == (0, 0, 400, 300)
    assert spanning.model_input_shape_hw == (300, 400)
    assert spanning.rgb.shape == (300, 400, 3)


def test_composer_resizes_one_source_image_with_exact_rgb_pixels():
    image = np.arange(100 * 100 * 3, dtype=np.uint32).reshape(100, 100, 3)
    image = (image % 256).astype(np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    composed = blip_mod.compose_verification_image(image, mask)
    expected = np.asarray(
        Image.fromarray(composed.source_composite, mode="RGB").resize(
            (composed.scaled_width, composed.scaled_height), Image.Resampling.BILINEAR
        )
    )
    assert np.array_equal(composed.rgb, expected)
    assert composed.scaled_shape_hw[0] >= 1 and composed.scaled_shape_hw[1] >= 1

    large = np.zeros((1600, 1600), dtype=bool)
    large[300:1300, 300:1300] = True
    downscaled = blip_mod.compose_verification_image(
        np.zeros((1600, 1600, 3), dtype=np.uint8), large
    )
    assert max(downscaled.scaled_shape_hw) == 768


class _SpotlightQA:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def answer(self, image, query, max_new_tokens):
        array = np.asarray(image)
        self.calls.append((image.copy(), query, max_new_tokens))
        white = np.all(array == np.array((220, 220, 220), dtype=np.uint8), axis=2)
        return "Yes" if np.any(white) else "No"


def test_mask_aware_positive_and_same_crop_hard_negative():
    scene = np.full((220, 220, 3), (80, 50, 30), dtype=np.uint8)
    scene[60:80, 60:80] = (220, 220, 220)
    positive_mask = np.zeros((220, 220), dtype=bool)
    positive_mask[60:80, 60:80] = True
    negative_mask = np.zeros((220, 220), dtype=bool)
    negative_mask[100:120, 100:120] = True
    config = {
        "solar_panel": {
            "question": "is this a photovoltaic panel?",
            "trueresult": "Yes",
            "falseresult": "No",
        }
    }

    qa = _SpotlightQA()
    positive = {"segmentation": positive_mask, "clip_label": "solar_panel", "clip_score": 0.2}
    blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32).filter_masks(
        [positive], scene, None, "ignored", service_safe_artifact_names=True
    )
    assert positive["clip_label"] == "solar_panel"
    assert positive["blip3_answer"] == "Yes"

    qa = _SpotlightQA()
    negative = {"segmentation": negative_mask, "clip_label": "solar_panel", "clip_score": 0.2}
    blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32).filter_masks(
        [negative], scene, None, "ignored", service_safe_artifact_names=True
    )
    assert negative["clip_label"] == "negative"
    assert negative["blip3_answer"] == "No"
    assert all(
        query.endswith(blip_mod.BLIP3_FIXED_INSTRUCTION)
        and "[TARGET QUESTION]\nis this a photovoltaic panel?\n[/TARGET QUESTION]" in query
        for _image, query, _tokens in qa.calls
    )


def test_service_debug_artifacts_are_fixed_png_names_and_exact_qa_arrays(tmp_path):
    class QA:
        device = "cpu"

        def __init__(self):
            self.images = []

        def answer(self, image, _query, max_new_tokens):
            assert max_new_tokens == 32
            self.images.append(np.asarray(image).copy())
            return "Yes"

    qa = QA()
    sink = BoundedMemoryArtifactSink()
    config = {
        "label/with hostile text": {"question": "is this safe?", "trueresult": "Yes", "debug": True}
    }
    masks = [
        {"segmentation": np.ones((20, 20), dtype=bool), "clip_label": "label/with hostile text"},
        {"segmentation": np.eye(20, dtype=bool), "clip_label": "label/with hostile text"},
    ]
    filter_ = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    filter_.filter_masks(
        masks,
        np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3),
        tmp_path,
        "../hostile/frame",
        artifact_sink=sink,
        service_safe_artifact_names=True,
    )
    assert sink.names() == (
        "blip3-verification-CANDIDATE-0001-QUESTION-0001.png",
        "blip3-verification-CANDIDATE-0002-QUESTION-0002.png",
    )
    for index, artifact in enumerate(sink.artifacts()):
        assert artifact.content_type == "image/png"
        assert np.array_equal(artifact.array, qa.images[index])
        buffer = io.BytesIO()
        Image.fromarray(artifact.array).save(buffer, format="PNG")
        repeated = io.BytesIO()
        Image.fromarray(artifact.array).save(repeated, format="PNG")
        decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())))
        assert np.array_equal(decoded, qa.images[index])
        assert buffer.getvalue() == repeated.getvalue()
        assert (
            hashlib.sha256(buffer.getvalue()).hexdigest()
            == hashlib.sha256(repeated.getvalue()).hexdigest()
        )
    assert not list(tmp_path.rglob("*"))

    cli_sink = BoundedMemoryArtifactSink()
    cli_filter = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    cli_filter.filter_masks(
        [masks[0]],
        np.zeros((20, 20, 3), dtype=np.uint8),
        tmp_path,
        "../frame",
        artifact_sink=cli_sink,
    )
    assert cli_sink.names() == ("frame-blip3-verification-CANDIDATE-0001-QUESTION-0001.png",)
    assert not any(
        fragment in cli_sink.names()[0] for fragment in ("hostile", "safe", "answer", "label")
    )
