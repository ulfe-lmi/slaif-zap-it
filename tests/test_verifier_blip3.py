import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from modules.verifier import blip3 as blip_mod
from src.core import BoundedMemoryArtifactSink


def test_initialize_dryrun_alternates_labels():
    state = blip_mod.initialize({}, dryrun=True, verbosity=0)
    masks = [
        {"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "initial"} for _ in range(3)
    ]
    filt = state["blip3_filter"]
    updated, answers = filt.filter_masks(masks, np.zeros((1, 1, 3), dtype=np.uint8), ".", "frame")
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
    state, processed, meta = blip_mod.run(
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


def _square_ring(mask, radius=4):
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for row_delta in range(2 * radius + 1):
        for col_delta in range(2 * radius + 1):
            dilated |= padded[
                row_delta : row_delta + height,
                col_delta : col_delta + width,
            ]
    return dilated & ~mask


def test_composer_rejects_invalid_image_mask_and_empty_mask():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=bool)
    with pytest.raises(TypeError):
        blip_mod.compose_verification_image(image.astype(np.int16), mask)
    with pytest.raises(TypeError):
        blip_mod.compose_verification_image(image, mask.astype(np.uint8))
    with pytest.raises(TypeError):
        blip_mod.compose_verification_image(image, np.zeros((8, 8), dtype=bool))
    with pytest.raises(ValueError, match="at least one"):
        blip_mod.compose_verification_image(image, mask)


def test_composer_crop_metadata_handles_borders_and_spanning_mask():
    image = np.zeros((300, 400, 3), dtype=np.uint8)

    ordinary = np.zeros((300, 400), dtype=bool)
    ordinary[100:120, 150:180] = True
    composed = blip_mod.compose_verification_image(image, ordinary)
    assert composed.crop_box_xyxy == (100, 45, 228, 173)
    assert composed.crop_shape_hw == (128, 128)
    assert composed.scaled_shape_hw == (256, 256)
    assert composed.paired.shape == (256, 516, 3)

    for rows, cols in (
        (slice(0, 10), slice(0, 10)),
        (slice(290, 300), slice(390, 400)),
        (slice(0, 10), slice(390, 400)),
        (slice(290, 300), slice(0, 10)),
    ):
        border_mask = np.zeros((300, 400), dtype=bool)
        border_mask[rows, cols] = True
        border_composed = blip_mod.compose_verification_image(image, border_mask)
        x0, y0, x1, y1 = border_composed.crop_box_xyxy
        assert 0 <= x0 < x1 <= 400
        assert 0 <= y0 < y1 <= 300
        assert x1 - x0 == 128
        assert y1 - y0 == 128

    spanning = blip_mod.compose_verification_image(image, np.ones((300, 400), dtype=bool))
    assert spanning.crop_box_xyxy == (0, 0, 400, 300)
    assert spanning.scaled_shape_hw == (300, 400)
    assert spanning.paired.shape == (300, 804, 3)


def test_composer_uses_one_exact_nearest_mapping_for_rgb_and_mask():
    image = np.arange(100 * 100 * 3, dtype=np.uint32).reshape(100, 100, 3)
    image = (image % 256).astype(np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    composed = blip_mod.compose_verification_image(image, mask)
    crop = image[
        composed.crop_box_xyxy[1] : composed.crop_box_xyxy[3],
        composed.crop_box_xyxy[0] : composed.crop_box_xyxy[2],
    ]
    crop_mask = mask[
        composed.crop_box_xyxy[1] : composed.crop_box_xyxy[3],
        composed.crop_box_xyxy[0] : composed.crop_box_xyxy[2],
    ]
    rows = blip_mod._nearest_indices(crop.shape[0], composed.scaled_height)
    cols = blip_mod._nearest_indices(crop.shape[1], composed.scaled_width)
    expected = crop[np.ix_(rows, cols)]
    expected_mask = crop_mask[np.ix_(rows, cols)]
    assert np.array_equal(composed.paired[:, : composed.scaled_width], expected)
    assert np.array_equal(composed.scaled_mask, expected_mask)
    assert composed.scaled_shape_hw[0] >= 1 and composed.scaled_shape_hw[1] >= 1

    large = np.zeros((1600, 1600), dtype=bool)
    large[300:1300, 300:1300] = True
    downscaled = blip_mod.compose_verification_image(
        np.zeros((1600, 1600, 3), dtype=np.uint8), large
    )
    assert max(downscaled.scaled_shape_hw) == 768


def test_spotlight_pixels_contour_and_dimming_are_exact_and_component_aware():
    image = np.full((80, 80, 3), (11, 101, 251), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:22, 20:22] = True
    mask[20:22, 58:60] = True
    composed = blip_mod.compose_verification_image(image, mask)
    left = composed.paired[:, : composed.scaled_width]
    right = composed.paired[:, composed.scaled_width + 4 :]
    expected_contour = _square_ring(composed.scaled_mask)
    assert np.array_equal(composed.contour, expected_contour)
    assert not np.any(composed.contour & composed.scaled_mask)
    assert np.array_equal(right[composed.scaled_mask], left[composed.scaled_mask])
    assert np.all(right[composed.contour] == np.array((255, 224, 0), dtype=np.uint8))
    exterior = ~composed.scaled_mask & ~composed.contour
    assert np.array_equal(right[exterior], (left[exterior].astype(np.uint16) * 2 // 5))
    assert np.all(composed.paired[:, composed.scaled_width : composed.scaled_width + 4] == 0)

    # A bbox-only rectangle would outline the gap between these components.
    component_row = np.flatnonzero(composed.scaled_mask.any(axis=1))[0]
    first_col = np.flatnonzero(composed.scaled_mask[component_row])
    assert first_col.size
    midpoint = (int(first_col.min()) + int(first_col.max())) // 2
    assert not composed.contour[component_row, midpoint]

    boundary_mask = np.zeros((80, 80), dtype=bool)
    boundary_mask[0:3, 0:3] = True
    boundary = blip_mod.compose_verification_image(image, boundary_mask)
    boundary_left = boundary.paired[:, : boundary.scaled_width]
    boundary_right = boundary.paired[:, boundary.scaled_width + 4 :]
    assert np.array_equal(boundary_right[boundary.scaled_mask], boundary_left[boundary.scaled_mask])


class _SpotlightQA:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def answer(self, image, query, max_new_tokens):
        array = np.asarray(image)
        self.calls.append((image.copy(), query, max_new_tokens))
        right = array[:, array.shape[1] // 2 + 2 :]
        panel_pixels = np.all(right == np.array((220, 220, 220), dtype=np.uint8), axis=2)
        return "Yes" if np.any(panel_pixels) else "No"


def test_mask_aware_positive_and_same_crop_hard_negative():
    scene = np.full((220, 220, 3), (80, 50, 30), dtype=np.uint8)
    scene[60:80, 60:80] = (220, 220, 220)
    scene[95:115, 95:115] = (220, 220, 220)
    positive_mask = np.zeros((220, 220), dtype=bool)
    positive_mask[60:80, 60:80] = True
    negative_mask = np.zeros((220, 220), dtype=bool)
    negative_mask[60:80, 100:120] = True
    config = {
        "solar_panel": {
            "question": "is this a photovoltaic panel?",
            "trueresult": "Yes",
            "falseresult": "No",
        }
    }

    qa = _SpotlightQA()
    positive = {"segmentation": positive_mask, "clip_label": "solar_panel", "clip_score": 0.2}
    filt = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    filt.filter_masks([positive], scene, None, "ignored", service_safe_artifact_names=True)
    assert positive["clip_label"] == "solar_panel"
    assert positive["blip3_answer"] == "Yes"

    qa = _SpotlightQA()
    negative = {"segmentation": negative_mask, "clip_label": "solar_panel", "clip_score": 0.2}
    filt = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    filt.filter_masks([negative], scene, None, "ignored", service_safe_artifact_names=True)
    assert negative["clip_label"] == "negative"
    assert negative["blip3_answer"] == "No"
    assert blip_mod.compose_verification_query(config["solar_panel"]["question"]).endswith(
        blip_mod.BLIP3_FIXED_INSTRUCTION
    )
    assert all(
        query.endswith(blip_mod.BLIP3_FIXED_INSTRUCTION)
        and "[TARGET QUESTION]\nis this a photovoltaic panel?\n[/TARGET QUESTION]" in query
        for _image, query, _tokens in qa.calls
    )

    crop_box = blip_mod.compose_verification_image(scene, negative_mask).crop_box_xyxy
    ordinary_crop = scene[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    assert np.any(np.all(ordinary_crop == (220, 220, 220), axis=2))
    assert not np.any(
        np.all(
            np.asarray(qa.calls[0][0])[:, qa.calls[0][0].width // 2 + 2 :] == (220, 220, 220),
            axis=2,
        )
    )


def test_any_and_label_rules_reuse_paired_image_and_queries():
    class QA:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def answer(self, image, query, max_new_tokens):
            self.calls.append((image, query, max_new_tokens))
            return "unclear" if len(self.calls) == 1 else "Yes"

    qa = QA()
    mask = {
        "segmentation": np.ones((20, 20), dtype=bool),
        "clip_label": "goat",
        "clip_score": 0.1,
    }
    config = {
        "any,0.5": {"question": "is there an animal?"},
        "goat": {"question": "is this a goat?", "trueresult": "Yes", "newcategory": "animal"},
    }
    filt = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    filt.filter_masks([mask], np.zeros((20, 20, 3), dtype=np.uint8), None, "frame")
    assert len(qa.calls) == 2
    assert qa.calls[0][0] is qa.calls[1][0]
    assert qa.calls[0][2] == qa.calls[1][2] == 32
    assert "[TARGET QUESTION]\nis there an animal?\n[/TARGET QUESTION]" in qa.calls[0][1]
    assert "[TARGET QUESTION]\nis this a goat?\n[/TARGET QUESTION]" in qa.calls[1][1]
    assert mask["clip_label"] == "animal"


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
        "label/with hostile text": {
            "question": "is this safe?",
            "trueresult": "Yes",
            "debug": True,
        }
    }
    masks = [
        {"segmentation": np.ones((20, 20), dtype=bool), "clip_label": "label/with hostile text"},
        {"segmentation": np.eye(20, dtype=bool), "clip_label": "label/with hostile text"},
    ]
    filt = blip_mod._Blip3Filter.from_qa(qa, config, max_questions=32, max_new_tokens=32)
    filt.filter_masks(
        masks,
        np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3),
        tmp_path,
        "../hostile/frame",
        artifact_sink=sink,
        service_safe_artifact_names=True,
    )
    assert sink.names() == (
        "blip3-verification-0000-0000.png",
        "blip3-verification-0001-0001.png",
    )
    for index, artifact in enumerate(sink.artifacts()):
        assert artifact.content_type == "image/png"
        assert np.array_equal(artifact.array, qa.images[index])
        buffer = io.BytesIO()
        Image.fromarray(artifact.array).save(buffer, format="PNG")
        first_bytes = buffer.getvalue()
        repeated = io.BytesIO()
        Image.fromarray(artifact.array).save(repeated, format="PNG")
        assert first_bytes == repeated.getvalue()
        assert (
            hashlib.sha256(first_bytes).hexdigest()
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
    assert cli_sink.names() == ("frame-blip3-verification-0000-0000.png",)
    assert not any(
        fragment in cli_sink.names()[0] for fragment in ("hostile", "safe", "answer", "label")
    )
