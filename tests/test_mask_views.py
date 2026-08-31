"""Generated-array acceptance tests for CLIP and single-image BLIP3 views."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image, ImageFilter

from modules.verifier import blip3 as blip3_module
from src.core import BoundedMemoryArtifactSink, CandidateViewConfig, build_mask_views
from src.core.errors import CoreError
from src.service.errors import ServiceError
from src.service.yaml_input import parse_hostile_config


def _clip_config(**overrides):
    values = {
        "mode": "mask_dilated",
        "context_fraction": 0.0,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
    }
    values.update(overrides)
    return CandidateViewConfig.from_mapping(values, stage="clip")


def _blip_config(**overrides):
    values = {
        "mode": "single_dilated_blur",
        "context_fraction": 0.20,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "crop_extent_multiplier": 2.0,
        "blur_sigma_fraction": 0.15,
        "contour_enabled": True,
        "contour_fraction": 0.02,
        "contour_min_pixels": 1,
        "contour_max_pixels": 3,
        "contour_rgb": [255, 224, 0],
    }
    values.update(overrides)
    return CandidateViewConfig.from_mapping(values, stage="blip3")


def _scene(shape=(120, 160)):
    rows, cols = np.indices(shape)
    return np.stack(
        (
            (rows * 7 + cols * 3 + 11) % 251,
            (rows * 5 + cols * 13 + 17) % 251,
            (rows * 19 + cols * 2 + 23) % 251,
        ),
        axis=2,
    ).astype(np.uint8)


def _oracle_dilate(mask, radius):
    result = np.zeros_like(mask, dtype=bool)
    for row, col in zip(*np.nonzero(mask)):
        y0, y1 = max(0, row - radius), min(mask.shape[0], row + radius + 1)
        x0, x1 = max(0, col - radius), min(mask.shape[1], col + radius + 1)
        yy, xx = np.indices((y1 - y0, x1 - x0))
        result[y0:y1, x0:x1] |= (yy + y0 - row) ** 2 + (xx + x0 - col) ** 2 <= radius**2
    return result


def _bbox_inclusive(mask):
    rows, cols = np.nonzero(mask)
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def test_clip_view_contract_remains_mask_derived_and_zero_filled():
    image = _scene((32, 36))
    mask = np.zeros((32, 36), dtype=bool)
    mask[8:24, 10:26] = True
    mask[12:20, 14:22] = False
    view = build_mask_views(image, mask, 7, _clip_config())
    assert view.target_bbox_xyxy == (10, 8, 26, 24)
    assert view.context_bbox_xyxy == view.target_bbox_xyxy
    assert np.all(view.target_rgb[~view.target_mask] == 0)
    assert np.array_equal(view.target_rgb[view.target_mask], image[8:24, 10:26][view.target_mask])


def test_exact_euclidean_primitive_matches_independent_oracle():
    from src.core.mask_views import _circular_dilate, exact_euclidean_dilate

    for shape, radius in [((1, 7), 3), ((7, 1), 3), ((13, 12), 5)]:
        for seed in range(4):
            rng = np.random.default_rng(seed)
            mask = rng.random(shape) < 0.22
            if not np.any(mask):
                mask[shape[0] // 2, shape[1] // 2] = True
            expected = _oracle_dilate(mask, radius)
            assert np.array_equal(_circular_dilate(mask, radius), expected)
            assert np.array_equal(exact_euclidean_dilate(mask, radius), expected)


def test_single_view_normal_geometry_pixels_and_repeat_bytes():
    image = _scene()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[30:70, 50:100] = True
    mask[43:55, 68:82] = False
    config = _blip_config()
    first = blip3_module.compose_single_blip3_view(image, mask, 7, config)
    second = blip3_module.compose_single_blip3_view(image, mask, 7, config)

    raw = _bbox_inclusive(mask)
    width = raw[2] - raw[0] + 1
    height = raw[3] - raw[1] + 1
    extent = max(width, height)
    support = _oracle_dilate(mask, math.ceil(0.20 * extent))
    contour = _oracle_dilate(support, 1) & ~support
    assert first.raw_mask_bbox_xyxy_inclusive == raw
    assert first.support_bbox_xyxy_inclusive == _bbox_inclusive(support)
    assert first.raw_context_radius == math.ceil(0.20 * extent)
    assert first.effective_context_radius == first.raw_context_radius
    assert first.raw_contour_width == math.ceil(0.02 * extent)
    assert first.effective_contour_width == 1
    assert first.effective_blur_sigma == pytest.approx(0.15 * extent)
    assert first.source_composite_shape_hw == (80, 100)
    assert first.model_input_shape_hw == (256, 320)
    x0, y0, x1, y1 = first.crop_bbox_xyxy_exclusive
    assert np.array_equal(first.support_mask, support[y0:y1, x0:x1])
    assert np.array_equal(first.contour, contour[y0:y1, x0:x1])
    assert np.array_equal(first.rgb, second.rgb)
    assert first.rgb.tobytes() == second.rgb.tobytes()
    assert not first.rgb.flags.writeable
    assert not first.source_composite.flags.writeable


def test_pixel_authority_is_source_oracle_plus_pillow_blur_and_contour():
    image = _scene((90, 110))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[30:55, 40:70] = True
    config = _blip_config(contour_rgb=[1, 200, 77], contour_fraction=0.25)
    composition = blip3_module.compose_single_blip3_view(image, mask, 1, config)
    x0, y0, x1, y1 = composition.crop_bbox_xyxy_exclusive
    source_crop = image[y0:y1, x0:x1]
    blurred = np.asarray(
        Image.fromarray(source_crop, mode="RGB").filter(
            ImageFilter.GaussianBlur(composition.effective_blur_sigma)
        )
    )
    expected = blurred.copy()
    expected[composition.support_mask] = source_crop[composition.support_mask]
    expected[composition.contour] = np.array(config.contour_rgb, dtype=np.uint8)
    assert np.array_equal(composition.source_composite, expected)
    assert np.array_equal(
        composition.source_composite[composition.support_mask],
        source_crop[composition.support_mask],
    )
    assert not np.any(composition.support_mask & composition.contour)
    assert np.all(composition.source_composite[composition.contour] == np.array([1, 200, 77]))
    assert np.array_equal(
        composition.rgb,
        np.asarray(
            Image.fromarray(expected, mode="RGB").resize(
                (composition.scaled_width, composition.scaled_height), Image.Resampling.BILINEAR
            )
        ),
    )


def test_fragmented_and_merged_masks_never_create_rectangular_bridge():
    image = _scene((120, 180))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[45:60, 30:45] = True
    mask[45:60, 120:135] = True
    composition = blip3_module.compose_single_blip3_view(
        image, mask, 2, _blip_config(context_fraction=0.0, min_context_pixels=4)
    )
    x0, y0, _, _ = composition.crop_bbox_xyxy_exclusive
    row = 52 - y0
    midpoint = (30 + 135) // 2 - x0
    assert not composition.support_mask[row, midpoint]
    assert not composition.contour[row, midpoint]

    merged = np.zeros_like(mask)
    merged[40:85, 55:105] = True
    merged_composition = blip3_module.compose_single_blip3_view(
        image, merged, 3, _blip_config(contour_enabled=False)
    )
    sx0, sy0, sx1, sy1 = merged_composition.crop_bbox_xyxy_exclusive
    source_crop = image[sy0:sy1, sx0:sx1]
    assert np.array_equal(
        merged_composition.source_composite[merged_composition.support_mask],
        source_crop[merged_composition.support_mask],
    )


@pytest.mark.parametrize("point", [(0, 0), (0, 39), (39, 0), (39, 39)])
def test_edge_and_corner_crop_clamps_without_wraparound(point):
    image = _scene((40, 40))
    mask = np.zeros((40, 40), dtype=bool)
    mask[point] = True
    composition = blip3_module.compose_single_blip3_view(
        image,
        mask,
        4,
        _blip_config(context_fraction=0, max_context_pixels=0, contour_enabled=False),
    )
    x0, y0, x1, y1 = composition.crop_bbox_xyxy_exclusive
    assert 0 <= x0 < x1 <= 40 and 0 <= y0 < y1 <= 40
    assert composition.support_mask.shape == composition.source_composite.shape[:2]
    assert np.count_nonzero(composition.raw_mask) == 1


def test_containment_rejection_is_candidate_local_and_does_not_mutate_following_flow():
    image = _scene((100, 140))
    rejected_mask = np.zeros(image.shape[:2], dtype=bool)
    rejected_mask[10, 10] = True
    valid_mask = np.zeros(image.shape[:2], dtype=bool)
    valid_mask[35:65, 50:90] = True
    with pytest.raises(blip3_module.Blip3CandidateViewRejected) as excinfo:
        blip3_module.compose_single_blip3_view(image, rejected_mask, 1, _blip_config())
    assert excinfo.value.reason == "crop_cannot_contain_support_and_contour"

    class QA:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def answer(self, model_image, query, max_new_tokens):
            self.calls.append((np.asarray(model_image).copy(), query, max_new_tokens))
            return "Yes"

    qa = QA()
    sink = BoundedMemoryArtifactSink()
    records = []
    masks = [
        {
            "segmentation": rejected_mask,
            "_source_index": 0,
            "_filtered_index": 0,
            "clip_label": "target",
            "clip_score": 0.1,
        },
        {
            "segmentation": valid_mask,
            "_source_index": 1,
            "_filtered_index": 1,
            "clip_label": "target",
            "clip_score": 0.1,
        },
    ]
    filt = blip3_module._Blip3Filter.from_qa(
        qa,
        {"target": {"question": "is this the target?", "debug": True}},
        max_questions=32,
        max_new_tokens=32,
    )
    filt.filter_masks(
        masks,
        image,
        None,
        "ignored",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_inputs=[],
        candidate_view_records=records,
    )
    assert len(qa.calls) == 1
    assert masks[0]["clip_label"] == "target"
    assert "blip3_answer" not in masks[0]
    assert masks[1]["clip_label"] == "target"
    assert [record["status"] for record in records] == ["rejected", "rendered"]
    assert records[0]["reason"] == "crop_cannot_contain_support_and_contour"
    assert sink.names() == ("blip3-verification-CANDIDATE-0002-QUESTION-0001.png",)


def test_query_is_exact_and_contains_no_superseded_visual_language():
    question = "is this a photovoltaic panel?"
    query = blip3_module.compose_verification_query(question)
    assert (
        query
        == f"[TARGET QUESTION]\n{question}\n[/TARGET QUESTION]\n{blip3_module.BLIP3_FIXED_INSTRUCTION}"
    )
    assert blip3_module.BLIP3_FIXED_INSTRUCTION == (
        "The unblurred region inside the yellow boundary is the selected candidate. "
        "The blurred surroundings are context only. Answer exactly Yes or No."
    )
    for phrase in ("left", "right", "pane", "divider", "zero-filled", "target-only"):
        assert phrase not in blip3_module.BLIP3_FIXED_INSTRUCTION.lower()


def test_inputs_are_not_mutated_and_debug_png_is_exact_model_input():
    image = _scene((100, 140))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[35:65, 50:90] = True
    image_before, mask_before = image.copy(), mask.copy()
    config = _blip_config(contour_enabled=False)
    composition = blip3_module.compose_single_blip3_view(image, mask, 8, config)
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)

    class QA:
        device = "cpu"

        def __init__(self):
            self.image = None

        def answer(self, model_image, _query, max_new_tokens):
            self.image = np.asarray(model_image).copy()
            return "Yes"

    qa = QA()
    sink = BoundedMemoryArtifactSink()
    filt = blip3_module._Blip3Filter.from_qa(
        qa,
        {"target": {"question": "is this the target?", "debug": True}},
        max_questions=32,
        max_new_tokens=32,
    )
    filt.filter_masks(
        [
            {
                "segmentation": mask,
                "_source_index": 7,
                "_filtered_index": 2,
                "clip_label": "target",
                "clip_score": 0.1,
            }
        ],
        image,
        None,
        "request",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_config=config,
    )
    assert np.array_equal(qa.image, composition.rgb)
    artifact = sink.artifacts()[0].array
    assert np.array_equal(artifact, composition.rgb)
    buffer = io.BytesIO()
    Image.fromarray(artifact, mode="RGB").save(buffer, format="PNG")
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(buffer.getvalue()))), composition.rgb)


def test_blip3_defaults_are_a_separate_exact_policy_from_clip():
    clip = CandidateViewConfig.from_mapping(None, stage="clip").as_dict(stage="clip")
    blip = CandidateViewConfig.from_mapping(None, stage="blip3").as_dict(stage="blip3")
    assert clip == {
        "mode": "mask_dilated",
        "context_fraction": 0.1,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
    }
    assert blip == {
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
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_fraction", True),
        ("context_fraction", None),
        ("context_fraction", float("nan")),
        ("crop_extent_multiplier", 0.9),
        ("blur_sigma_fraction", 0.6),
        ("contour_enabled", 1),
        ("contour_min_pixels", 0),
        ("contour_max_pixels", 4),
        ("contour_rgb", [1, 2, True]),
        ("contour_rgb", [1, 2]),
    ],
)
def test_blip3_config_rejects_strict_type_and_bound_violations(field, value):
    with pytest.raises(CoreError):
        CandidateViewConfig.from_mapping({field: value}, stage="blip3")


def test_blip3_config_rejects_legacy_fields_and_inverted_bounds():
    for value in (
        {"mode": "mask_dilated"},
        {"outside_fill": "zero"},
        {"context_intensity": 0.2},
        {"contour_width": 2},
        {"min_context_pixels": 4, "max_context_pixels": 3},
        {"contour_min_pixels": 3, "contour_max_pixels": 2},
    ):
        with pytest.raises(CoreError):
            CandidateViewConfig.from_mapping(value, stage="blip3")


def test_service_yaml_and_capabilities_expose_new_blip3_surface_only():
    valid = parse_hostile_config(
        b"""alpha: 0.5
candidate_views:
  blip3:
    context_fraction: 0.5
    min_context_pixels: 256
    max_context_pixels: 512
    crop_extent_multiplier: 1.0
    blur_sigma_fraction: 0.5
    contour_enabled: false
    contour_fraction: 0.25
    contour_min_pixels: 3
    contour_max_pixels: 3
    contour_rgb: [0, 1, 255]
""",
        verbosity=3,
    )
    blip = valid.effective_mapping["candidate_views"]["blip3"]
    assert blip["mode"] == "single_dilated_blur"
    assert blip["contour_rgb"] == [0, 1, 255]
    for raw in (
        b"candidate_views:\n  blip3:\n    contour_width: 1\n",
        b"candidate_views:\n  blip3:\n    outside_fill: zero\n",
        b"candidate_views:\n  blip3:\n    mode: mask_dilated\n",
        b"candidate_views:\n  blip3:\n    contour_rgb: [0, 1, true]\n",
    ):
        with pytest.raises(ServiceError) as excinfo:
            parse_hostile_config(b"alpha: 0.5\n" + raw, verbosity=3)
        assert excinfo.value.code in {"invalid_config", "unsupported_field"}


def test_contour_controls_use_effective_clamped_width_and_strict_exterior():
    image = _scene((120, 160))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[35:75, 55:105] = True
    for fraction, expected in ((0.0, 1), (0.05, 3), (0.25, 3)):
        composition = blip3_module.compose_single_blip3_view(
            image,
            mask,
            1,
            _blip_config(contour_fraction=fraction, contour_min_pixels=1, contour_max_pixels=3),
        )
        assert composition.effective_contour_width == expected
        assert not np.any(composition.contour & composition.support_mask)
    disabled = blip3_module.compose_single_blip3_view(
        image, mask, 1, _blip_config(contour_enabled=False, contour_fraction=0.25)
    )
    assert disabled.raw_contour_width == 13
    assert disabled.effective_contour_width == 0
    assert not np.any(disabled.contour)
