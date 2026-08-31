"""Generated-array acceptance tests for CLIP and single-image BLIP3 views."""

from __future__ import annotations

import io
import hashlib
import math
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageFilter

from modules.classifier import clip as clip_module
from modules.verifier import blip3 as blip3_module
from src.core import (
    BoundedMemoryArtifactSink,
    CandidateViewConfig,
    build_mask_views,
    build_raw_clip_crop,
)
from src.core.errors import CoreError
from src.service.capabilities import build_capabilities
from src.service.errors import ServiceError
from src.service.schemas import CandidateViewInputRecord
from src.service.settings import ServiceSettings
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


def _raw_clip_config(**overrides):
    values = {
        "mode": "raw_bbox_crop",
        "context_fraction": 0.10,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
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
        "mode": "raw_bbox_crop",
        "context_fraction": 0.1,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
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


def test_bbox_is_storage_only_and_context_is_exactly_dilated():
    image = np.zeros((32, 36, 3), dtype=np.uint8)
    mask = np.zeros((32, 36), dtype=bool)
    mask[8:24, 10:26] = True
    mask[12:20, 14:22] = False
    image[12:20, 14:22] = (250, 250, 250)
    image[7, 10] = (240, 10, 10)
    view = build_mask_views(image, mask, 7, _clip_config())

    assert view.target_bbox_xyxy == (10, 8, 26, 24)
    assert view.context_bbox_xyxy == view.target_bbox_xyxy
    assert np.all(view.target_rgb[~view.target_mask] == 0)
    assert np.all(view.context_rgb[~view.support_mask] == 0)
    assert np.array_equal(view.target_rgb[view.target_mask], image[8:24, 10:26][view.target_mask])
    assert np.array_equal(view.context_rgb[view.target_mask], image[8:24, 10:26][view.target_mask])
    assert not np.any(view.context_rgb[4:12, 4:12])


def test_exact_512_striped_rectangular_leakage_fixture_is_repeatable():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    rows, cols = np.indices((512, 512))
    image[:, :, 0] = ((rows * 5 + cols * 3) % 251 + 1).astype(np.uint8)
    image[:, :, 1] = ((rows * 7 + cols * 11) % 251 + 1).astype(np.uint8)
    image[:, :, 2] = ((rows * 13 + cols * 17) % 251 + 1).astype(np.uint8)
    mask = np.zeros((512, 512), dtype=bool)
    mask[64:448, 48:464] = True
    mask[192:320, 192:320] = False
    distractor = np.zeros((512, 512), dtype=bool)
    distractor[224:288, 224:288] = True
    image[distractor] = np.where(
        ((rows[distractor] // 4) % 2 == 0)[:, None],
        np.array((255, 8, 8), dtype=np.uint8),
        np.array((8, 255, 255), dtype=np.uint8),
    )

    config = _clip_config(context_fraction=0, min_context_pixels=0, max_context_pixels=0)
    first = build_mask_views(image, mask, 11, config)
    second = build_mask_views(image, mask, 11, config)
    x0, y0, x1, y1 = first.target_bbox_xyxy
    source_crop = image[y0:y1, x0:x1]
    distractor_crop = distractor[y0:y1, x0:x1]

    assert np.unique(image[distractor].reshape(-1, 3), axis=0).shape[0] == 2
    assert first.target_bbox_xyxy == (48, 64, 464, 448)
    assert first.context_bbox_xyxy == first.target_bbox_xyxy
    assert np.all(first.target_rgb[distractor_crop] == 0)
    assert np.all(first.context_rgb[distractor_crop] == 0)
    assert np.array_equal(first.target_rgb[first.target_mask], source_crop[first.target_mask])
    assert np.array_equal(first.context_rgb[first.target_mask], source_crop[first.target_mask])
    assert first.target_rgb.tobytes() == second.target_rgb.tobytes()
    assert first.context_rgb.tobytes() == second.context_rgb.tobytes()
    first_png = io.BytesIO()
    Image.fromarray(first.context_rgb, mode="RGB").save(first_png, format="PNG")
    second_png = io.BytesIO()
    Image.fromarray(second.context_rgb, mode="RGB").save(second_png, format="PNG")
    assert first_png.getvalue() == second_png.getvalue()
    assert (
        hashlib.sha256(first_png.getvalue()).hexdigest()
        == hashlib.sha256(second_png.getvalue()).hexdigest()
    )


def test_generated_visibility_markers_holes_components_and_radius_overrides():
    image = np.zeros((41, 47, 3), dtype=np.uint8)
    mask = np.zeros((41, 47), dtype=bool)
    mask[20, 22] = True
    image[20, 23] = (10, 20, 30)
    image[22, 25] = (40, 50, 60)
    view = build_mask_views(image, mask, 1, _clip_config(context_fraction=0.5))
    assert view.metadata["raw_radius"] == 1
    assert view.effective_radius == 1
    assert view.context_rgb[1, 2].tolist() == [3, 7, 10]
    assert not np.any(np.all(view.context_rgb == image[22, 25], axis=2))

    zero = build_mask_views(image, mask, 1, _clip_config(context_fraction=0.0))
    minimum = build_mask_views(
        image,
        mask,
        1,
        _clip_config(context_fraction=0.0, min_context_pixels=3, max_context_pixels=5),
    )
    maximum = build_mask_views(
        image,
        mask,
        1,
        _clip_config(context_fraction=0.5, min_context_pixels=0, max_context_pixels=0),
    )
    assert zero.effective_radius == 0
    assert minimum.effective_radius == 3
    assert maximum.metadata["raw_radius"] == 1 and maximum.effective_radius == 0

    ring_image = np.zeros((13, 13, 3), dtype=np.uint8)
    ring = np.zeros((13, 13), dtype=bool)
    ring[3:10, 3:10] = True
    ring[5:8, 5:8] = False
    ring_image[6, 6] = (121, 122, 123)
    before = build_mask_views(
        ring_image, ring, 2, _clip_config(min_context_pixels=1, max_context_pixels=1)
    )
    reached = build_mask_views(
        ring_image, ring, 2, _clip_config(min_context_pixels=2, max_context_pixels=2)
    )
    bx, by, _, _ = before.context_bbox_xyxy
    rx, ry, _, _ = reached.context_bbox_xyxy
    assert np.all(before.context_rgb[6 - by, 6 - bx] == 0)
    assert reached.context_rgb[6 - ry, 6 - rx].tolist() == [42, 42, 43]

    components = np.zeros((24, 36), dtype=bool)
    components[10:13, 4:7] = True
    components[10:13, 25:28] = True
    components_image = np.zeros((24, 36, 3), dtype=np.uint8)
    components_image[11, 5] = (201, 17, 91)
    components_image[11, 26] = (19, 211, 73)
    components_image[11, 16] = (251, 251, 251)
    component_view = build_mask_views(components_image, components, 3, _clip_config())
    assert component_view.context_bbox_xyxy == (4, 10, 28, 13)
    assert component_view.context_rgb[1, 1].tolist() == [201, 17, 91]
    assert component_view.context_rgb[1, 22].tolist() == [19, 211, 73]
    assert np.all(component_view.context_rgb[1, 12] == 0)


def test_border_corner_and_disconnected_source_pixels_have_no_wraparound():
    height, width = 17, 19
    rows, cols = np.indices((height, width))
    image = np.stack(
        (
            (rows * 17 + cols * 3 + 1) % 251,
            (rows * 5 + cols * 19 + 2) % 251,
            (rows * 23 + cols * 7 + 3) % 251,
        ),
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    for point in (
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (0, width // 2),
        (height - 1, width // 2),
        (height // 2, 0),
        (height // 2, width - 1),
    ):
        mask[point] = True
    view = build_mask_views(
        image, mask, 4, _clip_config(min_context_pixels=1, max_context_pixels=1)
    )
    x0, y0, x1, y1 = view.context_bbox_xyxy
    assert (x0, y0, x1, y1) == (0, 0, width, height)
    assert view.target_rgb.shape[:2] == view.target_mask.shape == view.support_mask.shape
    assert np.all(view.context_rgb[~view.support_mask] == 0)
    for row, col in zip(*np.nonzero(mask)):
        assert np.array_equal(view.target_rgb[row - y0, col - x0], image[row, col])
        assert np.array_equal(view.context_rgb[row - y0, col - x0], image[row, col])


@pytest.mark.parametrize("contour_enabled", [False, True])
def test_tiny_mask_builds_source_space_crop_before_resize_and_contour(contour_enabled):
    image = _scene((15, 15))
    mask = np.zeros((15, 15), dtype=bool)
    mask[5:8, 5:8] = True
    config = _blip_config(
        context_fraction=0,
        min_context_pixels=0,
        max_context_pixels=0,
        crop_extent_multiplier=2.0,
        contour_enabled=contour_enabled,
        contour_fraction=0.0,
        contour_min_pixels=1,
        contour_max_pixels=1,
    )
    composition = blip3_module.compose_single_blip3_view(image, mask, 5, config)
    assert composition.crop_bbox_xyxy_exclusive == (3, 3, 9, 9)
    assert composition.crop_shape_hw == (6, 6)
    assert composition.model_input_shape_hw == (256, 256)
    expected_source = image[3:9, 3:9]
    expected_mask = mask[3:9, 3:9]
    assert np.array_equal(composition.raw_mask, expected_mask)
    assert np.array_equal(composition.support_mask, expected_mask)
    assert np.array_equal(
        composition.source_composite[expected_mask], expected_source[expected_mask]
    )
    expected = np.asarray(
        Image.fromarray(composition.source_composite, mode="RGB").resize(
            (256, 256), Image.Resampling.BILINEAR
        )
    )
    assert np.array_equal(composition.rgb, expected)
    if contour_enabled:
        assert np.any(composition.contour)
        assert np.all(composition.source_composite[composition.contour] == [255, 224, 0])
    else:
        assert not np.any(composition.contour)


def test_euclidean_radius_formula_and_markers():
    image = np.zeros((25, 25, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=bool)
    mask[12, 12] = True
    image[12, 13] = (10, 20, 30)
    image[14, 14] = (40, 50, 60)
    view = build_mask_views(image, mask, 1, _clip_config(context_fraction=0.5))
    assert view.metadata["raw_radius"] == 1
    assert view.effective_radius == 1
    assert view.context_bbox_xyxy == (11, 11, 14, 14)
    assert view.context_rgb[1, 2, 0] == 3
    assert not np.any(view.context_rgb == image[14, 14])
    clamped = build_mask_views(
        image,
        mask,
        1,
        _clip_config(context_fraction=0.0, min_context_pixels=2, max_context_pixels=3),
    )
    assert clamped.effective_radius == 2


def test_radius_512_dilation_uses_bounded_local_resources():
    child = (
        "import resource\n"
        "import time\n"
        "import numpy as np\n"
        "from src.core.mask_views import _circular_dilate\n"
        "mask = np.zeros((941, 1672), dtype=bool)\n"
        "mask[470:472, 835:837] = True\n"
        "started = time.perf_counter()\n"
        "support = _circular_dilate(mask, 512)\n"
        "elapsed = time.perf_counter() - started\n"
        "assert support.dtype == bool and support.shape == mask.shape\n"
        "assert support[470, 835] and not support[0, 0]\n"
        "print(f'{elapsed:.6f} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child], check=True, capture_output=True, text=True, timeout=30
    )
    elapsed, rss_kib = (float(value) for value in completed.stdout.split())
    assert elapsed < 30.0
    assert rss_kib < 512 * 1024


@pytest.mark.parametrize(
    "mask_point",
    [(0, 0), (0, 24), (24, 0), (24, 24), (0, 12), (12, 0), (24, 12), (12, 24)],
)
def test_border_masks_are_clipped_without_wraparound(mask_point):
    image = np.zeros((25, 25, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=bool)
    mask[mask_point] = True
    view = build_mask_views(
        image, mask, 2, _clip_config(min_context_pixels=4, max_context_pixels=4)
    )
    x0, y0, x1, y1 = view.context_bbox_xyxy
    assert 0 <= x0 < x1 <= 25 and 0 <= y0 < y1 <= 25
    assert view.support_mask.shape == view.context_rgb.shape[:2]
    assert np.count_nonzero(view.target_mask) == 1


def test_results_are_immutable_and_inputs_are_not_mutated():
    image = np.arange(20 * 21 * 3, dtype=np.uint8).reshape(20, 21, 3)
    mask = np.zeros((20, 21), dtype=bool)
    mask[5:8, 6:9] = True
    image_before, mask_before = image.copy(), mask.copy()
    first = build_mask_views(image, mask, 4, _clip_config(context_fraction=0.2))
    second = build_mask_views(image, mask, 4, _clip_config(context_fraction=0.2))
    assert np.array_equal(first.context_rgb, second.context_rgb)
    assert not first.context_rgb.flags.writeable
    assert not first.support_mask.flags.writeable
    with pytest.raises(ValueError):
        first.context_rgb[0, 0, 0] = 1
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_contour_is_only_ring_and_single_image_has_no_rectangular_bridge():
    image = np.full((24, 32, 3), 200, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[8:10, 4:6] = True
    mask[7:17, 25:27] = True
    mask[7:17, 4:6] = True
    composition = blip3_module.compose_single_blip3_view(
        image,
        mask,
        3,
        _blip_config(
            context_fraction=0.0,
            min_context_pixels=2,
            max_context_pixels=2,
            contour_fraction=0.25,
            contour_min_pixels=2,
            contour_max_pixels=2,
        ),
    )
    x0, y0, _, _ = composition.crop_bbox_xyxy_exclusive
    row = 12 - y0
    midpoint = (4 + 26) // 2 - x0
    assert not composition.support_mask[row, midpoint]
    assert not composition.contour[row, midpoint]
    assert not np.any(composition.contour & composition.support_mask)
    assert np.all(composition.source_composite[composition.contour] == [255, 224, 0])


def test_resize_restores_high_contrast_target_pixels_after_interpolation():
    image = np.full((20, 20, 3), (249, 3, 241), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[7:12, 7:12] = True
    image[mask] = [1, 2, 3]
    config = _blip_config(
        context_fraction=0.0,
        min_context_pixels=2,
        max_context_pixels=2,
        contour_enabled=False,
    )
    composition = blip3_module.compose_single_blip3_view(image, mask, 1, config)
    expected = np.asarray(
        Image.fromarray(composition.source_composite, mode="RGB").resize(
            (composition.scaled_width, composition.scaled_height), Image.Resampling.BILINEAR
        )
    )
    assert np.array_equal(composition.rgb, expected)
    assert np.array_equal(composition.source_composite[composition.raw_mask], image[mask])


def test_single_image_composition_retains_only_crop_bounded_arrays():
    image = _scene((128, 160))
    mask = np.zeros((128, 160), dtype=bool)
    mask[60:68, 75:83] = True
    composition = blip3_module.compose_single_blip3_view(
        image, mask, 9, _blip_config(context_fraction=0.0, contour_enabled=True)
    )
    retained = [value for value in vars(composition).values() if isinstance(value, np.ndarray)]
    assert retained
    assert all(array.shape[:2] != image.shape[:2] for array in retained)
    assert all(
        array.nbytes <= composition.source_composite.nbytes for array in retained if array.ndim == 2
    )
    assert composition.source_composite.shape[:2] == composition.crop_shape_hw
    assert composition.source_composite.shape[:2] != image.shape[:2]


def test_capability_discloses_contour_rgb_limits_and_record_names():
    capabilities = build_capabilities(ServiceSettings())
    assert capabilities["candidate_views"]["blip3"]["fields"]["contour_rgb"] == {
        "type": "array",
        "min_items": 3,
        "max_items": 3,
        "item_type": "integer",
        "item_minimum": 0,
        "item_maximum": 255,
    }
    assert "min_items" not in capabilities["candidate_views"]["clip"]["fields"]["context_fraction"]
    base = {
        "stage": "clip",
        "source_candidate_id": 8,
        "filtered_index": 3,
        "artifact_name": "clip-candidate-view-CANDIDATE-0008.png",
        "target_bbox_xyxy": [1, 2, 3, 4],
        "context_bbox_xyxy": [0, 1, 4, 5],
        "effective_radius": 2,
        "source_dimensions": {"height": 8, "width": 8},
        "crop_dimensions": {"height": 4, "width": 4},
        "model_input_dimensions": {"height": 4, "width": 4},
    }
    assert CandidateViewInputRecord.model_validate(base).artifact_name == base["artifact_name"]
    with pytest.raises(ValueError):
        CandidateViewInputRecord.model_validate(
            {**base, "artifact_name": "clip-candidate-view-CANDIDATE-0007.png"}
        )
    blip = {
        **base,
        "stage": "blip3",
        "artifact_name": "blip3-verification-CANDIDATE-0008-QUESTION-0003.png",
        "question_id": 3,
        "raw_mask_bbox_xyxy_inclusive": [1, 2, 3, 4],
        "support_bbox_xyxy_inclusive": [0, 1, 4, 5],
        "crop_bbox_xyxy_exclusive": [0, 1, 4, 5],
        "raw_context_radius": 2,
        "effective_context_radius": 2,
        "raw_contour_width": 1,
        "effective_contour_width": 1,
        "effective_blur_sigma": 2.0,
        "source_composite_dimensions": {"height": 4, "width": 4},
    }
    assert CandidateViewInputRecord.model_validate(blip).question_id == 3
    with pytest.raises(ValueError):
        CandidateViewInputRecord.model_validate(
            {**blip, "artifact_name": "blip3-verification-CANDIDATE-0008-QUESTION-0004.png"}
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"candidate_views:\n  blip3:\n    contour_rgb: [0, 1, 256]\n",
        b"candidate_views:\n  blip3:\n    contour_rgb: [0, 1.0, 2]\n",
        b"candidate_views:\n  blip3:\n    contour_width: 1\n",
        b"candidate_views:\n  blip3:\n    crop_extent_multiplier: 0.9\n",
    ],
)
def test_blip3_strict_validation_rejects_machine_limit_violations(raw):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(b"alpha: 0.5\n" + raw, verbosity=3)
    assert excinfo.value.code in {"invalid_config", "unsupported_field"}


def test_real_clip_classify_single_receives_literal_raw_bbox_processor_view():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def item(self):
            return self.value

        def __float__(self):
            return float(self.value)

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float64)

        @property
        def T(self):
            return Tensor(self.data.T)

        def numel(self):
            return int(self.data.size)

        def norm(self, dim=-1, keepdim=False):
            return Tensor(np.linalg.norm(self.data, axis=dim, keepdims=keepdim))

        def __truediv__(self, other):
            return Tensor(self.data / other.data)

        def __getitem__(self, index):
            value = self.data[index]
            return Scalar(value) if np.ndim(value) == 0 else Tensor(value)

        def argmax(self):
            return Scalar(int(np.argmax(self.data)))

    class Torch:
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        def no_grad(self):
            return self._NoGrad()

        def is_tensor(self, value):
            return isinstance(value, Tensor)

        def tensor(self, data):
            return Tensor(data)

        def matmul(self, left, right):
            return Tensor(np.matmul(left.data, right.data))

    torch = Torch()

    class Processor:
        def __init__(self):
            self.images = []
            self.texts = []

        def __call__(self, *, images=None, text=None, return_tensors, padding=False):
            assert return_tensors == "pt"
            if text is not None:
                assert images is None
                assert padding is True
                self.texts.append(list(text))
            else:
                self.images.append(np.asarray(images).copy())
            return {}

    class Model:
        def get_text_features(self, **_inputs):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        def get_image_features(self, **_inputs):
            return torch.tensor([[3.0, 1.0]])

    processor = Processor()
    clip_filter = object.__new__(clip_module._ClipFilter)
    clip_filter._torch = torch
    clip_filter.device = "cpu"
    clip_filter.model_dtype = None
    clip_filter.processor = processor
    clip_filter.model = Model()
    clip_filter.text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clip_filter.class_idx = ["target", "distractor"]
    clip_filter.all_prompts = ["target prompt", "distractor prompt"]
    clip_filter.debug = True
    clip_filter.verbosity = 0
    clip_filter.log_print = lambda *_args, **_kwargs: None

    canonical_prompts = {
        "machine_id_with_underscore": "Natural-language value, with punctuation\nand a newline",
        "other_id": "another natural-language value",
    }
    clip_filter.class_map = clip_module._class_map_from(
        {"labels": canonical_prompts}, canonical_labels=True
    )
    clip_filter.all_prompts = [
        "Natural-language value, with punctuation\nand a newline",
        "another natural-language value",
    ]
    clip_filter._encode_text_prompts()
    assert processor.texts == [clip_filter.all_prompts]
    assert "machine_id_with_underscore" not in processor.texts[0][0]
    # Restore the independent two-label seam used by the image-boundary proof.
    clip_filter.class_map = {
        "target": ["target prompt"],
        "distractor": ["distractor prompt"],
    }
    clip_filter.class_idx = ["target", "distractor"]
    clip_filter.all_prompts = ["target prompt", "distractor prompt"]
    clip_filter.text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    image = _scene((32, 36))
    mask = np.zeros((32, 36), dtype=bool)
    mask[8:24, 10:26] = True
    mask[12:20, 14:22] = False
    image[10:12, 12:14] = (250, 250, 250)  # outside the mask, inside its tight bbox
    image[4:8, 6:10] = (3, 251, 127)  # distinctive padded context pixels
    image_before = image.copy()
    mask_before = mask.copy()
    sink = BoundedMemoryArtifactSink()
    records = []
    config = _raw_clip_config(
        context_fraction=0.1875,
        min_context_pixels=2,
        max_context_pixels=3,
    )
    candidate = {"segmentation": mask, "_source_index": 7, "_filtered_index": 2}
    clip_filter.filter_masks(
        [candidate],
        image,
        None,
        "frame",
        artifact_sink=sink,
        safe_artifact_names=True,
        candidate_view_config=config,
        candidate_view_inputs=records,
    )
    expected_view = build_raw_clip_crop(
        image,
        mask,
        8,
        config,
        filtered_index=2,
        debug=True,
    )
    expected = expected_view.rgb
    x0, y0, x1, y1 = expected_view.crop_bbox_xyxy_exclusive
    assert expected_view.raw_context_radius == int(math.floor(0.1875 * 16 + 0.5)) == 3
    assert expected_view.effective_context_radius == 3
    assert (x0, y0, x1, y1) == (7, 5, 29, 27)
    assert np.array_equal(expected, image[y0:y1, x0:x1])
    assert len(processor.images) == 1
    assert np.array_equal(processor.images[0], expected)
    assert processor.images[0].shape == (y1 - y0, x1 - x0, 3)
    assert np.array_equal(sink.artifacts()[0].array, processor.images[0])
    assert sink.names() == ("clip-candidate-view-CANDIDATE-0008.png",)
    png = io.BytesIO()
    Image.fromarray(sink.artifacts()[0].array, mode="RGB").save(png, format="PNG")
    decoded = np.asarray(Image.open(io.BytesIO(png.getvalue())))
    assert np.array_equal(decoded, processor.images[0])
    assert records[0]["source_candidate_id"] == 8
    assert records[0]["filtered_index"] == 2
    assert records[0]["crop_bbox_xyxy_exclusive"] == [x0, y0, x1, y1]
    assert records[0]["raw_context_radius"] == 3
    assert records[0]["effective_context_radius"] == 3
    assert records[0]["artifact_name"] == "clip-candidate-view-CANDIDATE-0008.png"
    assert list(records[0]["config"]) == [
        "mode",
        "context_fraction",
        "min_context_pixels",
        "max_context_pixels",
    ]
    assert records[0]["config"]["mode"] == "raw_bbox_crop"
    assert records[0]["config"]["context_fraction"] == 0.1875
    assert records[0]["config"]["min_context_pixels"] == 2
    assert records[0]["config"]["max_context_pixels"] == 3
    assert records[0]["target_bbox_xyxy"] == [10, 8, 25, 23]
    assert records[0]["model_input_dimensions"] == {"height": 22, "width": 22}
    # ``classify_single_scores`` returns every configured label in order.
    assert list(candidate["clip_scores"]) == ["target", "distractor"]
    assert candidate["clip_scores"]["target"] == pytest.approx(3.0 / np.sqrt(10.0))
    assert candidate["clip_scores"]["distractor"] == pytest.approx(1.0 / np.sqrt(10.0))
    assert candidate["clip_label"] == "target"
    assert candidate["clip_prompt"] == "target prompt"
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_resident_clip_debug_configuration_is_a_b_a_request_local():
    class TextEmbeds:
        def numel(self):
            return 1

    holder = object.__new__(clip_module._ClipFilter)
    holder.text_embeds = TextEmbeds()
    holder.debug = True
    holder.verbosity = 0
    holder.log_print = lambda *_args, **_kwargs: None
    updates = []
    classifications = []
    holder.update_labels = lambda config: updates.append(dict(config))

    def classify(patch, _index):
        classifications.append(patch.copy())
        return "target", 0.5, "target prompt"

    holder.classify_single = classify
    state = {"clip_filter": holder}
    image = np.full((18, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((18, 20), dtype=bool)
    mask[7:11, 8:12] = True

    def run_once(context_fraction):
        sink = BoundedMemoryArtifactSink()
        records = []
        clip_module.run(
            state,
            {
                "config": {"debug": True, "labels": {"target": "target"}},
                "masks": [{"segmentation": mask, "_source_index": 7}],
                "out_dir": None,
                "fname_stem": "request",
                "artifact_sink": sink,
                "safe_artifact_names": True,
                "candidate_view_config": _clip_config(context_fraction=context_fraction),
                "candidate_view_inputs": records,
            },
            image,
            verbosity=3,
        )
        return sink, records

    first_sink, first_records = run_once(0.1)
    second_sink, second_records = run_once(0.3)
    third_sink, third_records = run_once(0.1)
    assert first_sink.names() == third_sink.names() == ("clip-candidate-view-CANDIDATE-0008.png",)
    assert second_sink.names() == first_sink.names()
    assert first_records[0]["artifact_name"] == third_records[0]["artifact_name"]
    assert len(updates) == len(classifications) == 3
    assert not np.array_equal(classifications[0], classifications[1])
    assert np.array_equal(classifications[0], classifications[2])
    assert state["clip_filter"] is holder


def test_centered_crop_uses_inclusive_centers_and_independent_clamps():
    cases = [
        ((40, 50), (20, 15, 23, 17), 1.0),
        ((40, 50), (20, 15, 22, 18), 1.5),
        ((40, 50), (0, 0, 3, 2), 2.0),
        ((40, 50), (46, 35, 49, 38), 2.0),
    ]
    for (height, width), (raw_x0, raw_y0, raw_x1, raw_y1), multiplier in cases:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[raw_y0 : raw_y1 + 1, raw_x0 : raw_x1 + 1] = True
        config = _blip_config(
            crop_extent_multiplier=multiplier,
            context_fraction=0.0,
            min_context_pixels=0,
            max_context_pixels=0,
            contour_enabled=False,
        )
        composition = blip3_module.compose_single_blip3_view(image, mask, 1, config)
        raw_width = raw_x1 - raw_x0 + 1
        raw_height = raw_y1 - raw_y0 + 1
        nominal_width = math.ceil(multiplier * raw_width)
        nominal_height = math.ceil(multiplier * raw_height)

        # Independent expected arithmetic: inclusive pixel-center, odd/even
        # nominal sizes, and endpoint clamps without shifting the other end.
        expected_x0 = math.floor((raw_x0 + raw_x1) / 2 - (nominal_width - 1) / 2)
        expected_y0 = math.floor((raw_y0 + raw_y1) / 2 - (nominal_height - 1) / 2)
        expected_box = (
            max(0, min(width, expected_x0)),
            max(0, min(height, expected_y0)),
            max(0, min(width, expected_x0 + nominal_width)),
            max(0, min(height, expected_y0 + nominal_height)),
        )
        assert composition.crop_bbox_xyxy_exclusive == expected_box
        assert composition.crop_shape_hw == (
            expected_box[3] - expected_box[1],
            expected_box[2] - expected_box[0],
        )

    image = np.zeros((30, 30, 3), dtype=np.uint8)
    asymmetric = np.zeros((30, 30), dtype=bool)
    asymmetric[10:13, 10:14] = True
    config = _blip_config(
        crop_extent_multiplier=1.5,
        context_fraction=0.0,
        min_context_pixels=0,
        max_context_pixels=0,
        contour_fraction=0.0,
        contour_min_pixels=1,
        contour_max_pixels=1,
    )
    composition = blip3_module.compose_single_blip3_view(image, asymmetric, 1, config)
    assert composition.crop_bbox_xyxy_exclusive == (9, 9, 15, 14)
    assert composition.contour[11 - 9, 14 - 9]
    old_shifted_box = (8, 9, 14, 14)
    assert old_shifted_box[2] == 14 < composition.crop_bbox_xyxy_exclusive[2]
