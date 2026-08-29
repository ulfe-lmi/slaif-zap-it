"""Generated-array tests for mask-isolated CLIP/BLIP3 candidate views."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from modules.verifier.blip3 import compose_candidate_view_pair
from modules.verifier import blip3 as blip3_module
from modules.classifier import clip as clip_module
from src.core import BoundedMemoryArtifactSink
from src.core import CandidateViewConfig, build_mask_views
from src.service.errors import ServiceError
from src.service.yaml_input import parse_hostile_config


def _config(**overrides):
    values = {
        "mode": "mask_dilated",
        "context_fraction": 0.0,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
    }
    values.update(overrides)
    return CandidateViewConfig.from_mapping(values)


def _ring(shape=(32, 36)):
    mask = np.zeros(shape, dtype=bool)
    mask[8:24, 10:26] = True
    mask[12:20, 14:22] = False
    return mask


def test_bbox_is_storage_only_and_context_is_exactly_dilated():
    image = np.zeros((32, 36, 3), dtype=np.uint8)
    mask = _ring()
    image[12:20, 14:22] = (250, 250, 250)  # distractor in the bbox hole
    image[7, 10] = (240, 10, 10)  # outside D at radius zero
    view = build_mask_views(image, mask, 7, _config())

    assert view.target_bbox_xyxy == (10, 8, 26, 24)
    assert view.context_bbox_xyxy == view.target_bbox_xyxy
    assert np.all(view.target_rgb[~view.target_mask] == 0)
    assert np.all(view.context_rgb[~view.support_mask] == 0)
    assert np.all(view.target_rgb[view.target_mask] == image[8:24, 10:26][view.target_mask])
    assert np.all(view.context_rgb[view.target_mask] == image[8:24, 10:26][view.target_mask])
    assert not np.any(view.context_rgb[12 - 8 : 20 - 8, 14 - 10 : 22 - 10])


def test_euclidean_radius_formula_and_markers():
    image = np.zeros((25, 25, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=bool)
    mask[12, 12] = True
    image[12, 13] = (10, 20, 30)  # distance 1, inside radius 1
    image[14, 14] = (40, 50, 60)  # distance sqrt(8), outside radius 1
    view = build_mask_views(image, mask, 1, _config(context_fraction=0.5))
    assert view.metadata["raw_radius"] == 1
    assert view.effective_radius == 1
    assert view.context_bbox_xyxy == (11, 11, 14, 14)
    assert view.context_rgb[1, 2, 0] == 3  # floor(10 * 0.35)
    assert image[14, 14, 0] != 0
    assert not np.any(view.context_rgb == image[14, 14])

    clamped = build_mask_views(
        image,
        mask,
        1,
        _config(context_fraction=0.0, min_context_pixels=2, max_context_pixels=3),
    )
    assert clamped.effective_radius == 2


@pytest.mark.parametrize(
    "mask_point",
    [(0, 0), (0, 24), (24, 0), (24, 24), (0, 12), (12, 0), (24, 12), (12, 24)],
)
def test_border_masks_are_clipped_without_wraparound(mask_point):
    image = np.zeros((25, 25, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=bool)
    mask[mask_point] = True
    view = build_mask_views(image, mask, 2, _config(min_context_pixels=4, max_context_pixels=4))
    x0, y0, x1, y1 = view.context_bbox_xyxy
    assert 0 <= x0 < x1 <= 25 and 0 <= y0 < y1 <= 25
    assert view.support_mask.shape == view.context_rgb.shape[:2]
    assert np.count_nonzero(view.target_mask) == 1


def test_results_are_immutable_and_inputs_are_not_mutated():
    image = np.arange(20 * 21 * 3, dtype=np.uint8).reshape(20, 21, 3)
    mask = np.zeros((20, 21), dtype=bool)
    mask[5:8, 6:9] = True
    image_before, mask_before = image.copy(), mask.copy()
    first = build_mask_views(image, mask, 4, _config(context_fraction=0.2))
    second = build_mask_views(image, mask, 4, _config(context_fraction=0.2))
    assert np.array_equal(first.context_rgb, second.context_rgb)
    assert np.array_equal(first.target_mask, second.target_mask)
    assert not first.context_rgb.flags.writeable
    assert not first.support_mask.flags.writeable
    with pytest.raises(ValueError):
        first.context_rgb[0, 0, 0] = 1
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_contour_is_only_ring_and_blip_pair_has_no_rectangular_bridge():
    image = np.full((24, 32, 3), 200, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[8:10, 4:6] = True
    mask[8:10, 25:27] = True
    view = build_mask_views(
        image,
        mask,
        3,
        CandidateViewConfig.from_mapping(
            {
                **_config(
                    context_fraction=0.0,
                    min_context_pixels=4,
                    max_context_pixels=4,
                ).__dict__,
                "contour_width": 2,
            },
            stage="blip3",
        ),
        stage="blip3",
    )
    pair = compose_candidate_view_pair(view)
    left = pair.paired[:, : pair.scaled_width]
    right = pair.paired[:, pair.scaled_width + pair.divider_width :]
    assert np.all(left[~pair.scaled_mask] == 0)
    assert np.all(right[~pair.support_mask] == 0)
    assert np.all(pair.paired[:, pair.scaled_width : pair.scaled_width + 4] == 0)
    assert not np.any(pair.support_mask[:, pair.scaled_width // 2 - 2 : pair.scaled_width // 2 + 2])


def test_service_candidate_view_validation_is_strict_and_effective():
    valid = parse_hostile_config(
        b"alpha: 0.5\ncandidate_views:\n  clip:\n    context_fraction: 0.5\n  blip3:\n    contour_width: 0\n",
        verbosity=3,
    )
    assert valid.effective_mapping["candidate_views"]["clip"]["context_fraction"] == 0.5
    assert valid.effective_mapping["candidate_views"]["blip3"]["contour_width"] == 0
    for raw, code in (
        (b"candidate_views: null\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_fraction: true\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    outside_fill: neutral\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    contour_width: 1\n", "unsupported_field"),
        (
            b"candidate_views:\n  clip:\n    min_context_pixels: 9\n    max_context_pixels: 8\n",
            "invalid_config",
        ),
    ):
        with pytest.raises(ServiceError) as excinfo:
            parse_hostile_config(b"alpha: 0.5\n" + raw, verbosity=3)
        assert excinfo.value.code == code


def test_pair_png_is_lossless_for_exact_qa_array():
    image = np.arange(16 * 17 * 3, dtype=np.uint8).reshape(16, 17, 3)
    mask = np.zeros((16, 17), dtype=bool)
    mask[5:10, 6:11] = True
    view = build_mask_views(image, mask, 8, _config(context_fraction=0.1), stage="clip")
    pair = compose_candidate_view_pair(
        build_mask_views(
            image,
            mask,
            8,
            CandidateViewConfig.from_mapping(
                {**_config(context_fraction=0.0).__dict__, "contour_width": 2}, stage="blip3"
            ),
            stage="blip3",
        )
    )
    buffer = io.BytesIO()
    Image.fromarray(pair.paired).save(buffer, format="PNG")
    decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())))
    assert np.array_equal(decoded, pair.paired)
    assert view.source_candidate_id == 8


def test_blip_debug_uses_one_based_source_and_question_ids():
    class QA:
        device = "cpu"

        def __init__(self):
            self.images = []

        def answer(self, image, _query, max_new_tokens):
            assert max_new_tokens == 32
            self.images.append(np.asarray(image).copy())
            return "Yes"

    image = np.full((20, 24, 3), 90, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=bool)
    mask[5:10, 7:12] = True
    record_list = []
    sink = BoundedMemoryArtifactSink()
    qa = QA()
    filt = blip3_module._Blip3Filter.from_qa(
        qa,
        {"hostile/rule": {"question": "is this a target?", "debug": True}},
        max_questions=32,
        max_new_tokens=32,
    )
    filt.filter_masks(
        [
            {
                "segmentation": mask,
                "_source_index": 7,
                "_filtered_index": 3,
                "clip_label": "hostile/rule",
                "clip_score": 0.1,
            }
        ],
        image,
        None,
        "client-frame",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_inputs=record_list,
    )
    assert sink.names() == ("blip3-verification-0008-0001.png",)
    assert np.array_equal(sink.artifacts()[0].array, qa.images[0])
    assert record_list[0]["source_candidate_id"] == 8
    assert record_list[0]["filtered_index"] == 3
    assert record_list[0]["question_id"] == 1


def test_clip_debug_uses_exact_builder_view_and_fixed_source_name():
    class TextEmbeds:
        def numel(self):
            return 1

    clip_filter = object.__new__(clip_module._ClipFilter)
    clip_filter.text_embeds = TextEmbeds()
    clip_filter.debug = True
    clip_filter.verbosity = 0
    clip_filter.log_print = lambda *_args, **_kwargs: None
    captured = []

    def classify(patch, _index):
        captured.append(patch.copy())
        return "target", 0.5, "target prompt"

    clip_filter.classify_single = classify
    image = np.full((18, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((18, 20), dtype=bool)
    mask[7:11, 8:12] = True
    sink = BoundedMemoryArtifactSink()
    records = []
    clip_filter.filter_masks(
        [{"segmentation": mask, "_source_index": 7, "_filtered_index": 2}],
        image,
        None,
        "client-frame",
        artifact_sink=sink,
        safe_artifact_names=True,
        candidate_view_config=_config(context_fraction=0.0),
        candidate_view_inputs=records,
    )
    expected = build_mask_views(image, mask, 8, _config(context_fraction=0.0)).context_rgb
    assert sink.names() == ("clip-candidate-view-0008.png",)
    assert np.array_equal(captured[0], expected)
    assert np.array_equal(sink.artifacts()[0].array, captured[0])
    assert records[0]["filtered_index"] == 2
