"""Generated-array tests for mask-isolated CLIP/BLIP3 candidate views."""

from __future__ import annotations

import io
import inspect
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from modules.verifier.blip3 import compose_candidate_view_pair
from modules.verifier import blip3 as blip3_module
from modules.classifier import clip as clip_module
from src.core import BoundedMemoryArtifactSink
from src.core import CandidateViewConfig, build_mask_views
from src.core.mask_views import CANDIDATE_VIEW_DEFAULTS
from src.service.errors import ServiceError
from src.service.schemas import CandidateViewInputRecord
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


def test_circular_dilation_matches_independent_bruteforce_oracle():
    from src.core.mask_views import _circular_dilate

    for shape, radius in [((1, 7), 3), ((7, 1), 3), ((9, 11), 0), ((13, 12), 5)]:
        for seed in range(8):
            rng = np.random.default_rng(seed)
            mask = rng.random(shape) < 0.22
            if not np.any(mask):
                mask[shape[0] // 2, shape[1] // 2] = True
            expected = np.zeros(shape, dtype=bool)
            for row, col in zip(*np.nonzero(mask)):
                y0 = max(0, row - radius)
                y1 = min(shape[0], row + radius + 1)
                x0 = max(0, col - radius)
                x1 = min(shape[1], col + radius + 1)
                yy, xx = np.indices((y1 - y0, x1 - x0))
                expected[y0:y1, x0:x1] |= (yy + y0 - row) ** 2 + (xx + x0 - col) ** 2 <= radius**2
            assert np.array_equal(_circular_dilate(mask, radius), expected)

    assert "horizontal_cache" not in inspect.getsource(_circular_dilate)


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
        [sys.executable, "-c", child],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
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


def test_resize_restores_high_contrast_target_pixels_after_interpolation():
    image = np.full((9, 11, 3), (249, 3, 241), dtype=np.uint8)
    image[2, 3] = (1, 2, 3)
    image[3, 4] = (4, 5, 6)
    image[4, 5] = (7, 8, 9)
    mask = np.zeros((9, 11), dtype=bool)
    mask[2, 3] = True
    mask[3, 4] = True
    mask[4, 5] = True
    view = build_mask_views(
        image,
        mask,
        1,
        CandidateViewConfig.from_mapping(
            {
                **_config(
                    context_fraction=0.5,
                    min_context_pixels=2,
                    max_context_pixels=2,
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
    assert np.array_equal(right[pair.scaled_mask], left[pair.scaled_mask])
    assert np.all(right[~pair.support_mask] == 0)
    assert not np.any(pair.contour & pair.scaled_mask)
    assert not np.any(pair.contour & ~pair.support_mask)


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


def test_candidate_view_defaults_and_inclusive_endpoints_are_effective():
    omitted = parse_hostile_config(b"alpha: 0.5\n", verbosity=3)
    clip_defaults = dict(CANDIDATE_VIEW_DEFAULTS["clip"])
    assert omitted.effective_mapping["candidate_views"] == {
        "clip": clip_defaults,
        "blip3": {
            **clip_defaults,
            "contour_width": 2,
        },
    }
    endpoints = parse_hostile_config(
        b"""alpha: 0.5
candidate_views:
  clip:
    context_fraction: 0.5
    min_context_pixels: 256
    max_context_pixels: 512
    context_intensity: 1
  blip3:
    context_fraction: 0
    min_context_pixels: 0
    max_context_pixels: 0
    context_intensity: 0
    contour_width: 16
""",
        verbosity=3,
    )
    clip = endpoints.effective_mapping["candidate_views"]["clip"]
    blip3 = endpoints.effective_mapping["candidate_views"]["blip3"]
    assert (clip["context_fraction"], clip["min_context_pixels"], clip["max_context_pixels"]) == (
        0.5,
        256,
        512,
    )
    assert (blip3["context_fraction"], blip3["max_context_pixels"], blip3["contour_width"]) == (
        0.0,
        0,
        16,
    )


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"candidate_views: []\n", "invalid_config"),
        (b"candidate_views:\n  clip: null\n", "invalid_config"),
        (b"candidate_views:\n  blip3: null\n", "invalid_config"),
        (b"candidate_views:\n  sam2: {}\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    unknown: 1\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    mode: rectangle\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    outside_fill: neutral\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    contour_width: 1\n", "unsupported_field"),
        (b"candidate_views:\n  clip:\n    context_fraction: true\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    min_context_pixels: false\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_fraction: .nan\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_intensity: .inf\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_fraction: -0.01\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_fraction: 0.51\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    min_context_pixels: -1\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    min_context_pixels: 257\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    max_context_pixels: -1\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    max_context_pixels: 513\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_intensity: -0.01\n", "invalid_config"),
        (b"candidate_views:\n  clip:\n    context_intensity: 1.01\n", "invalid_config"),
        (b"candidate_views:\n  blip3:\n    contour_width: -1\n", "invalid_config"),
        (b"candidate_views:\n  blip3:\n    contour_width: 17\n", "invalid_config"),
        (
            b"candidate_views:\n  clip:\n    min_context_pixels: 9\n    max_context_pixels: 8\n",
            "invalid_config",
        ),
        (b"clip:\n  padding: 1\n", "unsupported_field"),
    ],
)
def test_candidate_view_validation_rejects_all_unsupported_and_out_of_range_values(raw, code):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(b"alpha: 0.5\n" + raw, verbosity=3)
    assert excinfo.value.code == code


def test_candidate_view_input_names_are_typed_and_match_ids():
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
    for artifact_name in (
        "clip-candidate-view-CANDIDATE-0008",
        "clip-candidate-view-CANDIDATE-0007.png",
        "clip-candidate-view-CANDIDATE-0008.jpg",
        "../clip-candidate-view-CANDIDATE-0008.png",
    ):
        with pytest.raises(ValueError):
            CandidateViewInputRecord.model_validate({**base, "artifact_name": artifact_name})

    blip = {
        **base,
        "stage": "blip3",
        "artifact_name": "blip3-verification-CANDIDATE-0008-QUESTION-0003.png",
        "question_id": 3,
    }
    assert CandidateViewInputRecord.model_validate(blip).question_id == 3
    with pytest.raises(ValueError):
        CandidateViewInputRecord.model_validate(
            {**blip, "artifact_name": "blip3-verification-CANDIDATE-0008-QUESTION-0004.png"}
        )


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
    assert sink.names() == ("blip3-verification-CANDIDATE-0008-QUESTION-0001.png",)
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
    assert sink.names() == ("clip-candidate-view-CANDIDATE-0008.png",)
    assert np.array_equal(captured[0], expected)
    assert np.array_equal(sink.artifacts()[0].array, captured[0])
    assert records[0]["filtered_index"] == 2


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

    def run_once(debug):
        sink = BoundedMemoryArtifactSink()
        records = []
        clip_module.run(
            state,
            {
                "config": {
                    "debug": debug,
                    "labels": {"target": "target"},
                },
                "masks": [{"segmentation": mask, "_source_index": 7}],
                "out_dir": None,
                "fname_stem": "request",
                "artifact_sink": sink,
                "safe_artifact_names": True,
                "candidate_view_inputs": records,
            },
            image,
            verbosity=3,
        )
        return sink, records

    first_sink, first_records = run_once(True)
    second_sink, second_records = run_once(False)
    third_sink, third_records = run_once(True)
    assert first_sink.names() == third_sink.names() == ("clip-candidate-view-CANDIDATE-0008.png",)
    assert second_sink.names() == ()
    assert first_records[0]["artifact_name"] == third_records[0]["artifact_name"]
    assert second_records == []
    assert len(updates) == len(classifications) == 3
    assert state["clip_filter"] is holder
