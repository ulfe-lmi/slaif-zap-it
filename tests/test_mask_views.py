"""Generated-array tests for mask-isolated CLIP/BLIP3 candidate views."""

from __future__ import annotations

import io
import hashlib
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


def _png_sha256(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_exact_512_striped_rectangular_leakage_fixture_is_repeatable():
    """The required high-contrast fixture proves bbox storage is not visibility."""
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

    config = _config(context_fraction=0.0, min_context_pixels=0, max_context_pixels=0)

    def once():
        view = build_mask_views(image, mask, 11, config)
        return {
            "view": view,
            "target_png_sha": _png_sha256(view.target_rgb),
            "context_png_sha": _png_sha256(view.context_rgb),
        }

    first = once()
    second = once()
    view = first["view"]
    x0, y0, x1, y1 = view.target_bbox_xyxy
    source_crop = image[y0:y1, x0:x1]
    distractor_crop = distractor[y0:y1, x0:x1]

    assert image[distractor].min() > 0
    assert np.unique(image[distractor].reshape(-1, 3), axis=0).shape[0] == 2
    assert view.target_bbox_xyxy == (48, 64, 464, 448)
    assert view.context_bbox_xyxy == view.target_bbox_xyxy
    assert view.metadata["raw_radius"] == 0
    assert view.metadata["effective_radius"] == 0
    assert view.effective_radius == int(np.ceil(0.0 * max(416, 384)))
    assert np.all(view.target_rgb[distractor_crop] == 0)
    assert np.all(view.context_rgb[distractor_crop] == 0)
    assert np.all(view.target_rgb[~view.target_mask] == 0)
    assert np.all(view.context_rgb[~view.support_mask] == 0)
    assert np.array_equal(view.target_rgb[view.target_mask], source_crop[view.target_mask])
    assert np.array_equal(view.context_rgb[view.target_mask], source_crop[view.target_mask])

    for key in ("target_rgb", "context_rgb", "target_mask", "support_mask"):
        assert np.array_equal(getattr(first["view"], key), getattr(second["view"], key))
    assert first["view"].target_bbox_xyxy == second["view"].target_bbox_xyxy
    assert first["view"].context_bbox_xyxy == second["view"].context_bbox_xyxy
    assert first["view"].metadata_dict() == second["view"].metadata_dict()
    assert first["target_png_sha"] == second["target_png_sha"]
    assert first["context_png_sha"] == second["context_png_sha"]


def test_generated_visibility_markers_holes_components_and_radius_overrides():
    image = np.zeros((41, 47, 3), dtype=np.uint8)
    mask = np.zeros((41, 47), dtype=bool)
    mask[20, 22] = True
    image[20, 23] = (10, 20, 30)  # exactly one Euclidean pixel away
    image[22, 25] = (40, 50, 60)  # outside radius one
    view = build_mask_views(image, mask, 1, _config(context_fraction=0.5))
    assert view.metadata["raw_radius"] == 1  # ceil(0.5 * max(1, 1))
    assert view.effective_radius == 1
    assert view.context_rgb[1, 2].tolist() == [3, 7, 10]
    assert not np.any(np.all(view.target_rgb == image[20, 23], axis=2))
    assert not np.any(np.all(view.context_rgb == image[22, 25], axis=2))
    assert np.all(view.context_rgb[view.target_mask] == image[20, 22])

    zero = build_mask_views(image, mask, 1, _config(context_fraction=0.0))
    assert zero.metadata["raw_radius"] == 0
    assert zero.effective_radius == 0
    minimum = build_mask_views(
        image,
        mask,
        1,
        _config(context_fraction=0.0, min_context_pixels=3, max_context_pixels=5),
    )
    assert minimum.metadata["raw_radius"] == 0
    assert minimum.effective_radius == 3
    maximum = build_mask_views(
        image,
        mask,
        1,
        _config(context_fraction=0.5, min_context_pixels=0, max_context_pixels=0),
    )
    assert maximum.metadata["raw_radius"] == 1
    assert maximum.effective_radius == 0

    ring_image = np.zeros((13, 13, 3), dtype=np.uint8)
    ring = np.zeros((13, 13), dtype=bool)
    ring[3:10, 3:10] = True
    ring[5:8, 5:8] = False
    ring_image[6, 6] = (121, 122, 123)
    before = build_mask_views(
        ring_image, ring, 2, _config(min_context_pixels=1, max_context_pixels=1)
    )
    reached = build_mask_views(
        ring_image, ring, 2, _config(min_context_pixels=2, max_context_pixels=2)
    )
    before_x0, before_y0, _, _ = before.context_bbox_xyxy
    reached_x0, reached_y0, _, _ = reached.context_bbox_xyxy
    assert np.all(before.context_rgb[6 - before_y0, 6 - before_x0] == 0)
    assert reached.context_rgb[6 - reached_y0, 6 - reached_x0].tolist() == [42, 42, 43]

    components_image = np.zeros((24, 36, 3), dtype=np.uint8)
    components = np.zeros((24, 36), dtype=bool)
    components[10:13, 4:7] = True
    components[10:13, 25:28] = True
    components_image[11, 5] = (201, 17, 91)
    components_image[11, 26] = (19, 211, 73)
    components_image[11, 16] = (251, 251, 251)
    component_view = build_mask_views(
        components_image, components, 3, _config(context_fraction=0.0)
    )
    assert component_view.context_bbox_xyxy == (4, 10, 28, 13)
    assert component_view.context_rgb[1, 1].tolist() == [201, 17, 91]
    assert component_view.context_rgb[1, 22].tolist() == [19, 211, 73]
    assert np.all(component_view.context_rgb[1, 12] == 0)
    assert np.all(component_view.target_rgb[~component_view.target_mask] == 0)


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
    for row, col in (
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (0, width // 2),
        (height - 1, width // 2),
        (height // 2, 0),
        (height // 2, width - 1),
    ):
        mask[row, col] = True
    view = build_mask_views(image, mask, 4, _config(min_context_pixels=1, max_context_pixels=1))

    x0, y0, x1, y1 = view.context_bbox_xyxy
    source_crop = image[y0:y1, x0:x1]
    expected_target = np.zeros_like(source_crop)
    expected_target[view.target_mask] = source_crop[view.target_mask]
    expected_context = np.zeros_like(source_crop)
    expected_context[view.target_mask] = source_crop[view.target_mask]
    expected_context[view.support_mask & ~view.target_mask] = (
        source_crop[view.support_mask & ~view.target_mask].astype(np.float32) * 0.35
    ).astype(np.uint8)
    assert np.array_equal(view.target_rgb, expected_target)
    assert np.array_equal(view.context_rgb, expected_context)
    assert (x0, y0, x1, y1) == (0, 0, width, height)
    assert view.target_rgb.shape[:2] == view.target_mask.shape == view.support_mask.shape
    assert np.all(view.context_rgb[~view.support_mask] == 0)
    for row, col in zip(*np.nonzero(mask)):
        assert np.array_equal(view.target_rgb[row - y0, col - x0], image[row, col])
        assert np.array_equal(view.context_rgb[row - y0, col - x0], image[row, col])


@pytest.mark.parametrize("contour_width", [0, 2])
def test_tiny_mask_builds_source_space_crop_before_resize_and_contour(contour_width):
    image = np.zeros((9, 11, 3), dtype=np.uint8)
    image[3, 4] = (240, 17, 91)
    prohibited = (253, 251, 249)
    image[0, 0] = prohibited
    mask = np.zeros((9, 11), dtype=bool)
    mask[3, 4] = True
    config = CandidateViewConfig.from_mapping(
        {
            **_config(min_context_pixels=2, max_context_pixels=2).__dict__,
            "contour_width": contour_width,
        },
        stage="blip3",
    )
    view = build_mask_views(image, mask, 5, config, stage="blip3")
    pair = compose_candidate_view_pair(view)
    assert view.target_bbox_xyxy == (4, 3, 5, 4)
    assert view.context_bbox_xyxy == (2, 1, 7, 6)
    assert view.context_rgb.shape[:2] == (5, 5)
    assert pair.crop_box_xyxy == view.context_bbox_xyxy
    assert pair.crop_shape_hw == (5, 5)
    assert pair.scaled_shape_hw == (256, 256)
    assert pair.scale == 256 / 5

    from modules.verifier.blip3 import _nearest_indices, _square_dilation

    row_indices = _nearest_indices(5, 256)
    col_indices = _nearest_indices(5, 256)
    indexer = np.ix_(row_indices, col_indices)
    target_mask = view.target_mask[indexer]
    support_mask = view.support_mask[indexer]
    expected_target = np.asarray(
        Image.fromarray(view.target_rgb).resize((256, 256), Image.Resampling.BILINEAR)
    ).copy()
    expected_context = np.asarray(
        Image.fromarray(view.context_rgb).resize((256, 256), Image.Resampling.BILINEAR)
    ).copy()
    expected_target[~target_mask] = 0
    expected_context[~support_mask] = 0
    expected_context[target_mask] = expected_target[target_mask]
    expected_contour = _square_dilation(target_mask, contour_width) & ~target_mask & support_mask
    expected_context[expected_contour] = np.array((255, 224, 0), dtype=np.uint8)
    assert np.array_equal(pair.paired[:, :256], expected_target)
    assert np.array_equal(pair.scaled_mask, target_mask)
    assert np.array_equal(pair.support_mask, support_mask)
    assert np.array_equal(pair.paired[:, 260:], expected_context)
    assert not np.any(np.all(pair.paired == prohibited, axis=2))

    assert np.array_equal(pair.contour, expected_contour)
    assert not np.any(pair.contour & pair.scaled_mask)
    assert not np.any(pair.contour & ~pair.support_mask)
    if contour_width:
        right = pair.paired[:, 260:]
        assert np.all(right[pair.contour] == np.array((255, 224, 0), dtype=np.uint8))
    else:
        assert not np.any(pair.contour)
    repeated = compose_candidate_view_pair(build_mask_views(image, mask, 5, config, stage="blip3"))
    assert np.array_equal(pair.paired, repeated.paired)
    assert pair.paired.tobytes() == repeated.paired.tobytes()


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
    expected_view = build_mask_views(
        image,
        mask,
        8,
        CandidateViewConfig.from_mapping(None, stage="blip3"),
        stage="blip3",
    )
    expected_pair = compose_candidate_view_pair(expected_view)
    assert np.array_equal(qa.images[0], expected_pair.paired)
    buffer = io.BytesIO()
    Image.fromarray(sink.artifacts()[0].array).save(buffer, format="PNG")
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(buffer.getvalue()))), expected_pair.paired
    )
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


def test_real_clip_classify_single_receives_literal_processor_context_view():
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

        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            self.images.append(np.asarray(images).copy())
            return {}

    class Model:
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

    image = np.zeros((32, 36, 3), dtype=np.uint8)
    mask = _ring()
    image[12:20, 14:22] = (250, 250, 250)
    sink = BoundedMemoryArtifactSink()
    records = []
    clip_filter.filter_masks(
        [{"segmentation": mask, "_source_index": 7, "_filtered_index": 2}],
        image,
        None,
        "frame",
        artifact_sink=sink,
        safe_artifact_names=True,
        candidate_view_config=_config(context_fraction=0.0),
        candidate_view_inputs=records,
    )
    expected = build_mask_views(image, mask, 8, _config(context_fraction=0.0))
    assert len(processor.images) == 1
    assert np.array_equal(processor.images[0], expected.context_rgb)
    assert np.all(processor.images[0][~expected.support_mask] == 0)
    assert not np.any(np.all(processor.images[0] == (250, 250, 250), axis=2))
    assert sink.names() == ("clip-candidate-view-CANDIDATE-0008.png",)
    buffer = io.BytesIO()
    Image.fromarray(sink.artifacts()[0].array).save(buffer, format="PNG")
    decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())))
    assert np.array_equal(decoded, processor.images[0])
    assert records[0]["source_candidate_id"] == 8
    assert records[0]["filtered_index"] == 2
    assert clip_filter.classify_single(expected.context_rgb, 0) == (
        "target",
        pytest.approx(3.0 / np.sqrt(10.0)),
        "target prompt",
    )


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
