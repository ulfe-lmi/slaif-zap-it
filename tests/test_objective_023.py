"""Deterministic CPU coverage for the centroid-radial BLIP3 fallback."""

from __future__ import annotations

import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFilter

from modules.verifier.blip3 import _Blip3Filter, compose_single_blip3_view
import src.core.radial_geometry as radial_geometry
from src.core import (
    CandidateViewConfig,
    build_centroid_radial_geometry,
)
from src.service.capabilities import build_capabilities
from src.service.errors import ServiceError
from src.service.schemas import Blip3CandidateViewRecord, CandidateViewInputRecord
from src.service.settings import ServiceSettings
from src.service.yaml_input import parse_hostile_config


def _config(**overrides) -> CandidateViewConfig:
    values = {
        "infeasible_geometry_policy": "centroid_radial_mask_chord",
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


def _mask(shape=(80, 90), box=(30, 30, 39, 39)) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    result[y0 : y1 + 1, x0 : x1 + 1] = True
    return result


def _image(shape=(80, 90)) -> np.ndarray:
    height, width = shape
    values = np.arange(height * width * 3, dtype=np.uint32) % 251
    return values.astype(np.uint8).reshape(height, width, 3)


def _horizontal_mask(mask: np.ndarray) -> None:
    mask[45:55, 15:85] = True


def _vertical_mask(mask: np.ndarray) -> None:
    mask[15:85, 45:55] = True


def _rotated_mask(mask: np.ndarray) -> None:
    yy, xx = np.indices(mask.shape)
    angle = np.deg2rad(31.0)
    dx, dy = xx - 50, yy - 50
    rotated_x = dx * np.cos(angle) + dy * np.sin(angle)
    rotated_y = -dx * np.sin(angle) + dy * np.cos(angle)
    mask[...] = (rotated_x / 38.0) ** 2 + (rotated_y / 8.0) ** 2 <= 1.0


def _concave_mask(mask: np.ndarray) -> None:
    mask[20:80, 20:28] = True
    mask[20:28, 20:80] = True


def _fragmented_mask(mask: np.ndarray) -> None:
    mask[35:45, 10:20] = True
    mask[35:45, 80:90] = True


def _hole_mask(mask: np.ndarray) -> None:
    mask[15:85, 15:85] = True
    mask[35:65, 35:65] = False


def test_policy_omission_reject_and_fallback_feasible_path_are_compatible():
    image = _image()
    mask = _mask()
    omitted = CandidateViewConfig.from_mapping({}, stage="blip3")
    explicit = CandidateViewConfig.from_mapping(
        {"infeasible_geometry_policy": "reject"}, stage="blip3"
    )
    fallback = CandidateViewConfig.from_mapping(
        {"infeasible_geometry_policy": "centroid_radial_mask_chord"}, stage="blip3"
    )
    first = compose_single_blip3_view(image, mask, 7, omitted)
    second = compose_single_blip3_view(image, mask, 7, explicit)
    third = compose_single_blip3_view(image, mask, 7, fallback)
    for candidate in (second, third):
        assert np.array_equal(candidate.rgb, first.rgb)
        assert np.array_equal(candidate.source_composite, first.source_composite)
        assert np.array_equal(candidate.raw_mask, first.raw_mask)
        assert np.array_equal(candidate.support_mask, first.support_mask)
        assert np.array_equal(candidate.contour, first.contour)
        assert candidate.crop_bbox_xyxy_exclusive == first.crop_bbox_xyxy_exclusive
        assert candidate.raw_context_radius == first.raw_context_radius
        assert candidate.effective_context_radius == first.effective_context_radius
    assert first.geometry_strategy_used == "euclidean_largest_axis"
    assert third.infeasible_geometry_policy == "centroid_radial_mask_chord"
    assert third.mask_centroid_xy is None
    assert third.external_boundary_pixel_count is None


def test_policy_is_strict_in_core_and_hostile_yaml():
    with pytest.raises(ValueError, match="infeasible_geometry_policy"):
        CandidateViewConfig.from_mapping({"infeasible_geometry_policy": True}, stage="blip3")
    with pytest.raises(ServiceError, match="infeasible_geometry_policy"):
        parse_hostile_config(
            b"candidate_views:\n  blip3:\n    infeasible_geometry_policy: other\n",
            verbosity=3,
        )


@pytest.mark.parametrize("box", [(30, 60, 49, 159), (60, 30, 159, 49)])
def test_rectangle_radial_chords_use_each_axis(box):
    mask = _mask((220, 220), box)
    config = _config(context_fraction=0.20, max_context_pixels=256)
    geometry = build_centroid_radial_geometry(mask, mask.shape, 1, config)
    assert int(geometry.raw_distances.min()) == 4
    assert int(geometry.raw_distances.max()) == 20
    assert geometry.raw_distances.dtype == np.intp


def test_rotated_elongated_mask_and_concave_mask_preserve_raw_pixels():
    shape = (180, 180)
    yy, xx = np.ogrid[:180, :180]
    dx, dy = xx - 90, yy - 90
    angle = np.deg2rad(31.0)
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    rotated = (rx / 52.0) ** 2 + (ry / 9.0) ** 2 <= 1.0
    composition = compose_single_blip3_view(_image(shape), rotated, 2, _config())
    x0, y0, x1, y1 = composition.crop_bbox_xyxy_exclusive
    raw_x0, raw_y0, raw_x1, raw_y1 = composition.raw_mask_bbox_xyxy_inclusive
    assert x1 - x0 <= min(shape[1], math.ceil(2.0 * (raw_x1 - raw_x0 + 1)))
    assert y1 - y0 <= min(shape[0], math.ceil(2.0 * (raw_y1 - raw_y0 + 1)))
    assert composition.geometry_strategy_used in {
        "euclidean_largest_axis",
        "centroid_radial_mask_chord_fallback",
    }
    concave = np.zeros(shape, dtype=bool)
    concave[55:125, 65:75] = True
    concave[55:65, 65:125] = True
    concave[115:125, 65:125] = True
    geometry = build_centroid_radial_geometry(concave, shape, 3, _config())
    support, _, _ = geometry.support_for_scale(1_000_000)
    wx0, wy0, _, _ = geometry.window_bbox_xyxy_exclusive
    assert np.all(support[concave[wy0 : wy0 + support.shape[0], wx0 : wx0 + support.shape[1]]])


def test_fragmented_chord_counts_positive_runs_after_a_gap():
    mask = np.zeros((24, 40), dtype=bool)
    mask[10:14, 2:6] = True
    mask[10:14, 30:34] = True
    geometry = build_centroid_radial_geometry(mask, mask.shape, 1, _config(context_fraction=0.5))
    indices = np.where(np.all(geometry.boundary_points == (2, 11), axis=1))[0]
    assert indices.size == 1
    # The complete chord sees both four-pixel foreground runs; stopping at the
    # first zero gap would produce a smaller value.
    assert int(geometry.raw_distances[indices[0]]) == 4
    assert geometry.centroid_xy[0] == pytest.approx(17.5)


def test_centroid_in_gap_and_degenerate_positive_x_convention():
    fragmented = np.zeros((30, 40), dtype=bool)
    fragmented[12:16, 4:8] = True
    fragmented[12:16, 28:32] = True
    geometry = build_centroid_radial_geometry(fragmented, fragmented.shape, 1, _config())
    assert geometry.centroid_xy == pytest.approx((17.5, 13.5))
    support, _, _ = geometry.support_for_scale(1_000_000)
    assert np.count_nonzero(support) >= np.count_nonzero(fragmented)

    one = np.zeros((15, 15), dtype=bool)
    one[7, 5] = True
    geometry = build_centroid_radial_geometry(one, one.shape, 1, _config(context_fraction=0.5))
    support, _, _ = geometry.support_for_scale(1_000_000)
    wx0, wy0, _, _ = geometry.window_bbox_xyxy_exclusive
    assert support[7 - wy0, 6 - wx0]


@pytest.mark.parametrize("point", [(0, 0), (39, 0), (0, 59), (39, 59)])
def test_edge_and_corner_masks_keep_nominal_crop_and_source_pixels(point):
    mask = np.zeros((60, 40), dtype=bool)
    x, y = point
    mask[y, x] = True
    composition = compose_single_blip3_view(
        np.zeros((*mask.shape, 3), dtype=np.uint8), mask, 1, _config(context_fraction=0.5)
    )
    x0, y0, x1, y1 = composition.crop_bbox_xyxy_exclusive
    assert 0 <= x0 < x1 <= 40 and 0 <= y0 < y1 <= 60
    assert np.count_nonzero(composition.raw_mask) == 1


def test_external_contours_are_deterministic_and_holes_are_not_seeds():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 15:25] = False
    mask[5, 5] = True
    first = build_centroid_radial_geometry(mask, mask.shape, 1, _config())
    second = build_centroid_radial_geometry(mask, mask.shape, 1, _config())
    assert first.contours == second.contours
    assert first.external_boundary_pixel_count == sum(map(len, first.contours))
    assert not any(15 <= x < 25 and 15 <= y < 25 for contour in first.contours for x, y in contour)
    assert not first.boundary_points.flags.writeable


def test_contour_reduction_disable_scaling_and_zero_context_precedence():
    image = _image((50, 50))
    mask = _mask((50, 50), (20, 20, 29, 29))
    reduced = compose_single_blip3_view(
        image,
        mask,
        1,
        _config(
            crop_extent_multiplier=1.2,
            context_fraction=0.0,
            contour_fraction=0.25,
            contour_min_pixels=1,
            contour_max_pixels=3,
        ),
    )
    assert reduced.geometry_adjustment == "contour_reduced"
    assert reduced.effective_contour_width == 1
    disabled = compose_single_blip3_view(
        image,
        mask,
        1,
        _config(
            crop_extent_multiplier=1.0,
            context_fraction=0.0,
            contour_fraction=0.25,
            contour_min_pixels=1,
            contour_max_pixels=3,
        ),
    )
    assert disabled.geometry_adjustment == "contour_disabled"
    scaled = compose_single_blip3_view(
        image,
        mask,
        1,
        _config(crop_extent_multiplier=1.5, context_fraction=0.5, contour_enabled=False),
    )
    assert scaled.geometry_adjustment == "radial_context_scaled"
    assert 0.0 < scaled.effective_radial_scale < 1.0
    zero = compose_single_blip3_view(
        image,
        mask,
        1,
        _config(crop_extent_multiplier=1.0, context_fraction=0.5, contour_enabled=False),
    )
    assert zero.geometry_adjustment == "zero_context_fallback"
    assert zero.effective_radial_scale == 0.0


def test_common_fixed_point_distance_rounding_is_exact():
    mask = _mask((50, 50), (20, 20, 29, 29))
    geometry = build_centroid_radial_geometry(mask, mask.shape, 1, _config(context_fraction=0.5))
    q = 333_333
    actual = geometry.distances_for_scale(q)
    expected = q * geometry.bounded_distances // 1_000_000
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, geometry.distances_for_scale(q))


def test_composite_pixel_provenance_and_contour_color_are_exact():
    image = _image((70, 70))
    mask = _mask((70, 70), (25, 25, 34, 34))
    config = _config(
        crop_extent_multiplier=1.5,
        context_fraction=0.5,
        contour_enabled=True,
        contour_fraction=0.25,
        contour_min_pixels=1,
        contour_max_pixels=3,
    )
    composition = compose_single_blip3_view(image, mask, 4, config)
    x0, y0, x1, y1 = composition.crop_bbox_xyxy_exclusive
    source_crop = image[y0:y1, x0:x1]
    blurred = np.asarray(
        Image.fromarray(source_crop, mode="RGB").filter(
            ImageFilter.GaussianBlur(composition.effective_blur_sigma)
        )
    )
    assert np.array_equal(
        composition.source_composite[composition.support_mask],
        source_crop[composition.support_mask],
    )
    assert not np.any(composition.contour & composition.support_mask)
    assert np.all(
        composition.source_composite[composition.contour] == np.asarray(config.contour_rgb)
    )
    other = ~(composition.support_mask | composition.contour)
    assert np.array_equal(composition.source_composite[other], blurred[other])


def _fake_filter() -> tuple[_Blip3Filter, list[Image.Image]]:
    received: list[Image.Image] = []

    def answer(image, _query, max_new_tokens):
        assert max_new_tokens == 768
        received.append(image.copy())
        return "yes"

    holder = object.__new__(_Blip3Filter)
    holder.label_cfg = {
        "target": {
            "question": "is this target?",
            "trueresult": "yes",
            "falseresult": "no",
            "debug": True,
        }
    }
    holder.max_questions = None
    holder.max_new_tokens = None
    holder.qa = SimpleNamespace(answer=answer)
    holder.verbosity = 3
    holder.log_print = lambda *_args, **_kwargs: None
    return holder, received


def test_one_debug_artifact_matches_the_single_qa_image_and_timer_boundaries():
    from src.core import MemoryArtifactSink

    holder, received = _fake_filter()
    records: list[dict] = []
    inputs: list[dict] = []
    image = _image((70, 70))
    mask = _mask((70, 70), (20, 20, 29, 29))
    config = _config(crop_extent_multiplier=1.0, context_fraction=0.5, contour_enabled=False)
    sink = MemoryArtifactSink()
    holder.filter_masks(
        [{"segmentation": mask, "clip_label": "target", "clip_score": 0.1}],
        image,
        None,
        "request",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_config=config,
        candidate_view_inputs=inputs,
        candidate_view_records=records,
    )
    assert len(received) == 1
    assert len(sink.artifacts()) == 1
    artifact = sink.artifacts()[0]
    encoded = io.BytesIO()
    Image.fromarray(artifact.array, mode="RGB").save(encoded, format="PNG")
    decoded = np.asarray(Image.open(io.BytesIO(encoded.getvalue())))
    assert np.array_equal(decoded, np.asarray(received[0]))
    assert inputs[0]["geometry_strategy_used"] == "centroid_radial_mask_chord_fallback"
    assert records[0]["geometry_strategy_used"] == "centroid_radial_mask_chord_fallback"
    assert holder._last_composition_time_ms > 0.0
    assert holder._last_verification_time_ms > 0.0


def test_fallback_records_are_schema_valid_and_capabilities_advertise_both_policies():
    from src.core import MemoryArtifactSink

    holder, _received = _fake_filter()
    records: list[dict] = []
    inputs: list[dict] = []
    mask = _mask((70, 70), (20, 20, 29, 29))
    holder.filter_masks(
        [{"segmentation": mask, "clip_label": "target", "clip_score": 0.1}],
        _image((70, 70)),
        None,
        "request",
        artifact_sink=MemoryArtifactSink(),
        candidate_view_config=_config(
            crop_extent_multiplier=1.0, context_fraction=0.5, contour_enabled=False
        ),
        candidate_view_inputs=inputs,
        candidate_view_records=records,
    )
    Blip3CandidateViewRecord.model_validate(records[0])
    CandidateViewInputRecord.model_validate(inputs[0])
    invalid = dict(records[0])
    invalid["raw_radial_distance_max"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        Blip3CandidateViewRecord.model_validate(invalid)
    invalid_input = dict(inputs[0])
    invalid_input["raw_radial_distance_max"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        CandidateViewInputRecord.model_validate(invalid_input)
    capabilities = build_capabilities(ServiceSettings())
    policy = capabilities["candidate_views"]["blip3"]["fields"]["infeasible_geometry_policy"]
    assert policy["allowed"] == ["reject", "centroid_radial_mask_chord"]
    assert (
        capabilities["candidate_views"]["blip3"]["defaults"]["infeasible_geometry_policy"]
        == "reject"
    )


def test_ray_batches_are_fixed_and_forced_small_batches_are_identical(monkeypatch):
    mask = _mask((199, 199), (5, 5, 193, 193))
    observed: list[int] = []
    original = radial_geometry._rasterize_lines

    def recording_batch(starts, ends, **kwargs):
        observed.append(int(np.asarray(starts).shape[0]))
        return original(starts, ends, **kwargs)

    monkeypatch.setattr(radial_geometry, "_rasterize_lines", recording_batch)
    geometry = build_centroid_radial_geometry(mask, mask.shape, 1, _config())
    production_support, _production_endpoints, _production_distances = geometry.support_for_scale(
        1_000_000
    )
    assert len(observed) > 1
    assert max(observed) <= radial_geometry._RAY_BATCH_SIZE

    origin = np.asarray(geometry.window_bbox_xyxy_exclusive[:2], dtype=np.intp)
    starts = geometry.boundary_points - origin
    ends = starts.copy()
    production_counts = radial_geometry._line_positive_counts(
        geometry.raw_mask_window, starts, ends
    )
    small_counts = radial_geometry._line_positive_counts(
        geometry.raw_mask_window, starts, ends, ray_batch_size=3
    )
    assert np.array_equal(production_counts, small_counts)
    assert production_support.dtype == np.dtype(bool)


def test_geometry_scratch_is_tight_bbox_local_and_translation_is_exact(monkeypatch):
    small = np.zeros((220, 220), dtype=bool)
    small[90:120, 80:106] = True
    offset_x, offset_y = 150, 170
    large = np.zeros((500, 500), dtype=bool)
    large[offset_y + 90 : offset_y + 120, offset_x + 80 : offset_x + 106] = True
    config = _config()

    small_geometry = build_centroid_radial_geometry(small, small.shape, 1, config)
    seen_shapes: list[tuple[int, int]] = []
    original_components = radial_geometry._component_pixels

    def recording_components(local_mask):
        seen_shapes.append(tuple(local_mask.shape))
        return original_components(local_mask)

    monkeypatch.setattr(radial_geometry, "_component_pixels", recording_components)
    large_geometry = build_centroid_radial_geometry(large, large.shape, 1, config)

    assert seen_shapes == [(30, 26)]
    assert large_geometry.raw_mask_window.shape == small_geometry.raw_mask_window.shape
    assert np.array_equal(large_geometry.raw_mask_window, small_geometry.raw_mask_window)
    assert large_geometry.raw_bbox_xyxy_inclusive == (
        small_geometry.raw_bbox_xyxy_inclusive[0] + offset_x,
        small_geometry.raw_bbox_xyxy_inclusive[1] + offset_y,
        small_geometry.raw_bbox_xyxy_inclusive[2] + offset_x,
        small_geometry.raw_bbox_xyxy_inclusive[3] + offset_y,
    )
    assert large_geometry.centroid_xy == pytest.approx(
        (small_geometry.centroid_xy[0] + offset_x, small_geometry.centroid_xy[1] + offset_y)
    )
    assert large_geometry.contours == tuple(
        tuple((x + offset_x, y + offset_y) for x, y in contour)
        for contour in small_geometry.contours
    )
    assert np.array_equal(
        large_geometry.boundary_points,
        small_geometry.boundary_points + np.asarray((offset_x, offset_y)),
    )
    assert np.array_equal(large_geometry.raw_distances, small_geometry.raw_distances)
    assert np.array_equal(large_geometry.bounded_distances, small_geometry.bounded_distances)
    small_support, small_endpoints, _ = small_geometry.support_for_scale(1_000_000)
    large_support, large_endpoints, _ = large_geometry.support_for_scale(1_000_000)
    assert np.array_equal(large_support, small_support)
    assert np.array_equal(
        large_endpoints,
        small_endpoints + np.asarray((offset_x, offset_y)),
    )


@pytest.mark.parametrize(
    "shape_builder",
    [_horizontal_mask, _vertical_mask, _rotated_mask, _concave_mask, _fragmented_mask, _hole_mask],
    ids=["horizontal", "vertical", "rotated", "concave", "fragmented", "hole"],
)
def test_generated_mask_families_are_deterministic_and_preserve_raw_pixels(shape_builder):
    mask = np.zeros((100, 100), dtype=bool)
    shape_builder(mask)
    first = compose_single_blip3_view(_image(mask.shape), mask, 1, _config())
    second = compose_single_blip3_view(_image(mask.shape), mask, 1, _config())
    assert np.array_equal(first.rgb, second.rgb)
    assert np.array_equal(first.source_composite, second.source_composite)
    assert np.array_equal(first.raw_mask, second.raw_mask)
    x0, y0, x1, y1 = first.crop_bbox_xyxy_exclusive
    assert np.array_equal(first.raw_mask, mask[y0:y1, x0:x1])
    assert int(np.count_nonzero(first.raw_mask)) == int(np.count_nonzero(mask))


@pytest.mark.parametrize(
    ("shape_builder", "expected_raw_min", "expected_raw_max"),
    [(_horizontal_mask, 2, 14), (_vertical_mask, 2, 14), (_rotated_mask, 3, 13)],
    ids=["horizontal", "vertical", "rotated"],
)
def test_elongated_masks_are_forced_through_fallback_and_keep_chord_stats(
    shape_builder, expected_raw_min, expected_raw_max
):
    mask = np.zeros((100, 100), dtype=bool)
    shape_builder(mask)
    composition = compose_single_blip3_view(
        _image(mask.shape),
        mask,
        1,
        _config(context_fraction=0.20, crop_extent_multiplier=1.0, contour_enabled=False),
    )
    assert composition.geometry_strategy_used == "centroid_radial_mask_chord_fallback"
    assert composition.raw_radial_distance_min == expected_raw_min
    assert composition.raw_radial_distance_max == expected_raw_max
    assert composition.effective_radial_scale == 0.0


def test_edge_containment_shift_is_reported_against_unshifted_nominal_crop():
    mask = np.zeros((100, 200), dtype=bool)
    mask[0:27, 60:160] = True
    composition = compose_single_blip3_view(
        _image(mask.shape),
        mask,
        1,
        _config(context_fraction=0.20, crop_extent_multiplier=2.0),
    )
    assert composition.geometry_strategy_used == "centroid_radial_mask_chord_fallback"
    assert composition.geometry_adjustment == "crop_shifted"
