# modules/verifier/blip3.py
"""BLIP-3 based verification module with unified interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
import unicodedata
from inspect import Parameter, signature
from typing import Any, Dict, Tuple
import os
import numpy as np
from PIL import Image, ImageFilter

from src.core.errors import CoreError


MAX_SERVICE_QUESTIONS = 256
MAX_SERVICE_NEW_TOKENS = 32
BLIP3_FIXED_INSTRUCTION = (
    "The unblurred region inside the yellow boundary is the selected candidate. "
    "The blurred surroundings are context only. Answer exactly Yes or No."
)
_BLIP3_TARGET_SHORT_SIDE = 256
_BLIP3_MAX_LONG_SIDE = 768
BLIP3_CANDIDATE_VIEW_REJECTION_REASON = "crop_cannot_contain_support_and_contour"


def normalize_blip3_token(value: str) -> str:
    """Normalize one configured/result token for exact equality matching."""
    if not isinstance(value, str):
        raise TypeError("BLIP3 result tokens must be strings")
    return unicodedata.normalize("NFKC", value).strip().casefold().rstrip(".!?,;:").strip()


class Blip3CandidateViewRejected(CoreError):
    """A candidate-local BLIP3 view could not contain its support/contour."""

    reason = BLIP3_CANDIDATE_VIEW_REJECTION_REASON

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        super().__init__(self.reason)


@dataclass(frozen=True)
class Blip3VerificationComposition:
    """Immutable one-image BLIP3 input and source-space composition facts."""

    rgb: np.ndarray
    image: Image.Image
    source_composite: np.ndarray
    raw_mask: np.ndarray
    support_mask: np.ndarray
    contour: np.ndarray
    raw_mask_bbox_xyxy_inclusive: Tuple[int, int, int, int]
    support_bbox_xyxy_inclusive: Tuple[int, int, int, int]
    crop_bbox_xyxy_exclusive: Tuple[int, int, int, int]
    crop_shape_hw: Tuple[int, int]
    source_composite_shape_hw: Tuple[int, int]
    model_input_shape_hw: Tuple[int, int]
    scale: float
    raw_context_radius: int
    effective_context_radius: int
    raw_contour_width: int
    effective_contour_width: int
    effective_blur_sigma: float
    source_candidate_id: int
    infeasible_geometry_policy: str = "reject"
    geometry_strategy_used: str = "euclidean_largest_axis"
    mask_centroid_xy: tuple[float, float] | None = None
    external_boundary_pixel_count: int | None = None
    raw_radial_distance_min: float | None = None
    raw_radial_distance_max: float | None = None
    raw_radial_distance_mean: float | None = None
    effective_radial_distance_min: float | None = None
    effective_radial_distance_max: float | None = None
    effective_radial_distance_mean: float | None = None
    effective_radial_scale: float | None = None
    geometry_adjustment: str = "none"

    @property
    def array(self) -> np.ndarray:
        """The sole final model-input RGB array."""
        return self.rgb

    @property
    def scaled_height(self) -> int:
        return self.model_input_shape_hw[0]

    @property
    def scaled_width(self) -> int:
        return self.model_input_shape_hw[1]

    @property
    def scaled_shape_hw(self) -> Tuple[int, int]:
        """Compatibility alias for final model-input dimensions."""
        return self.model_input_shape_hw

    def metadata_record(self, filtered_index: int, *, status: str = "rendered", reason=None):
        """Return the bounded L3 candidate-composition record."""
        return {
            "source_candidate_id": self.source_candidate_id,
            "filtered_index": int(filtered_index),
            "status": status,
            "reason": reason,
            "render_mode": "single_dilated_blur",
            "raw_mask_bbox_xyxy_inclusive": list(self.raw_mask_bbox_xyxy_inclusive),
            "support_bbox_xyxy_inclusive": list(self.support_bbox_xyxy_inclusive),
            "crop_bbox_xyxy_exclusive": list(self.crop_bbox_xyxy_exclusive),
            "raw_context_radius": self.raw_context_radius,
            "effective_context_radius": self.effective_context_radius,
            "raw_contour_width": self.raw_contour_width,
            "effective_contour_width": self.effective_contour_width,
            "effective_blur_sigma": self.effective_blur_sigma,
            "infeasible_geometry_policy": self.infeasible_geometry_policy,
            "geometry_strategy_used": self.geometry_strategy_used,
            "mask_centroid_xy": (
                list(self.mask_centroid_xy) if self.mask_centroid_xy is not None else None
            ),
            "external_boundary_pixel_count": self.external_boundary_pixel_count,
            "raw_radial_distance_min": self.raw_radial_distance_min,
            "raw_radial_distance_max": self.raw_radial_distance_max,
            "raw_radial_distance_mean": self.raw_radial_distance_mean,
            "effective_radial_distance_min": self.effective_radial_distance_min,
            "effective_radial_distance_max": self.effective_radial_distance_max,
            "effective_radial_distance_mean": self.effective_radial_distance_mean,
            "effective_radial_scale": self.effective_radial_scale,
            "geometry_adjustment": self.geometry_adjustment,
            "source_composite_dimensions": {
                "height": self.source_composite_shape_hw[0],
                "width": self.source_composite_shape_hw[1],
            },
            "model_input_dimensions": {
                "height": self.model_input_shape_hw[0],
                "width": self.model_input_shape_hw[1],
            },
        }


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _validate_single_view_inputs(image_rgb, segmentation_mask, source_candidate_id):
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.dtype(np.uint8)
        or image_rgb.shape[0] <= 0
        or image_rgb.shape[1] <= 0
    ):
        raise CoreError("BLIP3 view source must be a non-empty RGB uint8 array")
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != image_rgb.shape[:2]
        or segmentation_mask.dtype != np.dtype(bool)
        or not np.any(segmentation_mask)
    ):
        raise CoreError("BLIP3 view mask must be a non-empty boolean source-shaped array")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise CoreError("source candidate ID must be a positive integer")


def _tight_bbox_inclusive(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise CoreError("BLIP3 view mask must be non-empty")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _model_dimensions(height: int, width: int) -> tuple[int, int, float]:
    short_side = min(height, width)
    scale = _BLIP3_TARGET_SHORT_SIDE / float(short_side) if short_side < 256 else 1.0
    long_side = max(height, width)
    if long_side * scale > _BLIP3_MAX_LONG_SIDE:
        scale = _BLIP3_MAX_LONG_SIDE / float(long_side)
    scaled_width = max(1, int(math.floor(width * scale + 0.5)))
    scaled_height = max(1, int(math.floor(height * scale + 0.5)))
    return scaled_height, scaled_width, float(scale)


def _single_view_geometry(image_shape, segmentation_mask, source_candidate_id, config):
    from src.core.mask_views import exact_euclidean_dilate

    height, width = (int(image_shape[0]), int(image_shape[1]))
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != (height, width)
        or segmentation_mask.dtype != np.dtype(bool)
        or not np.any(segmentation_mask)
    ):
        raise CoreError("BLIP3 view mask must be a non-empty boolean source-shaped array")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise CoreError("source candidate ID must be a positive integer")
    raw_x0, raw_y0, raw_x1, raw_y1 = _tight_bbox_inclusive(segmentation_mask)
    bbox_width = raw_x1 - raw_x0 + 1
    bbox_height = raw_y1 - raw_y0 + 1
    extent = max(bbox_width, bbox_height)
    raw_radius = math.ceil(config.context_fraction * extent)
    effective_radius = min(max(raw_radius, config.min_context_pixels), config.max_context_pixels)
    support = exact_euclidean_dilate(segmentation_mask, effective_radius)
    support_x0, support_y0, support_x1, support_y1 = _tight_bbox_inclusive(support)
    raw_contour_width = math.ceil(config.contour_fraction * extent)
    effective_contour_width = (
        min(max(raw_contour_width, config.contour_min_pixels), config.contour_max_pixels)
        if config.contour_enabled
        else 0
    )
    contour = (
        exact_euclidean_dilate(support, effective_contour_width) & ~support
        if effective_contour_width
        else np.zeros_like(support, dtype=bool)
    )

    nominal_width = math.ceil(config.crop_extent_multiplier * bbox_width)
    nominal_height = math.ceil(config.crop_extent_multiplier * bbox_height)
    center_x = (raw_x0 + raw_x1) / 2.0
    center_y = (raw_y0 + raw_y1) / 2.0
    crop_x0_unclamped = math.floor(center_x - (nominal_width - 1) / 2.0)
    crop_y0_unclamped = math.floor(center_y - (nominal_height - 1) / 2.0)
    crop_x1_unclamped = crop_x0_unclamped + nominal_width
    crop_y1_unclamped = crop_y0_unclamped + nominal_height
    crop_x0 = max(0, min(width, crop_x0_unclamped))
    crop_y0 = max(0, min(height, crop_y0_unclamped))
    crop_x1 = max(0, min(width, crop_x1_unclamped))
    crop_y1 = max(0, min(height, crop_y1_unclamped))
    crop_box = (crop_x0, crop_y0, crop_x1, crop_y1)

    outside = (support | contour).copy()
    outside[crop_y0:crop_y1, crop_x0:crop_x1] = False
    metadata = {
        "source_candidate_id": source_candidate_id,
        "raw_mask_bbox_xyxy_inclusive": [raw_x0, raw_y0, raw_x1, raw_y1],
        "support_bbox_xyxy_inclusive": [support_x0, support_y0, support_x1, support_y1],
        "crop_bbox_xyxy_exclusive": list(crop_box),
        "raw_context_radius": raw_radius,
        "effective_context_radius": effective_radius,
        "raw_contour_width": raw_contour_width,
        "effective_contour_width": effective_contour_width,
        "effective_blur_sigma": min(max(config.blur_sigma_fraction * extent, 2.0), 20.0),
        "source_composite_dimensions": {
            "height": max(crop_y1 - crop_y0, 0),
            "width": max(crop_x1 - crop_x0, 0),
        },
        "infeasible_geometry_policy": config.infeasible_geometry_policy,
        "geometry_strategy_used": "euclidean_largest_axis",
        "mask_centroid_xy": None,
        "external_boundary_pixel_count": None,
        "raw_radial_distance_min": None,
        "raw_radial_distance_max": None,
        "raw_radial_distance_mean": None,
        "effective_radial_distance_min": None,
        "effective_radial_distance_max": None,
        "effective_radial_distance_mean": None,
        "effective_radial_scale": None,
        "geometry_adjustment": "none",
    }
    if crop_x1 > crop_x0 and crop_y1 > crop_y0:
        planned_height, planned_width, _planned_scale = _model_dimensions(
            crop_y1 - crop_y0, crop_x1 - crop_x0
        )
        metadata["model_input_dimensions"] = {
            "height": planned_height,
            "width": planned_width,
        }
    if np.any(outside) or crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise Blip3CandidateViewRejected(metadata)
    model_height, model_width, scale = _model_dimensions(crop_y1 - crop_y0, crop_x1 - crop_x0)
    metadata["model_input_dimensions"] = {"height": model_height, "width": model_width}
    return {
        "raw_bbox": (raw_x0, raw_y0, raw_x1, raw_y1),
        "support_bbox": (support_x0, support_y0, support_x1, support_y1),
        "crop_box": crop_box,
        "support": support,
        "contour": contour,
        "raw_radius": raw_radius,
        "effective_radius": effective_radius,
        "raw_contour_width": raw_contour_width,
        "effective_contour_width": effective_contour_width,
        "effective_sigma": metadata["effective_blur_sigma"],
        "model_height": model_height,
        "model_width": model_width,
        "scale": scale,
        "infeasible_geometry_policy": config.infeasible_geometry_policy,
        "geometry_strategy_used": "euclidean_largest_axis",
        "mask_centroid_xy": None,
        "external_boundary_pixel_count": None,
        "raw_radial_distance_min": None,
        "raw_radial_distance_max": None,
        "raw_radial_distance_mean": None,
        "effective_radial_distance_min": None,
        "effective_radial_distance_max": None,
        "effective_radial_distance_mean": None,
        "effective_radial_scale": None,
        "geometry_adjustment": "none",
    }


def _inclusive_bbox_for_window(
    mask: np.ndarray, window_origin: tuple[int, int]
) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise CoreError("BLIP3 view support must be non-empty")
    x0, y0 = window_origin
    return (
        int(cols.min()) + x0,
        int(rows.min()) + y0,
        int(cols.max()) + x0,
        int(rows.max()) + y0,
    )


def _union_inclusive_bboxes(*boxes: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _fallback_crop_box(
    image_shape: tuple[int, int],
    raw_bbox: tuple[int, int, int, int],
    required_bbox: tuple[int, int, int, int],
    multiplier: float,
) -> tuple[int, int, int, int] | None:
    """Choose the nearest valid full-size crop that contains ``required_bbox``."""
    height, width = image_shape
    raw_x0, raw_y0, raw_x1, raw_y1 = raw_bbox
    required_x0, required_y0, required_x1, required_y1 = required_bbox
    nominal_width = min(width, math.ceil(multiplier * (raw_x1 - raw_x0 + 1)))
    nominal_height = min(height, math.ceil(multiplier * (raw_y1 - raw_y0 + 1)))

    def choose_axis(
        raw_start: int,
        raw_end: int,
        required_start: int,
        required_end: int,
        nominal: int,
        source_size: int,
    ) -> int | None:
        centered_unclamped = math.floor((raw_start + raw_end) / 2.0 - (nominal - 1) / 2.0)
        legacy_origin = max(0, min(source_size, centered_unclamped))
        lower = max(0, required_end + 1 - nominal)
        upper = min(required_start, source_size - nominal)
        if lower > upper:
            return None
        # The valid origins form one integer interval, so clamping the old
        # centered origin is the unique nearest choice (lower wins ties).
        return min(max(legacy_origin, lower), upper)

    crop_x0 = choose_axis(raw_x0, raw_x1, required_x0, required_x1, nominal_width, width)
    crop_y0 = choose_axis(raw_y0, raw_y1, required_y0, required_y1, nominal_height, height)
    if crop_x0 is None or crop_y0 is None:
        return None
    return crop_x0, crop_y0, crop_x0 + nominal_width, crop_y0 + nominal_height


def _centroid_radial_fallback_geometry(
    image_shape,
    segmentation_mask: np.ndarray,
    source_candidate_id: int,
    config,
):
    """Build the expert-directed fallback after the Euclidean path rejects."""
    from src.core.mask_views import exact_euclidean_dilate
    from src.core.radial_geometry import build_centroid_radial_geometry

    height, width = (int(image_shape[0]), int(image_shape[1]))
    radial = build_centroid_radial_geometry(
        segmentation_mask, (height, width), source_candidate_id, config
    )
    raw_bbox = radial.raw_bbox_xyxy_inclusive
    raw_width = raw_bbox[2] - raw_bbox[0] + 1
    raw_height = raw_bbox[3] - raw_bbox[1] + 1
    extent = max(raw_width, raw_height)
    requested_raw_contour = math.ceil(config.contour_fraction * extent)
    requested_contour = (
        min(
            max(requested_raw_contour, config.contour_min_pixels),
            config.contour_max_pixels,
        )
        if config.contour_enabled
        else 0
    )
    window_origin = (
        radial.window_bbox_xyxy_exclusive[0],
        radial.window_bbox_xyxy_exclusive[1],
    )
    centered_box = _fallback_crop_box(
        (height, width), raw_bbox, raw_bbox, config.crop_extent_multiplier
    )

    def contour_for(support: np.ndarray, contour_width: int) -> np.ndarray:
        if contour_width == 0:
            return np.zeros_like(support, dtype=bool)
        return exact_euclidean_dilate(support, contour_width) & ~support

    def attempt(
        support: np.ndarray, contour_width: int, effective_distances: np.ndarray
    ) -> dict[str, Any] | None:
        support_bbox = _inclusive_bbox_for_window(support, window_origin)
        contour = contour_for(support, contour_width)
        required_bbox = (
            _union_inclusive_bboxes(
                support_bbox, _inclusive_bbox_for_window(contour, window_origin)
            )
            if np.any(contour)
            else support_bbox
        )
        crop_box = _fallback_crop_box(
            (height, width), raw_bbox, required_bbox, config.crop_extent_multiplier
        )
        if crop_box is None:
            return None
        return {
            "support": support,
            "contour": contour,
            "support_bbox": support_bbox,
            "crop_box": crop_box,
            "effective_distances": effective_distances,
        }

    initial_support, _initial_endpoints, initial_distances = radial.support_for_scale(1_000_000)
    selected = attempt(initial_support, requested_contour, initial_distances)
    selected_contour_width = requested_contour
    selected_scale_q = 1_000_000

    if selected is None and requested_contour:
        for contour_width in range(requested_contour - 1, 0, -1):
            selected = attempt(initial_support, contour_width, initial_distances)
            if selected is not None:
                selected_contour_width = contour_width
                break
        if selected is None:
            selected_contour_width = 0
            selected = attempt(initial_support, 0, initial_distances)

    if selected is None:
        # At this point support itself cannot fit.  Search the single common
        # fixed-point scale; a successful q=0 result is guaranteed because the
        # nominal crop is at least the raw bbox and the contour is disabled.
        selected_contour_width = 0
        low, high, best = 0, 1_000_000, 0
        while low <= high:
            candidate_q = (low + high) // 2
            support, _endpoints, distances = radial.support_for_scale(candidate_q)
            candidate = attempt(support, 0, distances)
            if candidate is None:
                high = candidate_q - 1
            else:
                best = candidate_q
                low = candidate_q + 1
        selected_scale_q = best
        support, _endpoints, distances = radial.support_for_scale(selected_scale_q)
        selected = attempt(support, 0, distances)
        while selected is None and selected_scale_q > 0:
            selected_scale_q -= 1
            support, _endpoints, distances = radial.support_for_scale(selected_scale_q)
            selected = attempt(support, 0, distances)
        if selected is None:
            raise CoreError("centroid-radial fallback could not contain the raw mask")
        if selected_scale_q > 0 and not np.any(selected["effective_distances"]):
            selected_scale_q = 0
            support, _endpoints, distances = radial.support_for_scale(0)
            selected = attempt(support, 0, distances)
            if selected is None:
                raise CoreError(
                    "centroid-radial zero-context fallback could not contain the raw mask"
                )

    effective_distances = selected["effective_distances"]
    if selected_scale_q == 0:
        adjustment = "zero_context_fallback"
    elif selected_scale_q < 1_000_000:
        adjustment = "radial_context_scaled"
    elif requested_contour and selected_contour_width == 0:
        adjustment = "contour_disabled"
    elif selected_contour_width < requested_contour:
        adjustment = "contour_reduced"
    else:
        crop_box = selected["crop_box"]
        adjustment = "crop_shifted" if centered_box != crop_box else "none"

    model_height, model_width, scale = _model_dimensions(
        selected["crop_box"][3] - selected["crop_box"][1],
        selected["crop_box"][2] - selected["crop_box"][0],
    )
    raw_distances = radial.raw_distances.astype(np.float64)
    effective_distances_float = effective_distances.astype(np.float64)
    return {
        "raw_bbox": raw_bbox,
        "support_bbox": selected["support_bbox"],
        "crop_box": selected["crop_box"],
        "support": selected["support"],
        "contour": selected["contour"],
        "raw_radius": int(np.max(radial.raw_distances)),
        "effective_radius": int(np.max(effective_distances)),
        "raw_contour_width": requested_raw_contour,
        "effective_contour_width": selected_contour_width,
        "effective_sigma": min(max(config.blur_sigma_fraction * extent, 2.0), 20.0),
        "model_height": model_height,
        "model_width": model_width,
        "scale": scale,
        "infeasible_geometry_policy": config.infeasible_geometry_policy,
        "geometry_strategy_used": "centroid_radial_mask_chord_fallback",
        "mask_centroid_xy": radial.centroid_xy,
        "external_boundary_pixel_count": radial.external_boundary_pixel_count,
        "raw_radial_distance_min": float(np.min(raw_distances)),
        "raw_radial_distance_max": float(np.max(raw_distances)),
        "raw_radial_distance_mean": float(np.mean(raw_distances)),
        "effective_radial_distance_min": float(np.min(effective_distances_float)),
        "effective_radial_distance_max": float(np.max(effective_distances_float)),
        "effective_radial_distance_mean": float(np.mean(effective_distances_float)),
        "effective_radial_scale": selected_scale_q / 1_000_000.0,
        "geometry_adjustment": adjustment,
        "window_origin": window_origin,
    }


def compose_single_blip3_view(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    source_candidate_id: int,
    config=None,
) -> Blip3VerificationComposition:
    """Compose one exact source-space BLIP3 candidate image.

    The raw mask and its Euclidean support are restored from the source crop;
    every other crop pixel is a Pillow Gaussian-blurred scene pixel.  A crop
    that cannot contain the full support plus exterior contour is rejected
    before image reads, blur, model calls, or debug artifact creation.
    """
    from src.core.mask_views import CandidateViewConfig

    view_config = (
        config
        if isinstance(config, CandidateViewConfig)
        else CandidateViewConfig.from_mapping(config, stage="blip3")
    )
    if view_config.stage != "blip3":
        view_config = CandidateViewConfig.from_mapping(
            view_config.as_dict(stage="blip3"), stage="blip3"
        )
    _validate_single_view_inputs(image_rgb, segmentation_mask, source_candidate_id)
    try:
        # Compatibility is intentional: the reviewed Euclidean path is always
        # evaluated first, even when the opt-in fallback is configured.
        geometry = _single_view_geometry(
            image_rgb.shape, segmentation_mask, source_candidate_id, view_config
        )
    except Blip3CandidateViewRejected:
        if view_config.infeasible_geometry_policy != "centroid_radial_mask_chord":
            raise
        geometry = _centroid_radial_fallback_geometry(
            image_rgb.shape, segmentation_mask, source_candidate_id, view_config
        )
    x0, y0, x1, y1 = geometry["crop_box"]
    raw_mask_crop = np.ascontiguousarray(segmentation_mask[y0:y1, x0:x1].copy())
    if "window_origin" in geometry:
        wx0, wy0 = geometry["window_origin"]

        def crop_local(source: np.ndarray) -> np.ndarray:
            result = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            overlap_x0, overlap_y0 = max(x0, wx0), max(y0, wy0)
            overlap_x1, overlap_y1 = min(x1, wx0 + source.shape[1]), min(y1, wy0 + source.shape[0])
            if overlap_x0 < overlap_x1 and overlap_y0 < overlap_y1:
                result[
                    overlap_y0 - y0 : overlap_y1 - y0,
                    overlap_x0 - x0 : overlap_x1 - x0,
                ] = source[
                    overlap_y0 - wy0 : overlap_y1 - wy0,
                    overlap_x0 - wx0 : overlap_x1 - wx0,
                ]
            return result

        support_crop = crop_local(geometry["support"])
        contour_crop = crop_local(geometry["contour"])
    else:
        support_crop = np.ascontiguousarray(geometry["support"][y0:y1, x0:x1].copy())
        contour_crop = np.ascontiguousarray(geometry["contour"][y0:y1, x0:x1].copy())
    source_crop = np.ascontiguousarray(image_rgb[y0:y1, x0:x1].copy())
    blurred = np.asarray(
        Image.fromarray(source_crop, mode="RGB").filter(
            ImageFilter.GaussianBlur(geometry["effective_sigma"])
        ),
        dtype=np.uint8,
    ).copy()
    composite = blurred
    composite[support_crop] = source_crop[support_crop]
    composite[contour_crop] = np.asarray(view_config.contour_rgb, dtype=np.uint8)
    model_height, model_width, scale = (
        geometry["model_height"],
        geometry["model_width"],
        geometry["scale"],
    )
    final_rgb = np.asarray(
        Image.fromarray(composite, mode="RGB").resize(
            (model_width, model_height), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    ).copy()
    return Blip3VerificationComposition(
        rgb=_readonly(final_rgb),
        image=Image.fromarray(final_rgb, mode="RGB"),
        source_composite=_readonly(composite),
        raw_mask=_readonly(raw_mask_crop),
        support_mask=_readonly(support_crop),
        contour=_readonly(contour_crop),
        raw_mask_bbox_xyxy_inclusive=geometry["raw_bbox"],
        support_bbox_xyxy_inclusive=geometry["support_bbox"],
        crop_bbox_xyxy_exclusive=geometry["crop_box"],
        crop_shape_hw=(int(composite.shape[0]), int(composite.shape[1])),
        source_composite_shape_hw=(int(composite.shape[0]), int(composite.shape[1])),
        model_input_shape_hw=(int(final_rgb.shape[0]), int(final_rgb.shape[1])),
        scale=scale,
        raw_context_radius=geometry["raw_radius"],
        effective_context_radius=geometry["effective_radius"],
        raw_contour_width=geometry["raw_contour_width"],
        effective_contour_width=geometry["effective_contour_width"],
        effective_blur_sigma=float(geometry["effective_sigma"]),
        source_candidate_id=source_candidate_id,
        infeasible_geometry_policy=geometry["infeasible_geometry_policy"],
        geometry_strategy_used=geometry["geometry_strategy_used"],
        mask_centroid_xy=geometry["mask_centroid_xy"],
        external_boundary_pixel_count=geometry["external_boundary_pixel_count"],
        raw_radial_distance_min=geometry["raw_radial_distance_min"],
        raw_radial_distance_max=geometry["raw_radial_distance_max"],
        raw_radial_distance_mean=geometry["raw_radial_distance_mean"],
        effective_radial_distance_min=geometry["effective_radial_distance_min"],
        effective_radial_distance_max=geometry["effective_radial_distance_max"],
        effective_radial_distance_mean=geometry["effective_radial_distance_mean"],
        effective_radial_scale=geometry["effective_radial_scale"],
        geometry_adjustment=geometry["geometry_adjustment"],
    )


def single_blip3_view_model_input_shape(
    image_shape, segmentation_mask, source_candidate_id: int, config=None
) -> tuple[int, int]:
    """Return final RGB dimensions for pre-model debug admission.

    This performs only bounded geometry and containment checks.  It does not
    read image pixels, blur, resize, call BLIP3, or create an artifact; the
    filter owns the one actual composition for an admitted candidate.
    """
    from src.core.mask_views import CandidateViewConfig

    view_config = (
        config
        if isinstance(config, CandidateViewConfig)
        else CandidateViewConfig.from_mapping(config, stage="blip3")
    )
    if view_config.stage != "blip3":
        view_config = CandidateViewConfig.from_mapping(
            view_config.as_dict(stage="blip3"), stage="blip3"
        )
    try:
        geometry = _single_view_geometry(
            image_shape, segmentation_mask, source_candidate_id, view_config
        )
    except Blip3CandidateViewRejected:
        if view_config.infeasible_geometry_policy != "centroid_radial_mask_chord":
            raise
        geometry = _centroid_radial_fallback_geometry(
            image_shape, segmentation_mask, source_candidate_id, view_config
        )
    return geometry["model_height"], geometry["model_width"]


def single_blip3_view_model_input_nbytes(
    image_shape, segmentation_mask, source_candidate_id: int, config=None
) -> int:
    """Return the exact uncompressed RGB bytes reserved for one model input."""
    height, width = single_blip3_view_model_input_shape(
        image_shape, segmentation_mask, source_candidate_id, config
    )
    return int(height * width * 3)


def compose_blip3_verification_image(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    config=None,
    source_candidate_id: int = 1,
) -> Blip3VerificationComposition:
    """Compatibility entry point for the one-image BLIP3 compositor."""
    return compose_single_blip3_view(image_rgb, segmentation_mask, source_candidate_id, config)


def compose_verification_image(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    config=None,
) -> Blip3VerificationComposition:
    """Compatibility name for the one-image BLIP3 compositor."""
    return compose_blip3_verification_image(image_rgb, segmentation_mask, config)


def compose_verification_query(target_question: str) -> str:
    """Keep the bounded client question before the fixed region task."""
    if not isinstance(target_question, str):
        raise TypeError("BLIP3 target question must be a string")
    return f"[TARGET QUESTION]\n{target_question}\n[/TARGET QUESTION]\n{BLIP3_FIXED_INSTRUCTION}"


class Blip3ResourceLimitError(ValueError):
    """Raised before generation when the service BLIP3 budget is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        planned_questions: int | None = None,
        allowed_limit: int | None = None,
    ) -> None:
        super().__init__(message)
        self.planned_questions = planned_questions
        self.allowed_limit = allowed_limit

    @property
    def details(self) -> dict[str, Any] | None:
        """Return bounded client-safe planning evidence when available."""
        if self.planned_questions is None or self.allowed_limit is None:
            return None
        alternatives = [
            "set clip_routing.max_candidates to a lower deterministic cap",
            "tighten the clip_routing predicates to route fewer candidates",
        ]
        if self.allowed_limit < MAX_SERVICE_QUESTIONS:
            alternatives.append(
                "raise SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS only up to the 256-question maximum"
            )
        return {
            "planned_questions": int(self.planned_questions),
            "allowed_limit": int(self.allowed_limit),
            "controlling_field": "SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS",
            "admissible_alternatives": alternatives,
        }


# -------------------------------------------------------------------------
# Safe fallback if transformers isn't present during dry-run
# -------------------------------------------------------------------------
try:
    from transformers import StoppingCriteria
except ImportError:  # pragma: no cover

    class StoppingCriteria:  # type: ignore
        def __call__(self, *args, **kwargs):
            raise RuntimeError("transformers is required for BLIP-3 execution")


class _EosListStoppingCriteria(StoppingCriteria):
    """Stops generation when the special BLIP-3 end-of-answer sequence appears."""

    def __init__(self, eos_sequence=(32007,)):
        self.eos_sequence = list(eos_sequence)

    def __call__(self, input_ids, _scores, **kwargs):
        if len(input_ids[0]) < len(self.eos_sequence):
            return False
        return input_ids[0][-len(self.eos_sequence) :].tolist() == self.eos_sequence


# -------------------------------------------------------------------------
# Patches
# -------------------------------------------------------------------------
def _install_safe_to_for_meta():
    """
    Patch torch.nn.Module.to to gracefully handle meta tensors by using
    to_empty(device=...) instead of raising NotImplementedError.
    """
    import torch.nn as nn

    if getattr(nn.Module, "_zap_it_meta_to_patched", False):
        return
    _orig_to = nn.Module.to

    def _safe_to(self, *args, **kwargs):
        try:
            return _orig_to(self, *args, **kwargs)
        except NotImplementedError:
            device = kwargs.get("device", None)
            if device is None and len(args) >= 1:
                device = args[0]
            if device is None:
                raise
            try:
                return self.to_empty(device=device)
            except Exception:
                raise

    nn.Module.to = _safe_to  # type: ignore[attr-defined]
    nn.Module._zap_it_meta_to_patched = True  # type: ignore[attr-defined]


def _force_openclip_default_pretrained(default_tag: str = "laion2b_s32b_b79k"):
    """
    If a ViT-H-14 backbone is created without 'pretrained', inject a sensible default.
    Not required for the fix; kept as an optional utility.
    """
    try:
        import open_clip.factory as ocf
    except Exception:
        return
    if getattr(ocf, "_zap_it_pretrained_wrapped", False):
        return
    _orig = ocf.create_model_and_transforms

    def _wrapped(model_name, *args, **kwargs):
        pt = kwargs.get("pretrained", None)
        if (pt in (None, "", False)) and ("ViT-H-14" in str(model_name)):
            kwargs["pretrained"] = default_tag
        return _orig(model_name, *args, **kwargs)

    ocf.create_model_and_transforms = _wrapped  # type: ignore[attr-defined]
    ocf._zap_it_pretrained_wrapped = True  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# BLIP-3 QA core
# -------------------------------------------------------------------------
class _Blip3QA:
    def __init__(
        self,
        blip_config: Dict[str, Any],
        device="cuda",
        verbosity: int = 1,
        log_print_func=None,
        *,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer,
        )

        self._torch = torch
        want_cuda = str(device).startswith("cuda")
        self.device = torch.device(device if (want_cuda and torch.cuda.is_available()) else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # ---- Config knobs (all optional) ----
        self.model_name = blip_config.get(
            "model_name", "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )
        self.revision = blip_config.get("revision")
        load_kwargs = {"revision": str(self.revision)} if self.revision else {}
        if local_files_only:
            load_kwargs["local_files_only"] = True

        # dtype: "auto" | "float16" | "bfloat16" | "float32"
        dtype_cfg = str(blip_config.get("dtype", "auto")).lower()

        def _bf16_ok() -> bool:
            return (self.device.type == "cuda") and getattr(
                torch.cuda, "is_bf16_supported", lambda: False
            )()

        if dtype_cfg == "auto":
            # XGen-MM (Phi-3) prefers BF16; mixing FP16/BF16 can break generate()
            if (
                "phi3" in self.model_name.lower() or "xgen-mm" in self.model_name.lower()
            ) and _bf16_ok():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        elif dtype_cfg == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_cfg == "float16":
            dtype = torch.float16
        elif dtype_cfg == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float32

        use_fast_tok = bool(blip_config.get("use_fast_tokenizer", True))
        use_fast_proc = bool(blip_config.get("use_fast_processor", True))

        self.log_print(f"[_Blip3QA] loading {self.model_name}", 1, self.verbosity)

        # Ensure .to(...) on meta-tensors falls back to to_empty(...)
        _install_safe_to_for_meta()

        # Keep OpenCLIP on CPU during construction to avoid early CUDA moves.
        os.environ.setdefault("OPENCLIP_DEFAULT_DEVICE", "cpu")

        # ---- Create the model with REAL CPU tensors (avoid init_empty_weights/meta),
        #      then move the fully materialized model to the target device + dtype.
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **load_kwargs,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,  # <<< critical: don't init on meta
            device_map=None,  # avoid accelerate sharding/meta route
            attn_implementation="eager",  # optional; remove if your stack complains
        )

        # Unify device & dtype; prevents BF16/FP16 mismatches in remote generate()
        self.model.to(device=self.device, dtype=dtype).eval()
        try:
            self.estimated_gpu_bytes = sum(
                int(parameter.numel()) * int(parameter.element_size())
                for parameter in self.model.parameters()
            )
        except (AttributeError, TypeError):
            self.estimated_gpu_bytes = 0
        try:
            # Some remote loaders keep nested modules in a different dtype; normalize them.
            if hasattr(self.model, "vlm") and hasattr(self.model.vlm, "lang_model"):
                self.model.vlm.lang_model.to(dtype=dtype)
        except Exception:
            pass

        # Tokenizer & processor (prefer fast to avoid warnings; falls back if unavailable)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **load_kwargs,
            trust_remote_code=True,
            use_fast=use_fast_tok,
            legacy=False,
        )
        if hasattr(self.model, "update_special_tokens"):
            self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            **load_kwargs,
            trust_remote_code=True,
            use_fast=use_fast_proc,
        )

        self._prompt = (
            "<|system|>\nA chat between a curious user and an artificial "
            "intelligence assistant. The assistant gives helpful, detailed, "
            "and polite answers to the user's questions.<|end|>\n"
            "<|user|>\n<image>\n{q}<|end|>\n<|assistant|>\n"
        )

        self.stopper = _EosListStoppingCriteria()

    def move_to(self, device: Any) -> None:
        """Move only the reusable model holder; request tensors are never kept."""
        target = self._torch.device(device)
        self.model.to(device=target)
        self.device = target

    def answer(self, image, query: str, max_new_tokens: int = 768) -> str:
        from PIL import Image as _PILImage

        if not isinstance(image, _PILImage.Image):
            image = _PILImage.fromarray(image)

        torch = self._torch

        vision_inputs = self.image_processor(
            [image],
            return_tensors="pt",
            image_aspect_ratio="anyres",
        )

        prompt = self._prompt.format(q=query)
        lang_inputs = self.tokenizer([prompt], return_tensors="pt")

        inputs = {**vision_inputs, **lang_inputs}

        # >>> CRITICAL: cast all floating tensors to the model's dtype (e.g., BF16) <<<
        model_dtype = next(self.model.parameters()).dtype

        def _to_dev_dtype(x):
            if torch.is_tensor(x):
                return (
                    x.to(self.device, dtype=model_dtype)
                    if x.is_floating_point()
                    else x.to(self.device)
                )
            if isinstance(x, (list, tuple)):
                return type(x)(_to_dev_dtype(t) for t in x)
            if isinstance(x, dict):
                return {kk: _to_dev_dtype(vv) for kk, vv in x.items()}
            return x

        inputs = {k: _to_dev_dtype(v) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                image_size=[image.size],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                top_p=None,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[self.stopper],
            )

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.split("<|end|>")[0].strip()


# -------------------------------------------------------------------------
# BLIP3 filter wrapper
# -------------------------------------------------------------------------
class _Blip3Filter:
    def __init__(
        self,
        blip_config: Dict[str, Any],
        device="cuda",
        verbosity: int = 1,
        log_print_func=None,
        *,
        qa: _Blip3QA | None = None,
        max_questions: int | None = None,
        max_new_tokens: int | None = None,
    ):
        import torch

        self._torch = torch
        self.device = torch.device(
            device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        self.label_cfg: Dict[str, Dict[str, Any]] = {}
        self.max_questions = max_questions
        self.max_new_tokens = max_new_tokens
        self.qa = qa
        self.update_rules(blip_config)
        if self.qa is None:
            model_cfg: Dict[str, Any] = {
                k: v for k, v in blip_config.items() if not isinstance(v, dict)
            }
            self.qa = _Blip3QA(
                model_cfg, device=device, verbosity=verbosity, log_print_func=self.log_print
            )

    def update_rules(self, blip_config: Dict[str, Any]) -> None:
        """Replace request rules without changing the reusable model holder."""
        self.label_cfg = {
            str(key): dict(value)
            for key, value in (blip_config or {}).items()
            if isinstance(value, dict)
        }

    @classmethod
    def from_qa(
        cls,
        qa: _Blip3QA,
        blip_config: Dict[str, Any],
        *,
        verbosity: int = 1,
        log_print_func=None,
        max_questions: int | None = None,
        max_new_tokens: int | None = None,
    ) -> "_Blip3Filter":
        return cls(
            blip_config,
            device=qa.device,
            verbosity=verbosity,
            log_print_func=log_print_func,
            qa=qa,
            max_questions=max_questions,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _legacy_frame_stem(fname_stem: Any) -> str:
        """Keep trusted CLI names bounded and independent of user rule text."""
        stem = str(fname_stem).replace("\\", "/").rsplit("/", 1)[-1]
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem).strip("._")
        return stem[:96] or "image"

    @staticmethod
    def _planned_question_count(masks, any_rules, label_rules) -> int:
        """Count every request question before any candidate view/model call."""
        planned_questions = 0
        for mask in masks:
            routing = mask.get("clip_routing")
            if isinstance(routing, dict):
                if mask.get("_route_to_blip3") and routing.get("chosen_target") in label_rules:
                    planned_questions += 1
                continue
            score = float(mask.get("clip_score", 0.0))
            planned_questions += sum(
                1 for threshold, _key, _rule in any_rules if score <= threshold
            )
            if mask.get("clip_label") in label_rules:
                planned_questions += 1
        return planned_questions

    def _enforce_question_limit(self, planned_questions: int) -> None:
        """Reject a planned workload before composition or BLIP3 generation."""
        if self.max_questions is not None and planned_questions > self.max_questions:
            raise Blip3ResourceLimitError(
                f"BLIP3 candidate count exceeds the fixed {self.max_questions}-question limit",
                planned_questions=planned_questions,
                allowed_limit=self.max_questions,
            )

    def _write_debug_artifact(
        self,
        model_input: np.ndarray,
        out_dir,
        fname_stem,
        candidate_index: int,
        question_index: int,
        artifact_sink,
        *,
        service_safe_artifact_names: bool,
    ):
        """Write only the exact lossless image passed to BLIP3."""
        if service_safe_artifact_names:
            image_name = (
                f"blip3-verification-CANDIDATE-{candidate_index:04d}-"
                f"QUESTION-{question_index:04d}.png"
            )
        else:
            image_name = (
                f"{self._legacy_frame_stem(fname_stem)}-blip3-verification-"
                f"CANDIDATE-{candidate_index:04d}-QUESTION-{question_index:04d}.png"
            )
        if artifact_sink is not None:
            artifact_sink.store_image(image_name, model_input, fmt="png")
        else:
            if out_dir is None:
                raise ValueError("BLIP3 debug requires an artifact sink or output directory")
            Image.fromarray(model_input).save(os.path.join(out_dir, image_name), format="PNG")
        self.log_print(f"[_Blip3Filter debug] => wrote {image_name}", 2, self.verbosity)
        return image_name

    def filter_masks(
        self,
        masks,
        image_np,
        out_dir,
        fname_stem,
        artifact_sink=None,
        *,
        service_safe_artifact_names: bool = False,
        candidate_view_config=None,
        candidate_view_inputs=None,
        candidate_view_records=None,
    ):
        from src.core.mask_views import CandidateViewConfig

        composition_time_ms = 0.0
        verification_time_ms = 0.0
        self._last_composition_time_ms = composition_time_ms
        self._last_verification_time_ms = verification_time_ms

        if not self.label_cfg:
            return masks, []

        view_config = (
            candidate_view_config
            if isinstance(candidate_view_config, CandidateViewConfig)
            else CandidateViewConfig.from_mapping(candidate_view_config, stage="blip3")
        )

        any_rules, label_rules = [], {}
        for key, rule in self.label_cfg.items():
            if isinstance(key, str) and key.startswith("any,"):
                try:
                    thr = float(key.split(",", 1)[1])
                except ValueError:
                    continue
                any_rules.append((thr, key, rule))
            else:
                label_rules[key] = rule

        if self.max_questions is not None:
            self._enforce_question_limit(
                self._planned_question_count(masks, any_rules, label_rules)
            )

        answers = []
        question_index = 0
        blip3_candidate_views = candidate_view_records

        for idx, m in enumerate(masks):
            lbl = m.get("clip_label")
            score = float(m.get("clip_score", 0.0))
            verification = None
            source_index = m.get("_source_index")
            source_candidate_id = (
                int(source_index) + 1
                if type(source_index) is int and source_index >= 0
                else idx + 1
            )
            filtered_index = int(m.get("_filtered_index", idx))

            routing = m.get("clip_routing")
            canonical_route = isinstance(routing, dict) and "chosen_target" in routing
            if canonical_route:
                target_label = routing.get("chosen_target")
                applicable_rules = (
                    [label_rules[target_label]]
                    if m.get("_route_to_blip3") and target_label in label_rules
                    else []
                )
            else:
                target_label = lbl
                applicable_rules = [cfg for threshold, _key, cfg in any_rules if score <= threshold]
                if lbl in label_rules:
                    applicable_rules.append(label_rules[lbl])
            if not applicable_rules:
                continue

            composition_started = time.perf_counter()
            try:
                verification = compose_single_blip3_view(
                    image_np,
                    m["segmentation"],
                    source_candidate_id if source_candidate_id > 0 else idx + 1,
                    view_config,
                )
            except Blip3CandidateViewRejected as exc:
                composition_time_ms += (time.perf_counter() - composition_started) * 1000.0
                m["_blip3_rejected"] = True
                record = dict(exc.metadata)
                record.update(
                    {
                        "filtered_index": filtered_index,
                        "status": "rejected",
                        "reason": exc.reason,
                        "render_mode": "single_dilated_blur",
                    }
                )
                if blip3_candidate_views is not None:
                    blip3_candidate_views.append(record)
                continue
            else:
                composition_time_ms += (time.perf_counter() - composition_started) * 1000.0

            if blip3_candidate_views is not None:
                blip3_candidate_views.append(
                    verification.metadata_record(filtered_index, status="rendered")
                )

            def ask(cfg):
                nonlocal question_index, verification_time_ms
                current_question_index = question_index
                question_index += 1
                question = cfg.get("question", "")
                query = compose_verification_query(question)
                debug_array = verification.rgb.copy() if cfg.get("debug") is True else None
                verification_started = time.perf_counter()
                try:
                    answer = self.qa.answer(
                        verification.image,
                        query,
                        max_new_tokens=(
                            self.max_new_tokens if self.max_new_tokens is not None else 768
                        ),
                    )
                finally:
                    verification_time_ms += (time.perf_counter() - verification_started) * 1000.0
                if cfg.get("debug") is True:
                    public_question_id = current_question_index + 1
                    artifact_name = self._write_debug_artifact(
                        debug_array,
                        out_dir,
                        fname_stem,
                        source_candidate_id,
                        public_question_id,
                        artifact_sink,
                        service_safe_artifact_names=service_safe_artifact_names,
                    )
                    if candidate_view_inputs is not None:
                        candidate_view_inputs.append(
                            {
                                "stage": "blip3",
                                "source_candidate_id": source_candidate_id,
                                "filtered_index": filtered_index,
                                "question_id": public_question_id,
                                "artifact_name": artifact_name,
                                "artifact_status": (
                                    artifact_sink.artifact_status(artifact_name)
                                    if artifact_sink is not None
                                    and hasattr(artifact_sink, "artifact_status")
                                    else "emitted"
                                ),
                                "raw_mask_bbox_xyxy_inclusive": list(
                                    verification.raw_mask_bbox_xyxy_inclusive
                                ),
                                "support_bbox_xyxy_inclusive": list(
                                    verification.support_bbox_xyxy_inclusive
                                ),
                                "crop_bbox_xyxy_exclusive": list(
                                    verification.crop_bbox_xyxy_exclusive
                                ),
                                "raw_context_radius": verification.raw_context_radius,
                                "effective_context_radius": verification.effective_context_radius,
                                "raw_contour_width": verification.raw_contour_width,
                                "effective_contour_width": verification.effective_contour_width,
                                "effective_blur_sigma": verification.effective_blur_sigma,
                                "source_composite_dimensions": {
                                    "height": int(verification.source_composite_shape_hw[0]),
                                    "width": int(verification.source_composite_shape_hw[1]),
                                },
                                "model_input_dimensions": {
                                    "height": int(verification.rgb.shape[0]),
                                    "width": int(verification.rgb.shape[1]),
                                },
                                "infeasible_geometry_policy": verification.infeasible_geometry_policy,
                                "geometry_strategy_used": verification.geometry_strategy_used,
                                "mask_centroid_xy": (
                                    list(verification.mask_centroid_xy)
                                    if verification.mask_centroid_xy is not None
                                    else None
                                ),
                                "external_boundary_pixel_count": (
                                    verification.external_boundary_pixel_count
                                ),
                                "raw_radial_distance_min": verification.raw_radial_distance_min,
                                "raw_radial_distance_max": verification.raw_radial_distance_max,
                                "raw_radial_distance_mean": verification.raw_radial_distance_mean,
                                "effective_radial_distance_min": (
                                    verification.effective_radial_distance_min
                                ),
                                "effective_radial_distance_max": (
                                    verification.effective_radial_distance_max
                                ),
                                "effective_radial_distance_mean": (
                                    verification.effective_radial_distance_mean
                                ),
                                "effective_radial_scale": verification.effective_radial_scale,
                                "geometry_adjustment": verification.geometry_adjustment,
                            }
                        )
                return (
                    answer,
                    query,
                    artifact_name if cfg.get("debug") is True else None,
                    current_question_index + 1,
                )

            def apply_answer(cfg, answer_info):
                answer, query, artifact_name, public_question_id = answer_info
                m["blip3_answer"] = answer
                answers.append(answer)
                true_value = str(cfg.get("trueresult", ""))
                false_value = str(cfg.get("falseresult", ""))
                true_token = normalize_blip3_token(true_value)
                false_token = normalize_blip3_token(false_value)
                answer_token = normalize_blip3_token(str(answer))
                if canonical_route:
                    if answer_token == true_token:
                        mapping_outcome = "true_match"
                        final_label = str(cfg["newcategory"])
                    elif answer_token == false_token:
                        mapping_outcome = "false_match"
                        final_label = str(cfg["falsecategory"])
                    else:
                        mapping_outcome = "unmatched_answer"
                        final_label = str(cfg["falsecategory"])
                else:
                    # Trusted legacy rules retain their historical substring
                    # behavior and implicit negative category.
                    answer_lower = str(answer).lower()
                    if false_value and false_value.lower() in answer_lower:
                        mapping_outcome = "false_match"
                        final_label = str(cfg.get("falsecategory", "negative"))
                    elif true_value and true_value.lower() in answer_lower:
                        mapping_outcome = "true_match"
                        final_label = str(cfg.get("newcategory", m.get("clip_label", lbl)))
                    else:
                        mapping_outcome = "unmatched_answer"
                        final_label = str(cfg.get("falsecategory", m.get("clip_label", lbl)))
                m["clip_label"] = final_label
                verification_record = {
                    "source_candidate_id": source_candidate_id,
                    "filtered_index": filtered_index,
                    "question_id": public_question_id,
                    "routing_target_label": target_label,
                    "routing_reason": (routing.get("primary_reason") if canonical_route else None),
                    "configured_question": str(cfg.get("question", "")),
                    "effective_question": query,
                    "raw_answer": str(answer),
                    "normalized_answer": answer_token,
                    "normalized_true_result": true_token,
                    "normalized_false_result": false_token,
                    "configured_true_result": true_value,
                    "configured_false_result": false_value,
                    "configured_true_label": str(cfg.get("newcategory", "")),
                    "configured_false_label": str(cfg.get("falsecategory", "negative")),
                    "mapping_outcome": mapping_outcome,
                    "input_artifact_name": artifact_name,
                    "input_artifact_status": (
                        artifact_sink.artifact_status(artifact_name)
                        if artifact_name
                        and artifact_sink is not None
                        and hasattr(artifact_sink, "artifact_status")
                        else "emitted"
                        if artifact_name
                        else "not_requested"
                    ),
                    "final_label": final_label,
                }
                m.setdefault("blip3_verifications", []).append(verification_record)
                m["blip3_verification"] = verification_record
                return mapping_outcome

            if canonical_route:
                apply_answer(applicable_rules[0], ask(applicable_rules[0]))
                continue

            processed = False
            # "any,<thr>" rules: only ask BLIP3 if CLIP score is <= thr
            for thr, _key, cfg in any_rules:
                if score > thr:
                    continue
                processed = apply_answer(cfg, ask(cfg)) != "unmatched_answer"
                break

            if processed:
                continue

            cfg = label_rules.get(lbl)
            if cfg:
                apply_answer(cfg, ask(cfg))

        self._last_composition_time_ms = max(0.0, composition_time_ms)
        self._last_verification_time_ms = max(0.0, verification_time_ms)
        return masks, answers


# -------------------------------------------------------------------------
# Dry-run fallback
# -------------------------------------------------------------------------
class _DryRunBlip3Filter:
    """Simulate BLIP-3 by deterministically approving/rejecting masks."""

    def __init__(self, *, verbosity: int = 1, log_print_func=None):
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

    def filter_masks(self, masks, _image_np, _out_dir, _fname_stem, artifact_sink=None):
        answers = []
        for idx, mask in enumerate(masks, start=1):
            if idx % 2 == 1:
                mask["clip_label"] = "negative"
                answer = "dryrun: rejected"
            else:
                answer = "dryrun: accepted"
            mask["blip3_answer"] = answer
            answers.append(answer)
        self.log_print(f"[_DryRunBlip3Filter] processed {len(masks)} masks", 2, self.verbosity)
        return masks, answers


# -------------------------------------------------------------------------
# Entry points
# -------------------------------------------------------------------------
def run(
    state: Dict[str, Any] | None,
    params: Dict[str, Any],
    images,
    *,
    verbosity: int = 1,
    log_print_func=None,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run BLIP-3 verification using the unified module interface."""
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    dryrun_mode = bool(params.get("dryrun", False))

    blip_filter = state.get("blip3_filter")
    holder = state.get("blip3_qa")
    request_state = state
    if holder is not None:
        # The holder is shared, but this filter and its rules are request-local.
        request_state = dict(state)
        blip_filter = _Blip3Filter.from_qa(
            holder,
            params.get("config", {}) or {},
            verbosity=verbosity,
            log_print_func=log,
            max_questions=params.get("max_questions"),
            max_new_tokens=params.get("max_new_tokens"),
        )
    elif blip_filter is None:
        blip_cfg = params.get("config", {})
        device = params.get("device", "cuda")
        blip_filter = (
            _DryRunBlip3Filter(verbosity=verbosity, log_print_func=log)
            if dryrun_mode
            else _Blip3Filter(blip_cfg, device=device, verbosity=verbosity, log_print_func=log)
        )
        state["blip3_filter"] = blip_filter

    image_np = images[0] if isinstance(images, (list, tuple)) else images

    masks = params.get("masks")
    if masks is None:
        raise ValueError("BLIP-3 verifier requires 'masks' in params")

    out_dir = params.get("out_dir")
    fname_stem = params.get("fname_stem", "image")
    artifact_sink = params.get("artifact_sink")
    filter_kwargs = {}
    if artifact_sink is not None:
        filter_kwargs["artifact_sink"] = artifact_sink
    try:
        filter_parameters = signature(blip_filter.filter_masks).parameters.values()
        accepts_kwargs = any(
            parameter.kind == Parameter.VAR_KEYWORD for parameter in filter_parameters
        )
        accepted_names = {parameter.name for parameter in filter_parameters}
    except (TypeError, ValueError):
        accepts_kwargs = False
        accepted_names = set()
    if (
        isinstance(blip_filter, _Blip3Filter)
        and "service_safe_artifact_names" in params
        and (accepts_kwargs or "service_safe_artifact_names" in accepted_names)
    ):
        filter_kwargs["service_safe_artifact_names"] = bool(params["service_safe_artifact_names"])
    if isinstance(blip_filter, _Blip3Filter) and (
        accepts_kwargs or "candidate_view_config" in accepted_names
    ):
        filter_kwargs["candidate_view_config"] = params.get("candidate_view_config")
        if accepts_kwargs or "candidate_view_inputs" in accepted_names:
            filter_kwargs["candidate_view_inputs"] = params.get("candidate_view_inputs")
        if accepts_kwargs or "candidate_view_records" in accepted_names:
            filter_kwargs["candidate_view_records"] = params.get("candidate_view_records")

    updated_masks, answers = blip_filter.filter_masks(
        masks, image_np, out_dir, fname_stem, **filter_kwargs
    )
    meta = {
        "answers": answers,
        "num_masks": len(updated_masks) if updated_masks is not None else 0,
        "verified_count": len(answers),
        "composition_time_ms": max(
            0.0, float(getattr(blip_filter, "_last_composition_time_ms", 0.0))
        ),
        "verification_time_ms": max(
            0.0, float(getattr(blip_filter, "_last_verification_time_ms", 0.0))
        ),
    }
    return (state if holder is not None else request_state), updated_masks, meta


def initialize(
    config: Dict[str, Any],
    *,
    dryrun: bool = False,
    device="cuda",
    verbosity: int = 1,
    log_print_func=None,
) -> Dict[str, Any]:
    """Prepare a BLIP-3 filter or its dry-run counterpart."""
    log = log_print_func or (lambda *a, **k: None)

    if dryrun:
        log("[verifier.blip3] Initializing dry-run BLIP-3 filter", 1, verbosity)
        return {"blip3_filter": _DryRunBlip3Filter(verbosity=verbosity, log_print_func=log)}

    log("[verifier.blip3] Initializing BLIP-3 filter", 1, verbosity)
    blip_filter = _Blip3Filter(config, device=device, verbosity=verbosity, log_print_func=log)
    return {"blip3_filter": blip_filter}


def initialize_holder(
    *,
    device: str = "cpu",
    verbosity: int = 0,
    log_print_func=None,
    local_files_only: bool = True,
    model_name: str | None = None,
    revision: str | None = None,
) -> Dict[str, Any]:
    """Initialize the pinned reusable BLIP3 holder without request rules."""
    from src.runtime.models import APPROVED_MODEL_SPECS

    spec = APPROVED_MODEL_SPECS["blip3"]
    model_name = model_name or spec.model_id
    revision = revision or spec.revision
    qa = _Blip3QA(
        {
            "model_name": model_name,
            "revision": revision,
            "dtype": "float16",
            "use_fast_tokenizer": True,
            "use_fast_processor": True,
        },
        device=device,
        verbosity=verbosity,
        log_print_func=log_print_func,
        local_files_only=local_files_only,
    )
    return {"blip3_qa": qa}


__all__ = [
    "Blip3ResourceLimitError",
    "Blip3CandidateViewRejected",
    "BLIP3_FIXED_INSTRUCTION",
    "BLIP3_CANDIDATE_VIEW_REJECTION_REASON",
    "MAX_SERVICE_NEW_TOKENS",
    "MAX_SERVICE_QUESTIONS",
    "Blip3VerificationComposition",
    "compose_single_blip3_view",
    "single_blip3_view_model_input_shape",
    "single_blip3_view_model_input_nbytes",
    "compose_blip3_verification_image",
    "compose_verification_image",
    "compose_verification_query",
    "normalize_blip3_token",
    "initialize",
    "initialize_holder",
    "run",
]
