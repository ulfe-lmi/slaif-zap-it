"""Pure, request-local candidate views for semantic model inputs.

The SAM2 mask, rather than its rectangular bounding box, is the authority for
which source pixels can reach a semantic model.  This module deliberately has
no model, filesystem, environment, or service dependencies so it can be used by
both the CLIP and BLIP3 adapters and tested entirely with generated arrays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import CoreError

__all__ = [
    "CANDIDATE_VIEW_DEFAULTS",
    "CANDIDATE_VIEW_LIMITS",
    "CandidateViewConfig",
    "MaskViewResult",
    "build_candidate_views",
    "build_mask_views",
    "default_candidate_view_configs",
    "effective_candidate_view_configs",
]


CANDIDATE_VIEW_DEFAULTS: dict[str, dict[str, Any]] = {
    "clip": {
        "mode": "mask_dilated",
        "context_fraction": 0.10,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
    },
    "blip3": {
        "mode": "mask_dilated",
        "context_fraction": 0.10,
        "min_context_pixels": 0,
        "max_context_pixels": 64,
        "outside_fill": "zero",
        "context_intensity": 0.35,
        "contour_width": 2,
    },
}

CANDIDATE_VIEW_LIMITS: dict[str, tuple[float, float]] = {
    "context_fraction": (0.0, 0.5),
    "min_context_pixels": (0, 256),
    "max_context_pixels": (0, 512),
    "context_intensity": (0.0, 1.0),
    "contour_width": (0, 16),
}


def _invalid(message: str) -> CoreError:
    return CoreError(message)


@dataclass(frozen=True)
class CandidateViewConfig:
    """Validated scalar policy for one candidate-view stage."""

    mode: str = "mask_dilated"
    context_fraction: float = 0.10
    min_context_pixels: int = 0
    max_context_pixels: int = 64
    outside_fill: str = "zero"
    context_intensity: float = 0.35
    contour_width: int = 0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, stage: str = "clip"
    ) -> "CandidateViewConfig":
        """Validate a stage mapping without coercing explicit values."""
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise _invalid(f"candidate_views.{stage} must be a mapping")
        allowed = {
            "mode",
            "context_fraction",
            "min_context_pixels",
            "max_context_pixels",
            "outside_fill",
            "context_intensity",
        }
        if stage == "blip3":
            allowed.add("contour_width")
        unknown = sorted(set(value).difference(allowed), key=str)
        if unknown:
            raise _invalid(
                f"candidate_views.{stage} has unsupported field(s): "
                + ", ".join(str(item) for item in unknown)
            )
        defaults = CANDIDATE_VIEW_DEFAULTS.get(stage, CANDIDATE_VIEW_DEFAULTS["clip"])
        mode = value.get("mode", defaults["mode"])
        if type(mode) is not str or mode != "mask_dilated":
            raise _invalid(f"candidate_views.{stage}.mode must be 'mask_dilated'")
        outside_fill = value.get("outside_fill", defaults["outside_fill"])
        if type(outside_fill) is not str or outside_fill != "zero":
            raise _invalid(f"candidate_views.{stage}.outside_fill must be 'zero'")

        fraction = value.get("context_fraction", defaults["context_fraction"])
        if type(fraction) not in (int, float) or not math.isfinite(float(fraction)):
            raise _invalid(f"candidate_views.{stage}.context_fraction must be a finite number")
        if (
            not CANDIDATE_VIEW_LIMITS["context_fraction"][0]
            <= float(fraction)
            <= CANDIDATE_VIEW_LIMITS["context_fraction"][1]
        ):
            raise _invalid(f"candidate_views.{stage}.context_fraction must be between 0 and 0.5")

        minimum = value.get("min_context_pixels", defaults["min_context_pixels"])
        maximum = value.get("max_context_pixels", defaults["max_context_pixels"])
        for field_name, candidate in (
            ("min_context_pixels", minimum),
            ("max_context_pixels", maximum),
        ):
            if type(candidate) is not int:
                raise _invalid(f"candidate_views.{stage}.{field_name} must be an integer")
            lower, upper = CANDIDATE_VIEW_LIMITS[field_name]
            if not lower <= candidate <= upper:
                raise _invalid(
                    f"candidate_views.{stage}.{field_name} must be between "
                    f"{int(lower)} and {int(upper)}"
                )
        if minimum > maximum:
            raise _invalid(
                f"candidate_views.{stage}.min_context_pixels must not exceed max_context_pixels"
            )

        intensity = value.get("context_intensity", defaults["context_intensity"])
        if type(intensity) not in (int, float) or not math.isfinite(float(intensity)):
            raise _invalid(f"candidate_views.{stage}.context_intensity must be a finite number")
        if not 0.0 <= float(intensity) <= 1.0:
            raise _invalid(f"candidate_views.{stage}.context_intensity must be between 0 and 1")

        contour = value.get("contour_width", defaults.get("contour_width", 0))
        if type(contour) is not int:
            raise _invalid(f"candidate_views.{stage}.contour_width must be an integer")
        if not 0 <= contour <= 16:
            raise _invalid(f"candidate_views.{stage}.contour_width must be between 0 and 16")

        return cls(
            mode=mode,
            context_fraction=float(fraction),
            min_context_pixels=minimum,
            max_context_pixels=maximum,
            outside_fill=outside_fill,
            context_intensity=float(intensity),
            contour_width=contour,
        )

    def as_dict(self, *, stage: str = "clip", applied: bool | None = None) -> dict[str, Any]:
        """Return the public effective values for this stage."""
        result: dict[str, Any] = {
            "mode": self.mode,
            "context_fraction": self.context_fraction,
            "min_context_pixels": self.min_context_pixels,
            "max_context_pixels": self.max_context_pixels,
            "outside_fill": self.outside_fill,
            "context_intensity": self.context_intensity,
        }
        if stage == "blip3":
            result["contour_width"] = self.contour_width
        if applied is not None:
            result["applied"] = bool(applied)
        return result


def default_candidate_view_configs() -> dict[str, CandidateViewConfig]:
    """Create independent safe defaults for both semantic stages."""
    return {
        stage: CandidateViewConfig.from_mapping(values, stage=stage)
        for stage, values in CANDIDATE_VIEW_DEFAULTS.items()
    }


def effective_candidate_view_configs(value: Any = None) -> dict[str, CandidateViewConfig]:
    """Normalize a complete or partial top-level candidate-view mapping."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise _invalid("candidate_views must be a mapping")
    unknown = sorted(set(value).difference({"clip", "blip3"}), key=str)
    if unknown:
        raise _invalid(
            "candidate_views has unsupported stage(s): " + ", ".join(str(item) for item in unknown)
        )
    return {
        stage: CandidateViewConfig.from_mapping(
            value[stage] if stage in value else None, stage=stage
        )
        for stage in ("clip", "blip3")
    }


def _validate_inputs(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    source_candidate_id: int,
) -> None:
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.dtype(np.uint8)
        or image_rgb.shape[0] <= 0
        or image_rgb.shape[1] <= 0
    ):
        raise _invalid("candidate view source must be a non-empty RGB uint8 array")
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != image_rgb.shape[:2]
        or segmentation_mask.dtype != np.dtype(bool)
        or not np.any(segmentation_mask)
    ):
        raise _invalid("candidate view mask must be a non-empty boolean source-shaped array")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise _invalid("source candidate ID must be a positive integer")


def _distance_transform_1d(values: np.ndarray) -> np.ndarray:
    """Return the exact squared distance to the nearest finite 1-D sample.

    This is the lower-envelope algorithm for parabolas.  It is used twice by
    :func:`_circular_dilate` to calculate an exact squared Euclidean distance
    transform without retaining one image-sized array per disk row.
    """
    length = int(values.size)
    result = np.full(length, np.inf, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result

    parabola_positions = np.empty(length, dtype=np.intp)
    intersections = np.empty(length + 1, dtype=np.float64)
    first = int(finite[0])
    parabola_positions[0] = first
    intersections[0] = -np.inf
    intersections[1] = np.inf
    envelope_size = 0

    for position_value in finite[1:]:
        position = int(position_value)
        while True:
            previous = int(parabola_positions[envelope_size])
            intersection = (
                (float(values[position]) + position * position)
                - (float(values[previous]) + previous * previous)
            ) / float(2 * (position - previous))
            if envelope_size == 0 or intersection > intersections[envelope_size]:
                break
            envelope_size -= 1
        envelope_size += 1
        parabola_positions[envelope_size] = position
        intersections[envelope_size] = intersection
        intersections[envelope_size + 1] = np.inf

    envelope_index = 0
    for position in range(length):
        while intersections[envelope_index + 1] < position:
            envelope_index += 1
        nearest = int(parabola_positions[envelope_index])
        delta = position - nearest
        result[position] = delta * delta + float(values[nearest])
    return result


def _exact_disk_dilate_window(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return exact disk dilation for a source-space local window."""
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    vertical_distances = np.empty((height, width), dtype=np.float64)
    for column in range(width):
        values = np.where(mask[:, column], 0.0, np.inf)
        vertical_distances[:, column] = _distance_transform_1d(values)

    squared_distances = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        squared_distances[row, :] = _distance_transform_1d(vertical_distances[row, :])
    return squared_distances <= float(radius * radius)


def _dilate_cropped(
    mask: np.ndarray,
    radius: int,
    target_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Dilate only the target-expanded window and return its tight support crop."""
    if radius < 0:
        raise ValueError("dilation radius must not be negative")
    height, width = mask.shape
    wx0 = max(0, target_bbox[0] - radius)
    wy0 = max(0, target_bbox[1] - radius)
    wx1 = min(width, target_bbox[2] + radius)
    wy1 = min(height, target_bbox[3] + radius)
    support_window = _exact_disk_dilate_window(mask[wy0:wy1, wx0:wx1], radius)
    local_bbox = _tight_bbox(support_window)
    sx0, sy0, sx1, sy1 = local_bbox
    context_bbox = (wx0 + sx0, wy0 + sy0, wx0 + sx1, wy0 + sy1)
    return support_window[sy0:sy1, sx0:sx1].copy(), context_bbox


def _circular_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate by the exact integer-pixel Euclidean disk, clipped to the source."""
    if radius < 0:
        raise ValueError("dilation radius must not be negative")
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    target_bbox = _tight_bbox(mask)
    support_crop, context_bbox = _dilate_cropped(mask, radius, target_bbox)
    result = np.zeros_like(mask, dtype=bool)
    x0, y0, x1, y1 = context_bbox
    result[y0:y1, x0:x1] = support_crop
    return result


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise _invalid("candidate view support mask is empty")
    return int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MaskViewResult:
    """Immutable source-space target/context views and audit metadata."""

    target_rgb: np.ndarray
    context_rgb: np.ndarray
    target_mask: np.ndarray
    support_mask: np.ndarray
    target_bbox_xyxy: tuple[int, int, int, int]
    context_bbox_xyxy: tuple[int, int, int, int]
    effective_radius: int
    source_candidate_id: int
    metadata: Mapping[str, Any]

    @property
    def target_only_rgb(self) -> np.ndarray:
        return self.target_rgb

    @property
    def dilated_context_rgb(self) -> np.ndarray:
        return self.context_rgb

    @property
    def dilated_mask(self) -> np.ndarray:
        return self.support_mask

    def metadata_dict(self) -> dict[str, Any]:
        """Return a mutable serialization copy without exposing arrays."""
        return {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in self.metadata.items()
        }


def build_mask_views(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    source_candidate_id: int,
    config: CandidateViewConfig | Mapping[str, Any] | None = None,
    *,
    stage: str = "clip",
) -> MaskViewResult:
    """Build a deterministic, mask-isolated target and dilated-context crop."""
    _validate_inputs(image_rgb, segmentation_mask, source_candidate_id)
    if isinstance(config, CandidateViewConfig):
        view_config = config
    else:
        view_config = CandidateViewConfig.from_mapping(config, stage=stage)

    target_bbox = _tight_bbox(segmentation_mask)
    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    extent = max(target_width, target_height)
    raw_radius = math.ceil(view_config.context_fraction * extent)
    effective_radius = min(
        max(raw_radius, view_config.min_context_pixels), view_config.max_context_pixels
    )
    support_crop, context_bbox = _dilate_cropped(segmentation_mask, effective_radius, target_bbox)
    x0, y0, x1, y1 = context_bbox
    target_crop = segmentation_mask[y0:y1, x0:x1].copy()
    source_crop = image_rgb[y0:y1, x0:x1]

    target_rgb = np.zeros_like(source_crop)
    target_rgb[target_crop] = source_crop[target_crop]
    context_rgb = np.zeros_like(source_crop)
    context_rgb[target_crop] = source_crop[target_crop]
    context_ring = support_crop & ~target_crop
    if np.any(context_ring):
        context_rgb[context_ring] = (
            source_crop[context_ring].astype(np.float32) * view_config.context_intensity
        ).astype(np.uint8)

    contour = np.zeros_like(target_crop)
    if stage == "blip3" and view_config.contour_width:
        contour = _circular_dilate(target_crop, view_config.contour_width)
        contour &= ~target_crop
        contour &= support_crop
        context_rgb[contour] = np.array((255, 224, 0), dtype=np.uint8)

    metadata = MappingProxyType(
        {
            "stage": stage,
            "source_candidate_id": source_candidate_id,
            "source_shape_hw": tuple(int(value) for value in image_rgb.shape[:2]),
            "target_bbox_xyxy": target_bbox,
            "context_bbox_xyxy": context_bbox,
            "target_shape_hw": tuple(int(value) for value in target_crop.shape),
            "context_shape_hw": tuple(int(value) for value in support_crop.shape),
            "raw_radius": raw_radius,
            "effective_radius": effective_radius,
            "config": MappingProxyType(view_config.as_dict(stage=stage)),
            "context_rounding": "floor(channel * context_intensity)",
            "contour_pixels": int(np.count_nonzero(contour)),
        }
    )
    return MaskViewResult(
        target_rgb=_readonly(target_rgb),
        context_rgb=_readonly(context_rgb),
        target_mask=_readonly(target_crop),
        support_mask=_readonly(support_crop),
        target_bbox_xyxy=target_bbox,
        context_bbox_xyxy=context_bbox,
        effective_radius=effective_radius,
        source_candidate_id=source_candidate_id,
        metadata=metadata,
    )


build_candidate_views = build_mask_views
