"""Bounded, deterministic visualizations of raw SAM2 candidates.

This module deliberately has no model, service, filesystem or environment
dependencies.  It turns the complete non-empty, original-resolution SAM2
candidate sequence into a small set of RGB arrays for the service memory
artifact sink.  The engine supplies the raw count and empty-candidate count in
the manifest; this renderer is concerned only with the candidates it receives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is a runtime dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

__all__ = [
    "RAW_CANDIDATE_ID_BASE",
    "RAW_CONTACT_SHEET_COLUMNS",
    "RAW_CONTACT_SHEET_ROWS",
    "RAW_CANDIDATES_PER_SHEET",
    "RAW_MAXIMUM_CONTACT_SHEETS",
    "RAW_MAXIMUM_REPRESENTED_CANDIDATES",
    "RAW_TILE_CONTENT_WIDTH",
    "RAW_TILE_CONTENT_HEIGHT",
    "RAW_TILE_LABEL_HEIGHT",
    "RAW_MASK_ALPHA",
    "RAW_CONTEXT_PADDING_FRACTION",
    "RAW_MIN_CONTEXT_PADDING_PIXELS",
    "RAW_MAX_DIAGNOSTIC_PIXELS",
    "RAW_FIXED_ARTIFACT_NAMES",
    "RAW_TRUNCATION_WARNING",
    "RAW_CONTACT_SHEET_WIDTH",
    "RAW_CONTACT_SHEET_HEIGHT",
    "RawSam2Visualization",
    "candidate_color",
    "diagnostic_dimensions",
    "raw_sam2_debug_rgb_bytes",
    "render_raw_sam2_visualizations",
]

RAW_CANDIDATE_ID_BASE = 1
RAW_CONTACT_SHEET_COLUMNS = 3
RAW_CONTACT_SHEET_ROWS = 4
RAW_CANDIDATES_PER_SHEET = RAW_CONTACT_SHEET_COLUMNS * RAW_CONTACT_SHEET_ROWS
RAW_MAXIMUM_CONTACT_SHEETS = 8
RAW_MAXIMUM_REPRESENTED_CANDIDATES = 96
RAW_TILE_CONTENT_WIDTH = 320
RAW_TILE_CONTENT_HEIGHT = 240
RAW_TILE_LABEL_HEIGHT = 28
RAW_CONTACT_SHEET_WIDTH = RAW_CONTACT_SHEET_COLUMNS * RAW_TILE_CONTENT_WIDTH
RAW_CONTACT_SHEET_HEIGHT = RAW_CONTACT_SHEET_ROWS * (
    RAW_TILE_CONTENT_HEIGHT + RAW_TILE_LABEL_HEIGHT
)
RAW_MASK_ALPHA = 0.45
RAW_CONTEXT_PADDING_FRACTION = 0.10
RAW_MIN_CONTEXT_PADDING_PIXELS = 4
RAW_MAX_DIAGNOSTIC_PIXELS = 2_000_000
RAW_SHEET_NEUTRAL = (32, 32, 32)
RAW_LABEL_BACKGROUND = (16, 16, 16)
RAW_LABEL_FOREGROUND = (255, 255, 255)
RAW_TRUNCATION_WARNING = "raw SAM2 visualization truncated after 96 represented candidates"
RAW_UNION_NAME = "sam2-union-coverage.png"
RAW_OVERLAP_NAME = "sam2-overlap-heatmap.png"
RAW_UNCOVERED_NAME = "sam2-uncovered-pixels.png"
RAW_FIXED_ARTIFACT_NAMES = (
    "sam2-candidates-page-{page:04d}.png",
    RAW_UNION_NAME,
    RAW_OVERLAP_NAME,
    RAW_UNCOVERED_NAME,
)

# The palette is an arithmetic function of the public candidate id only.  It
# is intentionally independent of image pixels, scores, labels and ordering
# after the source-index sort.
_PALETTE_MULTIPLIERS = (53, 97, 193)
_PALETTE_OFFSETS = (29, 71, 149)


@dataclass(frozen=True)
class RawSam2Visualization:
    """Arrays and typed facts produced by the raw-candidate renderer."""

    artifacts: tuple[tuple[str, np.ndarray], ...]
    summary: Mapping[str, Any]


def candidate_color(candidate_id: int) -> tuple[int, int, int]:
    """Return the fixed RGB palette color for one one-based candidate id."""

    if type(candidate_id) is not int or candidate_id < RAW_CANDIDATE_ID_BASE:
        raise ValueError("candidate_id must be a positive integer")
    return tuple(
        (multiplier * candidate_id + offset) % 256
        for multiplier, offset in zip(_PALETTE_MULTIPLIERS, _PALETTE_OFFSETS)
    )


def _require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for raw SAM2 visualizations")


def _as_rgb(image_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.shape[2] != 3 or image_rgb.shape[0] <= 0 or image_rgb.shape[1] <= 0:
        raise ValueError("image_rgb must have a positive RGB shape")
    if image_rgb.dtype != np.uint8:
        return np.asarray(image_rgb, dtype=np.uint8)
    return image_rgb


def _candidate_records(
    masks: Sequence[Mapping[str, Any]], *, height: int, width: int
) -> list[tuple[int, np.ndarray, Mapping[str, Any], tuple[int, int, int, int]]]:
    records: list[tuple[int, np.ndarray, Mapping[str, Any], tuple[int, int, int, int]]] = []
    seen: set[int] = set()
    for mask in masks:
        if not isinstance(mask, Mapping):
            raise ValueError("SAM2 candidates must be mappings")
        source_index = mask.get("_source_index")
        if type(source_index) is not int or source_index < 0:
            raise ValueError("SAM2 candidate source indexes must be non-negative integers")
        candidate_id = source_index + RAW_CANDIDATE_ID_BASE
        if candidate_id in seen:
            raise ValueError("SAM2 candidate source indexes must be unique")
        seen.add(candidate_id)
        segmentation = mask.get("segmentation")
        if not isinstance(segmentation, np.ndarray) or segmentation.shape != (height, width):
            raise ValueError("SAM2 candidate masks must match the source image shape")
        bool_mask = np.asarray(segmentation, dtype=bool)
        rows, columns = np.nonzero(bool_mask)
        if rows.size == 0:
            continue
        bbox = (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))
        records.append((candidate_id, bool_mask, mask, bbox))
    records.sort(key=lambda item: item[0])
    return records


def _score_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    return f"{numeric:.3f}" if math.isfinite(numeric) else "n/a"


def _candidate_label(candidate_id: int, record: Mapping[str, Any]) -> str:
    return (
        f"C{candidate_id:04d}  IoU {_score_text(record.get('predicted_iou'))}  "
        f"stability {_score_text(record.get('stability_score'))}"
    )


def _crop_bounds(
    bbox: tuple[int, int, int, int], *, height: int, width: int
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    padding = max(
        RAW_MIN_CONTEXT_PADDING_PIXELS,
        int(math.ceil(RAW_CONTEXT_PADDING_FRACTION * max(bbox_width, bbox_height))),
    )
    return (
        max(0, x_min - padding),
        max(0, y_min - padding),
        min(width, x_max + padding + 1),
        min(height, y_max + padding + 1),
    )


def _resampling(name: str) -> Any:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return getattr(resampling, name)
    return getattr(Image, name)


def _letterboxed_tile(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = _crop_bounds(bbox, height=height, width=width)
    crop_rgb = image_rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    crop_height, crop_width = crop_rgb.shape[:2]
    scale = min(
        RAW_TILE_CONTENT_WIDTH / crop_width,
        RAW_TILE_CONTENT_HEIGHT / crop_height,
        1.0,
    )
    resized_width = max(1, min(RAW_TILE_CONTENT_WIDTH, int(round(crop_width * scale))))
    resized_height = max(1, min(RAW_TILE_CONTENT_HEIGHT, int(round(crop_height * scale))))
    rgb_image = Image.fromarray(crop_rgb, mode="RGB")
    mask_image = Image.fromarray(np.where(crop_mask, 255, 0).astype(np.uint8), mode="L")
    rgb_resized = np.asarray(
        rgb_image.resize((resized_width, resized_height), _resampling("BILINEAR")),
        dtype=np.uint8,
    )
    mask_resized = np.asarray(
        mask_image.resize((resized_width, resized_height), _resampling("NEAREST")),
        dtype=np.uint8,
    )
    content = np.full(
        (RAW_TILE_CONTENT_HEIGHT, RAW_TILE_CONTENT_WIDTH, 3),
        RAW_SHEET_NEUTRAL,
        dtype=np.uint8,
    )
    left = (RAW_TILE_CONTENT_WIDTH - resized_width) // 2
    top = (RAW_TILE_CONTENT_HEIGHT - resized_height) // 2
    base = rgb_resized.copy()
    content[top : top + resized_height, left : left + resized_width] = base
    content_mask = np.zeros((RAW_TILE_CONTENT_HEIGHT, RAW_TILE_CONTENT_WIDTH), dtype=bool)
    content_mask[top : top + resized_height, left : left + resized_width] = mask_resized > 0
    return content, content_mask


def _render_tile(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    record: Mapping[str, Any],
    candidate_id: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    content, content_mask = _letterboxed_tile(image_rgb, mask, bbox)
    color = np.asarray(candidate_color(candidate_id), dtype=np.float64)
    selected = content_mask[..., None]
    composited = np.rint(
        content.astype(np.float64) * (1.0 - RAW_MASK_ALPHA) + color * RAW_MASK_ALPHA
    ).astype(np.uint8)
    tile = np.full(
        (RAW_TILE_CONTENT_HEIGHT + RAW_TILE_LABEL_HEIGHT, RAW_TILE_CONTENT_WIDTH, 3),
        RAW_SHEET_NEUTRAL,
        dtype=np.uint8,
    )
    tile[:RAW_TILE_CONTENT_HEIGHT] = np.where(selected, composited, content)
    tile[RAW_TILE_CONTENT_HEIGHT:] = RAW_LABEL_BACKGROUND
    pil_tile = Image.fromarray(tile, mode="RGB")
    draw = ImageDraw.Draw(pil_tile)
    font = ImageFont.load_default()
    label = _candidate_label(candidate_id, record)
    available_width = RAW_TILE_CONTENT_WIDTH - 8
    while draw.textbbox((0, 0), label, font=font)[2] > available_width and label:
        label = label[:-1]
    bounds = draw.textbbox((0, 0), label, font=font)
    text_height = bounds[3] - bounds[1]
    text_y = RAW_TILE_CONTENT_HEIGHT + (RAW_TILE_LABEL_HEIGHT - text_height) // 2 - bounds[1]
    draw.text((4, text_y), label, fill=RAW_LABEL_FOREGROUND, font=font)
    return np.asarray(pil_tile, dtype=np.uint8)


def diagnostic_dimensions(width: int, height: int) -> tuple[int, int]:
    """Return deterministic nearest-neighbor diagnostic dimensions."""

    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("diagnostic dimensions must be positive integers")
    pixels = width * height
    if pixels <= RAW_MAX_DIAGNOSTIC_PIXELS:
        return width, height
    scale = math.sqrt(RAW_MAX_DIAGNOSTIC_PIXELS / pixels)
    output_width = max(1, min(width, int(math.floor(width * scale))))
    output_height = max(1, min(height, int(math.floor(height * scale))))
    while output_width * output_height > RAW_MAX_DIAGNOSTIC_PIXELS:
        if output_width >= output_height and output_width > 1:
            output_width -= 1
        elif output_height > 1:
            output_height -= 1
        else:  # pragma: no cover - the loop cannot reach this for a valid limit
            break
    return output_width, output_height


def raw_sam2_debug_rgb_bytes(width: int, height: int) -> int:
    """Return the exact fixed worst-case RGB-array reservation for one request."""

    diagnostic_width, diagnostic_height = diagnostic_dimensions(width, height)
    sheets = RAW_MAXIMUM_CONTACT_SHEETS * RAW_CONTACT_SHEET_HEIGHT * RAW_CONTACT_SHEET_WIDTH * 3
    diagnostics = 3 * diagnostic_width * diagnostic_height * 3
    return sheets + diagnostics


def _nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[1] == width and array.shape[0] == height:
        return np.asarray(array, dtype=np.uint8).copy()
    return np.asarray(
        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").resize(
            (width, height), _resampling("NEAREST")
        ),
        dtype=np.uint8,
    )


def _heatmap(overlap: np.ndarray) -> np.ndarray:
    maximum = int(overlap.max()) if overlap.size else 0
    output = np.zeros((*overlap.shape, 3), dtype=np.uint8)
    if maximum:
        fraction = overlap.astype(np.float64) / maximum
        output[..., 0] = np.rint(255 * fraction).astype(np.uint8)
        output[..., 1] = np.rint(96 * fraction).astype(np.uint8)
        output[..., 2] = np.rint(255 * (1.0 - fraction)).astype(np.uint8)
    return output


def render_raw_sam2_visualizations(
    image_rgb: np.ndarray, masks: Sequence[Mapping[str, Any]]
) -> RawSam2Visualization:
    """Render fixed, bounded raw SAM2 candidate and coverage artifacts."""

    _require_pillow()
    source = _as_rgb(image_rgb)
    height, width = source.shape[:2]
    records = _candidate_records(masks, height=height, width=width)
    overlap = np.zeros((height, width), dtype=np.uint32)
    for _, mask, _, _ in records:
        np.add(overlap, mask.astype(np.uint32), out=overlap, casting="unsafe")

    maximum_overlap = int(overlap.max()) if overlap.size else 0
    covered = overlap > 0
    covered_count = int(np.count_nonzero(covered))
    uncovered_count = int(covered.size - covered_count)
    histogram_limit = min(maximum_overlap, 255)
    histogram = {
        str(count): int(np.count_nonzero(overlap == count)) for count in range(histogram_limit + 1)
    }
    overflow_count = int(np.count_nonzero(overlap >= 256))
    diagnostic_width, diagnostic_height = diagnostic_dimensions(width, height)
    union_source = np.repeat(np.where(covered[..., None], 255, 0).astype(np.uint8), 3, axis=2)
    uncovered_source = np.repeat(
        np.where((~covered)[..., None], 255, 0).astype(np.uint8), 3, axis=2
    )
    union = _nearest(union_source, diagnostic_width, diagnostic_height)
    heatmap = _nearest(_heatmap(overlap), diagnostic_width, diagnostic_height)
    uncovered = _nearest(uncovered_source, diagnostic_width, diagnostic_height)

    represented = records[:RAW_MAXIMUM_REPRESENTED_CANDIDATES]
    truncated_count = len(records) - len(represented)
    page_count = math.ceil(len(represented) / RAW_CANDIDATES_PER_SHEET)
    artifacts: list[tuple[str, np.ndarray]] = []
    for page in range(page_count):
        sheet = np.full(
            (RAW_CONTACT_SHEET_HEIGHT, RAW_CONTACT_SHEET_WIDTH, 3),
            RAW_SHEET_NEUTRAL,
            dtype=np.uint8,
        )
        for tile_index, (candidate_id, mask, record, bbox) in enumerate(
            represented[page * RAW_CANDIDATES_PER_SHEET : (page + 1) * RAW_CANDIDATES_PER_SHEET]
        ):
            tile = _render_tile(source, mask, record, candidate_id, bbox)
            row, column = divmod(tile_index, RAW_CONTACT_SHEET_COLUMNS)
            y0 = row * (RAW_TILE_CONTENT_HEIGHT + RAW_TILE_LABEL_HEIGHT)
            x0 = column * RAW_TILE_CONTENT_WIDTH
            sheet[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
        artifacts.append((f"sam2-candidates-page-{page + 1:04d}.png", sheet))
    artifacts.extend(
        (
            (RAW_UNION_NAME, union),
            (RAW_OVERLAP_NAME, heatmap),
            (RAW_UNCOVERED_NAME, uncovered),
        )
    )
    summary = {
        "enabled": True,
        "candidate_id_base": RAW_CANDIDATE_ID_BASE,
        "visualizable_candidate_count": len(records),
        "represented_candidate_count": len(represented),
        "represented_candidate_ids": [item[0] for item in represented],
        "truncated_candidate_count": truncated_count,
        "contact_sheet_count": page_count,
        "covered_pixel_count": covered_count,
        "uncovered_pixel_count": uncovered_count,
        "max_overlap_count": maximum_overlap,
        "overlap_histogram": histogram,
        "overlap_histogram_overflow_pixel_count": overflow_count,
        "overlap_histogram_truncated": maximum_overlap > 255,
        "source_dimensions": {"width": width, "height": height},
        "diagnostic_dimensions": {
            "width": diagnostic_width,
            "height": diagnostic_height,
        },
        "artifact_names": [name for name, _ in artifacts],
        "warnings": [RAW_TRUNCATION_WARNING] if truncated_count else [],
    }
    return RawSam2Visualization(artifacts=tuple(artifacts), summary=summary)
