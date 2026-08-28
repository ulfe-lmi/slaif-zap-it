"""Visualization helpers and pipeline orchestration for ZAP-IT."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "render_annotated",
    "build_2x2_composite",
    "build_composite_for_masks",
    "build_panoptic_final",
    "sanitize_visualization_label",
    "render_annotated_labelled",
    "generate_visualizations",
]


def render_annotated(
    image_np: np.ndarray, masks: Iterable[Dict[str, Any]], alpha: float = 0.5
) -> np.ndarray:
    """Return an alpha-blended overlay of ``masks`` on top of ``image_np``."""
    out = image_np.astype(np.float32, copy=True)
    sorted_masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    for color_index, ann in enumerate(sorted_masks):
        seg = ann.get("segmentation")
        if seg is None or not np.any(seg):
            continue
        # Stable palette: request A/B/A output must not differ merely because
        # NumPy's process-global RNG advanced between calls.
        color = np.asarray(
            (
                (53 + color_index * 97) % 256,
                (109 + color_index * 67) % 256,
                (191 + color_index * 31) % 256,
            ),
            dtype=np.uint8,
        )
        out[seg, :] = alpha * color + (1.0 - alpha) * out[seg, :]

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


_SAFE_LABEL_CHARS = frozenset(" _-.+")
_LABEL_REPLACEMENT = "_"
_LABEL_LIMIT = 48
_LABEL_SOURCE_LIMIT = 1024
_LABEL_PADDING = 2
_LABEL_MARGIN = 2


def sanitize_visualization_label(label: Any) -> str:
    """Return the bounded, display-only form of a final object label.

    Structured result metadata is never changed by this helper.  The visible
    form is deliberately ASCII-only so model-produced control characters,
    separators and other Unicode text cannot become confusing path-like or
    terminal-like output in a diagnostic image.
    """
    raw = "" if label is None else str(label)[:_LABEL_SOURCE_LIMIT]
    normalized = unicodedata.normalize("NFKC", raw)
    safe: list[str] = []
    for character in normalized:
        codepoint = ord(character)
        if codepoint < 32 or codepoint == 127 or character in {"/", "\\"}:
            replacement = _LABEL_REPLACEMENT
        elif character.isspace():
            replacement = " "
        elif character.isascii() and (character.isalnum() or character in _SAFE_LABEL_CHARS):
            replacement = character
        else:
            replacement = _LABEL_REPLACEMENT
        if replacement == _LABEL_REPLACEMENT and safe and safe[-1] == _LABEL_REPLACEMENT:
            continue
        safe.append(replacement)

    result = re.sub(r" +", " ", "".join(safe)).strip()
    result = re.sub(r"_+", "_", result)
    return result[:_LABEL_LIMIT] or "unknown"


def _object_value(obj: Any, name: str, default: Any = None) -> Any:
    """Read one ObjectResult value without importing the core module here."""
    value = getattr(obj, name, default)
    if value is not default:
        return value
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _object_mask(obj: Any) -> np.ndarray:
    mask = _object_value(obj, "mask")
    if mask is None and isinstance(obj, dict):
        mask = obj.get("segmentation")
    return np.asarray(mask, dtype=bool)


def _label_text(obj: Any, *, show_confidence: bool) -> str:
    label = sanitize_visualization_label(_object_value(obj, "label"))
    instance_id = int(_object_value(obj, "instance_id", 0))
    text = f"{label} {instance_id}"
    if show_confidence:
        score = _object_value(obj, "clip_score")
        try:
            finite_score = float(score)
        except (TypeError, ValueError):
            finite_score = math.nan
        if math.isfinite(finite_score):
            text += f"   CLIP {finite_score:.2f}"
    return text


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[int, int, tuple[int, int, int, int]]:
    bounds = draw.textbbox((0, 0), text, font=font)
    return max(int(bounds[2] - bounds[0]), 1), max(int(bounds[3] - bounds[1]), 1), bounds


def _fit_label_text(
    draw: ImageDraw.ImageDraw,
    obj: Any,
    *,
    show_confidence: bool,
    max_width: int,
    font: ImageFont.ImageFont,
) -> tuple[str, int, int, tuple[int, int, int, int]]:
    """Shorten only the visible label first, retaining the ID and score."""
    original = sanitize_visualization_label(_object_value(obj, "label"))[:_LABEL_LIMIT]
    instance_id = int(_object_value(obj, "instance_id", 0))
    suffix = f" {instance_id}"
    if show_confidence:
        score = _object_value(obj, "clip_score")
        try:
            finite_score = float(score)
        except (TypeError, ValueError):
            finite_score = math.nan
        if math.isfinite(finite_score):
            suffix += f"   CLIP {finite_score:.2f}"

    for length in range(len(original), -1, -1):
        text = f"{original[:length]}{suffix}"
        width, height, bounds = _text_size(draw, text, font)
        if width <= max_width:
            return text, width, height, bounds

    # A very small image cannot fit even the required instance suffix with
    # Pillow's bitmap font.  Keep the bounded rectangle and let Pillow clip the
    # final glyphs; ordinary service-sized images always take the branch above.
    text = f"{original}{suffix}"
    width, height, bounds = _text_size(draw, text, font)
    return text, width, height, bounds


def _clamp_label_box(
    left: float, top: float, box_width: int, box_height: int, width: int, height: int
) -> tuple[int, int, int, int]:
    bounded_width = min(max(int(box_width), 1), width)
    bounded_height = min(max(int(box_height), 1), height)
    max_left = max(width - bounded_width, 0)
    max_top = max(height - bounded_height, 0)
    bounded_left = min(max(int(round(left)), 0), max_left)
    bounded_top = min(max(int(round(top)), 0), max_top)
    return bounded_left, bounded_top, bounded_left + bounded_width, bounded_top + bounded_height


def _intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _label_candidate_boxes(
    bbox: tuple[int, int, int, int],
    centroid: tuple[float, float],
    box_width: int,
    box_height: int,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], ...]:
    x_min, y_min, x_max, y_max = bbox
    centroid_row, centroid_column = centroid
    centered_left = centroid_column - box_width / 2.0
    centered_top = centroid_row - box_height / 2.0
    candidates = (
        (centered_left, y_min - box_height - _LABEL_MARGIN),
        (centered_left, y_max + 1 + _LABEL_MARGIN),
        (centered_left, centered_top),
        (x_min - box_width - _LABEL_MARGIN, centered_top),
        (x_max + 1 + _LABEL_MARGIN, centered_top),
    )
    return tuple(
        _clamp_label_box(left, top, box_width, box_height, width, height)
        for left, top in candidates
    )


def _object_bbox_and_centroid(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[float, float]]:
    rows, columns = np.nonzero(mask)
    return (
        (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())),
        (float(rows.mean()), float(columns.mean())),
    )


def render_annotated_labelled(
    image_np: np.ndarray,
    final_objects: Sequence[Any],
    *,
    alpha: float = 0.5,
    show_confidence: bool = False,
) -> np.ndarray:
    """Render the deterministic final-object overlay with visible labels.

    ``final_objects`` must be the already ordered ``ObjectResult`` sequence.
    This renderer only reads objects and masks; it never changes structured
    metadata or source arrays.  Label rectangles are chosen greedily from a
    fixed above/below/inside/left/right sequence and are always clamped to the
    original image.
    """
    if not isinstance(show_confidence, bool):
        raise ValueError("show_confidence must be a boolean")
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("labelled visualization requires an RGB image")
    height, width = image_np.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("labelled visualization requires a non-empty image")

    objects = tuple(final_objects)
    overlay_masks = []
    masks: list[np.ndarray] = []
    for obj in objects:
        mask = _object_mask(obj)
        if mask.shape != (height, width):
            raise ValueError("final object mask does not match the image dimensions")
        masks.append(mask)
        overlay_masks.append({"segmentation": mask, "area": int(np.count_nonzero(mask))})

    output = render_annotated(image_np, overlay_masks, alpha=alpha)
    canvas = Image.fromarray(output, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    occupied: list[tuple[int, int, int, int]] = []

    for object_index, (obj, mask) in enumerate(zip(objects, masks)):
        if not np.any(mask):
            continue
        bbox, centroid = _object_bbox_and_centroid(mask)
        max_text_width = max(width - 2 * _LABEL_PADDING, 1)
        text, text_width, text_height, text_bounds = _fit_label_text(
            draw,
            obj,
            show_confidence=show_confidence,
            max_width=max_text_width,
            font=font,
        )
        box_width = min(width, text_width + 2 * _LABEL_PADDING)
        box_height = min(height, text_height + 2 * _LABEL_PADDING)
        candidates = _label_candidate_boxes(
            bbox,
            centroid,
            box_width,
            box_height,
            width=width,
            height=height,
        )
        selected = candidates[0]
        selected_overlap = None
        for candidate_index, candidate in enumerate(candidates):
            overlap = sum(_intersection_area(candidate, prior) for prior in occupied)
            if overlap == 0:
                selected = candidate
                selected_overlap = 0
                break
            score = (overlap, candidate_index)
            if selected_overlap is None or score < selected_overlap:
                selected = candidate
                selected_overlap = score
        occupied.append(selected)

        # The overlay palette is stable by final object order; darkening that
        # color creates a high-contrast bounded label swatch without any input
        # text or user-controlled visual resource.
        color = np.asarray(
            (
                (53 + object_index * 97) % 256,
                (109 + object_index * 67) % 256,
                (191 + object_index * 31) % 256,
            ),
            dtype=np.uint8,
        )
        background = tuple(int(channel * 0.65) for channel in color)
        luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
        foreground = (0, 0, 0) if luminance >= 140 else (255, 255, 255)
        left, top, right, bottom = selected
        draw.rectangle((left, top, right - 1, bottom - 1), fill=background)
        text_left = left + _LABEL_PADDING - text_bounds[0]
        text_top = top + _LABEL_PADDING - text_bounds[1]
        draw.text((text_left, text_top), text, font=font, fill=foreground)

    return np.asarray(canvas, dtype=np.uint8).copy()


def build_2x2_composite(
    base_np: np.ndarray,
    annotated_np: np.ndarray,
    mask_rand_np: np.ndarray,
    masked_np: np.ndarray,
) -> np.ndarray:
    """Build a 2×2 composite from the supplied quadrants."""
    h, w = base_np.shape[:2]

    def rez(img: np.ndarray) -> Image.Image:
        return Image.fromarray(img).resize((w, h), Image.Resampling.LANCZOS)

    tl = np.array(rez(base_np))
    tr = np.array(rez(annotated_np))
    bl = np.array(rez(mask_rand_np))
    br = np.array(rez(masked_np))
    top = np.hstack((tl, tr))
    bottom = np.hstack((bl, br))
    return np.vstack((top, bottom))


def build_composite_for_masks(
    orig_np: np.ndarray,
    mask_list: Iterable[Dict[str, Any]],
    alpha: float,
    verbosity: int,
    *,
    log_print_func: Callable[[str, int, int], None] | None = None,
    return_extra: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Create a 2×2 composite for ``mask_list`` overlays."""
    log = log_print_func or (lambda *a, **k: None)
    log("  => [build_composite_for_masks] building annotated overlay...", 2, verbosity)
    annotated = render_annotated(orig_np, mask_list, alpha=alpha)

    log("  => [build_composite_for_masks] building random color + masked array...", 2, verbosity)

    mask_list = list(mask_list)
    if mask_list:
        stack_pre = np.stack([m["segmentation"] for m in mask_list], axis=0)
        combined_pre = np.any(stack_pre, axis=0)
        mask_rand_pre = np.zeros_like(orig_np)
        for seg in stack_pre:
            color = np.random.randint(0, 255, 3)
            for channel in range(3):
                mask_rand_pre[..., channel][seg] = color[channel]
        masked_pre = np.zeros_like(orig_np)
        masked_pre[combined_pre] = orig_np[combined_pre]
    else:
        mask_rand_pre = np.zeros_like(orig_np)
        masked_pre = np.zeros_like(orig_np)

    log("  => [build_composite_for_masks] building 2x2 now...", 2, verbosity)
    composite_2x2 = build_2x2_composite(orig_np, annotated, mask_rand_pre, masked_pre)

    if return_extra:
        return composite_2x2, annotated
    return composite_2x2


def build_panoptic_final(orig_np: np.ndarray, final_masks: Iterable[Dict[str, Any]]) -> np.ndarray:
    """Create a detectron2-based panoptic overlay from ``final_masks``."""
    try:
        import torch
        from detectron2.data import Metadata
        from detectron2.structures import BitMasks, Instances
        from detectron2.utils.visualizer import ColorMode, Visualizer
    except ImportError as exc:
        raise RuntimeError(
            "panoptic visualization requires the optional detectron2 dependency"
        ) from exc

    final_masks = list(final_masks)
    if not final_masks:
        return orig_np

    height, width = orig_np.shape[:2]
    labels = [m.get("clip_label", "unknown") for m in final_masks]
    unique_labels = sorted(set(labels))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}

    meta = Metadata()
    meta.thing_classes = unique_labels

    instances = Instances((height, width))
    mask_tensors = []
    class_ids = []
    for mask in final_masks:
        seg_bool = mask.get("segmentation")
        if seg_bool is None or seg_bool.shape != (height, width):
            continue
        mask_tensors.append(torch.from_numpy(seg_bool))
        class_ids.append(label_to_id.get(mask.get("clip_label", "unknown"), 0))

    if not mask_tensors:
        return orig_np

    bitmasks = BitMasks(torch.stack(mask_tensors, dim=0))
    instances.pred_masks = bitmasks
    instances.pred_classes = torch.tensor(class_ids, dtype=torch.int64)

    visualizer = Visualizer(orig_np, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
    result = visualizer.draw_instance_predictions(instances)
    return result.get_image()


def _iter_stage_entries(vis_cfg: Dict[str, Any]) -> Iterable[tuple[str, Dict[str, Any]]]:
    for stage_name in ("sam2", "clip", "blip3"):
        entries = vis_cfg.get(stage_name, [])
        if not entries:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"visualization.{stage_name} must be a list of mappings")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"visualization.{stage_name} entries must be mappings")
            if "id" not in entry or "renderer" not in entry:
                raise ValueError(f"visualization.{stage_name} entries require 'id' and 'renderer'")
            yield stage_name, entry


def generate_visualizations(
    image_np: np.ndarray,
    masks_by_stage: Dict[str, Iterable[Dict[str, Any]]],
    vis_cfg: Dict[str, Any],
    *,
    default_alpha: float,
    verbosity: int = 1,
    log_print_func: Callable[[str, int, int], None] | None = None,
    final_objects: Sequence[Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Render configured streams, using final objects for labelled streams."""
    log = log_print_func or (lambda *a, **k: None)
    outputs: Dict[str, np.ndarray] = {}

    for stage_name, entry in _iter_stage_entries(vis_cfg):
        vis_id = entry["id"]
        renderer = entry["renderer"].lower()
        masks = list(masks_by_stage.get(stage_name, []))
        log(
            f"[visualizer] => rendering '{vis_id}' using stage '{stage_name}' and renderer '{renderer}'",
            2,
            verbosity,
        )

        if renderer in {"annotated", "alpha-overlay"}:
            alpha = float(entry.get("alpha", default_alpha))
            outputs[vis_id] = render_annotated(image_np, masks, alpha=alpha)
        elif renderer == "annotated-labelled":
            if stage_name != "blip3":
                raise ValueError(
                    "annotated-labelled visualization is only valid at the blip3 stage"
                )
            if final_objects is None:
                raise ValueError("annotated-labelled visualization requires final objects")
            alpha = float(entry.get("alpha", default_alpha))
            outputs[vis_id] = render_annotated_labelled(
                image_np,
                final_objects,
                alpha=alpha,
                show_confidence=entry.get("show_confidence", False),
            )
        elif renderer == "panoptic":
            outputs[vis_id] = build_panoptic_final(image_np, masks)
        else:
            raise ValueError(f"Unknown renderer '{renderer}' for visualization '{vis_id}'")

    return outputs
