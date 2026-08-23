"""Visualization helpers and pipeline orchestration for ZAP-IT."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

import numpy as np
from PIL import Image

__all__ = [
    "render_annotated",
    "build_2x2_composite",
    "build_composite_for_masks",
    "build_panoptic_final",
    "generate_visualizations",
]


def render_annotated(
    image_np: np.ndarray, masks: Iterable[Dict[str, Any]], alpha: float = 0.5
) -> np.ndarray:
    """Return an alpha-blended overlay of ``masks`` on top of ``image_np``."""
    out = image_np.astype(np.float32, copy=True)
    sorted_masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    for ann in sorted_masks:
        seg = ann.get("segmentation")
        if seg is None or not np.any(seg):
            continue
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        out[seg, :] = alpha * color + (1.0 - alpha) * out[seg, :]

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


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
) -> Dict[str, np.ndarray]:
    """Render all configured visualization streams and return them as an ordered dict."""
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

        if renderer == "alpha-overlay":
            alpha = float(entry.get("alpha", default_alpha))
            outputs[vis_id] = render_annotated(image_np, masks, alpha=alpha)
        elif renderer == "panoptic":
            outputs[vis_id] = build_panoptic_final(image_np, masks)
        else:
            raise ValueError(f"Unknown renderer '{renderer}' for visualization '{vis_id}'")

    return outputs
