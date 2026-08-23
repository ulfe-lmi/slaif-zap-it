"""SAM2-based segmentation module with unified interface."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class _DryRunMaskGenerator:
    """Lightweight stand-in for ``SAM2AutomaticMaskGenerator``."""

    def generate(self, image):
        image_np = image[0] if isinstance(image, (list, tuple)) else image
        height, width = image_np.shape[:2]
        rows, cols = 3, 4
        masks = []

        row_edges = np.linspace(0, height, rows + 1, dtype=int)
        col_edges = np.linspace(0, width, cols + 1, dtype=int)

        counter = 0
        for r in range(rows):
            for c in range(cols):
                counter += 1
                y0, y1 = row_edges[r], row_edges[r + 1]
                x0, x1 = col_edges[c], col_edges[c + 1]
                seg = np.zeros((height, width), dtype=bool)
                seg[y0:y1, x0:x1] = True
                masks.append(
                    {
                        "segmentation": seg,
                        "area": int(seg.sum()),
                        "predicted_iou": 1.0,
                        "stability_score": 1.0,
                        "bbox": [int(x0), int(y0), int(x1), int(y1)],
                        "dryrun_id": counter,
                    }
                )

        return masks


def initialize(
    config: Dict[str, Any],
    *,
    dryrun: bool = False,
    device=None,
    verbosity: int = 1,
    log_print_func=None,
) -> Dict[str, Any]:
    """Prepare the SAM2 runtime objects and return the initial state."""

    log = log_print_func or (lambda *a, **k: None)

    if dryrun:
        log("[segmenter.sam2] Initializing dry-run mask generator", 1, verbosity)
        return {"mask_generator": _DryRunMaskGenerator()}

    log("[segmenter.sam2] Building SAM2 model...", 1, verbosity)

    # Local imports so dry-run mode avoids pulling heavy dependencies.
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    target_device = device
    if target_device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model_name", "facebook/sam2-hiera-large")
    model = build_sam2_hf(model_name)
    model.eval().to(target_device)

    generator_kwargs = {
        "points_per_side": config.get("points_per_side"),
        "pred_iou_thresh": config.get("pred_iou_thresh"),
        "stability_score_thresh": config.get("stability_score_thresh"),
        "min_mask_region_area": config.get("min_mask_region_area"),
        "crop_n_layers": config.get("crop_n_layers"),
        "crop_n_points_downscale_factor": config.get("crop_n_points_downscale_factor"),
        "crop_overlap_ratio": config.get("crop_overlap_ratio"),
        "box_nms_thresh": config.get("box_nms_thresh"),
        "multimask_output": config.get("multimask_output"),
    }

    mask_generator = SAM2AutomaticMaskGenerator(model, **generator_kwargs)
    return {"mask_generator": mask_generator}


def run(
    state: Dict[str, Any] | None,
    params: Dict[str, Any],
    images,
    *,
    verbosity: int = 1,
    log_print_func=None,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run SAM2 segmentation using the unified module interface."""

    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    dryrun_mode = bool(params.get("dryrun", False))

    mask_generator = state.get("mask_generator")
    if mask_generator is None:
        mask_generator = params.get("mask_generator")
        if mask_generator is not None:
            state["mask_generator"] = mask_generator

    if mask_generator is None:
        raise ValueError("SAM2 segmenter requires a 'mask_generator' instance in state or params")

    alpha = params.get("alpha")

    # Support callers passing a list/tuple of images by taking the first item.
    image_np = images[0] if isinstance(images, (list, tuple)) else images

    if dryrun_mode:
        log("[segmenter.sam2] Generating dry-run masks...", 2, verbosity)
        # Ensure we are using the lightweight generator.
        if not isinstance(mask_generator, _DryRunMaskGenerator):
            mask_generator = _DryRunMaskGenerator()
            state["mask_generator"] = mask_generator
        masks = mask_generator.generate(image_np)
    else:
        log("[segmenter.sam2] Generating masks...", 2, verbosity)
        masks = mask_generator.generate(image_np)

    log(f"[segmenter.sam2] => produced {len(masks)} masks", 2, verbosity)

    meta = {
        "alpha": alpha,
        "num_masks": len(masks),
    }
    meta.update(params.get("extra_meta", {}))

    return state, masks, meta


__all__ = ["initialize", "run"]
