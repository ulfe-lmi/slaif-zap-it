"""SAM2-based segmentation module with unified interface."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np


# This is the intentionally small public SAM2 surface.  The service validator
# owns the untrusted-input policy; this module keeps the constructor seam
# equally fail-closed so a caller cannot accidentally forward arbitrary YAML
# mappings to the upstream class.
SAM2_GENERATOR_FIELDS = (
    "points_per_side",
    "points_per_batch",
    "pred_iou_thresh",
    "stability_score_thresh",
    "stability_score_offset",
    "mask_threshold",
    "box_nms_thresh",
    "crop_n_layers",
    "crop_nms_thresh",
    "crop_overlap_ratio",
    "crop_n_points_downscale_factor",
    "min_mask_region_area",
    "use_m2m",
    "multimask_output",
)

SAM2_DEFAULTS = {
    "points_per_side": 8,
    "points_per_batch": 8,
    "pred_iou_thresh": 0.5,
    "stability_score_thresh": 0.5,
    "stability_score_offset": 1.0,
    "mask_threshold": 0.0,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 512 / 1500,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 0,
    "use_m2m": False,
    "multimask_output": True,
}

SAM2_PROFILES = {
    "fast": {
        "points_per_side": 8,
        "points_per_batch": 8,
        "pred_iou_thresh": 0.5,
        "stability_score_thresh": 0.5,
        "crop_n_layers": 0,
    },
    "balanced": {
        "points_per_side": 16,
        "points_per_batch": 16,
        "pred_iou_thresh": 0.7,
        "stability_score_thresh": 0.8,
        "crop_n_layers": 0,
    },
    "quality": {
        "points_per_side": 32,
        "points_per_batch": 32,
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.85,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 50,
        "multimask_output": True,
    },
}


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


def _generator_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only explicitly supplied safe constructor scalar values."""
    return {key: config[key] for key in SAM2_GENERATOR_FIELDS if key in config}


def estimated_prompt_count(
    points_per_side: int, crop_n_layers: int, crop_n_points_downscale_factor: int
) -> int:
    """Estimate upstream prompt points using SAM2's crop/grid construction."""
    return sum(
        4**layer * int(points_per_side / (crop_n_points_downscale_factor**layer)) ** 2
        for layer in range(crop_n_layers + 1)
    )


def _default_generator_factory(model: Any, **kwargs: Any) -> Any:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    return SAM2AutomaticMaskGenerator(model, **kwargs)


def build_request_generator(
    model: Any,
    config: Mapping[str, Any],
    *,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Construct one request-local generator around an already resident model.

    ``point_grids`` and ``output_mode`` are fixed operator controls.  The
    request mapping is filtered a second time at this boundary, which keeps
    this seam safe even when used without the HTTP validator in a test or
    future adapter.
    """
    kwargs = _generator_kwargs(config)
    kwargs.update(point_grids=None, output_mode="binary_mask")
    return (factory or _default_generator_factory)(model, **kwargs)


def initialize(
    config: Dict[str, Any],
    *,
    dryrun: bool = False,
    device=None,
    verbosity: int = 1,
    log_print_func=None,
    local_files_only: bool = False,
    model_only: bool = False,
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
    revision = config.get("revision")
    if revision:
        # The upstream helper does not expose ``revision``.  Keep the model
        # identity pinned for operator qualification without changing the
        # legacy unpinned path used by existing CLI configurations.
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2

        config_file = config.get("config_file", "configs/sam2/sam2_hiera_l.yaml")
        checkpoint_name = config.get("checkpoint_name", "sam2_hiera_large.pt")
        checkpoint = hf_hub_download(
            repo_id=model_name,
            filename=checkpoint_name,
            revision=str(revision),
            local_files_only=local_files_only,
        )
        model = build_sam2(config_file=config_file, ckpt_path=checkpoint, device=target_device)
    else:
        model = build_sam2_hf(model_name, device=target_device)
    model.eval().to(target_device)
    if str(config.get("dtype", "auto")).lower() == "float16" and str(target_device).startswith(
        "cuda"
    ):
        model.half()

    if model_only:
        # The live service deliberately keeps only this pinned model object in
        # its resident registry.  Request generator state must never become a
        # resident holder or be written back into this dictionary.
        return {"model": model}

    # Legacy CLI callers retain their historical configured generator.  The
    # service uses ``model_only=True`` and the request seam below instead.
    generator_kwargs = _generator_kwargs(config)

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
    request_local_generator = False
    if mask_generator is None and state.get("model") is not None:
        mask_generator = build_request_generator(
            state["model"],
            params.get("mask_generator_config", {}),
            factory=state.get("generator_factory"),
        )
        request_local_generator = True
    if mask_generator is None:
        mask_generator = params.get("mask_generator")
        if mask_generator is not None and not request_local_generator:
            state["mask_generator"] = mask_generator

    if mask_generator is None:
        raise ValueError("SAM2 segmenter requires a 'mask_generator' instance in state or params")

    alpha = params.get("alpha")

    # Support callers passing a list/tuple of images by taking the first item.
    image_np = images[0] if isinstance(images, (list, tuple)) else images

    def generate_masks():
        if dryrun_mode:
            log("[segmenter.sam2] Generating dry-run masks...", 2, verbosity)
            if not isinstance(mask_generator, _DryRunMaskGenerator):
                return _DryRunMaskGenerator().generate(image_np)
            return mask_generator.generate(image_np)

        # SAM2's image transform returns float32 tensors.  The operator's
        # all-resident profile deliberately pins the model to FP16, so use the
        # framework autocast boundary to make those activations compatible
        # without mutating request data or model residency.
        model = getattr(getattr(mask_generator, "predictor", None), "model", None)
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                parameter = next(iter(parameters()))
            except (StopIteration, TypeError, RuntimeError):
                parameter = None
            dtype = getattr(parameter, "dtype", None)
            device = getattr(parameter, "device", None)
            if str(dtype) == "torch.float16" and str(device).startswith("cuda"):
                import torch

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return mask_generator.generate(image_np)
        log("[segmenter.sam2] Generating masks...", 2, verbosity)
        return mask_generator.generate(image_np)

    if dryrun_mode:
        log("[segmenter.sam2] Generating dry-run masks...", 2, verbosity)
        # Ensure we are using the lightweight generator.
        if not isinstance(mask_generator, _DryRunMaskGenerator):
            mask_generator = _DryRunMaskGenerator()
        masks = generate_masks()
    else:
        masks = generate_masks()

    log(f"[segmenter.sam2] => produced {len(masks)} masks", 2, verbosity)

    meta = {
        "alpha": alpha,
        "num_masks": len(masks),
    }
    meta.update(params.get("extra_meta", {}))

    return state, masks, meta


__all__ = [
    "SAM2_DEFAULTS",
    "SAM2_GENERATOR_FIELDS",
    "SAM2_PROFILES",
    "build_request_generator",
    "estimated_prompt_count",
    "initialize",
    "run",
]
