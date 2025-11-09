"""SAM2-based segmentation module with unified interface."""
from __future__ import annotations
from typing import Any, Dict, Tuple


def run(state: Dict[str, Any] | None,
        params: Dict[str, Any],
        images,
        *,
        verbosity: int = 1,
        log_print_func=None) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run SAM2 segmentation using the unified module interface.

    Parameters
    ----------
    state:
        Mutable dictionary holding reusable objects. Expected to contain a
        ``"mask_generator"`` entry with a ``SAM2AutomaticMaskGenerator``
        instance. When ``state`` is ``None`` an empty dictionary is created and
        ``mask_generator`` is pulled from ``params``.
    params:
        Dictionary with configuration values. Must provide ``"mask_generator"``
        when the state does not yet contain one. Optional keys:
        ``"alpha"`` (kept for compatibility) and ``"extra_meta"`` for any
        caller-provided metadata.
    images:
        A single RGB image represented as a NumPy array or a batch of such
        images. Only the first image is processed when a batch is provided.

    Returns
    -------
    tuple
        ``(state, masks, metaresult)``
    """
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    mask_generator = state.get("mask_generator")
    if mask_generator is None:
        mask_generator = params.get("mask_generator")
        if mask_generator is None:
            raise ValueError("SAM2 segmenter requires a 'mask_generator' instance in state or params")
        state["mask_generator"] = mask_generator

    alpha = params.get("alpha")

    # Support callers passing a list/tuple of images by taking the first item.
    image_np = images[0] if isinstance(images, (list, tuple)) else images

    log("[segmenter.sam2] Generating masks...", 2, verbosity)
    masks = mask_generator.generate(image_np)
    log(f"[segmenter.sam2] => produced {len(masks)} masks", 2, verbosity)

    meta = {
        "alpha": alpha,
        "num_masks": len(masks),
    }
    meta.update(params.get("extra_meta", {}))

    return state, masks, meta


__all__ = ["run"]
