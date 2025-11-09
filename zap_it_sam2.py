"""
zap-it-sam2.py

Contains the functions directly handling SAM2-based segmentation. Currently we
only expose the single-pass helper used throughout the pipeline.
"""

def process_single_pass(image_np, mask_generator, alpha, verbosity=1, log_print_func=None):
    """
    Calls mask_generator.generate(image_np) once (no tiling). Returns list of mask dicts:
      each dict => {"segmentation", "area", "predicted_iou", "stability_score", ...}

    :param image_np: The (H,W,3) image array (uint8).
    :param mask_generator: A SAM2AutomaticMaskGenerator instance.
    :param alpha: (not actually used here, but kept for signature parity).
    :param verbosity: 0..2
    :param log_print_func: optional logging function to show messages.
    """
    if log_print_func and verbosity >= 2:
        log_print_func("[process_single_pass] Generating masks (single pass)...", 2, verbosity)
    all_masks = mask_generator.generate(image_np)
    if log_print_func and verbosity >= 2:
        log_print_func(f"[process_single_pass] => got {len(all_masks)} masks total.", 2, verbosity)
    return all_masks

__all__ = ["process_single_pass"]
