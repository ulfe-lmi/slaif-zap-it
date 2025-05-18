"""
zap-it-sam2.py

Contains the functions directly handling SAM2-based segmentation, such as
single-pass vs. tiled approach and the helper for computing tile positions.
"""

import numpy as np

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

def compute_tile_positions(H, W, tile_size, overlap):
    """
    Returns a list of (x0,y0,x1,y1) for overlapping tiles.
    overlap=0.2 => each tile overlaps next by 20% in each dimension.
    """
    step = int(tile_size * (1 - overlap))
    positions = []
    y = 0
    while y < H:
        x = 0
        y_end = min(y + tile_size, H)
        while x < W:
            x_end = min(x + tile_size, W)
            positions.append((x, y, x_end, y_end))
            x += step
        y += step
    return positions

def process_tiled(image_np, mask_generator, alpha, tile_size, overlap, verbosity=1, log_print_func=None):
    """
    Splits the image into overlapping tiles => run SAM2 => merges partial masks
    into global coords => returns one big list of dicts: {"segmentation", "area", ...}

    :param image_np: (H,W,3) image array
    :param mask_generator: The SAM2 generator
    :param alpha: not directly used for mask generation (kept for signature consistency)
    :param tile_size: integer tile dimension
    :param overlap: fraction of overlap (0..1)
    :param verbosity: 0..2
    :param log_print_func: optional function for printing logs
    """
    H, W = image_np.shape[:2]
    pos = compute_tile_positions(H, W, tile_size, overlap)
    all_masks = []
    if log_print_func and verbosity >= 1:
        log_print_func(f"[process_tiled] Tiling => #tiles={len(pos)}", 1, verbosity)

    for i, (x0, y0, x1, y1) in enumerate(pos, start=1):
        tile_sub = image_np[y0:y1, x0:x1, :]
        tile_masks = mask_generator.generate(tile_sub)
        for m in tile_masks:
            seg_t = m["segmentation"]
            seg_global = np.zeros((H, W), dtype=bool)
            seg_global[y0:y0+seg_t.shape[0], x0:x0+seg_t.shape[1]] = seg_t
            all_masks.append({
                "segmentation": seg_global,
                "area": seg_global.sum(),
                "predicted_iou": m["predicted_iou"],
                "stability_score": m["stability_score"]
            })

    if log_print_func and verbosity >= 1:
        log_print_func(f"[process_tiled] => total {len(all_masks)} masks after merging all tiles.", 1, verbosity)
    return all_masks
