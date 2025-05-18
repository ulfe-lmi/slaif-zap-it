"""
zap_it_geometry.py

Provides a 'run_geometry_on_masks' function that:
  - For each final 'positive' mask => run Canny & Hough
  - Saves lines & intersections to TSV (one set of TSV files per mask)
  - Optionally writes out the raw Canny edge image if geometry.debug==true
  - Returns a list of geometry data for each mask, so we can draw them later.

If "geometry" is absent in the YAML config, the main pipeline won't call this.
"""

import os
import numpy as np
import cv2

def run_geometry_on_masks(final_masks, geometry_cfg, out_dir, base_filename,
                          orig_shape, log_print_func=None, verbosity=1):
    """
    For each mask in final_masks => run geometry analysis if it's a positive category mask.
      - Convert bool segmentation => 8-bit
      - Canny (threshold1, threshold2)
      - Hough transform (HoughLinesP)
      - Find line intersections
      - Save lines & intersections to TSV
      - If geometry.debug==true, also write out the Canny edge image
      - Return geometry data for caller to visualize/draw

    geometry_cfg can contain:
      debug: bool
      canny_threshold1: int
      canny_threshold2: int
      hough_threshold: int
      hough_min_line_length: int
      hough_max_line_gap: int
    """

    # If no logging function is provided, define a no-op
    if log_print_func is None:
        def log_print_func(msg, needed_level, current_level):
            pass

    debug_mode = bool(geometry_cfg.get("debug", False))
    cth1 = geometry_cfg.get("canny_threshold1", 50)
    cth2 = geometry_cfg.get("canny_threshold2", 150)
    h_thresh = geometry_cfg.get("hough_threshold", 30)
    h_min_len = geometry_cfg.get("hough_min_line_length", 20)
    h_max_gap = geometry_cfg.get("hough_max_line_gap", 10)

    # Optionally set OpenCV threads (for HPC usage)
    cv2.setNumThreads(24)

    all_geometry = []
    H, W = orig_shape[:2]

    for i, fm in enumerate(final_masks):
        label_str = fm.get("clip_label", "unknown")
        seg_bool = fm["segmentation"]  # shape=(H, W), bool

        # Convert bool->8u
        mask_8u = seg_bool.astype(np.uint8) * 255

        # 1) Canny
        edges = cv2.Canny(mask_8u, cth1, cth2, apertureSize=3)
        edge_count = np.count_nonzero(edges)

        # If geometry.debug==True and verbosity >=2 => log & write the canny image
        if debug_mode and verbosity >= 2:
            log_print_func(f"[geometry debug] => mask={i}, label={label_str}, "
                           f"Canny edges nonzero={edge_count}", 2, verbosity)
            canny_file = f"{base_filename}_mask{i}_canny.png"
            canny_path = os.path.join(out_dir, canny_file)
            cv2.imwrite(canny_path, edges)
            log_print_func(f"[geometry debug] => wrote canny image => {canny_path}", 2, verbosity)

        # 2) Hough transform
        lines_p = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=h_thresh,
            minLineLength=h_min_len,
            maxLineGap=h_max_gap
        )
        lines_list = []
        if lines_p is not None:
            for ln in lines_p:
                x1, y1, x2, y2 = ln[0]
                lines_list.append((x1, y1, x2, y2))

        if debug_mode and verbosity >= 2:
            if lines_list:
                log_print_func(f"[geometry debug] => found {len(lines_list)} lines for mask={i}, label={label_str}", 2, verbosity)
            else:
                log_print_func(f"[geometry debug] => no lines found for mask={i}, label={label_str}", 2, verbosity)

        # 3) Find intersections
        inters_list = []
        for idx_a in range(len(lines_list)):
            x1a, y1a, x2a, y2a = lines_list[idx_a]
            for idx_b in range(idx_a + 1, len(lines_list)):
                x1b, y1b, x2b, y2b = lines_list[idx_b]
                pt = line_intersection(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
                if pt is not None:
                    ix, iy = pt
                    # skip out-of-bounds
                    if 0 <= ix < W and 0 <= iy < H:
                        inters_list.append((ix, iy))

        # 4) Write lines.tsv
        lines_tsv = f"{base_filename}_mask{i}_lines.tsv"
        lines_path = os.path.join(out_dir, lines_tsv)
        with open(lines_path, "w") as lf:
            lf.write("x1\ty1\tx2\ty2\n")
            for (xa, ya, xb, yb) in lines_list:
                lf.write(f"{xa}\t{ya}\t{xb}\t{yb}\n")

        # 5) Write intersections.tsv
        inters_tsv = f"{base_filename}_mask{i}_intersections.tsv"
        inters_path = os.path.join(out_dir, inters_tsv)
        with open(inters_path, "w") as inf:
            inf.write("ix\tiy\n")
            for (ix, iy) in inters_list:
                inf.write(f"{ix}\t{iy}\n")

        all_geometry.append({
            "mask_index": i,
            "label": label_str,
            "lines": lines_list,
            "intersections": inters_list
        })

    return all_geometry


def line_intersection(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
    """
    Returns (ix, iy) if two line segments [x1a,y1a..x2a,y2a] and [x1b,y1b..x2b,y2b]
    intersect within the segments, else None.
    """
    A1 = float(y2a - y1a)
    B1 = float(x1a - x2a)
    C1 = A1*x1a + B1*y1a

    A2 = float(y2b - y1b)
    B2 = float(x1b - x2b)
    C2 = A2*x1b + B2*y1b

    det = A1*B2 - A2*B1
    if abs(det) < 1e-9:
        return None  # parallel

    ix = (B2*C1 - B1*C2) / det
    iy = (A1*C2 - A2*C1) / det

    if not is_between(ix, x1a, x2a) or not is_between(iy, y1a, y2a):
        return None
    if not is_between(ix, x1b, x2b) or not is_between(iy, y1b, y2b):
        return None

    return (ix, iy)


def is_between(val, e1, e2):
    return min(e1, e2) - 1e-9 <= val <= max(e1, e2) + 1e-9
