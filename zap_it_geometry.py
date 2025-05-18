"""
zap_it_geometry.py

Provides functions for line-based geometry analysis on final "positive" masks.

Exported:
  - apply_geometry_on_mask(...):
      Perform Canny + Hough on a single mask, save lines & intersections to TSV,
      optionally write the Canny .png image if debug==true, and return line data.
  - draw_geometry_on_image(...):
      Draw lines & intersections onto an RGB or BGR image array (OpenCV style).
  - line_intersection(...), is_between(...):
      Helpers for computing line-segment intersections.
"""

import os
import numpy as np
import cv2

def apply_geometry_on_mask(
    mask_bool,
    geometry_cfg,
    mask_index,
    out_dir,
    base_name,
    orig_shape,
    verbosity=1
):
    """
    Applies the geometry pipeline (Canny->Hough->line intersections) on `mask_bool`.

    Args:
      mask_bool (np.bool_ array): the final mask region, shape=(H,W).
      geometry_cfg (dict): geometry config with possible keys:
         debug (bool),
         canny_threshold1 (int),
         canny_threshold2 (int),
         canny_aperture (int),
         hough_rho (float),
         hough_theta (float),   # in degrees
         hough_threshold (int),
         hough_min_line_length (int), 
         hough_max_line_gap (int).
      mask_index (int): the index of this mask among final masks
      out_dir (str): where to write TSV and optional Canny image
      base_name (str): typically the image stem, e.g. "some_image"
      orig_shape (tuple): (H, W) of the full region for reference
      verbosity (int): log level: 0=none,1=some,2=full

    Returns:
      (lines_data, intersections):
        lines_data: list of (x1,y1,x2,y2) from cv2.HoughLinesP
        intersections: list of (ix, iy) float coords in image space
    """
    H, W = mask_bool.shape[:2]

    debug = bool(geometry_cfg.get("debug", False))
    thr1 = geometry_cfg.get("canny_threshold1", 50)
    thr2 = geometry_cfg.get("canny_threshold2", 150)
    aperture = int(geometry_cfg.get("canny_aperture", 3))

    rho = float(geometry_cfg.get("hough_rho", 1.0))
    theta_deg = float(geometry_cfg.get("hough_theta", 1.0))
    hough_thr = int(geometry_cfg.get("hough_threshold", 30))
    h_min_len = int(geometry_cfg.get("hough_min_line_length", 20))
    h_max_gap = int(geometry_cfg.get("hough_max_line_gap", 10))

    # Convert bool->8u for Canny
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    # Possibly log minimal info
    if debug and verbosity >= 1:
        print(f"[geometry] => applying canny on mask {mask_index}, shape=({H},{W}), "
              f"thr=({thr1},{thr2}), aperture={aperture}")

    # 1) Canny
    edges = cv2.Canny(mask_u8, threshold1=thr1, threshold2=thr2, apertureSize=aperture)
    edge_nonzero = np.count_nonzero(edges)

    # Write out the Canny image if debug
    if debug and verbosity >= 2:
        canny_file = f"{base_name}_mask{mask_index}_canny.png"
        canny_path = os.path.join(out_dir, canny_file)
        cv2.imwrite(canny_path, edges)
        print(f"[geometry debug] => wrote canny image => {canny_path} (nonzero={edge_nonzero})")

    # 2) Hough transform (probabilistic)
    theta_rad = np.deg2rad(theta_deg)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta_rad,
        threshold=hough_thr,
        minLineLength=h_min_len,
        maxLineGap=h_max_gap
    )

    lines_data = []
    if lines_p is not None:
        for ln in lines_p:
            x1, y1, x2, y2 = ln[0]
            lines_data.append((x1, y1, x2, y2))

    # Optional debug print
    if debug and verbosity >= 2:
        if lines_data:
            print(f"[geometry debug] => found {len(lines_data)} lines for mask={mask_index}")
        else:
            print(f"[geometry debug] => no lines found for mask={mask_index}")

    # 3) Intersections among lines
    intersections = []
    for i in range(len(lines_data)):
        x1a, y1a, x2a, y2a = lines_data[i]
        for j in range(i+1, len(lines_data)):
            x1b, y1b, x2b, y2b = lines_data[j]
            pt = line_intersection(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
            if pt is not None:
                (ix, iy) = pt
                if 0 <= ix < W and 0 <= iy < H:
                    intersections.append((ix, iy))

    # Write lines.tsv
    lines_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_lines.tsv")
    with open(lines_tsv, "w") as lf:
        lf.write("x1\ty1\tx2\ty2\n")
        for (xa, ya, xb, yb) in lines_data:
            lf.write(f"{xa}\t{ya}\t{xb}\t{yb}\n")

    # Write intersections.tsv
    inters_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_intersections.tsv")
    with open(inters_tsv, "w") as inf:
        inf.write("ix\tiy\n")
        for (ix, iy) in intersections:
            inf.write(f"{ix}\t{iy}\n")

    return (lines_data, intersections)


def draw_geometry_on_image(image_arr, lines_data, intersections, geometry_cfg, circle_radius_frac=0.01):
    """
    Draws lines (in green w/ black border) and intersections (in red w/ black border)
    onto 'image_arr' in-place. 'image_arr' is assumed to be an RGB array of shape=(H,W,3).

    We do 2-phase draws so there's a black outline around lines & circles:
      lines => thicker black, then green
      intersections => black outer circle, red circle, black inner circle

    circle_radius_frac => fraction of the image diagonal for intersection circles.
    """
    import cv2
    H, W = image_arr.shape[:2]
    diag_len = np.sqrt(H*H + W*W)
    base_r = int(diag_len * circle_radius_frac)
    if base_r < 2:
        base_r = 2

    # We'll convert 'image_arr' from RGB to BGR to use cv2 line/circle draws
    bgr = image_arr[..., ::-1].copy()

    # 1) Draw lines
    # we store them in (x1,y1,x2,y2). We'll do line in black with thickness=3, then green thickness=1
    for (x1,y1,x2,y2) in lines_data:
        # black
        cv2.line(bgr, (x1,y1), (x2,y2), (0,0,0), thickness=3, lineType=cv2.LINE_AA)
        # green
        cv2.line(bgr, (x1,y1), (x2,y2), (0,255,0), thickness=1, lineType=cv2.LINE_AA)

    # 2) Draw intersection circles
    for (ix, iy) in intersections:
        center = (int(ix), int(iy))
        # black outer
        cv2.circle(bgr, center, base_r+1, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)
        # red main
        cv2.circle(bgr, center, base_r, (0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        # black inner if base_r>=2
        if base_r >= 2:
            cv2.circle(bgr, center, base_r-1, (0,0,0), thickness=-1, lineType=cv2.LINE_AA)

    # Convert back to RGB
    image_arr[...] = bgr[..., ::-1]
    return image_arr


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
        return None  # parallel or nearly so

    ix = (B2*C1 - B1*C2) / det
    iy = (A1*C2 - A2*C1) / det

    if not is_between(ix, x1a, x2a) or not is_between(iy, y1a, y2a):
        return None
    if not is_between(ix, x1b, x2b) or not is_between(iy, y1b, y2b):
        return None

    return (ix, iy)


def is_between(val, end1, end2):
    """
    Return True if val is within the closed interval [min(end1,end2), max(end1,end2)].
    """
    return min(end1, end2) - 1e-9 <= val <= max(end1, end2) + 1e-9
