#!/usr/bin/env python3
"""
zap-it-batch.py

Main orchestrator script for the ZAP-IT Zero-shot Anything Pipeline for Image Tasks.
New: we have a 'geometry' step that operates on each final mask's binary region
(i.e., after all classification & filtering) to do lines & intersections detection.

We:
  1) load config from zap-it-config.py
  2) do SAM2 => post => clip => final filters
  3) optionally do "geometry" => canny + hough lines + intersection detection
  4) produce geometry TSV outputs & optionally draw the geometry on the final overlay image
  5) finalize JSON & summary outputs
"""

import os
import argparse
import shutil
import json
import numpy as np
from PIL import Image, ImageOps
import torch
import cv2   # For canny & hough lines

# Our modules:
from zap_it_config import load_config
from zap_it_sam2 import process_single_pass, process_tiled
from zap_it_clip import ClipFilter
from zap_it_postseg_processing import filter_by_area_bbox
from zap_it_visualization import (
    build_composite_for_masks,
    build_panoptic_final
)

# We still need to build the SAM2 mask_generator:
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_hf

def log_print(msg, needed_level, current_level):
    """
    A small function that prints 'msg' only if 'current_level' >= 'needed_level'.
    We'll keep the same 0=none, 1=some, 2=full logic.
    """
    if current_level >= needed_level:
        print(msg, flush=True)

def prepare_dirs(base_dir, verbosity=1):
    """
    Ensures we have an 'output/' folder in base_dir. Removes old if present.
    """
    out_dir = os.path.join(base_dir, "output")
    if os.path.exists(out_dir):
        log_print(f"[prepare_dirs] Removing old output: {out_dir}", 2, verbosity)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_print(f"[prepare_dirs] Created output folder: {out_dir}", 2, verbosity)
    return out_dir

def apply_geometry_on_mask(
    mask_bool, geometry_cfg, mask_index,
    out_dir, base_name, orig_shape, verbosity=1
):
    """
    Applies the geometry pipeline (Canny->Hough->line intersections) on 'mask_bool'.
    'mask_bool' is a (H,W) bool array for the final mask region of interest.

    :param geometry_cfg: dict with fields:
        debug: bool
        canny_threshold1, canny_threshold2, canny_aperture
        hough_rho, hough_theta, hough_threshold
    :param mask_index: which mask number is this
    :param out_dir: directory to place TSV files
    :param base_name: the prefix for output (like 'L_top'), we add suffix
    :param orig_shape: (H,W) of final image (for line & circle drawing)
    :return: lines_data (list of lines) and intersection_data (list of points)
             or an empty list if no geometry was performed
    """
    H, W = mask_bool.shape[:2]
    debug = bool(geometry_cfg.get("debug", False))
    thr1 = geometry_cfg.get("canny_threshold1", 10)
    thr2 = geometry_cfg.get("canny_threshold2", 70)
    aperture = geometry_cfg.get("canny_aperture", 3)
    rho = float(geometry_cfg.get("hough_rho", 1.0))
    theta_deg = float(geometry_cfg.get("hough_theta", 1.0))
    hough_thr = int(geometry_cfg.get("hough_threshold", 50))

    # Convert mask_bool => uint8 with 255 for True
    mask_u8 = (mask_bool * 255).astype(np.uint8)

    if debug and verbosity >= 1:
        print(f"[geometry] => applying canny on mask {mask_index}, shape={mask_u8.shape}, thr=({thr1},{thr2}), aperture={aperture}")

    # Canny:
    edges = cv2.Canny(mask_u8, threshold1=thr1, threshold2=thr2, apertureSize=aperture)

    # Hough lines:
    # Convert degrees => radians
    theta_rad = np.deg2rad(theta_deg)
    # cv2.HoughLines returns lines in polar form => shape=(N,1,2) => (rho,theta)
    lines = cv2.HoughLines(edges, rho=rho, theta=theta_rad, threshold=hough_thr)

    lines_data = []
    if lines is not None:
        for line_item in lines:
            # line_item is shape=(1,2), let's flatten
            r, t = line_item[0]
            # We store them for output:
            lines_data.append((r, t))  # we keep them in (rho,theta) form
    else:
        lines_data = []

    # We'll find intersection among all pairs of lines that are not parallel
    intersections = []
    # lines_data is a list of (rho,theta)
    # let's define a small function that given (r,theta) => returns line param
    # We'll do standard formula for intersection
    # line i => (ri, ti)
    # line j => (rj, tj)
    # We'll skip near-parallel lines => (ti ~ tj).
    def line_intersect(rt1, rt2):
        # Each line: r = x cos t + y sin t
        # Solve system => we do matrix method
        (r1, t1) = rt1
        (r2, t2) = rt2
        # We'll do:
        # cos t1 * x + sin t1 * y = r1
        # cos t2 * x + sin t2 * y = r2
        # => we solve for x,y
        A = np.array([
            [np.cos(t1), np.sin(t1)],
            [np.cos(t2), np.sin(t2)]
        ], dtype=np.float64)
        b = np.array([r1, r2], dtype=np.float64)
        # We do solve => might fail if parallel
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-9:
            return None
        xy = np.linalg.solve(A, b)
        return (xy[0], xy[1])

    for i in range(len(lines_data)):
        for j in range(i+1, len(lines_data)):
            (r1, t1) = lines_data[i]
            (r2, t2) = lines_data[j]
            # skip if angles are ~the same => parallel
            if abs(t1 - t2) < 1e-3:
                continue
            pnt = line_intersect((r1, t1), (r2, t2))
            if pnt is not None:
                (xx, yy) = pnt
                # we only keep intersection if it's within the bounding box
                # of the mask (since we skip the ROI boundary lines)
                if 0 <= xx < W and 0 <= yy < H:
                    intersections.append((xx, yy))
    # Done => we have lines_data, intersections

    # Write them to TSV
    lines_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_lines.tsv")
    with open(lines_tsv, "w") as f:
        f.write("rho\ttheta_degrees\n")
        for (r, t) in lines_data:
            # convert t => degrees for readability
            t_deg = np.rad2deg(t)
            f.write(f"{r:.3f}\t{t_deg:.3f}\n")

    intersects_tsv = os.path.join(out_dir, f"{base_name}_mask{mask_index}_intersections.tsv")
    with open(intersects_tsv, "w") as f:
        f.write("x\ty\n")
        for (xx, yy) in intersections:
            f.write(f"{xx:.2f}\t{yy:.2f}\n")

    # Return lines_data & intersections for overlay
    return (lines_data, intersections)

def draw_geometry_on_image(
    image_arr, lines_data, intersections, geometry_cfg,
    circle_radius_frac=0.01
):
    """
    Draws lines (in green with black border) + intersections (in red with black border)
    onto the image_arr (H,W,3) in-place. We assume 'image_arr' is color in BGR or RGB?
    We'll treat it as BGR if we use OpenCV. So let's do a small caution.

    circle_radius_frac => fraction of the diagonal for the circle radius.
    """
    h, w = image_arr.shape[:2]
    diag_len = np.sqrt(h*h + w*w)
    circle_radius = int(diag_len * circle_radius_frac)
    if circle_radius < 3:
        circle_radius = 3

    # We'll treat image_arr as BGR if we want to use cv2. If it's RGB, let's convert to BGR first:
    # But let's assume the final image is in RGB. We'll do a quick convert to BGR, draw, then back to RGB.

    bgr_img = image_arr[..., ::-1].copy()

    # We interpret lines_data => (rho, theta)
    # We'll do the standard param => x0,y0 => we pick a big length => drawSegment
    # We'll do 3 passes => to create black border thickness + actual color.

    def draw_line(r, t, color_bgr):
        # We'll pick two points far away:
        # see https://docs.opencv.org/3.4/dd/d1a/group__imgproc__draw.html
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        # We can pick a segment length
        length = 4000  # big enough for typical images
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        cv2.line(bgr_img, (x1,y1), (x2,y2), color_bgr, thickness=1, lineType=cv2.LINE_AA)

    # 1) black border => thickness=3
    for (r, t) in lines_data:
        draw_line(r, t, (0,0,0))  # black
    for (r, t) in lines_data:
        draw_line(r, t, (0,255,0))  # green

    # intersections => we do the same multi pass => black circle radius +/- 1?
    # circle is in red with black border => let's do circle in 3 passes
    # center pass => red => radius=circle_radius
    # outer pass => black => radius=circle_radius+1
    # inner pass => black => radius=circle_radius-1
    for (xx, yy) in intersections:
        center = (int(xx), int(yy))
        # if we are in range
        # pass # let's do 3 circle calls:
        # Outer
        cv2.circle(bgr_img, center, circle_radius+1, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
        # Middle
        cv2.circle(bgr_img, center, circle_radius, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # Inner
        if circle_radius > 2:
            cv2.circle(bgr_img, center, circle_radius-1, (0,0,0), thickness=1, lineType=cv2.LINE_AA)

    # Now convert back to RGB
    final_rgb = bgr_img[..., ::-1]
    image_arr[:] = final_rgb[:]  # copy back

def process_folder(base_dir, mask_generator, config, verbosity=1):
    """
    The main per-folder routine:
      - parse sub-config
      - read each .jpg => apply ROI or entire image => single-pass or tiled => SAM2
      - post-sam2 => area/bbox filter => optionally CLIP => final label filter
      - [NEW] geometry => for each final mask => canny->hough->intersection => store as TSV
                         if there's a final overlay => we draw these lines/circles
      - build optional 2x2 composites => store final JSON => store final detectron2 overlay
    """
    out_dir = prepare_dirs(base_dir, verbosity)

    images = sorted([
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(base_dir, f))
    ])
    if not images:
        log_print(f"No .jpg found in {base_dir}", 1, verbosity)
        return

    # Extract sub-configs
    prep = config.get("preprocessing", {})
    clip_cfg = config.get("clip", {})
    sam2_cfg = config.get("mask_generator", {})
    tile_cfg = config.get("tiled", {})
    alpha_val = config["alpha"]
    postsam2_cfg = config.get("postsam2processing", {})
    vis_cfg = config.get("visualization", {})

    # geometry sub-config if present
    geom_cfg = config.get("geometry", None)

    post_maxsize = postsam2_cfg.get("maxsize", 999999999)
    max_w = postsam2_cfg.get("max_w", 999999999)
    max_h = postsam2_cfg.get("max_h", 999999999)
    post_debug = bool(postsam2_cfg.get("debug", False))

    keep_labels_str = vis_cfg.get("labels", "")
    keep_labels = [s.strip() for s in keep_labels_str.split(",") if s.strip()]
    composite_sam2 = bool(vis_cfg.get("composite-sam2", False))
    composite_clip = bool(vis_cfg.get("composite-clip", False))
    final_flag = bool(vis_cfg.get("final", False))

    tile_size = tile_cfg.get("tile_size", 1024)
    overlap = tile_cfg.get("overlap", 0.2)

    roi_val = prep.get("roi", None)
    resize_val = prep.get("resize", None)
    prep_debug = bool(prep.get("debug", False))

    # Possibly create CLIP filter
    clip_filter = None
    if clip_cfg:
        clip_filter = ClipFilter(
            clip_cfg,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbosity=verbosity,
            log_print_func=log_print
        )

    from zap_it_visualization import build_composite_for_masks, build_panoptic_final

    for fname in images:
        log_print(f"\n[process_folder] => Handling image: {fname}", 1, verbosity)
        img_path = os.path.join(base_dir, fname)

        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        orig_np = np.array(image)
        H_orig, W_orig = orig_np.shape[:2]
        log_print(f" => Original shape = {W_orig}x{H_orig}", 1, verbosity)

        # A) ROI logic
        if roi_val:
            x, y, w, h = [int(v) for v in roi_val.split(",")]
            x2 = min(x + w, W_orig)
            y2 = min(y + h, H_orig)
            partial_np = orig_np[y:y2, x:x2, :]
            log_print(f" => ROI=({x},{y},{w},{h}) => partial shape={partial_np.shape[1]}x{partial_np.shape[0]}", 1, verbosity)
        else:
            x, y = 0, 0
            x2, y2 = W_orig, H_orig
            partial_np = orig_np

        if prep_debug and roi_val:
            roi_file = f"{os.path.splitext(fname)[0]}-roi01.jpg"
            roi_path = os.path.join(out_dir, roi_file)
            Image.fromarray(partial_np).save(roi_path, "JPEG")
            log_print(f" => saved ROI debug => {roi_file}", 1, verbosity)

        # B) Single-pass or Tiling
        from zap_it_sam2 import process_single_pass, process_tiled
        if resize_val is None:
            log_print(" => Tiling approach, no resize", 1, verbosity)
            partial_masks = process_tiled(
                partial_np,
                mask_generator,
                alpha_val,
                tile_size,
                overlap,
                verbosity=verbosity,
                log_print_func=log_print
            )
        else:
            rv = float(resize_val)
            if abs(rv - 1.0) < 1e-7:
                log_print(" => Single pass @native", 1, verbosity)
                partial_masks = process_single_pass(
                    partial_np,
                    mask_generator,
                    alpha_val,
                    verbosity=verbosity,
                    log_print_func=log_print
                )
            elif rv < 1.0:
                new_w = int(partial_np.shape[1] * rv)
                new_h = int(partial_np.shape[0] * rv)
                log_print(f" => downscaled => {new_w}x{new_h} (factor={rv:.2f})", 1, verbosity)
                partial_res = np.array(Image.fromarray(partial_np).resize((new_w, new_h), Image.Resampling.LANCZOS))
                partial_masks = process_single_pass(
                    partial_res,
                    mask_generator,
                    alpha_val,
                    verbosity=verbosity,
                    log_print_func=log_print
                )
            else:
                new_w = int(partial_np.shape[1] * rv)
                new_h = int(partial_np.shape[0] * rv)
                log_print(f" => upscaled => {new_w}x{new_h} (factor={rv:.2f})", 1, verbosity)
                partial_res = np.array(Image.fromarray(partial_np).resize((new_w, new_h), Image.Resampling.LANCZOS))
                partial_masks = process_single_pass(
                    partial_res,
                    mask_generator,
                    alpha_val,
                    verbosity=verbosity,
                    log_print_func=log_print
                )

        # C) Convert partial => global coords
        if resize_val is None:
            H_res, W_res = partial_np.shape[:2]
        else:
            if abs(float(resize_val) - 1.0) < 1e-7:
                H_res, W_res = partial_np.shape[:2]
            else:
                H_res, W_res = partial_res.shape[:2]

        scaleX = (x2 - x) / float(W_res)
        scaleY = (y2 - y) / float(H_res)

        all_masks_pre = []
        for m in partial_masks:
            seg_rs = m["segmentation"]
            rr, cc = np.where(seg_rs)
            if len(rr) == 0:
                continue
            seg_global = np.zeros((H_orig, W_orig), dtype=bool)
            for (rpos, cpos) in zip(rr, cc):
                Yg = y + int(rpos * scaleY)
                Xg = x + int(cpos * scaleX)
                if 0 <= Yg < H_orig and 0 <= Xg < W_orig:
                    seg_global[Yg, Xg] = True

            all_masks_pre.append({
                "segmentation": seg_global,
                "area": seg_global.sum(),
                "predicted_iou": m["predicted_iou"],
                "stability_score": m["stability_score"]
            })

        # D) If sam2 debug => raw patches
        if sam2_cfg.get("debug", False):
            log_print("[mask_generator debug] => saving raw SAM2 patches...", 1, verbosity)
            for idx, mm in enumerate(all_masks_pre):
                seg = mm["segmentation"]
                rr, cc = np.where(seg)
                if len(rr) == 0:
                    continue
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                patch = orig_np[y_min:y_max+1, x_min:x_max+1, :]
                patch_file = f"{os.path.splitext(fname)[0]}_sam2-patch{idx:04d}.jpg"
                patch_path = os.path.join(out_dir, patch_file)
                Image.fromarray(patch).save(patch_path, "JPEG")
                log_print(f"  => wrote {patch_file}", 2, verbosity)

        # E) Post-SAM2 area & bounding box filter
        filtered_for_clip = filter_by_area_bbox(
            all_masks_pre,
            post_maxsize, max_w, max_h,
            verbosity=verbosity, log_print_func=log_print
        )

        # F) CLIP classification
        if clip_filter:
            log_print(f"[clip_filter] => classifying {len(filtered_for_clip)} bounding boxes...", 1, verbosity)
            masked_after_clip = clip_filter.filter_masks(
                filtered_for_clip, orig_np, out_dir, os.path.splitext(fname)[0]
            )
            log_print("[clip_filter] => classification done, now final label filter...", 1, verbosity)
        else:
            masked_after_clip = filtered_for_clip

        # G) final label-based filter => keep only masks whose clip_label is in keep_labels
        final_masks = []
        for mm in masked_after_clip:
            lbl = mm.get("clip_label", None)
            if keep_labels:
                if lbl not in keep_labels:
                    continue
            final_masks.append(mm)

        # H) optional debug => store final patches
        if post_debug:
            log_print("[postsam2processing debug] => saving final patches after classification...", 1, verbosity)
            for idx, mm in enumerate(final_masks):
                seg = mm["segmentation"]
                rr, cc = np.where(seg)
                if len(rr) == 0:
                    continue
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                patch = orig_np[y_min:y_max+1, x_min:x_max+1, :]
                patch_file = f"{os.path.splitext(fname)[0]}_sam2-filtered-patch{idx:04d}.jpg"
                patch_path = os.path.join(out_dir, patch_file)
                Image.fromarray(patch).save(patch_path, "JPEG")
                log_print(f"  => wrote final patch => {patch_file}", 2, verbosity)

        # I) Build optional 2x2 composites
        pre_2x2 = None
        if composite_sam2:
            log_print("[visualization] => building 'pre' 2x2 composite (sam2) ...", 1, verbosity)
            pre_2x2 = build_composite_for_masks(
                orig_np, all_masks_pre, alpha_val,
                verbosity, log_print_func=log_print
            )

        post_2x2 = None
        final_image_array = None
        if composite_clip or final_flag:
            log_print("[visualization] => building 'post' 2x2 composite (clip) ...", 1, verbosity)
            post_2x2, final_image_array = build_composite_for_masks(
                orig_np, final_masks, alpha_val,
                verbosity, log_print_func=log_print,
                return_extra=True
            )

        final_2x4 = None
        if composite_sam2 or composite_clip:
            if pre_2x2 is None:
                if post_2x2 is not None:
                    pre_2x2 = np.zeros_like(post_2x2)
                else:
                    pre_2x2 = np.zeros((100, 100, 3), dtype=np.uint8)
            if post_2x2 is None:
                post_2x2 = np.zeros_like(pre_2x2)
            final_2x4 = np.vstack((pre_2x2, post_2x2))

        # J) final detectron2-based overlay if needed
        panoptic_arr = None
        final_jpg_path = None
        if final_flag and len(final_masks) > 0:
            log_print("[visualization] => generating panoptic final image with detectron2 Visualizer...", 1, verbosity)
            panoptic_arr = build_panoptic_final(orig_np, final_masks)
            final_jpg_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}-final.jpg")
            Image.fromarray(panoptic_arr).save(final_jpg_path, quality=95)
            log_print(f"[visualization] => wrote final single overlay => {final_jpg_path}", 1, verbosity)

        # K) GEOMETRY step: if geometry config is present, do it for each final mask
        # For each final mask => we do canny/hough on that binary => store lines & intersections => if we have panoptic => draw them
        if geom_cfg := config.get("geometry", None):
            # we do it. We'll do canny/hough on each final mask
            if verbosity>=1:
                print(f"[geometry] => geometry section found => applying canny/hough to each final mask... (#masks={len(final_masks)})")

            # If we are going to overlay => let's keep a local copy of the final panoptic if it is present
            # or we can build a 3-channel array from orig if not
            if panoptic_arr is None:
                # no final panoptic => let's make a 3channel copy from orig
                overlay_arr = np.array(orig_np, copy=True)
            else:
                overlay_arr = np.array(panoptic_arr, copy=True)

            for idx, mm in enumerate(final_masks):
                seg_bool = mm["segmentation"]
                lines_data, intersections = apply_geometry_on_mask(
                    seg_bool, geom_cfg, idx, out_dir,
                    os.path.splitext(fname)[0],
                    (H_orig, W_orig),
                    verbosity=verbosity
                )
                if lines_data or intersections:
                    # draw them
                    draw_geometry_on_image(overlay_arr, lines_data, intersections, geom_cfg)
            # after we do for all, let's write the new overlay if we want:
            # the user might want an "immediate" geometry overlay. Let's name it:
            geometry_jpg_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_geometry.jpg")
            Image.fromarray(overlay_arr).save(geometry_jpg_path, quality=95)
            if verbosity>=1:
                print(f"[geometry] => wrote geometry overlay => {geometry_jpg_path}")

            # if we had a separate final_jpg, let's replace it or name a new one
            # but let's keep them separate. The geometry overlay might differ from
            # the detectron2 overlay, or we might want to combine them.
            # For now let's keep them separate. If you want to combine them with detectron2,
            # we could do that before writing final_jpg above.

        # L) Save JSON for final masks
        out_json = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
        ser = []
        for fm in final_masks:
            d = {}
            for k, v in fm.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (np.int32, np.int64)):
                    d[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    d[k] = float(v)
                else:
                    d[k] = v
            ser.append(d)
        with open(out_json, "w") as f:
            json.dump(ser, f)
        log_print(f"[process_folder] => wrote JSON => {out_json}", 1, verbosity)

        # M) If final_2x4 => store summary
        if final_2x4 is not None:
            out_sum = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_summary.jpg")
            Image.fromarray(final_2x4).save(out_sum, quality=95)
            log_print(f"[visualization] => wrote summary => {out_sum}", 1, verbosity)

        log_print("[process_folder] => done with image.\n", 1, verbosity)


def segment_images(base_dir, recursive=False, parsed_config=None, verbosity_level="some"):
    """
    Main entry point. Expects 'parsed_config' from load_config.
    Builds SAM2 model + mask generator, then processes folder(s).
    """
    if not parsed_config:
        raise ValueError("No parsed config provided to segment_images.")

    if verbosity_level == "none":
        vb = 0
    elif verbosity_level == "full":
        vb = 2
    else:
        vb = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mg_cfg = parsed_config["mask_generator"]  # must exist

    print("[segment_images] Building SAM2 model...")
    model = build_sam2_hf("facebook/sam2-hiera-large")
    model.eval().to(device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=mg_cfg["points_per_side"],
        pred_iou_thresh=mg_cfg["pred_iou_thresh"],
        stability_score_thresh=mg_cfg["stability_score_thresh"],
        min_mask_region_area=mg_cfg["min_mask_region_area"],
        crop_n_layers=mg_cfg["crop_n_layers"],
        crop_n_points_downscale_factor=mg_cfg["crop_n_points_downscale_factor"],
        crop_overlap_ratio=mg_cfg["crop_overlap_ratio"],
        box_nms_thresh=mg_cfg["box_nms_thresh"],
        multimask_output=mg_cfg["multimask_output"]
    )

    if recursive:
        for root, dirs, files in os.walk(base_dir):
            process_folder(root, mask_generator, parsed_config, verbosity=vb)
    else:
        process_folder(base_dir, mask_generator, parsed_config, verbosity=vb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZAP-IT Zero-shot Anything Pipeline for Image Tasks - main orchestrator."
    )
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories.")
    parser.add_argument("--verbose", default="some", choices=["none", "some", "full"],
                        help="Verbosity level.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a valid directory.")
        exit(1)

    print("Starting script...")

    config_dict, vb = load_config(args.config, verbosity_level=args.verbose)

    segment_images(
        base_dir=args.dir,
        recursive=args.recursive,
        parsed_config=config_dict,
        verbosity_level=args.verbose
    )

    print("Done.")
