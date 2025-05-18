#!/usr/bin/env python3
"""
zap-it-batch.py

Main orchestrator script for the ZAP-IT Zero-shot Anything Pipeline for Image Tasks.
It uses:
  - zap-it-config.py to parse YAML config
  - zap-it-sam2.py for SAM2-based segmentation
  - zap-it-clip.py for CLIP-based classification
  - zap-it-postseg-processing.py for post-SAM2 area/bbox filtering
  - zap-it-visualization.py for final rendering
  - zap-it-geometry.py (NEW) for line-based geometry analysis

No fundamental processing logic changed, just reorganized into modules for clarity,
plus optional geometry step if 'geometry:' block is in the config.
"""

import os
import argparse
import shutil
import json
import numpy as np
from PIL import Image, ImageOps
import torch

# Import from the config & pipeline modules:
from zap_it_config import load_config
from zap_it_sam2 import process_single_pass, process_tiled
from zap_it_clip import ClipFilter
from zap_it_postseg_processing import filter_by_area_bbox
from zap_it_visualization import build_composite_for_masks, build_panoptic_final

# The following SAM2 library is still needed for constructing mask_generator:
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


def process_folder(base_dir, mask_generator, config, verbosity=1):
    """
    The main per-folder routine:
      - parse sub-config
      - read each .jpg => apply ROI or entire image => single-pass or tiled => SAM2
      - post-sam2 => area/bbox filter => optionally CLIP => final label filter
      - build optional 2x2 composites => store final JSON => store final detectron2 overlay
      - optionally run geometry step if 'geometry:' is in the config
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
    geometry_cfg = config.get("geometry", None)  # NEW geometry block, may be None

    # parse post-SAM2
    post_maxsize = postsam2_cfg.get("maxsize", 999999999)
    max_w = postsam2_cfg.get("max_w", 999999999)
    max_h = postsam2_cfg.get("max_h", 999999999)
    post_debug = bool(postsam2_cfg.get("debug", False))

    # parse visualization
    keep_labels_str = vis_cfg.get("labels", "")
    keep_labels = [s.strip() for s in keep_labels_str.split(",") if s.strip()]
    composite_sam2 = bool(vis_cfg.get("composite-sam2", False))
    composite_clip = bool(vis_cfg.get("composite-clip", False))
    final_flag = bool(vis_cfg.get("final", False))

    # parse tiling or ROI
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

        # D) If sam2_cfg debug => store raw SAM2 patches
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
        from zap_it_visualization import build_composite_for_masks
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
        final_image_path = None
        if final_flag and len(final_masks) > 0:
            log_print("[visualization] => generating panoptic final image with detectron2 Visualizer...", 1, verbosity)
            panoptic_arr = build_panoptic_final(orig_np, final_masks)
            final_jpg = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}-final.jpg")
            Image.fromarray(panoptic_arr).save(final_jpg, quality=95)
            log_print(f"[visualization] => wrote final single overlay => {final_jpg}", 1, verbosity)
            final_image_path = final_jpg  # we might want to open/draw on it if geometry is used

        # K) Save JSON
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

        if final_2x4 is not None:
            out_sum = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_summary.jpg")
            Image.fromarray(final_2x4).save(out_sum, quality=95)
            log_print(f"[visualization] => wrote summary => {out_sum}", 1, verbosity)

        log_print("[process_folder] => done with image.\n", 1, verbosity)

        # L) If geometry block => run geometry step
        if geometry_cfg and len(final_masks) > 0:
            from zap_it_geometry import run_geometry_on_masks
            from zap_it_visualization import draw_geometry_on_final
            import cv2

            log_print("[geometry] => computing lines & intersections for each final mask...", 1, verbosity)

            geometry_data = run_geometry_on_masks(
                final_masks,
                geometry_cfg,
                out_dir,
                os.path.splitext(fname)[0],  # base stem for lines.tsv etc.
                orig_np.shape,
                log_print_func=log_print,
                verbosity=verbosity
            )

            # We'll read the -final.jpg if it exists; otherwise fallback to final_image_array or orig_np
            if final_image_path and os.path.isfile(final_image_path):
                final_image_cv = cv2.imread(final_image_path, cv2.IMREAD_COLOR)  # BGR
                final_image_rgb = cv2.cvtColor(final_image_cv, cv2.COLOR_BGR2RGB)
            elif final_image_array is not None:
                final_image_rgb = final_image_array
            else:
                final_image_rgb = orig_np  # fallback

            # Draw geometry on final
            out_with_geom = draw_geometry_on_final(
                final_image_rgb, geometry_data, geometry_cfg,
                log_print_func=log_print, verbosity=verbosity
            )

            # Save as -final-geom.jpg
            geom_bgr = cv2.cvtColor(out_with_geom, cv2.COLOR_RGB2BGR)
            geom_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}-final-geom.jpg")
            cv2.imwrite(geom_path, geom_bgr)
            log_print(f"[geometry] => wrote geometry-annotated final => {geom_path}", 1, verbosity)


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

    # load config
    config_dict, vb = load_config(args.config, verbosity_level=args.verbose)

    # run main
    segment_images(
        base_dir=args.dir,
        recursive=args.recursive,
        parsed_config=config_dict,
        verbosity_level=args.verbose
    )

    print("Done.")
