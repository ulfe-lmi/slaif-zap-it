#!/usr/bin/env python3
"""
zap-it-batch.py

Main orchestrator script for the ZAP-IT Zero-shot Anything Pipeline for Image Tasks.

Steps in summary:
 1) Load config from zap_it_config.py
 2) Build SAM2 mask generator
 3) For each image:
    a) ROI or entire => optional resize => produce partial masks
    b) Scale partial => global => post-sam2 filters => clip => final label filter
    c) Optionally do geometry on each final mask if "geometry" is in config
    d) Produce summary composites, panoptic overlay, JSON, etc.
"""

import os
import argparse
import shutil
import json
import random
import multiprocessing as mp
import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    torch = None

# Our modules:
from zap_it_config import load_config
from modules.segmenter import initialize_sam2, run_sam2
from modules.classifier import initialize_clip, run_clip
from modules.verifier import initialize_blip3, run_blip3
from modules.input.images import (
    list_images,
    load_image,
    apply_roi,
    resize_image,
    save_roi_debug,
)
from zap_it_postseg_processing import filter_by_area_bbox
from modules.output.visualization import build_composite_for_masks, build_panoptic_final

# NEW: geometry code is in a separate file
from zap_it_geometry import apply_geometry_on_mask, draw_geometry_on_image

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


def _resolve_device(preferred=None):
    if preferred is not None:
        return preferred
    if torch is not None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "cpu"


def process_folder(
    base_dir,
    segmenter_state,
    config,
    *,
    dryrun=False,
    verbosity=1,
    randomize=False,
    yolo_exporter=None,
    images=None,
    out_dir=None,
    skip_prepare=False,
    device=None,
    clip_state=None,
    blip3_state=None,
):
    """
    For each .jpg in base_dir:
     - ROI + optional resize => SAM2 => post-sam2 => clip => final label filter
     - If geometry config => do canny/hough on each final mask => write TSV => draw lines/circles
     - Build composites + panoptic => save JSON, summary
    """
    if out_dir is None:
        out_dir = prepare_dirs(base_dir, verbosity)
    elif not skip_prepare:
        out_dir = prepare_dirs(base_dir, verbosity)

    if images is None:
        images = list_images(base_dir)
    else:
        images = list(images)
    if randomize:
        random.shuffle(images)
    if not images:
        log_print(f"No .jpg found in {base_dir}", 1, verbosity)
        return

    # Grab sub-configs
    prep = config.get("preprocessing", {})
    clip_cfg = config.get("clip", {})
    blip3_cfg = config.get("blip3", {})
    sam2_cfg = config.get("mask_generator", {})
    alpha_val = config["alpha"]
    postsam2_cfg = config.get("postsam2processing", {})
    vis_cfg = config.get("visualization", {})
    geometry_cfg = config.get("geometry", None)

    post_maxsize = postsam2_cfg.get("maxsize", 999999999)
    max_w = postsam2_cfg.get("max_w", 999999999)
    max_h = postsam2_cfg.get("max_h", 999999999)
    post_debug = bool(postsam2_cfg.get("debug", False))

    keep_labels_str = vis_cfg.get("labels", "")
    keep_labels = [s.strip() for s in keep_labels_str.split(",") if s.strip()]
    composite_sam2 = bool(vis_cfg.get("composite-sam2", False))
    composite_clip = bool(vis_cfg.get("composite-clip", False))
    final_flag = bool(vis_cfg.get("final", False))

    roi_val = prep.get("roi", None)
    resize_val = prep.get("resize", None)
    prep_debug = bool(prep.get("debug", False))

    # Module states reused across images
    if segmenter_state is None:
        segmenter_state = {}
    if clip_state is None:
        clip_state = {}
    if blip3_state is None:
        blip3_state = {}

    for fname in images:
        log_print(f"\n[process_folder] => Handling image: {fname}", 1, verbosity)
        img_path = os.path.join(base_dir, fname)

        _, orig_np = load_image(img_path)
        H_orig, W_orig = orig_np.shape[:2]
        log_print(f" => Original shape = {W_orig}x{H_orig}", 1, verbosity)

        # A) ROI
        partial_np, (x, y, x2, y2) = apply_roi(orig_np, roi_val)
        if roi_val:
            log_print(
                f" => ROI=({roi_val}) => partial shape={partial_np.shape[1]}x{partial_np.shape[0]}",
                1,
                verbosity,
            )

        if prep_debug and roi_val:
            roi_file = f"{os.path.splitext(fname)[0]}-roi01.jpg"
            roi_path = os.path.join(out_dir, roi_file)
            save_roi_debug(partial_np, roi_path)
            log_print(f" => saved ROI debug => {roi_file}", 1, verbosity)

        # B) segmentation with optional resizing
        resized_np, resize_info = resize_image(partial_np, resize_val)
        if resize_info["mode"] == "native":
            log_print(" => Single pass @native", 1, verbosity)
        else:
            new_w, new_h = resize_info["size"]
            log_print(
                f" => {resize_info['mode']} => {new_w}x{new_h} (factor={resize_info['factor']:.2f})",
                1,
                verbosity,
            )

        segmenter_params = {
            "alpha": alpha_val,
            "dryrun": dryrun,
        }
        if "mask_generator" in segmenter_state:
            segmenter_params["mask_generator"] = segmenter_state["mask_generator"]
        segmenter_state, partial_masks, _ = run_sam2(
            segmenter_state,
            segmenter_params,
            resized_np,
            verbosity=verbosity,
            log_print_func=log_print,
        )

        # C) scale partial => global
        H_res, W_res = resized_np.shape[:2]

        scaleX = (x2 - x) / float(W_res)
        scaleY = (y2 - y) / float(H_res)

        all_masks_pre = []
        for m in partial_masks:
            seg_rs = m["segmentation"]
            rr, cc = np.where(seg_rs)
            if len(rr)==0:
                continue
            seg_global = np.zeros((H_orig, W_orig), dtype=bool)
            for (rpos, cpos) in zip(rr, cc):
                Yg = y + int(rpos*scaleY)
                Xg = x + int(cpos*scaleX)
                if 0<=Yg<H_orig and 0<=Xg<W_orig:
                    seg_global[Yg, Xg] = True

            all_masks_pre.append({
                "segmentation": seg_global,
                "area": seg_global.sum(),
                "predicted_iou": m.get("predicted_iou", None),
                "stability_score": m.get("stability_score", None)
            })

        # D) sam2 debug => raw patches
        if sam2_cfg.get("debug", False):
            log_print("[mask_generator debug] => saving raw SAM2 patches...", 1, verbosity)
            for idx, mm in enumerate(all_masks_pre):
                seg = mm["segmentation"]
                rr, cc = np.where(seg)
                if len(rr)==0:
                    continue
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                patch = orig_np[y_min:y_max+1, x_min:x_max+1, :]
                patch_file = f"{os.path.splitext(fname)[0]}_sam2-patch{idx:04d}.jpg"
                patch_path = os.path.join(out_dir, patch_file)
                Image.fromarray(patch).save(patch_path, "JPEG")
                log_print(f"  => wrote {patch_file}", 2, verbosity)

        # E) post-sam2 area/bbox filter
        filtered_for_clip = filter_by_area_bbox(
            all_masks_pre,
            post_maxsize, max_w, max_h,
            verbosity=verbosity, log_print_func=log_print
        )

        # F) CLIP classification (if provided)
        if clip_cfg:
            log_print(f"[clip] => classifying {len(filtered_for_clip)} bounding boxes...", 1, verbosity)
            clip_params = {
                "config": clip_cfg,
                "device": _resolve_device(device),
                "masks": filtered_for_clip,
                "out_dir": out_dir,
                "fname_stem": os.path.splitext(fname)[0],
                "dryrun": dryrun,
            }
            clip_state, masked_after_clip, _ = run_clip(
                clip_state,
                clip_params,
                orig_np,
                verbosity=verbosity,
                log_print_func=log_print,
            )
            log_print("[clip] => classification done, now final label filter...", 1, verbosity)
        else:
            masked_after_clip = filtered_for_clip

        # Optional BLIP3 verification step
        if blip3_cfg:
            log_print("[blip3] => verifying masks...", 1, verbosity)
            blip3_params = {
                "config": blip3_cfg,
                "device": _resolve_device(device),
                "masks": masked_after_clip,
                "out_dir": out_dir,
                "fname_stem": os.path.splitext(fname)[0],
                "dryrun": dryrun,
            }
            blip3_state, masked_after_clip, _ = run_blip3(
                blip3_state,
                blip3_params,
                orig_np,
                verbosity=verbosity,
                log_print_func=log_print,
            )

        # G) final label-based filter => keep only masks in keep_labels
        final_masks = []
        for mm in masked_after_clip:
            lbl = mm.get("clip_label", None)
            if keep_labels and lbl not in keep_labels:
                continue
            final_masks.append(mm)

        # H) optional debug => final patch saving
        post_debug_flag = bool(postsam2_cfg.get("debug", False))
        if post_debug_flag:
            log_print("[postsam2processing debug] => saving final patches after classification...", 1, verbosity)
            for idx, mm in enumerate(final_masks):
                seg = mm["segmentation"]
                rr, cc = np.where(seg)
                if len(rr)==0:
                    continue
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                patch = orig_np[y_min:y_max+1, x_min:x_max+1, :]
                patch_file = f"{os.path.splitext(fname)[0]}_sam2-filtered-patch{idx:04d}.jpg"
                patch_path = os.path.join(out_dir, patch_file)
                Image.fromarray(patch).save(patch_path, "JPEG")
                log_print(f"  => wrote final patch => {patch_file}", 2, verbosity)

        # I) build optional 2x2 composites
        pre_2x2 = None
        if composite_sam2:
            log_print("[visualization] => building 'pre' 2x2 composite (sam2) ...", 1, verbosity)
            pre_2x2 = build_composite_for_masks(orig_np, all_masks_pre, alpha_val, verbosity, log_print_func=log_print)

        post_2x2 = None
        final_image_array = None
        if composite_clip or final_flag:
            log_print("[visualization] => building 'post' 2x2 composite (clip) ...", 1, verbosity)
            post_2x2, final_image_array = build_composite_for_masks(
                orig_np, final_masks, alpha_val, verbosity, log_print_func=log_print,
                return_extra=True
            )

        final_2x4 = None
        if composite_sam2 or composite_clip:
            if pre_2x2 is None and post_2x2 is not None:
                pre_2x2 = np.zeros_like(post_2x2)
            if post_2x2 is None and pre_2x2 is not None:
                post_2x2 = np.zeros_like(pre_2x2)
            if pre_2x2 is not None and post_2x2 is not None:
                final_2x4 = np.vstack((pre_2x2, post_2x2))

        # J) detectron2-based overlay if final_flag
        panoptic_arr = None
        if final_flag and len(final_masks)>0:
            log_print("[visualization] => generating panoptic final image with detectron2 Visualizer...", 1, verbosity)
            panoptic_arr = build_panoptic_final(orig_np, final_masks)
            final_jpg_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}-final.jpg")
            Image.fromarray(panoptic_arr).save(final_jpg_path, quality=95)
            log_print(f"[visualization] => wrote final single overlay => {final_jpg_path}", 1, verbosity)
        else:
            final_jpg_path = None

        # K) geometry step if config has "geometry"
        if geometry_cfg and len(final_masks)>0:
            if verbosity>=1:
                print(f"[geometry] => geometry section found => applying canny/hough to each final mask... (#masks={len(final_masks)})")

            # pick an overlay as base for geometry
            if panoptic_arr is not None:
                overlay_for_geometry = np.array(panoptic_arr, copy=True)
            else:
                # fallback to original
                overlay_for_geometry = orig_np.copy()

            # for each final mask => apply
            for idx, fm in enumerate(final_masks):
                seg_bool = fm["segmentation"]
                lines_data, intersections = apply_geometry_on_mask(
                    seg_bool, geometry_cfg, idx, out_dir,
                    os.path.splitext(fname)[0],
                    (H_orig, W_orig),
                    verbosity=verbosity
                )
                if lines_data or intersections:
                    draw_geometry_on_image(overlay_for_geometry, lines_data, intersections, geometry_cfg)

            # write overlay with geometry
            geometry_jpg_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_geometry.jpg")
            Image.fromarray(overlay_for_geometry).save(geometry_jpg_path, quality=95)
            if verbosity>=1:
                print(f"[geometry] => wrote geometry overlay => {geometry_jpg_path}")


        # L) Save JSON for final masks
        out_json = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
        ser = []
        for fm in final_masks:
            d = {}
            for k,v in fm.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v,(np.int32, np.int64)):
                    d[k]=int(v)
                elif isinstance(v,(np.float32, np.float64)):
                    d[k]=float(v)
                else:
                    d[k]=v
            ser.append(d)
        with open(out_json,"w") as f:
            json.dump(ser,f)
        log_print(f"[process_folder] => wrote JSON => {out_json}",1,verbosity)

        # M) if final_2x4 => store summary
        if final_2x4 is not None:
            out_sum = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_summary.jpg")
            Image.fromarray(final_2x4).save(out_sum,quality=95)
            log_print(f"[visualization] => wrote summary => {out_sum}",1,verbosity)

        if yolo_exporter is not None:
            yolo_exporter.process_image(orig_np, final_masks, roi_val=roi_val)

        log_print("[process_folder] => done with image.\n",1,verbosity)


def _worker_process(gpu_idx, images, base_dir, config, verbosity, out_dir, dryrun=False):
    """Worker helper for multi-GPU processing."""
    mg_cfg = config["mask_generator"]

    if dryrun or torch is None:
        device = "cpu"
    else:
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.device_count() > gpu_idx else "cpu")

    if verbosity >= 1:
        mode = "dry-run" if dryrun else "full"
        print(f"[worker {gpu_idx}] Initializing SAM2 ({mode}) on {device}...")

    segmenter_state = initialize_sam2(
        mg_cfg,
        dryrun=dryrun,
        device=device if torch is not None else None,
        verbosity=verbosity,
        log_print_func=log_print,
    )

    yolo_exporter = None
    if config.get("export_yolo_det"):
        from modules.output.yolo import YoloDatasetExporter
        yolo_exporter = YoloDatasetExporter(config, base_dir, verbosity=verbosity, log_print_func=log_print)

    clip_state = None
    if config.get("clip"):
        clip_state = initialize_clip(
            config.get("clip", {}),
            dryrun=dryrun,
            device=device if torch is not None else "cpu",
            verbosity=verbosity,
            log_print_func=log_print,
        )

    blip3_state = None
    if config.get("blip3"):
        blip3_state = initialize_blip3(
            config.get("blip3", {}),
            dryrun=dryrun,
            device=device if torch is not None else "cpu",
            verbosity=verbosity,
            log_print_func=log_print,
        )

    process_folder(
        base_dir,
        segmenter_state,
        config,
        dryrun=dryrun,
        verbosity=verbosity,
        randomize=False,
        yolo_exporter=yolo_exporter,
        images=images,
        out_dir=out_dir,
        skip_prepare=True,
        device=device,
        clip_state=clip_state,
        blip3_state=blip3_state,
    )


def process_folder_parallel(base_dir, config, ngpu, verbosity, randomize=False, recursive=False, dryrun=False):
    """Process images in ``base_dir`` using ``ngpu`` processes."""
    out_dir = prepare_dirs(base_dir, verbosity)

    images = list_images(base_dir)
    if randomize:
        random.shuffle(images)
    if not images:
        log_print(f"No .jpg found in {base_dir}", 1, verbosity)
        return

    chunks = [images[i::ngpu] for i in range(ngpu)]

    procs = []
    for idx, subset in enumerate(chunks):
        if not subset:
            continue
        p = mp.Process(target=_worker_process, args=(idx, subset, base_dir, config, verbosity, out_dir, dryrun))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

def segment_images(base_dir, recursive=False, parsed_config=None, verbosity_level="some", randomize=False, ngpu=1, dryrun=False):
    """
    Main entry point. Expects 'parsed_config' from load_config.
    Builds SAM2 model + mask generator, then calls process_folder.
    """
    if not parsed_config:
        raise ValueError("No parsed config provided to segment_images.")

    if verbosity_level=="none":
        vb=0
    elif verbosity_level=="full":
        vb=2
    else:
        vb=1

    mg_cfg = parsed_config["mask_generator"]

    if torch is None and not dryrun:
        raise RuntimeError("PyTorch is required for full runs. Install torch or use --dryrun.")

    if ngpu <= 1 or dryrun:
        yolo_exporter = None
        if parsed_config.get("export_yolo_det"):
            from modules.output.yolo import YoloDatasetExporter
            yolo_exporter = YoloDatasetExporter(parsed_config, base_dir, verbosity=vb, log_print_func=log_print)

        device = "cpu" if dryrun else _resolve_device(None)

        print("[segment_images] Initializing SAM2...")
        segmenter_state = initialize_sam2(
            mg_cfg,
            dryrun=dryrun,
            device=device if torch is not None else None,
            verbosity=vb,
            log_print_func=log_print,
        )

        clip_state = None
        if parsed_config.get("clip"):
            clip_state = initialize_clip(
                parsed_config.get("clip", {}),
                dryrun=dryrun,
                device=device,
                verbosity=vb,
                log_print_func=log_print,
            )

        blip3_state = None
        if parsed_config.get("blip3"):
            blip3_state = initialize_blip3(
                parsed_config.get("blip3", {}),
                dryrun=dryrun,
                device=device,
                verbosity=vb,
                log_print_func=log_print,
            )

        if recursive:
            for root, dirs, files in os.walk(base_dir):
                process_folder(
                    root,
                    segmenter_state,
                    parsed_config,
                    dryrun=dryrun,
                    verbosity=vb,
                    randomize=randomize,
                    yolo_exporter=yolo_exporter,
                    device=device,
                    clip_state=clip_state,
                    blip3_state=blip3_state,
                )
        else:
            process_folder(
                base_dir,
                segmenter_state,
                parsed_config,
                dryrun=dryrun,
                verbosity=vb,
                randomize=randomize,
                yolo_exporter=yolo_exporter,
                device=device,
                clip_state=clip_state,
                blip3_state=blip3_state,
            )
    else:
        process_folder_parallel(
            base_dir,
            parsed_config,
            ngpu,
            vb,
            randomize=randomize,
            recursive=recursive,
            dryrun=dryrun,
        )


if __name__=="__main__":
    import sys
    parser = argparse.ArgumentParser(description="ZAP-IT Zero-shot Anything Pipeline for Image Tasks.")
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories.")
    parser.add_argument("--verbose", default="some", choices=["none","some","full"], help="Verbosity level.")
    parser.add_argument("--randomize", action="store_true", help="Process images in random order")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use in parallel")
    parser.add_argument("--dryrun", action="store_true", help="Enable dry-run mode for SAM2/CLIP/BLIP3")
    args = parser.parse_args()

    if args.ngpu > 1 and not args.dryrun:
        mp.set_start_method("spawn", force=True)

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print("Starting script...")

    from zap_it_config import load_config
    config_dict, vb = load_config(args.config, verbosity_level=args.verbose)

    segment_images(
        base_dir=args.dir,
        recursive=args.recursive,
        parsed_config=config_dict,
        verbosity_level=args.verbose,
        randomize=args.randomize,
        ngpu=args.ngpu,
        dryrun=args.dryrun,
    )

    print("Done.")
