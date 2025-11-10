"""Reusable orchestration helpers for the batch segmentation pipeline."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import shutil
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

from modules.classifier import initialize_clip, run_clip
from modules.input.images import (
    apply_roi,
    list_images,
    load_image,
    resize_image,
    save_roi_debug,
)
from modules.output.images import build_image_writer
from modules.output.video import build_video_writer
from modules.segmenter import initialize_sam2, run_sam2
from modules.verifier import initialize_blip3, run_blip3
from modules.visualizer import generate_visualizations
from .postprocessing import filter_by_area_bbox


def log_print(msg: str, needed_level: int, current_level: int) -> None:
    """Print ``msg`` only when ``current_level`` meets ``needed_level``."""

    if current_level >= needed_level:
        print(msg, flush=True)


def prepare_dirs(base_dir: str, verbosity: int = 1) -> str:
    """Prepare the ``output/`` directory underneath ``base_dir``."""

    out_dir = os.path.join(base_dir, "output")
    if os.path.exists(out_dir):
        log_print(f"[prepare_dirs] Removing old output: {out_dir}", 2, verbosity)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_print(f"[prepare_dirs] Created output folder: {out_dir}", 2, verbosity)
    return out_dir


def _resolve_device(preferred: Optional[torch.device] = None):  # type: ignore[name-defined]
    """Resolve the torch device to use for computation."""

    if preferred is not None:
        return preferred
    if torch is not None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "cpu"


def process_folder(
    base_dir: str,
    segmenter_state: Optional[dict],
    config: dict,
    *,
    dryrun: bool = False,
    verbosity: int = 1,
    randomize: bool = False,
    yolo_exporter=None,
    images: Optional[Iterable[str]] = None,
    out_dir: Optional[str] = None,
    skip_prepare: bool = False,
    device=None,
    clip_state: Optional[dict] = None,
    blip3_state: Optional[dict] = None,
) -> None:
    """Run the batch pipeline for images located in ``base_dir``."""

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

    prep = config.get("preprocessing", {})
    clip_cfg = config.get("clip", {})
    blip3_cfg = config.get("blip3", {})
    sam2_cfg = config.get("mask_generator", {})
    alpha_val = config["alpha"]
    postsam2_cfg = config.get("postsam2processing", {})
    vis_cfg = config.get("visualization", {})

    post_maxsize = postsam2_cfg.get("maxsize", 999_999_999)
    max_w = postsam2_cfg.get("max_w", 999_999_999)
    max_h = postsam2_cfg.get("max_h", 999_999_999)

    labels_cfg = vis_cfg.get("labels", [])
    if isinstance(labels_cfg, str):
        keep_labels = [s.strip() for s in labels_cfg.split(",") if s.strip()]
    elif isinstance(labels_cfg, (list, tuple, set)):
        keep_labels = [str(item).strip() for item in labels_cfg if str(item).strip()]
    else:
        keep_labels = []

    image_writer = build_image_writer(config.get("images"), out_dir, verbosity=verbosity)
    video_writer = build_video_writer(config.get("video"), out_dir, verbosity=verbosity)

    roi_val = prep.get("roi", None)
    resize_val = prep.get("resize", None)
    prep_debug = bool(prep.get("debug", False))

    if segmenter_state is None:
        segmenter_state = {}
    if clip_state is None:
        clip_state = {}
    if blip3_state is None:
        blip3_state = {}

    try:
        for fname in images:
            log_print(f"\n[process_folder] => Handling image: {fname}", 1, verbosity)
            img_path = os.path.join(base_dir, fname)

            _, orig_np = load_image(img_path)
            H_orig, W_orig = orig_np.shape[:2]
            log_print(f" => Original shape = {W_orig}x{H_orig}", 1, verbosity)

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

            H_res, W_res = resized_np.shape[:2]

            scaleX = (x2 - x) / float(W_res)
            scaleY = (y2 - y) / float(H_res)

            all_masks_pre: List[dict] = []
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

                all_masks_pre.append(
                    {
                        "segmentation": seg_global,
                        "area": seg_global.sum(),
                        "predicted_iou": m.get("predicted_iou", None),
                        "stability_score": m.get("stability_score", None),
                    }
                )

            if sam2_cfg.get("debug", False):
                log_print("[mask_generator debug] => saving raw SAM2 patches...", 1, verbosity)
                for idx, mm in enumerate(all_masks_pre):
                    seg = mm["segmentation"]
                    rr, cc = np.where(seg)
                    if len(rr) == 0:
                        continue
                    y_min, y_max = rr.min(), rr.max()
                    x_min, x_max = cc.min(), cc.max()
                    patch = orig_np[y_min : y_max + 1, x_min : x_max + 1, :]
                    patch_file = f"{os.path.splitext(fname)[0]}_sam2-patch{idx:04d}.jpg"
                    patch_path = os.path.join(out_dir, patch_file)
                    Image.fromarray(patch).save(patch_path, "JPEG")
                    log_print(f"  => wrote {patch_file}", 2, verbosity)

            filtered_for_clip = filter_by_area_bbox(
                all_masks_pre,
                post_maxsize,
                max_w,
                max_h,
                verbosity=verbosity,
                log_print_func=log_print,
            )

            if clip_cfg:
                log_print(
                    f"[clip] => classifying {len(filtered_for_clip)} bounding boxes...",
                    1,
                    verbosity,
                )
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

            clip_only_masks = [dict(m) for m in masked_after_clip]

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

            final_masks = []
            for mm in masked_after_clip:
                lbl = mm.get("clip_label", None)
                if keep_labels and lbl not in keep_labels:
                    continue
                final_masks.append(mm)

            post_debug_flag = bool(postsam2_cfg.get("debug", False))
            if post_debug_flag:
                log_print(
                    "[postsam2processing debug] => saving final patches after classification...",
                    1,
                    verbosity,
                )
                for idx, mm in enumerate(final_masks):
                    seg = mm["segmentation"]
                    rr, cc = np.where(seg)
                    if len(rr) == 0:
                        continue
                    y_min, y_max = rr.min(), rr.max()
                    x_min, x_max = cc.min(), cc.max()
                    patch = orig_np[y_min : y_max + 1, x_min : x_max + 1, :]
                    patch_file = f"{os.path.splitext(fname)[0]}_sam2-filtered-patch{idx:04d}.jpg"
                    patch_path = os.path.join(out_dir, patch_file)
                    Image.fromarray(patch).save(patch_path, "JPEG")
                    log_print(f"  => wrote final patch => {patch_file}", 2, verbosity)

            stage_masks = {
                "sam2": all_masks_pre,
                "clip": clip_only_masks,
                "blip3": final_masks,
            }

            rendered = generate_visualizations(
                orig_np,
                stage_masks,
                vis_cfg,
                default_alpha=alpha_val,
                verbosity=verbosity,
                log_print_func=log_print,
            )

            if rendered:
                image_writer.write(rendered)
                video_writer.write(rendered)

            out_json = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
            ser: List[dict] = []
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

            if yolo_exporter is not None:
                yolo_exporter.process_image(orig_np, final_masks, roi_val=roi_val)

            log_print("[process_folder] => done with image.\n", 1, verbosity)
    finally:
        image_writer.close()
        video_writer.close()


def _worker_process(
    gpu_idx: int,
    images: Sequence[str],
    base_dir: str,
    config: dict,
    verbosity: int,
    out_dir: str,
    dryrun: bool = False,
) -> None:
    """Worker helper used by :func:`process_folder_parallel`."""

    mg_cfg = config["mask_generator"]

    if dryrun or torch is None:
        device = "cpu"
    else:
        device = torch.device(
            f"cuda:{gpu_idx}" if torch.cuda.device_count() > gpu_idx else "cpu"
        )

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

        yolo_exporter = YoloDatasetExporter(
            config, base_dir, verbosity=verbosity, log_print_func=log_print
        )

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


def process_folder_parallel(
    base_dir: str,
    config: dict,
    ngpu: int,
    verbosity: int,
    randomize: bool = False,
    recursive: bool = False,
    dryrun: bool = False,
) -> None:
    """Process ``base_dir`` using ``ngpu`` worker processes."""

    out_dir = prepare_dirs(base_dir, verbosity)

    images = list_images(base_dir)
    if randomize:
        random.shuffle(images)
    if not images:
        log_print(f"No .jpg found in {base_dir}", 1, verbosity)
        return

    chunks = [images[i::ngpu] for i in range(ngpu)]

    procs: List[mp.Process] = []
    for idx, subset in enumerate(chunks):
        if not subset:
            continue
        p = mp.Process(
            target=_worker_process,
            args=(idx, subset, base_dir, config, verbosity, out_dir, dryrun),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def segment_images(
    base_dir: str,
    recursive: bool = False,
    parsed_config: Optional[dict] = None,
    verbosity_level: str = "some",
    randomize: bool = False,
    ngpu: int = 1,
    dryrun: bool = False,
) -> None:
    """Main entry point used by ``zap-it-batch.py``."""

    if not parsed_config:
        raise ValueError("No parsed config provided to segment_images.")

    if verbosity_level == "none":
        vb = 0
    elif verbosity_level == "full":
        vb = 2
    else:
        vb = 1

    mg_cfg = parsed_config["mask_generator"]

    if torch is None and not dryrun:
        raise RuntimeError("PyTorch is required for full runs. Install torch or use --dryrun.")

    if ngpu <= 1 or dryrun:
        yolo_exporter = None
        if parsed_config.get("export_yolo_det"):
            from modules.output.yolo import YoloDatasetExporter

            yolo_exporter = YoloDatasetExporter(
                parsed_config, base_dir, verbosity=vb, log_print_func=log_print
            )

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
            for root, _dirs, _files in os.walk(base_dir):
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
