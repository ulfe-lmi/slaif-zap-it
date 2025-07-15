"""
zap_it_yolo_export.py

Utility to export a detection dataset in YOLO format from the ZAP-IT pipeline.
Based on configuration section `export_yolo_det` this script divides each image
(or ROI) into tiles, runs SAM2 segmentation on each tile and writes the tiles
and corresponding bounding boxes in YOLO text format. The resulting directory
structure matches what Ultralytics YOLO expects.
"""

import os
import random
import argparse
import yaml
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_hf

from zap_it_config import load_config
from zap_it_clip import ClipFilter
from zap_it_blip3 import Blip3Filter
from zap_it_postseg_processing import filter_by_area_bbox


def compute_tile_positions_rect(h: int, w: int, tw: int, th: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Return list of (x0,y0,x1,y1) rectangles for tiling."""
    step_x = max(1, int(tw * (1 - overlap)))
    step_y = max(1, int(th * (1 - overlap)))
    positions = []
    y = 0
    while y < h:
        x = 0
        y_end = min(y + th, h)
        while x < w:
            x_end = min(x + tw, w)
            positions.append((x, y, x_end, y_end))
            x += step_x
        y += step_y
    return positions


def export_yolo_dataset(base_dir: str, config: dict, randomize: bool = False, verbosity: int = 1) -> None:
    """Main entry for exporting YOLO dataset."""
    yolo_cfg = config.get("export_yolo_det")
    if not yolo_cfg:
        print("[export_yolo_dataset] export_yolo_det section missing in config")
        return

    labels_str = yolo_cfg.get("labels", "")
    labels = [s.strip() for s in labels_str.split(",") if s.strip()]
    train_split = float(yolo_cfg.get("trainsplit", 80)) / 100.0
    sample_roi = bool(yolo_cfg.get("sample_roi", False))
    dim_str = yolo_cfg.get("sample_dimension", "1024,1024")
    tw, th = [int(v) for v in dim_str.split(",")]

    overlap = float(config.get("tiled", {}).get("overlap", 0.0))

    results_root = os.path.join("results", "yolo")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(results_root, sub), exist_ok=True)

    dataset_yaml = os.path.join(results_root, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        yaml.safe_dump({
            "train": "images/train",
            "val": "images/val",
            "nc": len(labels),
            "names": labels,
        }, f)

    prep_cfg = config.get("preprocessing", {})
    roi_val = prep_cfg.get("roi")

    post_cfg = config.get("postsam2processing", {})
    post_max = post_cfg.get("maxsize", 99999999)
    max_w = post_cfg.get("max_w", 99999999)
    max_h = post_cfg.get("max_h", 99999999)

    clip_cfg = config.get("clip", {})
    blip3_cfg = config.get("blip3", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_filter = ClipFilter(clip_cfg, device=device, verbosity=verbosity) if clip_cfg else None
    blip3_filter = Blip3Filter(blip3_cfg, device=device, verbosity=verbosity) if blip3_cfg else None

    mg_cfg = config.get("mask_generator", {})
    model = build_sam2_hf("facebook/sam2-hiera-large")
    model.eval().to(device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=mg_cfg.get("points_per_side"),
        pred_iou_thresh=mg_cfg.get("pred_iou_thresh"),
        stability_score_thresh=mg_cfg.get("stability_score_thresh"),
        min_mask_region_area=mg_cfg.get("min_mask_region_area"),
        crop_n_layers=mg_cfg.get("crop_n_layers"),
        crop_n_points_downscale_factor=mg_cfg.get("crop_n_points_downscale_factor"),
        crop_overlap_ratio=mg_cfg.get("crop_overlap_ratio"),
        box_nms_thresh=mg_cfg.get("box_nms_thresh"),
        multimask_output=mg_cfg.get("multimask_output"),
    )

    images = [f for f in os.listdir(base_dir) if f.lower().endswith(".jpg")]
    if randomize:
        random.shuffle(images)
    tile_id = 0

    for fname in images:
        img_path = os.path.join(base_dir, fname)
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        img_np = np.array(image)
        H, W = img_np.shape[:2]

        if sample_roi and roi_val:
            rx, ry, rw, rh = [int(v) for v in str(roi_val).split(",")]
            x2 = min(rx + rw, W)
            y2 = min(ry + rh, H)
            roi_np = img_np[ry:y2, rx:x2, :]
        else:
            roi_np = img_np

        rh, rw = roi_np.shape[:2]
        tiles = compute_tile_positions_rect(rh, rw, tw, th, overlap)

        for (x0, y0, x1, y1) in tiles:
            tile_np = roi_np[y0:y1, x0:x1, :]
            tile_masks = mask_generator.generate(tile_np)

            filt = filter_by_area_bbox(tile_masks, post_max, max_w, max_h, verbosity=verbosity)
            if clip_filter:
                filt = clip_filter.filter_masks(filt, tile_np, results_root, f"{fname}_tile{tile_id}")
            if blip3_filter:
                filt = blip3_filter.filter_masks(filt, tile_np, results_root, f"{fname}_tile{tile_id}")

            final_masks = [m for m in filt if not labels or m.get("clip_label") in labels]

            set_name = "train" if random.random() < train_split else "val"
            img_out = os.path.join(results_root, "images", set_name, f"tile_{tile_id:06d}.jpg")
            lbl_out = os.path.join(results_root, "labels", set_name, f"tile_{tile_id:06d}.txt")
            Image.fromarray(tile_np).save(img_out, "JPEG", quality=95)

            lines = []
            for m in final_masks:
                seg = m["segmentation"]
                rr, cc = np.where(seg)
                if len(rr) == 0:
                    continue
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                cx = (x_min + x_max) / 2.0 / (x1 - x0)
                cy = (y_min + y_max) / 2.0 / (y1 - y0)
                bw = (x_max - x_min + 1) / (x1 - x0)
                bh = (y_max - y_min + 1) / (y1 - y0)
                cls_idx = labels.index(m.get("clip_label")) if m.get("clip_label") in labels else 0
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(lbl_out, "w") as f:
                for line in lines:
                    f.write(line + "\n")
            tile_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO detection dataset from ZAP-IT pipeline")
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--randomize", action="store_true", help="Process images in random order")
    parser.add_argument("--verbose", default="some", choices=["none", "some", "full"], help="Verbosity level")
    args = parser.parse_args()

    config, vb = load_config(args.config, verbosity_level=args.verbose)
    export_yolo_dataset(args.dir, config, randomize=args.randomize, verbosity=vb)
