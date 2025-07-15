"""
zap_it_yolo_export.py

Helper to export a detection dataset in YOLO format.  The main
:class:`YoloDatasetExporter` is designed to be reused by ``zap-it-batch.py``
so that YOLO annotations can be produced while running the standard
pipeline.  It can also be invoked as a standalone script for convenience.

The exporter divides each image (optionally restricted to the ROI) into
(possibly overlapping) tiles.  For each tile the provided masks are
converted into YOLO bounding box text files and the tile image is stored
under ``results/yolo`` following the directory structure expected by
Ultralytics YOLO.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
import yaml


# -----------------------------------------------------------------------------
# utility
# -----------------------------------------------------------------------------

def compute_tile_positions_rect(h: int, w: int, tw: int, th: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Return a list of ``(x0, y0, x1, y1)`` rectangles covering ``w``x``h``."""
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


# -----------------------------------------------------------------------------
# main class
# -----------------------------------------------------------------------------

class YoloDatasetExporter:
    """Builds a YOLO detection dataset under ``results/yolo``."""

    def __init__(self, config: dict) -> None:
        yolo_cfg = config.get("export_yolo_det")
        if not yolo_cfg:
            raise ValueError("export_yolo_det section missing in config")

        labels_str = yolo_cfg.get("labels", "")
        self.labels = [s.strip() for s in labels_str.split(",") if s.strip()]
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        self.train_split = float(yolo_cfg.get("trainsplit", 80)) / 100.0
        self.sample_roi = bool(yolo_cfg.get("sample_roi", False))
        dim_str = yolo_cfg.get("sample_dimension", "1024,1024")
        tw, th = [int(v) for v in dim_str.split(",")]
        self.tile_w = tw
        self.tile_h = th
        self.overlap = float(config.get("tiled", {}).get("overlap", 0.0))

        self.results_root = os.path.join("results", "yolo")
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(self.results_root, sub), exist_ok=True)

        dataset_yaml = os.path.join(self.results_root, "dataset.yaml")
        with open(dataset_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "train": "images/train",
                    "val": "images/val",
                    "nc": len(self.labels),
                    "names": self.labels,
                },
                f,
            )

        self.tile_id = 0

    # ------------------------------------------------------------------
    def process_image(
        self,
        image_np: np.ndarray,
        masks: Iterable[dict],
        roi_val: str | None = None,
    ) -> None:
        """Export tiles and annotations for one image."""
        h_img, w_img = image_np.shape[:2]
        offset_x, offset_y = 0, 0

        if self.sample_roi and roi_val:
            rx, ry, rw, rh = [int(v) for v in str(roi_val).split(",")]
            x2 = min(rx + rw, w_img)
            y2 = min(ry + rh, h_img)
            image_np = image_np[ry:y2, rx:x2, :]
            offset_x, offset_y = rx, ry

        h_roi, w_roi = image_np.shape[:2]
        tiles = compute_tile_positions_rect(h_roi, w_roi, self.tile_w, self.tile_h, self.overlap)

        for (x0, y0, x1, y1) in tiles:
            tile_np = image_np[y0:y1, x0:x1, :]
            lines = []
            for m in masks:
                label = m.get("clip_label")
                if self.labels and label not in self.labels:
                    continue
                seg = m["segmentation"]
                gx0 = offset_x + x0
                gy0 = offset_y + y0
                gx1 = offset_x + x1
                gy1 = offset_y + y1
                sub = seg[gy0:gy1, gx0:gx1]
                if not np.any(sub):
                    continue
                rr, cc = np.where(sub)
                y_min, y_max = rr.min(), rr.max()
                x_min, x_max = cc.min(), cc.max()
                cx = (x_min + x_max) / 2.0 / (x1 - x0)
                cy = (y_min + y_max) / 2.0 / (y1 - y0)
                bw = (x_max - x_min + 1) / (x1 - x0)
                bh = (y_max - y_min + 1) / (y1 - y0)
                cls_idx = self.label_to_idx.get(label, 0)
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            set_name = "train" if random.random() < self.train_split else "val"
            base = f"tile_{self.tile_id:06d}"
            img_out = os.path.join(self.results_root, "images", set_name, base + ".jpg")
            lbl_out = os.path.join(self.results_root, "labels", set_name, base + ".txt")
            Image.fromarray(tile_np).save(img_out, "JPEG", quality=95)
            with open(lbl_out, "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
            self.tile_id += 1


# -----------------------------------------------------------------------------
# stand-alone driver (uses the same pipeline as zap-it-batch)
# -----------------------------------------------------------------------------

def run_export_over_folder(base_dir: str, config: dict, randomize: bool = False, verbosity: int = 1) -> None:
    """Convenience wrapper to build a dataset directly from a folder of images."""
    from zap_it_sam2 import process_tiled, process_single_pass
    from zap_it_postseg_processing import filter_by_area_bbox
    from zap_it_clip import ClipFilter
    from zap_it_blip3 import Blip3Filter
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
    import torch

    exporter = YoloDatasetExporter(config)

    prep = config.get("preprocessing", {})
    roi_val = prep.get("roi")
    resize_val = prep.get("resize")
    tile_cfg = config.get("tiled", {})
    tile_size = tile_cfg.get("tile_size", 1024)
    overlap = tile_cfg.get("overlap", 0.2)

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

    for fname in images:
        img_path = os.path.join(base_dir, fname)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        if roi_val:
            rx, ry, rw, rh = [int(v) for v in str(roi_val).split(",")]
            x2 = min(rx + rw, w)
            y2 = min(ry + rh, h)
            roi_np = image_np[ry:y2, rx:x2, :]
        else:
            roi_np = image_np

        if resize_val is None:
            masks = process_tiled(roi_np, mask_generator, config["alpha"], tile_size, overlap, verbosity=verbosity)
        else:
            rv = float(resize_val)
            if abs(rv - 1.0) < 1e-7:
                masks = process_single_pass(roi_np, mask_generator, config["alpha"], verbosity=verbosity)
            else:
                new_w = int(roi_np.shape[1] * rv)
                new_h = int(roi_np.shape[0] * rv)
                res = np.array(Image.fromarray(roi_np).resize((new_w, new_h), Image.Resampling.LANCZOS))
                masks = process_single_pass(res, mask_generator, config["alpha"], verbosity=verbosity)

        masks = filter_by_area_bbox(masks, post_max, max_w, max_h, verbosity=verbosity)
        if clip_filter:
            masks = clip_filter.filter_masks(masks, roi_np, exporter.results_root, os.path.splitext(fname)[0])
        if blip3_filter:
            masks = blip3_filter.filter_masks(masks, roi_np, exporter.results_root, os.path.splitext(fname)[0])

        exporter.process_image(image_np, masks, roi_val=roi_val)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO detection dataset from ZAP-IT pipeline")
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--randomize", action="store_true", help="Process images in random order")
    parser.add_argument("--verbose", default="some", choices=["none", "some", "full"], help="Verbosity level")
    args = parser.parse_args()

    from zap_it_config import load_config

    config, vb = load_config(args.config, verbosity_level=args.verbose)
    run_export_over_folder(args.dir, config, randomize=args.randomize, verbosity=vb)
