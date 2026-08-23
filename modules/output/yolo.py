"""
modules.output.yolo

Helper to export a detection dataset in YOLO format.  The main
:class:`YoloDatasetExporter` is designed to be reused by ``zap-it-batch.py``
so that YOLO annotations can be produced while running the standard
pipeline.  It can also be invoked as a standalone script for convenience.

Each processed image (optionally restricted to the configured ROI) is
converted into a YOLO sample.  The exporter writes the cropped image (if an
ROI is used) alongside the bounding boxes derived from the final masks under
a ``yolo`` directory placed alongside the ``output`` folder within the
processed image directory, following the structure expected by Ultralytics
YOLO.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import yaml
from datetime import datetime

from modules.segmenter import run_sam2
from modules.classifier import run_clip
from modules.verifier import run_blip3
from src.postprocessing import filter_by_area_bbox


# -----------------------------------------------------------------------------
# main class
# -----------------------------------------------------------------------------


class YoloDatasetExporter:
    """Builds a YOLO detection dataset under ``<base_dir>/yolo``."""

    def __init__(
        self,
        config: dict,
        base_dir: str,
        verbosity: int = 1,
        log_print_func: Callable | None = None,
        output_root: str | None = None,
    ) -> None:
        yolo_cfg = config.get("export_yolo_det")
        if not yolo_cfg:
            raise ValueError("export_yolo_det section missing in config")

        self.verbosity = verbosity
        self.log_print_func = log_print_func

        def _log(msg: str, level: int = 1) -> None:
            if self.log_print_func is not None:
                self.log_print_func(msg, level, self.verbosity)
            elif self.verbosity >= level:
                print(msg, flush=True)

        # store method as bound method
        self._log = _log

        labels_str = yolo_cfg.get("labels", "")
        self.labels = [s.strip() for s in labels_str.split(",") if s.strip()]
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        self.train_split = float(yolo_cfg.get("trainsplit", 80)) / 100.0
        self.sample_roi = bool(yolo_cfg.get("sample_roi", False))

        if output_root:
            self.results_root = os.path.abspath(output_root)
        else:
            self.results_root = os.path.join(base_dir, "yolo")
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(self.results_root, sub), exist_ok=True)

        data_dict = {
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.labels),
            "names": self.labels,
        }
        dataset_yaml = os.path.join(self.results_root, "dataset.yaml")
        with open(dataset_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_dict, f)
        self._log(f"[YoloDatasetExporter] wrote {dataset_yaml}", 2)

        data_yaml = os.path.join(self.results_root, "data.yaml")
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_dict, f)
        self._log(f"[YoloDatasetExporter] wrote {data_yaml}", 2)

        self.sample_id = 0

    # ------------------------------------------------------------------
    def process_image(
        self,
        image_np: np.ndarray,
        masks: Iterable[dict],
        roi_val: str | None = None,
    ) -> None:
        """Export one image (optionally cropped to the ROI) and its annotations."""
        h_img, w_img = image_np.shape[:2]
        offset_x, offset_y = 0, 0

        if self.sample_roi and roi_val:
            rx, ry, rw, rh = [int(v) for v in str(roi_val).split(",")]
            x2 = min(rx + rw, w_img)
            y2 = min(ry + rh, h_img)
            sample_np = image_np[ry:y2, rx:x2, :]
            offset_x, offset_y = rx, ry
        else:
            sample_np = image_np

        h_sample, w_sample = sample_np.shape[:2]

        lines = []
        for m in masks:
            label = m.get("clip_label")
            if self.labels and label not in self.labels:
                continue
            seg = m["segmentation"]
            sub = seg[offset_y : offset_y + h_sample, offset_x : offset_x + w_sample]
            if not np.any(sub):
                continue
            rr, cc = np.where(sub)
            y_min, y_max = rr.min(), rr.max()
            x_min, x_max = cc.min(), cc.max()
            cx = (x_min + x_max) / 2.0 / w_sample
            cy = (y_min + y_max) / 2.0 / h_sample
            bw = (x_max - x_min + 1) / w_sample
            bh = (y_max - y_min + 1) / h_sample
            cls_idx = self.label_to_idx.get(label, 0)
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        set_name = "train" if random.random() < self.train_split else "val"
        base = f"sample_{self.sample_id:06d}"
        img_out = os.path.join(self.results_root, "images", set_name, base + ".jpg")
        lbl_out = os.path.join(self.results_root, "labels", set_name, base + ".txt")
        Image.fromarray(sample_np).save(img_out, "JPEG", quality=95)
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] wrote image {img_out}", 2)
        with open(lbl_out, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] wrote labels {lbl_out}", 2)
        self.sample_id += 1


# -----------------------------------------------------------------------------
# stand-alone driver (uses the same pipeline as zap-it-batch)
# -----------------------------------------------------------------------------


def run_export_over_folder(
    base_dir: str, config: dict, randomize: bool = False, verbosity: int = 1
) -> None:
    """Convenience wrapper to build a dataset directly from a folder of images."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
    import torch

    exporter = YoloDatasetExporter(config, base_dir, verbosity=verbosity)

    def module_log(msg, needed_level, _current_level):
        exporter._log(msg, needed_level)

    prep = config.get("preprocessing", {})
    roi_val = prep.get("roi")
    resize_val = prep.get("resize")
    post_cfg = config.get("postsam2processing", {})
    post_max = post_cfg.get("maxsize", 99999999)
    max_w = post_cfg.get("max_w", 99999999)
    max_h = post_cfg.get("max_h", 99999999)

    clip_cfg = config.get("clip", {})
    blip3_cfg = config.get("blip3", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_state = None
    blip3_state = None
    segmenter_state = None

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
            x, y = rx, ry
        else:
            roi_np = image_np
            x, y = 0, 0
            x2, y2 = w, h

        if resize_val is None:
            resized_np = roi_np
        else:
            rv = float(resize_val)
            if abs(rv - 1.0) < 1e-7:
                resized_np = roi_np
            else:
                new_w = int(roi_np.shape[1] * rv)
                new_h = int(roi_np.shape[0] * rv)
                resized_np = np.array(
                    Image.fromarray(roi_np).resize((new_w, new_h), Image.Resampling.LANCZOS)
                )

        segmenter_params = {
            "mask_generator": mask_generator,
            "alpha": config["alpha"],
        }
        segmenter_state, partial_masks, _ = run_sam2(
            segmenter_state,
            segmenter_params,
            resized_np,
            verbosity=verbosity,
            log_print_func=module_log,
        )

        H_res, W_res = resized_np.shape[:2]
        scaleX = (x2 - x) / float(W_res)
        scaleY = (y2 - y) / float(H_res)

        all_masks_pre = []
        for m in partial_masks:
            seg_rs = m["segmentation"]
            rr, cc = np.where(seg_rs)
            if len(rr) == 0:
                continue
            seg_global = np.zeros((h, w), dtype=bool)
            for rpos, cpos in zip(rr, cc):
                Yg = y + int(rpos * scaleY)
                Xg = x + int(cpos * scaleX)
                if 0 <= Yg < h and 0 <= Xg < w:
                    seg_global[Yg, Xg] = True

            all_masks_pre.append(
                {
                    "segmentation": seg_global,
                    "area": seg_global.sum(),
                    "predicted_iou": m.get("predicted_iou"),
                    "stability_score": m.get("stability_score"),
                }
            )

        masks = filter_by_area_bbox(all_masks_pre, post_max, max_w, max_h, verbosity=verbosity)
        if clip_cfg:
            clip_params = {
                "config": clip_cfg,
                "device": device,
                "masks": masks,
                "out_dir": exporter.results_root,
                "fname_stem": os.path.splitext(fname)[0],
            }
            clip_state, masks, _ = run_clip(
                clip_state,
                clip_params,
                image_np,
                verbosity=verbosity,
                log_print_func=module_log,
            )
        if blip3_cfg:
            blip3_params = {
                "config": blip3_cfg,
                "device": device,
                "masks": masks,
                "out_dir": exporter.results_root,
                "fname_stem": os.path.splitext(fname)[0],
            }
            blip3_state, masks, _ = run_blip3(
                blip3_state,
                blip3_params,
                image_np,
                verbosity=verbosity,
                log_print_func=module_log,
            )

        exporter.process_image(image_np, masks, roi_val=roi_val)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export YOLO detection dataset from ZAP-IT pipeline"
    )
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--randomize", action="store_true", help="Process images in random order")
    parser.add_argument(
        "--verbose",
        default="some",
        choices=["none", "some", "full"],
        help="Verbosity level",
    )
    args = parser.parse_args()

    from src.config import load_config

    config, vb = load_config(args.config, verbosity_level=args.verbose)
    run_export_over_folder(args.dir, config, randomize=args.randomize, verbosity=vb)
