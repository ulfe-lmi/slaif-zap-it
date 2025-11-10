"""Batch orchestration helpers for the ZAP-IT pipeline."""

import PIL.Image as Image

# restore old names expected by Detectron2
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR
if not hasattr(Image, "CUBIC"):
    Image.CUBIC = Image.BICUBIC
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from .batch import (
    log_print,
    prepare_dirs,
    prepare_output_dir,
    _resolve_device,
    process_folder,
    process_video,
    run_frame_pipeline,
    _worker_process,
    process_folder_parallel,
    segment_images,
    segment_video,
)
from .postprocessing import filter_by_area_bbox

__all__ = [
    "log_print",
    "prepare_dirs",
    "prepare_output_dir",
    "_resolve_device",
    "process_folder",
    "process_video",
    "run_frame_pipeline",
    "_worker_process",
    "process_folder_parallel",
    "segment_images",
    "segment_video",
    "filter_by_area_bbox",
]
