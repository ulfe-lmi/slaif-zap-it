"""Batch orchestration helpers for the ZAP-IT pipeline."""

import PIL.Image as Image

# Pillow 11+ moved resampling filters under ``Image.Resampling``.
_resampling = getattr(Image, "Resampling", None)
if not hasattr(Image, "BILINEAR") and _resampling is not None:
    if hasattr(_resampling, "BILINEAR"):
        Image.BILINEAR = _resampling.BILINEAR
if not hasattr(Image, "BILINEAR"):
    Image.BILINEAR = 2
if not hasattr(Image, "BICUBIC") and _resampling is not None:
    if hasattr(_resampling, "BICUBIC"):
        Image.BICUBIC = _resampling.BICUBIC
if not hasattr(Image, "BICUBIC"):
    Image.BICUBIC = 3
if not hasattr(Image, "LANCZOS") and _resampling is not None:
    if hasattr(_resampling, "LANCZOS"):
        Image.LANCZOS = _resampling.LANCZOS
if not hasattr(Image, "LANCZOS"):
    Image.LANCZOS = 1

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
    process_video_parallel,
    run_frame_pipeline,
    _worker_process,
    process_folder_parallel,
    segment_images,
    segment_video,
)
from .core import (
    CoreConfig,
    CoreError,
    FilesystemArtifactSink,
    IdentityMaskOverflowError,
    IdentityMaskProjectionError,
    MemoryArtifactSink,
    ObjectResult,
    PipelineResult,
    SingleImageOutcome,
    StageFunctions,
    StageStatus,
    classify_config_fields,
    config_digest,
    order_final_objects,
    render_identity_png,
    render_yolo,
    run_single_image,
)
from .postprocessing import filter_by_area_bbox

__all__ = [
    "log_print",
    "prepare_dirs",
    "prepare_output_dir",
    "_resolve_device",
    "process_folder",
    "process_video",
    "process_video_parallel",
    "run_frame_pipeline",
    "_worker_process",
    "process_folder_parallel",
    "segment_images",
    "segment_video",
    "filter_by_area_bbox",
    "CoreConfig",
    "CoreError",
    "FilesystemArtifactSink",
    "IdentityMaskOverflowError",
    "IdentityMaskProjectionError",
    "MemoryArtifactSink",
    "ObjectResult",
    "PipelineResult",
    "SingleImageOutcome",
    "StageFunctions",
    "StageStatus",
    "classify_config_fields",
    "config_digest",
    "order_final_objects",
    "render_identity_png",
    "render_yolo",
    "run_single_image",
]
