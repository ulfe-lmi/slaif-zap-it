"""Reusable orchestration helpers for the batch segmentation pipeline."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import shutil
import traceback
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from queue import Empty

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
from modules.input.video import FFmpegVideoReader, probe_video
from modules.segmenter import initialize_sam2, run_sam2
from modules.visualizer import generate_visualizations
from .core import (
    CandidateViewConfig,
    CoreConfig,
    default_candidate_view_configs,
    FilesystemArtifactSink,
    ObjectResult,
    PipelineResult,
    StageFunctions,
    run_single_image,
)
from .postprocessing import filter_by_area_bbox


def initialize_blip3(*args, **kwargs):
    """Lazy BLIP3 adapter import to keep pure-core package imports acyclic."""
    from modules.verifier import initialize_blip3 as initialize

    return initialize(*args, **kwargs)


def run_blip3(*args, **kwargs):
    """Lazy BLIP3 adapter import preserving the legacy batch seam."""
    from modules.verifier import run_blip3 as run

    return run(*args, **kwargs)


# Stage callables and legacy helpers are re-exported here so that historical
# monkeypatch targets (``src.batch.run_sam2``, ``src.batch.save_roi_debug``...)
# keep working through the adapter.
__all__ = [
    "FFmpegVideoReader",
    "FramePipelineResult",
    "PipelineContext",
    "apply_roi",
    "build_image_writer",
    "build_video_writer",
    "core_config_from_context",
    "filter_by_area_bbox",
    "generate_visualizations",
    "initialize_blip3",
    "initialize_clip",
    "initialize_sam2",
    "list_images",
    "load_image",
    "log_print",
    "pipeline_context_from_core_config",
    "prepare_dirs",
    "prepare_output_dir",
    "process_folder",
    "process_folder_parallel",
    "process_video",
    "process_video_parallel",
    "resize_image",
    "run_blip3",
    "run_clip",
    "run_frame_pipeline",
    "run_sam2",
    "save_roi_debug",
    "segment_images",
    "segment_video",
]


@dataclass
class PipelineContext:
    """Static configuration extracted from the user-provided config."""

    alpha: float
    roi_val: Optional[str]
    resize_val: Optional[object]
    prep_debug: bool
    clip_cfg: dict
    blip3_cfg: dict
    sam2_cfg: dict
    postsam2_cfg: dict
    vis_cfg: dict
    keep_labels: List[str]
    post_maxsize: int
    max_w: int
    max_h: int
    clip_routing_cfg: Mapping[str, object] = field(default_factory=dict)
    candidate_views: Mapping[str, CandidateViewConfig] = field(
        default_factory=default_candidate_view_configs
    )

    def to_core_config(self) -> CoreConfig:
        """Return the normalized core view of this context."""
        return CoreConfig(
            alpha=self.alpha,
            roi_val=self.roi_val,
            resize_val=self.resize_val,
            prep_debug=self.prep_debug,
            clip_cfg=self.clip_cfg,
            blip3_cfg=self.blip3_cfg,
            clip_routing_cfg=self.clip_routing_cfg,
            sam2_cfg=self.sam2_cfg,
            postsam2_cfg=self.postsam2_cfg,
            vis_cfg=self.vis_cfg,
            keep_labels=tuple(self.keep_labels),
            post_maxsize=self.post_maxsize,
            max_w=self.max_w,
            max_h=self.max_h,
            candidate_views=self.candidate_views,
        )


def core_config_from_context(context: PipelineContext) -> CoreConfig:
    """Public helper returning the normalized :class:`CoreConfig`."""
    return context.to_core_config()


@dataclass
class FramePipelineResult:
    """Outputs produced by :func:`run_frame_pipeline`.

    ``rendered``, ``final_masks`` and ``serialized`` preserve the historical
    adapter shape; ``core_result`` carries the typed in-memory core result.
    """

    rendered: Mapping[str, np.ndarray]
    final_masks: List[dict]
    serialized: List[dict]
    core_result: Optional[PipelineResult] = None


def log_print(msg: str, needed_level: int, current_level: int) -> None:
    """Print ``msg`` only when ``current_level`` meets ``needed_level``."""

    if current_level >= needed_level:
        print(msg, flush=True)


def prepare_output_dir(base_dir: str, *, subdir: Optional[str] = None, verbosity: int = 1) -> str:
    """Prepare an ``output`` directory underneath ``base_dir``.

    If ``subdir`` is provided a nested folder with that name is created inside the
    ``output`` directory. The target directory is removed if it already exists to
    ensure a clean slate for each run.
    """

    root_dir = os.path.join(base_dir, "output")
    if subdir:
        os.makedirs(root_dir, exist_ok=True)
        out_dir = os.path.join(root_dir, subdir)
    else:
        out_dir = root_dir
    if os.path.exists(out_dir):
        log_print(f"[prepare_dirs] Removing old output: {out_dir}", 2, verbosity)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_print(f"[prepare_dirs] Created output folder: {out_dir}", 2, verbosity)
    return out_dir


def prepare_dirs(base_dir: str, verbosity: int = 1) -> str:
    """Backward compatible wrapper that prepares ``base_dir/output``."""

    return prepare_output_dir(base_dir, verbosity=verbosity)


def _derive_run_subdir(base_dir: str, input_root: str) -> str:
    """Return the relative subdirectory for ``base_dir`` under ``input_root``."""

    rel = os.path.relpath(base_dir, input_root)
    root_name = os.path.basename(os.path.normpath(input_root))
    if root_name in ("", os.sep):
        root_name = ""
    if rel in (".", os.curdir):
        return root_name or "run"
    if root_name:
        return os.path.join(root_name, rel)
    return rel


def _prepare_output_paths(
    base_dir: str,
    *,
    input_root: Optional[str],
    image_output_root: Optional[str],
    video_output_root: Optional[str],
    verbosity: int,
    cleanup: bool,
    run_folder: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return the output directory along with image/video roots for ``base_dir``."""

    abs_base = os.path.abspath(base_dir)
    abs_root = os.path.abspath(input_root or base_dir)
    rel_subdir = _derive_run_subdir(abs_base, abs_root)
    if run_folder:
        rel_subdir = run_folder

    image_dir = os.path.join(image_output_root, rel_subdir) if image_output_root else None
    video_dir = os.path.join(video_output_root, rel_subdir) if video_output_root else None

    if image_dir or video_dir:
        out_dir = image_dir or video_dir  # Prefer image outputs when available
        abs_out_dir = os.path.abspath(out_dir)
        try:
            common = os.path.commonpath([abs_base, abs_out_dir])
        except ValueError:
            common = ""
        if cleanup and common == abs_base and os.path.exists(abs_out_dir):
            log_print(f"[prepare_dirs] Removing old output: {abs_out_dir}", 2, verbosity)
            shutil.rmtree(abs_out_dir)
        os.makedirs(abs_out_dir, exist_ok=True)
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
        else:
            image_dir = out_dir
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        else:
            video_dir = out_dir
    else:
        if cleanup:
            if run_folder:
                out_dir = prepare_output_dir(abs_base, subdir=run_folder, verbosity=verbosity)
            else:
                out_dir = prepare_dirs(abs_base, verbosity)
        else:
            if run_folder:
                out_dir = os.path.join(abs_base, "output", run_folder)
            else:
                out_dir = os.path.join(abs_base, "output")
            os.makedirs(out_dir, exist_ok=True)
        image_dir = out_dir
        video_dir = out_dir

    return out_dir, image_dir, video_dir


def _compute_yolo_root(
    base_dir: str,
    *,
    input_root: Optional[str],
    image_output_root: Optional[str],
    run_folder: Optional[str] = None,
) -> str:
    """Return the base directory where YOLO exports should be placed."""

    abs_base = os.path.abspath(base_dir)
    abs_root = os.path.abspath(input_root or base_dir)
    default_folder = _derive_run_subdir(abs_base, abs_root)
    if image_output_root:
        folder = run_folder or default_folder
        return os.path.join(image_output_root, folder, "yolo")
    return os.path.join(base_dir, "yolo")


def _build_pipeline_context(config: dict) -> PipelineContext:
    """Derive static configuration used by :func:`run_frame_pipeline`.

    Normalization is delegated to :meth:`CoreConfig.from_mapping` so the batch
    adapter and the in-memory core share one definition.
    """
    return pipeline_context_from_core_config(CoreConfig.from_mapping(config))


def pipeline_context_from_core_config(core_config: CoreConfig) -> PipelineContext:
    """Build a legacy :class:`PipelineContext` from normalized core config."""
    return PipelineContext(
        alpha=core_config.alpha,
        roi_val=core_config.roi_val,
        resize_val=core_config.resize_val,
        prep_debug=core_config.prep_debug,
        clip_cfg=dict(core_config.clip_cfg),
        blip3_cfg=dict(core_config.blip3_cfg),
        sam2_cfg=dict(core_config.sam2_cfg),
        postsam2_cfg=dict(core_config.postsam2_cfg),
        vis_cfg=dict(core_config.vis_cfg),
        keep_labels=list(core_config.keep_labels),
        post_maxsize=core_config.post_maxsize,
        max_w=core_config.max_w,
        max_h=core_config.max_h,
        clip_routing_cfg=dict(core_config.clip_routing_cfg),
        candidate_views=core_config.candidate_views,
    )


def _resolve_device(preferred: Optional[torch.device] = None):  # type: ignore[name-defined]
    """Resolve the torch device to use for computation."""

    if preferred is not None:
        return preferred
    if torch is not None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "cpu"


def _adapter_stage_functions() -> StageFunctions:
    """Resolve stage callables from this module's namespace.

    Reading the module globals at call time keeps legacy monkeypatching of
    ``src.batch.*`` stage functions working through the adapter.
    """
    module_globals = globals()
    return StageFunctions(
        apply_roi=module_globals["apply_roi"],
        resize_image=module_globals["resize_image"],
        run_sam2=module_globals["run_sam2"],
        filter_by_area_bbox=module_globals["filter_by_area_bbox"],
        run_clip=module_globals["run_clip"],
        run_blip3=module_globals["run_blip3"],
        generate_visualizations=module_globals["generate_visualizations"],
    )


def _objects_to_final_masks(objects: Sequence[ObjectResult]) -> List[dict]:
    """Rebuild the historical mask-dict view from typed core objects."""
    return [{**dict(obj.metadata), "segmentation": obj.mask} for obj in objects]


def run_frame_pipeline(
    frame_id: str,
    orig_np: np.ndarray,
    *,
    context: PipelineContext,
    segmenter_state: Optional[dict],
    clip_state: Optional[dict],
    blip3_state: Optional[dict],
    out_dir: str,
    dryrun: bool,
    verbosity: int,
    device=None,
    yolo_exporter=None,
) -> tuple[FramePipelineResult, dict, Optional[dict], Optional[dict]]:
    """Compatibility adapter executing one frame through the in-memory core.

    The core performs all computation; this adapter only binds the legacy
    ``out_dir`` as a filesystem artifact sink so configured debug artifacts
    keep landing in the operator-selected output directory, and forwards the
    optional YOLO dataset exporter hook.
    """
    outcome = run_single_image(
        orig_np,
        context.to_core_config(),
        frame_id=frame_id,
        segmenter_state=segmenter_state,
        clip_state=clip_state,
        blip3_state=blip3_state,
        dryrun=dryrun,
        verbosity=verbosity,
        device=device,
        log_print_func=log_print,
        artifact_sink=FilesystemArtifactSink(out_dir),
        stages=_adapter_stage_functions(),
    )

    final_masks = _objects_to_final_masks(outcome.result.objects)
    serialized = [obj.serialized_metadata() for obj in outcome.result.objects]

    if yolo_exporter is not None:
        yolo_exporter.process_image(orig_np, final_masks, roi_val=context.roi_val)

    result = FramePipelineResult(
        rendered=outcome.result.rendered or {},
        final_masks=final_masks,
        serialized=serialized,
        core_result=outcome.result,
    )
    return result, outcome.segmenter_state, outcome.clip_state, outcome.blip3_state


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
    device=None,
    clip_state: Optional[dict] = None,
    blip3_state: Optional[dict] = None,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
    input_root: Optional[str] = None,
    cleanup_output: bool = True,
) -> None:
    """Run the batch pipeline for images located in ``base_dir``."""

    out_dir, image_dir, video_dir = _prepare_output_paths(
        base_dir,
        input_root=input_root,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
        verbosity=verbosity,
        cleanup=cleanup_output,
    )

    if images is None:
        images = list_images(base_dir)
    else:
        images = list(images)
    if randomize:
        random.shuffle(images)
    if not images:
        log_print(f"No .jpg found in {base_dir}", 1, verbosity)
        return

    pipeline_context = _build_pipeline_context(config)

    image_writer = build_image_writer(config.get("images"), image_dir, verbosity=verbosity)
    video_writer = build_video_writer(config.get("video"), video_dir, verbosity=verbosity)

    try:
        for fname in images:
            frame_id = os.path.splitext(fname)[0]
            log_print(f"\n[process_folder] => Handling image: {fname}", 1, verbosity)
            img_path = os.path.join(base_dir, fname)

            _, orig_np = load_image(img_path)

            result, segmenter_state, clip_state, blip3_state = run_frame_pipeline(
                frame_id,
                orig_np,
                context=pipeline_context,
                segmenter_state=segmenter_state,
                clip_state=clip_state,
                blip3_state=blip3_state,
                out_dir=out_dir,
                dryrun=dryrun,
                verbosity=verbosity,
                device=device,
                yolo_exporter=yolo_exporter,
            )

            if result.rendered:
                image_writer.write(result.rendered)
                video_writer.write(result.rendered)

            out_json = os.path.join(out_dir, f"{frame_id}.json")
            with open(out_json, "w") as f:
                json.dump(result.serialized, f)
            log_print(f"[process_folder] => wrote JSON => {out_json}", 1, verbosity)

            log_print("[process_folder] => done with image.\n", 1, verbosity)
    finally:
        image_writer.close()
        video_writer.close()


def process_video(
    video_path: str,
    segmenter_state: Optional[dict],
    config: dict,
    *,
    dryrun: bool = False,
    verbosity: int = 1,
    yolo_exporter=None,
    device=None,
    clip_state: Optional[dict] = None,
    blip3_state: Optional[dict] = None,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
) -> None:
    """Run the batch pipeline over frames extracted from a video file."""

    video_dir = os.path.dirname(video_path) or "."
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    out_dir, image_dir, video_dir_out = _prepare_output_paths(
        video_dir,
        input_root=video_dir,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
        verbosity=verbosity,
        cleanup=True,
        run_folder=video_stem,
    )

    pipeline_context = _build_pipeline_context(config)

    metadata = probe_video(video_path)
    reader = FFmpegVideoReader(video_path, metadata)

    image_writer = build_image_writer(config.get("images"), image_dir, verbosity=verbosity)
    video_writer = build_video_writer(
        config.get("video"),
        video_dir_out,
        verbosity=verbosity,
        default_fps=metadata.fps,
    )

    try:
        for frame_idx, raw in enumerate(reader, start=1):
            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(
                metadata.height, metadata.width, 3
            )
            frame_id = f"{video_stem}-{frame_idx:07d}"
            log_print(f"\n[process_video] => Handling frame: {frame_id}", 1, verbosity)

            result, segmenter_state, clip_state, blip3_state = run_frame_pipeline(
                frame_id,
                frame_np,
                context=pipeline_context,
                segmenter_state=segmenter_state,
                clip_state=clip_state,
                blip3_state=blip3_state,
                out_dir=out_dir,
                dryrun=dryrun,
                verbosity=verbosity,
                device=device,
                yolo_exporter=yolo_exporter,
            )

            if result.rendered:
                image_writer.write(result.rendered)
                video_writer.write(result.rendered)

            out_json = os.path.join(out_dir, f"{frame_id}.json")
            with open(out_json, "w") as f:
                json.dump(result.serialized, f)
            log_print(f"[process_video] => wrote JSON => {out_json}", 1, verbosity)

            log_print("[process_video] => done with frame.\n", 1, verbosity)
    finally:
        reader.close()
        image_writer.close()
        video_writer.close()


def _video_worker_process(
    gpu_idx: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: dict,
    pipeline_context: PipelineContext,
    dryrun: bool,
    verbosity: int,
    out_dir: str,
) -> None:
    """Worker helper used by :func:`process_video_parallel`."""

    try:
        if dryrun or torch is None:
            device = "cpu"
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > gpu_idx:
                device = torch.device(f"cuda:{gpu_idx}")
            else:
                device = torch.device("cpu")

        segmenter_state = initialize_sam2(
            config["mask_generator"],
            dryrun=dryrun,
            device=device if torch is not None else None,
            verbosity=verbosity,
            log_print_func=log_print,
        )

        clip_state = None
        if config.get("clip"):
            clip_state = initialize_clip(
                config.get("clip", {}),
                dryrun=dryrun,
                device=device,
                verbosity=verbosity,
                log_print_func=log_print,
            )

        blip3_state = None
        if config.get("blip3"):
            blip3_state = initialize_blip3(
                config.get("blip3", {}),
                dryrun=dryrun,
                device=device,
                verbosity=verbosity,
                log_print_func=log_print,
            )

        while True:
            payload = task_queue.get()
            if payload is None:
                break

            frame_idx, frame_id, frame_np = payload
            log_print(
                f"\n[process_video_parallel worker {gpu_idx}] => Handling frame: {frame_id}",
                1,
                verbosity,
            )

            result, segmenter_state, clip_state, blip3_state = run_frame_pipeline(
                frame_id,
                frame_np,
                context=pipeline_context,
                segmenter_state=segmenter_state,
                clip_state=clip_state,
                blip3_state=blip3_state,
                out_dir=out_dir,
                dryrun=dryrun,
                verbosity=verbosity,
                device=device,
                yolo_exporter=None,
            )

            result_queue.put(("RESULT", frame_idx, frame_id, result))
    except Exception:  # pragma: no cover - defensive guard
        result_queue.put(("ERROR", gpu_idx, traceback.format_exc()))
    finally:
        result_queue.put(("DONE", gpu_idx))


def process_video_parallel(
    video_path: str,
    config: dict,
    *,
    dryrun: bool,
    verbosity: int,
    yolo_exporter=None,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
    ngpu: int,
) -> None:
    """Run the batch pipeline over frames extracted from a video file using multiple GPUs."""

    if ngpu <= 0:
        raise ValueError("ngpu must be a positive integer")

    video_dir = os.path.dirname(video_path) or "."
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    out_dir, image_dir, video_dir_out = _prepare_output_paths(
        video_dir,
        input_root=video_dir,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
        verbosity=verbosity,
        cleanup=True,
        run_folder=video_stem,
    )

    pipeline_context = _build_pipeline_context(config)

    metadata = probe_video(video_path)
    reader = FFmpegVideoReader(video_path, metadata)

    image_writer = build_image_writer(config.get("images"), image_dir, verbosity=verbosity)
    video_writer = build_video_writer(
        config.get("video"),
        video_dir_out,
        verbosity=verbosity,
        default_fps=metadata.fps,
    )

    task_queues: List[mp.Queue] = []
    workers: List[mp.Process] = []
    result_queue: Optional[mp.Queue] = None
    worker_error: Optional[Exception] = None

    def emit_frame(
        frame_id: str, result: FramePipelineResult, frame_np: Optional[np.ndarray]
    ) -> None:
        if result.rendered:
            image_writer.write(result.rendered)
            video_writer.write(result.rendered)

        out_json = os.path.join(out_dir, f"{frame_id}.json")
        with open(out_json, "w") as f:
            json.dump(result.serialized, f)
        log_print(f"[process_video_parallel] => wrote JSON => {out_json}", 1, verbosity)

        if yolo_exporter is not None and frame_np is not None:
            yolo_exporter.process_image(
                frame_np,
                result.final_masks,
                roi_val=pipeline_context.roi_val,
            )

    try:
        result_queue = mp.Queue(maxsize=max(ngpu * 2, 1))

        for idx in range(ngpu):
            queue = mp.Queue(maxsize=2)
            task_queues.append(queue)
            proc = mp.Process(
                target=_video_worker_process,
                args=(
                    idx,
                    queue,
                    result_queue,
                    config,
                    pipeline_context,
                    dryrun,
                    verbosity,
                    out_dir,
                ),
            )
            proc.start()
            workers.append(proc)

        pending_results: dict[int, Tuple[str, FramePipelineResult]] = {}
        pending_frames: dict[int, np.ndarray] = {}
        next_to_emit = 1
        processed = 0
        total_frames = 0
        done_workers: set[int] = set()

        def handle_message(msg) -> None:
            nonlocal processed, next_to_emit, worker_error
            kind = msg[0]
            if kind == "RESULT":
                _, frame_idx, frame_id, result = msg
                pending_results[frame_idx] = (frame_id, result)
                while next_to_emit in pending_results:
                    frame_id_local, result_local = pending_results.pop(next_to_emit)
                    frame_np_local = pending_frames.pop(next_to_emit, None)
                    emit_frame(frame_id_local, result_local, frame_np_local)
                    processed += 1
                    next_to_emit += 1
            elif kind == "ERROR":
                _, worker_idx, tb = msg
                worker_error = RuntimeError(f"Worker {worker_idx} failed:\n{tb}")
            elif kind == "DONE":
                done_workers.add(msg[1])

        def drain_nonblocking() -> None:
            if result_queue is None:
                return
            while True:
                try:
                    msg = result_queue.get_nowait()
                except Empty:
                    break
                handle_message(msg)

        try:
            for frame_idx, raw in enumerate(reader, start=1):
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(
                    metadata.height, metadata.width, 3
                )
                frame_id = f"{video_stem}-{frame_idx:07d}"
                log_print(
                    f"\n[process_video_parallel] => Scheduling frame: {frame_id}",
                    1,
                    verbosity,
                )

                total_frames += 1
                if yolo_exporter is not None:
                    pending_frames[frame_idx] = frame_np

                queue_idx = (frame_idx - 1) % len(task_queues)
                task_queues[queue_idx].put((frame_idx, frame_id, frame_np))
                drain_nonblocking()
        finally:
            for queue in task_queues:
                queue.put(None)

        while processed < total_frames and worker_error is None:
            assert result_queue is not None
            msg = result_queue.get()
            handle_message(msg)

        # Drain any remaining DONE/ERROR messages without blocking indefinitely
        drain_nonblocking()

        if worker_error is not None:
            raise worker_error

    finally:
        for queue in task_queues:
            try:
                queue.close()
            except Exception:
                pass
            try:
                queue.join_thread()
            except Exception:
                pass

        if result_queue is not None:
            try:
                result_queue.close()
            except Exception:
                pass
            try:
                result_queue.join_thread()
            except Exception:
                pass

        reader.close()
        image_writer.close()
        video_writer.close()

        for proc in workers:
            proc.join()

        for proc in workers:
            if proc.exitcode not in (0, None) and worker_error is None:
                worker_error = RuntimeError(f"Worker process exited with code {proc.exitcode}")

    if worker_error is not None:
        raise worker_error


def _worker_process(
    gpu_idx: int,
    images: Sequence[str],
    base_dir: str,
    config: dict,
    verbosity: int,
    dryrun: bool = False,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
    input_root: Optional[str] = None,
) -> None:
    """Worker helper used by :func:`process_folder_parallel`."""

    mg_cfg = config["mask_generator"]

    effective_input_root = input_root or base_dir

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

        yolo_exporter = YoloDatasetExporter(
            config,
            base_dir,
            output_root=_compute_yolo_root(
                base_dir,
                input_root=effective_input_root,
                image_output_root=image_output_root,
            ),
            verbosity=verbosity,
            log_print_func=log_print,
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
        device=device,
        clip_state=clip_state,
        blip3_state=blip3_state,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
        input_root=effective_input_root,
        cleanup_output=False,
    )


def process_folder_parallel(
    base_dir: str,
    config: dict,
    ngpu: int,
    verbosity: int,
    randomize: bool = False,
    recursive: bool = False,
    dryrun: bool = False,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
) -> None:
    """Process ``base_dir`` using ``ngpu`` worker processes."""

    _prepare_output_paths(
        base_dir,
        input_root=base_dir,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
        verbosity=verbosity,
        cleanup=True,
    )

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
            args=(idx, subset, base_dir, config, verbosity),
            kwargs={
                "dryrun": dryrun,
                "image_output_root": image_output_root,
                "video_output_root": video_output_root,
                "input_root": base_dir,
            },
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
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
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
                parsed_config,
                base_dir,
                output_root=_compute_yolo_root(
                    base_dir,
                    input_root=base_dir,
                    image_output_root=image_output_root,
                ),
                verbosity=vb,
                log_print_func=log_print,
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
                    image_output_root=image_output_root,
                    video_output_root=video_output_root,
                    input_root=base_dir,
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
                image_output_root=image_output_root,
                video_output_root=video_output_root,
                input_root=base_dir,
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
            image_output_root=image_output_root,
            video_output_root=video_output_root,
        )


def segment_video(
    video_path: str,
    *,
    parsed_config: Optional[dict] = None,
    verbosity_level: str = "some",
    dryrun: bool = False,
    image_output_root: Optional[str] = None,
    video_output_root: Optional[str] = None,
    ngpu: int = 1,
) -> None:
    """Main entry point for processing a single video file."""

    if not parsed_config:
        raise ValueError("No parsed config provided to segment_video.")

    if verbosity_level == "none":
        vb = 0
    elif verbosity_level == "full":
        vb = 2
    else:
        vb = 1

    if torch is None and not dryrun:
        raise RuntimeError("PyTorch is required for full runs. Install torch or use --dryrun.")

    device = "cpu" if dryrun else _resolve_device(None)

    yolo_exporter = None
    video_dir = os.path.dirname(video_path) or "."
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if parsed_config.get("export_yolo_det"):
        from modules.output.yolo import YoloDatasetExporter

        yolo_exporter = YoloDatasetExporter(
            parsed_config,
            video_dir,
            output_root=_compute_yolo_root(
                video_dir,
                input_root=video_dir,
                image_output_root=image_output_root,
                run_folder=video_stem,
            ),
            verbosity=vb,
            log_print_func=log_print,
        )

    if ngpu > 1 and not dryrun:
        log_print(
            f"[segment_video] Launching parallel video processing with {ngpu} GPUs",
            1,
            vb,
        )
        process_video_parallel(
            video_path,
            parsed_config,
            dryrun=dryrun,
            verbosity=vb,
            yolo_exporter=yolo_exporter,
            image_output_root=image_output_root,
            video_output_root=video_output_root,
            ngpu=ngpu,
        )
        return

    print("[segment_video] Initializing SAM2...")
    segmenter_state = initialize_sam2(
        parsed_config["mask_generator"],
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

    process_video(
        video_path,
        segmenter_state,
        parsed_config,
        dryrun=dryrun,
        verbosity=vb,
        yolo_exporter=yolo_exporter,
        device=device,
        clip_state=clip_state,
        blip3_state=blip3_state,
        image_output_root=image_output_root,
        video_output_root=video_output_root,
    )
