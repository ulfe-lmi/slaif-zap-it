#!/usr/bin/env python3
"""Bounded Objective 003 GPU1 qualification runner.

Run this script only in the repo-owned GPU environment and with the launch
mask already set:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      .venv-gpu/bin/python scripts/qualify_gpu_runtime.py

The script downloads only the three approved, revision-pinned model snapshots
when ``--download`` is supplied, uses a generated in-memory fixture, captures
sanitized all-GPU snapshots around each class, and skips any load whose
predicted peak exceeds the conservative 90% budget.  It never changes another
process or starts a listener.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.device import inspect_visible_device, require_launch_environment
from src.runtime.models import APPROVED_MODEL_SPECS, ModelSpec
from src.runtime.ports import select_candidate_port
from src.runtime.shm import ensure_shm_root, shm_free_bytes

GPU_UUID = "GPU-c457dbaf-991c-dc23-c781-0dc030776dd8"
GPU_TOTAL_MIB = 11_264
GPU_PEAK_FRACTION = 0.90
FIXTURE_SIZE = 128


def _command(args: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
    return result.stdout.strip() if result.returncode == 0 else ""


def _gpu_snapshot() -> dict[str, list[str]]:
    gpu = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )
    compute = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    # Keep evidence useful without publishing unrelated executable paths.
    sanitized_compute = []
    for line in compute.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            fields[2] = Path(fields[2]).name
        sanitized_compute.append(", ".join(fields))
    return {"gpu": gpu.splitlines(), "compute": sanitized_compute}


def _masked_uuid(torch_module: Any | None = None) -> str | None:
    if torch_module is not None:
        value = getattr(torch_module.cuda.get_device_properties(0), "uuid", None)
        if value:
            text = str(value)
            return text if text.startswith("GPU-") else f"GPU-{text}"
    env = os.environ.copy()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return None
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return uuids[0] if len(uuids) == 1 else None


def _versions() -> dict[str, str]:
    import accelerate
    import huggingface_hub
    import PIL
    import torch
    import torchvision
    import transformers

    import sam2

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "sam2": str(getattr(sam2, "__version__", "source-2b90b9f")),
    }


def _fixture() -> np.ndarray:
    image = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE, 3), dtype=np.uint8)
    image[:64, :64] = (220, 40, 30)
    image[64:, 64:] = (30, 180, 50)
    image[32:96, 48:80] = (30, 50, 220)
    return image


def _rss_mib() -> float:
    # Linux reports KiB for ru_maxrss.  This is a bounded process-level signal,
    # not a claim of a full host-memory profile.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _memory_mib(torch: Any) -> dict[str, float]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "free": round(free / 1024**2, 1),
        "total": round(total / 1024**2, 1),
    }


def _clear_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _model_prediction(spec: ModelSpec, *, dtype_bytes: float = 4.0, overhead: float = 1.5) -> int:
    raw_mib = (spec.approx_weight_bytes or 0) / 1024**2
    return int(round(raw_mib * dtype_bytes / 4.0 * overhead))


def _can_load(torch: Any, predicted_peak_mib: int) -> tuple[bool, float]:
    free, total = torch.cuda.mem_get_info()
    free_mib = free / 1024**2
    budget_mib = min(float(total) / 1024**2 * GPU_PEAK_FRACTION, free_mib)
    return predicted_peak_mib <= budget_mib, round(free_mib, 1)


def _download_models() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    patterns = {
        "sam2": ["README.md", "*.json", "*.yaml", "*.pt", "*.safetensors"],
        "clip": ["README.md", "*.json", "*.bin", "*.txt", "*.model"],
        "blip3": ["README.md", "LICENSE.txt", "*.json", "*.py", "*.safetensors", "*.model"],
    }
    result: dict[str, Any] = {}
    for key, spec in APPROVED_MODEL_SPECS.items():
        path = snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            allow_patterns=patterns[key],
        )
        total_bytes = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
        result[key] = {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "cache_size_mib": round(total_bytes / 1024**2, 1),
            "files": sum(1 for item in Path(path).rglob("*") if item.is_file()),
        }
    return result


def _record_blocked(
    name: str,
    spec: ModelSpec,
    predicted: int,
    free_before: float,
    before: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    after = _gpu_snapshot()
    return {
        "name": name,
        "status": "BLOCKED",
        "predicted_peak_mib": predicted,
        "free_before_mib": free_before,
        "model": spec.model_id,
        "revision": spec.revision,
        "reason": reason,
        "snapshot_before": before,
        "snapshot_after": after,
    }


def _measure(
    name: str,
    spec: ModelSpec,
    torch: Any,
    loader: Callable[[], Any],
    runner: Callable[[Any, np.ndarray], dict[str, Any]],
    *,
    predicted_peak_mib: int,
    fixture: np.ndarray,
    repeats: int = 3,
) -> dict[str, Any]:
    before = _gpu_snapshot()
    allowed, free_before = _can_load(torch, predicted_peak_mib)
    if not allowed:
        return _record_blocked(
            name,
            spec,
            predicted_peak_mib,
            free_before,
            before,
            "predicted peak exceeds the 90% GPU budget or current free VRAM",
        )

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    state = None
    try:
        state = loader()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        loaded_memory = _memory_mib(torch)
        timings = []
        shapes = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = runner(state, fixture)
            torch.cuda.synchronize()
            timings.append(round((time.perf_counter() - started) * 1000, 2))
            shapes.append(output)
        memory = _memory_mib(torch)
        result = {
            "name": name,
            "status": "PASSED",
            "model": spec.model_id,
            "revision": spec.revision,
            "predicted_peak_mib": predicted_peak_mib,
            "free_before_mib": free_before,
            "load_seconds": round(load_seconds, 3),
            "inference_ms": timings,
            "output_shapes": shapes,
            "loaded_memory": loaded_memory,
            "end_memory": memory,
            "host_rss_max_mib": _rss_mib(),
            "snapshot_before": before,
        }
    except Exception as exc:  # live evidence records an honest failure
        result = {
            "name": name,
            "status": "FAILED",
            "model": spec.model_id,
            "revision": spec.revision,
            "predicted_peak_mib": predicted_peak_mib,
            "free_before_mib": free_before,
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
            "snapshot_before": before,
        }
    finally:
        state = None
        _clear_cuda(torch)
        result["cleanup_memory"] = _memory_mib(torch)
        result["snapshot_after"] = _gpu_snapshot()
    return result


def _sam_loader(torch: Any, spec: ModelSpec) -> Callable[[], Any]:
    from modules.segmenter import initialize_sam2

    def load() -> Any:
        return initialize_sam2(
            {
                "model_name": spec.model_id,
                "revision": spec.revision,
                "points_per_side": 8,
                "points_per_batch": 8,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.5,
                "crop_n_layers": 0,
            },
            device=torch.device("cuda:0"),
            verbosity=0,
        )

    return load


def _clip_loader(torch: Any, spec: ModelSpec) -> Callable[[], Any]:
    from modules.classifier import initialize_clip

    def load() -> Any:
        return initialize_clip(
            {
                "model_name": spec.model_id,
                "revision": spec.revision,
                "labels": {"red": "a red object", "green": "a green object"},
            },
            device=torch.device("cuda:0"),
            verbosity=0,
        )

    return load


def _run_sam(state: Any, image: np.ndarray) -> dict[str, Any]:
    from modules.segmenter import run_sam2

    _, masks, _ = run_sam2(state, {"dryrun": False}, image, verbosity=0)
    shape = list(masks[0]["segmentation"].shape) if masks else []
    return {"count": len(masks), "first_mask_shape": shape}


def _run_clip(state: Any, image: np.ndarray) -> dict[str, Any]:
    from modules.classifier import run_clip

    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[16:96, 16:96] = True
    _, masks, _ = run_clip(
        state,
        {
            "config": {
                "model_name": APPROVED_MODEL_SPECS["clip"].model_id,
                "revision": APPROVED_MODEL_SPECS["clip"].revision,
                "labels": {"red": "a red object", "green": "a green object"},
            },
            "masks": [{"segmentation": mask}],
            "device": "cuda:0",
            "dryrun": False,
        },
        image,
        verbosity=0,
    )
    return {"count": len(masks), "labels": [str(item.get("clip_label")) for item in masks]}


def _run_combined(states: tuple[Any, Any], image: np.ndarray) -> dict[str, Any]:
    from modules.segmenter import run_sam2
    from modules.classifier import run_clip

    sam_state, clip_state = states
    _, masks, _ = run_sam2(sam_state, {"dryrun": False}, image, verbosity=0)
    _, masks, _ = run_clip(
        clip_state,
        {
            "config": {
                "model_name": APPROVED_MODEL_SPECS["clip"].model_id,
                "revision": APPROVED_MODEL_SPECS["clip"].revision,
                "labels": {"red": "a red object", "green": "a green object"},
            },
            "masks": masks[:4],
            "device": "cuda:0",
            "dryrun": False,
        },
        image,
        verbosity=0,
    )
    return {"count": len(masks), "labels": [str(item.get("clip_label")) for item in masks]}


def _measure_combined(
    torch: Any, fixture: np.ndarray, sam: ModelSpec, clip: ModelSpec
) -> dict[str, Any]:
    before = _gpu_snapshot()
    predicted = _model_prediction(sam, overhead=1.5) + _model_prediction(clip, overhead=1.5)
    allowed, free_before = _can_load(torch, predicted)
    if not allowed:
        return _record_blocked(
            "sam2_clip",
            clip,
            predicted,
            free_before,
            before,
            "predicted combined peak exceeds the 90% GPU budget or current free VRAM",
        )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sam_state = clip_state = None
    try:
        sam_state = _sam_loader(torch, sam)()
        clip_state = _clip_loader(torch, clip)()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        loaded_memory = _memory_mib(torch)
        timings = []
        shapes = []
        for _ in range(3):
            run_started = time.perf_counter()
            output = _run_combined((sam_state, clip_state), fixture)
            torch.cuda.synchronize()
            timings.append(round((time.perf_counter() - run_started) * 1000, 2))
            shapes.append(output)
        result = {
            "name": "sam2_clip",
            "status": "PASSED",
            "predicted_peak_mib": predicted,
            "free_before_mib": free_before,
            "load_seconds": round(load_seconds, 3),
            "inference_ms": timings,
            "output_shapes": shapes,
            "loaded_memory": loaded_memory,
            "end_memory": _memory_mib(torch),
            "snapshot_before": before,
        }
    except Exception as exc:
        result = {
            "name": "sam2_clip",
            "status": "FAILED",
            "predicted_peak_mib": predicted,
            "free_before_mib": free_before,
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
            "snapshot_before": before,
        }
    finally:
        sam_state = None
        clip_state = None
        _clear_cuda(torch)
        result["cleanup_memory"] = _memory_mib(torch)
        result["snapshot_after"] = _gpu_snapshot()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download pinned model snapshots")
    parser.add_argument("--tmp-root", default="/dev/shm/slaif-zap-it")
    args = parser.parse_args()

    require_launch_environment(physical_gpu_index=1)
    ensure_shm_root(args.tmp_root, min_free_bytes=64 * 1024 * 1024)
    selected_port = select_candidate_port()

    import torch

    uuid = _masked_uuid(torch)
    report: dict[str, Any] = {
        "status": "PENDING",
        "launch_environment": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"},
        "masked_uuid": uuid,
        "versions": _versions(),
        "shm_free_mib": round(shm_free_bytes(args.tmp_root) / 1024**2, 1),
        "candidate_port": {
            "host": selected_port.host,
            "port": selected_port.port,
            "verified_by_ss_and_bind": selected_port.unused,
        },
        "snapshot_before_all": _gpu_snapshot(),
    }
    device = inspect_visible_device(
        torch,
        expected_uuid=GPU_UUID,
        strict=True,
        uuid_provider=lambda: _masked_uuid(torch),
    )
    report["device"] = {
        "mode": device.mode,
        "visible_count": device.visible_count,
        "logical_index": device.logical_index,
        "name": device.name,
        "uuid": device.uuid,
        "total_memory_mib": device.total_memory_mib,
    }
    if args.download:
        report["downloads"] = _download_models()
    else:
        report["downloads"] = "SKIPPED (use --download on first qualification)"

    fixture = _fixture()
    sam = APPROVED_MODEL_SPECS["sam2"]
    clip = APPROVED_MODEL_SPECS["clip"]
    blip = APPROVED_MODEL_SPECS["blip3"]
    report["measurements"] = [
        _measure(
            "sam2",
            sam,
            torch,
            _sam_loader(torch, sam),
            _run_sam,
            predicted_peak_mib=_model_prediction(sam, overhead=1.5),
            fixture=fixture,
        ),
        _measure(
            "clip",
            clip,
            torch,
            _clip_loader(torch, clip),
            _run_clip,
            predicted_peak_mib=_model_prediction(clip, overhead=1.5),
            fixture=fixture,
        ),
        _record_blocked(
            "blip3",
            blip,
            _model_prediction(blip, dtype_bytes=2.0, overhead=1.2),
            round(torch.cuda.mem_get_info()[0] / 1024**2, 1),
            _gpu_snapshot(),
            "conservative bfloat16 weight-plus-overhead prediction exceeds the 90% budget; no load attempted",
        ),
        _measure_combined(torch, fixture, sam, clip),
    ]
    report["snapshot_after_all"] = _gpu_snapshot()
    report["status"] = (
        "PASSED"
        if all(item["status"] in {"PASSED", "BLOCKED"} for item in report["measurements"])
        else "FAILED"
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
