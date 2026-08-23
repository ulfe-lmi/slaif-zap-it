"""Operator wiring that turns the qualified GPU runtime into one loopback service.

This module is the objective 004 activation layer between the measured
objective 003 runtime and the objective 002 HTTP contract. It owns:

- operator-only launch configuration (strict physical-GPU1 masking, pinned
  GPU UUID, verified loopback port, RAM-backed temp root);
- a fail-closed preflight that runs before any CUDA library is imported;
- a resident model registry that loads the supported ``sam2_clip`` profile
  exactly once per process on ``cuda:0`` using the pinned model identities;
- an engine adapter forwarding resident states into the pure
  :func:`src.core.engine.run_single_image` pipeline;
- readiness composition with honest not-ready/failed transitions;
- the ``main()`` entrypoint used by ``scripts/serve_local.py``.

Nothing here persists request data or changes any protected host resource.
Heavy dependencies (torch, sam2, transformers) are imported lazily so CPU
test environments can exercise every seam without them.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .device import DeviceGuardError, inspect_visible_device, require_launch_environment
from .models import APPROVED_MODEL_SPECS
from .ports import PortCheck, verify_port_unused
from .shm import ShmError, ensure_shm_root, shm_free_bytes
from .strategy import (
    SUPPORTED_RESIDENT_PROFILES,
    SUPPORTED_RESIDENT_STRATEGY,
    RuntimePolicy,
    RuntimeReadiness,
)

__all__ = [
    "LiveServiceConfig",
    "LiveServiceError",
    "PreflightReport",
    "ResidentRegistry",
    "build_device_provider",
    "compose_readiness",
    "default_resident_loader",
    "live_engine_callable",
    "main",
    "masked_gpu_uuid",
    "preflight",
    "wrap_test_injection",
]

_LOOPBACK_HOST = "127.0.0.1"
_PHYSICAL_GPU_INDEX = 1
_SHM_MIN_FREE_BYTES = 64 * 1024 * 1024


class LiveServiceError(RuntimeError):
    """Raised when operator preflight or startup validation fails."""


@dataclass(frozen=True)
class LiveServiceConfig:
    """Validated operator-owned launch configuration."""

    host: str = _LOOPBACK_HOST
    port: int = 17891
    tmp_root: str = "/dev/shm/slaif-zap-it"
    model_cache_root: str | None = None
    expected_gpu_uuid: str | None = None
    physical_gpu_index: int = _PHYSICAL_GPU_INDEX
    strict_gpu: bool = True

    def __post_init__(self) -> None:
        if self.host != _LOOPBACK_HOST:
            raise LiveServiceError("the local service may only bind 127.0.0.1 in this objective")
        if not 1 <= int(self.port) <= 65535:
            raise LiveServiceError("SLAIF_ZAP_IT_PORT must be a valid TCP port")
        if int(self.physical_gpu_index) < 0:
            raise LiveServiceError("physical GPU index must be non-negative")
        if self.strict_gpu and int(self.physical_gpu_index) != _PHYSICAL_GPU_INDEX:
            raise LiveServiceError("the live service may expose only physical GPU index 1")
        if self.strict_gpu and not (self.expected_gpu_uuid or "").strip():
            raise LiveServiceError("strict GPU operation requires SLAIF_ZAP_IT_EXPECTED_GPU_UUID")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "LiveServiceConfig":
        env = os.environ if environ is None else environ
        raw_port = (env.get("SLAIF_ZAP_IT_PORT") or "").strip()
        if not raw_port:
            raise LiveServiceError(
                "SLAIF_ZAP_IT_PORT must be set to a verified-unused loopback port"
            )
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise LiveServiceError("SLAIF_ZAP_IT_PORT must be an integer") from exc
        raw_index = (env.get("SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX") or "1").strip()
        try:
            physical = int(raw_index)
        except ValueError as exc:
            raise LiveServiceError("SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX must be an integer") from exc
        strict_raw = (env.get("SLAIF_ZAP_IT_STRICT_GPU") or "1").strip().lower()
        strict = strict_raw not in {"0", "false", "no", "off"}
        return cls(
            host=(env.get("SLAIF_ZAP_IT_HOST") or _LOOPBACK_HOST).strip(),
            port=port,
            tmp_root=(env.get("SLAIF_ZAP_IT_TMP_ROOT") or "/dev/shm/slaif-zap-it").strip(),
            model_cache_root=(env.get("SLAIF_ZAP_IT_MODEL_CACHE_ROOT") or None),
            expected_gpu_uuid=(env.get("SLAIF_ZAP_IT_EXPECTED_GPU_UUID") or None),
            physical_gpu_index=physical,
            strict_gpu=strict,
        )


@dataclass(frozen=True)
class PreflightReport:
    """Sanitized pre-launch evidence; safe for logs and reports."""

    launch_environment_ok: bool
    shm_root: str
    shm_free_mib: float
    port_check: PortCheck


def preflight(config: LiveServiceConfig) -> PreflightReport:
    """Validate environment mask, temp root and port before CUDA imports."""
    try:
        require_launch_environment(None, physical_gpu_index=config.physical_gpu_index)
    except DeviceGuardError as exc:
        raise LiveServiceError(f"launch environment check failed: {exc}") from exc
    try:
        root = ensure_shm_root(config.tmp_root)
    except ShmError as exc:
        raise LiveServiceError(f"shared-memory workspace check failed: {exc}") from exc
    free_bytes = shm_free_bytes(root)
    if free_bytes < _SHM_MIN_FREE_BYTES:
        raise LiveServiceError("shared-memory workspace has insufficient free capacity")
    port_check = verify_port_unused(config.host, config.port)
    if not port_check.unused:
        raise LiveServiceError(
            f"{config.host}:{config.port} is not a freshly verified-unused loopback port"
        )
    return PreflightReport(
        launch_environment_ok=True,
        shm_root=str(root),
        shm_free_mib=round(free_bytes / (1024 * 1024), 1),
        port_check=port_check,
    )


def _startup_log_line(config: LiveServiceConfig, report: PreflightReport) -> str:
    """Format startup evidence without disclosing operator filesystem paths."""
    return (
        "serve_local: pid={pid} host={host}:{port} shm_ready=true shm_free_mib={shm_free}"
    ).format(
        pid=os.getpid(),
        host=config.host,
        port=config.port,
        shm_free=report.shm_free_mib,
    )


def masked_gpu_uuid(
    nvidia_smi: str = "nvidia-smi", *, torch_module: Any | None = None
) -> str | None:
    """Return the UUID of the one visible GPU, preferring masked torch state.

    Some NVIDIA utility versions report every physical GPU even when a child
    process has ``CUDA_VISIBLE_DEVICES=1``. PyTorch's device properties are
    already in the masked logical view and are therefore authoritative here;
    the ``nvidia-smi`` path remains a useful fallback when no torch module is
    available.
    """
    if torch_module is not None:
        try:
            value = getattr(torch_module.cuda.get_device_properties(0), "uuid", None)
        except Exception:
            value = None
        if value:
            text = str(value)
            return text if text.startswith("GPU-") else f"GPU-{text}"
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return uuids[0] if len(uuids) == 1 else None


class ResidentRegistry:
    """One-per-process home of the resident SAM2+CLIP model states.

    The registry loads the supported profile once in a worker thread while
    the HTTP surface already serves honest ``not_ready`` answers. It never
    reloads after success and records only sanitized failure categories.
    """

    def __init__(
        self,
        *,
        loader: Callable[[], dict[str, Any]],
        strategy: str = "sam2_clip_resident_blip3_rejected",
    ) -> None:
        self._loader = loader
        self._strategy = strategy
        self._lock = threading.Lock()
        self._states: dict[str, Any] | None = None
        self._ready = False
        self._failed = False
        self._error_type: str | None = None
        self.load_seconds: float | None = None
        self._load_thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed

    @property
    def error_type(self) -> str | None:
        with self._lock:
            return self._error_type

    def verdict(self) -> RuntimeReadiness:
        """Honest registry-only readiness verdict."""
        with self._lock:
            if self._ready:
                return RuntimeReadiness(True, f"resident {self._strategy} models are ready")
            if self._failed:
                return RuntimeReadiness(
                    False,
                    f"resident model load failed ({self._error_type}); see operator runbook",
                )
            return RuntimeReadiness(False, "resident model registry is still loading")

    def wait_until_settled(self, timeout: float = 600.0) -> bool:
        """Block until ready/failed; returns False on timeout without lying."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready or self.failed:
                return True
            time.sleep(0.05)
        return False

    def load(self) -> None:
        import time

        started = time.perf_counter()
        try:
            states = self._loader()
        except Exception as exc:  # recorded as a sanitized category, never rethrown
            with self._lock:
                self._failed = True
                self._error_type = type(exc).__name__
            return
        with self._lock:
            if not isinstance(states, dict) or "segmenter" not in states or "clip" not in states:
                self._failed = True
                self._error_type = "ValueError"
                return
            self._states = states
            self._ready = True
            self.load_seconds = round(time.perf_counter() - started, 3)

    def start_background_load(self) -> threading.Thread:
        with self._lock:
            if self._load_thread is not None:
                return self._load_thread
        thread = threading.Thread(target=self.load, name="zap-it-model-load", daemon=True)
        with self._lock:
            self._load_thread = thread
        thread.start()
        return thread

    def shutdown(self, timeout: float = 60.0) -> None:
        """Wait briefly for the one model-loading thread during process stop."""
        with self._lock:
            thread = self._load_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(float(timeout), 0.0))

    def states(self) -> dict[str, Any]:
        with self._lock:
            if self._states is None:
                raise LiveServiceError("resident model states are not loaded")
            return self._states


def default_resident_loader(
    *, device_name: str = "cuda:0", model_cache_root: str | None = None
) -> Callable[[], dict[str, Any]]:
    """Build the pinned SAM2+CLIP loader for the supported resident profile."""

    def load() -> dict[str, Any]:
        # Approved Objective 003 snapshots are operator assets.  A live
        # request must never cause a model download; missing snapshots simply
        # keep the registry not-ready.
        if model_cache_root:
            os.environ["HF_HOME"] = model_cache_root
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(model_cache_root) / "hub")
        import torch
        from modules.classifier import initialize_clip
        from modules.segmenter import initialize_sam2

        device = torch.device(device_name)
        sam_spec = APPROVED_MODEL_SPECS["sam2"]
        clip_spec = APPROVED_MODEL_SPECS["clip"]
        segmenter_state = initialize_sam2(
            {
                "model_name": sam_spec.model_id,
                "revision": sam_spec.revision,
                "points_per_side": 8,
                "points_per_batch": 8,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.5,
                "crop_n_layers": 0,
            },
            device=device,
            verbosity=0,
            local_files_only=True,
        )
        clip_state = initialize_clip(
            {
                "model_name": clip_spec.model_id,
                "revision": clip_spec.revision,
                # Resident CLIP loads without text prompts; effective labels
                # arrive per request through the uploaded YAML.
                "labels": {},
            },
            device=device,
            verbosity=0,
            local_files_only=True,
        )
        return {"segmenter": segmenter_state, "clip": clip_state}

    return load


def live_engine_callable(
    registry: ResidentRegistry,
    *,
    runner: Optional[Callable[..., Any]] = None,
    device_name: str = "cuda:0",
) -> Callable[..., Any]:
    """Adapt resident registry states into the pure engine signature.

    The HTTP transport passes placeholder ``None`` states and no device; this
    adapter replaces them with the process-resident objects and pins logical
    ``cuda:0``. Request state stays fresh per call; only reusable model
    holders are shared across requests.
    """

    def engine(
        image_rgb: Any,
        config: Any,
        *,
        frame_id: str = "image",
        segmenter_state: Optional[dict] = None,
        clip_state: Optional[dict] = None,
        blip3_state: Optional[dict] = None,
        dryrun: bool = False,
        verbosity: int = 1,
        device: Optional[Any] = None,
        log_print_func: Optional[Callable[..., None]] = None,
        artifact_sink: Any = None,
        stages: Any = None,
        class_labels: Any = (),
        **_: Any,
    ) -> Any:
        from src.service.errors import ServiceError

        del segmenter_state, clip_state, blip3_state, device  # replaced by resident values
        sam2_cfg = getattr(config, "sam2_cfg", None) or {}
        mutable_generator_keys = sorted(str(key) for key in sam2_cfg if key != "debug")
        if mutable_generator_keys:
            raise ServiceError(
                "request-level SAM2 generator parameters are fixed by the resident "
                "runtime: " + ", ".join(mutable_generator_keys),
                code="unsupported_field",
            )
        states = registry.states()
        if stages is not None:
            raise ServiceError(
                "custom stage sets are not accepted by the resident runtime",
                code="unsupported_field",
            )
        if runner is not None:
            resolved_runner = runner
        else:
            from src.core.engine import run_single_image as resolved_runner

        return resolved_runner(
            image_rgb,
            config,
            frame_id=frame_id,
            segmenter_state=states["segmenter"],
            clip_state=states["clip"],
            blip3_state=None,
            dryrun=dryrun,
            verbosity=verbosity,
            device=device_name,
            log_print_func=log_print_func,
            artifact_sink=artifact_sink,
            stages=None,
            class_labels=tuple(class_labels),
        )

    return engine


def wrap_test_injection(engine: Callable[..., Any]) -> Callable[..., Any]:
    """Add opt-in operator-only failure/delay injection for live verification.

    The hook is inert unless ``SLAIF_ZAP_IT_TEST_INJECT`` or
    ``SLAIF_ZAP_IT_TEST_DELAY_SECONDS`` is explicitly set by the operator. It
    is process-wide and never request-selectable; normal launches do not use it.
    """
    mode = (os.environ.get("SLAIF_ZAP_IT_TEST_INJECT") or "").strip().lower()
    raw_delay = (os.environ.get("SLAIF_ZAP_IT_TEST_DELAY_SECONDS") or "0").strip()
    try:
        delay = max(float(raw_delay), 0.0)
    except ValueError:
        delay = 0.0

    if not mode and delay <= 0:
        return engine

    def injected(*args: Any, **kwargs: Any) -> Any:
        if mode in {"failure", "fail", "inference_failure"}:
            raise RuntimeError("operator-injected inference failure")
        if mode in {"delay", "timeout", "cancel"} and delay > 0:
            time.sleep(delay)
        return engine(*args, **kwargs)

    return injected


def compose_readiness(
    device_provider: Callable[[], RuntimeReadiness],
    registry: ResidentRegistry,
) -> Callable[[], RuntimeReadiness]:
    """Join honest registry state with fail-closed device evidence."""

    def provider() -> RuntimeReadiness:
        registry_verdict = registry.verdict()
        if not registry_verdict.ready:
            return registry_verdict
        return device_provider()

    return provider


def build_device_provider(
    policy: RuntimePolicy,
    *,
    torch_module: Any,
    uuid_provider: Callable[[], str | None],
) -> Callable[[], RuntimeReadiness]:
    """Wrap the pinned device guard as a service readiness provider."""
    from .readiness import make_readiness_provider

    return make_readiness_provider(
        policy.with_model_registry_ready(True),
        torch_module=torch_module,
        uuid_provider=uuid_provider,
    )


def main() -> int:
    """Run exactly one loopback ZAP-IT service process (operator entrypoint).

    Exit codes: 2 = configuration/preflight failure, 3 = device guard
    failure. Uvicorn serves until SIGTERM/SIGINT triggers graceful shutdown.
    """
    import sys

    try:
        config = LiveServiceConfig.from_environment()
    except LiveServiceError as exc:
        print(f"serve_local: {exc}", file=sys.stderr)
        return 2
    if not config.strict_gpu:
        print("serve_local: strict physical GPU1 mode cannot be disabled", file=sys.stderr)
        return 2
    if config.model_cache_root:
        # Set the operator cache before importing model libraries.  The value
        # is never printed or returned to a client.
        os.environ["HF_HOME"] = config.model_cache_root
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(config.model_cache_root) / "hub")
    try:
        report = preflight(config)
    except LiveServiceError as exc:
        print(f"serve_local: {exc}", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print("serve_local: PyTorch runtime environment is unavailable", file=sys.stderr)
        return 3

    try:
        policy = RuntimePolicy.from_environment(expected_gpu_uuid=config.expected_gpu_uuid)
    except (TypeError, ValueError) as exc:
        print(f"serve_local: invalid runtime policy: {exc}", file=sys.stderr)
        return 2
    if (
        policy.strategy != SUPPORTED_RESIDENT_STRATEGY
        or policy.supported_profiles != SUPPORTED_RESIDENT_PROFILES
        or policy.physical_gpu_index != _PHYSICAL_GPU_INDEX
        or not policy.strict_gpu
    ):
        print("serve_local: runtime policy is not the qualified resident profile", file=sys.stderr)
        return 2

    def uuid_provider() -> str | None:
        return masked_gpu_uuid(torch_module=torch)

    try:
        device_report = inspect_visible_device(
            torch,
            expected_uuid=config.expected_gpu_uuid,
            strict=True,
            uuid_provider=uuid_provider,
        )
    except DeviceGuardError as exc:
        print(f"serve_local: device guard failed: {exc}", file=sys.stderr)
        return 3

    from src.service.app import create_app
    from src.service.settings import ServiceSettings

    registry = ResidentRegistry(
        loader=default_resident_loader(model_cache_root=config.model_cache_root),
        strategy=policy.strategy,
    )
    try:
        settings = ServiceSettings.from_environment()
    except (TypeError, ValueError) as exc:
        print(f"serve_local: invalid service settings: {exc}", file=sys.stderr)
        return 2
    runtime_metadata = {
        "strategy": policy.strategy,
        "supported_profiles": list(policy.supported_profiles),
        "device": {
            "physical_index": config.physical_gpu_index,
            "logical": "cuda:0",
            "visible_count": device_report.visible_count,
            "uuid": device_report.uuid,
            "name": device_report.name,
            "total_memory_mib": device_report.total_memory_mib,
        },
        "models": {
            name: {"id": spec.model_id, "revision": spec.revision}
            for name, spec in APPROVED_MODEL_SPECS.items()
            if name in {"sam2", "clip"}
        },
    }
    app = create_app(
        engine=wrap_test_injection(live_engine_callable(registry)),
        settings=settings,
        readiness_provider=compose_readiness(
            build_device_provider(policy, torch_module=torch, uuid_provider=uuid_provider),
            registry,
        ),
        runtime_policy=policy,
        runtime_metadata=runtime_metadata,
        shutdown_callback=registry.shutdown,
    )

    # Sanitized facts only: identifiers, counts, timings. Never secrets.
    print(_startup_log_line(config, report), flush=True)
    print(
        "serve_local: device mode={mode} count={count} logical={logical} "
        "name={name} uuid={uuid} total_mib={total} strategy={strategy} "
        "profiles={profiles}".format(
            mode=device_report.mode,
            count=device_report.visible_count,
            logical=device_report.logical_index,
            name=device_report.name,
            uuid=device_report.uuid,
            total=device_report.total_memory_mib,
            strategy=policy.strategy,
            profiles=",".join(policy.supported_profiles),
        ),
        flush=True,
    )

    registry.start_background_load()

    import uvicorn

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        reload=False,
        log_level="info",
        access_log=True,
    )
    return 0
