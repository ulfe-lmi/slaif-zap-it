"""FastAPI application factory implementing the ``/v1/completions`` contract.

The transport layer owns strict multipart parsing, hostile-input validation,
bounded concurrency, deadlines/cancellation, stable sanitized errors and the
optional API key. Inference is delegated to an injected engine callable with
the :func:`src.core.engine.run_single_image` signature; this objective ships
and documents only the deterministic CPU :class:`~src.service.fake_engine.FakeEngine`.
Everything lives in memory; the service never persists request data.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from src.core.config import CoreConfig, config_digest
from src.core.clip_prompts import ClipPromptValidationError
from src.core.sinks import ArtifactBudget, ArtifactSinkError, BoundedMemoryArtifactSink
from src.runtime.strategy import RuntimePolicy, UnsupportedProfileError

from .auth import verify_bearer_key
from .artifacts import ArtifactDeliveryLedger, ArtifactSelection
from .capabilities import CapabilitiesResponse, build_capabilities
from .envelope import (
    SCHEMA_VERSION,
    ResponseContext,
    bound_json_size,
    build_completion_json,
    build_completion_zip,
)
from .errors import ServiceError, error_envelope
from .fake_engine import EngineCallable, FakeEngine
from .gate import InferenceGate
from .image_input import decode_image_safely
from .metrics import CONTENT_TYPE_LATEST, ServiceMetrics
from .multipart import parse_strict_multipart
from .resources import check_request_resources
from .schemas import (
    CompletionResponse,
    ErrorEnvelope,
    HealthStatus,
    OpenAIErrorEnvelope,
    ResponsesRequest,
    ResponsesResponse,
)
from .responses import (
    ResponsesAuthenticationError,
    build_public_projection,
    build_responses_response,
    parse_responses_request,
    responses_error_body,
    responses_request_body_limit,
)
from .settings import ServiceSettings
from .yaml_input import parse_hostile_config

__all__ = ["ReadyState", "default_readiness_provider", "create_app"]

ReadinessProvider = Callable[[], "ReadyState"]


@dataclass(frozen=True)
class ReadyState:
    """Honest readiness verdict supplied by an injectable provider."""

    ready: bool
    detail: str = ""


@dataclass(frozen=True)
class _InferenceRun:
    """Request-local result shared by the native and Responses transports."""

    image_rgb: Any
    validated: Any
    core_config: CoreConfig
    outcome: Any
    sink: BoundedMemoryArtifactSink
    delivery_ledger: ArtifactDeliveryLedger
    class_labels: List[str]


def default_readiness_provider() -> ReadyState:
    return ReadyState(ready=False, detail="no readiness provider configured")


def _new_request_id() -> str:
    return secrets.token_hex(8)


def _request_id_for(request: Request) -> str:
    cached = getattr(request.state, "request_id", None)
    if not cached:
        cached = _new_request_id()
        request.state.request_id = cached
    return cached


def _error_response(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.envelope(_request_id_for(request)),
        headers=dict(exc.headers or {}),
    )


async def _run_engine_bounded(
    engine: EngineCallable,
    *,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    image_rgb: Any,
    core_config: CoreConfig,
    frame_id: str,
    verbosity: int,
    sink: BoundedMemoryArtifactSink,
    class_labels: List[str],
    timeout_seconds: float,
    render_visualizations: bool,
) -> Any:
    """Run one inference in the single-slot executor under the deadline."""

    def _call() -> Any:
        return engine(
            image_rgb,
            core_config,
            frame_id=frame_id,
            segmenter_state=None,
            clip_state=None,
            blip3_state=None,
            dryrun=False,
            verbosity=verbosity,
            device=None,
            log_print_func=None,
            artifact_sink=sink,
            stages=None,
            class_labels=tuple(class_labels),
            render_visualizations=render_visualizations,
            service_safe_artifact_names=True,
        )

    future = loop.run_in_executor(executor, _call)
    try:
        # Cancellation of an asyncio Future cannot stop a synchronous model
        # call already running in a worker thread. Shield it, drain it before
        # releasing the inference gate, and keep the one-model/one-request
        # invariant true even on timeout or client disconnect.
        return await asyncio.wait_for(asyncio.shield(future), timeout=max(timeout_seconds, 0.0))
    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            await asyncio.shield(future)
        except BaseException:
            # The outer request returns a sanitized timeout/cancel result; the
            # worker's exception must not escape as a second response failure.
            pass
        raise


def create_app(
    *,
    engine: EngineCallable,
    settings: Optional[ServiceSettings] = None,
    readiness_provider: Optional[ReadinessProvider] = None,
    runtime_policy: Optional[RuntimePolicy] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    shutdown_callback: Optional[Callable[[], None]] = None,
    model_registry: Any | None = None,
    lifecycle_controller: Any | None = None,
    enable_docs: bool = True,
    clip_prompt_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> FastAPI:
    """Build the service app around an explicitly injected engine.

    ``runtime_policy`` is operator-owned startup state.  It is deliberately
    separate from the uploaded YAML so a client cannot enable an unqualified
    model combination or alter device/resource settings.
    """
    resolved_settings = settings or ServiceSettings()
    readiness = readiness_provider or default_readiness_provider
    gate = InferenceGate(
        queue_depth=resolved_settings.queue_depth,
        retry_after_seconds=resolved_settings.retry_after_seconds,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zap-it-inference")
    control_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zap-it-model-control")
    metrics = ServiceMetrics()
    controller = lifecycle_controller
    if (
        controller is None
        and model_registry is not None
        and resolved_settings.model_control_mode == "explicit"
    ):
        from .model_control import ModelLifecycleController

        controller = ModelLifecycleController(
            registry=model_registry,
            gate=gate,
            control_executor=control_executor,
            operation_timeout_seconds=resolved_settings.model_control_operation_seconds,
            drain_timeout_seconds=resolved_settings.model_control_drain_seconds,
            metrics=metrics,
        )
    if controller is not None and resolved_settings.model_control_mode == "explicit":
        if controller.is_ready:
            gate.resume_now()
        else:
            gate.pause_now("model is not ready")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if controller is not None:
                await controller.shutdown()
            elif shutdown_callback is not None:
                shutdown_callback()
            executor.shutdown(wait=True, cancel_futures=True)
            control_executor.shutdown(wait=True, cancel_futures=True)

    def require_api_key(request: Request) -> None:
        expected = resolved_settings.api_key
        if expected is None:
            return
        verify_bearer_key(request.headers.get("authorization"), expected)

    def require_capabilities_key(request: Request) -> None:
        """Capabilities are always an authenticated operator-policy view."""
        expected = resolved_settings.api_key
        if expected is None:
            raise ServiceError("missing or invalid API credentials", code="unauthorized")
        verify_bearer_key(request.headers.get("authorization"), expected)

    def require_responses_api_key(request: Request) -> None:
        """Authenticate Responses before FastAPI or the handler reads JSON."""
        expected = resolved_settings.api_key
        if expected is None:
            return
        try:
            verify_bearer_key(request.headers.get("authorization"), expected)
        except ServiceError as exc:
            raise ResponsesAuthenticationError() from exc

    def record_error(request: Request, code: str) -> None:
        if getattr(request.state, "metrics_recorded", False):
            return
        metrics.observe_error(code)
        started = getattr(request.state, "metrics_started", None)
        if started is not None:
            metrics.request_duration.observe(time.monotonic() - started)
        serialization_started = getattr(request.state, "metrics_serialization_started", None)
        if serialization_started is not None and not getattr(
            request.state, "metrics_serialization_recorded", False
        ):
            metrics.serialization_duration.observe(time.monotonic() - serialization_started)
            request.state.metrics_serialization_recorded = True
        request.state.metrics_recorded = True

    app = FastAPI(
        title="SLAIF ZAP-IT API",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    app.state.settings = resolved_settings
    app.state.gate = gate
    app.state.runtime_policy = runtime_policy
    app.state.started_monotonic = time.monotonic()
    app.state.metrics = metrics
    app.state.model_controller = controller

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        record_error(request, exc.code)
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request.state.request_id = _request_id_for(request)
        return JSONResponse(
            status_code=400,
            content=error_envelope(
                "invalid_multipart", "malformed request", _request_id_for(request)
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(
                "http_error", "request could not be routed", _request_id_for(request)
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        record_error(request, "internal_error")
        return JSONResponse(
            status_code=500,
            content=error_envelope(
                "internal_error",
                "unexpected internal failure",
                _request_id_for(request),
            ),
        )

    @app.exception_handler(ResponsesAuthenticationError)
    async def handle_responses_authentication(
        request: Request, exc: ResponsesAuthenticationError
    ) -> JSONResponse:
        del exc
        request_id = _request_id_for(request)
        metrics.observe_response_error()
        return JSONResponse(
            status_code=401,
            content=responses_error_body(
                ServiceError("authentication failed", code="unauthorized"), request_id
            ),
            headers={"x-request-id": request_id},
        )

    async def run_shared_inference(
        request: Request,
        *,
        image_bytes: bytes,
        config_bytes: bytes,
        verbosity: int,
        request_id: str,
        deadline_monotonic: float,
        render_visualizations: bool,
    ) -> _InferenceRun:
        """Decode, validate and execute one request through the shared path.

        Both HTTP surfaces enter here with bounded bytes.  Keeping readiness,
        CLIP preflight, gate admission and the executor in one helper prevents
        a compatibility adapter from creating a second model or concurrency
        authority.
        """

        settings_local = resolved_settings

        def check_deadline() -> None:
            if time.monotonic() >= deadline_monotonic:
                raise ServiceError("request deadline exceeded", code="timeout")

        check_request_resources(settings_local)
        check_deadline()
        image_rgb = decode_image_safely(
            image_bytes,
            max_decoded_pixels=settings_local.max_decoded_pixels,
            max_width=settings_local.max_image_width,
            max_height=settings_local.max_image_height,
        )
        check_deadline()
        validated = parse_hostile_config(
            config_bytes,
            verbosity=verbosity,
            max_visualization_streams=settings_local.max_visualization_streams,
            settings=settings_local,
        )
        check_deadline()
        core_config = CoreConfig.from_mapping(validated.effective_mapping)
        core_config = dataclass_replace(
            core_config,
            sam2_metadata=validated.sam2_metadata,
            clip_prompt_metadata=validated.clip_prompt_metadata,
        )
        if runtime_policy is not None:
            try:
                runtime_policy.validate_config(core_config)
            except UnsupportedProfileError as exc:
                raise ServiceError(str(exc), code="unsupported_profile") from exc
        check_deadline()

        if controller is not None and not controller.is_ready:
            ready = ReadyState(False, controller.readiness_detail())
        else:
            ready = readiness()
        if not ready.ready:
            metrics.readiness.set(0)
            raise ServiceError(ready.detail or "engine not ready", code="not_ready")
        metrics.readiness.set(1)

        class_labels = list(validated.class_labels)
        delivery_ledger = ArtifactDeliveryLedger(
            ArtifactSelection.from_mapping(
                core_config.diagnostic_artifacts,
                applied=verbosity >= 3,
            ),
            max_response_artifacts=settings_local.max_response_artifacts,
            max_debug_artifacts=settings_local.max_debug_artifacts,
            max_single_artifact_bytes=settings_local.max_single_artifact_bytes,
            max_total_raw_artifact_bytes=settings_local.max_total_raw_artifact_bytes,
            max_response_bytes=settings_local.max_response_bytes,
            verbosity=verbosity,
        )
        sink = BoundedMemoryArtifactSink(
            ArtifactBudget(
                max_artifacts=settings_local.max_debug_artifacts,
                max_single_bytes=settings_local.max_single_artifact_bytes,
                max_total_bytes=settings_local.max_total_raw_artifact_bytes,
            ),
            admission=delivery_ledger,
        )
        loop = asyncio.get_running_loop()
        try:
            async with gate.slot():
                metrics.active.set(1)
                budget = deadline_monotonic - time.monotonic()
                if budget <= 0:
                    raise ServiceError("request deadline exceeded", code="timeout")
                if clip_prompt_validator is not None and core_config.clip_cfg:
                    try:
                        clip_prompt_validator(core_config.clip_cfg)
                    except ClipPromptValidationError as exc:
                        raise ServiceError(
                            exc.message,
                            code="invalid_config",
                            details=exc.details,
                        ) from exc
                    except Exception as exc:
                        raise ServiceError(
                            "CLIP prompt preflight failed",
                            code="inference_failure",
                        ) from exc
                try:
                    inference_started = time.monotonic()
                    metrics.reset_gpu_peaks()
                    outcome = await _run_engine_bounded(
                        engine,
                        loop=loop,
                        executor=executor,
                        image_rgb=image_rgb,
                        core_config=core_config,
                        frame_id=request_id,
                        verbosity=verbosity,
                        sink=sink,
                        class_labels=class_labels,
                        timeout_seconds=budget,
                        render_visualizations=render_visualizations,
                    )
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    metrics.update_gpu()
                except asyncio.TimeoutError as exc:
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    raise ServiceError(
                        "inference exceeded the request deadline", code="timeout"
                    ) from exc
                except asyncio.CancelledError as exc:
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    raise ServiceError(
                        "request was cancelled before completion", code="cancelled"
                    ) from exc
                except ServiceError:
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    raise
                except ArtifactSinkError as exc:
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    raise ServiceError(
                        "artifact budget exceeded", code="response_too_large"
                    ) from exc
                except Exception as exc:
                    metrics.inference_duration.observe(time.monotonic() - inference_started)
                    raise ServiceError(
                        "inference failed inside the engine",
                        code="inference_failure",
                    ) from exc
        except asyncio.CancelledError as exc:
            raise ServiceError("request was cancelled before completion", code="cancelled") from exc
        finally:
            metrics.active.set(0)

        return _InferenceRun(
            image_rgb=image_rgb,
            validated=validated,
            core_config=core_config,
            outcome=outcome,
            sink=sink,
            delivery_ledger=delivery_ledger,
            class_labels=class_labels,
        )

    @app.get("/healthz", response_model=HealthStatus, tags=["health"])
    async def healthz() -> Dict[str, str]:
        uptime = time.monotonic() - app.state.started_monotonic
        return {"status": "ok", "uptime_s": f"{uptime:.3f}"}

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        tags=["capabilities"],
        dependencies=[Depends(require_capabilities_key)],
    )
    async def capabilities() -> Dict[str, Any]:
        """Return static operator policy without readiness or inference admission."""
        return build_capabilities(resolved_settings)

    def repository_error(status_code: int, message: str) -> JSONResponse:
        """Return the KServe/Triton repository-extension error shape."""
        return JSONResponse(status_code=status_code, content={"error": message})

    def require_control_key(request: Request) -> None:
        if resolved_settings.model_control_mode != "explicit":
            raise ServiceError(
                "explicit model control is disabled",
                code="model_control_disabled",
            )
        if controller is None:
            raise ServiceError("model control is not configured", code="model_control_failure")
        from .auth import verify_bearer_key

        verify_bearer_key(
            request.headers.get("authorization"),
            resolved_settings.model_control_api_key or "",
        )

    async def repository_body(request: Request, *, allow_empty: bool) -> dict[str, Any]:
        if request.query_params:
            raise ServiceError(
                "repository request parameters are not supported", code="unsupported_field"
            )
        body = await request.body()
        if len(body) > 16 * 1024:
            raise ServiceError("repository request body is too large", code="payload_too_large")
        if not body.strip():
            if allow_empty:
                return {}
            raise ServiceError(
                "repository request body must be an object", code="invalid_multipart"
            )
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"application/json", ""}:
            raise ServiceError("repository request body must be JSON", code="invalid_multipart")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceError(
                "repository request body is malformed", code="invalid_multipart"
            ) from exc
        if not isinstance(value, dict):
            raise ServiceError(
                "repository request body must be an object", code="invalid_multipart"
            )
        return value

    def repository_status(exc: ServiceError) -> JSONResponse:
        return repository_error(exc.status_code, exc.message)

    @app.get("/v2", tags=["model-management"])
    async def v2_metadata() -> Dict[str, Any]:
        return {
            "name": "slaif-zap-it",
            "version": SCHEMA_VERSION,
            "extensions": ["model_repository"],
            "model_repository": {
                "management_subset": True,
                "inference_endpoint": "/v1/completions",
                "v2_tensor_inference": False,
            },
        }

    @app.post("/v2/repository/index", tags=["model-management"])
    async def repository_index(request: Request) -> Any:
        try:
            require_control_key(request)
            body = await repository_body(request, allow_empty=True)
            unknown = set(body).difference({"ready"})
            if unknown or ("ready" in body and not isinstance(body["ready"], bool)):
                raise ServiceError(
                    "repository index request contains unsupported fields", code="unsupported_field"
                )
            if controller is None:
                raise ServiceError("model control is not configured", code="model_control_failure")
            return controller.snapshot(
                ready_only=bool(body.get("ready", False)), model_name=resolved_settings.model_id
            )
        except ServiceError as exc:
            return repository_status(exc)

    async def validate_model_operation(request: Request, model_name: str) -> dict[str, Any]:
        require_control_key(request)
        if model_name != resolved_settings.model_id:
            raise ServiceError("unsupported model name", code="unsupported_model", status_code=404)
        return await repository_body(request, allow_empty=True)

    @app.post("/v2/repository/models/{model_name:path}/load", tags=["model-management"])
    async def repository_load(request: Request, model_name: str) -> Response:
        try:
            body = await validate_model_operation(request, model_name)
            if body != {}:
                raise ServiceError(
                    "load request body must be empty or {}", code="unsupported_field"
                )
            if controller is None:
                raise ServiceError("model control is not configured", code="model_control_failure")
            await controller.load()
            return Response(status_code=200, content=b"")
        except ServiceError as exc:
            return repository_status(exc)

    @app.post("/v2/repository/models/{model_name:path}/unload", tags=["model-management"])
    async def repository_unload(request: Request, model_name: str) -> Response:
        try:
            body = await validate_model_operation(request, model_name)
            if body != {}:
                raise ServiceError(
                    "unload request body must be empty or {}", code="unsupported_field"
                )
            if controller is None:
                raise ServiceError("model control is not configured", code="model_control_failure")
            await controller.unload()
            return Response(status_code=200, content=b"")
        except ServiceError as exc:
            return repository_status(exc)

    @app.get("/readyz", tags=["health"])
    async def readyz(request: Request) -> JSONResponse:
        try:
            check_request_resources(resolved_settings)
        except ServiceError as exc:
            metrics.readiness.set(0)
            record_error(request, exc.code)
            return _error_response(request, exc)
        if controller is not None and not controller.is_ready:
            state = ReadyState(False, controller.readiness_detail())
        else:
            state = readiness()
        request_id = _request_id_for(request)
        if state.ready:
            metrics.readiness.set(1)
            return JSONResponse(
                status_code=200, content={"status": "ready", "detail": state.detail}
            )
        metrics.readiness.set(0)
        return JSONResponse(
            status_code=503,
            content=error_envelope("not_ready", state.detail or "engine not ready", request_id),
        )

    error_responses: Dict[int | str, Dict[str, Any]] = {
        200: {"model": CompletionResponse},
        400: {"model": ErrorEnvelope},
        401: {"model": ErrorEnvelope},
        413: {"model": ErrorEnvelope},
        499: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
        504: {"model": ErrorEnvelope},
        507: {"model": ErrorEnvelope},
    }

    @app.post(
        "/v1/responses",
        response_model=ResponsesResponse,
        responses={
            400: {"model": OpenAIErrorEnvelope},
            401: {"model": OpenAIErrorEnvelope},
            413: {"model": OpenAIErrorEnvelope},
            499: {"model": OpenAIErrorEnvelope},
            500: {"model": OpenAIErrorEnvelope},
            503: {"model": OpenAIErrorEnvelope},
            504: {"model": OpenAIErrorEnvelope},
            507: {"model": OpenAIErrorEnvelope},
        },
        tags=["responses"],
        dependencies=[Depends(require_responses_api_key)],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ResponsesRequest"}
                    }
                },
            }
        },
    )
    async def responses(request: Request) -> Any:
        """Serve the narrow, non-streaming Responses facade."""
        started = time.monotonic()
        request.state.metrics_started = started
        request_id = _request_id_for(request)
        settings_local = resolved_settings
        deadline_monotonic = started + settings_local.request_deadline_seconds
        image_generation = False

        def check_deadline() -> None:
            if time.monotonic() >= deadline_monotonic:
                raise ServiceError("request deadline exceeded", code="timeout")

        def fail(exc: ServiceError) -> JSONResponse:
            record_error(request, exc.code)
            metrics.observe_response_error(image_generation=image_generation)
            headers = {"x-request-id": request_id}
            headers.update(exc.headers or {})
            return JSONResponse(
                status_code=exc.status_code,
                content=responses_error_body(exc, request_id),
                headers=headers,
            )

        try:
            if request.query_params:
                raise ServiceError("query parameters are not supported", code="unsupported_field")
            if request.headers.get("content-type", "").strip().lower() != "application/json":
                raise ServiceError(
                    "Content-Type must be application/json", code="unsupported_media_type"
                )
            check_request_resources(settings_local)
            body_limit = responses_request_body_limit(settings_local)
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > body_limit:
                raise ServiceError(
                    "request body exceeds the maximum allowed size", code="payload_too_large"
                )
            chunks: List[bytes] = []
            streamed_total = 0
            try:
                async for chunk in request.stream():
                    check_deadline()
                    if not chunk:
                        continue
                    streamed_total += len(chunk)
                    if streamed_total > body_limit:
                        raise ServiceError(
                            "request body exceeds the maximum allowed size",
                            code="payload_too_large",
                        )
                    chunks.append(chunk)
            except ClientDisconnect as exc:
                raise ServiceError(
                    "request was cancelled before completion", code="cancelled"
                ) from exc
            check_deadline()
            body = b"".join(chunks)

            def reject_constant(_value: str) -> None:
                raise ValueError("non-finite JSON constant")

            try:
                decoded_body = json.loads(body.decode("utf-8"), parse_constant=reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ServiceError(
                    "request body is malformed JSON", code="responses_invalid_json"
                ) from exc
            parsed = parse_responses_request(decoded_body, settings_local)
            image_generation = parsed.image_generation
            check_deadline()
            run = await run_shared_inference(
                request,
                image_bytes=parsed.image_bytes,
                config_bytes=parsed.config_bytes,
                verbosity=2,
                request_id=request_id,
                deadline_monotonic=deadline_monotonic,
                render_visualizations=False,
            )
            class_mapping: Dict[str, int] = {}
            for index, label in enumerate(run.class_labels):
                class_mapping.setdefault(str(label), index)
            projection = build_public_projection(
                run.outcome,
                model_id=settings_local.model_id,
                config_digest=config_digest(run.core_config),
                class_mapping=class_mapping,
                candidate_views={
                    "clip": run.core_config.candidate_view_config("clip").as_dict(
                        stage="clip", applied=bool(run.core_config.clip_cfg)
                    ),
                    "blip3": run.core_config.candidate_view_config("blip3").as_dict(
                        stage="blip3", applied=bool(run.core_config.blip3_cfg)
                    ),
                },
                clip_routing=dict(run.core_config.clip_routing_cfg),
                clip_prompt_metadata=run.core_config.clip_prompt_metadata,
                config_warnings=run.validated.warnings,
            )
            serialization_started = time.monotonic()
            request.state.metrics_serialization_started = serialization_started
            check_deadline()
            response, _png, _output_text = build_responses_response(
                projection,
                model_id=settings_local.model_id,
                image_rgb=run.image_rgb,
                objects=run.outcome.result.objects,
                image_generation=image_generation,
                settings=settings_local,
            )
            check_deadline()
            response_bytes = len(
                json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            metrics.serialization_duration.observe(time.monotonic() - serialization_started)
            request.state.metrics_serialization_recorded = True
            metrics.update_gpu()
            metrics.observe_response_success(
                image_generation=image_generation,
                response_bytes=response_bytes,
                objects=len(run.outcome.result.objects),
            )
            metrics.request_duration.observe(time.monotonic() - started)
            request.state.metrics_recorded = True
            return JSONResponse(
                status_code=200,
                content=response,
                headers={"x-request-id": request_id},
            )
        except ServiceError as exc:
            return fail(exc)
        except Exception:
            return fail(ServiceError("unexpected internal failure", code="internal_error"))

    @app.get("/metrics", tags=["health"], dependencies=[Depends(require_api_key)])
    async def metrics_endpoint() -> Response:
        return Response(content=metrics.scrape(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/completions",
        responses=error_responses,
        tags=["completions"],
        dependencies=[Depends(require_api_key)],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image", "config"],
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Exactly one JPEG/PNG/WebP image",
                                },
                                "config": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Exactly one UTF-8 YAML configuration document",
                                },
                                "verbosity": {
                                    "type": "string",
                                    "enum": ["0", "1", "2", "3"],
                                    "description": "Strict integers; textual aliases rejected in v1",
                                },
                                "response_format": {
                                    "type": "string",
                                    "enum": ["json", "zip"],
                                },
                                "model": {
                                    "type": "string",
                                    "enum": [resolved_settings.model_id],
                                },
                                "stream": {
                                    "type": "string",
                                    "enum": ["false"],
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def completions(request: Request) -> Any:
        started = time.monotonic()
        request.state.metrics_started = started
        request_id = _request_id_for(request)
        settings_local = resolved_settings
        deadline_monotonic = started + settings_local.request_deadline_seconds

        def remaining_budget() -> float:
            return deadline_monotonic - time.monotonic()

        def check_deadline() -> None:
            if time.monotonic() >= deadline_monotonic:
                raise ServiceError("request deadline exceeded", code="timeout")

        content_length = request.headers.get("content-length")
        check_request_resources(settings_local)
        if content_length and content_length.isdigit():
            if int(content_length) > settings_local.max_request_bytes:
                raise ServiceError(
                    "request body exceeds the maximum allowed size",
                    code="payload_too_large",
                )

        content_type = request.headers.get("content-type", "")
        chunks: List[bytes] = []
        streamed_total = 0
        try:
            async for chunk in request.stream():
                check_deadline()
                if not chunk:
                    continue
                streamed_total += len(chunk)
                if streamed_total > settings_local.max_request_bytes:
                    raise ServiceError(
                        "request body exceeds the maximum allowed size",
                        code="payload_too_large",
                    )
                chunks.append(chunk)
        except ClientDisconnect as exc:
            raise ServiceError("request was cancelled before completion", code="cancelled") from exc
        check_deadline()
        parsed = parse_strict_multipart(content_type, chunks, settings_local)
        check_deadline()

        run = await run_shared_inference(
            request,
            image_bytes=parsed.image_bytes,
            config_bytes=parsed.config_bytes,
            verbosity=parsed.verbosity,
            request_id=request_id,
            deadline_monotonic=deadline_monotonic,
            render_visualizations=parsed.verbosity >= 3,
        )
        validated = run.validated
        core_config = run.core_config
        outcome = run.outcome
        sink = run.sink
        delivery_ledger = run.delivery_ledger
        class_labels = run.class_labels

        class_mapping: Dict[str, int] = {}
        for index, label in enumerate(class_labels):
            class_mapping.setdefault(str(label), index)

        context = ResponseContext(
            request_id=request_id,
            model_id=settings_local.model_id,
            verbosity=parsed.verbosity,
            response_format=parsed.response_format,
            config_digest=config_digest(core_config),
            class_mapping=class_mapping,
            config_warnings=validated.warnings,
            runtime_metadata=runtime_metadata or {},
            max_objects=settings_local.max_objects,
            max_response_artifacts=settings_local.max_response_artifacts,
            max_single_artifact_bytes=settings_local.max_single_artifact_bytes,
            max_total_raw_artifact_bytes=settings_local.max_total_raw_artifact_bytes,
            max_mask_rle_runs_per_object=settings_local.max_mask_rle_runs_per_object,
            max_mask_rle_runs_total=settings_local.max_mask_rle_runs_total,
            max_response_bytes=settings_local.max_response_bytes,
            deadline_monotonic=deadline_monotonic,
            candidate_views={
                "clip": core_config.candidate_view_config("clip").as_dict(
                    stage="clip", applied=bool(core_config.clip_cfg)
                ),
                "blip3": core_config.candidate_view_config("blip3").as_dict(
                    stage="blip3", applied=bool(core_config.blip3_cfg)
                ),
            },
            clip_routing=dict(core_config.clip_routing_cfg),
            artifact_ledger=delivery_ledger,
            service_safe_artifact_names=True,
        )

        serialization_started = time.monotonic()
        request.state.metrics_serialization_started = serialization_started
        if settings_local.test_serialization_delay_seconds:
            await asyncio.sleep(settings_local.test_serialization_delay_seconds)
        check_deadline()
        if parsed.response_format == "zip":
            payload = build_completion_zip(
                outcome,
                context,
                sink=sink,
                max_bytes=settings_local.max_response_bytes,
            )
            check_deadline()
            response = Response(
                content=payload,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="zap-it-result.zip"'},
            )
            check_deadline()
            metrics.serialization_duration.observe(time.monotonic() - serialization_started)
            request.state.metrics_serialization_recorded = True
            metrics.update_gpu()
            metrics.observe_success(
                verbosity=parsed.verbosity,
                response_format="zip",
                response_bytes=len(payload),
                objects=len(outcome.result.objects),
                artifacts=delivery_ledger.document()["delivered_count"]
                + (1 if parsed.verbosity >= 1 else 0),
            )
            metrics.request_duration.observe(time.monotonic() - started)
            request.state.metrics_recorded = True
            return response

        document = build_completion_json(outcome, context, sink=sink)
        bound_json_size(
            document,
            settings_local.max_response_bytes,
            deadline_monotonic=deadline_monotonic,
        )
        check_deadline()
        response = JSONResponse(status_code=200, content=document)
        check_deadline()
        metrics.serialization_duration.observe(time.monotonic() - serialization_started)
        request.state.metrics_serialization_recorded = True
        metrics.update_gpu()
        metrics.observe_success(
            verbosity=parsed.verbosity,
            response_format="json",
            response_bytes=len(response.body or b""),
            objects=len(outcome.result.objects),
            artifacts=len(document["service"].get("artifacts", [])),
        )
        metrics.request_duration.observe(time.monotonic() - started)
        request.state.metrics_recorded = True
        return response

    # The handler intentionally accepts a raw Request so the body can be
    # streamed under the derived cap.  Register the strict Pydantic request
    # model explicitly in generated OpenAPI without asking FastAPI to parse
    # and preallocate the unbounded body for us.
    generated_openapi = app.openapi

    def openapi_with_responses_request() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = generated_openapi()
        request_schema = ResponsesRequest.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        definitions = request_schema.pop("$defs", {})
        schema.setdefault("components", {}).setdefault("schemas", {}).update(definitions)
        schema["components"]["schemas"]["ResponsesRequest"] = request_schema
        app.openapi_schema = schema
        return schema

    app.openapi = openapi_with_responses_request  # type: ignore[method-assign]
    return app


def create_default_app() -> FastAPI:
    """Convenience factory wiring the deterministic fake engine."""
    return create_app(engine=FakeEngine())
