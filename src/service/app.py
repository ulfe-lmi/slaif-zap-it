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
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from src.core.config import CoreConfig, config_digest
from src.core.sinks import MemoryArtifactSink

from .auth import verify_bearer_key
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
from .multipart import parse_strict_multipart
from .schemas import (
    CompletionResponse,
    ErrorEnvelope,
    HealthStatus,
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
        content=error_envelope(exc.code, exc.message, _request_id_for(request)),
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
    sink: MemoryArtifactSink,
    class_labels: List[str],
    timeout_seconds: float,
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
        )

    future = loop.run_in_executor(executor, _call)
    return await asyncio.wait_for(future, timeout=max(timeout_seconds, 0.0))


def create_app(
    *,
    engine: EngineCallable,
    settings: Optional[ServiceSettings] = None,
    readiness_provider: Optional[ReadinessProvider] = None,
) -> FastAPI:
    """Build the service app around an explicitly injected engine."""
    resolved_settings = settings or ServiceSettings()
    readiness = readiness_provider or default_readiness_provider
    gate = InferenceGate(
        queue_depth=resolved_settings.queue_depth,
        retry_after_seconds=resolved_settings.retry_after_seconds,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zap-it-inference")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def require_api_key(request: Request) -> None:
        expected = resolved_settings.api_key
        if expected is None:
            return
        verify_bearer_key(request.headers.get("authorization"), expected)

    app = FastAPI(
        title="SLAIF ZAP-IT API",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = resolved_settings
    app.state.gate = gate
    app.state.started_monotonic = time.monotonic()

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
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
        return JSONResponse(
            status_code=500,
            content=error_envelope(
                "internal_error",
                "unexpected internal failure",
                _request_id_for(request),
            ),
        )

    @app.get("/healthz", response_model=HealthStatus, tags=["health"])
    async def healthz() -> Dict[str, str]:
        uptime = time.monotonic() - app.state.started_monotonic
        return {"status": "ok", "uptime_s": f"{uptime:.3f}"}

    @app.get("/readyz", tags=["health"])
    async def readyz(request: Request) -> JSONResponse:
        state = readiness()
        request_id = _request_id_for(request)
        if state.ready:
            return JSONResponse(
                status_code=200, content={"status": "ready", "detail": state.detail}
            )
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
        request_id = _request_id_for(request)
        settings_local = resolved_settings

        def remaining_budget() -> float:
            return settings_local.request_deadline_seconds - (time.monotonic() - started)

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > settings_local.max_request_bytes:
                raise ServiceError(
                    "request body exceeds the maximum allowed size",
                    code="payload_too_large",
                )

        content_type = request.headers.get("content-type", "")
        chunks: List[bytes] = []
        streamed_total = 0
        async for chunk in request.stream():
            if not chunk:
                continue
            streamed_total += len(chunk)
            if streamed_total > settings_local.max_request_bytes:
                raise ServiceError(
                    "request body exceeds the maximum allowed size",
                    code="payload_too_large",
                )
            chunks.append(chunk)
        parsed = parse_strict_multipart(content_type, chunks, settings_local)

        image_rgb = decode_image_safely(
            parsed.image_bytes, max_decoded_pixels=settings_local.max_decoded_pixels
        )
        validated = parse_hostile_config(parsed.config_bytes, verbosity=parsed.verbosity)
        core_config = CoreConfig.from_mapping(validated.effective_mapping)

        ready = readiness()
        if not ready.ready:
            raise ServiceError(ready.detail or "engine not ready", code="not_ready")

        class_labels = list(validated.class_labels)
        sink = MemoryArtifactSink()
        loop = asyncio.get_running_loop()
        try:
            async with gate.slot():
                budget = remaining_budget()
                if budget <= 0:
                    raise ServiceError("request deadline exceeded", code="timeout")
                try:
                    outcome = await _run_engine_bounded(
                        engine,
                        loop=loop,
                        executor=executor,
                        image_rgb=image_rgb,
                        core_config=core_config,
                        frame_id=request_id,
                        verbosity=parsed.verbosity,
                        sink=sink,
                        class_labels=class_labels,
                        timeout_seconds=budget,
                    )
                except asyncio.TimeoutError as exc:
                    raise ServiceError(
                        "inference exceeded the request deadline", code="timeout"
                    ) from exc
                except asyncio.CancelledError as exc:
                    raise ServiceError(
                        "request was cancelled before completion", code="cancelled"
                    ) from exc
                except ServiceError:
                    raise
                except Exception as exc:
                    raise ServiceError(
                        "inference failed inside the engine",
                        code="inference_failure",
                    ) from exc
        except asyncio.CancelledError:
            raise ServiceError("request was cancelled before completion", code="cancelled")

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
        )

        if parsed.response_format == "zip":
            payload = build_completion_zip(
                outcome,
                context,
                sink=sink,
                max_bytes=settings_local.max_response_bytes,
            )
            return Response(
                content=payload,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="zap-it-result.zip"'},
            )

        document = build_completion_json(outcome, context, sink=sink)
        bound_json_size(document, settings_local.max_response_bytes)
        return JSONResponse(status_code=200, content=document)

    return app


def create_default_app() -> FastAPI:
    """Convenience factory wiring the deterministic fake engine."""
    return create_app(engine=FakeEngine())
