"""CPU/fake contract tests for the narrow OpenAI Responses facade."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import httpx
import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.responses import Response as SDKResponse
from PIL import Image
from pydantic import ValidationError

from modules.visualizer import render_annotated_labelled
from src.core.config import CoreConfig, config_digest
from src.service import ReadyState, ServiceSettings, create_app
from src.service.envelope import encode_png
from src.service.fake_engine import FakeEngine
from src.service.responses import (
    PUBLIC_SCHEMA_VERSION,
    _bounded_warning,
    build_public_projection,
    parse_responses_request,
    responses_request_body_limit,
)
from src.service.schemas import (
    OpenAIErrorEnvelope,
    PublicProjection,
    ResponsesRequest,
    ResponsesResponse,
)
from src.service.errors import ServiceError
from src.service.yaml_input import parse_hostile_config


def _png_bytes(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


IMAGE_BYTES = _png_bytes()
CONFIG_BYTES = b"alpha: 0.5\nclip:\n  labels:\n    goat: 'a goat'\n"


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _body(*, tool: bool = False, config: bytes = CONFIG_BYTES) -> dict:
    body = {
        "model": "zap-it-1",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": _data_url("image/png", IMAGE_BYTES),
                    },
                    {
                        "type": "input_file",
                        "filename": "task.yaml",
                        "file_data": _data_url("application/yaml", config),
                    },
                ],
            }
        ],
    }
    if tool:
        body["tools"] = [{"type": "image_generation"}]
    return body


def _client(engine: FakeEngine | None = None, settings: ServiceSettings | None = None):
    app = create_app(
        engine=engine or FakeEngine(),
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "fake engine ready"),
    )
    return TestClient(app, raise_server_exceptions=False), app


def _post(client: TestClient, *, tool: bool = False, config: bytes = CONFIG_BYTES):
    return client.post("/v1/responses", json=_body(tool=tool, config=config))


def test_bounded_warning_preserves_printable_controls_and_limit():
    assert _bounded_warning("normal warning: A/B 123") == "normal warning: A/B 123"
    assert _bounded_warning("left\nmiddle\ttab\x00nul\x1funit") == ("left middle tab nul unit")
    assert _bounded_warning(123) == "123"
    long_warning = "a" * 300
    assert _bounded_warning(long_warning) == long_warning[:256]
    assert len(_bounded_warning(long_warning)) == 256


def test_public_projection_sanitizes_top_level_and_sam2_resource_warnings():
    validated = parse_hostile_config(CONFIG_BYTES, verbosity=2, settings=ServiceSettings())
    config = CoreConfig.from_mapping(validated.effective_mapping)
    outcome = FakeEngine()(
        np.array(Image.open(io.BytesIO(IMAGE_BYTES)).convert("RGB"), dtype=np.uint8),
        config,
        verbosity=2,
        class_labels=list(validated.class_labels),
    )
    outcome = replace(
        outcome,
        result=replace(
            outcome.result,
            warnings=("top\nlevel",),
            sam2_metadata={"resource_warnings": ["resource\twarning"]},
        ),
    )
    projection = build_public_projection(
        outcome,
        model_id="zap-it-1",
        config_digest=config_digest(config),
        class_mapping={"goat": 0},
        candidate_views={},
        clip_routing={},
    )
    assert projection["warnings"] == ["top level"]
    assert projection["sam2"]["resource_warnings"] == ["resource warning"]


@pytest.mark.parametrize(
    ("config", "expected_warning"),
    [
        (
            CONFIG_BYTES + b"postsam2processing:\n  debug: true\n",
            "debug flag postsam2processing.debug ignored at verbosity below 3",
        ),
        (
            CONFIG_BYTES + b"diagnostic_artifacts:\n  stages: [sam2]\n",
            "diagnostic_artifacts selection is valid but not applied below verbosity 3",
        ),
    ],
)
def test_responses_warning_projection_preserves_complete_config_warning(config, expected_warning):
    client, _ = _client()
    response = _post(client, config=config)
    assert response.status_code == 200, response.text
    projection = json.loads(response.json()["output"][0]["content"][0]["text"])
    assert expected_warning in projection["warnings"]
    assert " ".join(expected_warning) not in projection["warnings"]


def test_canonical_request_uses_shared_engine_path_and_fixed_service_policy():
    engine = FakeEngine()
    client, _ = _client(engine)
    response = _post(client)
    assert response.status_code == 200, response.text
    assert len(engine.calls) == 1
    assert engine.calls[0]["verbosity"] == 2
    assert engine.calls[0]["has_sink"] is True
    assert response.json()["model"] == "zap-it-1"

    class SpyEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.render_flags = []

        def __call__(self, *args, **kwargs):
            self.render_flags.append(kwargs.get("render_visualizations"))
            return super().__call__(*args, **kwargs)

    spy = SpyEngine()
    spy_client, _ = _client(spy)
    assert _post(spy_client, tool=True).status_code == 200
    assert spy.render_flags == [False]
    assert spy.calls[0]["verbosity"] == 2


def test_public_objects_match_private_l2_object_record_semantics():
    private_client, _ = _client()
    private = private_client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", IMAGE_BYTES, "image/png"),
            "config": ("task.yaml", CONFIG_BYTES, "application/yaml"),
        },
        data={"verbosity": "2"},
    )
    public = _post(private_client).json()
    projection = json.loads(public["output"][0]["content"][0]["text"])
    assert projection["schema_version"] == PUBLIC_SCHEMA_VERSION
    assert projection["objects"] == private.json()["service"]["objects"]
    assert all("mask_rle" not in obj for obj in projection["objects"])
    assert projection["objects"][0]["source_candidate_id"] == 1
    assert projection["objects"][0]["filtered_index"] == 0


def test_public_projection_serialization_is_deterministic_and_finite():
    client, _ = _client()
    first = json.loads(_post(client).json()["output"][0]["content"][0]["text"])
    second = json.loads(_post(client).json()["output"][0]["content"][0]["text"])
    first_text = json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    second_text = json.dumps(second, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert first_text == second_text
    assert first["config_digest"] == second["config_digest"]
    assert "NaN" not in first_text and "Infinity" not in first_text
    assert PublicProjection.model_validate(first).schema_version == PUBLIC_SCHEMA_VERSION


def test_success_envelopes_have_bounded_unique_protocol_ids_and_timestamps():
    client, _ = _client()
    first = _post(client).json()
    second = _post(client).json()
    assert first["id"].startswith("resp_")
    assert first["output"][0]["id"].startswith("msg_")
    assert first["id"] != second["id"]
    assert first["output"][0]["id"] != second["output"][0]["id"]
    ResponsesResponse.model_validate(first)
    assert first["status"] == "completed"
    assert isinstance(first["created_at"], float)
    assert first["created_at"] <= first["completed_at"]


def test_tool_controls_exactly_one_canonical_png_output_and_truthful_metadata():
    client, _ = _client()
    no_tool = _post(client).json()
    with_tool = _post(client, tool=True).json()
    assert no_tool["tools"] == []
    assert no_tool["tool_choice"] == "none"
    assert no_tool["parallel_tool_calls"] is False
    assert with_tool["tools"] == [{"type": "image_generation"}]
    assert with_tool["tool_choice"] == "auto"
    assert with_tool["parallel_tool_calls"] is False
    assert [item["type"] for item in no_tool["output"]] == ["message"]
    assert [item["type"] for item in with_tool["output"]] == [
        "message",
        "image_generation_call",
    ]
    image_call = with_tool["output"][1]
    assert image_call["id"].startswith("ig_")
    assert image_call["status"] == "completed"
    png = base64.b64decode(image_call["result"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == (32, 24)


def test_response_schema_rejects_mismatched_tool_and_output_metadata():
    client, _ = _client()
    no_tool = _post(client).json()
    with_tool = _post(client, tool=True).json()

    image_call_with_no_tool_metadata = json.loads(json.dumps(with_tool))
    image_call_with_no_tool_metadata["tool_choice"] = "none"
    image_call_with_no_tool_metadata["tools"] = []
    with pytest.raises(ValidationError):
        ResponsesResponse.model_validate(image_call_with_no_tool_metadata)

    declared_tool_without_image_call = json.loads(json.dumps(no_tool))
    declared_tool_without_image_call["tool_choice"] = "auto"
    declared_tool_without_image_call["tools"] = [{"type": "image_generation"}]
    with pytest.raises(ValidationError):
        ResponsesResponse.model_validate(declared_tool_without_image_call)

    ResponsesResponse.model_validate(no_tool)
    ResponsesResponse.model_validate(with_tool)


def test_tool_png_matches_existing_renderer_and_shared_encoder():
    client, _ = _client()
    response = _post(client, tool=True).json()
    actual = base64.b64decode(response["output"][1]["result"], validate=True)
    validated = parse_hostile_config(CONFIG_BYTES, verbosity=2, settings=ServiceSettings())
    config = CoreConfig.from_mapping(validated.effective_mapping)
    outcome = FakeEngine()(  # type: ignore[call-arg]
        np.array(Image.open(io.BytesIO(IMAGE_BYTES)).convert("RGB"), dtype=np.uint8),
        config,
        verbosity=2,
        class_labels=list(validated.class_labels),
        render_visualizations=False,
    )
    expected = encode_png(
        render_annotated_labelled(
            np.array(Image.open(io.BytesIO(IMAGE_BYTES)).convert("RGB"), dtype=np.uint8),
            outcome.result.objects,
            alpha=0.5,
            show_confidence=False,
        )
    )
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest() == hashlib.sha256(expected).hexdigest()


def test_tool_toggle_does_not_change_public_projection_or_engine_configuration():
    first_engine = FakeEngine()
    first_client, _ = _client(first_engine)
    second_engine = FakeEngine()
    second_client, _ = _client(second_engine)
    no_tool = _post(first_client).json()
    with_tool = _post(second_client, tool=True).json()
    assert (
        no_tool["output"][0]["content"][0]["text"] == with_tool["output"][0]["content"][0]["text"]
    )
    assert first_engine.calls[0]["config_digest"] == second_engine.calls[0]["config_digest"]
    assert first_engine.calls[0]["verbosity"] == second_engine.calls[0]["verbosity"] == 2


def test_responses_strips_private_visualization_execution_and_evidence():
    class SpyEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.flags = []

        def __call__(self, *args, **kwargs):
            self.flags.append(kwargs.get("render_visualizations"))
            return super().__call__(*args, **kwargs)

    debug_config = CONFIG_BYTES + (
        b"mask_generator:\n  debug: true\n"
        b"visualization:\n  sam2:\n    - id: private-debug\n"
        b"      renderer: annotated\n"
    )
    engine = SpyEngine()
    client, _ = _client(engine)
    response = _post(client, config=debug_config)
    assert response.status_code == 200, response.text
    assert engine.flags == [False]
    body = response.json()
    projection = json.loads(body["output"][0]["content"][0]["text"])
    serialized = json.dumps(body)
    assert set(projection) == {
        "candidate_counts",
        "candidate_views",
        "class_mapping",
        "clip_prompts",
        "clip_routing",
        "config_digest",
        "image",
        "model",
        "objects",
        "sam2",
        "schema_version",
        "warnings",
    }
    assert all("mask_rle" not in item for item in projection["objects"])
    assert "identity-mask.png" not in serialized
    assert "private-debug" not in serialized
    assert "artifact_delivery" not in serialized


def test_two_http_surfaces_share_one_gate_and_never_overlap_engine_calls():
    engine = FakeEngine(delay_seconds=0.15)
    app = create_app(
        engine=engine,
        settings=ServiceSettings(queue_depth=1),
        readiness_provider=lambda: ReadyState(True, "fake"),
    )

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            completion = client.post(
                "/v1/completions",
                files={
                    "image": ("frame.png", IMAGE_BYTES, "image/png"),
                    "config": ("task.yaml", CONFIG_BYTES, "application/yaml"),
                },
            )
            response = client.post("/v1/responses", json=_body())
            return await asyncio.gather(completion, response)

    results = asyncio.run(scenario())
    assert {result.status_code for result in results} <= {200, 503}
    assert engine.max_observed_active == 1


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda body: body.update({"unsupported": 1}), "unsupported_field"),
        (lambda body: body.update({"model": "other"}), "unsupported_model"),
        (lambda body: body.update({"stream": True}), "stream_unsupported"),
        (lambda body: body.update({"store": True}), "unsupported_field"),
        (lambda body: body.update({"background": True}), "unsupported_field"),
        (lambda body: body.update({"tools": [{"type": "function"}]}), "unsupported_tool"),
        (
            lambda body: body.update({"tools": [{"type": "image_generation", "x": 1}]}),
            "unsupported_field",
        ),
    ],
)
def test_unsupported_top_level_controls_and_tools_are_explicit(mutate, code):
    engine = FakeEngine()
    client, _ = _client(engine)
    body = _body()
    mutate(body)
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == code
    assert error["type"] == "invalid_request_error"
    assert "task.yaml" not in response.text
    assert engine.calls == []


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (
            [
                {
                    "type": "input_file",
                    "filename": "task.yaml",
                    "file_data": _data_url("application/yaml", CONFIG_BYTES),
                }
            ],
            "missing_image",
        ),
        (
            [
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": _data_url("image/png", IMAGE_BYTES),
                }
            ],
            "missing_config",
        ),
        ([], "missing_image"),
        (
            [
                {
                    "type": "input_text",
                    "text": "not supported",
                }
            ],
            "unsupported_field",
        ),
        (
            [
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": _data_url("image/png", IMAGE_BYTES),
                },
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": _data_url("image/png", IMAGE_BYTES),
                },
            ],
            "duplicate_image",
        ),
        (
            [
                {
                    "type": "input_file",
                    "filename": "task.yaml",
                    "file_data": _data_url("application/yaml", CONFIG_BYTES),
                },
                {
                    "type": "input_file",
                    "filename": "other.yaml",
                    "file_data": _data_url("application/yaml", CONFIG_BYTES),
                },
            ],
            "duplicate_config",
        ),
    ],
)
def test_image_and_yaml_cardinality_fails_before_inference(content, code):
    engine = FakeEngine()
    client, _ = _client(engine)
    body = _body()
    body["input"][0]["content"] = content
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == code
    assert error["type"] == "invalid_request_error"
    if code in {"missing_image", "missing_config", "duplicate_image", "duplicate_config"}:
        assert error["param"] == "input[0].content"
    else:
        assert error["param"] == "input[0].content[0].type"
    assert set(error) == {"message", "type", "param", "code"}
    assert "input[0].content" not in error["message"]
    assert engine.calls == []


def test_invalid_yaml_is_typed_before_engine():
    engine = FakeEngine()
    client, _ = _client(engine)
    response = _post(client, config=b"not: [valid")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_config"
    assert error["type"] == "invalid_request_error"
    assert set(error) == {"message", "type", "param", "code"}
    assert engine.calls == []


def test_each_http_surface_admits_resources_exactly_once(monkeypatch):
    admissions = []

    def record_admission(settings):
        admissions.append(settings)

    monkeypatch.setattr("src.service.app.check_request_resources", record_admission)
    engine = FakeEngine()
    client, _ = _client(engine)
    native = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", IMAGE_BYTES, "image/png"),
            "config": ("task.yaml", CONFIG_BYTES, "application/yaml"),
        },
    )
    assert native.status_code == 200, native.text
    assert len(admissions) == 1
    responses = _post(client)
    assert responses.status_code == 200, responses.text
    assert len(admissions) == 2
    assert len(engine.calls) == 2


def test_first_resource_admission_failure_precedes_body_parsing_on_both_surfaces(monkeypatch):
    calls = []

    def fail_admission(settings):
        calls.append(settings)
        raise ServiceError("resource floor", code="insufficient_memory")

    monkeypatch.setattr("src.service.app.check_request_resources", fail_admission)
    engine = FakeEngine()
    client, _ = _client(engine)
    native = client.post(
        "/v1/completions",
        content=b"not-multipart",
        headers={"content-type": "application/octet-stream"},
    )
    assert native.status_code == 507
    assert native.json()["error"]["code"] == "insufficient_memory"
    response = client.post(
        "/v1/responses",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 507
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert error["type"] == "server_error"
    assert set(error) == {"message", "type", "param", "code"}
    assert len(calls) == 2
    assert engine.calls == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("image_url", "https://example.invalid/a.png", "responses_unsupported_source"),
        ("image_url", "data:image/png;base64,%%%", "responses_invalid_data_url"),
        ("image_url", _data_url("image/jpeg", IMAGE_BYTES), "responses_mime_mismatch"),
        ("filename", "../task.yaml", "responses_unsafe_filename"),
        ("filename", "task.txt", "responses_unsafe_filename"),
        (
            "file_data",
            _data_url("application/octet-stream", CONFIG_BYTES),
            "unsupported_media_type",
        ),
        ("file_data", "data:application/yaml;base64,", "responses_invalid_data_url"),
    ],
)
def test_data_sources_filename_and_mime_are_strict(field, value, code):
    engine = FakeEngine()
    client, _ = _client(engine)
    body = _body()
    target = body["input"][0]["content"][0 if field == "image_url" else 1]
    target[field] = value
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"]
    assert engine.calls == []


def test_limits_reject_before_engine_and_response_size_is_typed():
    engine = FakeEngine()
    client, _ = _client(engine, ServiceSettings(max_image_upload_bytes=8))
    response = _post(client)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert engine.calls == []

    large_engine = FakeEngine()
    large_client, _ = _client(large_engine, ServiceSettings(max_response_bytes=128))
    response = _post(large_client)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "response_too_large"


def test_malformed_json_not_ready_and_inference_failure_are_openai_shaped():
    client, _ = _client()
    malformed = client.post(
        "/v1/responses", content=b"{", headers={"content-type": "application/json"}
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "responses_invalid_json"
    OpenAIErrorEnvelope.model_validate(malformed.json())

    not_ready_app = create_app(
        engine=FakeEngine(), readiness_provider=lambda: ReadyState(False, "private detail")
    )
    with TestClient(not_ready_app, raise_server_exceptions=False) as not_ready_client:
        not_ready = _post(not_ready_client)
    assert not_ready.status_code == 503
    assert not_ready.json()["error"]["code"] == "not_ready"
    assert "private detail" not in not_ready.text
    failing, _ = _client(FakeEngine(fail=True))
    failure = _post(failing)
    assert failure.status_code == 500
    assert failure.json()["error"]["code"] == "inference_failure"


def test_authentication_finishes_before_body_processing_and_completion_auth_is_unchanged():
    engine = FakeEngine()
    client, _ = _client(engine, ServiceSettings(api_key="k" * 32))
    response = client.post(
        "/v1/responses",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert set(response.json()["error"]) == {"message", "type", "param", "code"}
    assert response.headers["x-request-id"]
    assert engine.calls == []

    completion = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", IMAGE_BYTES, "image/png"),
            "config": ("task.yaml", CONFIG_BYTES, "application/yaml"),
        },
    )
    assert completion.status_code == 401
    assert set(completion.json()["error"]) == {"code", "message", "request_id"}


def test_no_response_state_or_files_are_retained(tmp_path: Path):
    engine = FakeEngine()
    client, _ = _client(engine, ServiceSettings(tmp_root=str(tmp_path)))
    assert _post(client).status_code == 200
    assert list(tmp_path.rglob("*")) == []


def test_openapi_and_capabilities_describe_both_distinct_surfaces():
    client, _ = _client(settings=ServiceSettings(api_key="k" * 32))
    openapi = client.get("/openapi.json").json()
    assert "/v1/responses" in openapi["paths"]
    assert "/v1/responses/{response_id}" not in openapi["paths"]
    assert "ResponsesRequest" in openapi["components"]["schemas"]
    assert "ResponsesResponse" in openapi["components"]["schemas"]
    assert "OpenAIErrorEnvelope" in openapi["components"]["schemas"]
    response_schema = openapi["components"]["schemas"]["ResponsesResponse"]
    assert response_schema["properties"]["tool_choice"]["enum"] == ["none", "auto"]
    assert response_schema["properties"]["tools"]["maxItems"] == 1
    assert response_schema["properties"]["tools"]["items"] == {
        "$ref": "#/components/schemas/ResponsesTool"
    }
    response_schema = openapi["paths"]["/v1/responses"]["post"]
    assert response_schema["requestBody"]["content"]["application/json"]
    capabilities = client.get("/v1/capabilities", headers={"authorization": "Bearer " + "k" * 32})
    assert capabilities.status_code == 200
    surfaces = capabilities.json()["response_evidence"]["api_surfaces"]
    assert surfaces["completions"]["path"] == "/v1/completions"
    assert surfaces["completions"]["classification"].startswith("private")
    assert surfaces["responses"]["path"] == "/v1/responses"
    assert surfaces["responses"]["projection_version"] == PUBLIC_SCHEMA_VERSION
    assert surfaces["responses"][
        "encoded_request_body_limit_bytes"
    ] == responses_request_body_limit(ServiceSettings())
    assert surfaces["responses"]["token_usage"] == "omitted"


def test_official_sdk_serializes_request_and_parses_typed_response_and_png():
    app_client, _ = _client()
    request_body = _body(tool=True)

    def handler(request):
        result = app_client.request(
            request.method,
            str(request.url.path),
            headers=dict(request.headers),
            content=request.content,
        )
        return httpx2.Response(
            result.status_code,
            headers=dict(result.headers),
            content=result.content,
            request=request,
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as http_client:
        client_options = {"api_" + "key": "fixture"}
        sdk = OpenAI(
            **client_options,
            base_url="http://test/v1",
            http_client=http_client,
        )
        response = sdk.responses.create(**request_body)
    assert isinstance(response, SDKResponse)
    assert response.object == "response"
    assert response.status == "completed"
    assert response.tool_choice == "auto"
    assert len(response.tools) == 1
    assert response.tools[0].type == "image_generation"
    projection = json.loads(response.output_text)
    assert projection["schema_version"] == PUBLIC_SCHEMA_VERSION
    calls = [item for item in response.output if item.type == "image_generation_call"]
    assert len(calls) == 1
    png = base64.b64decode(calls[0].result, validate=True)
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.size == (32, 24)


def test_parser_and_schema_agree_on_fixed_subset_and_derived_body_limit():
    settings = ServiceSettings()
    parsed = parse_responses_request(_body(tool=True), settings)
    assert parsed.model == "zap-it-1"
    assert parsed.image_generation is True
    assert len(parsed.image_bytes) == len(IMAGE_BYTES)
    assert parsed.config_bytes == CONFIG_BYTES
    assert responses_request_body_limit(settings) > len(IMAGE_BYTES) + len(CONFIG_BYTES)
    schema = ResponsesRequest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"model", "input", "tools", "store", "stream", "background"}
