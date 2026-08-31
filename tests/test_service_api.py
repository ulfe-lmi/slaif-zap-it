"""HTTP contract tests for /v1/completions (CPU-only, fake engine).

Sequential behavior runs through Starlette's TestClient; concurrency, busy,
timeout and cancellation scenarios drive the app through httpx's ASGI
transport inside a single event loop so scheduling matches the production
single-process/single-loop model.
"""

import asyncio
import base64
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.service import (
    ERROR_STATUS_CODES,
    ReadyState,
    ServiceSettings,
    create_app,
)
from src.service.fake_engine import FakeEngine
from src.runtime.strategy import RuntimePolicy

IO = __import__("io") if False else io


def png_bytes(width=32, height=24, color=(200, 10, 10)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


VALID_CONFIG = b"alpha: 0.5\npreprocessing:\n  roi: false\nclip:\n  labels:\n    goat: 'a goat'\n"

FILES = {
    "image": ("frame.png", png_bytes(), "image/png"),
    "config": ("config.yaml", VALID_CONFIG, "application/yaml"),
}


def _ready_provider():
    return ReadyState(True, "fake engine ready")


def make_client(
    engine=None,
    settings=None,
    readiness=None,
):
    app = create_app(
        engine=engine or FakeEngine(),
        settings=settings,
        readiness_provider=readiness or (lambda: ReadyState(True, "fake engine ready")),
    )
    return TestClient(app, raise_server_exceptions=False), app


# ---------------------------------------------------------------------------
# health / readiness
# ---------------------------------------------------------------------------


def test_healthz_reports_process_up():
    client, _ = make_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_default_is_honest_not_ready():
    app = create_app(engine=FakeEngine())
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "not_ready"
    assert body["error"]["request_id"]


def test_readyz_with_ready_provider():
    client, _ = make_client(readiness=lambda: ReadyState(True, "fake engine ready"))
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "detail": "fake engine ready"}


def test_operator_runtime_policy_accepts_sequential_blip3_profile():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target", model_registry_ready=True)
    app = create_app(
        engine=FakeEngine(),
        readiness_provider=_ready_provider,
        runtime_policy=policy,
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/completions",
        files={
            "image": ("frame.png", png_bytes(), "image/png"),
            "config": (
                "config.yaml",
                b"blip3:\n  thing:\n    question: 'is this a thing?'\n",
                "application/yaml",
            ),
        },
        data={"verbosity": "2"},
    )
    assert response.status_code == 200
    assert response.json()["service"]["objects"]


def test_not_ready_blocks_inference_but_parsing_errors_come_first():
    app = create_app(engine=FakeEngine())
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/v1/completions", files=FILES, data={"verbosity": "9"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_verbosity"
    ok_shape = client.post("/v1/completions", files=FILES)
    assert ok_shape.status_code == 503
    assert ok_shape.json()["error"]["code"] == "not_ready"


# ---------------------------------------------------------------------------
# verbosity levels L0-L3 monotonicity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def level_bodies():
    client, _ = make_client()
    bodies = {}
    for level in ("0", "1", "2", "3"):
        response = client.post("/v1/completions", files=FILES, data={"verbosity": level})
        assert response.status_code == 200, response.text
        bodies[int(level)] = response.json()
    return bodies


def test_l0_minimal_metadata_and_yolo_text(level_bodies):
    doc = level_bodies[0]
    assert doc["object"] == "text_completion"
    assert doc["usage"] is None
    assert doc["model"] == "zap-it-1"
    assert doc["schema_version"] == "zap-it.v1"
    assert doc["id"].startswith("cmpl-")
    assert doc["choices"][0]["finish_reason"] == "stop"
    text = doc["choices"][0]["text"]
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 2
    for line in lines:
        fields = line.split(" ")
        assert len(fields) == 5
        assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])
    service = doc["service"]
    assert service["verbosity"] == 0
    assert service["package_version"] == "0.1.0"
    assert service["class_mapping"] == {"goat": 0}
    assert len(service["config_digest"]) == 64
    assert service["image"] == {"width": 32, "height": 24}
    assert "artifacts" not in service
    assert "objects" not in service
    assert "stage_statuses" not in service
    assert "post_filter_diagnostics" not in service


def test_l1_adds_identity_mask_artifact(level_bodies):
    l0, l1 = level_bodies[0], level_bodies[1]
    assert "artifacts" not in l0["service"]
    artifacts = l1["service"]["artifacts"]
    assert [a["name"] for a in artifacts] == ["identity-mask.png"]
    artifact = artifacts[0]
    assert artifact["media_type"] == "image/png"
    assert artifact["encoding"] == "base64"
    raw = base64.b64decode(artifact["data"])
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
    assert artifact["size"] == len(raw)
    decoded = np.array(Image.open(io.BytesIO(raw)))
    assert decoded.dtype == np.uint16
    assert decoded.shape == (24, 32)
    assert set(np.unique(decoded)) == {0, 1, 2}
    assert "post_filter_diagnostics" not in l1["service"]


def test_l2_adds_object_records(level_bodies):
    l2 = level_bodies[2]
    objects = l2["service"]["objects"]
    assert len(objects) == 2
    by_id = {obj["instance_id"]: obj for obj in objects}
    assert sorted(by_id) == [1, 2]
    first = by_id[1]
    assert first["label"] == "goat"
    x1, y1, x2, y2 = first["bbox_xyxy"]
    assert 0 <= x1 <= x2 <= 32 and 0 <= y1 <= y2 <= 24
    assert len(first["bbox_normalized"]) == 4
    assert first["area_px"] > 0
    for field in ("predicted_iou", "stability_score", "clip_score"):
        assert field in first
    assert "blip3_answer" not in first
    assert "post_filter_diagnostics" not in l2["service"]


def test_l3_adds_stage_provenance_and_warnings(level_bodies):
    l3 = level_bodies[3]
    service = l3["service"]
    names = [status["name"] for status in service["stage_statuses"]]
    assert {"sam2", "clip"} <= set(names)
    assert isinstance(service["timings_ms"], dict)
    assert service["provenance"]["core_version"]
    assert any("fake" in warning for warning in service["warnings"])
    assert "candidate_counts" in service
    diagnostics = service["post_filter_diagnostics"]
    assert set(diagnostics) == {
        "limits",
        "evaluated",
        "removed_by_maxsize",
        "removed_empty_mask",
        "removed_by_max_w",
        "removed_by_max_h",
        "retained",
        "reason_precedence",
        "rejections",
        "rejections_truncated",
    }
    assert diagnostics["reason_precedence"] == ["maxsize", "empty_mask", "max_w", "max_h"]
    assert diagnostics["evaluated"] == diagnostics["retained"]
    assert diagnostics["rejections"] == []
    assert diagnostics["rejections_truncated"] == 0
    assert service["candidate_counts"]["sam2_candidates"] == diagnostics["evaluated"]
    assert service["candidate_counts"]["after_area_bbox"] == diagnostics["retained"]


# ---------------------------------------------------------------------------
# cardinality / field validation over HTTP
# ---------------------------------------------------------------------------


def _post_parts(client, parts, data=None):
    return client.post("/v1/completions", files=parts, data=data or {})


def test_missing_config_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={"image": ("f.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_part"


def test_missing_image_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={"config": ("c.yaml", VALID_CONFIG, "application/yaml")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_part"


def test_duplicate_image_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files=[
            ("image", ("a.png", png_bytes(), "image/png")),
            ("image", ("b.png", png_bytes(), "image/png")),
            ("config", ("c.yaml", VALID_CONFIG, "application/yaml")),
        ],
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "duplicate_part"


def test_unknown_field_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files=FILES,
        data={"output_dir": "/tmp"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"


def test_non_multipart_content_type_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("verbosity", "low", "unsupported_verbosity"),
        ("verbosity", "5", "unsupported_verbosity"),
        ("response_format", "xml", "unsupported_format"),
        ("model", "gpt-x", "unsupported_model"),
        ("stream", "true", "stream_unsupported"),
    ],
)
def test_scalar_field_validation(field, value, code):
    client, _ = make_client()
    response = _post_parts(client, FILES, data={field: value})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


# ---------------------------------------------------------------------------
# image and size limits over HTTP
# ---------------------------------------------------------------------------


def test_invalid_image_media_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={
            "image": ("f.gif", buffer.getvalue(), "image/gif"),
            "config": ("c.yaml", VALID_CONFIG, "application/yaml"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_corrupt_image_rejected():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={
            "image": ("f.png", b"\x89PNG broken data", "image/png"),
            "config": ("c.yaml", VALID_CONFIG, "application/yaml"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_decoded_pixel_cap_enforced():
    settings = ServiceSettings(max_decoded_pixels=100)
    client, _ = make_client(settings=settings)
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_upload_byte_cap_enforced():
    settings = ServiceSettings(max_image_upload_bytes=64)
    client, _ = make_client(settings=settings)
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_content_length_precheck_rejects_early():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files=FILES,
        headers={"Content-Length": str(ServiceSettings().max_request_bytes * 4)},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


# ---------------------------------------------------------------------------
# hostile YAML over HTTP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_text", "code"),
    [
        ("alpha: 0.5\ngeometry:\n  debug: true\n", "unsupported_field"),
        ("alpha: 0.5\nimages:\n  out: x\n", None),
        ("alpha: 0.5\nclip:\n  path: /etc\n", "unsafe_config"),
        ("alpha: 0.5\nclip:\n  padding: 'http://x'\n", "unsafe_config"),
        ("- a\n- b\n", "invalid_config"),
        ("a: [unclosed\n", "invalid_config"),
    ],
)
def test_yaml_policy_over_http(yaml_text, code):
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={
            "image": ("f.png", png_bytes(), "image/png"),
            "config": ("c.yaml", yaml_text.encode(), "application/yaml"),
        },
    )
    if code is None:
        assert response.status_code != 500
        return
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_batch_only_fields_reported_as_ignored_at_l3():
    client, _ = make_client()
    yaml_text = VALID_CONFIG + b"export_yolo_det:\n  labels: goat\n"
    response = client.post(
        "/v1/completions",
        files={
            "image": ("f.png", png_bytes(), "image/png"),
            "config": ("c.yaml", yaml_text, "application/yaml"),
        },
        data={"verbosity": "3"},
    )
    assert response.status_code == 200
    warnings = response.json()["service"]["warnings"]
    assert any("'export_yolo_det'" in w for w in warnings)


# ---------------------------------------------------------------------------
# JSON / ZIP parity
# ---------------------------------------------------------------------------


def test_zip_response_contains_manifest_detections_and_mask():
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files=FILES,
        data={"verbosity": "2", "response_format": "zip"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert "manifest.json" in names
    assert "detections.yolo.txt" in names
    assert "identity-mask.png" in names
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["service"]["package_version"] == "0.1.0"
    assert manifest["usage"] is None
    assert manifest["choices"][0]["text"] == archive.read("detections.yolo.txt").decode()
    listed = {entry["name"]: entry for entry in manifest["service"]["artifacts"]}
    for name, entry in listed.items():
        payload = archive.read(name)
        import hashlib

        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["size"]
    assert "data" not in listed["identity-mask.png"]


def test_json_and_zip_share_detection_semantics():
    client, _ = make_client()
    json_response = client.post("/v1/completions", files=FILES, data={"verbosity": "2"})
    zip_response = client.post(
        "/v1/completions",
        files=FILES,
        data={"verbosity": "2", "response_format": "zip"},
    )
    document = json_response.json()
    archive = zipfile.ZipFile(io.BytesIO(zip_response.content))
    manifest = json.loads(archive.read("manifest.json"))
    assert document["choices"][0]["text"] == manifest["choices"][0]["text"]
    assert document["service"]["objects"] == manifest["service"]["objects"]


def test_l3_post_filter_diagnostics_have_closed_numeric_contract_and_zip_parity():
    config = VALID_CONFIG + (
        b"postsam2processing:\n  maxsize: 100000\n  max_w: 0\n  max_h: 100000\n"
    )
    files = {
        "image": ("frame.png", png_bytes(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }
    client, _ = make_client()
    json_response = client.post("/v1/completions", files=files, data={"verbosity": "3"})
    repeat_response = client.post("/v1/completions", files=files, data={"verbosity": "3"})
    zip_response = client.post(
        "/v1/completions",
        files=files,
        data={"verbosity": "3", "response_format": "zip"},
    )
    assert (
        json_response.status_code == repeat_response.status_code == zip_response.status_code == 200
    )
    document = json_response.json()
    repeat = repeat_response.json()
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    diagnostics = document["service"]["post_filter_diagnostics"]
    assert diagnostics == repeat["service"]["post_filter_diagnostics"]
    assert diagnostics == manifest["service"]["post_filter_diagnostics"]
    assert diagnostics["evaluated"] >= 1
    assert diagnostics["retained"] == 0
    assert diagnostics["removed_by_max_w"] == diagnostics["evaluated"]
    assert (
        sum(
            diagnostics[key]
            for key in (
                "retained",
                "removed_by_maxsize",
                "removed_empty_mask",
                "removed_by_max_w",
                "removed_by_max_h",
            )
        )
        == diagnostics["evaluated"]
    )
    assert diagnostics["rejections_truncated"] == 0
    assert all(
        set(record) == {"source_index", "reason", "area_px", "bbox_width_px", "bbox_height_px"}
        and record["reason"] == "max_w"
        and record["bbox_width_px"] > 0
        for record in diagnostics["rejections"]
    )

    l2 = client.post("/v1/completions", files=files, data={"verbosity": "2"}).json()
    assert "post_filter_diagnostics" not in l2["service"]


def test_empty_detections_produce_empty_text_and_zero_mask():
    keep_nothing = VALID_CONFIG + b"visualization:\n  labels: ['nonexistent-label']\n"
    client, _ = make_client()
    response = client.post(
        "/v1/completions",
        files={
            "image": ("f.png", png_bytes(), "image/png"),
            "config": ("c.yaml", keep_nothing, "application/yaml"),
        },
        data={"verbosity": "1"},
    )
    assert response.status_code == 200
    document = response.json()
    assert document["choices"][0]["text"] == ""
    artifact = document["service"]["artifacts"][0]
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(artifact["data"]))))
    assert np.count_nonzero(decoded) == 0


# ---------------------------------------------------------------------------
# engine failures, timeout, cancellation
# ---------------------------------------------------------------------------


def test_engine_failure_maps_to_sanitized_inference_failure():
    class _Boom(FakeEngine):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    client, _ = make_client(engine=_Boom())
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "inference_failure"
    assert "secret" not in json.dumps(body)


def test_deadline_exceeded_maps_to_timeout():
    engine = FakeEngine(hang_seconds=0.6)
    settings = ServiceSettings(request_deadline_seconds=0.15)
    app = create_app(engine=engine, settings=settings, readiness_provider=_ready_provider)

    async def scenario():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as async_client:
            response = await async_client.post(
                "/v1/completions",
                files={
                    "image": ("f.png", png_bytes(), "image/png"),
                    "config": ("c.yaml", VALID_CONFIG, "application/yaml"),
                },
            )
            return response

    response = asyncio.run(scenario())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"


def test_cancellation_maps_to_cancelled_code():
    class _CancelEngine(FakeEngine):
        def __call__(self, *args, **kwargs):
            raise asyncio.CancelledError()

    client, _ = make_client(engine=_CancelEngine())
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == ERROR_STATUS_CODES["cancelled"]
    assert response.json()["error"]["code"] == "cancelled"


# ---------------------------------------------------------------------------
# concurrency / busy semantics
# ---------------------------------------------------------------------------


def _multipart_for_async():
    return {
        "files": {
            "image": ("f.png", png_bytes(), "image/png"),
            "config": ("c.yaml", VALID_CONFIG, "application/yaml"),
        }
    }


def _drive_async(app, count, stagger=0.05):
    import httpx

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        responses = []

        async with httpx.AsyncClient(transport=transport, base_url="http://t") as async_client:

            async def send(index):
                if index:
                    await asyncio.sleep(stagger)
                return await async_client.post("/v1/completions", **_multipart_for_async())

            responses = await asyncio.gather(*(send(i) for i in range(count)))
        return responses

    return asyncio.run(scenario())


def test_busy_when_slot_full_and_queue_depth_zero():
    engine = FakeEngine(delay_seconds=0.25)
    app = create_app(
        engine=engine, settings=ServiceSettings(queue_depth=0), readiness_provider=_ready_provider
    )
    responses = _drive_async(app, 2)
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 503]
    busy = next(r for r in responses if r.status_code == 503)
    body = busy.json()
    assert body["error"]["code"] == "service_busy"
    assert busy.headers.get("retry-after")
    assert engine.calls[0]["has_sink"] is True
    assert len(engine.calls) == 1
    assert engine.max_observed_active == 1


def test_queue_depth_one_admits_second_and_refuses_third():
    engine = FakeEngine(delay_seconds=0.15)
    app = create_app(
        engine=engine, settings=ServiceSettings(queue_depth=1), readiness_provider=_ready_provider
    )
    responses = _drive_async(app, 3)
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 200, 503]
    assert engine.max_observed_active == 1
    assert len(engine.calls) == 2
    frame_ids = {call["frame_id"] for call in engine.calls}
    assert len(frame_ids) == 2


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def test_auth_disabled_by_default_on_loopback():
    client, _ = make_client()
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == 200


def _auth_client():
    return make_client(settings=ServiceSettings(api_key="k-secret-1"))


def test_auth_missing_header_rejected():
    client, _ = _auth_client()
    response = client.post("/v1/completions", files=FILES)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_auth_wrong_key_rejected_and_never_echoed():
    client, _ = _auth_client()
    response = client.post(
        "/v1/completions",
        files=FILES,
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
    assert "k-secret-1" not in response.text


def test_auth_correct_key_accepted():
    client, _ = _auth_client()
    response = client.post(
        "/v1/completions",
        files=FILES,
        headers={"Authorization": "Bearer k-secret-1"},
    )
    assert response.status_code == 200


def test_auth_health_endpoints_stay_open():
    client, _ = _auth_client()
    assert client.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# isolation, cleanup and schema documentation
# ---------------------------------------------------------------------------


def test_request_ids_are_unique_per_call():
    engine = FakeEngine()
    client, _ = make_client(engine=engine)
    ids = []
    for _ in range(2):
        response = client.post("/v1/completions", files=FILES)
        assert response.status_code == 200
        ids.append(response.json()["service"]["request_id"])
    assert ids[0] != ids[1]
    assert {call["frame_id"] for call in engine.calls} == set(ids)


def test_no_files_created_by_service_requests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, _ = make_client()
    assert client.post("/v1/completions", files=FILES).status_code == 200
    bad = client.post(
        "/v1/completions",
        files={
            "image": ("f.png", b"junk", "image/png"),
            "config": ("c.yaml", VALID_CONFIG, "application/yaml"),
        },
    )
    assert bad.status_code == 400
    engine = FakeEngine(hang_seconds=0.4)
    app = create_app(
        engine=engine,
        settings=ServiceSettings(request_deadline_seconds=0.05),
        readiness_provider=_ready_provider,
    )
    with TestClient(app, raise_server_exceptions=False) as timeout_client:
        timeout_client.post("/v1/completions", files=FILES)
    assert list(tmp_path.iterdir()) == []


def test_openapi_documents_contract():
    client, app = make_client()
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths) >= {"/v1/completions", "/healthz", "/readyz"}
    completions = paths["/v1/completions"]["post"]
    multipart_schema = completions["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert set(multipart_schema["properties"]) >= {
        "image",
        "config",
        "verbosity",
        "response_format",
        "model",
        "stream",
    }
    refs = json.dumps(completions["responses"])
    assert "CompletionResponse" in refs
    assert "ErrorEnvelope" in refs
    component_models = schema["components"]["schemas"]
    assert {
        "CompletionResponse",
        "ErrorEnvelope",
        "ObjectRecord",
        "PostFilterLimits",
        "PostFilterRejection",
        "PostFilterDiagnostics",
    } <= set(component_models)
    assert component_models["PostFilterRejection"]["properties"]["reason"]["enum"] == [
        "maxsize",
        "empty_mask",
        "max_w",
        "max_h",
        "min_area",
        "max_area",
        "min_width",
        "max_width",
        "min_height",
        "max_height",
        "min_aspect_ratio",
        "max_aspect_ratio",
        "border_touching",
    ]
    assert component_models["PostFilterDiagnostics"]["properties"]["rejections"]["maxItems"] == 256


def test_openapi_schema_is_deterministic_across_instances():
    first = make_client()[0].get("/openapi.json").content
    second = make_client()[0].get("/openapi.json").content
    assert first == second


def test_error_responses_use_frozen_envelope_everywhere():
    client, _ = make_client()
    for request_kwargs, expected_status in [
        ({"files": {"image": ("f.png", png_bytes(), "image/png")}}, 400),
        ({"files": FILES, "data": {"verbosity": "7"}}, 400),
    ]:
        response = client.post("/v1/completions", **request_kwargs)
        assert response.status_code == expected_status
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "request_id"}
