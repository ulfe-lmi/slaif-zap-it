"""Unit tests for the service boundary modules (no HTTP transport)."""

import asyncio

import numpy as np
import pytest
from PIL import Image

from src.core import ObjectResult, PipelineResult, Provenance, SingleImageOutcome
from src.service.envelope import ResponseContext, build_completion_json
from src.service.errors import ERROR_STATUS_CODES, ServiceError, error_envelope
from src.service.fake_engine import FakeEngine
from src.service.gate import InferenceGate
from src.service.image_input import decode_image_safely
from src.service.multipart import parse_strict_multipart
from src.service.settings import SERVICE_MODEL_ID, ServiceSettings
from src.service.yaml_input import parse_hostile_config


def png_bytes(width=8, height=6, color=(10, 200, 30)):
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width=8, height=6):
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (width, height), (0, 0, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


def build_multipart(parts, boundary=b"zapittest"):
    """parts: list of (name, filename, content_type, payload) tuples."""
    out = bytearray()
    for name, filename, content_type, payload in parts:
        out += b"--" + boundary + b"\r\n"
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        out += f"Content-Disposition: {disposition}\r\n".encode()
        if content_type:
            out += f"Content-Type: {content_type}\r\n".encode()
        out += b"\r\n"
        out += payload + b"\r\n"
    out += b"--" + boundary + b"--\r\n"
    return bytes(out)


def parse(body_parts, settings=None, boundary=b"zapittest", content_type=None):
    body = build_multipart(body_parts, boundary=boundary)
    ctype = content_type or f"multipart/form-data; boundary={boundary.decode()}"
    return parse_strict_multipart(ctype, [body], settings or ServiceSettings())


VALID_CONFIG = b"alpha: 0.5\npreprocessing:\n  roi: false\n"


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


def test_frozen_error_codes_present_with_statuses():
    required = {
        "invalid_multipart",
        "missing_part",
        "duplicate_part",
        "invalid_image",
        "invalid_config",
        "unsafe_config",
        "unsupported_field",
        "unsupported_verbosity",
        "unsupported_format",
        "unsupported_model",
        "stream_unsupported",
        "payload_too_large",
        "image_too_large",
        "service_busy",
        "not_ready",
        "timeout",
        "cancelled",
        "inference_failure",
        "insufficient_memory",
        "response_too_large",
    }
    assert required.issubset(ERROR_STATUS_CODES)
    assert ERROR_STATUS_CODES["service_busy"] == 503
    assert ERROR_STATUS_CODES["payload_too_large"] == 413


def test_error_envelope_shape():
    envelope = error_envelope("service_busy", "busy message", "req-123")
    assert envelope == {
        "error": {"code": "service_busy", "message": "busy message", "request_id": "req-123"}
    }


def test_service_error_busy_headers():
    err = ServiceError("busy", code="service_busy", headers={"Retry-After": "7"})
    assert err.status_code == 503
    assert err.headers["Retry-After"] == "7"


def test_identity_projection_failure_maps_to_stable_sanitized_service_error():
    mask = np.ones((1, 1), dtype=bool)
    objects = (
        ObjectResult(instance_id=1, source_index=0, mask=mask),
        ObjectResult(instance_id=2, source_index=1, mask=mask),
    )
    result = PipelineResult(
        image_height=1,
        image_width=1,
        roi_box=(0, 0, 0, 0),
        resize_info={},
        objects=objects,
        stage_statuses=(),
        candidate_counts={},
        rendered={},
        warnings=(),
        timings={},
        provenance=Provenance(config_digest="digest"),
    )
    context = ResponseContext(
        request_id="req-identity",
        model_id="zap-it-1",
        verbosity=1,
        response_format="json",
        config_digest="digest",
        class_mapping={},
    )
    with pytest.raises(ServiceError) as excinfo:
        build_completion_json(
            SingleImageOutcome(result, segmenter_state=None, clip_state=None, blip3_state=None),
            context,
        )
    assert excinfo.value.code == "inference_failure"
    assert str(excinfo.value) == (
        "identity representation cannot preserve a distinct source pixel for every object"
    )


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def test_settings_defaults_match_frozen_limits():
    s = ServiceSettings()
    assert s.max_image_upload_bytes == 20 * 1024 * 1024
    assert s.max_config_upload_bytes == 256 * 1024
    assert s.max_decoded_pixels == 64_000_000
    assert s.max_response_bytes == 256 * 1024 * 1024
    assert s.request_deadline_seconds == 120.0
    assert s.queue_depth == 0
    assert s.model_id == "zap-it-1"


def test_settings_from_environment_overrides():
    env = {
        "SLAIF_ZAP_IT_API_KEY": "secret-key",
        "SLAIF_ZAP_IT_MAX_IMAGE_UPLOAD_BYTES": "1024",
        "SLAIF_ZAP_IT_QUEUE_DEPTH": "2",
        "SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS": "3.5",
        "SLAIF_ZAP_IT_TEST_SERIALIZATION_DELAY_SECONDS": "0.25",
    }
    s = ServiceSettings.from_environment(env)
    assert s.api_key == "secret-key"
    assert s.max_image_upload_bytes == 1024
    assert s.queue_depth == 2
    assert s.request_deadline_seconds == 3.5
    assert s.test_serialization_delay_seconds == 0.25


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError):
        ServiceSettings(max_decoded_pixels=-1)
    with pytest.raises(ValueError):
        ServiceSettings(queue_depth=-1)
    with pytest.raises(ValueError):
        ServiceSettings.from_environment({"SLAIF_ZAP_IT_MAX_DECODED_PIXELS": "nope"})


# ---------------------------------------------------------------------------
# strict multipart parsing
# ---------------------------------------------------------------------------


def _base_parts(**overrides):
    parts = [
        ("image", "img.png", "image/png", png_bytes()),
        ("config", "c.yaml", "application/yaml", VALID_CONFIG),
    ]
    for name, value in overrides.items():
        parts.append((name, None, None, value.encode() if isinstance(value, str) else value))
    return parts


def test_parse_happy_path_defaults():
    parsed = parse(_base_parts())
    assert parsed.verbosity == 0
    assert parsed.response_format == "json"
    assert parsed.model is None
    assert parsed.stream is False
    assert parsed.image_bytes.startswith(b"\x89PNG")
    assert parsed.config_bytes == VALID_CONFIG


def test_parse_explicit_scalar_fields():
    parsed = parse(
        _base_parts(verbosity="3", response_format="zip", model=SERVICE_MODEL_ID, stream="false")
    )
    assert parsed.verbosity == 3
    assert parsed.response_format == "zip"
    assert parsed.model == SERVICE_MODEL_ID


@pytest.mark.parametrize("bad", ["low", "4", "-1", "1.0", "+2"])
def test_parse_rejects_noncanonical_verbosity(bad):
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(verbosity=bad))
    assert excinfo.value.code == "unsupported_verbosity"


def test_parse_rejects_bad_response_format():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(response_format="xml"))
    assert excinfo.value.code == "unsupported_format"


def test_parse_rejects_wrong_model():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(model="gpt-4"))
    assert excinfo.value.code == "unsupported_model"


def test_parse_rejects_stream_true():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(stream="true"))
    assert excinfo.value.code == "stream_unsupported"


def test_missing_image_and_config():
    with pytest.raises(ServiceError) as excinfo:
        parse([("config", "c.yaml", None, VALID_CONFIG)])
    assert excinfo.value.code == "missing_part"
    with pytest.raises(ServiceError) as excinfo:
        parse([("image", "i.png", None, png_bytes())])
    assert excinfo.value.code == "missing_part"


def test_duplicate_part_rejected():
    parts = _base_parts() + [("image", "again.png", "image/png", png_bytes())]
    with pytest.raises(ServiceError) as excinfo:
        parse(parts)
    assert excinfo.value.code == "duplicate_part"


def test_unknown_field_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(output_dir="/tmp/evil"))
    assert excinfo.value.code == "unsupported_field"


def test_wrong_content_type_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(), content_type="application/json")
    assert excinfo.value.code == "invalid_multipart"


def test_missing_boundary_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(), content_type="multipart/form-data")
    assert excinfo.value.code == "invalid_multipart"


def test_garbage_body_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse_strict_multipart(
            "multipart/form-data; boundary=zapittest",
            [b"this is not multipart at all"],
            ServiceSettings(),
        )
    assert excinfo.value.code == "invalid_multipart"


def test_oversized_image_part_rejected_midstream():
    settings = ServiceSettings(max_image_upload_bytes=16)
    big_png = png_bytes(width=64, height=64)
    assert len(big_png) > 16
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(), settings=settings)
    assert excinfo.value.code == "payload_too_large"


def test_oversized_config_part_rejected_midstream():
    settings = ServiceSettings(max_config_upload_bytes=32)
    parts = [
        ("image", "img.png", "image/png", png_bytes()),
        ("config", "c.yaml", "application/yaml", VALID_CONFIG * 20),
    ]
    with pytest.raises(ServiceError) as excinfo:
        parse(parts, settings=settings)
    assert excinfo.value.code == "payload_too_large"


def test_scalar_part_cap_enforced():
    settings = ServiceSettings()
    huge_model = "x" * 5000
    with pytest.raises(ServiceError) as excinfo:
        parse(_base_parts(model=huge_model), settings=settings)
    assert excinfo.value.code == "payload_too_large"


# ---------------------------------------------------------------------------
# image decoding safety
# ---------------------------------------------------------------------------


def test_decode_png_to_rgb_array():
    array = decode_image_safely(png_bytes(), max_decoded_pixels=10_000)
    assert array.shape == (6, 8, 3)
    assert array.dtype == np.uint8
    assert array.flags.writeable


def test_decode_jpeg_allowed():
    array = decode_image_safely(jpeg_bytes(), max_decoded_pixels=10_000)
    assert array.shape == (6, 8, 3)


def test_decode_rejects_gif():
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="GIF")
    with pytest.raises(ServiceError) as excinfo:
        decode_image_safely(buffer.getvalue(), max_decoded_pixels=10_000)
    assert excinfo.value.code == "invalid_image"


def test_decode_rejects_text_payload():
    with pytest.raises(ServiceError) as excinfo:
        decode_image_safely(b"definitely not an image", max_decoded_pixels=10_000)
    assert excinfo.value.code == "invalid_image"


def test_decode_rejects_empty_payload():
    with pytest.raises(ServiceError) as excinfo:
        decode_image_safely(b"", max_decoded_pixels=10_000)
    assert excinfo.value.code == "invalid_image"


def test_decode_pixel_cap_before_allocation():
    with pytest.raises(ServiceError) as excinfo:
        decode_image_safely(png_bytes(64, 64), max_decoded_pixels=100)
    assert excinfo.value.code == "image_too_large"


def test_decode_truncated_png_rejected():
    payload = png_bytes(32, 32)[:40]
    with pytest.raises(ServiceError) as excinfo:
        decode_image_safely(payload, max_decoded_pixels=10**9)
    assert excinfo.value.code == "invalid_image"


# ---------------------------------------------------------------------------
# hostile YAML policy
# ---------------------------------------------------------------------------


def test_valid_config_parses_with_alpha_normalization():
    result = parse_hostile_config(VALID_CONFIG, verbosity=1)
    assert result.effective_mapping["alpha"] == 0.6
    assert result.class_labels == ()
    assert result.warnings == ()


def test_visualization_alpha_wins_like_legacy():
    raw = b"alpha: 0.5\nvisualization:\n  alpha: 0.75\n"
    result = parse_hostile_config(raw, verbosity=1)
    assert result.effective_mapping["alpha"] == 0.75


def test_batch_only_fields_ignored_with_warning():
    raw = VALID_CONFIG + b"images:\n  stream: out\nexport_yolo_det:\n  labels: goat\n"
    result = parse_hostile_config(raw, verbosity=1)
    assert "images" not in result.effective_mapping
    assert any("images" in warning for warning in result.warnings)


def test_unknown_top_level_field_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(VALID_CONFIG + b"geometry:\n  debug: true\n", verbosity=1)
    assert excinfo.value.code == "unsupported_field"


def test_blip2_section_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(VALID_CONFIG + b"blip2:\n  enabled: true\n", verbosity=1)
    assert excinfo.value.code == "unsupported_field"


@pytest.mark.parametrize(
    "snippet",
    [
        b"clip:\n  padding: '../escape'\n",
        b"clip:\n  padding: 'http://evil.example/x'\n",
        b"clip:\n  padding: '/absolute/path'\n",
        b"clip:\n  padding: 'C:\\\\win\\\\path'\n",
        b"clip:\n  padding: '~root/x'\n",
    ],
)
def test_hostile_string_values_rejected(snippet):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(VALID_CONFIG + snippet, verbosity=1)
    assert excinfo.value.code == "unsafe_config"


@pytest.mark.parametrize(
    "snippet",
    [
        b"preprocessing:\n  device: cuda\n",
        b"clip:\n  model_name: Salesforce/xgen-mm\n",
        b"mask_generator:\n  checkpoint: /models/sam.pt\n",
        b"visualization:\n  output_dir: /var/out\n",
        b"clip:\n  api_key: hunter2\n",
    ],
)
def test_forbidden_keys_rejected_anywhere(snippet):
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(VALID_CONFIG + snippet, verbosity=1)
    assert excinfo.value.code == "unsafe_config"


def test_yaml_alias_rejected():
    raw = ("alpha: 0.5\nanchor: &a\n  x: 1\nclip:\n  labels:\n    *a : {}\n").encode()
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=1)
    assert excinfo.value.code == "unsafe_config"


def test_deeply_nested_config_rejected():
    nested = "leaf: 1"
    for _ in range(24):
        nested = "a:\n" + "\n".join("  " + line for line in nested.splitlines())
    raw = ("alpha: 0.5\n" + nested).encode()
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(raw, verbosity=1)
    assert excinfo.value.code == "unsafe_config"


def test_oversized_scalar_rejected():
    huge = "x" * 20_000
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(f"alpha: 0.5\nclip:\n  padding: '{huge}'\n".encode(), verbosity=1)
    assert excinfo.value.code == "unsafe_config"


def test_invalid_utf8_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(b"alpha: \xff\xfe\n", verbosity=1)
    assert excinfo.value.code == "invalid_config"


def test_non_mapping_document_rejected():
    for raw in (b"- a\n- b\n", b"just a scalar", b""):
        with pytest.raises(ServiceError) as excinfo:
            parse_hostile_config(raw, verbosity=1)
        assert excinfo.value.code == "invalid_config"


def test_unparseable_yaml_rejected():
    with pytest.raises(ServiceError) as excinfo:
        parse_hostile_config(b"alpha: [unclosed\n  bad:", verbosity=1)
    assert excinfo.value.code == "invalid_config"


def test_clip_labels_become_class_labels_in_order():
    raw = (
        "alpha: 0.5\nclip:\n  labels:\n    zebra: 'a zebra'\n    negative: 'empty scene'\n"
    ).encode()
    result = parse_hostile_config(raw, verbosity=2)
    assert result.class_labels == ("zebra", "negative")


def test_debug_flags_stripped_below_verbosity_three():
    raw = (
        "alpha: 0.5\n"
        "preprocessing:\n  roi: false\n  debug: true\n"
        "mask_generator:\n  points_per_side: 8\n"
    ).encode()
    result = parse_hostile_config(raw, verbosity=1)
    assert result.effective_mapping["preprocessing"]["debug"] is False
    assert any("debug flag preprocessing.debug" in w for w in result.warnings)
    kept = parse_hostile_config(raw, verbosity=3)
    assert kept.effective_mapping["preprocessing"]["debug"] is True
    assert not [w for w in kept.warnings if "debug" in w]


# ---------------------------------------------------------------------------
# gate semantics
# ---------------------------------------------------------------------------


def test_gate_serializes_and_rejects_overflow():
    async def scenario():
        gate = InferenceGate(queue_depth=0, retry_after_seconds=3)
        busy_errors = []
        overlap = 0

        async def worker(delay: float):
            nonlocal overlap
            try:
                async with gate.slot():
                    overlap += 1
                    await asyncio.sleep(delay)
                    overlap -= 1
            except ServiceError as exc:
                busy_errors.append(exc)

        task_a = asyncio.create_task(worker(0.05))
        await asyncio.sleep(0.01)
        await worker(0.0)
        await task_a
        return busy_errors, overlap

    busy_errors, overlap = asyncio.run(scenario())
    assert len(busy_errors) == 1
    assert busy_errors[0].code == "service_busy"
    assert busy_errors[0].headers["Retry-After"] == "3"
    assert overlap <= 1


def test_gate_queue_admits_then_refuses():
    async def scenario():
        gate = InferenceGate(queue_depth=1)
        results = []
        busy = []

        async def worker(index: int, delay: float):
            try:
                async with gate.slot():
                    await asyncio.sleep(delay)
                    results.append(index)
            except ServiceError as exc:
                busy.append(exc)

        first = asyncio.create_task(worker(1, 0.05))
        await asyncio.sleep(0.005)
        queued = asyncio.create_task(worker(2, 0.0))
        await asyncio.sleep(0.005)
        third = asyncio.create_task(worker(3, 0.0))
        await asyncio.sleep(0.005)
        fourth = asyncio.create_task(worker(4, 0.0))
        await asyncio.gather(first, queued, third, fourth)
        assert sorted(results) == [1, 2]
        assert len(busy) == 2
        assert all(err.code == "service_busy" for err in busy)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# fake engine determinism and isolation
# ---------------------------------------------------------------------------


def test_fake_engine_deterministic_objects_ordered_by_area():
    from src.core.config import CoreConfig

    engine = FakeEngine(object_count=2)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    config = CoreConfig(alpha=0.5, roi_val=None, resize_val=None, prep_debug=False)
    outcome_one = engine(image, config, frame_id="r1", class_labels=("goat", "sign"))
    outcome_two = engine(image, config, frame_id="r2", class_labels=("goat", "sign"))
    areas = [obj.area_px for obj in outcome_one.result.objects]
    assert areas == sorted(areas, reverse=True)
    ids = [obj.instance_id for obj in outcome_two.result.objects]
    assert ids == [1, 2]
    assert outcome_one.result.objects[0].metadata["clip_label"] == "goat"
    assert outcome_one.result is not outcome_two.result
