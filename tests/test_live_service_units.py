"""Objective 004 unit tests: live-service wiring without CUDA or models."""

from __future__ import annotations

import importlib.util
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.runtime import live_service
from src.runtime.live_service import (
    LiveServiceConfig,
    LiveServiceError,
    ResidentRegistry,
    compose_readiness,
    live_engine_callable,
    masked_gpu_uuid,
    preflight,
)
from src.runtime.ports import PortCheck
from src.runtime.strategy import RuntimeReadiness
from src.service.errors import ServiceError


def _free_port_check(host: str, port: int) -> PortCheck:
    return PortCheck(host=host, port=port, ss_listener=False, bind_succeeded=True)


# --------------------------------------------------------------------------- #
# Launch configuration
# --------------------------------------------------------------------------- #


def test_config_requires_explicit_port():
    with pytest.raises(LiveServiceError, match="SLAIF_ZAP_IT_PORT"):
        LiveServiceConfig.from_environment({})


def test_config_rejects_non_loopback_host():
    with pytest.raises(LiveServiceError, match="127.0.0.1"):
        LiveServiceConfig.from_environment(
            {
                "SLAIF_ZAP_IT_PORT": "17891",
                "SLAIF_ZAP_IT_HOST": "0.0.0.0",
                "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": "GPU-x",
            }
        )


def test_config_strict_mode_requires_uuid():
    with pytest.raises(LiveServiceError, match="EXPECTED_GPU_UUID"):
        LiveServiceConfig.from_environment({"SLAIF_ZAP_IT_PORT": "17891"})


def test_config_happy_path_and_defaults():
    config = LiveServiceConfig.from_environment(
        {
            "SLAIF_ZAP_IT_PORT": "23654",
            "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": "GPU-abc",
            "SLAIF_ZAP_IT_TMP_ROOT": "/dev/shm/slaif-zap-it",
        }
    )
    assert config.host == "127.0.0.1"
    assert config.port == 23654
    assert config.physical_gpu_index == 1
    assert config.strict_gpu is True
    assert config.expected_gpu_uuid == "GPU-abc"


def test_config_rejects_bad_integers():
    with pytest.raises(LiveServiceError):
        LiveServiceConfig.from_environment(
            {"SLAIF_ZAP_IT_PORT": "not-a-port", "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": "GPU-x"}
        )
    with pytest.raises(LiveServiceError):
        LiveServiceConfig.from_environment(
            {
                "SLAIF_ZAP_IT_PORT": "17891",
                "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": "GPU-x",
                "SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX": "one",
            }
        )


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def test_preflight_fails_without_gpu_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    config = LiveServiceConfig.from_environment(
        {"SLAIF_ZAP_IT_PORT": "17891", "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": "GPU-x"}
    )
    with pytest.raises(LiveServiceError, match="launch environment"):
        preflight(config)


def test_preflight_reports_verified_state(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(live_service, "verify_port_unused", _free_port_check)
    config = LiveServiceConfig(
        host="127.0.0.1",
        port=39001,
        tmp_root=str(tmp_path / "shm"),
        expected_gpu_uuid="GPU-x",
    )
    report = preflight(config)
    assert report.launch_environment_ok is True
    assert report.port_check.unused is True
    assert report.shm_free_mib > 0
    root = Path(report.shm_root)
    assert (root.stat().st_mode & 0o777) == 0o700


def test_preflight_rejects_busy_port(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    def busy(host: str, port: int) -> PortCheck:
        return PortCheck(host=host, port=port, ss_listener=True, bind_succeeded=False)

    monkeypatch.setattr(live_service, "verify_port_unused", busy)
    config = LiveServiceConfig(
        host="127.0.0.1",
        port=39002,
        tmp_root=str(tmp_path / "shm"),
        expected_gpu_uuid="GPU-x",
    )
    with pytest.raises(LiveServiceError, match="verified-unused loopback port"):
        preflight(config)


def test_main_exit_code_2_when_port_missing(monkeypatch, capsys):
    monkeypatch.delenv("SLAIF_ZAP_IT_PORT", raising=False)
    assert live_service.main() == 2
    assert "SLAIF_ZAP_IT_PORT" in capsys.readouterr().err


def test_main_exit_code_2_when_mask_missing(monkeypatch, capsys):
    monkeypatch.setenv("SLAIF_ZAP_IT_PORT", "17891")
    monkeypatch.setenv("SLAIF_ZAP_IT_EXPECTED_GPU_UUID", "GPU-x")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert live_service.main() == 2
    assert "launch environment" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# UUID probe
# --------------------------------------------------------------------------- #


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_masked_uuid_single_line(monkeypatch):
    monkeypatch.setattr(
        live_service.subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(0, "GPU-c457dbaf-991c\n"),
    )
    assert masked_gpu_uuid() == "GPU-c457dbaf-991c"


def test_masked_uuid_ambiguous_or_failed(monkeypatch):
    monkeypatch.setattr(
        live_service.subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(0, "GPU-a\nGPU-b\n"),
    )
    assert masked_gpu_uuid() is None
    monkeypatch.setattr(live_service.subprocess, "run", lambda *a, **k: _FakeCompleted(1, ""))
    assert masked_gpu_uuid() is None


# --------------------------------------------------------------------------- #
# Resident registry
# --------------------------------------------------------------------------- #


def test_registry_transitions_loading_ready_and_reuses_states():
    segmenter = SimpleNamespace(name="sam2")
    clip = SimpleNamespace(name="clip")
    calls = []

    def loader() -> dict:
        calls.append(1)
        return {"segmenter": segmenter, "clip": clip}

    registry = ResidentRegistry(loader=loader)
    assert registry.verdict().ready is False
    assert "still loading" in registry.verdict().detail
    registry.load()
    assert registry.ready is True
    assert not registry.failed
    assert registry.load_seconds is not None
    first = registry.states()
    second = registry.states()
    assert first["segmenter"] is segmenter and first["clip"] is clip
    assert first is second
    assert len(calls) == 1
    verdict = registry.verdict()
    assert verdict.ready is True and "sam2_clip_resident" in verdict.detail


def test_registry_records_sanitized_failure_only():
    def loader() -> dict:
        raise RuntimeError("/secret/cache/path exploded with details")

    registry = ResidentRegistry(loader=loader)
    registry.load()
    assert registry.failed is True
    assert registry.ready is False
    assert registry.error_type == "RuntimeError"
    detail = registry.verdict().detail
    assert "/secret/cache/path" not in detail
    assert "exploded" not in detail


def test_registry_rejects_malformed_states():
    registry = ResidentRegistry(loader=lambda: {"unexpected": object()})
    registry.load()
    assert registry.failed is True
    assert registry.error_type == "ValueError"


# --------------------------------------------------------------------------- #
# Engine adapter
# --------------------------------------------------------------------------- #


class _StubConfig:
    def __init__(self, sam2_cfg: dict | None = None) -> None:
        self.sam2_cfg = sam2_cfg or {}


def test_engine_adapter_forwards_resident_states_and_device():
    segmenter = SimpleNamespace(name="sam2")
    clip = SimpleNamespace(name="clip")
    registry = ResidentRegistry(loader=lambda: {"segmenter": segmenter, "clip": clip})
    registry.load()

    captured: dict = {}

    def runner(image_rgb, config, **kwargs):
        captured.update(kwargs)
        return "outcome"

    engine = live_engine_callable(registry, runner=runner, device_name="cuda:0")
    outcome = engine(
        "image-array",
        _StubConfig(sam2_cfg={"debug": True}),
        frame_id="req-1",
        segmenter_state=None,
        clip_state=None,
        device="ignored",
        verbosity=2,
        artifact_sink="sink",
        class_labels=("red", "green"),
    )
    assert outcome == "outcome"
    assert captured["segmenter_state"] is segmenter
    assert captured["clip_state"] is clip
    assert captured["blip3_state"] is None
    assert captured["device"] == "cuda:0"
    assert captured["frame_id"] == "req-1"
    assert captured["verbosity"] == 2
    assert captured["artifact_sink"] == "sink"
    assert captured["class_labels"] == ("red", "green")


def test_engine_adapter_rejects_request_level_generator_params():
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})
    registry.load()
    engine = live_engine_callable(registry, runner=lambda *a, **k: "x")
    with pytest.raises(ServiceError) as excinfo:
        engine("img", _StubConfig(sam2_cfg={"points_per_side": 16}), verbosity=0)
    assert excinfo.value.code == "unsupported_field"
    message = str(excinfo.value)
    assert "points_per_side" in message
    assert "resident" in message


def test_engine_adapter_requires_loaded_registry():
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})
    engine = live_engine_callable(registry, runner=lambda *a, **k: "x")
    with pytest.raises(LiveServiceError):
        engine("img", _StubConfig(), verbosity=0)


# --------------------------------------------------------------------------- #
# Readiness composition
# --------------------------------------------------------------------------- #


def test_compose_readiness_registry_gate_comes_first():
    device_calls = []
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})

    def device_provider() -> RuntimeReadiness:
        device_calls.append(1)
        return RuntimeReadiness(True, "device ok")

    provider = compose_readiness(device_provider, registry)
    verdict = provider()
    assert verdict.ready is False
    assert device_calls == []
    registry.load()
    verdict = provider()
    assert verdict.ready is True
    assert len(device_calls) == 1


def test_compose_readiness_propagates_device_failures():
    registry = ResidentRegistry(loader=lambda: {"segmenter": {}, "clip": {}})
    registry.load()

    def device_provider() -> RuntimeReadiness:
        return RuntimeReadiness(False, "visible GPU UUID does not match the operator pin")

    verdict = compose_readiness(device_provider, registry)()
    assert verdict.ready is False
    assert "UUID" in verdict.detail


# --------------------------------------------------------------------------- #
# Operator assets (scripts, deploy template)
# --------------------------------------------------------------------------- #

SCRIPTS = [
    REPO_ROOT / "scripts" / "serve_local.sh",
    REPO_ROOT / "scripts" / "serve_local_stop.sh",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_launcher_scripts_parse(script):
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_launcher_publishes_pid_only_after_exact_entrypoint_ownership():
    launcher = (REPO_ROOT / "scripts" / "serve_local.sh").read_text()
    assert "wait_for_owned_pid" in launcher
    assert 'wait_healthy "$pid" "$SLAIF_ZAP_IT_PORT"' in launcher
    assert "nohup setsid" not in launcher
    assert "sed -n '2p'" in launcher


def test_stop_wrapper_delegates_to_start_script():
    wrapper = (REPO_ROOT / "scripts" / "serve_local_stop.sh").read_text()
    assert 'serve_local.sh" stop' in wrapper


def test_systemd_template_is_uninstalled_optional_asset():
    unit = REPO_ROOT / "deploy" / "zap-it-local.service"
    text = unit.read_text()
    assert "[Unit]" in text and "[Install]" in text
    assert "ExecStart=" in text and "serve_local.sh start" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "SLAIF_ZAP_IT_EXPECTED_GPU_UUID" in text
    assert "127.0.0.1" in text
    assert "SHIPPED UNINSTALLED" in text


def test_entrypoint_script_wires_module_main():
    text = (REPO_ROOT / "scripts" / "serve_local.py").read_text()
    assert "live_service import main" in text.replace("from src.runtime.", "")


# --------------------------------------------------------------------------- #
# Smoke tool helpers (loaded from scripts/ without packaging)
# --------------------------------------------------------------------------- #


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_local_service", REPO_ROOT / "scripts" / "smoke_local_service.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke():
    return _load_smoke_module()


def test_smoke_fixture_is_deterministic_png(smoke):
    first = smoke.make_fixture_png(size=64)
    second = smoke.make_fixture_png(size=64)
    assert first == second
    with Image.open(io.BytesIO(first)) as image:
        assert image.size == (64, 64)
        assert image.mode == "RGB"


def test_smoke_config_keeps_fixture_at_native_scale(smoke):
    from src.service.yaml_input import parse_hostile_config

    validated = parse_hostile_config(smoke.SAFE_CONFIG_YAML, verbosity=0)
    assert validated.effective_mapping["preprocessing"]["resize"] == 1.0


def test_smoke_multipart_cardinality(smoke):
    body, content_type = smoke.build_multipart(
        image_bytes=b"png-bytes",
        config_bytes=b"key: value\n",
        verbosity=2,
        response_format="zip",
        boundary="BOUNDARY",
    )
    text = body.decode("latin-1")
    assert 'name="image"' in text and 'name="config"' in text
    assert 'name="verbosity"' in text and "2" in text
    assert 'name="response_format"' in text and "zip" in text
    assert content_type == "multipart/form-data; boundary=BOUNDARY"


def test_smoke_yolo_parser_bounds_and_shapes(smoke):
    parsed = smoke.parse_yolo_lines("0 0.5 0.25 0.125 0.0625\n2 1.0 0.0 0.5 1.0\n")
    assert [item[0] for item in parsed] == [0, 2]
    assert parsed[0][1:] == (0.5, 0.25, 0.125, 0.0625)
    with pytest.raises(AssertionError):
        smoke.parse_yolo_lines("0 0.5 0.25 0.125\n")
    with pytest.raises(AssertionError):
        smoke.parse_yolo_lines("0 1.5 0.25 0.125 0.5\n")


def test_smoke_error_code_and_invalid_config_fixture(smoke):
    assert smoke.error_code({"error": {"code": "invalid_image"}}) == "invalid_image"
    assert smoke.error_code({"error": {"message": "sanitized"}}) == "?"
    assert smoke.INVALID_CONFIG_YAML == b"[unterminated\n"


def test_smoke_identity_mask_invariants(smoke):
    array = np.zeros((32, 48), dtype=np.uint16)
    array[:10, :12] = 1
    array[20:, 30:] = 2
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    info = smoke.check_identity_png(buffer.getvalue(), 48, 32, expected_ids=2)
    assert info == {"dtype": "uint16", "dims": [32, 48], "object_count": 2}

    wrong_ids = np.zeros((32, 48), dtype=np.uint16)
    wrong_ids[0, 0] = 5
    buffer_wrong = io.BytesIO()
    Image.fromarray(wrong_ids).save(buffer_wrong, format="PNG")
    with pytest.raises(AssertionError):
        smoke.check_identity_png(buffer_wrong.getvalue(), 48, 32, expected_ids=2)


# --------------------------------------------------------------------------- #
# Resident CLIP label resync (no torch required)
# --------------------------------------------------------------------------- #


def test_clip_class_map_helper_matches_legacy_parsing():
    from modules.classifier.clip import _class_map_from

    config = {
        "labels": {"red": "a red object,\nbright red", "green": " a green object "},
        "label blue": "a blue thing",
        "padding": 7,
        "model_name": "openai/clip-vit-base-patch32",
    }
    class_map = _class_map_from(config)
    assert class_map["red"] == ["a red object", "bright red"]
    assert class_map["green"] == ["a green object"]
    assert class_map["blue"] == ["a blue thing"]
    assert class_map != {} and "padding" not in class_map


def test_clip_update_labels_resident_resync_semantics():
    from modules.classifier.clip import _ClipFilter

    instance = _ClipFilter.__new__(_ClipFilter)
    instance.class_map = {}
    instance._torch = None
    instance.processor = None
    instance.model = None
    instance.device = "cpu"
    instance.text_embeds = None
    encodes = []
    instance._encode_text_prompts = lambda: encodes.append(1)

    assert instance.update_labels({"labels": {}}) is False
    changed = instance.update_labels({"labels": {"red": "a red object"}})
    assert changed is True and len(encodes) == 1
    assert instance.class_map == {"red": ["a red object"]}
    assert instance.all_prompts == ["a red object"]
    assert instance.class_idx == ["red"]
    back_to_empty = instance.update_labels({})
    assert back_to_empty is True and len(encodes) == 1
    assert instance.all_prompts == [] and instance.text_embeds is None
