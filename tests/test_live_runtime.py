from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime import live_service
from src.runtime.ports import PortCheck
from src.runtime.strategy import (
    SUPPORTED_RESIDENT_PROFILES,
    SUPPORTED_RESIDENT_STRATEGY,
)


TARGET_UUID = "GPU-c457dbaf-991c-dc23-c781-0dc030776dd8"


def test_live_config_is_loopback_ephemeral_and_pinned_to_physical_gpu1():
    config = live_service.LiveServiceConfig(
        port=17891,
        expected_gpu_uuid=TARGET_UUID,
        tmp_root="/dev/shm/slaif-zap-it",
    )
    assert config.host == "127.0.0.1"
    assert config.physical_gpu_index == 1

    with pytest.raises(live_service.LiveServiceError, match="physical GPU index 1"):
        live_service.LiveServiceConfig(
            port=17891,
            expected_gpu_uuid=TARGET_UUID,
            physical_gpu_index=0,
        )
    # CPU preflight tests may use pytest's isolated temporary directory; the
    # real entrypoint enforces the /dev/shm boundary before serving.
    assert (
        live_service.LiveServiceConfig(
            port=17891,
            expected_gpu_uuid=TARGET_UUID,
            tmp_root="/tmp/zap-it",
        ).tmp_root
        == "/tmp/zap-it"
    )


def test_live_config_reads_operator_cache_and_port_without_client_fields():
    config = live_service.LiveServiceConfig.from_environment(
        {
            "SLAIF_ZAP_IT_PORT": "23654",
            "SLAIF_ZAP_IT_EXPECTED_GPU_UUID": TARGET_UUID,
            "SLAIF_ZAP_IT_TMP_ROOT": "/dev/shm/slaif-zap-it",
            "SLAIF_ZAP_IT_MODEL_CACHE_ROOT": "/dev/shm/model-cache",
        }
    )
    assert config.port == 23654
    assert config.model_cache_root == "/dev/shm/model-cache"


def test_preflight_checks_all_operator_boundaries(monkeypatch):
    calls = []
    monkeypatch.setattr(
        live_service, "require_launch_environment", lambda *a, **k: calls.append("env")
    )
    monkeypatch.setattr(live_service, "ensure_shm_root", lambda root: root)
    monkeypatch.setattr(live_service, "shm_free_bytes", lambda root: 128 * 1024 * 1024)
    monkeypatch.setattr(
        live_service,
        "verify_port_unused",
        lambda host, port: PortCheck(host, port, False, True),
    )
    result = live_service.preflight(
        live_service.LiveServiceConfig(
            port=17891,
            expected_gpu_uuid=TARGET_UUID,
            tmp_root="/dev/shm/slaif-zap-it",
        )
    )
    assert result.launch_environment_ok
    assert result.port_check.unused
    assert result.shm_free_mib == 128.0
    assert calls == ["env"]


def test_resident_registry_has_one_load_and_honest_failure_state():
    load_calls = []
    registry = live_service.ResidentRegistry(
        loader=lambda: load_calls.append("load") or {"segmenter": object(), "clip": object()}
    )
    assert not registry.verdict().ready
    registry.start_background_load()
    assert registry.wait_until_settled(timeout=2)
    assert registry.ready
    registry.start_background_load()
    assert load_calls == ["load"]
    assert registry.states()["segmenter"] is not None
    registry.shutdown()

    failed = live_service.ResidentRegistry(loader=lambda: (_ for _ in ()).throw(OSError("hidden")))
    failed.load()
    assert failed.failed
    assert not failed.verdict().ready
    assert "OSError" in failed.verdict().detail


def test_live_engine_uses_resident_states_and_logical_cuda0():
    registry = live_service.ResidentRegistry(loader=lambda: {"segmenter": "sam", "clip": "clip"})
    registry.load()
    captured = {}

    def runner(image, config, **kwargs):
        captured.update(kwargs)
        return "outcome"

    engine = live_service.live_engine_callable(registry, runner=runner)
    config = SimpleNamespace(sam2_cfg={}, blip3_cfg={})
    assert engine("image", config, class_labels=("red",)) == "outcome"
    assert captured["segmenter_state"] == "sam"
    assert captured["clip_state"] == "clip"
    assert captured["device"] == "cuda:0"

    with pytest.raises(Exception, match="fixed"):
        engine("image", SimpleNamespace(sam2_cfg={"points_per_side": 8}, blip3_cfg={}))


def test_operator_profile_constants_are_the_qualified_strategy():
    assert SUPPORTED_RESIDENT_STRATEGY == "sam2_clip_resident_blip3_rejected"
    assert SUPPORTED_RESIDENT_PROFILES == ("sam2", "clip", "sam2_clip")
