from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime.device import (
    DeviceGuardError,
    inspect_visible_device,
    launch_environment,
    require_launch_environment,
)
from src.runtime.ports import PortCheck
from src.runtime.readiness import make_readiness_provider
from src.runtime.shm import ShmError, ShmWorkspace, ensure_shm_root
from src.runtime.strategy import RuntimePolicy, UnsupportedProfileError


class _FakeCuda:
    def __init__(self, *, available=True, count=1, uuid="GPU-target"):
        self._available = available
        self._count = count
        self._uuid = uuid

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_name(self, index):
        assert index == 0
        return "NVIDIA GeForce RTX 2080 Ti"

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(uuid=self._uuid, total_memory=11264 * 1024 * 1024)


def _fake_torch(**kwargs):
    return SimpleNamespace(cuda=_FakeCuda(**kwargs))


def test_launch_environment_is_the_physical_gpu1_contract():
    expected = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}
    assert launch_environment() == expected
    require_launch_environment(expected)


def test_launch_environment_rejects_missing_mask():
    with pytest.raises(DeviceGuardError):
        require_launch_environment({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})


def test_device_guard_reports_one_pinned_visible_gpu():
    report = inspect_visible_device(_fake_torch(), expected_uuid="GPU-target", strict=True)
    assert report.mode == "gpu"
    assert report.visible_count == 1
    assert report.logical_index == 0
    assert report.uuid == "GPU-target"
    assert report.total_memory_mib == 11264


def test_device_guard_fails_closed_on_wrong_uuid_and_count():
    with pytest.raises(DeviceGuardError, match="UUID"):
        inspect_visible_device(_fake_torch(uuid="GPU-other"), expected_uuid="GPU-target")
    with pytest.raises(DeviceGuardError, match="exactly one"):
        inspect_visible_device(_fake_torch(count=2), expected_uuid="GPU-target")


def test_device_guard_allows_explicit_cpu_test_mode():
    report = inspect_visible_device(
        _fake_torch(available=False, count=0), expected_uuid=None, strict=False
    )
    assert report.mode == "cpu"
    assert report.logical_index is None


def test_runtime_policy_rejects_client_requested_unsupported_profile():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target")
    config = SimpleNamespace(clip_cfg={"labels": {"thing": "a thing"}}, blip3_cfg={})
    assert policy.validate_config(config) == "sam2_clip"
    blip_config = SimpleNamespace(clip_cfg={}, blip3_cfg={"thing": {"question": "what?"}})
    with pytest.raises(UnsupportedProfileError):
        policy.validate_config(blip_config)


def test_runtime_policy_readiness_is_not_ready_until_device_and_registry_are_ready():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target")
    assert not policy.readiness(None).ready
    report = inspect_visible_device(_fake_torch(), expected_uuid="GPU-target")
    assert not policy.readiness(report).ready
    ready = policy.with_model_registry_ready().readiness(report)
    assert ready.ready


def test_readiness_provider_joins_device_guard_and_operator_registry_state():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target").with_model_registry_ready()
    provider = make_readiness_provider(
        policy,
        torch_module=_fake_torch(),
        environ=launch_environment(),
    )
    assert provider().ready


def test_readiness_provider_fails_closed_on_device_mismatch():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target").with_model_registry_ready()
    provider = make_readiness_provider(
        policy,
        torch_module=_fake_torch(uuid="GPU-other"),
        environ=launch_environment(),
    )
    state = provider()
    assert not state.ready
    assert "UUID" in state.detail


def test_readiness_provider_fails_closed_without_the_launch_mask():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target").with_model_registry_ready()
    provider = make_readiness_provider(policy, torch_module=_fake_torch(), environ={})
    state = provider()
    assert not state.ready
    assert "strict GPU mode" in state.detail


def test_shm_workspace_has_opaque_permissions_and_cleans_up(tmp_path):
    root = ensure_shm_root(tmp_path / "shm")
    assert root.stat().st_mode & 0o777 == 0o700
    with ShmWorkspace(root) as workspace:
        request_dir = workspace.path
        assert request_dir is not None
        assert request_dir.stat().st_mode & 0o777 == 0o700
        artifact = workspace.write_bytes("config.yaml", b"alpha: 0.6\n")
        assert artifact.read_bytes() == b"alpha: 0.6\n"
        assert artifact.stat().st_mode & 0o777 == 0o600
        with pytest.raises(ShmError):
            workspace.write_bytes("../escape", b"no")
    assert request_dir is not None
    assert not request_dir.exists()


def test_shm_root_refuses_insecure_existing_directory(tmp_path):
    root = tmp_path / "shm"
    root.mkdir(mode=0o755)
    with pytest.raises(ShmError, match="0700"):
        ensure_shm_root(root)


def test_port_check_requires_both_inspection_and_transient_bind():
    assert PortCheck("127.0.0.1", 17891, False, True).unused
    assert not PortCheck("127.0.0.1", 17891, True, True).unused
    assert not PortCheck("127.0.0.1", 17891, False, False).unused
