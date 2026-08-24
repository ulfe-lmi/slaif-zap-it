from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

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
from src.runtime.strategy import RuntimePolicy


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


@pytest.fixture
def shm_test_root():
    root = Path("/dev/shm") / f"zap-it-pytest-{uuid4().hex}"
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


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


def test_runtime_policy_supports_blip3_profiles():
    policy = RuntimePolicy(expected_gpu_uuid="GPU-target")
    config = SimpleNamespace(clip_cfg={"labels": {"thing": "a thing"}}, blip3_cfg={})
    assert policy.validate_config(config) == "sam2_clip"
    blip_config = SimpleNamespace(clip_cfg={}, blip3_cfg={"thing": {"question": "what?"}})
    assert policy.validate_config(blip_config) == "sam2_blip3"


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


def test_shm_workspace_has_opaque_permissions_and_cleans_up(shm_test_root):
    root = ensure_shm_root(shm_test_root)
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


def test_shm_root_accepts_a_canonical_descendant(shm_test_root):
    root = ensure_shm_root(shm_test_root / "nested" / ".." / "canonical")
    assert root == (shm_test_root / "canonical").resolve()
    assert root.stat().st_mode & 0o777 == 0o700


def test_shm_root_refuses_insecure_existing_directory(shm_test_root):
    root = shm_test_root
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(ShmError, match="0700"):
        ensure_shm_root(root)


@pytest.mark.parametrize("kind", ["escape", "root", "persistent"])
def test_shm_root_rejects_outside_or_non_descendant_paths(kind, shm_test_root):
    if kind == "escape":
        candidate = Path("/dev/shm/../../tmp") / f"zap-it-{uuid4().hex}"
    elif kind == "root":
        candidate = Path("/dev/shm")
    else:
        candidate = Path("/tmp") / f"zap-it-{uuid4().hex}"
    with pytest.raises(ShmError, match="strict descendant") as excinfo:
        ensure_shm_root(candidate)
    if kind != "root":
        assert str(candidate) not in str(excinfo.value)


def test_shm_root_rejects_intermediate_symlink_escape(shm_test_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    intermediate = shm_test_root / "intermediate"
    shm_test_root.mkdir(mode=0o700)
    intermediate.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ShmError, match="strict descendant"):
        ensure_shm_root(intermediate / "child")


def test_shm_root_rejects_final_symlink(shm_test_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    shm_test_root.mkdir(mode=0o700)
    final = shm_test_root / "final"
    final.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ShmError, match="must not be a symlink"):
        ensure_shm_root(final)


def test_port_check_requires_both_inspection_and_transient_bind():
    assert PortCheck("127.0.0.1", 17891, False, True).unused
    assert not PortCheck("127.0.0.1", 17891, True, True).unused
    assert not PortCheck("127.0.0.1", 17891, False, False).unused
