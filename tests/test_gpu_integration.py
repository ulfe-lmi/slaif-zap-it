"""Opt-in, serialized physical-GPU1 smoke checks.

Normal CPU CI does not collect live model tests as a requirement.  Operators
run this module explicitly with ``ZAP_IT_RUN_GPU=1`` and the launch mask.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from src.runtime.device import inspect_visible_device, require_launch_environment
from src.runtime.shm import ensure_shm_root

pytestmark = pytest.mark.gpu

if os.environ.get("ZAP_IT_RUN_GPU") != "1":
    pytest.skip("live GPU tests require ZAP_IT_RUN_GPU=1", allow_module_level=True)


@pytest.fixture()
def gpu_test_lock():
    """Serialize live checks with a mode-0600 lock under the RAM root."""
    import fcntl

    root = ensure_shm_root(os.environ.get("SLAIF_ZAP_IT_TMP_ROOT", "/dev/shm/slaif-zap-it"))
    lock_path = root / "gpu-test.lock"
    handle = lock_path.open("a+b")
    os.chmod(lock_path, 0o600)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_masked_device_is_pinned_gpu1_and_gpu0_is_untouched(gpu_test_lock, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    require_launch_environment(physical_gpu_index=1)
    import torch

    expected_uuid = os.environ.get("SLAIF_ZAP_IT_EXPECTED_GPU_UUID")
    if not expected_uuid:
        pytest.skip("SLAIF_ZAP_IT_EXPECTED_GPU_UUID is required for strict live tests")
    before = _compute_processes()
    report = inspect_visible_device(torch, expected_uuid=expected_uuid, strict=True)
    after = _compute_processes()
    assert report.logical_index == 0
    assert report.uuid == expected_uuid
    assert report.visible_count == 1
    assert [line for line in after if "GPU-4c129e25" in line] == [
        line for line in before if "GPU-4c129e25" in line
    ]
