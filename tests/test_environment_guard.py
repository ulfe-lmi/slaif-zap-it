"""Environment-guard tests proving the CPU suite stays offline and GPU-free."""

import sys


def test_no_cuda_path_executes():
    import torch

    # Under the stub harness this is always False; under any future environment
    # where real torch were present it must STILL be False for this suite.
    assert not torch.cuda.is_available()
    assert torch.cuda.device_count() == 0


def test_torch_module_is_stub():
    torch = sys.modules.get("torch")
    assert torch is not None, "torch stub must be installed by conftest"
    # The stub is a synthetic module without a real extension file on disk.
    assert not getattr(torch, "__file__", None)


def test_network_sockets_are_blocked():
    import socket

    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except AssertionError as exc:
        assert "sockets are disabled" in str(exc)
    else:
        raise AssertionError("socket creation unexpectedly succeeded; the offline guard failed")
