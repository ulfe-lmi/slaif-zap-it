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

    # Socket objects may be constructed (asyncio event loops need internal
    # self-pipe sockets), but any outbound connection attempt outside
    # loopback must fail loudly so the CPU suite stays offline.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            sock.connect(("example.com", 80))
        except AssertionError as exc:
            assert "disabled during the ZAP-IT CPU test suite" in str(exc)
            assert "example.com" in str(exc)
        else:
            raise AssertionError(
                "outbound connect unexpectedly succeeded; the offline guard failed"
            )
    finally:
        sock.close()


def test_loopback_connections_remain_available():
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(("127.0.0.1", port))
            accepted, _ = server.accept()
            accepted.close()
        finally:
            client.close()
    finally:
        server.close()
