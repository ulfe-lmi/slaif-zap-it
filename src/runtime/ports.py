"""Non-persistent explicitly scoped port qualification."""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass

__all__ = ["PortCheck", "select_candidate_port", "verify_port_unused"]


@dataclass(frozen=True)
class PortCheck:
    host: str
    port: int
    ss_listener: bool
    bind_succeeded: bool

    @property
    def unused(self) -> bool:
        return not self.ss_listener and self.bind_succeeded


def _ss_has_listener(host: str, port: int) -> bool:
    result = subprocess.run(
        ["ss", "-H", "-ltn"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ss listener inspection failed")
    suffix = f":{port}"
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if not local.endswith(suffix):
            continue
        if local.startswith(f"{host}:") or local.startswith("0.0.0.0:") or local.startswith("*:"):
            return True
    return False


def verify_port_unused(host: str, port: int) -> PortCheck:
    """Use ``ss`` plus a transient bind; never leave a listener behind."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be in the TCP range")
    listener = _ss_has_listener(host, port)
    bind_ok = False
    if not listener:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            bind_ok = True
        except OSError:
            bind_ok = False
        finally:
            sock.close()
    return PortCheck(host, port, listener, bind_ok)


def select_candidate_port(
    host: str = "127.0.0.1",
    *,
    preferred: tuple[int, ...] = (17891, 23654),
    fallback_start: int = 20000,
    fallback_end: int = 40000,
) -> PortCheck:
    """Select the first verified-unused candidate without reserving it."""
    candidates = list(preferred) + list(range(fallback_start, fallback_end + 1))
    for port in candidates:
        check = verify_port_unused(host, port)
        if check.unused:
            return check
    raise RuntimeError("no candidate service port was verified unused")
