#!/usr/bin/env python3
"""Install private operator config and a user unit without disclosing secrets."""

from __future__ import annotations

import argparse
import os
import secrets
import tempfile
from pathlib import Path


def _atomic_write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_api_key(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name == "SLAIF_ZAP_IT_API_KEY" and len(value) >= 32:
            return value
    return None


def install(
    *,
    repo_root: Path,
    config_path: Path,
    unit_path: Path,
    host: str,
    cidr: str,
    port: int,
    physical_gpu_index: int,
    expected_gpu_uuid: str,
    model_cache_root: Path,
) -> tuple[Path, Path, bool]:
    """Write the private environment and user unit; return whether a key was created."""
    for value in (host, cidr, expected_gpu_uuid, str(model_cache_root), str(repo_root)):
        if not value or any(character in value for character in "\r\n\0"):
            raise ValueError("operator values must be non-empty single-line strings")
    if not 1 <= port <= 65535 or physical_gpu_index < 0:
        raise ValueError("port/index is outside its valid range")
    python = repo_root / ".venv-gpu" / "bin" / "python"
    entrypoint = repo_root / "scripts" / "serve_local.py"
    if not python.is_file() or not os.access(python, os.X_OK) or not entrypoint.is_file():
        raise ValueError("reviewed repository GPU runtime/entrypoint is unavailable")

    existing_key = _existing_api_key(config_path)
    api_key = existing_key or secrets.token_urlsafe(48)
    environment = "\n".join(
        (
            "# Private ZAP-IT operator configuration. Never commit or share this file.",
            f"SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX={physical_gpu_index}",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
            f"CUDA_VISIBLE_DEVICES={physical_gpu_index}",
            f"SLAIF_ZAP_IT_EXPECTED_GPU_UUID={expected_gpu_uuid}",
            "SLAIF_ZAP_IT_NETWORK_SCOPE=private_lan",  # pragma: allowlist secret
            f"SLAIF_ZAP_IT_HOST={host}",
            f"SLAIF_ZAP_IT_PRIVATE_LAN_CIDR={cidr}",
            f"SLAIF_ZAP_IT_PORT={port}",
            "SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it",  # pragma: allowlist secret
            f"SLAIF_ZAP_IT_MODEL_CACHE_ROOT={model_cache_root}",
            "SLAIF_ZAP_IT_STRICT_GPU=1",
            f"SLAIF_ZAP_IT_API_KEY={api_key}",
            "SLAIF_ZAP_IT_MODEL_CONTROL_MODE=none",
            "SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS=120",
            "SLAIF_ZAP_IT_QUEUE_DEPTH=0",
            "SLAIF_ZAP_IT_RETRY_AFTER_SECONDS=5",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "",
        )
    )
    unit = f"""[Unit]
Description=SLAIF ZAP-IT authenticated private-LAN service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile={config_path}
WorkingDirectory={repo_root}
ExecStart={python} {entrypoint}
Restart=on-failure
RestartSec=5
TimeoutStartSec=600
TimeoutStopSec=180
KillMode=control-group
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/dev/shm/slaif-zap-it
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=default.target
"""
    _atomic_write(config_path, environment, 0o600)
    _atomic_write(unit_path, unit, 0o644)
    return config_path, unit_path, existing_key is None


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--cidr", required=True)
    parser.add_argument("--port", type=int, default=17891)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--model-cache-root", type=Path, default=home / ".cache/huggingface")
    parser.add_argument(
        "--config-path", type=Path, default=home / ".config/slaif-zap-it/service.env"
    )
    parser.add_argument(
        "--unit-path", type=Path, default=home / ".config/systemd/user/zap-it-lan.service"
    )
    args = parser.parse_args(argv)
    config, unit, created = install(
        repo_root=repo_root,
        config_path=args.config_path,
        unit_path=args.unit_path,
        host=args.host,
        cidr=args.cidr,
        port=args.port,
        physical_gpu_index=args.physical_gpu_index,
        expected_gpu_uuid=args.expected_gpu_uuid,
        model_cache_root=args.model_cache_root,
    )
    action = "generated" if created else "preserved"
    print(f"private LAN config installed: {config} (mode 0600; API key {action}, not shown)")
    print(f"user unit installed: {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
