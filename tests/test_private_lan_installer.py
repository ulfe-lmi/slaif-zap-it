from __future__ import annotations

import stat
from pathlib import Path

from scripts.install_private_lan_service import install


REPO_ROOT = Path(__file__).resolve().parents[1]


def _value(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key == name:
            return value
    raise AssertionError(f"missing {name}")


def test_installer_writes_private_config_and_preserves_fixed_key(tmp_path):
    config = tmp_path / "config" / "service.env"
    unit = tmp_path / "systemd" / "zap-it-lan.service"
    kwargs = {
        "repo_root": REPO_ROOT,
        "config_path": config,
        "unit_path": unit,
        "host": "10.8.132.76",
        "cidr": "10.8.132.0/24",
        "port": 17891,
        "physical_gpu_index": 0,
        "expected_gpu_uuid": "GPU-a91444df-4e87-011e-3347-9b3a4b9f9575",
        "model_cache_root": tmp_path / "cache",
    }
    _, _, created = install(**kwargs)
    first_key = _value(config, "SLAIF_ZAP_IT_API_KEY")
    assert created and len(first_key) >= 32
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert _value(config, "SLAIF_ZAP_IT_NETWORK_SCOPE") == "private_lan"
    assert _value(config, "SLAIF_ZAP_IT_MODEL_CONTROL_MODE") == "none"
    assert "0.0.0.0" not in config.read_text(encoding="utf-8")
    assert first_key not in unit.read_text(encoding="utf-8")
    assert "NoNewPrivileges=true" in unit.read_text(encoding="utf-8")

    _, _, created_again = install(**kwargs)
    assert not created_again
    assert _value(config, "SLAIF_ZAP_IT_API_KEY") == first_key
