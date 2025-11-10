"""Configuration loading utilities for ZAP-IT."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import yaml


def _print_enabled_modules(config: Dict[str, Any], verbosity: int = 1) -> None:
    """Print a short summary of which optional modules are enabled."""
    modules = {
        "SAM2 mask generator": True,
        "CLIP filter": bool(config.get("clip")),
        "BLIP3 verification": bool(config.get("blip3")),
        "YOLO export": bool(config.get("export_yolo_det")),
    }

    if verbosity >= 1:
        print("[config] Module overview:")
        for name, flag in modules.items():
            status = "enabled" if flag else "disabled"
            print(f"  - {name}: {status}")


def load_config(config_path: str, verbosity_level: str = "some") -> Tuple[Dict[str, Any], int]:
    """Load and parse the YAML config file in one step."""
    if verbosity_level == "none":
        vb = 0
    elif verbosity_level == "full":
        vb = 2
    else:
        vb = 1

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    vis_cfg = raw_config.get("visualization", {})
    alpha = vis_cfg.get("alpha", 0.6)
    raw_config["alpha"] = alpha

    prep_cfg = raw_config.get("preprocessing", {})
    roi_val = prep_cfg.get("roi", None)
    if roi_val is False:
        prep_cfg["roi"] = None
        raw_config["preprocessing"] = prep_cfg

    _print_enabled_modules(raw_config, vb)

    return raw_config, vb


__all__ = ["load_config", "_print_enabled_modules"]
