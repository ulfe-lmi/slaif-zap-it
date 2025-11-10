"""
zap-it-config.py

Houses the function(s) for loading & parsing the YAML configuration once.
Add or refine any config-specific utilities here.
"""

import yaml


def _print_enabled_modules(config: dict, verbosity: int = 1) -> None:
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

def load_config(config_path, verbosity_level="some"):
    """
    Load and parse the YAML config file in one step. Also do minor housekeeping:
      1) Convert 'roi: False' into a Python None in 'preprocessing' if present.
      2) Copy 'visualization.alpha' to top-level config['alpha'] for convenience.

    :param config_path: Path to YAML config file.
    :param verbosity_level: textual indication ("none", "some", "full").
    :return: (config_dict, vb) => (parsed YAML dict, integer verbosity)
    """
    if verbosity_level == "none":
        vb = 0
    elif verbosity_level == "full":
        vb = 2
    else:
        vb = 1

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # 1) If 'alpha' is in visualization, store it top-level
    vis_cfg = raw_config.get("visualization", {})
    alpha = vis_cfg.get("alpha", 0.6)
    raw_config["alpha"] = alpha

    # 2) Convert 'roi: False' => None
    prep_cfg = raw_config.get("preprocessing", {})
    roi_val = prep_cfg.get("roi", None)
    if roi_val is False:
        prep_cfg["roi"] = None
        raw_config["preprocessing"] = prep_cfg

    _print_enabled_modules(raw_config, vb)

    return raw_config, vb
