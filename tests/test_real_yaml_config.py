"""Config-grammar tests that exercise the REAL ``yaml.safe_load`` path.

The historical test harness stubs ``yaml`` when PyYAML is absent, so most
legacy tests never parse true YAML. The dev environment installs PyYAML, which
takes precedence over the stub, so this module validates both the shipped
example configurations and :func:`src.config.load_config` behavior against the
real parser.
"""

import copy
from pathlib import Path

import pytest
import yaml

from src.config import _print_enabled_modules, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
# Goat YAML is a local-only academic input and is intentionally absent from
# tracked source distributions. CI tests use the remaining redistributable
# examples; the opt-in goat harness owns its local regression path.
EXAMPLE_CONFIGS = sorted(
    path for path in CONFIG_DIR.glob("*.yaml") if path.name not in {"goats.yaml", "goats2.yaml"}
)


def _real_yaml_available() -> bool:
    """Guard: skip honestly if PyYAML was stubbed by the test harness."""
    return getattr(yaml, "__file__", None) is not None


pytestmark = pytest.mark.skipif(
    not _real_yaml_available(), reason="real PyYAML required for this module"
)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_configs_parse_with_real_yaml(config_path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    # Every example must at least define SAM2 mask-generator settings.
    assert "mask_generator" in raw


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_configs_load_through_load_config(config_path):
    config, verbosity = load_config(str(config_path), verbosity_level="none")
    assert verbosity == 0
    assert isinstance(config, dict)
    # load_config always injects an alpha value used by visualizers.
    assert 0.0 <= config["alpha"] <= 1.0


def test_roi_false_is_normalized_to_none(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
preprocessing:
  roi: false
  resize: 1.0
""",
        encoding="utf-8",
    )
    config, _ = load_config(str(config_path), verbosity_level="none")
    assert config["preprocessing"]["roi"] is None


def test_alpha_defaults_applied_when_visualization_missing(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("mask_generator: {}\n", encoding="utf-8")
    config, _ = load_config(str(config_path), verbosity_level="none")
    assert config["alpha"] == 0.6


def test_alpha_from_visualization_section(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
visualization:
  alpha: 0.25
""",
        encoding="utf-8",
    )
    config, _ = load_config(str(config_path), verbosity_level="none")
    assert config["alpha"] == 0.25


@pytest.mark.parametrize(
    ("verbosity_level", "expected"),
    [("none", 0), ("some", 1), ("full", 2)],
)
def test_verbosity_mapping(tmp_path, verbosity_level, expected):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _, verbosity = load_config(str(config_path), verbosity_level=verbosity_level)
    assert verbosity == expected


def test_module_overview_reflects_enabled_flags(capsys):
    config = {"clip": {"padding": 4}, "export_yolo_det": {"labels": "x"}}
    _print_enabled_modules(copy.deepcopy(config), verbosity=1)
    out = capsys.readouterr().out
    assert "[config] Module overview:" in out
    assert "- CLIP filter: enabled" in out
    assert "- BLIP3 verification: disabled" in out
    assert "- YOLO export: enabled" in out


def test_safe_load_rejects_non_mapping_top_level(tmp_path):
    """A top-level sequence parses fine in YAML but is not a valid ZAP-IT config."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(AttributeError):
        load_config(str(config_path), verbosity_level="none")
