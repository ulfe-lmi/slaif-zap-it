import pytest

from src.config import _print_enabled_modules, load_config
import zap_it_config
import zap_it_postseg_processing


def test_load_config_normalizes_and_reports(tmp_path, capsys):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
alpha: 0.1
visualization:
  alpha: 0.8
  labels: ["keep"]
preprocessing:
  roi: false
mask_generator: {}
clip: {enabled: true}
"""
    )

    cfg, verbosity = load_config(str(cfg_path), verbosity_level="full")

    assert cfg["alpha"] == pytest.approx(0.8)
    assert cfg["preprocessing"]["roi"] is None
    assert verbosity == 2

    captured = capsys.readouterr().out
    assert "Module overview" in captured
    assert "CLIP filter" in captured


def test_print_enabled_modules_lists_expected(capsys):
    _print_enabled_modules(
        {
            "clip": {"enabled": True},
            "blip3": {},
            "export_yolo_det": {},
        },
        verbosity=1,
    )
    out = capsys.readouterr().out
    assert "SAM2 mask generator" in out
    assert "CLIP filter" in out
    assert "YOLO export" in out


def test_shims_forward_and_warn(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("alpha: 0.3\nmask_generator: {}\n")

    shim_cfg, _ = zap_it_config.load_config(str(cfg_path))
    orig_cfg, _ = load_config(str(cfg_path))
    assert shim_cfg == orig_cfg

    sample_mask = {
        "area": 1,
        "segmentation": [[True]],
    }
    with pytest.warns(DeprecationWarning):
        zap_it_postseg_processing.filter_by_area_bbox([sample_mask], 10, 1, 1)
