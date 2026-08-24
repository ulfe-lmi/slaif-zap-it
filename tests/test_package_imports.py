"""Package/import smoke tests for the packaged modules.

These run inside the canonical CPU suite where ``tests/conftest.py`` provides
the documented stub harness, proving the package layout imports cleanly
without GPU libraries. Importing outside this harness requires the qualified
GPU environment (see TESTING.md and docs/runtime.md).
"""

from importlib import import_module
from pathlib import Path

PACKAGES = [
    "src",
    "src.batch",
    "src.config",
    "src.postprocessing",
    "src.runtime",
    "src.runtime.device",
    "src.runtime.strategy",
    "modules",
    "modules.visualizer",
    "modules.input",
    "modules.input.images",
    "modules.input.video",
    "modules.output",
    "modules.output.images",
    "modules.output.video",
    "modules.output.yolo",
    "modules.segmenter",
    "modules.segmenter.sam2",
    "modules.classifier",
    "modules.classifier.clip",
    "modules.verifier",
    "modules.verifier.blip3",
    "modules.geometry",
    "modules.geometry.geometry",
]


def test_all_packages_import():
    for name in PACKAGES:
        import_module(name)


def test_shims_re_export_current_implementations():
    zap_it_config = import_module("zap_it_config")
    from src import config as src_config

    assert set(zap_it_config.__all__) == {"load_config", "_print_enabled_modules"}
    assert zap_it_config.load_config is src_config.load_config
    assert zap_it_config._print_enabled_modules is src_config._print_enabled_modules

    zap_it_postseg = import_module("zap_it_postseg_processing")
    from src import postprocessing as src_postprocessing

    # The deprecated shim wraps the real filter; verify it still delegates.
    assert zap_it_postseg.__all__ == ["filter_by_area_bbox"]
    assert zap_it_postseg._filter_by_area_bbox is src_postprocessing.filter_by_area_bbox


def test_src_public_surface_resolves():
    src = import_module("src")
    for name in src.__all__:
        assert getattr(src, name, None) is not None, f"src.{name} missing"


def test_repo_root_scripts_present():
    root = Path(__file__).resolve().parents[1]
    for script in (
        "zap-it-batch.py",
        "zap_it_config.py",
        "zap_it_postseg_processing.py",
        "huggingface_downloader.py",
    ):
        assert (root / script).is_file(), f"{script} missing"
