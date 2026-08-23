"""CPU-only release-candidate invariants using generated data only."""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_central_crop_uses_exact_integer_middle_half_without_disk_write(tmp_path):
    goats = _load_script("smoke_local_goats")
    image = Image.new("RGB", (13, 9), (12, 34, 56))
    raw = io.BytesIO()
    image.save(raw, format="PNG")

    cropped, original, dimensions = goats.central_crop(raw.getvalue())

    assert original == (13, 9)
    assert dimensions == (6, 4)
    with Image.open(io.BytesIO(cropped)) as decoded:
        assert decoded.size == dimensions
    assert list(tmp_path.iterdir()) == []


def test_goat_derivation_is_allowlisted_and_strips_operator_controls():
    goats = _load_script("smoke_local_goats")
    raw = yaml.safe_dump(
        {
            "mask_generator": {"points_per_side": 4},
            "clip": {"labels": {"synthetic": "a synthetic object"}},
            "model_repo": "private-model",
            "blip3": {"question": "private prompt"},
            "images": {"input_dir": "/private/input"},
            "output_dir": "/private/output",
        }
    ).encode()

    derived, stripped = goats.derive_api_safe_config(raw)
    mapping = yaml.safe_load(derived)

    assert mapping["mask_generator"] == {}
    assert "model_repo" not in mapping
    assert "blip3" not in mapping
    assert "images" not in mapping
    assert "output_dir" not in mapping
    assert stripped == 5


def test_goat_path_guard_rejects_symlink_and_out_of_root(tmp_path):
    goats = _load_script("smoke_local_goats")
    root = tmp_path / "root"
    root.mkdir()
    safe = root / "safe.yaml"
    safe.write_text("alpha: 0.6\n", encoding="utf-8")
    outside = tmp_path / "outside.yaml"
    outside.write_text("alpha: 0.6\n", encoding="utf-8")

    assert goats._safe_path(safe, root, "config") == safe.resolve()
    with pytest.raises(ValueError):
        goats._safe_path(outside, root, "config")
    link = root / "link.yaml"
    link.symlink_to(safe)
    with pytest.raises(ValueError):
        goats._safe_path(link, root, "config")


def test_artifact_verifier_rejects_payload_names_and_traversal(tmp_path):
    verifier = _load_script("verify_release_artifacts")
    with pytest.raises(verifier.ArtifactError):
        verifier._validate_name("demos/goats/goats2.jpg", 1)
    with pytest.raises(verifier.ArtifactError):
        verifier._validate_name("../escape.txt", 1)

    archive_path = tmp_path / "safe.whl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("zap_it-0.1.0.dist-info/METADATA", "Name: zap-it\n")
        archive.writestr("zap_it-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr("zap_it-0.1.0.dist-info/entry_points.txt", "[console_scripts]\n")
        archive.writestr("zap_it-0.1.0.dist-info/licenses/LICENSE", "MIT\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/README.md", "readme\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/THIRD_PARTY_NOTICES.md", "notice\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/CHANGELOG.md", "change\n")
    evidence = verifier.inspect_archive(str(archive_path))
    assert evidence["member_count"] == 7
