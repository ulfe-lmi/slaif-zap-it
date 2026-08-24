"""CPU-only release-candidate invariants using generated data only."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import re
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

REMOVED_DEMO_DIRECTORIES = (
    "demos/glasswool",
    "demos/icecream",
    "demos/industrial",
    "demos/soccer",
)


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


def test_purged_demo_datasets_remain_absent_and_ignored():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for relative in REMOVED_DEMO_DIRECTORIES:
        assert not (REPO_ROOT / relative).exists()
        assert f"/{relative}/" in ignore


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
            "blip3": {"synthetic": {"question": "private prompt"}},
            "images": {"input_dir": "/private/input"},
            "output_dir": "/private/output",
        }
    ).encode()

    derived, stripped = goats.derive_api_safe_config(raw)
    mapping = yaml.safe_load(derived)

    assert mapping["mask_generator"] == {}
    assert "model_repo" not in mapping
    assert mapping["blip3"]["synthetic"]["question"] == "private prompt"
    assert "images" not in mapping
    assert "output_dir" not in mapping
    assert stripped == 4


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
    with pytest.raises(verifier.ArtifactError):
        verifier._validate_name("/absolute.txt", 1)

    archive_path = tmp_path / "safe.whl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("zap_it-0.1.0.dist-info/METADATA", "Name: zap-it\n")
        archive.writestr("zap_it-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr("zap_it-0.1.0.dist-info/entry_points.txt", "[console_scripts]\n")
        archive.writestr("zap_it-0.1.0.dist-info/licenses/LICENSE", "MIT\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/README.md", "readme\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/THIRD_PARTY_NOTICES.md", "notice\n")
        archive.writestr("zap_it-0.1.0.data/data/share/zap-it/CHANGELOG.md", "change\n")
        archive.writestr("src/__init__.py", "__version__ = '0.1.0'\n")
        archive.writestr("src/runtime/live_service.py", "def main(): pass\n")
        archive.writestr("src/service/app.py", "def create_app(): pass\n")
    evidence = verifier.inspect_archive(str(archive_path))
    assert evidence["member_count"] == 10


def _write_sdist(path: Path, *, missing: set[str] | None = None) -> None:
    verifier = _load_script("verify_release_artifacts")
    missing = missing or set()
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(verifier.REQUIRED_SDIST_NAMES - missing):
            payload = b"safe generated release fixture\n"
            info = tarfile.TarInfo(f"zap-it-0.1.0/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sdist_requires_install_lock_baseline_release_and_service_inputs(tmp_path):
    verifier = _load_script("verify_release_artifacts")
    complete = tmp_path / "complete.tar.gz"
    _write_sdist(complete)
    assert verifier.inspect_archive(str(complete))["member_count"] == len(
        verifier.REQUIRED_SDIST_NAMES
    )
    missing = tmp_path / "missing.tar.gz"
    _write_sdist(missing, missing={"INSTALL.md"})
    with pytest.raises(verifier.ArtifactError, match="required release members"):
        verifier.inspect_archive(str(missing))


@pytest.mark.parametrize(
    "name",
    [
        "output/result.json",
        "unexpected/output/result.json",
        "modules/output/readme.txt",
        ".env",
        ".env.local",
        "service.env",
        "private/service.env.example",
        "config/secrets.yaml",
        "demos/goats/goats1.jpg",
        "weights/model.safetensors",
    ],
)
def test_artifact_verifier_rejects_release_denylist_members(name):
    verifier = _load_script("verify_release_artifacts")
    with pytest.raises(verifier.ArtifactError):
        verifier._validate_name(name, 1)


def test_artifact_verifier_allows_only_public_env_and_package_output():
    verifier = _load_script("verify_release_artifacts")
    verifier._validate_name("deploy/service.env.example", 1)
    verifier._validate_name("modules/output", 0)
    verifier._validate_name("modules/output/images.py", 1)


def test_artifact_verifier_rejects_oversize_symlink_and_hardlink(tmp_path):
    verifier = _load_script("verify_release_artifacts")
    with pytest.raises(verifier.ArtifactError):
        verifier._validate_name("safe.txt", verifier.MAX_MEMBER_BYTES + 1)

    symlink = tmp_path / "symlink.whl"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("link.txt")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")
    with pytest.raises(verifier.ArtifactError, match="symlink"):
        verifier.inspect_archive(str(symlink))

    hardlink = tmp_path / "hardlink.tar.gz"
    with tarfile.open(hardlink, "w:gz") as archive:
        info = tarfile.TarInfo("link.txt")
        info.type = tarfile.LNKTYPE
        info.linkname = "target.txt"
        archive.addfile(info)
    with pytest.raises(verifier.ArtifactError, match="non-regular"):
        verifier.inspect_archive(str(hardlink))


def test_source_version_and_public_docs_reference_real_paths():
    import src

    assert src.__version__ == "0.1.0"
    assert (REPO_ROOT / "INSTALL.md").is_file()
    assert (REPO_ROOT / "requirements-gpu-cu124.lock").is_file()
    assert "configs/example.yaml" not in (REPO_ROOT / "README.md").read_text()
    assert "--config configs/goats.yaml" not in (REPO_ROOT / "INSTALL.md").read_text()
    for document in [
        REPO_ROOT / "README.md",
        REPO_ROOT / "INSTALL.md",
        *sorted((REPO_ROOT / "docs").glob("*.md")),
    ]:
        body = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)", body):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (document.parent / target).exists(), f"{document}: {target} missing"


def test_tracked_secret_baseline_passes_explained_finding_and_rejects_new_one(tmp_path):
    scanner = _load_script("scan_release_artifacts")
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("synthetic value", encoding="utf-8")
    hash_field = "hash" + "ed_secret"
    baseline = {
        "version": "1.5.0",
        "results": {"fixture.txt": [{"type": "Keyword", hash_field: "explained-hash"}]},
    }

    def explained(root):
        return {str(root / "fixture.txt"): [{"type": "Keyword", hash_field: "explained-hash"}]}

    scanner._tracked_paths = lambda root: [fixture]
    scanner._scan_findings = lambda root, paths=None: explained(root)
    assert scanner.scan_tracked_tree(tmp_path, baseline) == 1

    def unexpected(root):
        return {str(root / "fixture.txt"): [{"type": "Keyword", hash_field: "new-hash"}]}

    scanner._scan_findings = lambda root, paths=None: unexpected(root)
    with pytest.raises(RuntimeError, match="baseline mismatch"):
        scanner.scan_tracked_tree(tmp_path, baseline)

    with pytest.raises(RuntimeError, match="malformed"):
        scanner._baseline_findings({"results": []})

    def failed_scan(root, paths=None):
        raise RuntimeError("scanner failed")

    scanner._scan_findings = failed_scan
    with pytest.raises(RuntimeError, match="scanner failed"):
        scanner.scan_tracked_tree(tmp_path, baseline)


def test_two_image_academic_harness_uses_distinct_crops_in_exact_aba_order(
    tmp_path, monkeypatch, capsys
):
    goats = _load_script("smoke_local_goats")
    image_a = tmp_path / "image-a.png"
    image_b = tmp_path / "image-b.png"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image_a)
    Image.new("RGB", (16, 12), (30, 20, 10)).save(image_b)
    config = tmp_path / "goats2.yaml"
    config.write_text("alpha: 0.6\n", encoding="utf-8")
    calls = []

    def fake_level_case(
        host, port, *, verbosity, response_format, fixture_png, config_bytes, api_key
    ):
        calls.append((hashlib.sha256(fixture_png).hexdigest(), verbosity, response_format))
        return {
            "case": f"l{verbosity}-{response_format}",
            "passed": True,
            "http_status": 200,
            "latency_ms": 1.0,
            "response_bytes": 2,
            "objects": 1,
        }

    monkeypatch.setattr(goats, "run_level_case", fake_level_case)
    assert (
        goats.main(
            [
                "--port",
                "39000",
                "--image-a",
                str(image_a),
                "--image-b",
                str(image_b),
                "--config",
                str(config),
                "--fixture-root",
                str(tmp_path),
                "--tmp-root",
                str(tmp_path / "shm"),
            ]
        )
        == 0
    )
    crop_a = goats.central_crop(image_a.read_bytes())[0]
    crop_b = goats.central_crop(image_b.read_bytes())[0]
    expected = [hashlib.sha256(item).hexdigest() for item in (crop_a, crop_a, crop_a)]
    expected = [expected[0]] * 3 + [hashlib.sha256(crop_b).hexdigest()] * 3 + [expected[0]] * 3
    assert [item[0] for item in calls] == expected
    assert [item[1:] for item in calls] == [(2, "json"), (3, "json"), (3, "zip")] * 3
    output = json.loads(capsys.readouterr().out)
    assert output["fixture_aliases"] == ["a1", "b", "a2"]
    assert output["image_a"]["image_sha256"] != output["image_b"]["image_sha256"]
    assert output["zero_persistence"] is True
