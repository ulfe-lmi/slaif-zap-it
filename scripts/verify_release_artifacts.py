#!/usr/bin/env python3
"""Verify wheel/sdist member safety and emit deterministic digest evidence.

The verifier reads archive members in memory and never extracts an untrusted
archive. It is intentionally conservative: release packages contain Python,
documentation, generated-fixture tests and operator templates, not media,
weights, caches, outputs, credentials or local academic inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
import zipfile
from pathlib import PurePosixPath
from typing import Iterable, Mapping

MAX_MEMBER_BYTES = 2 * 1024 * 1024
FORBIDDEN_BASENAMES = frozenset({"goats.yaml", "goats2.yaml", "goats1.jpg", "goats2.jpg"})
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".pt",
        ".pth",
        ".safetensors",
        ".onnx",
        ".bin",
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "last_results",
        "outputs",
        "cache",
        "caches",
        "demos",
        "assets",
        "oap",
    }
)
REQUIRED_TEXT_NAMES = frozenset({"LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "CHANGELOG.md"})
REQUIRED_SDIST_NAMES = frozenset(
    {
        "INSTALL.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "requirements-gpu-cu124.lock",
        ".secrets.baseline",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "README.md",
        "CHANGELOG.md",
        "RELEASE_NOTES.md",
        "SECURITY.md",
        "TESTING.md",
        "deploy/service.env.example",
        "deploy/zap-it-local.service",
        "scripts/verify_release_artifacts.py",
        "scripts/scan_release_artifacts.py",
        "scripts/check_documentation.py",
        "scripts/smoke_installed_package.py",
        "scripts/smoke_model_control.py",
        "src/runtime/live_service.py",
        "src/service/app.py",
        "src/service/model_control.py",
        "src/service/envelope.py",
        "src/service/fake_engine.py",
        "src/service/settings.py",
        "src/service/multipart.py",
        "src/service/yaml_input.py",
        "src/service/schemas.py",
        "configs/glasswool.yaml",
        "configs/icecream.yaml",
        "configs/soccer.yaml",
        "configs/tomato.yaml",
        "docs/README.md",
    }
)
REQUIRED_WHEEL_MODULES = frozenset(
    {
        "src/__init__.py",
        "src/runtime/live_service.py",
        "src/service/app.py",
    }
)
PUBLIC_ENV_MEMBER = "deploy/service.env.example"
PRIVATE_ENV_BASENAMES = frozenset(
    {
        "env",
        ".env",
        "environment",
        "credentials",
        "secrets",
        "private",
        "service.env",
        "credentials.yaml",
        "credentials.yml",
        "credentials.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.json",
        "private.yaml",
        "private.yml",
        "private.json",
    }
)


class ArtifactError(ValueError):
    """Raised when a distribution violates the release content contract."""


def _validate_name(name: str, size: int) -> None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if normalized.startswith("/") or path.is_absolute() or "\x00" in normalized:
        raise ArtifactError(f"unsafe absolute archive member: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError(f"unsafe archive member path: {name!r}")
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & FORBIDDEN_PARTS:
        raise ArtifactError(f"forbidden archive member directory: {name!r}")
    if "output" in lower_parts:
        relative_parts = path.parts
        if relative_parts and relative_parts[0].lower().startswith(("zap-it-", "zap_it-")):
            relative_parts = relative_parts[1:]
        is_package_output = (
            len(relative_parts) >= 2
            and tuple(part.lower() for part in relative_parts[:2]) == ("modules", "output")
            and (len(relative_parts) == 2 or path.suffix.lower() == ".py")
        )
        if not is_package_output:
            raise ArtifactError(f"forbidden archive output member: {name!r}")
    basename = path.name.lower()
    if basename in FORBIDDEN_BASENAMES:
        raise ArtifactError(f"forbidden local fixture member: {name!r}")
    relative_normalized = normalized
    if path.parts and path.parts[0].lower().startswith(("zap-it-", "zap_it-")):
        relative_normalized = str(PurePosixPath(*path.parts[1:]))
    if basename.endswith(".env.example") and relative_normalized != PUBLIC_ENV_MEMBER:
        raise ArtifactError(f"unexpected environment template member: {name!r}")
    is_public_env = normalized == PUBLIC_ENV_MEMBER
    is_private_env = (
        basename in PRIVATE_ENV_BASENAMES
        or basename.startswith(".env.")
        or basename.endswith(".env")
        or basename.startswith("private.")
        or basename.startswith("secret.")
    )
    if is_private_env and not is_public_env:
        raise ArtifactError(f"forbidden private environment/config member: {name!r}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ArtifactError(f"forbidden payload member: {name!r}")
    if size > MAX_MEMBER_BYTES:
        raise ArtifactError(f"archive member exceeds size budget: {name!r}")


def _member_digest(name: str, payload: bytes) -> dict[str, object]:
    _validate_name(name, len(payload))
    return {"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest_digest(members: Iterable[Mapping[str, object]]) -> str:
    canonical = json.dumps(list(members), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _sdist_relative_name(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) > 1 and parts[0].startswith(("zap-it-", "zap_it-")):
        return str(PurePosixPath(*parts[1:]))
    return name


def inspect_zip(path: str) -> tuple[list[dict[str, object]], set[str]]:
    members: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                _validate_name(info.filename.rstrip("/"), 0)
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(mode):
                raise ArtifactError(f"symlink archive member: {info.filename!r}")
            members.append(_member_digest(info.filename, archive.read(info)))
    return members, {str(item["name"]) for item in members}


def inspect_tar(path: str) -> tuple[list[dict[str, object]], set[str]]:
    members: list[dict[str, object]] = []
    with tarfile.open(path, mode="r:*") as archive:
        for info in archive.getmembers():
            _validate_name(info.name.rstrip("/"), int(info.size))
            if info.issym() or info.islnk() or not (info.isfile() or info.isdir()):
                raise ArtifactError(f"non-regular archive member: {info.name!r}")
            if info.isdir():
                continue
            extracted = archive.extractfile(info)
            if extracted is None:
                raise ArtifactError(f"unreadable archive member: {info.name!r}")
            members.append(_member_digest(info.name, extracted.read()))
    return members, {str(item["name"]) for item in members}


def inspect_archive(path: str) -> dict[str, object]:
    """Return safe digest evidence or raise ArtifactError."""
    if path.endswith(".whl"):
        members, names = inspect_zip(path)
    elif path.endswith((".tar.gz", ".tar")):
        members, names = inspect_tar(path)
    else:
        raise ArtifactError(f"unsupported distribution extension: {path!r}")
    if path.endswith(".whl"):
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel = [name for name in names if name.endswith(".dist-info/WHEEL")]
        entrypoints = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        wheel_license = [
            name
            for name in names
            if name.endswith(".dist-info/LICENSE") or name.endswith(".dist-info/licenses/LICENSE")
        ]
        wheel_docs = {
            name.rsplit("/", 1)[-1] for name in names if ".data/data/share/zap-it/" in name
        }
        if not metadata or not wheel or not entrypoints or not wheel_license:
            raise ArtifactError("wheel metadata or console entrypoint is missing")
        missing_modules = sorted(REQUIRED_WHEEL_MODULES - names)
        if missing_modules:
            raise ArtifactError(f"wheel installed modules missing: {missing_modules}")
        if not {"README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"} <= wheel_docs:
            raise ArtifactError("wheel public notices/data files are missing")
    if path.endswith(".tar.gz"):
        relative_names = {_sdist_relative_name(name) for name in names}
        missing = sorted(REQUIRED_SDIST_NAMES - relative_names)
        if missing:
            raise ArtifactError(f"sdist required release members missing: {missing}")
    with open(path, "rb") as handle:
        raw = handle.read()
    return {
        "path": path,
        "size": os.path.getsize(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "member_count": len(members),
        "member_manifest_sha256": _manifest_digest(members),
        "members": members,
    }


def compare_wheel_members(left: str, right: str) -> None:
    """Require two wheels to contain identical member bytes and names."""
    left_evidence = inspect_archive(left)
    right_evidence = inspect_archive(right)
    left_members = sorted(
        (str(item["name"]), int(item["size"]), str(item["sha256"]))
        for item in left_evidence["members"]
    )
    right_members = sorted(
        (str(item["name"]), int(item["size"]), str(item["sha256"]))
        for item in right_evidence["members"]
    )
    if left_members != right_members:
        raise ArtifactError("direct and sdist-built wheel members differ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", help="wheel and/or sdist paths")
    parser.add_argument("--compare-wheels", nargs=2, metavar=("DIRECT", "FROM_SDIST"))
    parser.add_argument("--json", dest="json_path", help="write full digest evidence to a file")
    args = parser.parse_args(argv)
    if not args.archives and not args.compare_wheels:
        parser.error("provide archives or --compare-wheels")
    try:
        if args.compare_wheels:
            compare_wheel_members(*args.compare_wheels)
        evidence = [inspect_archive(path) for path in args.archives]
    except (ArtifactError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.write("\n")
    public = [
        {
            key: item[key]
            for key in ("path", "size", "sha256", "member_count", "member_manifest_sha256")
        }
        for item in evidence
    ]
    print(json.dumps(public, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
