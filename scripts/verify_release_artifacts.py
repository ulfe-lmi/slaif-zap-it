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
    basename = path.name.lower()
    if basename in FORBIDDEN_BASENAMES:
        raise ArtifactError(f"forbidden local fixture member: {name!r}")
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
    required = {
        name
        for name in REQUIRED_TEXT_NAMES
        if any(PurePosixPath(member).name == name for member in names)
    }
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
        if not {"README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"} <= wheel_docs:
            raise ArtifactError("wheel public notices/data files are missing")
    if required != REQUIRED_TEXT_NAMES and path.endswith(".tar.gz"):
        missing = sorted(REQUIRED_TEXT_NAMES - required)
        raise ArtifactError(f"sdist required notices missing: {missing}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", help="wheel and/or sdist paths")
    parser.add_argument("--json", dest="json_path", help="write full digest evidence to a file")
    args = parser.parse_args(argv)
    try:
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
