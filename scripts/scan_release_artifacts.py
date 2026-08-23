#!/usr/bin/env python3
"""Run the pinned secret scanner over safely unpacked wheel/sdist members."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from verify_release_artifacts import inspect_archive


def _relative_name(name: str, archive_name: str) -> str:
    parts = PurePosixPath(name).parts
    if archive_name.endswith(".tar.gz") and len(parts) > 1:
        return str(PurePosixPath(*parts[1:]))
    return name


def _scan(root: Path, baseline: dict[str, object]) -> int:
    scanner = shutil.which("detect-secrets") or str(
        Path(sys.executable).with_name("detect-secrets")
    )
    command = [scanner, "scan", "--all-files", "--no-verify", str(root)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    findings = json.loads(completed.stdout).get("results", {})
    allowed = {
        (filename, item.get("type"), item.get("hashed_secret"))
        for filename, entries in (baseline.get("results") or {}).items()
        for item in entries
    }
    unexpected = []
    for filename, entries in findings.items():
        relative = str(Path(filename).relative_to(root))
        for item in entries:
            key = (relative, item.get("type"), item.get("hashed_secret"))
            if key not in allowed:
                unexpected.append(key)
    if unexpected:
        raise RuntimeError(f"unexpected secret findings in unpacked artifact: {len(unexpected)}")
    return sum(len(entries) for entries in findings.values())


def scan_archive(path: str, baseline: dict[str, object]) -> int:
    inspect_archive(path)
    with tempfile.TemporaryDirectory(prefix="zap-it-release-scan-") as temporary:
        root = Path(temporary)
        # Populate bytes without ever extracting symlinks. inspect_archive
        # already validated every member before this pass.
        if path.endswith(".whl"):
            import zipfile

            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        name = _relative_name(info.filename, path)
                        target = root / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(info))
        else:
            import tarfile

            with tarfile.open(path, "r:*") as archive:
                for info in archive.getmembers():
                    if info.isfile():
                        extracted = archive.extractfile(info)
                        assert extracted is not None
                        target = root / _relative_name(info.name, path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(extracted.read())
        return _scan(root, baseline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--baseline", default=".secrets.baseline")
    args = parser.parse_args(argv)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    counts = [scan_archive(path, baseline) for path in args.archives]
    print(json.dumps({"archives": len(counts), "known_findings": sum(counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
