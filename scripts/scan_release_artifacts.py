#!/usr/bin/env python3
"""Run the pinned secret scanner over the tracked tree and release artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from verify_release_artifacts import inspect_archive


def _relative_name(name: str, archive_name: str) -> str:
    parts = PurePosixPath(name).parts
    if archive_name.endswith(".tar.gz") and len(parts) > 1:
        return str(PurePosixPath(*parts[1:]))
    return name


def _scanner_command() -> str:
    return shutil.which("detect-secrets") or str(Path(sys.executable).with_name("detect-secrets"))


def _baseline_findings(baseline: Mapping[str, object]) -> set[tuple[str, str, str]]:
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("results"), Mapping):
        raise RuntimeError("malformed secret baseline results")
    findings: set[tuple[str, str, str]] = set()
    for filename, entries in baseline["results"].items():
        if not isinstance(filename, str) or not isinstance(entries, list):
            raise RuntimeError("malformed secret baseline entry")
        for item in entries:
            if not isinstance(item, Mapping):
                raise RuntimeError("malformed secret baseline finding")
            detector = item.get("type")
            hashed_secret = item.get("hashed_secret")
            if not all(isinstance(value, str) and value for value in (detector, hashed_secret)):
                raise RuntimeError("malformed secret baseline finding fields")
            findings.add((filename, detector, hashed_secret))
    return findings


def _scan_findings(root: Path, paths: Sequence[Path] | None = None) -> Mapping[str, object]:
    command = [_scanner_command(), "scan", "--no-verify"]
    command.extend(str(path) for path in (paths or [root]))
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError("detect-secrets scanner failed to start") from exc
    if completed.returncode != 0:
        raise RuntimeError("detect-secrets scanner failed")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("detect-secrets returned malformed JSON") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("results"), Mapping):
        raise RuntimeError("detect-secrets returned malformed results")
    return document["results"]


def _finding_set(root: Path, findings: Mapping[str, object]) -> set[tuple[str, str, str]]:
    normalized: set[tuple[str, str, str]] = set()
    for filename, entries in findings.items():
        if not isinstance(filename, str) or not isinstance(entries, list):
            raise RuntimeError("detect-secrets returned malformed findings")
        try:
            relative = str(Path(filename).resolve().relative_to(root.resolve()))
        except ValueError as exc:
            raise RuntimeError("detect-secrets returned a path outside the scan root") from exc
        for item in entries:
            if not isinstance(item, Mapping):
                raise RuntimeError("detect-secrets returned malformed finding")
            detector = item.get("type")
            hashed_secret = item.get("hashed_secret")
            if not all(isinstance(value, str) and value for value in (detector, hashed_secret)):
                raise RuntimeError("detect-secrets returned malformed finding fields")
            normalized.add((relative, detector, hashed_secret))
    return normalized


def _scan(root: Path, baseline: dict[str, object]) -> int:
    current = _finding_set(root, _scan_findings(root))
    allowed = _baseline_findings(baseline)
    unexpected = current - allowed
    if unexpected:
        raise RuntimeError(f"unexpected secret findings in unpacked artifact: {len(unexpected)}")
    return len(current)


def _tracked_paths(root: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError("could not enumerate tracked files") from exc
    if completed.returncode != 0:
        raise RuntimeError("could not enumerate tracked files")
    return [root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def scan_tracked_tree(root: Path, baseline: dict[str, object]) -> int:
    """Enforce exact baseline findings over the committed tracked tree only."""
    baseline_findings = _baseline_findings(baseline)
    scan_paths = [
        path
        for path in _tracked_paths(root)
        if path.relative_to(root).as_posix() != ".secrets.baseline"
    ]
    current = _finding_set(root, _scan_findings(root, scan_paths))
    added = current - baseline_findings
    removed = baseline_findings - current
    if added or removed:
        raise RuntimeError(
            f"tracked secret baseline mismatch: additions={len(added)} removals={len(removed)}"
        )
    return len(current)


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
    parser.add_argument("archives", nargs="*")
    parser.add_argument("--baseline", default=".secrets.baseline")
    parser.add_argument(
        "--tracked-tree",
        action="store_true",
        help="compare findings in the git tracked tree exactly with the baseline",
    )
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise RuntimeError("malformed secret baseline")
        tracked_count = scan_tracked_tree(Path.cwd(), baseline) if args.tracked_tree else None
        counts = [scan_archive(path, baseline) for path in args.archives]
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    if not args.tracked_tree and not counts:
        parser.error("provide archives or --tracked-tree")
    print(
        json.dumps(
            {
                "archives": len(counts),
                "known_findings": sum(counts),
                "tracked_findings": tracked_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
