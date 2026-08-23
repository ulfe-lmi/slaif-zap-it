#!/usr/bin/env python3
"""Validate and atomically append one strategic-authored CRITICAL.md entry."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path

ID_RE = re.compile(r"^CRIT-[0-9]{4}$")
REQUIRED_TEXT = (
    "- Status: OPEN — HUMAN ADJUDICATION REQUIRED",
    "- Human adjudication required before:",
    "- Threshold attestation: ALL FIVE CRITICAL-ENTRY CONDITIONS SATISFIED",
    "### Dilemma",
    "### Provisional decision",
    "### Why this decision",
    "### Strongest case that this decision is wrong",
    "### Alternatives considered",
    "### Assumptions",
    "### Failure mode and blast radius",
    "### Mitigations and evidence",
    "### Reversibility and rollback",
    "### Exact question for the human adjudicator",
    "### Autonomous follow-up allowed before adjudication",
)
FORBIDDEN = ("DRAFT UNTIL", "VERIFY:", "TODO:", "TBD")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not ID_RE.fullmatch(args.id):
        raise SystemExit(f"Invalid critical ID: {args.id}")

    repo = args.repo_root.resolve()
    source = args.source.resolve()
    register = repo / "CRITICAL.md"
    if not (repo / ".git").is_dir():
        raise SystemExit(f"Not a Git checkout: {repo}")
    if not register.is_file():
        raise SystemExit(f"Critical register missing: {register}")
    if not source.is_file():
        raise SystemExit(f"Entry source missing: {source}")

    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("Critical entry is empty")
    heading_re = re.compile(rf"^## {re.escape(args.id)}\s+[—-]\s+\S")
    if not heading_re.search(text.lstrip()):
        raise SystemExit(f"Entry must begin with a level-2 heading for {args.id}")
    for required in REQUIRED_TEXT:
        if required not in text:
            raise SystemExit(f"Missing required critical-entry field: {required}")
    for marker in FORBIDDEN:
        if marker in text:
            raise SystemExit(f"Critical entry still contains unfinished marker: {marker}")
    if re.search(r"^#(?!#)", text, re.MULTILINE):
        raise SystemExit("Critical entry may not contain a level-1 heading")
    if "## HUMAN ADJUDICATION" in text:
        raise SystemExit("Agents may not append a human adjudication section")

    normalized = text.strip().encode("utf-8") + b"\n"
    lock_path = repo / ".git" / "oap-critical.lock"
    lock_path.touch(mode=0o600, exist_ok=True)

    with lock_path.open("r+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = register.read_bytes()
        if re.search(rb"^## " + args.id.encode("ascii") + rb"\b", current, re.MULTILINE):
            raise SystemExit(f"Critical ID already exists: {args.id}")
        if args.dry_run:
            print(f"would append {source} to {register} as {args.id}")
            return 0

        separator = b"\n" if current.endswith(b"\n\n") else b"\n\n"
        with register.open("ab", buffering=0) as stream:
            stream.write(separator + normalized)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(register.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)

    print(f"appended {args.id} to {register}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
