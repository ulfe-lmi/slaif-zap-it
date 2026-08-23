#!/usr/bin/env python3
"""Atomically publish one finalized strategic order and active pointer."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

ID_RE = re.compile(r"^[0-9]{3}-[a-z]$")
CRITICAL_ID_RE = re.compile(r"^CRIT-[0-9]{4}$")
FORBIDDEN = ("DRAFT UNTIL", "VERIFY:")
DECISION_RE = re.compile(
    r"^- Decision:\s*`?(NONE|APPEND (CRIT-[0-9]{4}))`?\s*$", re.MULTILINE
)


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def validate_adjudication_section(text: str) -> None:
    if "## Deferred human adjudication" not in text:
        raise SystemExit("Order lacks required Deferred human adjudication section")
    matches = DECISION_RE.findall(text)
    if len(matches) != 1:
        raise SystemExit(
            "Order must contain exactly one '- Decision: NONE' or "
            "'- Decision: APPEND CRIT-NNNN' line"
        )
    decision, critical_id = matches[0]
    if decision == "NONE":
        return
    if not CRITICAL_ID_RE.fullmatch(critical_id):
        raise SystemExit(f"Invalid critical entry ID: {critical_id}")
    required = (
        f"## {critical_id}",
        "- Status: OPEN — HUMAN ADJUDICATION REQUIRED",
        "- Threshold attestation: ALL FIVE CRITICAL-ENTRY CONDITIONS SATISFIED",
        "### Strongest case that this decision is wrong",
        "### Exact question for the human adjudicator",
        "### Autonomous follow-up allowed before adjudication",
    )
    for marker in required:
        if marker not in text:
            raise SystemExit(f"APPEND decision lacks exact critical-entry content: {marker}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not ID_RE.fullmatch(args.id):
        raise SystemExit(f"Invalid OAP ID: {args.id}")
    repo = args.repo_root.resolve()
    source = args.source.resolve()
    if not (repo / ".git").is_dir():
        raise SystemExit(f"Not Git checkout: {repo}")
    if not source.is_file() or not source.name.startswith(args.id + "-") or source.suffix != ".md":
        raise SystemExit("Invalid order source/name")

    data = source.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("Order must be UTF-8") from exc
    if not text.strip():
        raise SystemExit("Order empty")
    for marker in FORBIDDEN:
        if marker in text:
            raise SystemExit(f"Order still contains non-final marker: {marker}")
    if args.id not in text[:500]:
        raise SystemExit("Order ID absent near start")
    validate_adjudication_section(text)

    orders = repo / "oap" / "orders"
    target = orders / source.name
    active = repo / "oap" / "active"
    existing = sorted(orders.glob(f"{args.id}-*.md"))
    if existing and (
        existing != [target] or not target.is_file() or target.read_bytes() != data
    ):
        raise SystemExit(f"Conflicting order for {args.id}: {existing}")

    if args.dry_run:
        print(f"would publish {source} -> {target}")
        print(f"would set {active} -> {args.id}")
        return 0
    if not target.exists():
        atomic_write(target, data)
    atomic_write(active, (args.id + "\n").encode("ascii"))
    print(target)
    print(active)
    return 0


if __name__ == "__main__":
    sys.exit(main())
