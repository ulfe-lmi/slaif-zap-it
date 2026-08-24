#!/usr/bin/env python3
"""Offline documentation integrity and current-truth checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]

ROOT_DOCUMENTS = (
    "README.md",
    "INSTALL.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "SECURITY.md",
    "TESTING.md",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
)

REQUIRED_DOCUMENTS = ROOT_DOCUMENTS + (
    "docs/README.md",
    "docs/API.md",
    "docs/CONFIG.md",
    "docs/ALGORITHMS.md",
    "docs/CORE.md",
    "docs/runtime.md",
    "docs/RUNBOOK.md",
    "docs/SERVICE-DATASHEET.md",
    "docs/OUTPUT-PARITY.md",
    "docs/GATEWAY-INTEGRATION.md",
    "docs/RELEASE-GATE-INVENTORY.md",
    "docs/history/README.md",
)

CURRENT_DOCUMENTS = ROOT_DOCUMENTS + tuple(
    str(path.relative_to(REPO_ROOT)) for path in sorted((REPO_ROOT / "docs").glob("*.md"))
)

FORBIDDEN_CURRENT_PATTERNS = {
    "obsolete residency strategy": re.compile(r"sam2_clip_resident_blip3_rejected"),
    "obsolete open CRIT-0001 claim": re.compile(
        r"CRIT-0001.{0,80}(?:OPEN|BLOCKING)|(?:OPEN|BLOCKING).{0,80}CRIT-0001",
        re.IGNORECASE | re.DOTALL,
    ),
    "obsolete goat rights claim": re.compile(r"NONREDISTRIBUTABLE", re.IGNORECASE),
    "deleted algorithm document": re.compile(r"ALGORITHMS-DETAILED\.md"),
    "deleted runtime document": re.compile(r"GPU-RUNTIME\.md"),
    "generated documentation dump": re.compile(r"everything\.txt"),
    "obsolete all-resident qualification claim": re.compile(
        r"all-resident qualification remains separate", re.IGNORECASE
    ),
    "obsolete pending all-resident qualification claim": re.compile(
        r"pending a separate live qualification", re.IGNORECASE
    ),
    "obsolete low-card-only BLIP3 claim": re.compile(
        r"only live-qualified BLIP3 residency", re.IGNORECASE
    ),
    "obsolete fake-tested all-resident claim": re.compile(
        r"implemented and CPU/fake-tested.{0,100}not\s+(?:claimed as\s+)?live-qualified",
        re.IGNORECASE,
    ),
}

MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
CONFIG_REFERENCE = re.compile(r"(?<![\w/])(configs/[A-Za-z0-9_.-]+\.ya?ml)\b")

SLAIF_BLOCK = """<div style="text-align: center;">
  <a href="https://www.slaif.si">
    <img src="https://slaif.si/img/logos/SLAIF_logo_ANG_barve.svg" width="400" height="400" alt="SLAIF">
  </a>
</div>"""


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#", "data:"))


def _target_path(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if _is_external(target):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return None
    return (document.parent / target).resolve()


def check_repository(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_DOCUMENTS:
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")

    all_documents = [root / name for name in ROOT_DOCUMENTS if (root / name).is_file()]
    all_documents.extend(sorted((root / "docs").rglob("*.md")))
    all_documents.append(root / "oap" / "README.md")

    for document in all_documents:
        if not document.is_file():
            continue
        relative = document.relative_to(root)
        body = document.read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```|~~~.*?~~~", "", body, flags=re.DOTALL)
        headings = re.findall(r"^# (?!#).+", prose, flags=re.MULTILINE)
        if len(headings) != 1:
            errors.append(f"{relative}: expected exactly one H1, found {len(headings)}")
        for raw_target in MARKDOWN_LINK.findall(body) + HTML_LINK.findall(body):
            resolved = _target_path(document, raw_target)
            if resolved is not None and not resolved.exists():
                errors.append(f"{relative}: missing local link target {raw_target!r}")

    for relative in CURRENT_DOCUMENTS:
        document = root / relative
        if not document.is_file():
            continue
        body = document.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_CURRENT_PATTERNS.items():
            if pattern.search(body):
                errors.append(f"{relative}: {label}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if not readme.startswith(SLAIF_BLOCK):
        errors.append("README.md: SLAIF logo block does not match the project convention")
    for workflow in ("ci.yml", "codeql.yml"):
        if f"actions/workflows/{workflow}/badge.svg" not in readme:
            errors.append(f"README.md: missing {workflow} status badge")

    for relative in ("README.md", "INSTALL.md", "docs/ALGORITHMS.md"):
        body = (root / relative).read_text(encoding="utf-8")
        for config_path in CONFIG_REFERENCE.findall(body):
            if not (root / config_path).is_file():
                errors.append(
                    f"{relative}: referenced config is not tracked/present: {config_path}"
                )

    return sorted(set(errors))


def main() -> int:
    errors = check_repository()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"documentation check passed ({len(CURRENT_DOCUMENTS)} current documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
