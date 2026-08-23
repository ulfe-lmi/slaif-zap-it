#!/usr/bin/env python3
"""Read-only OAP/FIFO/Git/critical-register consistency snapshot."""

from __future__ import annotations

import argparse
import collections
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"^[0-9]{3}-[a-z]$")
CRITICAL_ID_RE = re.compile(r"^## (CRIT-[0-9]{4})\b", re.MULTILINE)
HUMAN_ADJUDICATION_RE = re.compile(
    r"^## HUMAN ADJUDICATION — (CRIT-[0-9]{4}) — [^\n]+\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)
HUMAN_DECISION_RE = re.compile(
    r"^- Decision: (ACCEPTED|REJECTED|CHANGE REQUIRED|DEFERRED)\s*$", re.MULTILINE
)


def run_git(repo: Path, *args: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.rstrip("\n"),
        "stderr": result.stderr.rstrip("\n"),
    }


def fifo_state(path: Path) -> str:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return "MISSING"
    return "FIFO" if stat.S_ISFIFO(mode) else "NOT_FIFO"


def critical_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "state": "MISSING",
            "entry_ids": [],
            "duplicate_ids": [],
            "latest_human_disposition": {},
            "accepted_ids": [],
            "open_ids": [],
        }
    text = path.read_text(encoding="utf-8")
    entry_ids = CRITICAL_ID_RE.findall(text)
    counts = collections.Counter(entry_ids)
    duplicate_ids = sorted(identifier for identifier, count in counts.items() if count > 1)
    latest_disposition: dict[str, str] = {}
    for identifier, body in HUMAN_ADJUDICATION_RE.findall(text):
        decision = HUMAN_DECISION_RE.search(body)
        latest_disposition[identifier] = decision.group(1) if decision else "MALFORMED"
    accepted_ids = sorted(
        identifier for identifier, decision in latest_disposition.items() if decision == "ACCEPTED"
    )
    open_ids = sorted(set(entry_ids) - set(accepted_ids))
    return {
        "path": str(path),
        "state": "FILE",
        "entry_ids": entry_ids,
        "duplicate_ids": duplicate_ids,
        "latest_human_disposition": dict(sorted(latest_disposition.items())),
        "accepted_ids": accepted_ids,
        "open_ids": open_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--strategic-home", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    strategic = args.strategic_home.resolve()
    active_path = repo / "oap" / "active"
    active = active_path.read_text("ascii").strip() if active_path.exists() else None
    valid_active = bool(active and ID_RE.fullmatch(active))
    orders = (
        sorted(str(path) for path in (repo / "oap" / "orders").glob(f"{active}-*.md"))
        if valid_active
        else []
    )
    reports = (
        sorted(str(path) for path in (repo / "oap" / "reports").glob(f"{active}-*.md"))
        if valid_active
        else []
    )
    critical = critical_state(repo / "CRITICAL.md")

    snapshot = {
        "repo_root": str(repo),
        "strategic_home": str(strategic),
        "control_fifo": fifo_state(strategic / "control.fifo"),
        "response_fifo": fifo_state(strategic / "response.fifo"),
        "active": active,
        "active_valid": valid_active,
        "matching_orders": orders,
        "matching_reports": reports,
        "order_count_valid": (len(orders) == 1) if valid_active else None,
        "report_count": len(reports),
        "critical_register": critical,
        "git_branch": run_git(repo, "branch", "--show-current"),
        "git_head": run_git(repo, "rev-parse", "HEAD"),
        "git_status": run_git(repo, "status", "--short"),
        "git_remote": run_git(repo, "remote", "-v"),
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))

    structural_ok = (
        snapshot["control_fifo"] == "FIFO"
        and snapshot["response_fifo"] == "FIFO"
        and critical["state"] == "FILE"
        and not critical["duplicate_ids"]
        and (not valid_active or len(orders) == 1)
    )
    return 0 if structural_ok else 1


if __name__ == "__main__":
    sys.exit(main())
