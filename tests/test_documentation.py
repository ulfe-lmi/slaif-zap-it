from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = REPO_ROOT / "scripts" / "check_documentation.py"
    spec = importlib.util.spec_from_file_location("check_documentation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_documentation_is_coherent_and_stale_claims_are_absent():
    assert _checker().check_repository(REPO_ROOT) == []


def test_audited_obsolete_residency_claims_are_rejected():
    checker = _checker()
    samples = (
        "all-resident qualification remains separate",
        "pending a separate live qualification",
        "the only live-qualified BLIP3 residency",
        "implemented and CPU/fake-tested but is not live-qualified",
    )
    for sample in samples:
        assert any(
            pattern.search(sample) for pattern in checker.FORBIDDEN_CURRENT_PATTERNS.values()
        )
