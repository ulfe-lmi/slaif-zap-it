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


def test_audited_obsolete_gpu_governance_claims_are_rejected():
    checker = _checker()
    samples = (
        "local service -> physical GPU 1 only",
        "Physical GPU0 is never used",
        "Never use physical GPU0, even when it is idle",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1",
    )
    for sample in samples:
        assert any(
            pattern.search(sample)
            for name, pattern in checker.FORBIDDEN_CURRENT_PATTERNS.items()
            if name.startswith("obsolete ")
        )


def test_operator_assigned_gpu_wording_is_allowed():
    checker = _checker()
    samples = (
        "The active order names an explicit operator-assigned physical index and UUID.",
        "CUDA_VISIBLE_DEVICES=<assigned-physical-index>",
        "Every unassigned device and unrelated workload remains protected.",
        "Historical physical GPU1 evidence is retained as host-specific context.",
    )
    for sample in samples:
        assert not any(
            pattern.search(sample) for pattern in checker.FORBIDDEN_CURRENT_PATTERNS.values()
        )
