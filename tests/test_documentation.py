from __future__ import annotations

import importlib.util
import re
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


def test_gateway_documentation_selects_only_the_future_responses_mapping():
    gateway = (REPO_ROOT / "docs" / "GATEWAY-INTEGRATION.md").read_text(encoding="utf-8")

    assert re.search(r"JSON\s+`POST /v1/responses`", gateway)
    assert re.search(r"fixed model\s+`zap-it-1`", gateway)
    assert "input_image" in gateway
    assert "input_file" in gateway
    assert 'tools: [{"type": "image_generation"}]' in gateway
    assert "image_generation_call.result" in gateway
    assert re.search(
        r"must not route public or general requests through\s+the native\s+`/v1/completions`",
        gateway,
    )

    obsolete_backend_mapping = re.compile(
        r"\b(?:backend|gateway)\s+(?:request|call|mapping)\b[^.\n]{0,160}"
        r"(?:multipart|/v1/completions|verbosity|response_format|JSON/ZIP)",
        re.IGNORECASE,
    )
    assert not obsolete_backend_mapping.search(gateway)
    obsolete_artifact_mapping = re.compile(
        r"\b(?:backend|gateway)\b[^.\n]{0,160}"
        r"\b(?:map|forward|return|emit|proxy)\b[^.\n]{0,160}"
        r"(?:JSON/ZIP|JSON or ZIP|debug artifacts)",
        re.IGNORECASE,
    )
    assert not obsolete_artifact_mapping.search(gateway)


def test_api_documentation_keeps_both_inference_surfaces_and_auth_boundary():
    api = (REPO_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    authentication = api.split("## Authentication", 1)[1].split("## Errors", 1)[0]

    opening = api.split("## Endpoints", 1)[0]
    assert "/v1/completions" in opening
    assert "/v1/responses" in opening
    assert "native/private" in opening
    assert re.search(r"not the general-public SLAIF\s+surface", opening)
    assert "future gateway/public compatibility surface" in opening
    assert "only inference contract" not in api.lower()
    assert "do not implement KServe V2 tensor inference" in api

    assert re.search(
        r"SLAIF_ZAP_IT_API_KEY.*?/v1/completions.*?/v1/responses",
        authentication,
        re.IGNORECASE | re.DOTALL,
    )
    assert re.search(
        r"private_lan.*?/v1/completions.*?/v1/responses",
        authentication,
        re.IGNORECASE | re.DOTALL,
    )
