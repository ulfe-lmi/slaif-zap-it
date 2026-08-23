"""Pinned model identities used by the qualified operator runtime.

Client configuration never selects any of these values.  The revision fields
are immutable provenance anchors captured from the approved Hugging Face model
repositories during Objective 003 qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = ["APPROVED_MODEL_SPECS", "ModelSpec"]


@dataclass(frozen=True)
class ModelSpec:
    """One approved model repository and its pinned revision metadata."""

    model_id: str
    revision: str
    source_url: str
    license_name: str
    license_url: str
    trust_remote_code: bool = False
    approx_weight_bytes: int | None = None


APPROVED_MODEL_SPECS: Mapping[str, ModelSpec] = {
    "sam2": ModelSpec(
        model_id="facebook/sam2-hiera-large",
        revision="e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251",
        source_url="https://huggingface.co/facebook/sam2-hiera-large",
        license_name="Apache-2.0",
        license_url="https://huggingface.co/facebook/sam2-hiera-large/blob/main/README.md",
        approx_weight_bytes=897_952_466,
    ),
    "clip": ModelSpec(
        model_id="openai/clip-vit-base-patch32",
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        source_url="https://huggingface.co/openai/clip-vit-base-patch32",
        license_name="OpenAI CLIP research model-card terms; no SPDX license field",
        license_url="https://huggingface.co/openai/clip-vit-base-patch32/blob/main/README.md",
        approx_weight_bytes=605_247_071,
    ),
    "blip3": ModelSpec(
        model_id="Salesforce/xgen-mm-phi3-mini-instruct-r-v1",
        revision="1d91d356d3b6fbc141140edf490b39890417af44",
        source_url="https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1",
        license_name="CC-BY-NC-4.0",
        license_url=(
            "https://huggingface.co/Salesforce/"
            "xgen-mm-phi3-mini-instruct-r-v1/blob/main/LICENSE.txt"
        ),
        trust_remote_code=True,
        approx_weight_bytes=18_357_535_724,
    ),
}
