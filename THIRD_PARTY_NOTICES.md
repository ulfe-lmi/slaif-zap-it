# Third-party notices and provenance

ZAP-IT 0.1.0 is an unpublished release candidate and is licensed under the MIT
License (see [LICENSE](LICENSE)). It builds on
the following third-party software and models. This file records pointers for
license review; it is not legal advice. The qualified runtime uses the pinned
repo-owned pip lock in `requirements-gpu-cu124.lock`; model weights remain
operator cache assets and are never committed or redistributed.

## Python packages

### CPU/dev toolchain (declared in `pyproject.toml`)

| Package | Role | License |
| --- | --- | --- |
| numpy 1.26.4 | array math | BSD-3-Clause |
| pillow 10.4.0 | image I/O | HPND (MIT-CMU) |
| pyyaml 6.0.2 | YAML config parsing | MIT |
| fastapi, python-multipart, uvicorn | service transport | MIT / Apache-2.0 / BSD-3-Clause |
| prometheus-client 0.21.1 | metrics | Apache-2.0 |
| pytest, pytest-cov, coverage, ruff, build, detect-secrets, openai 3.7.0 | CPU/release tooling and official Responses qualification client; SDK is dev-only | MIT / MIT / Apache-2.0 / MIT / MIT / Apache-2.0 / MIT |

### GPU/runtime stack (operator-managed lock)

| Package | Role | License pointer |
| --- | --- | --- |
| pytorch 2.5.1+cu124 / torchvision 0.20.1 / torchaudio 2.5.1 | tensor/GPU runtime | BSD-3-Clause |
| pytest 9.1.1 | opt-in live GPU marker tests | MIT |
| SAM-2 source commit `2b90b9f…` | segmentation implementation | Apache-2.0; optional CUDA extension disabled in this lock |
| transformers 4.41.1 | model loading/inference glue | Apache-2.0 |
| open-clip-torch 2.24.0, einops, einops-exts | XGen-MM remote-code dependencies | MIT / MIT / MIT; review before redistribution |
| opencv-python-headless, hydra-core, iopath | image/optional SAM2 support | Apache-2.0 / MIT / Apache-2.0 |
| huggingface-hub, accelerate, safetensors, sentencepiece | hub/model support | Apache-2.0 / Apache-2.0 / Apache-2.0 / Apache-2.0 |
| detectron2 | optional panoptic visualization only | Not installed by the qualified runtime; legacy behavior remains available when supplied |

## Pretrained models referenced by the downloader/pipeline

Downloaded by `huggingface_downloader.py` or named in configs; weights are
never committed to this repository.

| Model repo | Used by | Notes |
| --- | --- | --- |
| `facebook/sam2-hiera-large` @ `e6a8e880…` | `modules/segmenter/sam2.py` | HF model card declares Apache-2.0; exact revision/provenance and cache size are in [runtime.md](docs/runtime.md). |
| `openai/clip-vit-base-patch32` @ `3d74acf9…` | `modules/classifier/clip.py` | Pinned OpenAI research model card has no SPDX field and marks deployed use out of scope; do not treat it as a commercial/deployment license. |
| `Salesforce/xgen-mm-phi3-mini-instruct-r-v1` @ `1d91d356…` | `modules/verifier/blip3.py` | HF model card and `LICENSE.txt` identify CC-BY-NC-4.0; `trust_remote_code` audit is in [runtime.md](docs/runtime.md). |

Model revisions and the reviewed remote-code files are pinned. BLIP3 is
live-qualified through the sequential host-RAM/GPU lifecycle on the historical
11 GB host and through the all-resident lifecycle on the assigned 24,576-MiB
RTX 3090; Objectives 007–009 exercise all four supported service profiles.
Pinning and successful execution do not authorize client-selected model loading
or commercial redistribution. These are bounded local research measurements,
not an SLA, accuracy claim, license clearance, or external deployment.

## Distribution and rights status

demos/LICENSE.txt records a license pointer for demo imagery, but the release
allowlist excludes every demo and repository media payload and does not infer
rights for an unlisted path. Repository banner/logo assets are described as
project-owned, yet they are also excluded from release artifacts pending
independent inventory. The full path inventory and human gates are in
docs/RELEASE-GATE-INVENTORY.md.

The repository owner has explicitly confirmed redistribution rights for
`configs/goats.yaml`, `configs/goats2.yaml`, `demos/goats/goats1.jpg`, and
`demos/goats/goats2.jpg`; CRIT-0001 is human `ACCEPTED`. The files remain absent
from the current tracked tip and release artifacts as defense in depth, while
the local academic harness may use operator-held copies.

No model license or this notice authorizes commercial/deployed model use or
weight redistribution. Model-use and remaining media/release review are
separate from the cleared goat-fixture adjudication.
