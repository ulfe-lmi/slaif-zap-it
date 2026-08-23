# Third-party notices and provenance

ZAP-IT is licensed under the MIT License (see [LICENSE](LICENSE)). It builds on
the following third-party software and models. This file records pointers for
license review; it is not legal advice. Objective 003-a uses the pinned
repo-owned pip lock in `requirements-gpu-cu124.lock`; model weights remain
operator cache assets and are never committed or redistributed.

## Python packages

### CPU/dev toolchain (declared in `pyproject.toml`)

| Package | Role | License |
| --- | --- | --- |
| numpy | array math | BSD-3-Clause |
| pillow | image I/O | HPND (MIT-CMU) |
| pyyaml | YAML config parsing | MIT |
| pytest, pytest-cov, coverage | test/coverage tooling | MIT / Apache-2.0 (coverage) |
| ruff | formatter/linter | MIT |
| build | packaging frontend | MIT |

### GPU/runtime stack (Objective 003-a pip lock, operator-managed)

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

Model revisions and the remote-code file hashes are pinned for Objective 003-a.
The BLIP3 profile remains rejected on the verified 11 GiB GPU; pinning does not
authorize client-selected model loading or commercial redistribution.

## Assets

`demos/LICENSE.txt` records licensing for bundled demo imagery. Repository
banner/logo assets are project-owned.
