# Third-party notices and provenance

ZAP-IT is licensed under the MIT License (see [LICENSE](LICENSE)). It builds on
the following third-party software and models. This file records pointers for
license review; it is not legal advice. Runtime GPU dependencies are installed
by the operator conda environment (`environment.yml`), not by `pip install`.

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

### GPU/runtime stack (conda environment `environment.yml`, operator-managed)

| Package | Role | License pointer |
| --- | --- | --- |
| pytorch 2.3.x / torchvision / torchaudio | tensor/GPU runtime | BSD-3-Clause |
| detectron2 (incl. bundled ops) | panoptic visualization utilities | Apache-2.0 |
| SAM-2 (`facebookresearch/sam2`) | segmentation implementation | Apache-2.0 |
| transformers >= 4.41 | model loading/inference glue | Apache-2.0 |
| open-clip-torch | optional CLIP backends | MIT |
| opencv-python-headless | Canny/Hough geometry stage | Apache-2.0 |
| huggingface-hub, accelerate, safetensors, sentencepiece | hub/model support | Apache-2.0 / MIT/HF variants — review before redistribution |
| matplotlib | composite rendering | PSF-like (Matplotlib license) |

## Pretrained models referenced by the downloader/pipeline

Downloaded by `huggingface_downloader.py` or named in configs; weights are
never committed to this repository.

| Model repo | Used by | Notes |
| --- | --- | --- |
| `facebook/sam2-hiera-large` | `modules/segmenter/sam2.py` | SAM 2 checkpoints; check current model-card license before any redistribution/commercial use. |
| `openai/clip-vit-base-patch32` | `modules/classifier/clip.py` | OpenAI CLIP weights; MIT-style model card license at time of writing — re-verify. |
| `Salesforce/xgen-mm-phi3-mini-instruct-r-v1` | `modules/verifier/blip3.py` | BLIP-3 family releases have carried non-commercial research terms in the past; **verify the exact model license before any production or commercial deployment.** |

Model revisions are not pinned yet; pinning/reviewing remote-code models and
revisions is a planned modernization objective. Until then, treat model
downloads as operator actions outside CI.

## Assets

`demos/LICENSE.txt` records licensing for bundled demo imagery. Repository
banner/logo assets are project-owned.
