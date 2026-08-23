![ZAP-IT Banner](assets/banner.jpg)

# ZAP-IT - Zero-shot Anything Pipeline for Image Tasks

ZAP-IT is a high level computer‑vision pipeline built from foundational models. The goal is to describe your image task in a YAML file and let the pipeline handle segmentation and classification while preserving optional legacy geometry helpers. The local service deliberately does not activate geometry.

Version 0.1.0 is an unpublished release candidate. It is a bounded local
handoff, not a public-release, accuracy, SLA or rights-clearance claim.

- **Installation instructions:** see [INSTALL.md](INSTALL.md).
- **Configuration guide:** see [docs/CONFIG.md](docs/CONFIG.md).
- **Algorithms overview:** see [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## Quick Start

1. Install the CPU or qualified GPU environment as described in [INSTALL.md](INSTALL.md).
2. Pick or create a YAML file in `configs/` describing your task (see [docs/CONFIG.md](docs/CONFIG.md)).
3. Run the batch script over a folder of `.jpg` files:
   ```bash
   python zap-it-batch.py --config configs/example.yaml --input-image-dir path/to/images --verbose full
   ```
   The output images and JSON metadata are written to `output/` inside the given folder.
4. If the configuration contains an `export_yolo_det` section the batch script
   will also build a YOLO formatted dataset under a `yolo/` directory placed
   alongside the `output/` folder inside the given images directory.
   You can randomise the processing order of images with `--randomize`:
   ```bash
   python zap-it-batch.py --config configs/example.yaml --input-image-dir path/to/images --randomize
   ```
5. To process images on multiple GPUs simultaneously provide `--ngpu` with the
   desired number, e.g. `--ngpu 2` will handle two images in parallel.
6. To segment a video instead of an image directory provide `--input-video`:
   ```bash
   python zap-it-batch.py --config configs/example.yaml --input-video path/to/video.mp4
   ```
   Video outputs are written under `output/<video-stem>/` alongside the source file.
   Pass `--ngpu` with a value greater than one to fan frames across multiple GPUs; the
   parent process buffers frames to preserve ordering, so expect additional memory use
   when enabling this parallel mode.
7. Upon loading the YAML configuration the script prints a short summary of
   which optional modules are enabled (CLIP, BLIP3, YOLO export, etc.).

## Repository Layout

- `zap-it-batch.py` – orchestrates the full pipeline.
- `src/config.py` – loads and fixes the YAML configuration.
- `modules/input/` – helpers for loading and pre-processing image/video inputs.
- `modules/output/` – image/video writers and YOLO dataset exporter.
- `modules/visualizer.py` – rendering utilities that turn masks into RGB overlays.
- `modules/segmenter/` – SAM2 segmentation implementation via the unified module API.
- `modules/classifier/` – CLIP-based zero-shot classification module.
- `modules/verifier/` – BLIP-3 verification module.
- `src/postprocessing.py` – filters masks by size and bounding-box limits.
- `modules/geometry/` – optional line-based geometry analysis with Canny/Hough.
- `configs/` – example YAML files.
- `assets/` – logos and banner image used in this README.

The original README contained raw console logs from early experiments; it has been replaced with this concise guide. Use the documentation files for further details.

## Development and testing

A CPU-only development environment (no torch/GPU/model downloads) supports
linting, packaging and the unit-test suite:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,service]'
.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

Note that the CPU suite exercises pipeline logic through a documented stub
harness; see [docs/BASELINE.md](docs/BASELINE.md) for the precise inventory of
what is tested against real libraries versus stubs. The qualified pip GPU
environment from [INSTALL.md](INSTALL.md) is required for actual inference.

CI runs formatting/lint/wheel-build checks plus the CPU test suite on Python
3.10–3.12 and CodeQL security analysis on every PR and push to `main`.

## Additional documentation

- [INSTALL.md](INSTALL.md) – CPU development and pinned GPU runtime setup.
- [docs/CONFIG.md](docs/CONFIG.md) – YAML configuration reference.
- [docs/API.md](docs/API.md) – HTTP `/v1/completions` contract (CPU/fake-engine).
- [docs/OUTPUT-PARITY.md](docs/OUTPUT-PARITY.md) – complete legacy/current output classification.
- [docs/SERVICE-DATASHEET.md](docs/SERVICE-DATASHEET.md) – bounded service levels, limits, evidence and prerequisites.
- [ALGORITHMS-DETAILED.md](ALGORITHMS-DETAILED.md) – stage-by-stage walkthrough.
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md) – short algorithm overview.
- [CONTRIBUTING.md](CONTRIBUTING.md) – development workflow and rules.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) – license/provenance pointers.
- [SECURITY.md](SECURITY.md) – input/data/host security law.
- [docs/runtime.md](docs/runtime.md) – Objective 003-a hardware, provenance and measured resource strategy.
- [docs/RUNBOOK.md](docs/RUNBOOK.md) – operator start/stop, live smoke, evidence and rollback.
- [docs/GATEWAY-INTEGRATION.md](docs/GATEWAY-INTEGRATION.md) – separate future gateway contract (not implemented).
- [docs/RELEASE-GATE-INVENTORY.md](docs/RELEASE-GATE-INVENTORY.md) – rights, model, media and repository-setting gates.
- [CHANGELOG.md](CHANGELOG.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md) – unpublished candidate history and limitations.

ZAP-IT ships the batch CLI, the CPU-tested HTTP contract (`POST
/v1/completions`) and the operator-controlled Objective 004 loopback launcher.
The live service is deliberately limited to the measured resident SAM2+CLIP
profile on freshly verified physical GPU1; BLIP3, LAN/public exposure and
release readiness is governed by the unpublished 0.1.0 candidate gates in
docs/RELEASE-GATE-INVENTORY.md. Wheels and sdists deliberately contain no
demos, media, goat fixtures, model weights, caches, outputs or OAP transcript
material.
