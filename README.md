![ZAP-IT Banner](assets/banner.jpg)

# ZAP-IT - Zero-shot Anything Pipeline for Image Tasks

ZAP-IT is a high level computer‑vision pipeline built from foundational models. The goal is to describe your image task in a YAML file and let the pipeline handle segmentation, classification and optional geometry analysis. It assumes access to a powerful machine (GPU) so you can focus on the configuration rather than coding details.

- **Installation instructions:** see [INSTALL.md](INSTALL.md).
- **Configuration guide:** see [docs/CONFIG.md](docs/CONFIG.md).
- **Algorithms overview:** see [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## Quick Start

1. Install the dependencies and create the conda environment as described in [INSTALL.md](INSTALL.md).
2. Pick or create a YAML file in `configs/` describing your task (see [docs/CONFIG.md](docs/CONFIG.md)).
3. Run the batch script over a folder of `.jpg` files:
   ```bash
   python zap-it-batch.py --config configs/example.yaml --dir path/to/images --verbose full
   ```
   The output images and JSON metadata are written to `output/` inside the given folder.

## Repository Layout

- `zap-it-batch.py` – orchestrates the full pipeline.
- `zap_it_config.py` – loads and fixes the YAML configuration.
- `zap_it_sam2.py` – wraps the SAM2 segmentation model for single-pass or tiled processing.
- `zap_it_clip.py` – optional CLIP-based zero-shot classification for each mask.
- `zap_it_blip3.py` – optional BLIP-3 verification after classification.
- `zap_it_postseg_processing.py` – filters masks by size and bounding-box limits.
- `zap_it_visualization.py` – builds summary composites and panoptic overlays.
- `zap_it_geometry.py` – optional line-based geometry analysis with Canny/Hough.
- `configs/` – example YAML files.
- `assets/` – logos and banner image used in this README.

The original README contained raw console logs from early experiments; it has been replaced with this concise guide. Use the documentation files for further details.
