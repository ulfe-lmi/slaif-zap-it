# Installation

ZAP-IT relies on PyTorch with CUDA acceleration, Hugging Face transformers, SAM2, BLIP-3, CLIP and a few additional packages. The recommended way to reproduce our software stack is to create a Conda (or Mamba) environment from the supplied `environment.yml` file and then pull the model checkpoints from Hugging Face.

## 0. CPU-only development environment (tests, linting, packaging)

If you only want to run the test suite, linters or package builds — no
inference, no model downloads — use a plain virtualenv:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

This installs numpy/Pillow/PyYAML plus pytest, ruff, coverage and build tooling
only; heavy GPU libraries are intentionally excluded (see `pyproject.toml`).
Canonical verification commands are listed in [CONTRIBUTING.md](CONTRIBUTING.md),
and [docs/BASELINE.md](docs/BASELINE.md) documents exactly which behaviors the
CPU suite covers via its stub harness versus what needs the full stack below.

## 1. Prerequisites

1. **CUDA-enabled GPU** – the pipeline is GPU-first. Ensure that the GPU drivers available on your machine (or via HPC modules) match CUDA 12.1.
2. **Conda or Mamba** – the instructions below assume that either `conda` or `mamba` is available in your shell. On an HPC system, this is often exposed via a module such as `module load anaconda3` or `module load mambaforge`.
3. **Hugging Face account (optional but recommended)** – use `huggingface-cli login` if you need authenticated access to gated models.

### HPC-specific guidance

On multi-user clusters the CUDA toolkit, compilers and Python are typically provided as environment modules. Before creating the Conda environment, load the stack that matches the versions pinned in `environment.yml`:

```bash
module load gcc/11.3.0              # or a newer GCC that your site supports
module load cuda/12.1.1             # matches pytorch-cuda=12.1 from environment.yml
module load cudnn/8.9.7             # optional, only if provided separately
module load nccl/2.18.3             # optional, recommended for multi-GPU jobs
module load anaconda3/2023.09       # provides conda; adjust to your site's module name
```

> 💡 **Tip:** Ask your system administrators for the module names that correspond to CUDA 12.1, cuDNN 8.9+ and GCC 11+. Once the modules are loaded you can create the Conda environment exactly as on a workstation.

If your cluster offers Mamba it will significantly reduce environment solve time:

```bash
module load mambaforge
```

## 2. Create the Conda environment

With the required modules loaded (or on a workstation with recent NVIDIA drivers) run:

```bash
# Using conda (works everywhere)
conda env create -f environment.yml

# OR using mamba for faster solves
mamba env create -f environment.yml

conda activate zap-it
```

The `environment.yml` file pins PyTorch 2.3 with CUDA 12.1 support, along with the compiler toolchain (`cmake`, `ninja`) required by Detectron2 and other native extensions.

### CPU-only fallback

If you only have CPU access, remove the `pytorch-cuda=12.1` entry from `environment.yml` and let Conda resolve a CPU build of PyTorch instead. Keep in mind that inference will be significantly slower without a GPU.

## 3. Model code and checkpoint downloads

The environment already installs the latest SAM2 code directly from GitHub as well as the supporting libraries (`transformers`, `open-clip-torch`, `accelerate`, etc.). You still need to download the pretrained model weights. The easiest way is via the Hugging Face CLI:

```bash
conda activate zap-it
huggingface-cli login  # only if the models require authentication

# SAM 2
huggingface-cli download facebook/sam2-hiera-base --local-dir models/sam2

# BLIP-3 (choose the variant that matches your use case; base is shown here)
huggingface-cli download Salesforce/blip3-itm-base --local-dir models/blip3

# CLIP (ViT-L/14 example; swap in the architecture you need)
huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/clip
```

All commands above respect the `HF_HOME` environment variable. On a cluster you can set it to a scratch location to avoid filling your home directory:

```bash
export HF_HOME=/scratch/$USER/huggingface
```

If you prefer to script the downloads inside Python, `huggingface_hub.snapshot_download()` is available because it is included in the Conda environment.

## 4. Verifying the install

After the environment is active and the models are downloaded, run a quick smoke test to confirm that CUDA is visible:

```bash
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

When running on a Slurm-style HPC scheduler, remember to request a GPU in your job script (e.g. `#SBATCH --gpus=1`) and load the same modules you used when creating the environment before activating Conda in the job.

## 5. Additional resources

- Example configuration files are in the `configs/` directory.
- See [docs/CONFIG.md](docs/CONFIG.md) for details on creating custom YAML configurations.
- Hugging Face model cards provide the most up-to-date checkpoints:
  - [facebook/sam2-hiera-base](https://huggingface.co/facebook/sam2-hiera-base)
  - [Salesforce/blip3-itm-base](https://huggingface.co/Salesforce/blip3-itm-base)
  - [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
