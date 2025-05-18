# Installation

ZAP-IT relies on PyTorch, Hugging Face transformers and a few additional packages. We recommend creating a conda environment using the provided `environment.yml` file.

## Using Conda

```bash
conda env create -f environment.yml
conda activate zap-it
```

The `environment.yml` installs PyTorch with CUDA support, plus all core Python libraries. After activating the environment, install the SAM2 code from its repository:

```bash
pip install git+https://github.com/facebookresearch/SAM2.git
```

This command fetches the latest SAM2 implementation and weights via Hugging Face.

## Additional Notes

- The pipeline expects a CUDA-capable GPU for reasonable speed.
- Example configuration files are in the `configs/` directory.
- See [docs/CONFIG.md](docs/CONFIG.md) for how to create your own YAML configuration.
