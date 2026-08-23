# Installation

ZAP-IT has a light CPU development environment and a separate operator-managed
GPU environment. The GPU path is pinned for the verified Objective 003-a host;
it does not start a service or expose a listener.

## 0. CPU-only development

This path runs tests, linting and packaging without CUDA, model downloads or
network access from the test suite:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m build
```

The CPU tests use fakes for heavyweight model libraries. See
[docs/BASELINE.md](docs/BASELINE.md) for the exact coverage boundary.

## 1. Qualified GPU environment

Requirements are CPython 3.12, a compatible NVIDIA driver and `uv`. The lock
uses PyTorch 2.5.1/cu124 wheels and the pinned SAM2/Transformers support stack.
The verified host uses physical GPU index 1; GPU0 is protected.

```bash
uv venv .venv-gpu --python python3.12
SAM2_BUILD_CUDA=0 uv pip install --python .venv-gpu/bin/python \
  -r requirements-gpu-cu124.lock

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
export SLAIF_ZAP_IT_STRICT_GPU=1
export SLAIF_ZAP_IT_RESOURCE_STRATEGY=sam2_clip_resident_blip3_rejected
export SLAIF_ZAP_IT_SUPPORTED_PROFILES=sam2,clip,sam2_clip
```

`SAM2_BUILD_CUDA=0` avoids compiling an optional extension with the host's
system nvcc minor. The Python SAM2 implementation is still installed and
qualified. Detectron2 is intentionally not installed: it is lazy and optional
for the legacy panoptic visualization path.

The historical `environment.yml` is retained as a legacy reference and is not
the reproducible Objective 003-a environment on this host. Do not use it to
change system packages or CUDA.

## 2. Pinned model snapshots and qualification

Download only the approved, immutable model revisions into the operator's
default Hugging Face cache and run the bounded qualification:

```bash
.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download
.venv-gpu/bin/python scripts/qualify_gpu_runtime.py
```

The script uses an in-memory generated fixture, captures all-GPU snapshots,
checks the masked UUID, measures SAM2/CLIP separately and together, and
rejects BLIP3 before load when its conservative prediction exceeds the 90%
budget. It never writes weights or request data to the repository or
`/dev/shm`.

Exact revisions, licenses, remote-code audit, measured tables, supported
profiles and the selected Objective 004 port are recorded in
[docs/runtime.md](docs/runtime.md). Model use remains subject to the notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); weights are not committed.

## 3. Optional live GPU test

The live test is explicit, serialized and excluded from CPU CI:

```bash
ZAP_IT_RUN_GPU=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8 \
ZAP_IT_TESTS_ALLOW_SOCKETS=1 \
.venv-gpu/bin/pytest -m gpu tests/test_gpu_integration.py
```

Without `ZAP_IT_RUN_GPU=1`, the module skips honestly. It sets/checks the mask
before importing Torch, uses a RAM-backed lock, and verifies that GPU0's
compute-process evidence is unchanged.

## 4. Legacy batch usage

The batch CLI and YAML examples remain available after the GPU environment is
installed:

```bash
.venv-gpu/bin/python zap-it-batch.py \
  --config configs/goats.yaml \
  --input-image-dir path/to/images
```

The Objective 003 runtime policy is operator-owned. Uploaded API YAML cannot
select a model repository, revision, device, cache, path, command or resource
strategy. Objective 004 is required before starting the local API service.
