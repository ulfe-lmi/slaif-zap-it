# Installation

ZAP-IT separates lightweight development tooling from its operator-managed GPU
runtime. CPU installs never pull CUDA frameworks or model weights implicitly.

## Requirements

- Linux for the qualified service workflow;
- Python 3.10–3.12 for CPU development;
- Python 3.12, `uv`, a compatible NVIDIA driver, and the pinned CUDA runtime for
  the qualified GPU environment;
- sufficient host RAM for model construction and sequential BLIP3 residency;
- a RAM-backed `/dev/shm` workspace for service compatibility operations.

The published package version is not yet released: `0.1.0` remains an
unpublished release candidate.

## CPU development environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev,service]'

.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m build --wheel --sdist
```

CPU tests use injected fakes for heavyweight model libraries, disable network
access, and do not require CUDA. See [TESTING.md](TESTING.md).

## Install a built candidate

Build artifacts should be installed outside the checkout:

```bash
python3 -m venv /path/to/zap-it-venv
/path/to/zap-it-venv/bin/pip install dist/zap_it-0.1.0-py3-none-any.whl
/path/to/zap-it-venv/bin/python -c 'import src; print(src.__version__)'
test -x /path/to/zap-it-venv/bin/zap-it-service
```

The wheel contains the lightweight package and service entrypoint, not the GPU
stack or model snapshots. Do not publish artifacts or create a final release
until the current release-gate inventory has been reviewed.

## Pinned GPU environment

Create the repository-local runtime from the lock file:

```bash
uv venv .venv-gpu --python python3.12
SAM2_BUILD_CUDA=0 uv pip install --python .venv-gpu/bin/python \
  -r requirements-gpu-cu124.lock
```

`SAM2_BUILD_CUDA=0` skips an optional compiled extension; the qualified Python
SAM2 path remains installed. Detectron2 is intentionally absent because the
panoptic renderer is not a service capability.

The historical `environment.yml` is a legacy reference, not the qualified
runtime definition.

## Pinned model snapshots

The approved model IDs and revisions are fixed in `src/runtime/models.py`.
Download them once into an operator-managed Hugging Face cache:

```bash
.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download
```

Normal service launches must be offline. Model weights and caches must never be
placed in the repository, package, request workspace, or release artifacts.
Model licenses and provenance are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Device and service configuration

Before every launch, inspect all physical GPUs and processes. Use the exact
operator-assigned physical index and matching UUID; the historical maelstrom1
qualification used GPU1, while the 008-a hinton2 qualification used GPU0.
All other devices and processes remain protected.

Copy the environment template and replace its placeholders:

```bash
install -d -m 700 ~/.config/slaif-zap-it
install -m 600 deploy/service.env.example \
  ~/.config/slaif-zap-it/service.env
```

The private file supplies the expected GPU UUID, physical index, scoped port,
model-cache root, shared-memory root, API key, fixed resource limits, and the
startup-only SAM2 field/estimated-work caps. Set a
specific port after verifying that it is unused when subsequent commands need
to reference `$SLAIF_ZAP_IT_PORT`. Source the file before launching:

```bash
set -a
. ~/.config/slaif-zap-it/service.env
set +a
```

Do not
set the removed `SLAIF_ZAP_IT_RESOURCE_STRATEGY` or
`SLAIF_ZAP_IT_SUPPORTED_PROFILES` variables; residency is service-owned.

Launch through the repository wrapper:

```bash
scripts/serve_local.sh start
scripts/serve_local.sh status
scripts/serve_local.sh stop
```

Or run the installed `zap-it-service` foreground entrypoint from the prepared
GPU environment. The service must inherit:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=<verified physical GPU index>
CUDA_VISIBLE_DEVICES=<verified physical GPU index>
```

It must see exactly one logical device as `cuda:0` and match the pinned UUID.

Cards below 24,576 MiB use the live-qualified sequential SAM2+CLIP/BLIP3
stage-boundary lifecycle measured on the historical 11 GB RTX 2080 Ti. Cards at
or above 24,576 MiB use the live-qualified all-resident lifecycle measured on
the assigned RTX 3090. The Objective 009 matrix covers all four supported
profiles; both modes expose only logical `cuda:0` after the explicit operator
index and UUID pin. This is bounded local research evidence, not a production,
license, SLA, accuracy, or external-deployment claim.

## Optional GPU integration test

The explicit GPU test is serialized and excluded from public CI:

```bash
ZAP_IT_RUN_GPU=1 \
ZAP_IT_TESTS_ALLOW_SOCKETS=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=<assigned-physical-index> \
CUDA_VISIBLE_DEVICES=<assigned-physical-index> \
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=<fresh-target-uuid> \
.venv-gpu/bin/pytest -q -m gpu tests/test_gpu_integration.py
```

Without `ZAP_IT_RUN_GPU=1`, the test skips honestly. It must never allocate or
signal a protected GPU or change drivers, CUDA, firewall, or system services.

## Optional user-systemd template

`deploy/zap-it-local.service` is an uninstalled `Type=simple` template. Before
manual installation, replace the executable placeholder, verify the private
environment file, and keep the bind address loopback-only. Package verification
does not install, enable, reload, or start systemd units.

For an explicitly authorized private-LAN installation, use
`scripts/install_private_lan_service.py`. It writes a mode-0600 operator file,
generates or preserves a fixed bearer without printing it, and installs a
user-unit definition. LAN mode accepts only an exact RFC1918 host inside the
configured RFC1918 CIDR; wildcard/public binds fail closed. Review the generated
files, then run `systemctl --user daemon-reload` and enable
`zap-it-lan.service`.

## Batch CLI

After the GPU environment and model cache are available:

```bash
.venv-gpu/bin/python zap-it-batch.py \
  --config configs/tomato.yaml \
  --input-image-dir /path/to/images \
  --verbose full
```

The batch CLI is trusted to use configured input/output paths and retains image,
video, debug, visualization, and YOLO dataset-export behavior that the service
intentionally rejects.

For operation and rollback, continue with [docs/RUNBOOK.md](docs/RUNBOOK.md).
