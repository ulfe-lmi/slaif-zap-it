<div style="text-align: center;">
  <a href="https://www.slaif.si">
    <img src="https://slaif.si/img/logos/SLAIF_logo_ANG_barve.svg" width="400" height="400" alt="SLAIF">
  </a>
</div>

# ZAP-IT

[![CI](https://github.com/ulfe-lmi/slaif-zap-it/actions/workflows/ci.yml/badge.svg)](https://github.com/ulfe-lmi/slaif-zap-it/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ulfe-lmi/slaif-zap-it/actions/workflows/codeql.yml/badge.svg)](https://github.com/ulfe-lmi/slaif-zap-it/actions/workflows/codeql.yml)

**Zero-shot Anything Pipeline for Image Tasks**

ZAP-IT turns a YAML description of an image task into segmentation masks,
open-vocabulary labels, optional visual-language verification, visualizations,
and YOLO annotations. It combines SAM2, CLIP, and optional BLIP3 while
preserving the original batch and video workflows.

The repository also provides a bounded, local, single-image HTTP service. The
service accepts one image and one API-safe YAML configuration through
`POST /v1/completions` and returns deterministic JSON or ZIP artifacts at four
verbosity levels. It is a ZAP-IT-specific multimodal contract, not a generic
OpenAI text-completions implementation.

Version `0.1.0` is an unpublished release candidate. The current evidence is a
local research/development qualification, not a production SLA, accuracy
guarantee, commercial model-use clearance, or public-deployment authorization.

## Capabilities

| Capability | Batch CLI | Local service |
| --- | :---: | :---: |
| SAM2 automatic segmentation | Yes | Yes |
| CLIP zero-shot classification | Yes | Yes |
| BLIP3 verification and relabeling | Yes | Yes |
| In-memory YOLO, identity PNG, object metadata, RLE, and overlays | Adapter-dependent | Yes |
| YOLO dataset export | Yes | No |
| Image-directory and video processing | Yes | No |
| Canny/Hough geometry helpers | Legacy integrations | No |
| Panoptic/Detectron2 rendering | Optional legacy helper | No |
| LAN, public, multi-worker, or multi-tenant deployment | No | No |

### GPU residency

The live-qualified service strategy keeps SAM2 and CLIP on the selected GPU and
retains pinned FP16 BLIP3 in host RAM. A BLIP3 request runs SAM2 and CLIP first,
swaps those holders to CPU for the BLIP3 stage, and restores the baseline before
returning. On the qualified 11 GB RTX 2080 Ti, isolated BLIP3 peaked at
9,532 MiB reserved (88.09% of CUDA-visible memory); ten repeated central-crop
pipeline calls completed in approximately 10.2–11.5 seconds each.

The all-three-model resident implementation exists for cards with at least
24,576 MiB, but it has not yet completed its separate live qualification. No
request can select a model, revision, device, dtype, or residency policy.

## Quick start: development

The CPU environment runs formatting, packaging, and the complete fake-engine
test suite without CUDA or model downloads:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,service]'
.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

See [INSTALL.md](INSTALL.md) for the pinned GPU environment and installed
service workflow.

## Quick start: batch pipeline

After installing the GPU environment and model assets:

```bash
.venv-gpu/bin/python zap-it-batch.py \
  --config configs/tomato.yaml \
  --input-image-dir /path/to/images \
  --verbose full
```

The trusted batch workflow writes results below the supplied image directory.
It also supports `--input-video`, `--randomize`, and the legacy multi-GPU
`--ngpu` mode. Filesystem destinations and batch controls are intentionally not
accepted by the HTTP service.

## Quick start: local service

Copy and secure the operator environment template, replace the UUID/cache/API
key placeholders, set a freshly verified free loopback port, source the file,
and use the repository-owned launcher:

```bash
cp deploy/service.env.example ~/.config/slaif-zap-it/service.env
chmod 600 ~/.config/slaif-zap-it/service.env
${EDITOR:-vi} ~/.config/slaif-zap-it/service.env
set -a
. ~/.config/slaif-zap-it/service.env
set +a

scripts/serve_local.sh start
curl --fail-with-body http://127.0.0.1:${SLAIF_ZAP_IT_PORT}/healthz
curl --fail-with-body http://127.0.0.1:${SLAIF_ZAP_IT_PORT}/readyz
```

Example completion request:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer ${SLAIF_ZAP_IT_API_KEY}" \
  -F image=@/path/to/image.jpg \
  -F config=@configs/tomato.yaml \
  -F verbosity=2 \
  -F response_format=json \
  http://127.0.0.1:${SLAIF_ZAP_IT_PORT}/v1/completions
```

Stop the service with `scripts/serve_local.sh stop`. Keep it loopback-only and
use exactly one worker and one active inference request.

## Output levels

| Verbosity | Response additions |
| ---: | --- |
| 0 | Completion envelope and normalized five-field YOLO lines |
| 1 | Deterministic uint16 identity-mask PNG |
| 2 | Per-object mask-derived geometry and available SAM2/CLIP/BLIP3 metadata |
| 3 | Bounded timings, stage metadata, RLE masks, overlays, warnings, and provenance |

Binary artifacts are base64 descriptors in JSON or files in a bounded ZIP.
Request bytes and intermediate results remain in RAM or the validated
`/dev/shm` workspace and are removed after every request.

## Documentation

Start with the [documentation index](docs/README.md), or go directly to:

- [Installation](INSTALL.md)
- [Configuration reference](docs/CONFIG.md)
- [HTTP API](docs/API.md)
- [Algorithms](docs/ALGORITHMS.md)
- [Architecture](ARCHITECTURE.md)
- [Core library](docs/CORE.md)
- [GPU runtime and measured evidence](docs/runtime.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Service datasheet](docs/SERVICE-DATASHEET.md)
- [Security policy](SECURITY.md)
- [Testing](TESTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Release status and gates](docs/RELEASE-GATE-INVENTORY.md)

## Repository layout

```text
modules/       model and algorithm adapters
src/core/      typed in-memory pipeline and renderers
src/service/   multipart API, limits, errors, metrics, and serialization
src/runtime/   device guard, model residency, readiness, and local launcher
configs/       tracked batch/API-safe example configurations
scripts/       qualification, smoke, release, and operator tools
docs/          user, operator, architecture, evidence, and historical docs
tests/         CPU/fake tests plus explicit opt-in GPU tests
```

## Security and data handling

Treat images and uploaded YAML as hostile. The service bounds uploads, decoded
dimensions, YAML structure, artifacts, objects, deadlines, concurrency, host
memory, shared memory, and BLIP3 questions/tokens. Uploaded YAML cannot control
paths, downloads, models, devices, code, network access, credentials, or service
settings. Model caches remain operator assets outside Git.

See [SECURITY.md](SECURITY.md) for the complete boundary and responsible
disclosure information.

## Maintainer

Janez Perš<br>
Laboratory for Machine Intelligence (LMI)<br>
Faculty of Electrical Engineering, University of Ljubljana<br>
[LMI](https://lmi.fe.uni-lj.si/en) · [SLAIF](https://www.slaif.si)

Security reports: janez.pers@fe.uni-lj.si

## Acknowledgement

We acknowledge the support of the EC/EuroHPC JU and the Slovenian Ministry of
HESI through the SLAIF project (grant agreement 101254461).
