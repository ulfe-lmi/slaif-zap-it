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

At L3, `annotated` remains the mask-only overlay. The final-stage
`annotated-labelled` stream is a deterministic, Detectron2-free overlay whose
labels and manifest instance numbers come from the final structured objects;
structured labels remain available whether or not a visualization is requested.

BLIP3 verification is mask-aware: each executed question receives a deterministic
side-by-side RGB image with untouched context on the left and an exact
mask-highlighted candidate on the right. The service's L3 `debug: true` rules
expose that exact paired image as a fixed-name lossless PNG; this is an audit
artifact, not a guarantee of semantic accuracy.

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
| Authenticated fixed-host private-LAN deployment | No | Yes |
| Public/WAN, multi-worker, or multi-tenant deployment | No | No |

### GPU residency

Residency is selected from fresh physical capacity after an explicit operator
index/UUID pin, and the process exposes the selected card only as logical
`cuda:0`. A card below 24,576 MiB uses the live-qualified sequential
stage-boundary lifecycle on the historical 11 GB RTX 2080 Ti. A card at or
above 24,576 MiB uses the live-qualified all-resident lifecycle on the assigned
24,576-MiB RTX 3090. Objectives 007–009 provide real evidence for all four
supported profiles (`sam2`, `sam2_clip`, `sam2_blip3`, and
`sam2_clip_blip3`). These are bounded local research measurements, not an SLA,
accuracy claim, commercial-license clearance, or external deployment.

Geometry/panoptic behavior and gateway, deployment, licensing, media, and
final-release gates remain separate unsupported or release-gated scope; none
is represented as GPU-memory-blocked work. No request can select a model,
revision, device, dtype, or residency policy.

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

The authenticated `GET /v1/capabilities` route documents the strict
request-local SAM2 generator policy without requiring readiness. It exposes the
exact defaults, `fast`/`balanced`/`quality` profiles, intrinsic ranges,
startup operator caps and bounded estimation formulas; it never exposes
credentials, operator paths, GPU topology or mutable request state.

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

Stop the service with `scripts/serve_local.sh stop`. Loopback remains the
default. A human-authorized fixed-host private-LAN deployment is documented in
the runbook and requires a strong on-disk bearer. Use exactly one worker and one
active inference request.

## Output levels

| Verbosity | Response additions |
| ---: | --- |
| 0 | Completion envelope, normalized five-field YOLO lines, and `service.sam2` |
| 1 | Deterministic uint16 identity-mask PNG |
| 2 | Per-object mask-derived geometry and available SAM2/CLIP/BLIP3 metadata |
| 3 | Bounded timings, stage metadata, post-filter diagnostics, RLE masks, overlays, warnings, provenance, and BLIP3 verification PNGs |

Binary artifacts are base64 descriptors in JSON or files in a bounded ZIP.
Request bytes and intermediate results remain in RAM or the validated
`/dev/shm` workspace and are removed after every request.

L3 post-filter diagnostics report one short-circuit outcome per evaluated SAM2
candidate: `maxsize`, `empty_mask`, `max_w`, `max_h`, or retained, in that
precedence order. The area comparison is terminal and occurs before segmentation
access; a `maxsize` rejection records its exact area and `0/0` bbox dimensions
because bbox dimensions were not evaluated. Empty masks also report `0/0` for
their distinct reason. Other bbox dimensions are inclusive extents of the
remapped segmentation. Thresholds are strict for rejection and inclusive for
retention. Counts satisfy `evaluated = retained + all four removal counts` and
cross-check `candidate_counts.sam2_candidates` and `after_area_bbox`. Rejection
records contain only numeric area, bbox dimensions, and source index; at most
256 are retained in input order, with the remainder in `rejections_truncated`.
This is configured-filter evidence, not a SAM2 recall or model-accuracy claim.
The two-wide-candidate roof regression is programmatic
CPU filter evidence, not a real roof-image benchmark.

At L3, `mask_generator.debug: true` adds a bounded raw-SAM2 audit. The normal
combined overlay cannot explain overlapping proposals, so the service emits
separate source-order candidate tiles plus union coverage, overlap heatmap and
uncovered-pixel PNGs. IDs are one-based `_source_index + 1` values and scores
are labelled to three decimals (or `n/a`). Pages are fixed 3x4 sheets with
320x240 content and a 28-pixel label bar, capped at eight sheets/96 candidates;
the fixed names are `sam2-candidates-page-0001.png` through `-0008.png`,
`sam2-union-coverage.png`, `sam2-overlap-heatmap.png`, and
`sam2-uncovered-pixels.png`. All non-empty raw masks contribute to exact source
coverage/overlap counts, including truncated candidates. Diagnostics never
upscale and use nearest-neighbor downscaling to at most 2,000,000 pixels.
The `service.sam2.raw_visualization` manifest child reports dimensions,
histogram overflow and explicit truncation. Lower levels and legacy CLI
rectangular JPEG debug patches are unchanged.

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
