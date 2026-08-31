# ZAP-IT architecture

## Purpose and boundary

ZAP-IT is a YAML-driven computer-vision pipeline and a bounded local service.
It orchestrates existing foundation models; it is not a model-training system,
generic image generator, object store, or OpenAI text-completions replacement.

The repository owns two supported entry paths:

1. a trusted batch/video CLI that may write configured outputs and YOLO datasets;
2. a stateless single-image HTTP service that keeps request data in memory and
   exposes only an allowlisted subset of the YAML surface.

Both paths use the same algorithm adapters and in-memory core. Legacy helpers
remain available where documented, but their mere presence does not make them a
service capability.

## System overview

```text
Trusted batch CLI                         Local API client
       |                                        |
       | trusted YAML + paths                   | image + API-safe YAML
       v                                        v
filesystem adapters                    request/auth/resource guards
       |                                        |
       +------------------+  +------------------+
                          v  v
                    typed CoreConfig
                          |
                          v
       preprocess -> SAM2 -> filter -> CLIP -> optional BLIP3
                          |
                          v
                  deterministic result
             objects + masks + stages + timings
                    /                    \
                   v                      v
          CLI writers/YOLO          JSON/ZIP renderers
```

The HTTP service defaults to loopback and may bind one explicit RFC1918 address
only in authenticated `private_lan` mode. It runs one worker and one active
inference and exposes exactly one operator-pinned physical GPU as logical `cuda:0`. An
optional explicit model controller keeps that process/listener live while its
fixed model is cold.

## Components

### Configuration

`src.core.CoreConfig` is the normalized algorithm boundary. Trusted batch
configuration may additionally contain input/output, video, debug, and dataset
export controls.

The service parser accepts only bounded algorithm sections. It uses safe YAML
composition/loading, rejects aliases and excessive structures, and forbids
paths, URLs, commands, imports, environment variables, devices, credentials,
model repositories/revisions, downloads, caches, and service settings. Uploaded
configuration never becomes host authority.

### In-memory core

`src.core.run_single_image()` executes one decoded RGB array and returns a typed
`SingleImageOutcome`. It performs preprocessing, SAM2 mask generation,
post-filtering, optional CLIP and BLIP3 stages, final label filtering,
deterministic ordering, final-object visualization, and provenance assembly.

The core does not require a filesystem. Debug-capable modules receive an
artifact sink. The service uses a bounded memory sink; the trusted CLI can use a
filesystem sink through its compatibility adapter.

The shared pure candidate-view module constructs a complete source-byte-exact
rectangular CLIP `raw_bbox_crop` from a mask-derived inclusive bbox. Its
half-up context radius affects only the crop boundary; no mask or fill reaches
CLIP. The service maps each safe identifier to one indivisible prompt string or
an ordered array of independent prompts, trims only prompt boundaries, encodes
each item separately, and aggregates by maximum similarity into one complete
ordered semantic-class cosine vector. The pinned tokenizer limit is 77 tokens;
invalid counts, characters, tokens, and trimmed duplicates are structured
`invalid_config` errors before inference. An explicitly selected trusted-CLI
`mask_dilated` compatibility builder and multi-prompt labels remain separate.
BLIP3 uses a separate pure single-image compositor: an
inclusive raw-mask bbox determines a nominal centered crop, exact Euclidean
dilation determines support, and a second exact dilation determines an exterior
contour. Source RGB pixels under support D are restored from source bytes; the
exterior contour is painted with the configured RGB color, and every other crop
pixel is Pillow Gaussian-blurred scene context. The crop is rejected
locally if support plus contour cannot fit after endpoint clamping. Only the
fully composed image is bilinearly resized for QA, with short side 256 and long
side capped at 768. The fixed instruction follows the delimited client
question. This is pixel-boundary evidence, not semantic-accuracy evidence.

### Models and residency

Approved SAM2, CLIP, and BLIP3 identities/revisions are fixed in operator code.
Snapshots live in an operator-managed Hugging Face cache and are loaded with
network access disabled during service operation.

Residency is selected from fresh physical capacity after an explicit operator
index and UUID pin, and the process exposes the selected card only as logical
`cuda:0`. Below 24,576 MiB the live-qualified strategy is sequential:

```text
normal request:  SAM2 + CLIP on GPU, BLIP3 in host RAM

BLIP3 request:   SAM2 -> CLIP on GPU
                    |
                    v
                 SAM2 + CLIP to CPU
                 BLIP3 to GPU -> verify single mask-aware images
                 BLIP3 to CPU
                 SAM2 + CLIP restored to GPU
```

The swap happens at the BLIP3 stage boundary, after SAM2 and CLIP have run. A
context-managed `finally` restores baseline residency before response return;
restoration failure makes readiness fail until restart. Requests are serialized
so two transitions cannot overlap.

An all-three-GPU-resident implementation is selected automatically from fresh
physical capacity at or above 24,576 MiB. It requires all pinned holders to
prove residency on logical `cuda:0` before readiness and performs no request-
time movement. Objective 009's authenticated real matrix covers all four
supported profiles on the assigned 24,576-MiB RTX 3090. Both strategies are
bounded local research evidence, not an SLA, accuracy claim, commercial-license
clearance, or external deployment. Geometry/panoptic and deployment/release
gates remain separate for reasons other than GPU memory.

### Device guard

Strict startup requires:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<operator-selected physical GPU index>
```

Startup independently checks physical index, UUID, PCI address, model, capacity,
and compute processes, then
cross-checks the single masked CUDA device. It refuses an occupied or mismatched
target instead of selecting another GPU. Unassigned devices and unrelated
workloads are outside ZAP-IT authority.

### HTTP service

FastAPI exposes:

- `GET /healthz` — process liveness;
- `GET /readyz` — device and model-registry readiness;
- `GET /metrics` — bounded Prometheus evidence;
- `POST /v1/completions` — one image plus one YAML configuration.

The optional fixed-model management subset also exposes `GET /v2`,
`POST /v2/repository/index`, and fixed-name `load`/`unload` paths. These follow
the KServe/Triton Model Repository Extension vocabulary but do not implement
KServe V2 tensor inference. `SLAIF_ZAP_IT_MODEL_CONTROL_MODE=none` preserves
background startup loading; `explicit` requires a separate loopback bearer
credential and starts `UNAVAILABLE`.

Lifecycle state and inference admission share one authority:

```text
UNAVAILABLE -> LOADING -> READY -> UNLOADING -> UNAVAILABLE
```

Only `READY` admits inference. Load/unload work runs on one control executor;
unload atomically pauses new/queued requests, drains the active call, drops
all holders, runs bounded CUDA/GC cleanup, and proves the 64-MiB logical Torch
allocated/reserved cold bound and measured loaded-memory release. Failed or
cancelled operations settle in a stable sanitized state.

The completion path enforces multipart cardinality, upload and decoded-image
limits, YAML policy, host and shared-memory floors, one active inference,
absolute deadlines, object/artifact/response bounds, optional bearer auth, and
sanitized errors. The API returns no token-usage fiction; `usage` remains null.

### Results and renderers

One deterministic object order drives every representation:

1. descending mask area;
2. ascending centroid row;
3. ascending centroid column;
4. ascending source candidate index.

The service produces normalized five-field YOLO lines, a uint16 identity-mask
PNG, per-object metadata, overlap-preserving RLE, post-filter diagnostics,
annotated overlays, warnings, timings, and provenance according to verbosity.
`annotated` is mask-only;
`annotated-labelled` is an L3-only, deterministic, Detectron2-free final-object
overlay with sanitized labels and exact instance IDs. JSON binary data uses
bounded base64 descriptors; ZIP uses a deterministic manifest and names.

Post-filter diagnostics use optional canonical area, bbox, aspect-ratio, and
border rules with equality retained and fixed first-reason precedence. Every
candidate, including empty masks, is accounted for; non-empty rejections carry
their inclusive bbox, dimensions, area, reason, configured limit, and source ID.
L3 records remain input-ordered and capped at 256 with an explicit truncation
count; lower response levels do not serialize this sidecar. CLIP routing keeps
complete semantic-class score vectors for every post-geometry candidate and
applies OR logic
for top-1, top-k, margin, minimum score, and explicit uncertainty before the
deterministic candidate cap.

### Ephemeral storage

Requests normally remain entirely in RAM. Compatibility operations receive an
opaque mode-0700 workspace below `/dev/shm/slaif-zap-it`, with mode-0600 files
and unconditional cleanup. The service never falls back silently to persistent
disk and never writes client-controlled paths.

## Trust boundaries

| Boundary | Trusted | Untrusted / forbidden |
| --- | --- | --- |
| Operator startup | GPU UUID/index, cache, port, API key, fixed limits | Request overrides |
| Uploaded YAML | Allowlisted bounded algorithm rules | Paths, models, devices, network, code, secrets |
| Model assets | Pinned local snapshots and reviewed remote code | Request-selected repositories/downloads |
| Request data | In-memory processing within one call | Persistence, raw-body logs, content metric labels |
| Network | Loopback, or human-authorized explicit RFC1918 host/CIDR with fixed bearer | Wildcard, public/WAN, unauthenticated LAN |

## Supported versus legacy-only

The service supports preprocessing, SAM2, post-filtering, CLIP, BLIP3,
deterministic result rendering, bounded mask-only annotated visualizations, and
the final-object `annotated-labelled` L3 stream. Geometry,
panoptic/Detectron2 rendering, batch folders, video, dataset export, and legacy
filesystem debug workflows are not service stages.

The CLI remains backward compatible for trusted workflows and continues to own
image-directory/video orchestration and YOLO dataset generation.

## Verification model

Public CI runs static/package checks, Python 3.10–3.12 CPU/fake tests, release
artifact audits, secret checks, and CodeQL without CUDA or model downloads.
Explicit local GPU qualification verifies the pinned device, models, memory,
service behavior, repeated requests, failure recovery, cleanup, and GPU0
non-interference. Green CI is necessary but does not substitute for live GPU
evidence.

See [docs/runtime.md](docs/runtime.md) for measurements,
[docs/API.md](docs/API.md) for the wire contract, and
[SECURITY.md](SECURITY.md) for the complete security boundary.
