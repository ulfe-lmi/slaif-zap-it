# ZAP-IT local service datasheet

Status: bounded evidence for the unpublished `0.1.0` local research candidate.
This is not a production SLA, public-deployment approval, accuracy guarantee,
or claim of leak-proof operation.

## Purpose and non-goals

The service accepts one image and one hostile-but-validated YAML document and
returns deterministic segmentation/classification artifacts through loopback or
an explicitly authorized authenticated RFC1918 `POST /v1/completions` endpoint.
It is a ZAP-IT multimodal contract, not generic
OpenAI text-completion compatibility.

It does not provide public/WAN exposure, TLS, gateway integration, async jobs,
history, persistence, training, multi-worker CUDA, geometry activation,
Detectron2 panoptic output, or customer-data handling.

## Model-control subset

The optional `explicit` mode adds authenticated `/v2` repository `index`,
`load`, and `unload` management for the immutable `zap-it-1` model only. It is
not KServe V2 tensor inference and makes no generic repository or cross-process
ownership claim. `READY` alone admits inference; unload pauses admission,
drains the active call, releases holders, and requires the 64-MiB logical Torch
allocated/reserved cold proof. The default `none` mode retains background
startup loading.

## Exact response levels

| Level | Contract |
|---|---|
| 0 | Envelope, five-field normalized YOLO text, dimensions, class map and config digest |
| 1 | Level 0 plus uint16 `identity-mask.png` |
| 2 | Level 1 plus produced object bbox/area/centroid/SAM/CLIP/BLIP/geometry fields |
| 3 | Level 2 plus stage metadata, post-filter diagnostics, timings, warnings, provenance, one bounded BLIP3 composition record per applicable candidate, bounded debug/annotated artifacts and exact per-object mask RLE |

L3 RLE uses `coco_rle_uncompressed`, `size: [height, width]`,
`order: column-major`, and alternating background/foreground counts. It is
round-trippable source-mask truth, including disconnected components and
overlap. The identity PNG remains a documented single-valued projection.

The L3 `post_filter_diagnostics` field shares the post-SAM2 filter's exact
short-circuit evaluator: `maxsize`, `empty_mask`, `max_w`, `max_h`, then
retained. The area comparison is terminal and occurs before segmentation access;
`maxsize` records carry the exact area and zero bbox dimensions because they were
not evaluated. Empty masks also carry zero dimensions for their distinct reason.
Values strictly above a limit are rejected and equal values are retained; other
bbox dimensions are inclusive extents on the remapped mask. Counts must
reconcile evaluated with retained and all four removal counters and must
cross-check `candidate_counts.sam2_candidates`/`after_area_bbox`. Rejection
records contain only numeric source index, reason, area and bbox dimensions, are
input-ordered and capped at 256 with `rejections_truncated`. This is
configured-filter evidence, not a SAM2 recall or accuracy claim; the roof test
is programmatic CPU evidence. JSON and ZIP carry the same diagnostic values,
while L0-L2 omit the field.

Every level also includes `service.sam2`. The service keeps the pinned SAM2
model resident and constructs exactly one fresh automatic-mask generator around
it per request. The manifest reports the requested safe scalars, all 14 total
safe generator scalars including `use_m2m`, all effective
values, independent `explicit`/`profile`/`default` sources, selected profile,
exact prompt and mask-prediction estimates, raw generator candidate count,
three-decimal SAM2 duration and deterministic resource warnings. Timing is
excluded from content-determinism comparisons; JSON and ZIP metadata otherwise
agree.

Candidate-view debug artifacts are L3-only lossless PNGs of the exact arrays
passed to the semantic processors. CLIP uses
`clip-candidate-view-CANDIDATE-####.png`; BLIP3 uses
`blip3-verification-CANDIDATE-####-QUESTION-####.png`. BLIP3 passes one image
per candidate: exact source pixels in Euclidean support are restored, the
exterior contour is painted, and every other crop pixel is Gaussian-blurred
scene context. The source crop uses inclusive raw/support bboxes and a
half-open array-slice bbox; its endpoints are independently clamped and must
contain support plus contour. The full composition is bilinearly resized to a
256-pixel short-side target with a 768-pixel long-side cap. The decoded PNG is
byte-identical to the sole image passed to QA; this artifact documents pixel
identity and does not guarantee semantic accuracy.

The exact radius-512 disk dilation uses a local-window squared distance
transform with a constant number of temporary arrays. Debug resource admission
is two-phase: CLIP artifacts are admitted before CLIP, and actual post-CLIP
labels/scores admit single-image BLIP3 debug questions before QA. The separate
L3 `blip3_candidate_views` list has one bounded record per applicable candidate,
including rendered/rejected status and the fixed containment diagnostic. A
rejected candidate receives no QA call, debug artifact or label mutation.

At L3, `mask_generator.debug: true` enables the bounded raw-SAM2 diagnostic.
The ordinary combined overlay is insufficient to audit overlap ownership, so
the service emits independent candidate contact-sheet tiles and exact
all-candidate union, overlap and uncovered diagnostics. Candidate IDs are
one-based source-order IDs and may have gaps for empty proposals. Pages are
fixed 3x4 layouts with 320x240 content and a 28-pixel label bar; no more than
eight pages or 96 candidates are represented. Scores use three decimals or
`n/a`, and no client text enters an artifact name.

Padded candidate crops may be enlarged to the 320x240 content area for
readability; the three full-image diagnostics never upscale.

The fixed names are `sam2-candidates-page-0001.png` through
`sam2-candidates-page-0008.png`, `sam2-union-coverage.png`,
`sam2-overlap-heatmap.png`, and `sam2-uncovered-pixels.png`. Union is black
uncovered/white covered, overlap is black at zero with a fixed observed-maximum
scaled ramp, and uncovered is the exact inverse of union at source resolution.
The typed `service.sam2.raw_visualization` child reports source and diagnostic
dimensions, exact coverage counts, bounded overlap histogram/overflow,
represented IDs and truncation. Preflight reserves at most 11 diagnostic
artifacts and the exact fixed RGB-array formula before readiness or inference;
existing encoded response limits still apply. This is bounded visualization
evidence, not segmentation-quality validation or a solar-array recall/precision
benchmark.

## Supported stages

ROI/resize, SAM2 candidate filtering, CLIP label refresh, deterministic ordering,
YOLO, identity PNG, mask-only annotated overlays, final-object
`annotated-labelled` overlays and L3 RLE are supported. BLIP3 rules are
supported with a pinned FP16 holder. Below 24,576 MiB the historical 11 GB
RTX 2080 Ti uses the live-qualified sequential stage-boundary lifecycle; at or
above 24,576 MiB the assigned RTX 3090 uses the live-qualified all-resident
lifecycle. Objective 009's real matrix covers `sam2`, `sam2_clip`,
`sam2_blip3`, and `sam2_clip_blip3`. Both modes expose only logical `cuda:0`
after an explicit operator index and UUID pin. These are bounded local research
measurements, not an SLA, accuracy claim, commercial-license clearance, or
external deployment. Geometry/panoptic and deployment/release gates remain
separate for reasons other than GPU memory, as documented in
[OUTPUT-PARITY.md](OUTPUT-PARITY.md).

## Limits (operator startup settings)

| Setting | Default | Environment variable |
|---|---:|---|
| Encoded image | 20 MiB | `SLAIF_ZAP_IT_MAX_IMAGE_UPLOAD_BYTES` |
| Encoded YAML | 256 KiB | `SLAIF_ZAP_IT_MAX_CONFIG_UPLOAD_BYTES` |
| Decoded pixels | 64,000,000 | `SLAIF_ZAP_IT_MAX_DECODED_PIXELS` |
| Image width / height | 8192 / 8192 | `SLAIF_ZAP_IT_MAX_IMAGE_WIDTH/HEIGHT` |
| Objects | 256 | `SLAIF_ZAP_IT_MAX_OBJECTS` |
| Visualization streams | 8 | `SLAIF_ZAP_IT_MAX_VISUALIZATION_STREAMS` |
| Response artifacts | 64 | `SLAIF_ZAP_IT_MAX_RESPONSE_ARTIFACTS` |
| Debug artifacts | 48 | `SLAIF_ZAP_IT_MAX_DEBUG_ARTIFACTS` |
| Single raw artifact | 32 MiB | `SLAIF_ZAP_IT_MAX_SINGLE_ARTIFACT_BYTES` |
| Total raw artifacts | 128 MiB; L3 annotated RGB reservations are deducted before debug sink admission; raw-SAM2 debug reserves its fixed worst case before inference | `SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES` |
| RLE runs/object | 250,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT` |
| RLE runs/response | 1,000,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL` |
| Response | 256 MiB | `SLAIF_ZAP_IT_MAX_RESPONSE_BYTES` |
| Host available floor | 2 GiB | `SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES` |
| `/dev/shm` free floor | 64 MiB | `SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES` |
| Deadline / queue | 120 s / 0 | `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS`, `SLAIF_ZAP_IT_QUEUE_DEPTH` |
| BLIP3 questions / generated tokens | 32 / 32 | Fixed service policy; not uploaded controls |
| SAM2 points per side / batch | 64 / 64 | `SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_SIDE`, `SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_BATCH` |
| SAM2 crop layers | 2 | `SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS` |
| SAM2 estimated prompts / predictions | 8,192 / 24,576 | `SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_PROMPTS`, `SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_MASK_PREDICTIONS` |
| SAM2 minimum region area | 1,000,000 | `SLAIF_ZAP_IT_SAM2_MAX_MIN_MASK_REGION_AREA` |

Budgets are validated at startup and cannot be changed by request YAML. For L3,
each supported annotated stream reserves exactly `height * width * 3` raw bytes
before inference; raw SAM2 debug additionally reserves
`8 * 960 * 1072 * 3 + 3 * diagnostic_width * diagnostic_height * 3` bytes and
11 artifact slots. A per-stream/fixed-debug overflow or combined overflow is
rejected before model execution, and the remaining configured-stream bytes are
the debug sink budget. L0-L2 do not render or reserve visualization arrays. Raw artifacts are checked before
retention/encoding; JSON checks base64 expansion before encoding, and ZIP writes
prepared raw bytes directly. RLE and every serialization loop check the absolute
120-second request deadline. There is no post-hoc artifact truncation.

## SAM2 capability policy

The authenticated `GET /v1/capabilities` route is static policy metadata. It
does not require readiness or acquire the inference gate. It documents the
strict public ranges, exact defaults/profiles, formula
`sum(4^layer * int(points_per_side / downscale_factor^layer)^2)` and the
multimask multiplier. Fixed model revision, logical device, dtype, residency,
cache/checkpoint/config paths, output destinations, `point_grids`,
`output_mode=binary_mask` and arbitrary kwargs are operator-owned; sensitive
path, credential, topology and process values are withheld. Invalid intrinsic
values return `invalid_config` 400. Exceeding an operator field or estimate cap
returns non-retryable `resource_limit` 413 before readiness, gate acquisition,
generator construction or inference.

Nested BLIP3 debug flags are stripped to false below L3 with one bounded
aggregate warning. At L3 no duplicate answer-text debug artifact is generated;
structured BLIP3 answers and labels remain normal response metadata.

## Hardware and software matrix

| Item | Qualified value |
|---|---|
| Historical sequential target | NVIDIA GeForce RTX 2080 Ti, 11,264 MiB, physical index 1; UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` |
| Assigned all-resident target | NVIDIA GeForce RTX 3090, 24,576 MiB, physical index 0; UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` |
| Visible application device | `cuda:0` after the explicit assigned index is masked and UUID-checked |
| Driver / CUDA Torch | Historical 580.178.04 / CUDA 12.4; assigned 610.43.02 / CUDA 12.4 / Torch 2.5.1+cu124 |
| Service stack | FastAPI 0.141.1, Uvicorn 0.52.4, python-multipart 0.0.32 |
| Process model | One Uvicorn worker, one inference executor, one request in flight |
| Protected devices | Every unassigned physical GPU and unrelated workload remain protected; only the active-order-assigned index+UUID is exposed as logical `cuda:0` |

## Measured evidence

The initial 128×128 synthetic service baseline measured approximately
434–437 ms steady request latency, approximately 0.9 s first request,
approximately 1,984,604 KiB process RSS, approximately 1,849 MiB GPU1 ready and
5,749 MiB during inference, returning to approximately 6 MiB after stop. The
The bounded 32-request table, live RLE/metrics/recovery evidence, and
final GPU/process snapshots are published in `oap/reports/005-a-report.md`.
These measurements are bounded fixture evidence, not an SLA or soak test.

The sequential qualification adds a real BLIP3 gate and ten-request
central-crop benchmark to its immutable report. It records startup, transition,
restore, latency and memory evidence without committing goat bytes or response
content. Objectives 008–009 add assigned-RTX-3090 all-resident evidence and the
exact eight-call four-profile matrix; all measurements are bounded local
research evidence and do not authorize deployment, commercial use, or release.

## Academic fixture policy

The repository owner confirmed redistribution rights for the two goat images
and two YAML files; CRIT-0001 is accepted. The files nevertheless remain
ignored, operator-supplied academic inputs and are excluded from packages and
release artifacts as defense in depth.

The harness uses in-memory central-50% crops under sanitized aliases. Source
bytes, crop bytes, prompts, labels, and raw responses are not copied into CI,
packages, reports, or generated evidence. This is semantic/state-isolation and
resource evidence, not an accuracy benchmark.

## Metrics and privacy

`GET /metrics` exposes a custom process-local Prometheus registry without
default process collectors. Labels are limited to stable outcome code,
verbosity, `json|zip`, fixed model component/outcome and fixed transition
direction/outcome; duration/size/object/artifact histograms are otherwise
unlabeled.
Readiness, active inference, logical `cuda:0` allocated/reserved/peak/free and
host RSS gauges are also exposed when available. Request IDs, labels, prompts, answers, filenames,
paths, headers, credentials and raw content are never metric labels or logs.
Metrics reset on process restart.

Requests remain in memory. Any filesystem compatibility call must use a unique
0700 directory below the configured `/dev/shm` root, 0600 files, and unconditional
cleanup. The current resident service has no request persistence.

## Deployment prerequisites and limitations

Use the pinned GPU lock, freshly verify the selected physical GPU's
UUID/process/memory state, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and the matching
`CUDA_VISIBLE_DEVICES`, and select a verified-unused scoped port. Private-LAN
mode additionally requires an exact RFC1918 host/CIDR and strong fixed bearer.
Protected GPUs must remain unchanged. Gateway integration and final release
remain separately governed.
