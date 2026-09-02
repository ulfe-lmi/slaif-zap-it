# ZAP-IT local service datasheet

Status: bounded evidence for the unpublished `0.1.0` local research candidate.
This is not a production SLA, public-deployment approval, accuracy guarantee,
or claim of leak-proof operation.

## Purpose and non-goals

The service accepts one image and one hostile-but-validated YAML document and
returns deterministic segmentation/classification artifacts through loopback or
an explicitly authorized authenticated RFC1918 `POST /v1/completions` endpoint.
This native endpoint is the private operator/research/debug contract: it is not
generic OpenAI Completions compatibility, not the `slaif-api-gateway` backend
contract, and not the general-public SLAIF surface.

`POST /v1/responses` is the separate future gateway/public compatibility
facade. It is stateless, non-streaming, and returns the bounded public JSON
projection with an optional standard annotated PNG; gateway integration and
public/WAN deployment are not claimed here.

It does not provide public/WAN exposure, TLS, gateway integration, async jobs,
history, persistence, training, multi-worker CUDA, Canny/Hough geometry activation,
Detectron2 panoptic output, or customer-data handling.

## Responses facade

`POST /v1/responses` is a separate, stateless, non-streaming JSON transport for
the future gateway surface. It accepts exactly one `user` message with one
inline strict-base64 JPEG/PNG/WebP `input_image` and one inline strict-base64
UTF-8 YAML `input_file`; the file name is a safe ASCII `.yaml`/`.yml` basename
used only as metadata. The required model is `zap-it-1`, and only `store: false`,
`stream: false`, `background: false`, and the optional standard
`image_generation` declaration are supported. URLs, file IDs, text, state,
hosted tools and extra fields are rejected before inference.

The assistant `output_text` is canonical `zap-it.public.v1` JSON containing
public final-object evidence without masks/RLE, private debug artifacts,
runtime/GPU facts, paths, ZIPs or token usage. The image declaration appends
one `image_generation_call` whose standard base64 result is the existing final
annotated renderer at fixed `alpha=0.5` and `show_confidence=false`; it is not
generative image output. The route shares the completion gate, executor,
readiness, model holders and request limits. The derived encoded body cap and
decoded limits are authenticated through `/v1/capabilities`.

Responses errors use the OpenAI-shaped `message/type/param/code` body and put
the opaque request ID in `x-request-id`. The official `openai==3.7.0` CPU and
operator qualification uses typed SDK parsing. `slaif-api-gateway` is unchanged
and its end-to-end Responses multimodal/image-generation path is not qualified.

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

L3 service visualization descriptors use fixed ordinal names such as
`visualization/stream-0001.png`. They carry the configured stream as bounded
logical `visualization_id` metadata in JSON and ZIP manifests, including a
matching omission record when budget admission omits the bytes; identity and
candidate/debug artifacts omit that field.

The L3 `post_filter_diagnostics` field evaluates optional canonical geometry
rules for every candidate, including empties. Inclusive bbox, area,
aspect-ratio, and border checks retain equality and use fixed precedence.
Rejections carry source ID, nullable bbox, area, dimensions, configured limit,
and reason, are input-ordered and capped at 256 with `rejections_truncated`.
Counts reconcile evaluated, retained, and each removal reason. This is
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
per candidate: exact source pixels in Euclidean support D are restored from
source bytes, the exterior contour is painted with configured RGB, and every
other crop pixel is Gaussian-blurred
scene context. The source crop uses inclusive raw/support bboxes and a
half-open array-slice bbox; its endpoints are independently clamped and must
contain support plus contour. The full composition is bilinearly resized to a
256-pixel short-side target with a 768-pixel long-side cap. RGB pixels decoded
from the lossless PNG equal the sole image passed to QA; encoded PNG bytes are
not raw RGB bytes. This artifact documents pixel
identity and does not guarantee semantic accuracy.

The BLIP3 `infeasible_geometry_policy` defaults to `reject`. The explicit
`centroid_radial_mask_chord` value is tried only after the existing Euclidean
containment rejection. It uses a whole-mask centroid, ordered external
contours, cross-gap chord counts, inclusive spokes/quadrilaterals, and a common
millionth fixed-point scale. Candidate-local tight-bbox/local-window scratch and
fixed-size ray batches keep the geometry bounded. Contour reduction, contour
disabling, crop shifting, radial scaling, and zero-context fallback are reported
in precedence order in L3 composition/input records. Raw radial statistics are
pre-clamp diagnostics and may exceed `max_context_pixels`; effective statistics
remain bounded. Existing feasible views retain their prior pixels and metadata;
composition and QA timings are reported separately.

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
represented IDs and truncation. Optional diagnostic bytes are admitted by the
shared post-inference ledger; count and byte misses become typed omissions while
existing essential response limits still apply. This is bounded visualization
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
| Total raw artifacts | 128 MiB; optional L3 artifacts are admitted greedily and omitted with typed ledger reasons when over budget | `SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES` |
| RLE runs/object | 250,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT` |
| RLE runs/response | 1,000,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL` |
| Response | 256 MiB | `SLAIF_ZAP_IT_MAX_RESPONSE_BYTES` |
| Host available floor | 2 GiB | `SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES` |
| `/dev/shm` free floor | 64 MiB | `SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES` |
| Deadline / queue | 120 s / 0 | `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS`, `SLAIF_ZAP_IT_QUEUE_DEPTH` |
| BLIP3 rule definitions / planned questions / generated tokens | max 32 uploaded rules / 1..256 (default 256) planned questions / 32 | Rule count is a request YAML structural limit; `SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS` is operator-only and planned excess is `resource_limit` 413 before generation |
| CLIP semantic classes / prompts | 1..32 / 1..64 per class, 1..256 total | Canonical YAML policy; arrays are independent prompts, scalar values are indivisible |
| CLIP prompt size / tokenizer context | 512 Unicode codepoints / 77 tokens including special tokens | Invalid values return sanitized `invalid_config` 400 before inference |
| SAM2 points per side / batch | 64 / 64 | `SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_SIDE`, `SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_BATCH` |
| SAM2 crop layers | 2 | `SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS` |
| SAM2 estimated prompts / predictions | 8,192 / 24,576 | `SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_PROMPTS`, `SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_MASK_PREDICTIONS` |
| SAM2 minimum region area | 1,000,000 | `SLAIF_ZAP_IT_SAM2_MAX_MIN_MASK_REGION_AREA` |

Budgets are validated at startup and cannot be changed by request YAML. For L3,
supported artifacts are offered in deterministic pipeline/name order after
stage work. Stage, candidate and page selection exclusions do not set
`truncated`; operator count/raw/response omissions do. L0-L2 do not render or
deliver optional diagnostics. JSON checks base64 expansion before encoding and
ZIP writes prepared raw bytes directly. Response-byte omission rebuilds the
delivered tuple and preserves the essential identity mask; CLIP/BLIP3 records
retain candidate evidence with `omitted_response_limit` status. RLE and every
serialization loop check the absolute 120-second request deadline. An essential
response can still return `response_too_large` after optional tail omission.
Visualization members retain fixed ordinal names (`visualization/stream-####.png`)
and expose the validated configured ID as logical `visualization_id` metadata in
JSON, ZIP manifests, and omission records; the ID is never a path or member name.

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
generator construction or inference and includes sanitized estimates, causes,
public limits and admissible same-validator alternatives.

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

The service accepts domain-neutral routing configuration: CLIP labels are
bounded identifiers plus exactly one natural-language value each, `raw_bbox_crop`
is the only API CLIP view, and every surviving candidate receives a complete
ordered cosine vector before `clip_routing` selects request-authored BLIP3
rules through deterministic OR conditions. Canonical geometry records losses
with source IDs and inclusive bboxes. Optional artifact delivery is non-fatal;
selection, pagination, budgets, exact delivered bytes and typed omissions are
reported in `service.artifact_delivery`. CPU/fake evidence does not prove
semantic accuracy.
