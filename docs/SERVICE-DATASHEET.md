# ZAP-IT local service datasheet

Status: bounded evidence for the unpublished `0.1.0` local research candidate.
This is not a production SLA, public-deployment approval, accuracy guarantee,
or claim of leak-proof operation.

## Purpose and non-goals

The service accepts one image and one hostile-but-validated YAML document and
returns deterministic segmentation/classification artifacts through a loopback
`POST /v1/completions` endpoint. It is a ZAP-IT multimodal contract, not generic
OpenAI text-completion compatibility.

It does not provide LAN/public exposure, TLS, gateway integration, async jobs,
history, persistence, training, multi-worker CUDA, geometry activation,
Detectron2 panoptic output, or customer-data handling.

## Exact response levels

| Level | Contract |
|---|---|
| 0 | Envelope, five-field normalized YOLO text, dimensions, class map and config digest |
| 1 | Level 0 plus uint16 `identity-mask.png` |
| 2 | Level 1 plus produced object bbox/area/centroid/SAM/CLIP/BLIP/geometry fields |
| 3 | Level 2 plus stage metadata, timings, warnings, provenance, bounded debug/annotated artifacts and exact per-object mask RLE |

L3 RLE uses `coco_rle_uncompressed`, `size: [height, width]`,
`order: column-major`, and alternating background/foreground counts. It is
round-trippable source-mask truth, including disconnected components and
overlap. The identity PNG remains a documented single-valued projection.

## Supported stages

ROI/resize, SAM2 candidate filtering, CLIP label refresh, deterministic ordering,
YOLO, identity PNG, annotated overlays and L3 RLE are supported. BLIP3 rules are
supported with a pinned FP16 holder. The live-qualified service swaps SAM2+CLIP
and BLIP3 at the BLIP3 stage boundary. The 11 GB sequential path is the only
live-qualified BLIP3 residency; all-resident qualification is separate. Geometry and panoptic remain
unsupported as documented in [OUTPUT-PARITY.md](OUTPUT-PARITY.md).

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
| Total raw artifacts | 128 MiB; L3 annotated RGB reservations are deducted before debug sink admission | `SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES` |
| RLE runs/object | 250,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT` |
| RLE runs/response | 1,000,000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL` |
| Response | 256 MiB | `SLAIF_ZAP_IT_MAX_RESPONSE_BYTES` |
| Host available floor | 2 GiB | `SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES` |
| `/dev/shm` free floor | 64 MiB | `SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES` |
| Deadline / queue | 120 s / 0 | `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS`, `SLAIF_ZAP_IT_QUEUE_DEPTH` |
| BLIP3 questions / generated tokens | 32 / 32 | Fixed service policy; not uploaded controls |

Budgets are validated at startup and cannot be changed by request YAML. For L3,
each supported annotated stream reserves exactly `height * width * 3` raw bytes
before inference; a per-stream overflow or combined overflow is rejected before
model execution, and the remaining bytes are the debug sink budget. L0-L2 do not
render or reserve visualization arrays. Raw artifacts are checked before
retention/encoding; JSON checks base64 expansion before encoding, and ZIP writes
prepared raw bytes directly. RLE and every serialization loop check the absolute
120-second request deadline. There is no post-hoc artifact truncation.

## Hardware and software matrix

| Item | Qualified value |
|---|---|
| Physical target | NVIDIA GeForce RTX 2080 Ti, 11,264 MiB, physical index 1 |
| Target UUID | `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` |
| Visible application device | `cuda:0` after `CUDA_VISIBLE_DEVICES=1` |
| Driver / CUDA Torch | 580.178.04 / CUDA 12.4 / Torch 2.5.1+cu124 |
| Service stack | FastAPI 0.141.1, Uvicorn 0.52.4, python-multipart 0.0.32 |
| Process model | One Uvicorn worker, one inference executor, one request in flight |
| Protected device | Physical GPU0 and its unrelated workload are never used |

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
content. The >=24-GB all-resident path is implemented and CPU/fake-tested but
is intentionally not claimed as live-qualified.

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
`CUDA_VISIBLE_DEVICES`, and select a verified-unused loopback port. Protected
GPUs must remain unchanged. Gateway integration, LAN exposure, and final release
remain separately governed.
