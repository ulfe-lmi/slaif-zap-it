# ZAP-IT HTTP API

```text
POST /v1/completions
Content-Type: multipart/form-data
```

This is a **ZAP-IT-specific multimodal pipeline endpoint** using a conventional
path. It is **not** drop-in OpenAI `text-completions` compatibility.

The contract is tested with a CPU fake engine and qualified locally with pinned
SAM2, CLIP, and BLIP3 models. Operational residency is capacity-selected: the
historical 11-GB profile uses serialized host-RAM/GPU transitions for BLIP3,
while a strictly qualified card at or above 24 GiB keeps all three pinned
holders on logical `cuda:0`. Objective 009's authenticated real matrix covers
all four supported profiles on the assigned 24,576-MiB RTX 3090. This remains
bounded local research evidence, not an SLA, accuracy claim, commercial-license
clearance, or external deployment; geometry/panoptic and deployment/release
gates remain separate. No persistent listener is started by the package factory.

## Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/v1/completions` | POST | one image + one YAML config -> one result |
| `/v1/capabilities` | GET | authenticated static SAM2/candidate-view policy and schema |
| `/healthz` | GET | process/event-loop health (always unauthenticated) |
| `/readyz` | GET | engine readiness via injected provider; honest `not_ready` |
| `/metrics` | GET | process-local finite-cardinality Prometheus text |
| `/v2` | GET | bounded metadata advertising the repository-management subset |
| `/v2/repository/index` | POST | authenticated fixed-model lifecycle index |
| `/v2/repository/models/zap-it-1/load` | POST | authenticated synchronous fixed-model load |
| `/v2/repository/models/zap-it-1/unload` | POST | authenticated synchronous fixed-model unload |

The `/v2` paths implement only the [Triton Model Repository
Extension](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2450/user-guide/docs/protocol/extension_model_repository.html)
management vocabulary (`index`, `load`, `unload`) for the immutable `zap-it-1`
holder. They do not implement KServe V2 tensor inference; `/v1/completions`
remains the only inference contract. This deliberately does not claim the
[KServe V2 inference protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol).
Load and unload return an empty HTTP 200 body only after their lifecycle work
has completed.

## Explicit model control

`SLAIF_ZAP_IT_MODEL_CONTROL_MODE=none` is the default and preserves background
startup loading. In `explicit` mode startup is cold (`UNAVAILABLE`) and requires
the separate `SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY`; it must not equal
`SLAIF_ZAP_IT_API_KEY`. Management uses `Authorization: Bearer <control-key>`
even on loopback. The inference key never authorizes management, and neither
credential is accepted in multipart YAML or echoed in responses.

The repository index accepts only an empty body or `{ "ready": true|false }`.
Load and unload accept only an empty body or `{}`. Unknown fields, query
parameters, arbitrary model names, URL/path/revision/device overrides and
percent-encoded name tricks are rejected before lifecycle work. Index entries
use only `UNAVAILABLE`, `LOADING`, `READY` and `UNLOADING` states and a generic
sanitized reason.

Unload atomically pauses new and queued inference, drains the active synchronous
call, releases all holders and verifies logical Torch allocated and reserved
memory are each at most 64 MiB. A drain or memory-proof failure leaves the
service honest and never returns a misleading success. The process, listener
and `/healthz` remain live throughout.

## Request fields

Exactly one request = exactly one `image`, one `config`, one result.

| Field | Required | Rules |
|---|---|---|
| `image` | yes | exactly one part; JPEG/PNG/WebP only; encoded bytes <= limit; decoded pixels <= limit |
| `config` | yes | exactly one part; UTF-8 YAML document; byte limit enforced before parse |
| `verbosity` | no | strict integers `0\|1\|2\|3`; default `0`; textual aliases rejected in v1 |
| `response_format` | no | `json` (default) or `zip` |
| `model` | no | optional; only the fixed service id `zap-it-1` is accepted |
| `stream` | no | omitted or `false` only |

Unknown multipart fields, duplicate fields and missing required parts are
rejected with stable codes before any expensive work.

## Verbosity levels (monotonic information)

- **L0**: completion envelope + normalized YOLO lines in `choices[0].text`
  + minimal metadata (request id, image dims, class mapping, config digest),
  the complete `service.sam2` configuration/provenance object, and effective
  CLIP/BLIP3 candidate-view configuration with application status.
- **L1**: L0 + lossless uint16 identity PNG artifact (`identity-mask.png`,
  background `0`, instance ids `1..N`). Overlaps use the larger-area winner;
  if that would fully occlude an object, the service reserves a deterministic
  source pixel so the PNG IDs remain bijective with YOLO/object records.
- **L2**: L1 + per-object records containing only fields actually produced
  (bbox pixel+normalized, area, centroid, SAM quality, CLIP score, BLIP3
  answer when present, geometry hook when present, one-based
  `source_candidate_id` and zero-based post-SAM2 `filtered_index`).
- **L3**: L2 + stage statuses, candidate counts, post-filter diagnostics, timings, provenance,
  aggregate warnings, bounded annotated/debug artifacts, one-for-one numeric
  candidate-view input records, and one exact per-object uncompressed
  column-major COCO-style mask RLE. `annotated` remains
  mask-only; the optional `annotated-labelled` stream is final-stage, labelled,
  deterministic and Detectron2-free.

Lower levels never trigger extra optional stages solely to enrich output.
Configured filesystem-style debug flags are honored only at verbosity 3
(where they map to bounded logical artifacts in memory); below that they are
stripped from the effective config with an explicit warning.

### L3 post-filter diagnostics

The L3 `service.post_filter_diagnostics` sibling of `candidate_counts` records
one mutually exclusive outcome per candidate evaluated by the post-SAM2 filter.
The area comparison is terminal and occurs before segmentation access, so a
`maxsize` rejection records its exact `area_px` and zero `bbox_width_px` and
`bbox_height_px` because bbox dimensions were not evaluated. Precedence is
`maxsize`, `empty_mask`, `max_w`, then `max_h`; rejection uses strict `>` and
equality is retained. Empty masks also report zero dimensions for their distinct
reason; other bbox dimensions are inclusive extents of the exact remapped mask.
Counts satisfy
`evaluated = retained + removed_by_maxsize + removed_empty_mask +
removed_by_max_w + removed_by_max_h`, and the canonical engine cross-checks
`candidate_counts.sam2_candidates == evaluated` and
`candidate_counts.after_area_bbox == retained`.

Rejections contain only `source_index`, closed `reason`, `area_px`,
`bbox_width_px`, and `bbox_height_px`. They are ordered by filter input and
capped at 256; `rejections_truncated` reports rejected candidates beyond the
cap. The field is absent at L0-L2, has no artifact of its own, and is shared by
the JSON response and ZIP manifest. The two-wide-candidate roof case is a
programmatic CPU filter regression, not a real-image SAM2 accuracy benchmark.

When a BLIP3 rule executes, the verifier passes a deterministic paired RGB image
to every QA call: a target-only mask view on the left and the same candidate's
zero-filled, dimmed Euclidean-dilated context view on the right, with a
four-pixel dark divider and an exterior-only contour. The bbox is storage-only;
holes and disconnected components are not bridged. RGB is bilinearly resized,
masks nearest-neighbor resized, and support masks are reapplied before the
bounded 256-short-side/768-long-side policy. At L3, an effective rule with
`debug: true` adds only the exact paired image passed to QA as
`blip3-verification-CANDIDATE-####-QUESTION-####.png`; CLIP similarly emits
`clip-candidate-view-CANDIDATE-####.png`. Public candidate/question IDs are
one-based, filtered indices are zero-based, and no client text enters a name.

## Completion envelope

```json
{
  "id": "cmpl-<opaque>",
  "object": "text_completion",
  "created": 1770000000,
  "model": "zap-it-1",
  "choices": [{"index": 0, "text": "<YOLO lines>", "finish_reason": "stop"}],
  "usage": null,
  "schema_version": "zap-it.v1",
  "service": { "...level-gated metadata..." }
}
```

- YOLO lines are `<class_id> <cx> <cy> <w> <h>` with six-decimal coordinates
  normalized to the ORIGINAL image; empty detections yield an empty string.
- `usage` stays `null`: no invented token counts.
- Request ids are opaque server tokens; user filenames/content never enter
  ids, logs or responses.

Binary artifacts use one stable object shape:

```json
{"name": "...", "media_type": "...", "encoding": "base64",
 "sha256": "...", "size": 1234, "data": "<base64>"}
```

ZIP responses contain `manifest.json` (the full envelope without base64
payloads), `detections.yolo.txt`, `identity-mask.png` when applicable, and
level-gated artifacts with matching hashes/sizes/media types. ZIP assembly
is deterministic and uses the prepared raw artifacts directly; it does not
construct a duplicate base64 JSON payload first.

At L3 each object has:

```json
{"encoding":"coco_rle_uncompressed", "size":[height,width],
 "order":"column-major", "counts":[0, 3, 2]}
```

Counts begin with background and round-trip the complete source mask, including
disconnected components and overlap.

### SAM2 response manifest

Every JSON response and ZIP `manifest.json` contains `service.sam2`. Its
`effective` and `sources` mappings contain the 14 total safe generator scalars,
including `use_m2m`; `requested` contains only supplied `profile` and safe
scalar values.
`actual_candidate_count` is the raw count returned by the automatic generator,
before empty-mask removal, remapping, filtering or classification. It is
distinct from L3 `candidate_counts.sam2_candidates`. The measured
`execution_time_ms` includes request-local generator construction and
generation, is rounded to three decimals, and is observability data rather
than a deterministic-content field. No debug flag, unknown field, raw YAML or
operator control is echoed.

When verbosity is 3 and `mask_generator.debug: true`, `service.sam2` also has a
typed `raw_visualization` child. Its `candidate_id_base` is 1 and its raw
candidate count is the generator count before empty-mask removal; the
visualizable count plus omitted-empty count reconciles to that raw count.
`represented_candidate_ids` are the first 96 non-empty source-order IDs, so
empty proposals can create gaps and truncation is explicit. The child reports
exact original-resolution covered/uncovered counts, maximum overlap, histogram
keys 0..255 plus overflow pixels, contact-sheet count and source/diagnostic
dimensions. Union, overlap and uncovered artifacts are computed from all
non-empty candidates, not only those shown in the sheets.

Raw candidate sheets use fixed names `sam2-candidates-page-0001.png` through
`sam2-candidates-page-0008.png`; the other fixed names are
`sam2-union-coverage.png`, `sam2-overlap-heatmap.png`, and
`sam2-uncovered-pixels.png`. Pages have three columns, four rows, 320x240
content and a 28-pixel label bar. Candidate labels are
`C0001  IoU 0.843  stability 0.912`-style text with three decimals or `n/a`;
no user-controlled text is rendered. Crops use at least four pixels of clamped
`ceil(10%)` context padding, RGB bilinear/mask nearest-neighbor letterboxing
and 45% mask alpha; small padded crops are enlarged to fill the tile while the
three full-image diagnostics never upscale. Union is black/white
uncovered/covered, overlap is a fixed
blue-to-red ramp scaled by its observed maximum, and uncovered is the exact
inverse at source resolution before nearest-neighbor downscale to at most
2,000,000 pixels. Equal inputs are deterministic within a pinned environment;
arbitrary Pillow-version PNG byte identity is outside the claim.

Before readiness or gate admission, a debug request reserves the fixed maximum
of 11 diagnostic artifacts and the exact RGB formula
`8 * 960 * 1072 * 3 + 3 * diagnostic_width * diagnostic_height * 3` (the
2,000,000-pixel case is 42,698,880 bytes), in addition to configured streams.
Existing per-artifact, total raw-artifact, encoded JSON, ZIP and response-size
limits remain authoritative. Lower levels do not render or reserve these
diagnostics, and trusted CLI debug retains its historical rectangular JPEG
patch names and format.

### Capabilities

`GET /v1/capabilities` uses the ordinary inference bearer and is available
without model readiness or inference-gate admission. It returns an explicit
OpenAPI schema for strict field types/ranges, defaults, exact profile
overrides, current operator maxima and estimation formulas. Candidate-view
policy is one required top-level `candidate_views` object, independent of the
`raw_sam2_debug` policy; the response model, runtime JSON and OpenAPI schema
declare the same fields. Fixed model,
device, dtype, residency, cache/checkpoint/config and artifact-destination
controls are described as policy; sensitive operator paths, credentials,
topology and process state are not disclosed. `/docs` and `/openapi.json`
remain disabled on the private-LAN listener, while this authenticated route
remains available.

## Configuration policy (hostile uploads)

Parsing uses `yaml.safe_load` semantics behind bounds enforced during
composition: max depth 16, max 10 000 nodes, max 512 entries per collection,
max 16 384 characters per scalar, zero aliases/anchors accepted.

Top-level allowlist derived from the core boundary:

- **Accepted** (algorithmic): `alpha`, `preprocessing`, `mask_generator`,
  `postsam2processing`, `clip`, `blip3`, `candidate_views`, `visualization`.
- **Ignored with warning** (batch-only, never honored):
  `images`, `video`, `export_yolo_det`.
- **Rejected** (`unsupported_field`): anything else — including legacy
  `geometry` and `blip2` sections, which the single-image core does not consume.

Anywhere in the document:

- forbidden keys (paths/outputs/URLs/endpoints/devices/model repos/revisions/
  checkpoints/cache roots/credentials/env/service settings) ->
  `unsafe_config`;
- string values containing path/URL separators (`/`, `\`, `://`), leading `~`,
  drive-letter forms or control characters -> `unsafe_config`.

Uploaded configs can therefore never select filesystem paths, URLs,
commands, imports, Python symbols, devices, model repositories/revisions or
deployment settings. Legacy `visualization.alpha` hoisting matches CLI
normalization (default `0.6`). CLIP `labels` keys define the class mapping in
document order.

### Candidate-view policy

`candidate_views` is a typed request-local section with `clip` and `blip3`
children. Both default to `mode: mask_dilated`, `context_fraction: 0.10`,
`min_context_pixels: 0`, `max_context_pixels: 64`, `outside_fill: zero`, and
`context_intensity: 0.35`; BLIP3 also defaults to `contour_width: 2`. The exact
limits are fraction 0..0.5, minimum 0..256, maximum 0..512, intensity 0..1 and
BLIP3 contour 0..16. Null, bool-as-number, non-finite, unknown, out-of-range,
cross-field and unsupported values are rejected without clamping. `clip.padding`
is an unsupported service field; clients must use `candidate_views.clip`.

For `L = max(mask_bbox_width, mask_bbox_height)`, the builder reports
`raw_radius = ceil(context_fraction * L)` and
`effective_radius = min(max(raw_radius, min_context_pixels),
max_context_pixels)`. Dilation is an exact Euclidean disk clipped to the source.
The target is retained only where `M` is true; context is retained only in `D`,
with `floor(channel * context_intensity)` in `D - M`. Candidate and question
IDs are one-based; the post-SAM2 `filtered_index` is zero-based. L3 debug records
are one-for-one with fixed-name lossless model-input PNGs and contain only
bounded numeric provenance, not image pixels or client text. CLIP capacity is
admitted before the CLIP processor/model call; BLIP3 capacity is admitted after
actual CLIP labels/scores and before any QA call.

### Visualization streams

The service accepts bounded RGB streams only at L3. `annotated` and its legacy
`alpha-overlay` spelling draw masks without text. `annotated-labelled` is
accepted only under `visualization.blip3`; it draws the sanitized final
structured label and exact `ObjectResult.instance_id`. `show_confidence` is a
strict boolean and adds a finite two-decimal CLIP suffix when true. It never
changes structured labels, object ordering, masks, class mapping, or artifact
paths. `panoptic` remains unsupported.

```yaml
visualization:
  blip3:
    - id: labelled-result
      renderer: annotated-labelled
      alpha: 0.55
      show_confidence: true
```

## Limits (operator-overridable at startup only)

| Limit | Default | Env var |
|---|---|---|
| max image upload | 20 MiB | `SLAIF_ZAP_IT_MAX_IMAGE_UPLOAD_BYTES` |
| max config upload | 256 KiB | `SLAIF_ZAP_IT_MAX_CONFIG_UPLOAD_BYTES` |
| max decoded pixels | 64 000 000 | `SLAIF_ZAP_IT_MAX_DECODED_PIXELS` |
| max image width / height | 8192 / 8192 | `SLAIF_ZAP_IT_MAX_IMAGE_WIDTH`, `SLAIF_ZAP_IT_MAX_IMAGE_HEIGHT` |
| max objects | 256 | `SLAIF_ZAP_IT_MAX_OBJECTS` |
| max visualization streams | 8 | `SLAIF_ZAP_IT_MAX_VISUALIZATION_STREAMS` |
| max response/debug artifacts | 64 / 48 | `SLAIF_ZAP_IT_MAX_RESPONSE_ARTIFACTS`, `SLAIF_ZAP_IT_MAX_DEBUG_ARTIFACTS` |
| max single/total raw artifacts | 32 / 128 MiB; L3 annotated RGB reservations are deducted from the debug total | `SLAIF_ZAP_IT_MAX_SINGLE_ARTIFACT_BYTES`, `SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES` |
| max RLE runs/object/response | 250 000 / 1 000 000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT`, `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL` |
| max total response | 256 MiB | `SLAIF_ZAP_IT_MAX_RESPONSE_BYTES` |
| min available RAM / shm | 2 GiB / 64 MiB | `SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES`, `SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES` |
| request deadline | 120 s | `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS` |
| inference queue depth | 0 | `SLAIF_ZAP_IT_QUEUE_DEPTH` |
| Retry-After value | 5 s | `SLAIF_ZAP_IT_RETRY_AFTER_SECONDS` |

Encoded sizes are enforced while streaming (limit+1 pattern) before decode;
decoded width/height are checked from headers before pixel allocation; L3
annotated streams are preflighted as `height * width * 3` raw RGB bytes before
engine execution; host RAM and `/dev/shm` floors are checked at readiness and
request admission.

## Concurrency semantics

Exactly one inference executes at a time (single worker thread plus an async
gate). With the slot busy and the queue exhausted (default depth 0), arrivals
fail fast with `503 service_busy` and a `Retry-After` header. Deadlines wrap
the whole request, including RLE, artifact preparation, base64 expansion and
ZIP assembly; expiry returns `504 timeout` without a partial response.
Cancellation releases state immediately. The production
process model is one Uvicorn process, workers=1, no fork after CUDA init.

## Authentication

Strict-loopback deployments default to NO key. Setting
`SLAIF_ZAP_IT_API_KEY` before start enables mandatory
`Authorization: Bearer <key>` on `/v1/completions` and `/metrics` using
constant-time comparison. Keys are never logged or echoed; health endpoints
stay open.

Human-authorized private-LAN deployment sets
`SLAIF_ZAP_IT_NETWORK_SCOPE=private_lan`, one explicit RFC1918 host and a
containing RFC1918 CIDR. Startup fails unless the inference key is at least 32
characters. Wildcard, public, hostname and scope-mismatched binds are rejected.
`/docs` and `/openapi.json` are disabled on that listener;
`/v1/completions` and `/metrics` require the fixed bearer. TLS/WAN and
multi-user authorization are not provided by this mode.

## Errors

Stable sanitized envelope on every failure:

```json
{"error": {"code": "<stable_snake_code>", "message": "<sanitized>",
           "request_id": "<opaque>"}}
```

| Code | HTTP | Cause |
|---|---|---|
| `invalid_multipart` | 400 | not multipart / malformed body / too many parts |
| `missing_part` | 400 | image or config missing |
| `duplicate_part` | 400 | any field provided twice |
| `invalid_image` | 400 | corrupt/unknown media outside JPEG/PNG/WebP |
| `invalid_config` | 400 | non-UTF-8, non-mapping, unparseable YAML |
| `unsafe_config` | 400 | forbidden keys/values, aliases, bound violations |
| `unsupported_field` | 400 | unknown multipart field / unknown top-level key |
| `unsupported_verbosity` | 400 | not canonical 0..3 |
| `unsupported_format` | 400 | response_format outside json\|zip |
| `unsupported_model` | 400 | model != zap-it-1 |
| `stream_unsupported` | 400 | stream other than false |
| `unauthorized` | 401 | missing/wrong bearer key |
| `payload_too_large` | 413 | upload/body byte limits |
| `image_too_large` | 413 | decoded pixels over cap |
| `resource_limit` | 413 | SAM2 field or estimated-work cap exceeded; non-retryable |
| `response_too_large` | 413 | assembled JSON/ZIP over response cap |
| `cancelled` | 499* | cancelled before completion |
| `inference_failure` | 500 | engine failure (sanitized) |
| `internal_error` | 500 | unexpected internal failure (sanitized) |
| `service_busy` | 503 | slot busy + queue exhausted (+ `Retry-After`) |
| `not_ready` | 503 | readiness provider reports not ready |
| `timeout` | 504 | request deadline exceeded |
| `insufficient_memory` | 507 | reserved: RAM/`/dev/shm` pressure |
| `insufficient_shm` | 507 | configured `/dev/shm` floor is not available |

\* nginx-convention client-closed status, kept stable for this contract.

Messages never include stack traces, raw YAML/image bytes, host paths,
secrets or environment data.

## Data lifecycle

Everything lives in process memory; no request image/config/result is ever
persisted. A documented production workspace root
(`/dev/shm/slaif-zap-it`, `SLAIF_ZAP_IT_TMP_ROOT`) is reserved for later
objectives that require filesystem-backed stages; nothing writes there today.

## Example (fake-engine local test)

```python
from fastapi.testclient import TestClient
from src.service import FakeEngine, create_app

app = create_app(engine=FakeEngine())
client = TestClient(app)
response = client.post(
    "/v1/completions",
    files={
        "image": ("f.png", png_bytes, "image/png"),
        "config": ("c.yaml", yaml_bytes, "application/yaml"),
    },
    data={"verbosity": "2"},
)
```

OpenAPI: run the app and read `/openapi.json` (or `/docs`). The schema is
snapshot-tested against this document's contract.

## Metrics and parity

`/metrics` is process-local and resets on restart. It has no default process
collectors and uses only finite stable labels. See
[OUTPUT-PARITY.md](OUTPUT-PARITY.md) for the complete legacy/current output
classification and [SERVICE-DATASHEET.md](SERVICE-DATASHEET.md) for hardware,
measured evidence and deployment prerequisites.

## Limitations

- Live readiness requires the operator launcher, a freshly pinned exclusive
  GPU, and the complete local model cache.
- No streaming; no asynchronous jobs; no video API.
- BLIP3 request rules are bounded to 32 questions and 32 generated tokens per
  question. Model identity, revision, dtype, device and residency are fixed
  operator policy; they cannot be selected by YAML or multipart fields.
- Geometry and panoptic visualization remain explicitly unsupported service
  capabilities; `annotated-labelled` is the supported final-object labelled
  visualization and geometry activation requires a separate scientific-stage
  order.
- `geometry`/`blip2` config sections are rejected until the core consumes them.
