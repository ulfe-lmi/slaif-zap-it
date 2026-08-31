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
  aggregate warnings, bounded annotated/debug artifacts, one bounded
  `blip3_candidate_views` record per applicable candidate, one-for-one debug
  input records, and one exact per-object uncompressed
  column-major COCO-style mask RLE. `annotated` remains
  mask-only; the optional `annotated-labelled` stream is final-stage, labelled,
  deterministic and Detectron2-free.

Lower levels never trigger extra optional stages solely to enrich output.
Configured filesystem-style debug flags are honored only at verbosity 3
(where they map to bounded logical artifacts in memory); below that they are
stripped from the effective config with an explicit warning.

### L3 post-filter diagnostics

The L3 `service.post_filter_diagnostics` sibling of `candidate_counts` records
one optional-geometry outcome per candidate evaluated by the post-SAM2 filter,
including empty masks. It records fixed precedence across area, inclusive bbox,
aspect-ratio, and border rules; equality is retained. Each rejection includes
source candidate ID, nullable inclusive bbox, area, dimensions, configured limit
field/value and reason. Counts reconcile exactly and bounded records report
truncation at 256. The field is absent at L0-L2 and is shared by JSON and ZIP.

When a BLIP3 rule executes, the verifier composes one deterministic RGB image
per applicable candidate and passes that same image to every QA call for the
candidate. The inclusive raw-mask bbox determines only a centered nominal crop;
exact Euclidean dilation supplies support and a second exact dilation supplies
an exterior contour. Source pixels under support D are restored from source
bytes; exterior contour pixels are painted with the configured RGB color, and
all remaining source-scene pixels are Gaussian-blurred with Pillow. A clamped
crop that cannot contain support plus contour is rejected locally before image,
QA, or debug work. The complete composition is bilinearly resized under the
256-short-side/768-long-side policy. At L3, an effective rule with `debug: true`
adds the exact sole QA image as
`blip3-verification-CANDIDATE-####-QUESTION-####.png`; CLIP retains its separate
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

Optional L3 artifacts are admitted greedily after the stages produce them. The
service never reserves the fixed raw-SAM2 maximum or visualization RGB arrays
before inference. `service.artifact_delivery` records the requested/effective
selection, page, candidate filter, operator budgets, exact delivered names and
hashes, and bounded omission reasons (`not_selected_*` or `omitted_*`). Optional
count, per-artifact, aggregate-raw and response-byte overflow sets
`truncated: true` and preserves the successful inference. Before the final hard
response check, admitted optional artifacts are removed from the tail in
reverse order as needed; each omission rebuilds the immutable artifact tuple,
descriptors, hashes, byte totals and ZIP members. The required
`identity-mask.png` is never selected for optional omission. CLIP/BLIP3 debug
records retain their structured candidate evidence while their artifact status
becomes `omitted_response_limit`. Only an essential response that still cannot
fit returns `response_too_large`; trusted CLI debug keeps its historical
rectangular JPEG patch behavior.

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

`configuration.field_catalog` is the ordered, typed inventory of every accepted
service YAML leaf. Each record contains an OpenAPI-enumerated `path`, a
`CapabilityField` descriptor, and explicit required/nullable/default semantics;
the compatibility `configuration.fields` dictionary is generated from the same
inventory. L3 response metadata uses named `StageStatus`, `CandidateCounts`,
`TimingMetadata`, `ProvenanceMetadata` and `ClipRoutingConfiguration` models.
Timing values are finite, non-negative milliseconds keyed by dynamic
`stage.<name>` timers; per-label CLIP scores and sanitized runtime model maps
remain bounded typed maps.

## Configuration policy (hostile uploads)

Parsing uses `yaml.safe_load` semantics behind bounds enforced during
composition: max depth 16, max 10 000 nodes, max 512 entries per collection,
max 16 384 characters per scalar, zero aliases/anchors accepted.

Top-level allowlist derived from the core boundary:

- **Accepted** (algorithmic): `alpha`, `preprocessing`, `mask_generator`,
  `postsam2processing`, `clip`, `clip_routing`, `blip3`, `candidate_views`,
  `visualization`, `diagnostic_artifacts`.
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
normalization (default `0.6`). CLIP `labels` keys are routing identifiers in
document order; terminal class mapping comes from `visualization.labels`.

### Candidate-view policy

`candidate_views` is a typed request-local section with independent `clip` and
`blip3` children. CLIP defaults to and the API accepts only
`mode: raw_bbox_crop`, `context_fraction: 0.10`, `min_context_pixels: 0`, and
`max_context_pixels: 64`. The crop is a complete source-coordinate RGB
rectangle; its mask-derived bbox and half-up radius never mask, fill, dim,
blur, or otherwise alter pixels. Trusted CLI may explicitly use the separate
`mask_dilated` compatibility builder. BLIP3 defaults to
`mode: single_dilated_blur`, `context_fraction: 0.20`, context limits `0..64`,
`crop_extent_multiplier: 2.0`, `blur_sigma_fraction: 0.15`, enabled contour
fraction `0.02`, contour width limits `1..3`, and `contour_rgb: [255, 224, 0]`.
BLIP3 accepts only its new field set; the old `mode: mask_dilated`,
`outside_fill`, `context_intensity`, and `contour_width` fields are rejected.
Null, bool-as-number, non-finite, unknown, out-of-range, cross-field and
unsupported values are rejected without clamping. `clip.padding` remains an
unsupported service field.

For `L = max(raw_mask_width, raw_mask_height)`, BLIP3 reports
`raw_context_radius = ceil(context_fraction * L)` and
`effective_context_radius = min(max(raw_context_radius, min_context_pixels),
max_context_pixels)`. Support is the exact squared-Euclidean disk dilation
`D`; contour is `exact_euclidean_dilate(D, effective_contour_width) & ~D`.
`raw_contour_width = ceil(contour_fraction * L)` is bounded by the configured
1..3 limits. The crop dimensions are `ceil(crop_extent_multiplier * W/H)`;
raw and support bboxes are inclusive `xyxy`, while the array-slice crop bbox is
half-open `xyxy`. Gaussian sigma is
`min(max(blur_sigma_fraction * L, 2), 20)`. Candidate and question IDs are
one-based; the post-SAM2 `filtered_index` is zero-based. L3
`blip3_candidate_views` records one composition attempt per applicable
candidate, while debug records remain one-for-one with fixed-name lossless
model-input PNGs. Both contain only bounded numeric provenance, not image
pixels or client text. BLIP3 capacity is admitted after actual CLIP
labels/scores and before any QA call.

### Optional diagnostic artifact selection

The optional top-level `diagnostic_artifacts` section is a strict request-local
delivery selector:

```yaml
diagnostic_artifacts:
  stages: [sam2, clip, blip3, visualization]
  candidate_ids: null
  page: 1
  page_size: 48
```

`stages` is a unique subset of the four fixed stage tokens. `candidate_ids` is
null or a unique list of one-based source candidate IDs from 1 through 256;
the requested order is retained while the effective filter is sorted. `page`
is 1..65535 and `page_size` is 1..48. Pagination follows stage and candidate
selection in deterministic pipeline/name order. The section narrows eligible
L3 delivery and never enables a debug flag. At L0-L2 it remains valid but is
reported as not applied; lower levels do not execute optional diagnostics.

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
| max single/total raw artifacts | 32 / 128 MiB; optional L3 artifacts are greedily admitted and typed omissions are recorded | `SLAIF_ZAP_IT_MAX_SINGLE_ARTIFACT_BYTES`, `SLAIF_ZAP_IT_MAX_TOTAL_RAW_ARTIFACT_BYTES` |
| max RLE runs/object/response | 250 000 / 1 000 000 | `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_PER_OBJECT`, `SLAIF_ZAP_IT_MAX_MASK_RLE_RUNS_TOTAL` |
| max total response | 256 MiB | `SLAIF_ZAP_IT_MAX_RESPONSE_BYTES` |
| min available RAM / shm | 2 GiB / 64 MiB | `SLAIF_ZAP_IT_MIN_HOST_AVAILABLE_BYTES`, `SLAIF_ZAP_IT_MIN_SHM_FREE_BYTES` |
| request deadline | 120 s | `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS` |
| inference queue depth | 0 | `SLAIF_ZAP_IT_QUEUE_DEPTH` |
| Retry-After value | 5 s | `SLAIF_ZAP_IT_RETRY_AFTER_SECONDS` |

Encoded sizes are enforced while streaming (limit+1 pattern) before decode;
decoded width/height are checked from headers before pixel allocation; L3
annotated streams are admitted after rendering; their bytes are not reserved
before engine execution. Host RAM and `/dev/shm` floors are checked at
readiness and request admission.

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
| `resource_limit` | 413 | SAM2 field or estimated-work cap exceeded; details include sanitized alternatives |
| `response_too_large` | 413 | essential JSON/ZIP document still exceeds response cap after optional tail omission |
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
- Canny/Hough geometry and panoptic visualization remain explicitly unsupported
  service capabilities; `annotated-labelled` is the supported final-object
  labelled visualization. Optional `postsam2processing` impossibility geometry
  is an independent in-memory filter supported by this contract.
- `postsam2processing` accepts the canonical optional geometry rules
  (`min_area`, `max_area`, `min_width`, `max_width`, `min_height`,
  `max_height`, aspect-ratio bounds, and `allow_border_touching`). Legacy
  `maxsize`, `max_w`, and `max_h` are compatibility aliases with migration
  warnings. The unrelated batch-only `geometry` and `blip2` sections remain
  rejected.

## Objective 020 semantic contract

API CLIP labels are mappings of safe identifiers to one complete natural-language
prompt. `candidate_views.clip.mode` is always `raw_bbox_crop`; masked, filled,
dimmed, or padded views are not accepted by the API. `clip_routing.route_to_blip3`
uses OR conditions for top-1, top-k, score margin, minimum target score, and
uncertain winners, with deterministic reasons and a source-ID-ranked cap.
Complete vectors and routing diagnostics are L3-only; effective policy is
available in service metadata. The selected target rule asks BLIP3 once using
the existing delimited question and exact normalized true/false token mapping,
with an exact true match selecting `newcategory`, an exact false or unmatched
answer selecting configured `falsecategory`, and the mapping recorded.
L2 objects carry their own semantic evidence; JSON and ZIP metadata agree.
