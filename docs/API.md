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
holders on logical `cuda:0`. No persistent listener is started by the package
factory.

## Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/v1/completions` | POST | one image + one YAML config -> one result |
| `/healthz` | GET | process/event-loop health (always unauthenticated) |
| `/readyz` | GET | engine readiness via injected provider; honest `not_ready` |
| `/metrics` | GET | process-local finite-cardinality Prometheus text |

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
  + minimal metadata (request id, image dims, class mapping, config digest).
- **L1**: L0 + lossless uint16 identity PNG artifact (`identity-mask.png`,
  background `0`, instance ids `1..N`). Overlaps use the larger-area winner;
  if that would fully occlude an object, the service reserves a deterministic
  source pixel so the PNG IDs remain bijective with YOLO/object records.
- **L2**: L1 + per-object records containing only fields actually produced
  (bbox pixel+normalized, area, centroid, SAM quality, CLIP score, BLIP3
  answer when present, geometry hook when present).
- **L3**: L2 + stage statuses, candidate counts, timings, provenance,
  aggregate warnings, bounded annotated/debug artifacts, and one exact
  per-object uncompressed column-major COCO-style mask RLE.

Lower levels never trigger extra optional stages solely to enrich output.
Configured filesystem-style debug flags are honored only at verbosity 3
(where they map to bounded logical artifacts in memory); below that they are
stripped from the effective config with an explicit warning.

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

## Configuration policy (hostile uploads)

Parsing uses `yaml.safe_load` semantics behind bounds enforced during
composition: max depth 16, max 10 000 nodes, max 512 entries per collection,
max 16 384 characters per scalar, zero aliases/anchors accepted.

Top-level allowlist derived from the core boundary:

- **Accepted** (algorithmic): `alpha`, `preprocessing`, `mask_generator`,
  `postsam2processing`, `clip`, `blip3`, `visualization`.
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
  capabilities; geometry activation requires a separate scientific-stage order.
- `geometry`/`blip2` config sections are rejected until the core consumes them.
