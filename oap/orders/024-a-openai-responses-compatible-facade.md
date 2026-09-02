# OAP Work Order 024-a — additive OpenAI Responses-compatible facade

## Objective

Add one deliberately narrow, stateless, non-streaming `POST /v1/responses`
facade over the existing ZAP-IT service. It is the future gateway/public
compatibility surface. It must execute the same typed, in-memory
SAM2 -> geometry filtering -> CLIP -> routing -> optional BLIP3 -> final
filtering/ordering pipeline as the existing service and may additionally adapt
the already-final result into one canonical labelled PNG.

Keep `POST /v1/completions` behaviorally unchanged. That route remains the
ZAP-IT-native multipart operator/research/debug API with verbosity, JSON/ZIP,
identity masks, exact mask RLE, candidate views, contact sheets and other
bounded diagnostics. It is not OpenAI Completions compatibility, is not the
SLAIF public contract, and is not intended to pass through
`slaif-api-gateway`.

This is one new numeric objective and one new PR. It is a thin transport/output
adapter, public projection, schema, documentation, test and local
qualification objective. It is not authority to change inference or deployment
semantics.

## Deferred human adjudication

- Decision: NONE

The human has explicitly selected the facade architecture and service-specific
meaning of the canonical `image_generation` wire item. The current service
remains on its authenticated private RFC1918 listener; this objective does not
cross a public/WAN, production, gateway, credential, or irreversible boundary.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Verified remote default branch: `main` at merge SHA
  `32812032781c5d7daf54d5b7586b3c01d3270c48`, the accepted merge of Objective
  023 PR #87.
- Remote-main CI run `33551883499` and CodeQL run `33551883504` are successful.
- All seven PR-head checks for Objective 023 were successful before merge.
- The local worktree is clean at Objective 023 product commit
  `9f573e3c5f2d2df54100360bc5d05561fbba8254`; `origin/main` contains it.
- Open PRs #79-#86 are Dependabot-only and unrelated. No product objective PR
  is open.
- Current immutable `oap/active` is `023-c`; Objective 023 is complete and
  merged.
- Create branch `oap/024-a-openai-responses-compatible-facade` from the exact
  verified `origin/main` SHA and exactly one PR titled
  `Objective 024: OpenAI Responses-compatible facade`.
- Do not amend an old branch/PR, merge, enable auto-merge, or alter Dependabot
  PRs.

Before mutation, refresh remote main/open PRs/checks, the clean worktree,
`CRITICAL.md`, active/order state, listener/service state, environment-file
mode/digest, tmpfs, and every assigned-GPU/process fact. Stop only for a real
authority, safety or protocol contradiction.

## Strategic architectural finding

Strategic inspected current HEAD. The requirement is implementable as a thin
facade:

- `src/service/app.py` already owns bounded transport, decoding, hostile YAML
  validation, readiness, the shared single-slot gate, CLIP prompt preflight,
  deadline/cancellation handling and `_run_engine_bounded`.
- `run_single_image` already returns one typed `SingleImageOutcome` with final
  ordered `ObjectResult` identities and metadata.
- `render_annotated_labelled` already deterministically renders the final
  objects without Detectron2 or user-controlled paths.
- `decode_image_safely`, `parse_hostile_config`, the runtime policy and service
  settings already provide the required image/config/model/resource boundary.

Factor one shared prepared-inference seam from the existing completion route so
both transports reach the same validation, readiness, gate and engine call.
Do not make one HTTP route call the other; do not duplicate the pipeline or
create a second YAML validator. Preserve the completion transport and serializer
bytes/behavior as the compatibility gate.

## Frozen official Responses wire evidence

Implement against the current official contract reviewed on 2026-09-02:

- Create Response: `POST /v1/responses`, official reference
  <https://developers.openai.com/api/reference/cli/resources/responses/methods/create>.
- File inputs: `input_file` with `filename` and inline `file_data` data URL,
  official guide <https://developers.openai.com/api/docs/guides/file-inputs>.
- Image input: `input_image.image_url` accepts a base64 data URL.
- Image tool declaration and output extraction:
  `tools=[{"type":"image_generation"}]`, then iterate `response.output`,
  select `item.type == "image_generation_call"`, and decode `item.result`,
  official guide <https://developers.openai.com/api/docs/guides/image-generation>.
- Current official Python SDK is `openai==3.7.0`, tag `v3.7.0`, commit
  `ab76ab5c64b8d19761ce838891acc80743cd944a` in
  <https://github.com/openai/openai-python>. Its generated `Response` model
  requires `id`, `created_at`, `model`, `object`, `output`,
  `parallel_tool_calls`, `tool_choice` and `tools`; `usage` is optional. Its
  image call item has `id`, `type`, `status`, and optional `result`.

There is no protocol contradiction. ZAP-IT deliberately assigns the standard
image-generation declaration the documented service-specific meaning "also
return the canonical final annotated ZAP visualization"; it does not claim to
invoke an image-generation model. Do not rename the tool or output item.

If official schema evidence changes materially before implementation begins,
record the exact source/version and report the contradiction before substituting
another protocol. Otherwise use the frozen contract above consistently.

## Exact supported request subset

Accept only `Content-Type: application/json`, no query parameters, and one JSON
object with these top-level fields:

- required `model`, exactly the existing operator-fixed service model ID
  `zap-it-1`;
- required `input`;
- optional `tools`, absent/empty or exactly one
  `{"type":"image_generation"}` declaration;
- optional `store`, absent or strict boolean `false`;
- optional `stream`, absent or strict boolean `false`;
- optional `background`, absent or strict boolean `false`.

Reject every other top-level field as unsupported, including explicit
`previous_response_id`, `conversation`, `instructions`, `prompt`, `metadata`,
`include`, token/text/reasoning controls, `tool_choice`, hosted-tool controls,
and arbitrary extensions. Presence is unsupported even when its value is null;
do not silently ignore it. Reject `store: true`, `stream: true`, and
`background: true` with the applicable existing typed error. Reject null,
numeric/string lookalikes and other wrong types as `invalid_config`.

For this first subset, `input` is exactly one ordinary user message:

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,..."
      },
      {
        "type": "input_file",
        "filename": "task.yaml",
        "file_data": "data:application/yaml;base64,..."
      }
    ]
  }
]
```

The message may omit `type` or set it to exactly `message`. Its role is exactly
`user`; its content list has exactly two parts in either order. There must be
exactly one image and exactly one YAML file. No input text, audio, assistant,
system/developer message, prior output item or item reference is supported.

The image part requires `type`, `detail: "auto"`, and `image_url`; it accepts
only strict base64 data URLs for `image/png`, `image/jpeg`, or `image/webp`.
Do not fetch URLs or accept `file_id`. Verify the declared MIME against the
decoded image format instead of trusting either alone.

The file part requires `type`, a safe filename ending in `.yaml` or `.yml`, and
`file_data`. Accept strict base64 data URLs only for a documented small set of
YAML/text MIME types: `application/yaml`, `application/x-yaml`, `text/yaml`,
`text/x-yaml`, and `text/plain`. Do not accept `file_id`, `file_url`, server
paths or percent-encoded data. The filename is metadata only and never a path;
require a bounded ASCII basename, reject separators, traversal, controls and
unsafe suffixes, and never use it in a path, artifact name, response ID, log or
model prompt.

Use strict base64 validation. Reject malformed, empty, non-base64 or wrong-MIME
data URLs before inference with a bounded path-specific client error.

## Input/resource/security boundary

Authenticate `/v1/responses` with the same fixed deployment bearer key and
constant-time comparison used by `/v1/completions`. Authentication must finish
before reading the request body. Preserve the existing key-optional strict-
loopback development mode; do not change the keyed deployment policy. Never
echo/log credentials or request bodies.

Stream the JSON request body under a hard encoded-size cap; do not call an
unbounded `request.body()`. Derive the Responses body cap from the existing
decoded image and config caps using exact base64 expansion plus a fixed bounded
JSON-envelope allowance. Expose the derived value in capabilities; do not add a
client override or silently reduce the existing decoded upload limits.

Enforce, before inference where applicable:

- existing image upload byte cap after base64 decode;
- existing config upload byte cap after base64 decode;
- existing safe image formats, decoded dimensions and pixel cap;
- existing hostile UTF-8 YAML structure/content/configuration validator;
- existing runtime/profile and CLIP prompt preflight validation;
- existing request resource, readiness, gate/queue, deadline/cancellation,
  object-count and response-size policy;
- one process/worker/active inference and the same resident model holders.

Use request-local data only in memory or the already-authorized `/dev/shm`
boundary. Do not persist the image, YAML, result or response. Do not introduce
URL fetching, users, RBAC, quotas, rate limiting, billing, TLS, gateway logic,
new model/device/path controls, or new service credentials.

## Shared inference seam and fixed public execution policy

Extract a small typed internal helper/data record that accepts already-bounded
image/config bytes plus request/deadline context and performs the common decode,
existing YAML validation, `CoreConfig` construction, runtime validation,
readiness, shared gate admission, CLIP preflight and one `_run_engine_bounded`
call. Both endpoints must use it.

The Responses route uses the existing validator at fixed service verbosity 2
and calls the engine with `verbosity=2`, `render_visualizations=False`, the same
safe artifact-name mode and a request-local bounded sink/ledger. This preserves
final object evidence while stripping all debug flags through the existing
policy and preventing private L3 visualizations/debug artifacts from being
created. Uploaded visualization label filtering remains part of normal
inference; uploaded diagnostic rendering requests do not become public output.

Do not change the effective inference configuration merely because
`image_generation` is or is not requested. The tool controls only the
post-inference output adapter. Requests with and without it must produce the
same underlying final objects/config digest.

Keep both surfaces on the one existing `InferenceGate` and executor. Add a
cross-surface concurrency regression proving that a request on either route
occupies the same single inference slot; never construct a second engine/model
registry/gate.

## Deterministic public inference projection

Define and validate a dedicated typed public JSON projection, versioned exactly
as `zap-it.public.v1`. Serialize it to the assistant output text with
`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)`.

The projection is exactly:

- `schema_version`;
- `model`;
- deterministic effective `config_digest`;
- original `image` width and height;
- final `class_mapping`;
- `sam2` with requested/effective values, per-field source, selected profile,
  estimated prompt count, estimated mask-prediction count, actual candidate
  count and deterministic resource warnings;
- complete canonical `candidate_counts` after pipeline stages;
- effective `candidate_views` for CLIP and BLIP3;
- effective `clip_routing` configuration;
- bounded CLIP prompt-count metadata when configured;
- final ordered `objects` using the existing object-record semantics without
  `mask_rle`: instance ID, stable source candidate ID, filtered index, class ID,
  final label, bbox, normalized bbox, area, centroid, SAM scores, complete
  per-class CLIP scores/routing evidence, BLIP3 configured/effective question,
  raw/normalized answer/mapping/final label, geometry and bounded object
  warnings when present;
- deterministic configuration/result warnings, bounded and sanitized.

Factor/reuse the existing object-record builder so private completion object
records and public object records cannot drift. Preserve its key/value
semantics and private output ordering. Do not put any of these in the public
projection: request/response/message/tool-call IDs, timestamps, wall-clock
timings, package/runtime provenance, host/GPU/device facts, paths, operator
limit values, masks/RLE, identity images, raw candidates, geometry rejection
lists, candidate-view records, debug-artifact metadata/data, ZIP members or
contact sheets.

The projection string and optional PNG are deterministic for identical decoded
image, effective YAML, fixed model outputs and renderer version. The outer
Responses IDs and timestamps are protocol metadata and are intentionally unique,
not part of deterministic-content comparisons.

## Exact successful Responses envelope

Return a canonical completed response object that current `openai==3.7.0`
parses normally. It must contain:

```json
{
  "id": "resp_<safe-unique-id>",
  "object": "response",
  "created_at": 0.0,
  "completed_at": 0.0,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "model": "zap-it-1",
  "output": [
    {
      "id": "msg_<safe-unique-id>",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {"type": "output_text", "text": "<canonical JSON>", "annotations": []}
      ]
    }
  ],
  "parallel_tool_calls": false,
  "tool_choice": "none",
  "tools": []
}
```

The example zero timestamps only show shape; use bounded Unix seconds at
creation/completion. Use independent cryptographically random bounded IDs with
the shown canonical prefixes. Preserve ZAP instance/source candidate IDs inside
the projection; do not manufacture replacement inference-object identities.

Omit `usage` entirely. It is optional in the current official SDK model, and
SAM2/CLIP/BLIP3 work must never be misreported as LLM token usage. Do not add a
gateway quota/accounting convention.

## Canonical image-generation output item

When and only when the request contains exactly
`tools: [{"type":"image_generation"}]`, append exactly one output item after
the assistant message:

```json
{
  "id": "ig_<safe-unique-id>",
  "type": "image_generation_call",
  "status": "completed",
  "result": "<raw PNG bytes as standard base64, not a data URL>"
}
```

This does not call a generative model. After inference, call the existing
`modules.visualizer.render_annotated_labelled` on the original decoded RGB image
and the final ordered `ObjectResult` sequence using its canonical defaults,
`alpha=0.5` and `show_confidence=false`. The output therefore contains
deterministic masks, final class labels and final instance IDs. It is independent
of, and does not mutate, uploaded private visualization streams.

Factor/reuse the current deterministic PNG encoder so the public bytes are
exactly those produced by the existing renderer/encoder pair. Do not duplicate
renderer logic. Check raw render estimate against the existing total-raw budget,
encoded PNG against the existing single-artifact cap, object count against the
existing maximum and the complete JSON/base64 envelope against
`max_response_bytes`. A requested public image is essential, not a diagnostic
tail; return typed `response_too_large` rather than silently omitting it.

When the tool is absent/empty, render no public image and return no
`image_generation_call`. Never return identity masks, unlabelled overlays,
private configured visualization streams, candidate views, contact sheets,
debug images, ZIPs or arbitrary artifact descriptors through this facade.

## OpenAI-shaped error adapter

Catch route-local failures and return exactly this bounded shape without
changing the existing global/private handlers:

```json
{
  "error": {
    "message": "sanitized bounded message",
    "type": "invalid_request_error",
    "param": "bounded.request.path or null",
    "code": "existing_typed_service_code"
  }
}
```

Use `authentication_error` for 401, `invalid_request_error` for client/input/
capacity/response-size 4xx failures, and `server_error` for readiness/busy/
timeout/inference/internal 5xx-style failures. Preserve existing HTTP status,
typed code and safe `Retry-After` where applicable. Put the safe request ID in
`x-request-id`, not in a noncanonical error-body field. Do not expose raw YAML,
base64, filenames beyond a bounded field path, prompts/answers, model internals,
paths, credentials, stack traces or `ServiceError.details`.

At minimum keep these distinct by code/status/param where applicable:
malformed JSON/envelope, missing/duplicate image, missing/duplicate config,
unsafe filename, unsupported source/content/MIME, invalid image, invalid YAML,
unsupported model/state/stream/tool, payload/image/resource/response limits,
service busy/not ready, timeout/cancel, inference failure and internal failure.
No prompt/config/input error may become HTTP 500.

Because the route accepts a raw `Request` for bounded streaming, ensure
validation and authentication errors are shaped by this adapter and do not fall
through to the existing native completion envelope. Conversely, do not change a
single `/v1/completions` error field.

## Typed schemas, OpenAPI and capabilities

Add strict Pydantic/OpenAPI models for the accepted request shape, response
message/text and image-call union, public projection, and OpenAI error envelope.
Use `extra="forbid"`, bounded strings/lists/maps and finite-number constraints.
The runtime parser may remain manual where necessary for preallocation/body
bounds, but its accepted/rejected contract must agree with the generated schema.

Extend authenticated `/v1/capabilities` with typed, deterministic API-surface
metadata that clearly states:

- `/v1/completions`: native/private multipart research/debug surface, not
  OpenAI Completions and not gateway-facing;
- `/v1/responses`: OpenAI Responses-compatible narrow facade and future gateway
  surface;
- fixed model ID and endpoint;
- stateless, non-streaming, non-background, `store=false` subset;
- exact input cardinality, accepted inline data sources/MIME types and safe
  filename rule;
- only supported tool and its ZAP-specific meaning;
- message/output-text and optional image-call output types;
- public projection version and included/excluded evidence;
- omission of token usage;
- decoded and derived encoded request/response bounds;
- fixed bearer authentication;
- current gateway qualification status: not yet end-to-end qualified because
  `slaif-api-gateway` lacks its canonical Responses multimodal/image-generation
  path.

Do not advertise URL/file-ID support, streaming, state, persistence, hosted
tools or completed gateway integration.

## Metrics

Count public successes/errors through the existing finite-cardinality request,
duration, inference, serialization, response-byte, object and artifact metrics.
Do not relabel a Responses success as a private completion. Add one finite
Responses success counter/method if needed, with at most a fixed
image-generation-present boolean dimension. Never use model, label, prompt,
filename, response ID or user input as a metric label. Preserve every existing
completion metric name and meaning.

## Official SDK dependency and qualification tool

Add exact `openai==3.7.0` only to the development/test extra. Server/runtime code
must not import or depend on the SDK. Keep runtime deployment dependencies
unchanged.

Add a bounded operator qualification script using only the official client:

```python
client = OpenAI(base_url="http://HOST:PORT/v1", api_key=key)
response = client.responses.create(...)
public_result = json.loads(response.output_text)
for item in response.output:
    if item.type == "image_generation_call":
        png_bytes = base64.b64decode(item.result, validate=True)
```

It must build one deterministic small RGB PNG and safe YAML in memory (or accept
explicit repository-owned paths), convert both to canonical inline data URLs,
request the image tool, validate the typed SDK object, projection, one image
item and PNG decode, and print only bounded statuses/counts/sizes/hashes/timing.
Read the key only from `SLAIF_ZAP_IT_API_KEY`; never accept it on a command line,
print it, log bodies/prompts/answers, or write request data. An optional output
PNG must be written only to an explicit operator path with mode 0600; the live
order uses a mode-0700 `/dev/shm` evidence directory.

The CPU contract test must exercise official SDK serialization and generated
object parsing, not a home-grown Responses client/parser. An in-process ASGI
transport adapter is acceptable for CPU/fake tests only if the returned object
is the official SDK `Response` type and normal `output_text`/typed item access is
used. The mandatory live qualification below uses real HTTP.

## Documentation

Update at least `README.md`, `ARCHITECTURE.md`, `src/service/__init__.py`,
`docs/API.md`, `docs/SERVICE-DATASHEET.md`, `docs/RUNBOOK.md`,
`docs/OUTPUT-PARITY.md`, generated OpenAPI/capabilities descriptions and the
documentation-integrity checker/tests. Add a focused Responses facade document
if that keeps the complete field/error examples readable.

Documentation must unambiguously distinguish the two endpoints and explain:

- exact supported request JSON and data URL construction;
- safe filename/MIME/body/resource limits;
- public projection schema and determinism boundary;
- optional canonical annotated PNG and fixed renderer policy;
- all unsupported state/tool/source behaviors and explicit errors;
- one fixed deployment bearer credential;
- no token usage accounting;
- no request persistence and no URL fetching;
- the official SDK usage example and qualification command;
- the private debug artifacts available only from `/v1/completions`;
- `slaif-api-gateway` is unchanged and does not yet provide end-to-end
  Responses image-generation/multimodal routing; that is a later cross-repo
  qualification.

Do not use language suggesting this objective authorizes general-public/WAN
exposure, production readiness, per-user controls, gateway completion or model
accuracy.

## Required automated tests

Add focused pure/parser/projection/route/SDK tests, while retaining every
existing test unchanged and green. Prove at minimum:

1. A canonical request with one inline image and one inline YAML file reaches
   the same `FakeEngine`/shared path exactly once with fixed verbosity 2,
   visualization rendering false and the same effective config digest.
2. Public final objects equal the existing private L2 object-record semantics
   for the same decoded input/config/outcome; stable instance/source/filtered
   IDs, complete class scores/routing and BLIP3 evidence survive.
3. The canonical output-text JSON is byte-identical across repeated projection
   builds from the same outcome and contains no nonfinite values.
4. Outer response/message/image IDs are correctly prefixed, bounded and unique;
   normal protocol timestamps/status fields parse.
5. No-tool requests have exactly one assistant message and no image call; tool
   requests have that message plus exactly one completed
   `image_generation_call`.
6. Decoded `item.result` is a valid PNG. Its bytes and RGB pixels exactly equal
   a direct existing `render_annotated_labelled(..., alpha=0.5,
   show_confidence=False)` plus the shared encoder for the same final outcome.
7. Toggling the tool changes only outer output/image bytes, not effective
   config, engine call or public structured result.
8. A YAML requesting debug/private visualization proves the Responses path
   strips debug through the existing validator, calls no engine visualization,
   and exposes no RLE, identity mask, candidate view, contact sheet, debug
   artifact, ZIP or private artifact metadata.
9. Existing `/v1/completions` request/response/error behavior and deterministic
   renderer/ZIP artifact bytes remain unchanged. All pre-existing completion
   tests run unmodified.
10. The two endpoints share one gate/executor: cross-surface overlap permits
    only the configured one active request/queue policy and never overlaps two
    engine calls.
11. Missing, duplicate and multiple image/config cases fail before inference.
12. Malformed JSON, scalar/string input, unknown fields, unsafe filename,
    invalid/empty base64, MIME mismatch, corrupt image, invalid YAML, wrong
    model and wrong types produce precise OpenAI-shaped errors and zero engine
    calls.
13. `stream:true`, `store:true`, `background:true`, any previous response or
    conversation, URL/file-ID sources, input text, extra message/items, and
    every unsupported tool (function/custom/MCP/web/file/code interpreter)
    fail explicitly rather than being ignored.
14. Encoded body, decoded image/config, dimensions/pixels, SAM2/BLIP3 capacity,
    object, single PNG, raw artifact and full response limits use the existing
    typed status/code boundaries. Tiny-limit tests prove no large allocation or
    inference occurs when rejection can happen earlier.
15. Service busy, not-ready, timeout/cancel and inference failures keep distinct
    sanitized OpenAI envelopes; key/raw input/internal exception text is absent.
16. Configured bearer authentication is required before body processing; wrong
    keys are never echoed. Health behavior and `/v1/completions` auth remain
    unchanged.
17. Requests create no files and retain no response state; no retrieve/update/
    delete/list Responses endpoints appear.
18. OpenAPI exactly documents the supported subset/output union/errors and
    forbids unsupported fields. Capabilities are authenticated, deterministic
    and agree with runtime constants/ranges/defaults.
19. The current official `openai==3.7.0` client can create the Response, obtain
    `response.output_text`, parse it as JSON, iterate typed `response.output`,
    recognize `image_generation_call`, retrieve/decode `item.result`, and open
    the PNG without a custom response parser.
20. Fake/CPU tests pass on Python 3.10, 3.11 and 3.12; release artifacts and
    installed-wheel smoke remain clean. The SDK remains dev-only.

Use self-contained synthetic arrays/images for deterministic unit tests. Do not
add a generated image or external photograph merely to test the facade.

## Local live qualification and service continuity

Current independently refreshed host facts are:

- host `hinton2`;
- `zap-it-lan.service` active, PID 767111, restart count 0, started
  2026-09-01 21:26:29 CEST;
- exact listener `10.8.132.76:17891`, never wildcard;
- assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24576 MiB;
- the service is the only compute process on that assigned device, using about
  11122 MiB; every unassigned device/workload remains protected;
- `/dev/shm` is 12 GiB with about 9.7 GiB free;
- environment file remains outside Git, mode 0600, and the key was not
  disclosed;
- current keyed health/readiness/capabilities checks are 200 and unauthenticated
  capabilities is 401.

After all CPU tests and the PR product commit are ready, reverify every device,
UUID/PCI/name/VRAM/process, driver/CUDA/PyTorch, tmpfs, service/env file, exact
listener and port before one controlled user-service restart. Keep
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, expose only physical index 0, and require logical
`cuda:0`; do not touch any other GPU/process, system service, network, firewall,
driver, cache or credential.

Use the existing fixed key without printing it. Install the new dev-only SDK in
the non-service client environment as needed; do not add it to model runtime.
Create one mode-0700 evidence directory under `/dev/shm` and retain only bounded
mode-0600 response summary/PNG/hash evidence there.

Perform over real HTTP:

1. health 200, readiness 200, unauthenticated Responses 401 with OpenAI-shaped
   error, authenticated capabilities 200 and exact non-wildcard listener;
2. the official-SDK qualification with the deterministic small synthetic image
   and safe low-work SAM2+CLIP YAML, requesting image generation;
3. SDK `Response` type/model/status, parsed `output_text`, final object/count
   summary, exactly one image call, strict base64 decode, PNG magic/dimensions,
   byte count and SHA-256;
4. one no-tool request may use CPU/fake proof instead of a second costly live
   inference unless live evidence finds a protocol-only uncertainty;
5. one existing authenticated `/v1/completions` smoke after restart, proving its
   native envelope remains available;
6. post-request service PID/listener/readiness, assigned GPU UUID/process/memory,
   tmpfs request-root cleanliness, logs free of tracebacks/secrets/input data,
   and no restart loop.

Leave the newest service code running on the same authorized private LAN
address/key/GPU after successful qualification. If qualification fails, preserve
bounded evidence, fix within this objective, restart only when needed, and never
weaken the contract or alter inference parameters to force success.

This is ZAP-IT-side qualification only. Do not access or modify
`slaif-api-gateway`, and do not claim the absent
SDK -> gateway -> ZAP-IT image-output path is qualified.

## Required commands/evidence

At minimum report exact commands and outcomes for:

- `ruff format --check .`
- `ruff check .`
- `python scripts/check_documentation.py`
- focused Objective 024 tests
- full `pytest -q --cov=src --cov=modules --cov-report=term-missing`
- `python -m build --wheel --sdist`
- release artifact verification/secret scans/twine checks already required by
  project CI
- official SDK CPU contract test and exact installed SDK version/source
- real keyed live SDK qualification and private completion smoke
- Git status/diff/stat, commit lineage, PR URL/head SHA, all required checks and
  changed-file inventory.

For the report, include the public projection SHA-256 and annotated PNG SHA-256
but not their base64/body contents, key, YAML text, prompt/answer text, paths to
model caches or other private operator data.

## Non-goals

- no change to SAM2 parameters/generation, geometry filters, CLIP crops/prompts/
  scoring/routing, BLIP3 views/questions/answers/generation, final filtering,
  ordering or existing visualization algorithms;
- no change to model IDs/revisions, holders, residency, GPU policy, dtype,
  generation settings, caches or artifact destinations;
- no `/v1/chat/completions`, OpenAI Completions emulation, streaming/SSE,
  background jobs, state/store/retrieve/list/delete, conversations or previous
  responses;
- no function/custom/MCP/web/file/code-interpreter or arbitrary hosted tools;
- no file upload endpoint, file IDs, server URL fetching or gateway bypass;
- no public diagnostic images/archives/RLE/masks;
- no token-cost fiction, quota, rate limit, billing, users, RBAC or multi-key
  design;
- no gateway, TLS, WAN/public-bind, firewall, system service, driver or unrelated
  dependency modernization;
- no edits in another repository and no model-accuracy tuning.

## Merge gate and report

Report every changed file, request/config migration notes, test commands/results,
SDK version and parsing evidence, live service facts, response/PNG hashes, PR
head/commits/checks and any retained compatibility behavior. Explicitly answer
the strongest reason not to merge: a superficially OpenAI-shaped route could
silently diverge from official wire semantics or from the native pipeline while
leaking private diagnostics. The answer requires official SDK object parsing,
one shared engine seam, exact renderer-byte proof, explicit rejection tests,
private completion regressions and complete documentation—not HTTP 200 alone.

Coding must not merge. Strategic will independently inspect the actual diff,
commit lineage, response schemas, tests, CI, official SDK evidence, live service,
listener/GPU/resource facts, logs and report. Merge only when every required
check is present/successful with none missing, pending or failed, the diff is
bounded to this objective, `/v1/completions` remains unchanged, no credential or
request data is present, and no unrecorded critical dilemma exists.

