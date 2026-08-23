# OAP Work Order — 002-a — `/v1/completions` API contract with fake engine

Objective `002-a`. Implement the first complete CPU-testable HTTP service
contract around the typed single-image core from Objective 001: FastAPI/
Pydantic/Uvicorn service code, strict multipart parsing, hostile-input
validation, monotonic verbosity levels, JSON/ZIP serialization, stable errors,
health/readiness behavior, bounded concurrency semantics and optional local
API-key plumbing using a fake/injected engine. Freeze the public
`/v1/completions` contract sufficiently for clients and later GPU activation,
but do not start a persistent live GPU service in this objective.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `002 / 002-a`
- Mode: `CREATE_NEW_PR`
- Objective 001 merged on remote `main`: squash merge commit
  `08fed7bacc83d6121e33d508fd79e0b6533345fc` (PR #45); post-merge main runs
  CI SUCCESS + CodeQL SUCCESS — prerequisite satisfied.
- Verified current default branch and 40-hex base SHA: `main` @
  `08fed7bacc83d6121e33d508fd79e0b6533345fc` (verified live via `git fetch`
  immediately before publication).
- Required new branch name: `oap/002-a-v1-completions-api-contract`, created
  explicitly from remote `main` at that SHA. The shared local clone may still
  be checked out on a finished objective branch; never reset/clean unrelated
  local state.
- Existing objective-002 PR: none (no open PRs after #45 merged).
- Required PR title: `Objective 002-a: /v1/completions multipart API contract
  (fake engine, CPU-only)`

## Verified current state (post-001 tree, evidence gathered live 2026-08-23)

- Canonical in-memory engine entry point and dependency-injection seam:
  `src.core.run_single_image(image_rgb: np.ndarray, config: CoreConfig, *,
  frame_id="image", segmenter_state=None, clip_state=None, blip3_state=None,
  dryrun=False, verbosity=1, device=None, log_print_func=None,
  artifact_sink: Optional[ArtifactSink]=None, stages: Optional[StageFunctions]=None,
  class_labels: Sequence[str]=()) -> SingleImageOutcome`. Stage callables are
  injectable via `StageFunctions`/`default_stage_functions()` (historical
  monkeypatch targets preserved); request state is fresh per call; only model
  holder state dicts thread through.
- Typed result/object/artifact interfaces and renderer behavior:
  `ObjectResult` (instance_id, label, area_px, bbox_xyxy, normalized_bbox,
  centroid_rc, predicted_iou, stability_score, clip_score, blip3_answer,
  geometry() -> None hook, serialized_metadata()), `StageStatus`, `Provenance`,
  `PipelineResult` (object_by_id, stage_status, serialized_records),
  `SingleImageOutcome`; renderers `render_yolo(objects, image_width,
  image_height)` producing deterministic five-field six-decimal text and
  `render_identity_png(objects, width, height)` producing real lossless uint16
  PNG bytes with background 0 / ids 1..N and larger-area overlap winner;
  sinks `MemoryArtifactSink`/`FilesystemArtifactSink` over validated logical
  names (`StoredArtifact`; traversal rejected); `MAX_IDENTITY_OBJECTS` guard;
  `CoreError`, `ArtifactSinkError`, `IdentityMaskOverflowError`.
- Normalized configuration representation and batch-only classification:
  `CoreConfig.from_mapping` normalizes trusted configs; `classify_config_fields`
  splits `ALGORITHMIC_TOP_LEVEL_FIELDS` versus `BATCH_ONLY_TOP_LEVEL_FIELDS`;
  `config_digest` provides a stable digest hook for envelopes/provenance.
- Current package extras/dependency policy and supported Python: runtime deps
  numpy/pillow/pyyaml only; dev extra build/coverage/pytest/pytest-cov/ruff;
  requires-python >=3.10,<3.13; wheel packages include `src.core`.
  **FastAPI/Pydantic/Uvicorn/python-multipart are NOT yet dependencies** — this
  objective adds them as an isolated service extra so CPU-only installs remain
  light.
- Current error/warning/status types reusable by service: typed core errors
  plus per-call warnings list and stage timings exposed on the outcome — map
  these onto stable HTTP error codes without leaking internals.
- Exact CPU CI/test commands and coverage baseline: `.venv/bin/pip install -e
  '.[dev]'`; `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=
  term-missing` (137 passed / 0 failed / 0 skipped; branch coverage 71%,
  gate fail_under=64); ruff format/check clean; wheel builds.
- Current docs/API target differences requiring reconciliation: docs/CORE.md
  documents the core seam but no HTTP surface exists yet; ARCHITECTURE-for-agents
  defines the target endpoint/verbosity/artifact/error law that this objective
  implements and documents (docs/API.md).

## Public request contract (frozen by this order)

```text
POST /v1/completions   Content-Type: multipart/form-data
```

Exactly one request = exactly one image, one YAML config, one result:

- `image`: required; JPEG/PNG/WebP only (frozen safe set);
- `config`: required UTF-8 YAML document/upload;
- `verbosity`: integer `0|1|2|3`, default `0`; textual aliases are NOT accepted
  in v1 (strict ints; reserved for later versioned relaxation);
- `response_format`: `json|zip`, default `json`;
- `model`: optional; the fixed service identifier is `zap-it-1`; any other
  value is rejected with a stable unsupported-model error;
- `stream`: omitted or `false` only; anything else is a stable rejection.

This is a ZAP-IT-specific image-pipeline endpoint using the conventional path;
do not claim drop-in OpenAI text-completions compatibility.

## Response semantics

Verbosity is monotonic in returned information, not computational cost:

- **L0**: completion envelope + YOLO text in `choices[0].text` + minimum safe
  metadata (model, created, request id, image dims, class mapping, config
  digest, verbosity, finish reason);
- **L1**: L0 + uint16 identity PNG artifact;
- **L2**: L1 + per-object metadata actually produced by enabled stages;
- **L3**: L2 + bounded full safe artifacts/stage statuses/warnings/timings/
  provenance already produced by the executed pipeline.

Lower levels must not trigger optional stages solely to enrich output;
disabled/unavailable stages are represented honestly, never fabricated.

JSON binary artifacts use a stable object: logical name, media type,
encoding `base64`, SHA-256, byte size, data. ZIP responses contain
`manifest.json`, `detections.yolo.txt`, `identity-mask.png` when applicable,
and level-gated artifacts with deterministic logical names; manifest hashes/
sizes/media types match actual bytes; ZIP assembly bounded in memory or a
service-owned `/dev/shm` workspace, never caller paths.

### Frozen strategic decisions (binding)

- Busy semantics: exactly one active inference slot; queue depth default 0
  (operator-tunable upward). An arriving request while the slot is full and the
  queue is exhausted returns HTTP **503** with stable code `service_busy` and a
  `Retry-After` header. Rationale: capacity unavailability, not client
  misbehavior; deterministic and testable.
- Error envelope shape (frozen): `{"error": {"code": "<stable_snake_code>",
  "message": "<sanitized>", "request_id": "<opaque>"}}` with appropriate HTTP
  status; required codes include at minimum: invalid_multipart, missing_part,
  duplicate_part, invalid_image, invalid_config, unsafe_config, unsupported_field,
  unsupported_verbosity, unsupported_format, unsupported_model, stream_unsupported,
  payload_too_large, image_too_large, service_busy, not_ready, timeout,
  cancelled, inference_failure, insufficient_memory, response_too_large.
  No stack traces, raw YAML/image bytes, host paths, secrets or environment data.
- Auth policy: strict-loopback deployments default to NO key. Setting operator
  env `SLAIF_ZAP_IT_API_KEY` enables required `Authorization: Bearer <key>`;
  comparison via constant-time primitive; keys never logged or echoed. This
  path must exist and be tested now so later LAN exposure cannot ship without it.
- Default limits (operator-overridable at startup, never per-request):
  max image upload 20 MiB; max config upload 256 KiB; max decoded pixels
  64,000,000; max total response 256 MiB; request deadline 120 s.
- Request ID: opaque server-generated token; echoed in envelope metadata and
  error envelope; no user filenames/content.

## Scope

1. Add service package boundaries (FastAPI/Pydantic/Uvicorn +
   python-multipart under a dedicated `[service]`/`[api]` extra) with
   transport/validation/lifecycle/error mapping separated from the engine.
2. Dependency injection/fake engine: deterministic fake outcome factory for all
   API tests; no CUDA/model/network; no production-code monkeypatch detours.
3. Strict multipart cardinality/parsing rejections per frozen codes.
4. Bounded upload reads (limit+1 pattern); encoded sizes enforced before decode;
   UTF-8 validation; filenames used for diagnostics only, never paths.
5. Image safety: bounded dimensions/pixels decode, decompression-bomb and
   malformed-codec protection, RGB normalization for the engine; tests cover
   oversized compressed and decoded cases without pathological fixtures.
6. Hostile YAML safety: `yaml.safe_load` plus byte/depth/collection/string/alias
   bounds and a typed allowlist of algorithm fields derived from
   `ALGORITHMIC_TOP_LEVEL_FIELDS`; reject/ignore-with-warning mapping decided
   per field class and documented; uploaded config can never control paths,
   URLs, commands, imports, Python symbols, devices, environment, credentials,
   service settings, model repos/revisions, cache roots, debug destinations or
   batch/video controls.
7. Legacy debug/path normalization: safe algorithmic debug options map to
   bounded logical artifacts via the sink when a level permits; legacy
   filesystem paths are never honored silently (reject or ignore-with-warning,
   explicit list).
8. Stable completion envelope: schema version, service/model identifier,
   request ID shape, one choice, YOLO `choices[0].text`, finish reason, image
   dimensions, class mapping, config digest, verbosity, artifact list; `usage`
   stays null (no invented token counts).
9. JSON/ZIP parity per the semantics section; response size bounded.
10. Stable sanitized errors per frozen envelope/codes.
11. Concurrency boundary per frozen busy semantics; tests prove overlapping
    fake requests never run inference concurrently and busy/queue behavior is
    bounded/deterministic.
12. Timeout/cancellation cleanup: deadlines enforced; service-owned artifacts/
    workspaces cleaned on success, validation failure, inference failure and
    cancellation; tests use isolated temp roots; documented production default
    remains `/dev/shm/slaif-zap-it` for later objectives.
13. `/healthz` process/event-loop health; `/readyz` delegates to injected
    readiness provider and reports not-ready honestly; no GPU init here.
14. Optional API-key plumbing per frozen auth policy, fully tested.
15. OpenAPI accuracy + deterministic curl/Python examples using fake/local
    semantics; explicit non-claim of live model readiness.
16. Comprehensive CPU contract tests: cardinality, media types, limits,
    hostile YAML, path/device/model attempts, every verbosity level, JSON/ZIP,
    empty detections, overlaps, hashes, busy/concurrency, timeout/error,
    readiness, auth, cleanup.

## Non-goals

- no actual SAM2/CLIP/BLIP3 loading or GPU inference;
- no persistent listener/service activation outside tests;
- no physical GPU allocation, UUID enforcement against real hardware or
  driver/CUDA setup;
- no systemd/Docker/Compose, firewall/VPN or LAN/public exposure;
- no video API;
- no arbitrary remote model selection/download;
- no asynchronous job queue/background persistence;
- no implementation of Objective 003+.

## Acceptance criteria

1. Importable FastAPI app exposes `/v1/completions`, `/healthz`, `/readyz`
   with documented schemas and injectable engine/readiness dependencies.
2. CPU TestClient/HTTPX tests exercise the complete routes without CUDA/network
   or model downloads.
3. Exactly-one-image/config cardinality and all request limits are enforced
   before expensive work.
4. Uploaded YAML cannot select host paths, network, devices, commands, code,
   credentials, model repositories/revisions or deployment settings.
5. L0–L3 monotonic, matching Objective 001 ordering/renderers; lower levels do
   not trigger optional stages solely for enrichment.
6. JSON and ZIP artifacts have correct bytes, logical names, hashes, sizes,
   media types; response size bounded.
7. Stable sanitized errors tested; no stack/raw content/path/secret leakage.
8. Concurrency tests prove at most one inference executes simultaneously;
   busy/queue behavior deterministic/bounded per frozen 503 decision.
9. Success, validation failure, inference failure, timeout and cancellation
   leave no workspace residue.
10. Health/readiness honest and distinguish process-up from engine-ready.
11. Optional auth constant-time/operator-controlled/tested; loopback default
    explicit.
12. OpenAPI/docs/examples agree with actual wire contract.
13. Canonical CPU suite, Ruff/package/coverage/CI and CodeQL green.
14. No listener remains after tests; no GPU process/state changes occur.
15. Correct branch/one PR/report-only SELF contract satisfied; coding never
    merges.

## Required verification (exact commands/states)

- Predecessor remote-main/CI state: `main` @ `08fed7ba…` CI+CodeQL SUCCESS
  (re-confirm at round start; report observed values)
- Canonical package/Ruff/static: `.venv/bin/ruff format --check . &&
  .venv/bin/ruff check .` and `.venv/bin/python -m build --wheel` — PASSED
- Full CPU pytest+coverage: `.venv/bin/pytest -q --cov=src --cov=modules
  --cov-report=term-missing` — all green incl. new API suite; counts/duration;
  64% gate held or honestly raised with measured value
- Dedicated API/TestClient contract suite — named, PASSED
- OpenAPI schema snapshot/validation — PASSED
- Hostile YAML/property/limit tests — PASSED
- JSON/ZIP artifact parity/hash tests — PASSED
- Concurrency/busy/timeout/cancel/cleanup tests — PASSED
- Optional auth tests — PASSED
- Listener scan before/after proving no persistent service (e.g. `ss -tlnp`
  diff scoped to the test user) — PASSED
- Read-only GPU before/after snapshot proving zero allocation
  (`nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` + compute-apps)
- GitHub checks: `static (format, lint, build)`, `tests (py3.10)`,
  `tests (py3.11)`, `tests (py3.12)`, `Analyze (python)`, `CodeQL` — all
  present and SUCCESS, none pending/failed/missing

## Documentation and provenance

Update docs/API.md (request/response schemas, limits, allowlist, error codes,
auth policy, busy semantics, examples, limitations) plus README navigation.
Explicitly state Objective 002 is a CPU/fake-engine contract and does not prove
live model/GPU readiness. Preserve provenance established in Objectives 000–001.

## Security/resource constraints

CPU-only. Do not download models, allocate either GPU, modify CUDA/driver/system
packages, open ports persistently, change firewall/VPN, install system services
or alter global OpenCode/provider credentials. Tests may bind ephemeral loopback
sockets managed by the test process. Request bytes/config/results stay in memory
or isolated ephemeral test storage and are cleaned.

## Deferred human adjudication

- Decision: `NONE`

Field names, error codes, the 503-vs-429 choice, queue-depth default, loopback
auth default and limit values are ordinary reversible engineering decisions
frozen by this order and fully testable locally; none crosses a security
boundary, exposes the public internet, touches real customer data or affects
release authority. The deployment/exposure gate remains a later-objective human
matter. If implementation exposes a genuinely material dilemma meeting all five
register conditions, report it as a candidate and continue all unambiguous safe
scope; strategic decides next round. Coding may not invent the entry.

## GitHub publication and report

Create exactly one objective branch `oap/002-a-v1-completions-api-contract`
from remote `main` @ `08fed7bacc83d6121e33d508fd79e0b6533345fc` and exactly one
PR titled as specified. Carry exact order and `oap/active` transcript in
implementation history. Push all implementation before the final report-only
SELF commit (literal implementation SHA parent, single report path, `Report
publication commit: SELF`), exercise/fix current-head CI, never merge. Send
response `OK` only after remote head/parent/bytes verification. Report exact
wire schema, limits, security policy, test counts/coverage, artifact
semantics, error table, concurrency/cleanup behavior, docs, skips/failures/
limitations and host safety evidence including both GPU snapshots and listener
scans.
