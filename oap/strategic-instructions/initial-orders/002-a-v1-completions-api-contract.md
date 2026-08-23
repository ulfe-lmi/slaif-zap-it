# OAP Work Order — 002-a — `/v1/completions` API contract with fake engine

> DRAFT UNTIL Objective 001 is merged and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** this draft is preloaded human engineering intent. Strategic may refine exact implementation details only where verified post-001 evidence requires it; preserve the intended API boundary, security posture, monotonic verbosity and no-live-GPU constraint.

## Objective

Implement the first complete CPU-testable HTTP service contract around the typed
single-image core from Objective 001. Add FastAPI/Pydantic/Uvicorn service code,
strict multipart parsing, hostile-input validation, monotonic verbosity levels,
JSON/ZIP serialization, stable errors, health/readiness behavior, bounded
concurrency semantics and optional local API-key plumbing using a fake/injected
engine. Freeze the public `/v1/completions` contract sufficiently for clients and
later GPU activation, but do not start a persistent live GPU service in this
objective.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `002 / 002-a`
- Mode: `CREATE_NEW_PR`
- Objective 001 merged on remote `main`, merge SHA and checks: VERIFY:
- Verified current default branch and 40-hex base SHA: VERIFY:
- Required new branch name: VERIFY:
- Existing objective-002 PR: N/A after strategic confirms none: VERIFY:
- Required PR title: VERIFY:

Do not activate until Objective 001's in-memory core/result/renderers are merged
and green.

## Verified current state

Strategic must replace with post-001 evidence:

- canonical in-memory engine entry point and dependency-injection seam: VERIFY:
- typed result/object/artifact interfaces and exact renderer behavior: VERIFY:
- normalized configuration representation and batch-only/path fields: VERIFY:
- current package extras/dependency policy and supported Python: VERIFY:
- current error/warning/status types reusable by service: VERIFY:
- exact CPU CI/test commands and coverage baseline: VERIFY:
- current docs/API target differences requiring reconciliation: VERIFY:

## Public request contract

Canonical route:

```text
POST /v1/completions
Content-Type: multipart/form-data
```

Exactly one request means exactly one image, one YAML config and one result.
Canonical fields:

- `image`: required upload; initially JPEG/PNG/WebP only unless strategic evidence
  justifies a narrower safe set;
- `config`: required UTF-8 YAML document/upload;
- `verbosity`: integer `0..3`, default may be 0 if documented consistently;
- `response_format`: `json|zip`, default `json`;
- `model`: optional fixed service identifier; arbitrary repositories/revisions are
  forbidden;
- `stream`: omitted or `false` only; true is rejected with a stable error.

This is a ZAP-IT-specific image-pipeline endpoint using the requested path. Do
not claim drop-in OpenAI text-completions request compatibility.

## Required response semantics

Verbosity is monotonic in returned information, not in computational cost:

- **L0**: completion envelope + YOLO text + minimum safe metadata;
- **L1**: L0 + uint16 identity PNG artifact;
- **L2**: L1 + per-object metadata actually produced by enabled stages;
- **L3**: L2 + bounded full safe artifacts/provenance/timings/warnings available
  from the already-executed pipeline.

A lower response level must not trigger an expensive optional pipeline stage only
to populate the response. A disabled/unavailable stage is represented honestly,
never by fabricated defaults.

JSON binary artifacts use a stable object containing logical name, media type,
encoding (`base64`), SHA-256, byte size and data. ZIP responses contain a stable
`manifest.json`, `detections.yolo.txt`, `identity-mask.png` when applicable, and
level-gated artifacts with deterministic logical names. ZIP assembly must be
bounded and may use memory or a service-owned `/dev/shm` workspace; never a caller
path.

## Scope

1. **Add service package boundaries** using FastAPI/Pydantic/Uvicorn unless the
   post-001 repository proves another already-adopted stack is superior. Keep
   transport, validation, lifecycle and error mapping separate from the engine.
2. **Dependency injection/fake engine.** API tests must run without CUDA, model
   downloads or network by injecting a deterministic fake engine/result factory.
   Do not monkeypatch production code into a fundamentally different path solely
   for tests.
3. **Strict multipart cardinality and parsing.** Reject missing/duplicate image or
   config parts, malformed multipart, unsupported field names where policy is
   strict, unsupported `stream`, invalid verbosity/format/model and invalid
   content types with stable codes.
4. **Bounded upload reads.** Read at most configured limit+1 bytes. Enforce encoded
   image/config sizes before expensive parsing; validate UTF-8 and normalized
   filenames only for diagnostics, never for paths.
5. **Image safety.** Decode with bounded dimensions/pixels, protect against
   decompression bombs and malformed codecs, normalize to RGB as the engine
   expects, and reject unsupported/ambiguous image inputs. Tests cover oversized
   compressed and decoded dimensions without allocating pathological fixtures.
6. **Hostile YAML safety.** Use `yaml.safe_load` or a stricter safe mechanism.
   Enforce byte/depth/collection/string/alias bounds and a typed allowlist of
   supported API-safe algorithm fields. Uploaded config must not control input or
   output paths, URLs, commands, imports, Python symbols, devices, environment,
   credentials, service settings, model repositories/revisions, cache roots,
   arbitrary debug destinations or batch directory/video controls.
7. **Normalize legacy debug/path semantics.** Explicitly reject, ignore with a
   documented warning, or map safe algorithmic debug options to bounded logical
   artifacts. Never silently honor a legacy filesystem path from uploaded YAML.
8. **Stable completion envelope.** Freeze schema version, service/model identifier,
   request ID shape, one choice, `choices[0].text` YOLO content, finish reason,
   image dimensions, class mapping, config digest, verbosity and artifact list.
   `usage` remains null unless a meaningful non-token resource schema is later
   versioned; do not invent token counts.
9. **JSON and ZIP parity.** Equivalent logical information at the same verbosity
   must be represented consistently across formats. Manifest hashes/sizes/media
   types must match actual bytes.
10. **Stable sanitized errors.** Define versioned/sufficiently stable codes for at
    least malformed multipart, invalid image, invalid/unsafe YAML, unsupported
    field/level/format/model/stream, too large, busy, not ready, timeout/cancel,
    inference failure, insufficient memory/shared memory and response too large.
    No raw YAML/image, stack trace, secret, environment or host path.
11. **Concurrency boundary.** Establish one active inference slot and deterministic
    busy/queue behavior suitable for one future GPU request. Tests prove two
    overlapping fake requests do not run inference concurrently and that queue or
    rejection semantics are bounded. Exact `429` vs `503` choice is strategic and
    should be frozen/documented here.
12. **Timeout/cancellation cleanup.** Define request deadlines and ensure service-
    owned artifacts/workspaces clean up on success, validation error, inference
    error and cancellation. CPU tests may use temporary `/dev/shm`-like roots or
    isolated temp fixtures; production default remains `/dev/shm/slaif-zap-it`.
13. **Health/readiness.** `/healthz` reports process/event-loop health without
    claiming model readiness. `/readyz` delegates to an injected readiness/device
    provider and returns not-ready honestly. No GPU initialization is required in
    this objective.
14. **Optional API-key plumbing.** Implement operator-configured bearer/API-key
    authentication if it can remain simple and testable. Loopback-only operation
    may default to no key, but the code path must be ready before any later LAN
    exposure. Compare secrets in constant time and never log them.
15. **OpenAPI and client examples.** Ensure generated OpenAPI accurately describes
    multipart fields and response types where representable. Add deterministic
    curl/Python examples using fake/local semantics; no claim of live model
    readiness.
16. **Comprehensive CPU contract tests.** Cover cardinality, media types, limits,
    hostile YAML, path/device/model attempts, every verbosity level, JSON/ZIP,
    no detections, overlaps, binary hashes, busy/concurrency, timeout/error,
    readiness, optional auth and cleanup.

## Non-goals

- no actual SAM2/CLIP/BLIP3 loading or GPU inference;
- no persistent listener/service activation outside tests;
- no physical GPU allocation, UUID enforcement test against real hardware or
  driver/CUDA setup;
- no systemd/Docker/Compose, firewall/VPN or LAN/public exposure;
- no video API;
- no arbitrary remote model selection/download;
- no asynchronous job queue/background persistence;
- no implementation of Objective 003+.

## Acceptance criteria

1. The importable FastAPI application exposes `/v1/completions`, `/healthz` and
   `/readyz` with documented schemas and injectable engine/readiness dependencies.
2. CPU TestClient/HTTPX tests exercise the complete route without CUDA/network or
   model downloads.
3. Exactly-one-image/config cardinality and all request limits are enforced before
   expensive inference.
4. Uploaded YAML cannot select host paths, network, devices, commands, code,
   credentials, model repositories/revisions or deployment settings.
5. L0–L3 are monotonic and match Objective 001 object ordering/renderers. Lower
   levels do not trigger optional stages solely for output enrichment.
6. JSON and ZIP artifacts have correct bytes, logical names, hashes, sizes and
   media types; response size is bounded.
7. Stable sanitized errors are tested and reveal no stack/raw content/path/secret.
8. Concurrency tests prove at most one inference call executes simultaneously and
   busy/queue behavior is deterministic/bounded.
9. Success, validation failure, inference failure, timeout and cancellation leave
   no request workspace residue.
10. Health/readiness semantics are honest and distinguish process-up from engine-
    ready.
11. Optional auth, if enabled, is constant-time, operator-controlled and fully
    tested; loopback default policy is explicit.
12. OpenAPI/docs/examples agree with the actual wire contract.
13. Canonical CPU suite, Ruff/package/coverage/CI and CodeQL are green.
14. No listener remains after tests and no GPU process/state changes occur.
15. Correct branch/one PR/report-only SELF contract is satisfied; coding never
    merges.

## Required verification

- predecessor remote-main/CI state: VERIFY:
- canonical package/Ruff/static checks: VERIFY:
- full CPU pytest+coverage: VERIFY:
- dedicated API/TestClient contract suite: VERIFY:
- OpenAPI schema snapshot/validation: VERIFY:
- hostile YAML/property/limit tests: VERIFY:
- JSON/ZIP artifact parity/hash tests: VERIFY:
- concurrency/busy/timeout/cancel/cleanup tests: VERIFY:
- optional auth tests: VERIFY:
- listener scan before/after proving no persistent service: VERIFY:
- read-only GPU before/after snapshot proving zero allocation: VERIFY:
- GitHub CI/CodeQL checks: VERIFY:

## Documentation and provenance

Update API reference, security model, configuration allowlist, error codes,
OpenAPI/client examples and limitations. Explicitly state that Objective 002 is a
CPU/fake-engine contract and does not prove live model/GPU readiness.

## Security/resource constraints

This objective is CPU-only. Do not download models, allocate GPU0/GPU1, modify
CUDA/driver/system packages, open ports persistently, change firewall/VPN, install
system services or alter global OpenCode/provider credentials. Tests may bind an
ephemeral loopback socket managed by the test process. Request bytes/config/results
must remain memory or isolated ephemeral test storage and be cleaned.

## Deferred human adjudication

- Decision: `NONE`
- Exact field names, error codes, queue-vs-busy choice and loopback auth default are
  strategic decisions to make and test, not automatic CRITICAL entries.
- If a verified trust/security dilemma satisfies all five register conditions,
  strategic must decide provisionally and replace this section with exact
  `APPEND CRIT-NNNN` bytes before activation.

## GitHub publication and report

Create one objective-002 branch/PR from verified remote `main`, carry exact order
and active transcript, push all implementation before the final report-only SELF
commit, independently exercise/fix current-head CI and never merge. Report exact
wire schema, limits, security policy, test counts/coverage, artifact semantics,
errors, concurrency/cleanup behavior, docs, skips/failures/limitations and host
safety evidence.