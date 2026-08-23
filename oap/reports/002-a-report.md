# OAP Coding-Agent Report — 002-a

## Work order
- Identifier: `002-a` (numeric objective `002`, round `002-a`)
- Order: `oap/orders/002-a-v1-completions-api-contract.md` — `/v1/completions`
  API contract with fake engine, CPU-only; PR mode `CREATE_NEW_PR`
- Exact `oap/active` transcript carried in implementation commit `abe9878`
  (`oap/active` = `002-a`, order file added verbatim)

## Status
COMPLETE

## Executive summary

Implemented the frozen v1 HTTP service contract around the objective-001 core:
a new `src.service` package (FastAPI/Pydantic/Uvicorn under a dedicated
`[service]` extra) with strict streaming multipart parsing, hostile-YAML
allowlist policy, bounded image decoding, single-slot concurrency with honest
503 busy semantics, L0–L3 monotonic completion envelopes with JSON/ZIP parity,
stable sanitized errors, optional constant-time API-key auth,
health/readiness endpoints, and a deterministic fake engine. All work is
CPU-only; no GPU was allocated and no persistent listener was started.
PR #46 created from the required branch; all six required GitHub checks are
SUCCESS on the implementation head. No live model/GPU readiness is claimed.

## Authoritative GitHub state
- Repository: `ulfe-lmi/slaif-zap-it`
- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/46 — OPEN, not merged
- Base: `main` @ `08fed7bacc83d6121e33d508fd79e0b6533345fc` (re-verified via
  `git fetch` immediately before publication; unchanged)
- Branch: `oap/002-a-v1-completions-api-contract`
- Starting SHA: `08fed7bacc83d6121e33d508fd79e0b6533345fc`
- Implementation head SHA: 327a401ecef673d8dda1fad8a3528520d59b5461
- Report publication commit: SELF
- New PR yes (#46); no existing PR amended; coding merge NO

## Changes/files

Commit `abe9878003ef49b7000e58ba8c5b2332b064f25d` (implementation):
- `pyproject.toml` — `[service]` extra (`fastapi>=0.115,<1`,
  `python-multipart>=0.0.9`, `uvicorn>=0.30,<1`), `httpx` added to dev,
  `src.service` packaged
- `src/service/errors.py` — frozen code→status table, `ServiceError`, envelope builder
- `src/service/settings.py` — operator-overridable startup limits/env parsing
- `src/service/multipart.py` — strict streaming multipart parser (limit+1 caps,
  cardinality/duplicate/unknown-field rejections, scalar field validation)
- `src/service/image_input.py` — JPEG/PNG/WebP-only decode with pre-allocation
  pixel cap and malformed-codec mapping
- `src/service/yaml_input.py` — composition-time bounds (depth/nodes/collection/
  scalar), alias rejection at event level, typed allowlist, batch-only
  ignore-with-warning, forbidden key/value scan, debug-flag stripping below L3,
  legacy alpha hoisting, CLIP label→class-mapping derivation
- `src/service/fake_engine.py` — deterministic outcome factory with
  concurrency-observation counters (no CUDA/models/network)
- `src/service/gate.py` — one-active-slot gate, queue depth, Retry-After
- `src/service/auth.py` — constant-time Bearer verification
- `src/service/envelope.py` — L0–L3 JSON assembly, base64 artifact objects
  (name/media_type/encoding/sha256/size/data), deterministic ZIP + manifest
- `src/service/schemas.py` — Pydantic models documenting the wire contract
- `src/service/app.py` — app factory with injected engine/readiness/settings,
  Content-Length precheck, streamed body caps, deadline/cancel handling,
  sanitized exception handlers, routes
- `tests/test_service_units.py`, `tests/test_service_api.py` — new suites
- `tests/conftest.py` — offline guard evolved to block non-loopback outbound
  connects while permitting asyncio internal sockets (suite stays offline)
- `.github/workflows/ci.yml` — installs `.[dev,service]`
- `docs/API.md` (new), `docs/API-TARGET.md`, `README.md` — contract docs/navigation
- `oap/active`, `oap/orders/002-a-v1-completions-api-contract.md` — governance transcripts

Commit `327a401ecef673d8dda1fad8a3528520d59b5461`: removed one test assertion
(`"example.com" in str(exc)`) that CodeQL flagged as
`py/incomplete-url-substring-sanitization`; guard semantics unchanged.

## Acceptance evidence
1. Importable FastAPI app exposes `/v1/completions`, `/healthz`, `/readyz` with
   injectable engine/readiness/settings — PASSED (`create_app`,
   `test_openapi_documents_contract`)
2. CPU TestClient/HTTPX tests exercise all routes without CUDA/network/model
   downloads — PASSED (offline suite guard active throughout)
3. Exactly-one-image/config cardinality and all request limits enforced before
   expensive work — PASSED (missing/duplicate/unknown-part tests; byte caps
   enforced mid-stream before decode; decoded-pixel check pre-allocation)
4. Uploaded YAML cannot select host paths/network/devices/commands/code/
   credentials/model repos/revisions/deployment settings — PASSED
   (parametrized hostile-key/value/alias/depth/size tests over module and HTTP)
5. L0–L3 monotonic per Objective-001 ordering/renderers; lower levels do not
   trigger extra optional stages — PASSED (`level_bodies` fixture asserts exact
   key sets per level; YOLO text identical across levels; identity PNG uint16
   ids {0,1,2}; debug flags stripped below L3 with warnings)
6. JSON and ZIP artifacts have correct bytes/names/hashes/sizes/media types;
   response size bounded — PASSED (manifest hash/size equality recomputed from
   archive bytes; `response_too_large` paths implemented)
7. Stable sanitized errors tested; no stack/raw content/path/secret leakage —
   PASSED (envelope shape asserted on every error path; secret-echo negative test)
8. Concurrency tests prove at most one inference executes simultaneously;
   busy/queue deterministic/bounded — PASSED (queue_depth=0 → [200,503];
   queue_depth=1 → [200,200,503]; fake-engine `max_observed_active == 1`;
   unit-level gate tests)
9. Success, validation failure, inference failure, timeout and cancellation
   leave no workspace residue — PASSED (`test_no_files_created_by_service_requests`;
   memory-only lifecycle; no filesystem workspace used)
10. Health/readiness honest and distinguish process-up from engine-ready — PASSED
11. Optional auth constant-time/operator-controlled/tested; loopback default explicit — PASSED
12. OpenAPI/docs/examples agree with actual wire contract — PASSED (schema
    documents multipart body fields and response/error models; deterministic
    snapshot across instances; docs/API.md matches codes/limits)
13. Canonical CPU suite, Ruff/package/coverage/CI and CodeQL green — PASSED
14. No listener remains after tests; no GPU process/state changes — PASSED
15. Correct branch/one PR/report-only SELF contract satisfied; coding did not merge — PASSED

## Verification
- Predecessor state: `main` @ `08fed7ba…` CI SUCCESS + CodeQL SUCCESS
  (re-confirmed live at round start and immediately before report publication) — PASSED
- `.venv/bin/ruff format --check .` — PASSED (99 files already formatted)
- `.venv/bin/ruff check .` — PASSED (All checks passed!)
- `.venv/bin/python -m build --wheel` — PASSED (zap_it-0.1.0-py3-none-any.whl;
  includes 12 `src/service` modules)
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing` —
  PASSED: **244 passed / 0 failed / 0 skipped**, ~8.2 s wall, branch coverage
  **76%** (gate fail_under=64 held; measured 76.15–76.40% across runs)
- Dedicated API/TestClient contract suite `tests/test_service_api.py::…` (47 tests) — PASSED
- Unit boundary suite `tests/test_service_units.py::…` (58 tests) — PASSED
- OpenAPI schema snapshot/validation — PASSED (deterministic bytes across instances)
- Hostile YAML/property/limit tests — PASSED
- JSON/ZIP artifact parity/hash tests — PASSED
- Concurrency/busy/timeout/cancel/cleanup tests — PASSED
- Optional auth tests — PASSED
- Listener scan before/after (`ss -tlnp`) — PASSED: 10 LISTEN entries before
  and after, all pre-existing system services (sshd, cups, nfs, resolved, plus
  an unrelated third-party listener on 10.8.132.72:8000); none created or
  modified by this work
- Read-only GPU before/after snapshots — PASSED: identical outputs (below)

## CI/checks
All on head SHA 327a401ecef673d8dda1fad8a3528520d59b5461 (PR #46):
- static (format, lint, build): SUCCESS
- tests (py3.10): SUCCESS
- tests (py3.11): SUCCESS
- tests (py3.12): SUCCESS
- Analyze (python): SUCCESS
- CodeQL: SUCCESS
(One intermediate head `abe9878` had CodeQL FAILURE from a single high-severity
code-scanning alert `py/incomplete-url-substring-sanitization` triggered by a
test-file substring assertion; fixed in `327a401` by deleting that redundant
assertion; all checks then SUCCESS. No production code involved.)

## GPU/service/resource evidence
- Physical GPU snapshot BEFORE suite:
  `0, GPU-4c129e25-8e59-eee4-b49c-56c40e294182, 2161 MiB` /
  `1, GPU-c457dbaf-991c-dc23-c781-0dc030776dd8, 6 MiB`;
  compute-apps: only `GPU-4c129e25…, PID 66522, 2152 MiB` (unrelated workload)
- Physical GPU snapshot AFTER suite: byte-identical CSV; compute-apps identical
- Zero allocation on either GPU; physical GPU0 untouched; no CUDA context created
- Visible-device mapping: not applicable (no GPU code executed; no
  CUDA_VISIBLE_DEVICES set or needed in this CPU-only objective)
- Ports/services: no server started; TestClient/httpx ASGI transports only;
  listener set unchanged before/after
- Memory//dev/shm: all request/response data in process memory; no files
  written anywhere by service requests (asserted by test against isolated cwd);
  `/dev/shm/slaif-zap-it` remains documented default for later objectives only

## Documentation/provenance
- `docs/API.md`: request/response schemas, multipart field rules, verbosity
  levels, artifact object, ZIP layout, config allowlist policy, limits table
  with env vars, busy/concurrency semantics, auth policy, full error table
  incl. status choices (timeout=504, cancelled=499 nginx-convention,
  insufficient_memory=507 reserved), example, limitations
- Explicit non-claim of live model/GPU readiness stated in docs/API.md and README
- Objective 000–001 provenance preserved; no license changes; MIT retained;
  no model weights/caches committed; dependencies pinned with version bounds

## Deferred human adjudication
- Critical register action: NONE
- Order decision `NONE` honored; no CRITICAL entry appended, none authored,
  no candidate meeting the five-condition threshold was encountered

## Safety/scope confirmations
- No secrets, credentials, raw images/YAML, or environment values entered logs,
  OAP artifacts or this report
- Scope stayed inside order 002-a: no SAM2/CLIP/BLIP3 loading, no persistent
  listener, no GPU allocation, no systemd/Docker/firewall/VPN changes, no video
  API, no remote model selection/download, no background persistence, no
  objective 003+ implementation
- Protected resources untouched: unrelated port-8000 workload, GPU0 workload
  (PID 66522), system services, global agent credentials
- Working tree preserved; strategic-authored `oap/active` and order file
  carried verbatim into implementation history

## Limitations/blockers
- Contract exercised exclusively via fake engine; real inference wiring is
  objective 003+
- Timed-out inferences cannot be hard-killed inside a Python worker thread;
  the slot is released honestly but the thread may finish in background until
  the call returns (documented Python limitation; serialization still holds)
- Identity-mask overlap uses the larger-area winner; overlap-preserving
  per-object masks are objective 005 scope
- `geometry`/`blip2` legacy sections are rejected (`unsupported_field`) because
  the core does not consume them
- Gate adopts a new event loop only when quiescent (single-loop production
  model per architecture law); concurrent cross-loop sharing is rejected

## Factual strategic follow-up
- Coverage ratchet can be raised above the measured ~76% in a later order if
  desired (gate currently 64%)
- Error-code statuses for `cancelled` (499) and `insufficient_memory` (507) are
  provisional-but-frozen choices documented in docs/API.md; revisit only via a
  versioned contract change
- Objective 003 must pin model revisions/licenses and measure GPU1 residency
  before any live activation; objective 004 owns loopback port selection
