# OAP Coding-Agent Report — 024-a

## Work order

- Identifier: `024-a-openai-responses-compatible-facade`
- Objective: additive narrow OpenAI Responses-compatible facade
- PR mode: new numeric objective / new PR
- Repository: `ulfe-lmi/slaif-zap-it`

## Status

COMPLETE

## Executive summary

Added `POST /v1/responses` as a stateless, non-streaming, inline-data facade
over the existing typed in-memory pipeline. The route accepts exactly one
fixed-model user message containing one strict-base64 image and one strict-
base64 YAML file. It reuses the existing decoder, hostile YAML validator,
readiness, gate, executor, model holders, and final object-record semantics.

The assistant output is deterministic `zap-it.public.v1` JSON. The optional
standard `image_generation` declaration adds exactly one canonical final
annotated PNG from the existing renderer; it does not invoke a generative
model. Private completion behavior and diagnostics remain unchanged.

## Authoritative GitHub state

- PR: [#88](https://github.com/ulfe-lmi/slaif-zap-it/pull/88), `OPEN`
- Base: `main` at `32812032781c5d7daf54d5b7586b3c01d3270c48`
- Starting SHA: `32812032781c5d7daf54d5b7586b3c01d3270c48`
- Implementation commits: `36ca098c4d9941f7ab127370333279137de2474a`,
  `3e6e44aed3a4a2dc0708a8032aa4e244fb4ddd89`
- Implementation head SHA: `3e6e44aed3a4a2dc0708a8032aa4e244fb4ddd89`
- Report publication commit: SELF
- New PR: YES; amended existing PR: NO; coding merge: NO

The remote branch was verified at the implementation head before this report.
The implementation lineage is `3281203 -> 36ca098 -> 3e6e44a -> SELF`.

## Changes/files

- `src/service/app.py`: shared decode/validation/readiness/gate/executor seam,
  Responses route, pre-body authentication handler, OpenAPI registration.
- `src/service/responses.py`: bounded JSON/data-URL parser, public projection,
  canonical response/image adapter, and sanitized error adapter.
- `src/service/schemas.py`: strict request, response, output-item, public
  projection and OpenAI error schemas.
- `src/service/envelope.py`: named shared object-record and PNG encoder seams;
  existing private encoder alias retained for compatibility.
- `src/service/capabilities.py`, `src/service/errors.py`,
  `src/service/metrics.py`: typed surface disclosures, error codes, and finite
  Responses outcome metrics.
- `scripts/qualify_responses.py`: official SDK operator qualification using
  bounded in-memory synthetic inputs and mode-0600 RAM-backed evidence.
- `tests/test_objective_024.py`: focused parser, projection, renderer, error,
  auth, concurrency, OpenAPI, capability and official SDK contract tests.
- `pyproject.toml`: development-only exact `openai==3.7.0` dependency.
- `MANIFEST.in`: qualification script inclusion in source artifacts.
- `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `THIRD_PARTY_NOTICES.md`,
  `docs/API.md`, `docs/README.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md`, `docs/OUTPUT-PARITY.md`,
  `docs/RESPONSES-FACADE.md`: current contract and qualification documentation.
- `src/service/__init__.py`: service-surface documentation.
- `oap/active` and `oap/orders/024-a-openai-responses-compatible-facade.md`:
  exact unchanged active/order orchestration transcript.

No request/config migration is required. Existing `/v1/completions` remains
the native multipart JSON/ZIP research/debug contract.

## Acceptance evidence

- Canonical inline image/YAML input reaches the shared fake engine once at
  fixed verbosity 2 with `render_visualizations=False`; tool presence does not
  alter effective config digest or engine invocation.
- Public object records reuse the private L2 builder and retain stable instance,
  source, filtered, class, score/routing and BLIP3 evidence while omitting
  `mask_rle`. Canonical output text is byte-stable for equal outcomes and uses
  `allow_nan=false`.
- No-tool output contains one assistant message and no image call. Tool output
  contains that message plus one completed `image_generation_call`; strict
  base64 decoding produced a valid 32x24 RGB PNG.
- PNG bytes exactly matched a direct
  `render_annotated_labelled(..., alpha=0.5, show_confidence=False)` call plus
  the shared encoder.
- Unknown fields, wrong model/types, malformed JSON, wrong/empty base64, MIME
  mismatch, corrupt image, unsafe filename, invalid YAML, missing/duplicate
  parts, unsupported tools/sources, state controls and query parameters are
  rejected before engine calls with bounded OpenAI-shaped errors.
- The body is streamed under the decoded-upload-derived base64 cap;
  authentication runs before body processing. Existing image/YAML, dimensions,
  resource, object, artifact and response limits remain active.
- Responses and completions use one `InferenceGate`/executor; an async
  cross-surface overlap observed at most one active fake engine call.
- Capabilities disclose both surfaces, fixed model, exact inline sources/MIME
  types, public projection, derived body bound, omitted usage, private evidence
  exclusions and unqualified gateway status. No retrieve/update/delete/list
  Responses routes exist.
- The public route uses no request persistence. Final live `/dev/shm` evidence
  contains only two mode-0700 directories and 389-byte mode-0600 summaries.

The official wire references were checked before implementation:
[Create a model response](https://developers.openai.com/api/reference/cli/resources/responses/methods/create),
[file inputs](https://developers.openai.com/api/docs/guides/file-inputs), and
[image generation output extraction](https://developers.openai.com/api/docs/guides/image-generation).

## Verification

- `.venv/bin/pytest -q tests/test_objective_024.py`: PASSED — 33 tests.
- `.venv/bin/pytest -q tests/test_objective_024.py tests/test_service_api.py tests/test_labelled_visualization.py`: PASSED — 113 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 962 passed, 1 explicit GPU test skipped, coverage 82.83% against a 64% gate.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 28 current documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — zero archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exactly 7 reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `openai==3.7.0` official SDK CPU contract: PASSED — typed `Response`, `output_text`, typed output iteration, image-call recognition, strict PNG decode.
- Authenticated live `scripts/qualify_responses.py`: PASSED — official SDK, one image call, 2 objects, projection 2500 bytes, PNG 997 bytes.
- Authenticated existing native completion L2/JSON smoke: PASSED — HTTP 200, 8 objects, identity mask, 8176-byte response.
- `git diff --cached --check` excluding the immutable strategic order path: PASSED. The strategic order retains its authored trailing blank line byte-for-byte.

## CI/checks

All seven required checks passed on implementation head
`3e6e44aed3a4a2dc0708a8032aa4e244fb4ddd89`:

- [static (format, lint, build)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118198/job/100170371191): PASSED
- [release (artifact audit)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118198/job/100170371054): PASSED
- [tests (py3.10)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118198/job/100170371072): PASSED
- [tests (py3.11)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118198/job/100170370988): PASSED
- [tests (py3.12)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118198/job/100170370917): PASSED
- [Analyze (python)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33606118259/job/100170371038): PASSED
- [CodeQL](https://github.com/ulfe-lmi/slaif-zap-it/runs/100170600164): PASSED

## GPU/service/resource evidence

- Host: `hinton2`; assigned physical GPU 0 only, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`.
- Runtime compatibility: Torch `2.5.1+cu124`, CUDA `12.4`; the service
  exposed one visible device as logical `cuda:0`.
- Launch mapping remained `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=0`, expected physical index/UUID pinned by the
  existing operator environment. No unassigned device or unrelated process
  was touched.
- Service: user unit `zap-it-lan.service`, PID `805444`, `NRestarts=0`, active;
  listener exactly `10.8.132.76:17891` (non-wildcard); health/readiness 200;
  authenticated capabilities 200; unauthenticated capabilities 401.
- Final assigned-card sample: `nvidia-smi` reported 10595 MiB used, with only
  PID 805444 using 10572 MiB. Service cgroup current memory was 12,834,586,624
  bytes and peak memory 19,436,224,512 bytes.
- `/dev/shm`: 12 GiB total, approximately 9.7 GiB free at final check; request
  data/results were not persisted. Environment file remained mode 0600 with
  digest `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- Post-request logs passed the bounded hygiene scan: no traceback, bearer,
  request body, filename, prompt/answer or API-key content was found.

## Documentation/provenance

The public facade is documented separately from the native completion API in
`docs/RESPONSES-FACADE.md`; API, architecture, runbook, datasheet, parity,
README, testing and third-party provenance documents identify the future
gateway surface and its non-public/non-production qualification boundary.
The SDK is development-only and server/runtime source imports no OpenAI SDK.

## Deferred human adjudication

- Critical register action: NONE
- The active order explicitly specifies `Decision: NONE`; `CRITICAL.md` was
  read and no bytes were changed.

## Safety/scope confirmations

Only Objective 024-a was implemented. No model, inference parameter, holder,
residency, device, CUDA/driver, firewall, gateway repository, TLS, public/WAN
bind, credential, unrelated service or protected process was changed. No
merge, release, tag or external deployment was performed. The live service was
restarted only through the explicitly authorized user unit and was left on the
same private address, port, fixed credential and assigned GPU.

## Limitations/blockers

The live proof establishes ZAP-IT-side HTTP/SDK/schema/renderer integration and
bounded local execution, not semantic accuracy, recall, precision, commercial
model licensing, public deployment safety or production readiness. The
`slaif-api-gateway` Responses multimodal/image-generation route remains absent
and was neither accessed nor qualified.

## Factual strategic follow-up

The strongest reason not to merge autonomously is the coding role’s authority
boundary: it cannot accept or merge. The strongest technical risk is that an
OpenAI-shaped adapter could drift from official wire semantics or leak native
diagnostics. This round addresses that risk with official SDK parsing, one
shared inference seam, exact existing-renderer byte comparison, explicit
hostile-input rejection tests, private completion regressions, capability
disclosure and green multi-version CI; strategic should independently inspect
the PR diff, final report lineage and live evidence before acceptance.
