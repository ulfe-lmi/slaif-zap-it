# OAP Coding-Agent Report — 025-a

## Work order

- Identifier: `025-a-correct-responses-image-tool-metadata`
- Objective: correct successful Responses image-tool metadata
- PR mode: new numeric objective / new PR
- Repository: `ulfe-lmi/slaif-zap-it`

## Status

COMPLETE

## Executive summary

Corrected the successful `POST /v1/responses` envelope for the accepted
`image_generation` tool. Tool-bearing responses now echo the typed tool,
report `tool_choice: "auto"`, retain `parallel_tool_calls: false`, and emit
one completed image-generation call. No-tool responses retain `tools: []`,
`tool_choice: "none"`, and `parallel_tool_calls: false`.

The maintained Pydantic response schema now uses the bounded typed
`ResponsesTool` array, allows only the two effective choices, and rejects
inconsistent tool/output combinations. Existing inference, projection,
renderer, limits, authentication, native completions, and gateway boundary
behavior remain unchanged.

## Authoritative GitHub state

- PR: [#89](https://github.com/ulfe-lmi/slaif-zap-it/pull/89), `OPEN`
- Title: `Objective 025: correct Responses image-tool metadata`
- Base: `main` at `fae1397bac15792bd6c064ee943c2f0f615aea9d`
- Starting SHA: `fae1397bac15792bd6c064ee943c2f0f615aea9d`
- Branch: `oap/025-a-correct-responses-image-tool-metadata`
- Implementation head SHA: `febc88d0494d747a28324c8230057eac527b6661`
- Report publication commit: SELF
- New PR: YES; amended existing PR: NO; coding merge: NO

The implementation branch is based directly on current `origin/main`, which
contains the merged Objective 024 report-only commit. The implementation
commit is the only non-report commit added this round and includes the exact
025-a active/order transcript. The report-only child is created after this
report is complete.

## Changes/files

Implementation commit `febc88d…` changes eight paths, with 401 insertions and
11 deletions:

- `src/service/responses.py`: derive successful `tools` and `tool_choice`
  metadata from the already validated `image_generation` request decision.
- `src/service/schemas.py`: replace the empty untyped response tool list with
  bounded `ResponsesTool` items, allow `none|auto`, and enforce canonical
  assistant/message/image-call consistency.
- `tests/test_objective_024.py`: extend no-tool/tool metadata, schema mismatch,
  OpenAPI, and official SDK contract assertions.
- `scripts/qualify_responses.py`: fail live qualification on wrong effective
  tool metadata and retain only bounded tool-choice/type summary fields.
- `docs/RESPONSES-FACADE.md`, `docs/API.md`: document conditional successful
  metadata and the unchanged private/native boundary.
- `oap/active`, `oap/orders/025-a-correct-responses-image-tool-metadata.md`:
  exact round transcript committed unchanged.

The product diff excluding the orchestration transcript is seven paths with
95 insertions and 11 deletions. No `CRITICAL.md`, gateway repository,
dependency, model, inference, renderer, deployment, or native-completion
implementation path was changed.

## Acceptance evidence

1. **Tool-bearing metadata — PASSED.** Before this round the merged builder
   emitted `tools: []` and `tool_choice: "none"` beside an image-generation
   call. The corrected response emits exactly
   `tools: [{"type":"image_generation"}]`, `tool_choice: "auto"`, and
   `parallel_tool_calls: false`.
2. **No-tool preservation — PASSED.** CPU/fake tests prove message-only output
   with `tools: []`, `tool_choice: "none"`, and `parallel_tool_calls: false`.
3. **Schema/OpenAPI — PASSED.** The response model uses typed
   `ResponsesTool`, `maxItems: 1`, `tool_choice` enum `none|auto`, exactly one
   assistant message, and the cross-field invariant for tool/call agreement.
   Focused tests reject an image call with no-tool metadata and a declared
   tool without its call; generated OpenAPI advertises the bounded typed shape.
4. **Official SDK — PASSED.** The CPU contract and live qualification with
   pinned `openai==3.7.0` parse `Response`, expose `tool_choice == "auto"`,
   expose exactly one typed image-generation tool, preserve `output_text`,
   and expose one typed completed image call with a valid PNG result.
5. **Projection and renderer preservation — PASSED.** Tool toggling leaves
   the public projection text and engine configuration equal. Existing direct
   `render_annotated_labelled(..., alpha=0.5, show_confidence=False)` plus the
   shared encoder remains byte-identical to the tool result.
6. **Bounds/errors/private behavior — PASSED.** The existing final response
   size calculation validates the corrected envelope bytes. Focused and full
   suites retain unsupported request/tool, authentication, cardinality,
   capacity, error, and private `/v1/completions` coverage.
7. **Documentation — PASSED.** Current API/facade docs state the conditional
   metadata and preserve the native/private, future-facade, and not-yet-
   qualified gateway distinctions.
8. **Critical register — PASSED.** The order specifies `Decision: NONE`;
   `CRITICAL.md` was read and no register bytes changed.

## Verification

- `.venv/bin/pytest -q tests/test_objective_024.py`: PASSED — 40 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 971 passed, 1 explicit GPU test skipped, 82.78% total coverage
  against the 64% gate; two existing warnings were reported.
- `.venv/bin/ruff format --check .`: PASSED — 162 files formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 28 current
  documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED — existing setuptools
  license metadata deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl
  dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz
  --baseline .secrets.baseline`: PASSED — zero archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: PASSED — seven reviewed baseline findings.
- `.venv/bin/python scripts/verify_release_artifacts.py --compare-wheels
  dist/*.whl <wheel-rebuilt-from-sdist>`: PASSED — no member differences.
- `.venv/bin/python scripts/scan_release_artifacts.py <wheel-rebuilt-from-sdist>
  --baseline .secrets.baseline`: PASSED — zero archive findings.
- `.venv/bin/python -m twine check dist/* <wheel-rebuilt-from-sdist>`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check origin/main...HEAD`: PASSED.
- CPU official SDK contract within the focused suite: PASSED — typed response,
  output text, echoed tool, image call, strict base64, and PNG decode.
- Authenticated live qualification:
  `.venv/bin/python scripts/qualify_responses.py --host "$SLAIF_ZAP_IT_HOST"
  --port "$SLAIF_ZAP_IT_PORT" --evidence-root "$SLAIF_ZAP_IT_TMP_ROOT"`:
  PASSED — content-free summary only.
- Authenticated native L2 smoke using the repository smoke helper and the
  operator-supplied bearer environment: PASSED — HTTP 200, native envelope,
  eight objects, and uint16 identity mask.
- Content-free live probes: PASSED — health 200, readiness 200, authenticated
  capabilities 200, unauthenticated Responses 401, exact listener ownership.

## CI/checks

All seven required implementation-head checks passed on
`febc88d0494d747a28324c8230057eac527b6661`:

- [static (format, lint, build)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989381/job/100211371500): PASSED
- [release (artifact audit)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989381/job/100211371725): PASSED
- [tests (py3.10)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989381/job/100211371757): PASSED
- [tests (py3.11)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989381/job/100211371756): PASSED
- [tests (py3.12)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989381/job/100211371668): PASSED
- [Analyze (python)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33618989348/job/100211370517): PASSED
- [CodeQL](https://github.com/ulfe-lmi/slaif-zap-it/runs/100211636973): PASSED

The final SELF checks are intentionally inspected after publication because
the report is immutable and no post-publication report edit is permitted.

## GPU/service/resource evidence

- Host: `hinton2`.
- Authorized physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`.
- Runtime compatibility: Torch `2.5.1+cu124`, CUDA `12.4`.
- Launch mapping remained `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`; application visibility is logical `cuda:0`, with
  the expected UUID pinned. No unassigned device or unrelated process was
  touched.
- The existing PID `815951` was stopped once through
  `scripts/serve_local.sh restart`. The corrected service is left running as
  PID `837516`, started `2026-09-02 12:23:59 CEST`, with the exact listener
  `10.8.132.76:17891` and no wildcard listener. No rollback was needed.
- Post-qualification health/readiness remained 200/200 and the
  unauthenticated Responses probe remained 401. The assigned-card sample
  reported only PID `837516` at 10572 MiB; total used/free was 10595/13529
  MiB.
- `/dev/shm` is a 12 GiB tmpfs with approximately 9.7 GiB free. The scoped
  root and runtime directory are mode 700; retained qualification summaries
  are mode 600. No request image, YAML, response body or bearer was persisted.
- The bounded live SDK summary reported: status `PASSED`, SDK `3.7.0`,
  response `response`/`completed`, `tool_choice=auto`, one response tool of
  type `image_generation`, one image call, two public objects, 2500 projection
  bytes, 997 PNG bytes, projection SHA-256
  `a0324ac28ba64e2da5c70b9c18123069ad227b549ef5b98503f422b0cab315d6`, and
  PNG SHA-256
  `1848669e4a0816187975f600327352748fa6a6aa3efc3ff8208c638e043eb527`.
- The authenticated native smoke reported HTTP 200, object
  `text_completion`, L2 JSON, eight objects/YOLO lines, a 128×128 uint16
  identity mask, and an 8175-byte response.
- The live service log hygiene scan found no traceback, bearer/API-key text,
  input filename, data URL, input content, or response content.

## Documentation/provenance

The current docs now state the official-semantic correction: the accepted
image-generation tool is echoed and selected with `auto`, while no-tool
success remains `none` with an empty tool array. The official SDK remains a
development-only dependency; runtime code does not import it. The gateway
repository was not accessed or changed and remains unqualified for this
multimodal path. No request/config migration is required.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was read as ordered and was not modified.

## Safety/scope confirmations

Only Objective 025-a was implemented. No model, inference parameter, holder,
residency, device policy, CUDA/driver, firewall, network/VPN, credential,
gateway repository, unrelated service/process, release/tag, or merge was
changed. The one live restart used the repository launcher, exact assigned
GPU, existing private-LAN address/port, and existing operator environment.
No raw request data, credentials, model weights, or customer data entered the
report or OAP evidence.

## Limitations/blockers

The live proof establishes ZAP-IT-side deterministic HTTP/SDK/schema/renderer
integration and native preservation. It does not establish semantic model
accuracy, recall, precision, commercial licensing, gateway end-to-end
qualification, public/WAN deployment safety, production readiness, or final
release authority. Coding cannot merge or accept PR #89.

## Factual strategic follow-up

The strongest reason not to merge autonomously is the coding role’s authority
boundary: it cannot accept or merge. The strongest technical risk is that an
OpenAI-shaped adapter could drift from official tool semantics or expose a
tool call without declaring the tool. This round answers that risk with the
official SDK semantic assertions and live proof, a typed bounded schema with
cross-field validation, exact renderer-byte preservation, retained native
regressions, complete package/security checks, and green implementation-head
CI/CodeQL. Strategic should independently inspect the PR diff, implementation
to SELF lineage, and final SELF checks before any acceptance decision.
