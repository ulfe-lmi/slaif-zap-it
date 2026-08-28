# OAP Coding-Agent Report — 012-a

## Work order

- Identifier: `012-a`
- Objective: add the bounded final-object `annotated-labelled` visualization to
  the supported HTTP API.
- PR mode: one new Objective-012 branch and PR from remote `main`.
- Deferred human adjudication decision in the order: `NONE`.

## Status

PASSED

## Executive summary

Implemented and tested a deterministic Pillow-only `annotated-labelled` L3
renderer. It begins with the existing alpha mask overlay and then reads the
exact final ordered `ObjectResult` sequence, so visible instance numbers and
sanitized labels agree with post-CLIP, post-BLIP3, post-filter structured
results. Optional finite CLIP confidence is rendered with the specified
two-decimal suffix. Bounded candidate placement keeps label rectangles inside
the image and avoids complete overlap when a fixed candidate permits a free
position.

The API accepts this renderer only under `visualization.blip3` and strictly
validates `show_confidence`. Existing mask-only renderers, artifact budgets,
JSON/ZIP manifests, authentication and private-LAN behavior remain intact.
The persistent private-LAN service was restarted once as authorized, remained
enabled/active/ready, and passed real JSON/ZIP/repeat labelled-artifact
evidence on the assigned GPU.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/68`
- PR state: `OPEN`, merge state `CLEAN`; base `main` at
  `ce41b0becfb53cfe96ac11570a1af23b2d963311`.
- Branch: `oap/012-a-labelled-api-visualization`.
- Starting checkout SHA: `5b57095ea66730b6906ad95688a413da46561d5b`.
- Implementation head SHA: `05abf3795658e0b0ad0e5ebefb47affc415bf834`.
- Report publication commit: `SELF`.
- New PR: yes, exactly PR #68; amended existing PR: no.
- Coding merge/auto-merge: `NO`.

## Changes/files

- Added `modules.visualizer.render_annotated_labelled` and deterministic display
  sanitization, fixed candidate placement, confidence formatting, and bounded
  Pillow text rendering.
- Refactored `src/core/engine.py` so final ordering and IDs precede labelled
  visualization while legacy stage-mask renderers retain their existing inputs.
- Extended `src/service/yaml_input.py` for the final-stage renderer and strict
  `show_confidence` policy.
- Added pixel-level, sanitization, placement, confidence, final-label authority,
  JSON/ZIP parity, policy and L0-L2 execution tests in
  `tests/test_labelled_visualization.py`.
- Updated `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `docs/API.md`,
  `docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md` and `docs/runtime.md`.
- Carried the exact active/order transcript in `oap/active` and
  `oap/orders/012-a-labelled-api-visualization.md`.

The implementation commit is the single pushed commit
`05abf3795658e0b0ad0e5ebefb47affc415bf834`; this final child changes only this
report.

## Acceptance evidence

1. **Final-object renderer authority — PASSED.** The core constructs ordered
   `ObjectResult`s after CLIP, BLIP3 mutation and final label filtering, then
   supplies that exact sequence to `annotated-labelled`. Focused tests prove a
   BLIP3-mutated label wins over the earlier stage label, structured metadata
   and masks are unchanged, and manifest IDs/labels use the same final objects.
2. **Rendering contract — PASSED.** Pixel tests prove actual RGB label
   backgrounds and glyph pixels exist. Tests cover NFKC, whitespace,
   control/separator replacement, ASCII allowlist, empty fallback, 48-character
   cap, finite two-decimal confidence, absent/non-finite confidence, tiny and
   border masks, dynamic shortening, deterministic candidate placement and
   nearby-label non-complete-overlap behavior.
3. **API policy and compatibility — PASSED.** `annotated-labelled` is accepted
   only in `visualization.blip3`; earlier-stage placement, panoptic, unknown
   renderers, unsafe IDs, unknown fields and non-boolean confidence are rejected
   before inference. `annotated` remains mask-only. Existing annotated tests and
   the full suite remain green. Labels remain structural regardless of
   visualization selection.
4. **Limits and artifact parity — PASSED.** The renderer is one ordinary RGB
   stream; existing eight-stream, `height * width * 3`, per-artifact, total,
   response, deadline and artifact-count controls are unchanged. L0-L2 do not
   execute or reserve visualization arrays. Focused tests and real evidence
   prove JSON base64 and ZIP member hashes/sizes agree and artifact names remain
   based only on the safe visualization ID.
5. **Real private-LAN evidence — PASSED.** One authorized restart was followed
   by real verbosity-3 BLIP3 requests using the new renderer and a mask-only
   comparator in JSON, ZIP and repeat JSON. The labelled PNG was RGB at the
   original crop dimensions, differed from the mask-only PNG, had stable digest
   across repeats, and every final manifest object carried an instance ID and
   label.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 439 passed, 1 opt-in GPU test honestly skipped, 77.82% total
  coverage.
- `.venv/bin/pytest -q tests/test_labelled_visualization.py tests/test_visualizer.py tests/test_core_engine.py tests/test_parity_hardening.py`:
  `PASSED` — 59 passed.
- `.venv/bin/ruff format --check .`: `PASSED` — 142 files formatted.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current
  documents.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  `PASSED` — wheel and sdist member audits.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  `PASSED` — exactly 7 reviewed baseline findings.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  `PASSED` — no unexpected archive findings.
- `.venv/bin/python -m twine check dist/*`: `PASSED`.
- `git diff --check`: `PASSED`.
- Shell syntax check: `SKIPPED` — no shell files changed in this order.
- No model download or CUDA was used by the CPU suite or public CI checks.

Two live probe attempts failed before any authorized labelled inference and are
disclosed here as required:

- The first auth probe accidentally used `GET /v1/completions`; the expected
  `401` assertion received `405`. No labelled request was sent. Corrective
  change: use a POST auth probe.
- The second Python probe used POST but encountered a connection reset during
  the auth-only sequence. The service remained active with the same PID and
  readiness; direct POST checks then returned the required `401` statuses. No
  labelled request was sent by that probe. Corrective change: use the direct
  non-echoing POST checks and rerun the authorized labelled sequence.

The final authorized live probe was `PASSED`; no further restart or corrective
service mutation was needed.

## CI/checks

All current-head checks completed `SUCCESS` for implementation head
`05abf3795658e0b0ad0e5ebefb47affc415bf834` on PR #68:

- `static (format, lint, build)` — `PASSED`.
- `tests (py3.10)` — `PASSED`.
- `tests (py3.11)` — `PASSED`.
- `tests (py3.12)` — `PASSED`.
- `release (artifact audit)` — `PASSED`.
- `Analyze (python)` — `PASSED`.
- `CodeQL` — `PASSED`.

## GPU/service/resource evidence

- Host/service: hinton2, persistent `zap-it-lan.service`, enabled and active;
  final readiness `200`.
- Listener: exactly one `10.8.132.76:17891` listener, final MainPID `388703`,
  `NRestarts=0`, and the same PID/listener across all successful requests.
- Assigned GPU only: physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; visible inside the process
  as logical `cuda:0`. The final compute snapshot contained only PID `388703`
  on the assigned UUID. No unassigned GPU or unrelated process was touched.
- Final GPU sample: 13,547 MiB used and 10,577 MiB free. Service metrics
  recorded latest logical Torch peak allocated `11,732,496,384` bytes and peak
  reserved `13,841,203,200` bytes.
- Request fixture evidence used a sanitized in-memory crop with original
  dimensions `5568x4176` and request dimensions `2784x2088`; no raw image or
  configuration was copied into Git, OAP or chat.
- Real response evidence: 5 final objects; JSON response sizes `19,052,992`
  and `19,052,996` bytes; ZIP response size `14,245,869` bytes. The labelled
  PNG was `7,134,565` bytes with SHA-256
  `9ba5ee026fe1e11e8f3db28990de31cba47c8f56cef2e7cb03938f5288b125b1`; the
  comparator mask-only PNG was `7,134,565` bytes with SHA-256
  `2adad8f261e57202891963363922327e5ccc208cd644e700eda5229ebc5323ab`.
  JSON/ZIP descriptor/member hashes and sizes agreed; repeated labelled PNG
  digest was identical.
- Auth/docs: missing and wrong inference POST keys returned `401`; `/docs` and
  `/openapi.json` returned `404`.
- Shared memory and credentials: `/dev/shm/slaif-zap-it` remained empty; the
  root remained mode `0700`; the operator environment remained mode `0600`;
  the inference-key SHA-256 remained
  `cd7fb7f4189d1e5b0d759d09f718b309058724cb20294a40c296ce1fbb45cc51`.
- Sanitized journal inspection found no request filenames, bearer material or
  request-content markers. Authenticated metrics response size was `11,933`
  bytes and contained only bounded service metrics.

## Documentation/provenance

The API/config example, L3-only and final-stage rules, mask-only compatibility,
sanitization and confidence semantics, final-object authority, deterministic
placement, limits, artifact parity, runbook and datasheet claims are updated in
the changed documentation. No runtime dependency, model identity, model
revision, model residency strategy, font dependency, or package version changed.
The renderer uses the repository/runtime Pillow bitmap default font and no
network or user-selected filesystem resource.

## Deferred human adjudication

- Critical register action: `NONE`.
- No `CRITICAL.md` mutation was ordered or performed.

## Safety/scope confirmations

- No merge, auto-merge, release/tag/upload, public/WAN bind, TLS/gateway,
  firewall/route/VPN, driver/CUDA, unrelated systemd, second service, key
  rotation/disclosure, model lifecycle change, model download, or unassigned
  GPU operation was performed.
- The only service mutation was the single explicitly authorized restart of
  the owned `zap-it-lan.service`; it was left enabled, active, ready and
  serving the existing inference key.
- Request image/config/result data remained in memory. Raw labels, prompts,
  answers, YAML, images, credentials and user filenames were not written to
  the report, Git history, logs or chat.

## Limitations/blockers

The service remains a single private-LAN process with bearer authentication and
no TLS/WAN/public exposure or multi-user authorization. The labelled text is a
bounded diagnostic overlay; it does not change structured output or claim OCR
semantics. No blocker remains for this order.

## Factual strategic follow-up

Review PR #68 and its implementation/live evidence. The service remains enabled,
active and ready at the ordered private address/GPU. Coding does not merge,
accept, release or select a subsequent OAP order.
