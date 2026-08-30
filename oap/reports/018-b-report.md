# OAP Coding-Agent Report — 018-b

## Work order

- Identifier/order/objective/PR mode: `018-b`, make resize and residency proof
  independent, amend the existing Objective-018 PR.
- Active selector: `018-b`.
- Exact order SHA-256: `38f327d48886dcda3c3c4fd5da59068ec227ea0fb7339861a0147b9ea311f2ad`.

## Status

COMPLETE

## Executive summary

Corrected the two proof-quality defects identified by strategic review without
changing runtime/product behavior. The tiny-mask test now derives source-space
Euclidean support, the tight context bbox, center-based nearest-neighbor
mapping, source contour, and paired-image contour from bounded test-owned
oracles. The L0-L3 A/B/A service test now guards the actual CLIP fallback
initializer and BLIP3 holder constructor, records forbidden attempts, and
asserts stable resident-holder identities on every request.

## Authoritative GitHub state

- Repository/PR: `ulfe-lmi/slaif-zap-it`, [#74 — Objective 018: close
  mask-view acceptance matrix](https://github.com/ulfe-lmi/slaif-zap-it/pull/74).
- PR state: open, non-draft, mergeable, base `main`; coding did not merge,
  accept, close, or enable auto-merge.
- Branch: `oap/018-a-close-mask-view-acceptance-matrix`.
- Base SHA: `03def697373f2ae83d03494315aa96c800f0bcdf`.
- Starting SHA: `5fbcf43561a023e771fdfde6fb5795275b50227e` (018-a report SELF).
- Implementation head SHA: `ab85fe7f3c46ea3bf6cd3f0a75d517e9486031a9`.
- Implementation parent: `5fbcf43561a023e771fdfde6fb5795275b50227e`.
- Report publication commit: SELF.
- New PR: no; amended existing PR #74: yes.
- Implementation/control commit includes the exact unchanged order and
  `oap/active` selector (`018-b`).

## Changes/files

The implementation/control commit changes exactly five paths:

- `tests/test_mask_views.py`: independent bounded Euclidean, source-contour,
  center-mapping, bilinear, and square-contour expected values, including
  explicit eligible/ineligible marker assertions.
- `tests/test_candidate_view_api.py`: fail-if-invoked guards on
  `clip.initialize` and `blip3._Blip3QA`, observed attempt-list assertions,
  and per-request resident identity assertions.
- `TESTING.md`: documentation aligned with the observed evidence.
- `oap/active`: exact selector `018-b`.
- `oap/orders/018-b-make-resize-and-residency-proof-independent.md`: exact
  active order.

No runtime/product module, schema, default, dependency, lockfile, model,
environment, service, GPU, credential, network, or CRITICAL-register path was
changed.

## Acceptance evidence

1. **Independent tiny-mask oracle — PASSED.** The expected squared Euclidean
   distance field is computed by a small bounded source-coordinate brute-force
   oracle. Its support bbox and target bbox come from the generated mask; the
   expected RGB crops are independently neutralized from the generated image.
   The test-owned pixel-center formula derives the 256-pixel mask mappings, and
   Pillow bilinear interpolation is applied only to those independent crops.
   Source-stage contour uses a bounded test-owned Euclidean oracle, while the
   paired-image contour uses a separate test-owned square-neighborhood oracle.
   No production mapper or dilation helper derives an expected value.

2. **Independent masks/RGB/contour and legacy assertions — PASSED.** The
   independent source bboxes, target/support masks, source RGB arrays, resized
   target/context arrays, mapped masks, and contour equal the production
   builder/pair outputs for width zero and positive width. Zero fill, prohibited
   pixels, target restoration, contour placement/color, and repeated-byte
   determinism remain asserted.

3. **Marker boundary — PASSED.** The generated inside-radius marker is outside
   target-only and contributes a nonzero eligible context display (or the
   documented contour color); the outside-radius marker is absent from both
   target-only and context. Both are checked against the independent expected
   arrays and actual paired output.

4. **Observed resident-holder proof — PASSED.** The actual CLIP
   `modules.classifier.clip.initialize` seam and BLIP3 `modules.verifier.blip3._Blip3QA`
   construction seam are monkeypatched to record and fail on invocation while
   resident holders are supplied. The observed forbidden-initialization list
   is empty, its count is zero, and exact CLIP/BLIP3 holder identity tuples are
   equal on every request across the L0-L3 A/B/A matrix. BLIP3's request-local
   rule wrapper remains allowed and is not treated as holder construction.

5. **Documentation — PASSED.** `TESTING.md` describes the test-owned
   Euclidean/nearest-neighbor/contour expected values and the fail-if-invoked
   resident initialization seams without claiming semantic-model accuracy.

6. **Scope and verification — PASSED.** The implementation diff is test,
   documentation, and OAP transcript only. Focused/full CPU, static, package,
   release-integrity, archive, secret-baseline, metadata, and implementation
   head CI checks are green. The one live GPU test remains honestly skipped
   because this order authorizes no GPU phase.

7. **Service preservation — PASSED.** The already-running private-LAN service
   remains enabled, active, ready, and unchanged; no restart, reload,
   reconfiguration, or inference request was performed.

## Verification

- `.venv/bin/pytest -q tests/test_mask_views.py::test_tiny_mask_builds_source_space_crop_before_resize_and_contour tests/test_mask_views.py::test_real_clip_classify_single_receives_literal_processor_context_view tests/test_mask_views.py::test_blip_debug_uses_one_based_source_and_question_ids tests/test_candidate_view_api.py::test_candidate_view_policy_levels_and_stable_resident_ab_a_isolation tests/test_candidate_view_api.py::test_source_identity_survives_filter_semantics_order_visualization_json_and_zip`:
  **PASSED** — 6 passed, 1 warning.
- `.venv/bin/pytest -q tests/test_mask_views.py tests/test_candidate_view_api.py`:
  **PASSED** — 55 passed, 1 warning.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  **PASSED** — 791 passed, 1 honest GPU test skipped, 2 warnings, total
  coverage 81.50%, 83.93 seconds.
- `.venv/bin/ruff format --check .`: **PASSED** — 149 files already formatted.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules scripts tests`: **PASSED**.
- `.venv/bin/python scripts/check_documentation.py`: **PASSED** — 27 current
  documents.
- `git diff --check`: **PASSED**.
- `.venv/bin/python -m build --wheel --sdist`: **PASSED** — wheel and sdist
  built; existing setuptools license/deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py` on the direct wheel and
  sdist: **PASSED** — 67 wheel members and 160 sdist members.
- Wheel built from the sdist and
  `scripts/verify_release_artifacts.py --compare-wheels`: **PASSED** — direct
  and sdist-built wheel member manifests/bytes matched.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree` on the
  direct wheel, sdist, and sdist-built wheel: **PASSED** — three archives,
  zero unexpected archive findings, and seven unchanged reviewed tracked-tree
  findings.
- Outside-checkout installation of the sdist-built wheel plus
  `scripts/smoke_installed_package.py`: **PASSED** — console script and JSON/
  ZIP fake-service smoke reported package version `0.1.0` from site-packages.
- `.venv/bin/python -m twine check` on direct wheel, sdist, and rebuilt wheel:
  **PASSED**.
- `systemd-analyze verify deploy/zap-it-local.service`: **PASSED**.
- No live GPU phase or inference request: **SKIPPED** by order scope; the
  canonical GPU marker was skipped honestly for missing explicit opt-in.

## CI/checks

All seven required checks were **PASSED** on implementation SHA
`ab85fe7f3c46ea3bf6cd3f0a75d517e9486031a9`; none was pending, missing,
cancelled, failed, or unexpectedly skipped:

- `static (format, lint, build)`: **PASSED**, run `33283989022`, job
  `99183855532`.
- `release (artifact audit)`: **PASSED**, run `33283989022`, job
  `99183855681`.
- `tests (py3.10)`: **PASSED**, run `33283989022`, job `99183855684`.
- `tests (py3.11)`: **PASSED**, run `33283989022`, job `99183855678`.
- `tests (py3.12)`: **PASSED**, run `33283989022`, job `99183855610`.
- `Analyze (python)`: **PASSED**, run `33283988986`, job `99183855402`.
- `CodeQL`: **PASSED**, check run `99183971364`.

After this report-only SELF child is published, the same seven named checks
must be inspected on the final report head; the final verification below
records their result and no further mutation is permitted.

## GPU/service/resource evidence

- User unit `zap-it-lan.service`: **PASSED** read-only preservation check —
  enabled, active/running, PID `528963`, `NRestarts=0`.
- Exactly one listener: **PASSED** — `10.8.132.76:17891`, owned by PID
  `528963`.
- Assigned physical GPU: **PASSED** — index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; one compute process, PID
  `528963`, reported 11,178 MiB used. The process environment asserts
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, and the operator
  physical index 0; the application mapping is logical `cuda:0`.
- Endpoints: **PASSED** — `/healthz` 200, `/readyz` 200, unauthenticated
  `/v1/capabilities` 401, `/docs` 404, and `/openapi.json` 404.
- RAM workspace: **PASSED** — `/dev/shm/slaif-zap-it` mode 0700 and empty.
- Operator environment digest: **PASSED** — mode 0600 and digest
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- Service log sensitivity scan: **PASSED** — zero matches for credential,
  authorization, prompt, answer, traceback, OOM, temporary-path, or shared-
  memory patterns. No key, raw image/config, or request data was printed,
  copied, persisted, or added to OAP evidence.
- No service restart/reload/reconfiguration, live inference, GPU mutation,
  unrelated-process action, port/network/firewall action, or credential change
  occurred.

## Documentation/provenance

The current testing documentation now accurately identifies the independent
test-owned source-space and resize/contour oracles and observed resident-holder
guards. Historical 018-a order/report bytes were not changed. The evidence is
generated CPU/fake boundary and provenance evidence, not semantic accuracy,
recall, or precision measurement.

## Deferred human adjudication

- Critical register action: NONE.
- No CRITICAL entry was read, appended, edited, reordered, or closed.

## Safety/scope confirmations

- Only generated arrays, bounded fake processors/holders, and read-only service
  probes were used; no real model, download, GPU test, or live inference was
  used.
- No request image/config/result was persisted by the tests or service probes;
  no repository output path was used for request data.
- Only the five implementation/control paths listed above are in the
  implementation commit. This final commit changes only this report path.
- The PR remains open for strategic review. Coding did not merge, accept,
  release, tag, publish, or advance the objective.

## Limitations/blockers

The checks are deterministic boundary/provenance checks and do not measure
semantic-model accuracy. GPU integration remains skipped because this order
authorizes no live GPU phase. Final-head CI is required to be green after the
report SELF child and is verified below before signaling.

The strongest reason not to merge is that a post-implementation test can still
compare production logic to itself and pass while both are wrong. This round
answers that risk for the reviewed seams with bounded brute-force source-space
and contour oracles, an explicit test-owned center mapping, uniquely identified
markers, and fail-if-invoked resident initialization seams. It does not claim
that unrelated future tests cannot have the same weakness.

## Factual strategic follow-up

PR #74 remains open at the implementation/control head pending this immutable
report child and strategic review. The final child must have parent
`ab85fe7f3c46ea3bf6cd3f0a75d517e9486031a9`, change only
`oap/reports/018-b-report.md`, and be the final remote head before the exact
response FIFO `OK` is sent.
