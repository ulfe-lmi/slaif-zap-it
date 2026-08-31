# OAP Coding-Agent Report — 019-a

## Work order

- Identifier/order/objective/PR mode: `019-a`, implement the single-image
  BLIP3 candidate view and create one new PR.
- Active selector: `019-a`.
- Immutable order SHA-256 in the implementation commit:
  `832924040b9a9b11a8a8a01c5a2074acaaa0fcee2b503579b28ef56de96477ba`.

## Status

COMPLETE

## Executive summary

Replaced the BLIP3 target/context two-panel input with one deterministic,
request-local source-space composition per applicable candidate. The compositor
uses exact squared-Euclidean support and contour dilation, an independently
clamped centered crop, Pillow Gaussian blur for all non-support scene pixels,
byte-exact support restoration, and bounded bilinear RGB resizing. A crop that
cannot contain support plus contour is rejected locally before image/model or
debug work without changing the candidate label, score, or answer. BLIP3
reuses the identical one-image PIL input for all questions about a candidate;
L3 composition records are separate from per-question debug artifacts.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- PR: [#75 — Objective 019: single-image BLIP3 candidate view](https://github.com/ulfe-lmi/slaif-zap-it/pull/75).
- PR state: open, non-draft, base `main`; coding did not merge, accept, close,
  or enable auto-merge.
- Branch: `oap/019-a-single-image-blip3-candidate-view`.
- Verified remote base SHA: `4acff3a8f7717a08481b86338453d09e754c1e86`.
- Starting local execution SHA: `40824614a613acd799190735f5f2346a2d833d63`.
- Implementation head SHA: `573f5ed4d92e1b988baf325c65140b334eeed9ee`.
- Implementation parent: `4acff3a8f7717a08481b86338453d09e754c1e86`.
- Report publication commit: SELF.
- New PR: yes. Existing PR amended: no.
- Remote implementation branch was verified at the implementation SHA before
  report publication.

## Changes/files

The implementation/control commit contains exactly these 29 paths:

- `ARCHITECTURE.md`, `README.md`, `TESTING.md`.
- `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`.
- `modules/verifier/__init__.py`, `modules/verifier/blip3.py`.
- `oap/active`, `oap/orders/019-a-single-image-blip3-candidate-view.md`.
- `src/__init__.py`, `src/batch.py`, `src/core/__init__.py`,
  `src/core/engine.py`, `src/core/mask_views.py`, `src/core/results.py`.
- `src/service/capabilities.py`, `src/service/envelope.py`,
  `src/service/schemas.py`, `src/service/yaml_input.py`.
- `tests/test_candidate_view_api.py`, `tests/test_core_engine.py`,
  `tests/test_mask_views.py`, `tests/test_service_units.py`,
  `tests/test_verifier_blip3.py`.

No dependency, lockfile, model identity/revision, generation limit, auth,
network policy, service unit/environment, visualization, geometry, YOLO,
release, or CRITICAL-register change was made.

## Acceptance evidence

1. **Single-image BLIP3 seam — PASSED.** The production filter composes one
   image per applicable candidate and reuses the same PIL object for multiple
   questions. No pane, divider, duplicate, fill, dimming policy, or untouched
   rectangular context remains in the current product path.
2. **Exact source authority — PASSED.** Generated nonuniform RGB arrays and
   boolean masks independently verify inclusive raw/support bboxes, exact
   Euclidean dilation, exterior-only contour, Pillow Gaussian blur, source-pixel
   restoration, and final bilinear resize bytes.
3. **Geometry/configuration — PASSED.** The new BLIP3 fields and defaults are
   strict and request-local: `single_dilated_blur`, context `0.20`/`0..64`,
   crop multiplier `2.0`, blur fraction `0.15`, enabled contour fraction
   `0.02`, contour width `1..3`, and RGB `[255,224,0]`. Raw/effective radii
   and widths, inclusive bboxes, half-open crop, sigma, and dimensions are
   recorded. CLIP retains its prior field set and view behavior.
4. **Candidate-local rejection — PASSED.** The dedicated
   `Blip3CandidateViewRejected` reason is exactly
   `crop_cannot_contain_support_and_contour`. Rejected candidates receive no
   QA call or debug artifact, retain their CLIP fields, and do not block a
   following valid candidate.
5. **Question/model/debug identity — PASSED.** Client questions remain
   byte-preserved before the exact fixed instruction. Captured QA input and
   decoded lossless debug PNG are byte-identical; fixed names remain
   `blip3-verification-CANDIDATE-####-QUESTION-####.png`.
6. **L3 metadata and response parity — PASSED.** The separate bounded
   `blip3_candidate_views` record is one-per-applicable-candidate and includes
   status/reason, IDs, radii, widths, bboxes, sigma, source-composite and
   model-input dimensions. It is emitted only at L3 and is identical in JSON
   and ZIP manifests; debug artifacts remain one-per-question.
7. **Resource/state isolation — PASSED.** BLIP3 debug admission reserves the
   exact one-image uncompressed RGB size before QA, rejected candidates consume
   no artifact budget, and generated A/B/A API requests change and restore
   view bytes without resident-holder replacement.
8. **Regression compatibility — PASSED.** Legacy CLI/batch seams, CLIP
   candidate views, final labels/objects, identity/source indices, response
   bounds, auth/privacy, and unrelated pipeline tests remain green.
9. **Verification gates — PASSED.** Focused and full CPU tests, package/static
   checks, release audits, secret baseline, docs, systemd-unit verification,
   and all seven implementation-head GitHub checks passed. The canonical GPU
   marker is honestly skipped because 019-a authorizes no live GPU phase.
10. **Documentation/service preservation — PASSED.** Current contract docs
    agree with capabilities and contain no stale BLIP3 paired-view contract.
    The existing service remained running and unchanged during coding.

## Verification

- `.venv/bin/pytest -q tests/test_mask_views.py tests/test_verifier_blip3.py tests/test_candidate_view_api.py tests/test_sam2_configuration.py::test_capabilities_are_authenticated_static_deterministic_and_explicit`: **PASSED** — 35 passed, 1 warning.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: **PASSED** — 760 passed, 1 honest GPU test skipped, 2 warnings, total coverage `80.92%`.
- `.venv/bin/ruff format --check .`: **PASSED** — 149 files already formatted.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules scripts tests`: **PASSED**.
- `.venv/bin/python scripts/check_documentation.py`: **PASSED** — 27 current documents.
- `git diff --check`: **PASSED**.
- `.venv/bin/python -m build --wheel --sdist`: **PASSED** — direct wheel and sdist built; existing setuptools deprecation warnings only.
- `.venv/bin/python -m twine check dist/*`: **PASSED** — direct wheel and sdist.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: **PASSED** — 67 wheel members and 160 sdist members.
- Wheel built from the sdist plus `scripts/verify_release_artifacts.py --compare-wheels`: **PASSED** — direct and sdist-built wheel member manifests/bytes matched.
- `.venv/bin/python scripts/scan_release_artifacts.py --baseline .secrets.baseline --tracked-tree dist/*`: **PASSED** — 7 unchanged reviewed tracked-tree findings and 0 unexpected archive findings.
- Outside-checkout wheel installation plus `scripts/smoke_installed_package.py`: **PASSED** — site-packages import, console script, JSON smoke, and ZIP smoke all reported version `0.1.0`.
- `systemd-analyze verify deploy/zap-it-local.service`: **PASSED** — exit 0.
- Focused documentation/runtime/release tests: **PASSED** — 52 passed.
- Live model/download/inference/GPU phase: **SKIPPED** — not authorized by this order; no GPU allocation or model call was performed.

## CI/checks

All seven required implementation-head checks were **PASSED** on
`573f5ed4d92e1b988baf325c65140b334eeed9ee`:

- `static (format, lint, build)`: **PASSED**, CI run `33379301269`, job
  `99447746738`.
- `release (artifact audit)`: **PASSED**, CI run `33379301269`, job
  `99447746649`.
- `tests (py3.10)`: **PASSED**, CI run `33379301269`, job `99447747013`.
- `tests (py3.11)`: **PASSED**, CI run `33379301269`, job `99447746690`.
- `tests (py3.12)`: **PASSED**, CI run `33379301269`, job `99447746776`.
- `Analyze (python)`: **PASSED**, CodeQL run `33379301160`, job
  `99447746174`.
- `CodeQL`: **PASSED**, check run `99447982853`.

The final report-only child must be checked again on the final PR head before
the FIFO signal; no later non-report mutation is permitted.

## GPU/service/resource evidence

- No live GPU phase or inference was authorized or performed. Read-only
  verification found the exact assigned physical device unchanged: index `0`,
  UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`,
  NVIDIA GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; one compute process,
  PID `528963`, reported 13,408 MiB used. The process environment retained
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, and physical index
  `0`; application visibility is logical `cuda:0`.
- The user unit remained enabled/active/running with PID `528963`,
  `NRestarts=0`, and active timestamp `Sun 2026-08-30 01:28:56 CEST`.
- The sole required listener remained `10.8.132.76:17891`. Read-only endpoint
  checks returned `/healthz` 200, `/readyz` 200, unauthenticated
  `/v1/capabilities` 401, `/docs` 404, and `/openapi.json` 404.
- `/dev/shm/slaif-zap-it` remained mode `0700` and empty. The mode-`0600`
  operator environment retained SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- No unassigned GPU, unrelated process/service, firewall, VPN, port, network,
  credential, model cache, or host resource was modified. The API key was not
  printed or read into report evidence.

## Documentation/provenance

`ARCHITECTURE.md`, `README.md`, `TESTING.md`, `docs/ALGORITHMS.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`,
and `docs/SERVICE-DATASHEET.md` document the exact field surface, formulas,
source crop/blur/resize sequence, bbox conventions, rejection semantics,
single-image debug identity, bounded records, resource admission, and the
pixel-boundary—not semantic-accuracy—nature of this evidence. Historical OAP
orders/reports and history documents were not rewritten. The order selector is
`019-a`; its active file hash is
`1d3bf9f89785b900331cab679422044e23a71467c40708910a0deb9eafa4cc76`.

## Deferred human adjudication

- Critical register action: NONE.
- No CRITICAL entry was read, appended, edited, reordered, or closed.

## Safety/scope confirmations

- Only generated CPU arrays, bounded fake QA/processor seams, and read-only
  service/GPU probes were used; no real model, download, or live inference was
  used.
- No request image/config/result was persisted by the service or probes; no
  repository output path was used for request data.
- The implementation commit contains the exact active selector and order. The
  final child is required to change only this report path.
- PR #75 remains open for strategic review. Coding did not merge, accept,
  release, tag, publish, or advance the objective.

## Limitations/blockers

The generated-array and fake-model evidence proves deterministic pixel,
containment, provenance, resource-admission, and artifact identity behavior. It
does not measure BLIP3 semantic accuracy, recall, or precision. GPU/model/live
inference evidence is intentionally skipped under this order. Final-head CI
must be green after this report-only SELF child.

The strongest reason not to merge is that a visually plausible image could be
formed by a bbox-based or debug-only path while a different image reaches QA.
This round addresses that risk with independent source-coordinate Euclidean and
Pillow oracles, asymmetric generated arrays, literal QA capture, decoded-PNG
byte equality, one-composition/multi-question tests, and zero-call containment
rejection/resource tests. It does not claim semantic-model accuracy.

## Factual strategic follow-up

PR #75 remains open at implementation head
`573f5ed4d92e1b988baf325c65140b334eeed9ee` pending strategic review. The final
child must have that exact parent, change only
`oap/reports/019-a-report.md`, be pushed as the remote PR head, and be followed
by the exact response FIFO `OK` after all final-head checks are green.
