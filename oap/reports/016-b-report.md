# OAP Coding-Agent Report — 016-b

## Work order

- Identifier/order/objective/PR mode: `016-b` — correct heatmap zero coverage,
  enlarge raw candidate tiles and close raw-visualization proof gaps; amend
  Objective-016 PR #72.

## Status

COMPLETE

## Executive summary

Corrected the raw-SAM2 overlap heatmap so zero-coverage pixels remain exactly
black, including after diagnostic nearest-neighbor resizing. Small padded
candidate crops now enlarge into their fixed 320x240 tiles while the three
full-image diagnostics retain their no-upscale policy. The actual Pillow label
drawing seam is bounded and preserves complete ID/score fields, empty contact
sheet cells use fixed fills, and overlapping candidates remain isolated in
their own tile rectangles.

Added a pure cross-field manifest finalization/validation seam, wired before
service artifact-sink publication, and reused it from the Pydantic manifest
schema. Added CPU/API proofs for arithmetic tampering, level/name stripping,
hostile frame/candidate metadata, JSON/ZIP parity and the legacy rectangular
JPEG path. The corrected implementation passed the full CPU/static/release
gates, all current-head GitHub checks, and the bounded live requalification.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/72`
- PR state: OPEN, non-draft, MERGEABLE; coding did not merge or enable
  auto-merge.
- Base branch: `main` at
  `8081152403657f5e737ab0b491e0b89f587209e1`.
- Starting PR head SHA: `a0f6fabbf74d30a8d4a0fe0c99966cd3d2d10d21`.
- Branch: `oap/016-a-bounded-raw-sam2-visualizations`.
- Implementation head SHA: `8097166285db44020b92b7b9661688a619dd8994`.
- Report publication commit: SELF.
- New PR: no. Amended existing PR: yes. Coding merge/auto-merge: NO.

## Changes/files

Implementation commit
`8097166285db44020b92b7b9661688a619dd8994` contains the exact active selector,
016-b order transcript, implementation, tests and documentation changes:

- `oap/active`
- `oap/orders/016-b-correct-heatmap-and-close-raw-visualization-proofs.md`
- `src/core/raw_visualizations.py`, `src/core/engine.py` and
  `src/core/__init__.py`
- `src/service/capabilities.py`, `src/service/fake_engine.py` and
  `src/service/schemas.py`
- `tests/test_raw_sam2_visualizations.py` and the capability assertion in
  `tests/test_sam2_configuration.py`
- `README.md` and the applicable current `docs/ALGORITHMS.md`, `docs/API.md`,
  `docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`
  and `docs/SERVICE-DATASHEET.md`

The report publication commit changes only `oap/reports/016-b-report.md`. No
prior OAP report or CRITICAL entry was edited.

## Acceptance evidence

1. **Black-zero heatmap — PASSED.** The ramp is applied only where overlap is
   positive. CPU proofs cover all-zero, mixed 0/1/2, source-resolution union
   equivalence and nearest-neighbor resized equivalence; positive overlap
   values remain non-black and distinct.
2. **Enlarged candidate tiles — PASSED.** The tile-only scale is the smaller
   of `320/crop_width` and `240/crop_height` without a 1.0 ceiling. Generated
   border-touching one-pixel and disconnected masks prove enlargement,
   multi-pixel selected regions, exact-mask-only alpha and bounded content
   placement. Full-image diagnostics still use deterministic no-upscale
   dimensions and nearest-neighbor downscaling.
3. **Label and pagination seams — PASSED.** The actual draw helper records the
   complete finite label `C0001  IoU 0.843  stability 0.912`, exact `n/a`
   fields, and `C24576` without truncation; returned Pillow text bounds stay
   inside the 320x28 label bar. The 13-candidate second page proves candidate
   13 placement, neutral/background fixed empty cells, rectangle isolation and
   independent visibility for overlapping source masks. No client label,
   prompt, question, frame or path metadata is used.
4. **Manifest arithmetic — PASSED.** `validate_raw_sam2_manifest` and
   `finalize_raw_sam2_visualization` validate generator/empty/represented/
   truncated counts, IDs, page names/counts, source pixel totals, bounded
   histogram/overflow flags and deterministic diagnostic dimensions before
   sink publication. Tampered representative counts, IDs, page names, pixel
   totals and histogram facts are rejected with the fixed typed core error;
   Pydantic reuses the same arithmetic validator.
5. **API levels and legacy routing — PASSED.** For uploaded
   `mask_generator.debug: true` at L0, L1 and L2, the effective fake-engine
   config has debug false, `service.sam2.raw_visualization` is absent and no
   artifact begins with `sam2-`; the L1 identity artifact remains unchanged.
   Two service-safe runs with different frame IDs and hostile candidate
   metadata produced the same fixed raw names and sanitized label. The existing
   trusted legacy test continues to require `<frame>_sam2-patch0000.jpg`.
6. **JSON/ZIP parity — PASSED.** Focused tests and live requests compare raw
   metadata, PNG bytes, SHA-256, sizes and `image/png` media descriptors across
   JSON and ZIP representations and repeats.

## Verification

- `.venv/bin/pytest -q tests/test_raw_sam2_visualizations.py`: PASSED — 36
  focused raw renderer/API/resource/schema tests.
- `.venv/bin/pytest -q tests/test_raw_sam2_visualizations.py tests/test_sam2_configuration.py tests/test_service_api.py tests/test_service_units.py tests/test_core_engine.py tests/test_run_frame_pipeline.py`: PASSED — 388 focused renderer/API/resource/schema/core/legacy tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 725 passed, 1 explicit GPU skip, 79.96% total coverage; the skip is the opt-in GPU integration test without live-test enablement.
- `.venv/bin/ruff format --check .`: PASSED — 146 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh scripts/serve_local.py scripts/smoke_local_service.py`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED — wheel and sdist built with the corrected core module.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — zero unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — seven reviewed baseline findings before report publication, unchanged.
- `.venv/bin/python -m twine check dist/*`: PASSED — wheel and sdist.
- Required CPU, hostile-YAML, schema, API, artifact, auth, metrics, lifecycle,
  post-filter, BLIP3, package, resource, cancellation/timeout and legacy
  regressions: PASSED through the canonical suite.

One live harness attempt was FAILED as a post-check: three native large-crop
HTTP requests returned 200, then the harness correctly detected that its
2784x2088 source required diagnostic downscaling, contrary to that harness's
temporary no-downscale assertion. No inference request failed and no second
service restart was performed. The corrected 256x192 in-memory harness below
passed.

## CI/checks

All seven required checks are SUCCESS/COMPLETED on implementation SHA
`8097166285db44020b92b7b9661688a619dd8994`:

- `static (format, lint, build)` — PASSED, CI run `33224672344`.
- `tests (py3.10)` — PASSED, CI run `33224672344`.
- `tests (py3.11)` — PASSED, CI run `33224672344`.
- `tests (py3.12)` — PASSED, CI run `33224672344`.
- `release (artifact audit)` — PASSED, CI run `33224672344`.
- `Analyze (python)` — PASSED, CodeQL workflow run `33224672319`.
- `CodeQL` — PASSED, check run `99025968267`.

## GPU/service/resource evidence

- Exact assigned target only: physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; Torch
  `2.5.1+cu124`, CUDA build `12.4`. The process exposed one device as logical
  `cuda:0`; no unassigned device was allocated, stopped or reconfigured.
- Before the authorized restart, the user unit was enabled/active with PID
  `465389`, one listener on `10.8.132.76:17891` and one assigned-card compute
  process. Exactly one controlled restart of only user-level
  `zap-it-lan.service` produced PID `476019`; it remained enabled/active/ready
  with `NRestarts=0`, one listener and one compute process throughout the
  qualification.
- Post-readiness endpoint matrix: health 200, readiness 200, missing/wrong
  credentials 401, authenticated capabilities/metrics 200, `/docs` 404 and
  `/openapi.json` 404.
- Corrected live JSON/ZIP/ZIP requests were all HTTP 200. Raw metadata and all
  six raw PNG members were identical across repeats and JSON/ZIP parity. The
  256x192 diagnostic had 28 raw/28 represented candidates, three pages, six
  raw artifacts, 47,560 covered pixels, 1,592 uncovered pixels and maximum
  overlap 3. Union-black, uncovered-white and heatmap-zero masks matched;
  every covered heatmap pixel was non-black; contact-sheet content and label
  regions were nonblank.
- The live proposal counts differ from the 016-a qualification (47,491
  covered / 1,661 uncovered / maximum overlap 5) because the regenerated
  proposal set changed; the new relational assertions and manifest arithmetic
  passed. The earlier native large-crop check was not used as evidence.
- Metrics showed exactly one successful `component="registry"`,
  `outcome="success"` initialization and zero residency transitions. Sampling
  during corrected requests observed assigned-GPU peak used `11,617 MiB`,
  minimum free `12,507 MiB`, and peak service RSS `4,557,372 KiB`; final RSS
  was `4,556,380 KiB`.
- Final assigned-GPU snapshot was 11,617 MiB used and 12,507 MiB free. The
  `/dev/shm` root mode was 0700, the request workspace had zero entries, and
  the configured mode-0600 environment digest remained
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- Sanitized journal checks found no bearer credential, fixture name, prompt,
  answer or question content. No request image/config/response bytes were
  persisted or printed.

## Documentation/provenance

Updated the applicable README, API, configuration, algorithms, core,
output-parity, runbook and service-datasheet statements to distinguish enlarged
candidate tiles from diagnostics that never upscale. Capabilities now states
the candidate-tile enlargement policy while retaining the exact Objective-016
constants. Model identities/revisions, licenses, auth/network/device/cache/
residency policy, response defaults, filters, CLIP/BLIP3, geometry, identity
PNG, YOLO and CRIT-0001 were not changed.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Exactly active order `016-b` was executed; no adjacent order was selected.
- Exactly PR #72 was amended for numeric Objective 016; no new PR, merge,
  auto-merge, release/tag/upload, key rotation, firewall/VPN/network change,
  unrelated unit change or history rewrite occurred.
- Only the exact order-assigned physical GPU0/UUID was used for live work; all
  unassigned devices and unrelated processes remained protected.
- The active selector and exact 016-b order were committed with the bounded
  implementation before the implementation SHA was captured. The final
  report-only commit changes only this report and has the implementation SHA as
  its first parent.
- No request data entered repository output directories, persistent disk or
  OAP evidence.

## Limitations/blockers

None for the ordered scope. The live proposal counts changed from the prior
qualification as disclosed above. Deterministic PNG byte identity is claimed
for equal inputs in the pinned Pillow/NumPy environment, not across arbitrary
encoder/library versions. Live evidence is bounded local qualification, not a
segmentation-quality, solar-array recall/precision, production-readiness or
public-deployment claim.

## Factual strategic follow-up

PR #72 is open with all seven current-head checks green and is ready for
strategic review/acceptance. Coding has not merged it, enabled auto-merge,
selected a next order or authorized deployment.
