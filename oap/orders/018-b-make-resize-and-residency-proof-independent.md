# OAP Work Order 018-b — Make resize and residency proof independent

## Objective

Amend Objective-018 PR #74 with two narrow proof-quality corrections found by
strategic review of the immutable 018-a report and actual tests. Do not change
runtime/product behavior. Make the tiny-mask resize/contour expected result
independent of production mapping/dilation helpers, and replace the synthetic
constant called an initialization count with observed fail-if-reinitialized
resident-holder seams.

## Verified GitHub state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Default branch/base SHA: `main` at
  `03def697373f2ae83d03494315aa96c800f0bcdf`.
- Existing PR: #74, `Objective 018: close mask-view acceptance matrix`.
- Branch: `oap/018-a-close-mask-view-acceptance-matrix`.
- PR head/report commit:
  `5fbcf43561a023e771fdfde6fb5795275b50227e`.
- 018-a implementation parent:
  `74ab1d8dc685f2ea1ce8d9c0e38bb6fac9de5184`.
- The report head is a correct one-path report-only SELF commit.
- PR is open, non-draft, mergeable/clean, with all seven final-head checks
  successful.
- Amend this exact PR and branch. Do not create a new PR, merge, or enable
  auto-merge.

## Strategic review findings

### 1. Tiny-mask expected mapping is not independent

`test_tiny_mask_builds_source_space_crop_before_resize_and_contour` imports and
calls production `_nearest_indices` and `_square_dilation` to build the expected
target/support mapping and contour. That makes the new test partly tautological
and contradicts the 018-a order's strongest-reason answer requiring independent
expected masks/mappings. `TESTING.md` currently overstates that this is an
independent assertion.

### 2. Initialization count is asserted, not observed

`_CandidateViewEngine.holder_initialization_counts` is initialized directly to
`{"clip": 1, "blip3": 1}` and never updated by an initialization seam. Asserting
that constant does not prove A/B/A avoided resident-holder reinitialization.
Stable holder IDs are useful but do not by themselves fail if a hidden
initialization path runs and discards a new holder.

These are ordinary test defects. They do not meet the CRITICAL threshold and do
not imply a runtime defect.

## Required corrections

### 1. Test-owned resize and contour oracles

In the tiny-mask test, do not import or call production `_nearest_indices`,
`_square_dilation`, `_circular_dilate`, `build_mask_views`, or another production
helper to derive the expected mask mapping/support/contour that is being checked.
The production builder may still produce the actual result under test.

Construct expected values from the original generated source image/mask and the
documented contract using test-owned logic:

- independently compute the exact source-space Euclidean `D` for the small fixed
  radius, e.g. by a bounded brute-force distance oracle over source coordinates;
- derive the expected tight context bbox from that independent `D`;
- independently compute target-pixel-center to source-pixel nearest-neighbor
  indices from the documented center mapping, without calling the production
  mapper;
- use Pillow bilinear only as the documented external interpolation primitive
  for RGB, starting from independently neutralized source crops;
- independently build the expected positive-width contour with a bounded
  brute-force/test-owned square-neighborhood oracle, not the production
  dilation helper; and
- keep the width-zero, prohibited-pixel, target/support zero, exact target
  restoration and repeated-byte assertions.

Assert the independently derived source bbox, masks, mapped masks, contour and
paired RGB equal the production output. Include an explicit assertion that both
the inside-radius and outside-radius marker are absent from target-only, while
only the eligible inside marker appears in context.

The oracle must be simple, bounded and visibly independent; do not copy the
production optimized distance-transform code.

### 2. Observed resident-holder non-reinitialization

Remove `holder_initialization_counts` or turn it into real observed evidence.
In the stable-service A/B/A test, use `monkeypatch` or equivalent bounded
instrumentation at the actual fallback construction seams:

- the CLIP module's `initialize` path must raise/assert if invoked while the
  supplied resident `clip_filter` exists;
- BLIP3 `_Blip3QA` construction (or the exact holder-construction seam used by
  current code) must raise/assert if invoked while supplied resident `blip3_qa`
  exists; and
- retain exact stable holder identity assertions across every request.

Record the actual forbidden-initialization attempt list/count and assert it is
empty. It is acceptable and expected for BLIP3 to construct a request-local
rule/filter wrapper around the same QA holder; do not conflate that with model
holder reinitialization.

Exercise this observed guard across the L0-L3 A/B/A matrix already in the test.
Do not replace runtime code or monkeypatch the behavior being proven (candidate
view construction/model input generation).

### 3. Honest documentation

Adjust `TESTING.md` only as needed so “independent” accurately describes the
test-owned Euclidean, nearest-neighbor and contour expected values and the
fail-if-called resident initialization seams. Do not rewrite 018-a's immutable
order/report. Add this 018-b order and selector normally.

## Scope and non-goals

Expected implementation paths are `tests/test_mask_views.py`,
`tests/test_candidate_view_api.py`, `TESTING.md`, `oap/active`, this order, and
the new immutable 018-b report. No runtime/product module, schema, default,
dependency, lockfile, model, environment, unit, credential, GPU or network
change is authorized.

No real model, download, GPU test, live inference, service restart/reload,
release/tag/publish, public exposure, destructive history operation or CRITICAL
register mutation is allowed.

If the independent oracle exposes a runtime defect, preserve the failing case
and return PARTIAL without weakening it or modifying runtime code. Strategic
will decide a later 018-c correction.

## Verification

- Run the corrected focused tests, including the exact tiny-mask, literal CLIP
  processor, BLIP3 QA, L0-L3 A/B/A and source-identity cases.
- Run the full canonical CPU/offline suite with coverage and record exact
  pass/skip/warning counts and coverage.
- Run Ruff format/check, compileall, documentation integrity, `git diff --check`,
  wheel/sdist builds, release artifact verification, sdist-built outside-tree
  smoke, wheel comparison, archive/tracked-tree secret scans, `twine check`, and
  tracked systemd-unit verification.
- Push the correction implementation/control commit and require all seven
  current-head CI/CodeQL checks successful.
- Publish exactly one immutable `oap/reports/018-b-report.md` as a report-only
  SELF child; require all seven checks successful again on that final head.

## Service preservation

Do not restart, stop, reload, reconfigure or send inference to
`zap-it-lan.service`. Read-only start/final checks must show the already-qualified
service remains enabled, active and ready on PID `528963` unless an external
failure changed it, `NRestarts=0`, one listener at `10.8.132.76:17891`, one
assigned-GPU process on physical GPU0 UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, empty mode-0700
`/dev/shm/slaif-zap-it`, and unchanged mode-0600 environment digest
`bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.

Health/readiness must remain 200, unauthenticated protected endpoints denied,
and docs/OpenAPI 404. Never print, log, copy, rotate or mutate the key. Preserve
private-LAN scope and every unrelated service/device/process.

## Acceptance criteria

1. The tiny-mask expected `D`, bbox, target/support nearest mapping and contour
   are derived by visibly independent test-owned oracles, not production helpers.
2. Independent expected masks/RGB/contour equal actual builder/pair outputs;
   zero/prohibited/target-restoration/determinism assertions remain green.
3. Both dilation markers are explicitly absent target-only; only the eligible
   marker appears in context.
4. The A/B/A test has fail-if-invoked guards on actual CLIP and BLIP3 holder
   construction seams, observes zero attempts, and retains stable holder IDs.
5. Documentation claims exactly the evidence implemented.
6. Diff remains test/docs/OAP-only; all focused/full/static/package/final-head
   CI and CodeQL evidence is green.
7. The service remains unchanged, running and ready with no restart or inference.

The strongest reason not to merge remains that a test can compare production
logic to itself and pass while both are wrong. Answer it with small brute-force
source-space and contour oracles, an explicit test-owned center-mapping formula,
uniquely identifiable pixels and fail-if-called initialization seams.

## Deferred human adjudication

- Decision: NONE

## Publication and report contract

- Include this exact order and `oap/active` in the correction
  implementation/control commit on PR #74.
- Record the order SHA-256 and exact parent/head topology.
- After all implementation-head evidence and checks pass, publish exactly one
  immutable `oap/reports/018-b-report.md` report-only SELF child changing only
  that path, and verify its parent plus final-head checks.
- Report each finding/correction, independent-oracle design, observed
  initialization evidence, file/diff scope, all test/package/CI facts, service
  preservation, limitations, strongest reason not to merge, and critical action
  `NONE`.
- Status is COMPLETE only if every criterion passes. Otherwise report PARTIAL
  and do not claim the gap is closed.
- Coding must not merge, accept, enable auto-merge, release, restart the service,
  send live inference or print the key.
