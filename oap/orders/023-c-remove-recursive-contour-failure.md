# OAP Work Order 023-c — remove recursive contour failure

## Objective

Amend Objective 023 PR #87 in place. Replace the recursive external-boundary
repair walk with an iterative, candidate-local traversal that preserves the
exact established contour ordering and therefore preserves every existing
candidate-view pixel. Close the remaining deterministic timing, explicit-reject,
and artifact-budget proof gaps identified during strategic review.

This is a narrow robustness continuation. Do not change the accepted
centroid-radial geometry, the default `reject` policy, fallback eligibility,
crop/support/contour/blur/resize semantics, recognition behavior, or API field
set.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote `main` remains
  `515d5200e43feb0fa8b48c0762157491487dac3b`.
- Amend only open, mergeable PR #87 on branch
  `oap/023-a-centroid-radial-mask-chord-fallback`.
- Reviewed remote head is immutable 023-b report-only SELF commit
  `04d97b85b04ced307afe40932ba6226bb57a75bc`; its first parent is 023-b
  implementation commit `9887f00f46740f045acf46b606d24083d4c632e2`.
- All seven CI/CodeQL checks are green on both SHAs. Do not rewrite, amend,
  delete, or alter either `oap/reports/023-a-report.md` or
  `oap/reports/023-b-report.md`.
- Continue on the same branch and PR. Do not merge, enable auto-merge, create a
  new PR, or touch other branches/PRs.

Refresh GitHub, worktree, active/order state, `CRITICAL.md`, service/listener,
fixture/evidence hashes, tmpfs, and GPU/process facts before mutation. Preserve
the authoritative 023-a and 023-b evidence directories.

## Independently reproduced blocker

`src/core/radial_geometry.py::_moore_walk` falls back from the ordinary Moore
walk to nested recursive `visit()` calls over the external-boundary graph. A
valid high-boundary mask can exceed Python's recursion depth even though its
dimensions are far below the service's documented 8192-by-8192 image limit.

Strategic reproduced this at 023-b report head using a boolean `n`-by-`n` comb:

```python
mask = np.zeros((n, n), dtype=bool)
mask[n // 2, :] = True
for x in range(0, n, 150):
    mask[:, x] = True
_contours(mask)
```

- `n=500`: succeeds, 2,496 foreground pixels, normalized contour length 4,990;
- `n=600`: raises `RecursionError`, with only 2,996 foreground pixels;
- `n=800`: raises `RecursionError`, with only 5,594 foreground pixels.

The failure occurs before radial support or BLIP3 verification and can escape as
an inference failure for an otherwise valid opt-in candidate. This contradicts
the required deterministic behavior for concave, thin, fragmented/
high-boundary masks and the rule that valid candidates must not be lost because
fallback geometry cannot execute.

The recursive branch is not dead code. A 400-by-400 comb with teeth every four
pixels enters it and currently produces one normalized contour of shape
`(80598, 2)` with SHA-256
`e7220b5627a0d3185f7467fa16bc58c2877fe5e4659e3ad807c43d8d8c77b3ba`
over its contiguous `int64` `(x,y)` bytes. This is the compatibility oracle.

All 122 BLIP3 debug PNGs from the 023-a and literal-head 023-b live runs were
independently compared after the coding response: every encoded PNG hash and
every decoded contiguous RGB hash matches, including the twelve fallback views.
The final labelled visualization also matches exactly. Preserve this stronger
compatibility result.

The 023-b benchmark disclosure was also independently resolved. With the
service resident and idle, strategic reran the exact command and obtained
`806.603981`, `805.820917`, and `815.454911` ms, median `806.603981` ms,
maximum `815.454911` ms, status `PASSED`, and the same deterministic digest
`50d3f782c6de4ed272888bff896973e4884cb7d505fbcb8ad93b4a3580e8a8fe`.
No performance redesign or threshold change is authorized.

## Required implementation

### Iterative traversal with exact order preservation

Replace only the recursive DFS repair inside `_moore_walk` with an explicit
iterative stack. Preserve exactly the recursive algorithm's observable order:

- same external-boundary set and 8-neighbour definition;
- same start point;
- same neighbour sort key and stable order;
- depth-first visitation;
- same parent sample appended after returning from each child;
- same handling of any deterministically sorted unvisited remainder;
- same subsequent `_normalize_contour` behavior, component order, hole
  exclusion, quadrilateral adjacency, and metadata count.

Use explicit stack frames containing the current point and next-neighbour
position, or an equivalently bounded iterative formulation. Do not call a
recursive helper, change Python's recursion limit, catch `RecursionError`, drop
boundary samples, split a component, simplify the contour, or substitute a new
contour library/dependency.

Memory must remain candidate-local and O(external boundary sample count) plus
the already bounded geometry arrays. The traversal must be deterministic across
runs and Python versions supported by CI. Keep `_RAY_BATCH_SIZE=256` and all
023-b local-coordinate/ray batching behavior unchanged.

### No semantic or pixel changes

- The 400-by-400 compatibility-oracle contour shape and digest above must remain
  exact after the iterative rewrite.
- Existing 023-a/023-b focused tests and the complete suite must remain green.
- Default omission and explicit `reject` must remain identical.
- Existing feasible Euclidean views and existing radial views must remain
  byte-identical.
- No metadata value may change for the exact reference workload merely because
  recursion was removed.

### Focused production-path tests

Add tests that prove all of the following without deleting or weakening current
tests:

1. The 400-by-400 comb exercises the repair traversal and preserves the exact
   `(80598, 2)` contour shape and frozen SHA-256 above.
2. The 600-by-600, teeth-every-150 comb completes deterministically without
   recursion failure, preserves all foreground pixels, returns the same contour
   and support on two runs, and does not introduce a component-to-component
   rectangular bridge.
3. A source-embedded version forced through
   `centroid_radial_mask_chord_fallback` produces exactly one valid composed
   image and no containment/inference rejection. Keep the test bounded enough
   for normal CI while exercising the production public geometry/compositor,
   not only an alternate helper.
4. A deliberately low temporary Python recursion limit does not change the
   iterative contour result. Restore the interpreter limit in `finally`; do not
   leak process-global test state.
5. Nontrivial chord rays produce identical counts with forced batch size 3 and
   production batch size 256. Do not use zero-length `starts == ends` as the
   sole batch-equivalence proof.
6. An infeasible mask under omitted policy and explicit `reject` returns the
   same `crop_cannot_contain_support_and_contour` record/metadata and no BLIP3
   call.
7. A deterministic mocked `perf_counter` assigns known disjoint durations to
   composition and QA. Assert exact recorded values and prove debug-artifact
   encoding/planning is outside both intervals; do not rely only on `> 0` wall
   time assertions.
8. With a deliberately small diagnostic-artifact budget, fallback inference and
   all candidate/count/geometry metadata still succeed while artifacts are
   omitted/truncated under the existing contract. No artifact-budget or
   pagination implementation change is authorized unless this focused test
   reveals a real existing regression; if it does, stop and report rather than
   broaden scope silently.

Tests may use only generated arrays and mocked processors. No photograph, GPU,
model download, timing assertion, or new dependency belongs in CI.

## Benchmark and documentation

Run the existing strengthened standalone benchmark exactly:

```text
.venv/bin/python scripts/benchmark_centroid_radial_geometry.py --repeat 3 --warmup 1
```

Do not change its corpus, threshold, pass rule, digest, or published semantics.
Report every total, median, maximum, host/load context, and digest. A transient
failure must be disclosed, not hidden by retries. One pass under a documented
idle/normal CPU state is required; this remains a qualification rather than CI
wall-clock enforcement.

Update documentation only if it currently implies recursion. State that the
boundary repair walk uses an explicit candidate-local iterative stack and does
not depend on Python recursion depth. Do not make unrelated editorial changes.

## Exact live regression on the literal 023-c implementation head

After implementation/tests are complete, commit and push the literal 023-c
implementation SHA. Verify all seven required checks on that SHA before live
work. Then re-verify the private listener/auth boundary, service unit/environment
digest, tmpfs, every GPU/process, and assigned physical GPU0 UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` (RTX 3090, 24576 MiB).

One additional controlled restart of `zap-it-lan.service` is authorized for
this continuation. Use the existing launcher/environment only. Verify the new
PID start time is later than the literal 023-c implementation commit and the
loaded product code comes from that SHA. Leave the newest keyed private-LAN
service running with one process/worker/request and `NRestarts=0`.

Use the preserved exact 1280-by-720 fixture and unchanged exact fallback YAML.
Verify image SHA-256
`a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`
and config SHA-256
`0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`.
Run one request for each required global artifact page 1, 2, and 3; do not
change prompts, thresholds, question, routing, pagination, or visualization.

Require on every page: HTTP 200, prompt counts 32/15/15/20/15 total 97, stage
counts 205/137/137/122/122, zero containment rejection, exactly the same twelve
fallback IDs and 110 Euclidean IDs, identical strategies/adjustments/metadata,
and schema-valid JSON/ZIP records. Compare all 122 new debug PNGs—not only the
110 Euclidean views—against the literal-head 023-b ZIPs, both encoded and
decoded RGB. Require 122/122 matches. Require the same final labelled image
hash and produce or verify a contact sheet derived from the current twelve live
fallback artifacts.

Report composition, verification, BLIP3-stage, SAM2, and total request timings.
Do not tune or retry a model answer. If a live request fails, preserve sanitized
evidence and stop without widening scope or changing limits.

After verifying that the two abandoned 023-b setup directories
`/dev/shm/slaif-zap-it-geometry-review.023b.256n5I` and
`/dev/shm/slaif-zap-it-geometry-review.023b.99n7qL` contain only a mode-0600
copy of `page-1.yaml`, remove exactly those two redundant directories. Preserve
the authoritative `023b.lNBfGD` directory and all 023-a/baseline evidence.

## Verification and CI

Run focused geometry/compositor/API tests first, then all canonical checks:

```text
.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m compileall -q src modules scripts tests
.venv/bin/python scripts/check_documentation.py
.venv/bin/python -m build --wheel --sdist
```

Also run release-artifact verification/scanning, tracked-tree secret scan,
Twine, direct-versus-sdist wheel member comparison, isolated installed-wheel
JSON/ZIP schema smokes, `git diff --check`, and all required CI/CodeQL checks on
both the implementation SHA and final report head. Skipped, pending, failed, or
missing is not pass. Public CI must not download or execute models.

## Scope, safety, and non-goals

- Expected product diff is limited to `src/core/radial_geometry.py`, focused
  Objective 023/API tests, directly affected documentation if necessary, and
  OAP transcript/report.
- No schema/capability/config field change, dependency, SAM2/geometry-filter/
  CLIP/BLIP3 semantic change, prompt/question/answer change, final-filter/
  visualization change, artifact contract change, model/revision/dtype/
  residency change, service network/auth/key change, operator limit increase,
  CUDA/driver/firewall/VPN mutation, release, or tag.
- No request-controlled path or raw image/YAML/prompt/answer in Git, logs,
  metrics, or filenames. Preserved evidence remains mode-0700/0600 in tmpfs.
- Protect all unassigned devices and unrelated processes. Use only assigned
  physical GPU0 exposed as logical `cuda:0`.

## Acceptance and publication

Success requires removal of recursion-depth dependence with exact contour-order
compatibility, the high-boundary production-path proof, nontrivial batch and
deterministic timer tests, explicit-reject/artifact-budget coverage, unchanged
benchmark digest with sub-second idle qualification, full package/CI green, and
literal-head live proof that all 122 input images and all reference metadata
remain unchanged.

Commit/push the 023-c implementation plus unchanged published order/active
transcript to PR #87. After all implementation-head checks and live evidence
pass, publish one immutable `oap/reports/023-c-report.md`. Its final report-only
SELF commit must have the literal 023-c implementation SHA as first parent and
change only that report. Push and verify remote head/parent/exact report bytes
and every final check before sending exact response FIFO `OK`.

The report must include changed files, exact recursion reproduction/closure,
400-oracle digest, 600-comb result, all focused/full commands and results,
deterministic timer values, artifact-budget result, benchmark measurements,
CI URLs/SHAs, literal implementation/service chronology, exact live counts and
IDs, 122 encoded/decoded comparisons, current contact-sheet/final-image paths
and hashes, timing/hardware/GPU/service/security/tmpfs facts,
`Critical register action: NONE`, limitations, and the strongest reason not to
merge with an evidence-based answer.
