# OAP Work Order 023-b — close bounded geometry and live-head proof

## Objective

Amend Objective 023 PR #87 in place. Retain the accepted centroid-radial
mask-chord semantics and the successful reference behavior from 023-a, while
closing the independently verified resource-bound, response-schema, test-proof,
metadata, benchmark, and live-revision gaps. This is a same-PR continuation, not
a redesign and not a new recognition-pipeline objective.

The fallback remains explicitly opt-in, the default remains `reject`, and the
existing Euclidean-largest-axis compositor must still run first and remain
pixel-identical whenever feasible. Do not tune SAM2, geometry filters, CLIP,
BLIP3 questions/answers, final filtering, visualization, or artifact selection.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote `main` remains
  `515d5200e43feb0fa8b48c0762157491487dac3b`.
- Amend only open PR #87, branch
  `oap/023-a-centroid-radial-mask-chord-fallback`, titled
  `Objective 023: centroid-radial mask-chord fallback`.
- The reviewed remote head is the immutable 023-a report-only SELF commit
  `f032aa4787ef3e8170340eb2b715dc5849cad78a`; its first parent is implementation
  commit `6d63de7a2fce65b94e8212543e34a7bab25b79a4`.
- All seven CI/CodeQL checks on that head are green and the PR is mergeable, but
  green CI is insufficient because the defects below are independently
  reproduced.
- Continue on the same branch and PR. Commit the published 023-b order/active
  transcript and corrective implementation after the immutable 023-a report.
  Do not modify, replace, or delete `oap/reports/023-a-report.md`.
- Do not merge, enable auto-merge, rewrite history, amend existing commits, or
  touch other PRs.

Refresh GitHub, branch/worktree, current service/listener, exact fixture,
tmpfs evidence, `CRITICAL.md`, and all GPU/process facts before mutation. Preserve
valid 023-a evidence; do not rerun expensive live inference until the corrected
literal implementation head is committed, pushed, and locally loaded.

## Strategic review decision

The expert-directed geometry is accepted. Current implementation and reference
artifacts prove the original twelve losses were exactly containment failures:
205 SAM2 candidates, 137 after geometry, 137 CLIP-scored, 122 routed, 110
verified, and twelve records rejected only as
`crop_cannot_contain_support_and_contour`, for source IDs
`6, 11, 20, 105, 113, 120, 124, 139, 142, 154, 167, 178`. The 023-a opt-in run
preserved the first four counts, verified all 122, produced zero containment
rejections, used fallback for exactly those twelve IDs, and preserved all 110
previously feasible encoded PNG and decoded RGB hashes.

The implementation is nevertheless not mergeable yet for the following exact
reasons.

### Finding 1 — full-source allocation and unbounded ray matrix

`src/core/radial_geometry.py::_component_pixels(mask)` allocates
`visited = np.zeros_like(mask)` over the complete source-shaped mask and walks
`np.flatnonzero(mask)`. `build_centroid_radial_geometry` obtains bbox/centroid and
contours from the complete source mask before constructing its later local
window. `_rasterize_lines` constructs a single two-dimensional sampling matrix
for every boundary ray at once, sized by total boundary sample count times the
longest ray.

This violates the ordered requirement to operate on the tight candidate window
plus required margin and to vectorize ray sampling in bounded batches. It makes
per-candidate scratch memory depend on the source canvas and permits a
fragmented/high-perimeter mask to allocate an unbounded ray matrix.

### Finding 2 — legal raw metadata can fail the response schema

`Blip3CandidateViewRecord` caps raw radial min/max/mean at 512, although 512 is
the configured maximum for the *effective*, post-policy distance. Strategic
reproduced this with a source-shaped 1200-by-20 rectangular mask,
`context_fraction: 0.5`, `max_context_pixels: 512`, extent multiplier 1.0 and
contour disabled. Composition correctly used the fallback and reported raw max
600 and effective max 0 after fit scaling, but Pydantic response validation
failed at `raw_radial_distance_max` with `less_than_equal le=512.0`.

Raw diagnostics must accept every value possible for an otherwise admitted
image/configuration. Effective values remain bounded by the validated policy.
No valid inference may become a response-schema failure.

### Finding 3 — proof matrix and benchmark are incomplete

The new tests establish the primary reference behavior but do not prove the
full ordered matrix. In particular they do not exhaustively prove local scratch
bounds, bounded ray batches, all existing-fixture compatibility, explicit
reject parity, forced rotated fallback, deterministic concave/disconnected/hole
composition, adjustment precedence, response JSON/ZIP parity and verbosity
omission, deterministic timing accounting, A/B/A request-local isolation, or
artifact-budget coexistence. The benchmark accepts the minimum of repeated
runs and uses only small simple shapes; it does not represent the ordered
elongated, concave, fragmented, and high-boundary workload up to 199 by 199.

### Finding 4 — live proof did not run the literal final implementation parent

The private service PID 737533 started at 2026-09-01 19:45:03 CEST. Commit
`507550f62b9157d3b00cdadf5d34c674cfb154a5` predates that restart, but the final
023-a implementation parent `6d63de7a2fce65b94e8212543e34a7bab25b79a4`
was committed at 19:55:43 CEST. The second commit changes the runtime response
schema. Therefore the live evidence is valuable algorithm evidence, but it is
not proof that the literal final implementation head was loaded.

## Required corrective implementation

### 1. Preserve geometry semantics and compatibility

- Do not change centroid, external-contour, chord, cross-gap accumulation,
  radial-distance, endpoint, quadrilateral, common-scale, contour precedence,
  blur, resize, or fallback eligibility semantics fixed by 023-a.
- Preserve default/explicit `reject` behavior and exact existing feasible-image
  bytes.
- Preserve all 023-a metadata and API field names.
- Keep one BLIP3 image per candidate/question.
- Do not delete or weaken any 023-a tests merely to make this continuation pass.

### 2. Make all geometry scratch work candidate-local

After validating the non-empty source-shaped mask, compute its tight bbox and
immediately create one contiguous bbox-local boolean mask. Perform component
labelling, external-background determination, contour extraction/order,
centroid/chord sampling, and raw-distance calculation on that local mask. Carry
one explicit `(x0, y0)` origin to translate coordinates back to source space.

Construct support/contour only within a bounded local window derived from the
tight bbox, validated maximum effective radial distance, contour margin,
nominal crop dimensions, and source edges. Do not allocate source-height by
source-width visited/support/contour scratch arrays. A source-shaped input mask
is unavoidable at the public function boundary, but its size must not be
duplicated for geometry scratch.

The centroid must remain the float64 center of gravity of all mask-positive
pixels and be reported in source coordinates. Local-coordinate conversion must
not change any 023-a geometry result or source-edge behavior.

### 3. Bound vectorized ray sampling

Use a documented fixed maximum ray-batch size in production code. Rasterize and
count no more than that many rays in one sampling matrix; accumulate results in
stable boundary order. Do not use a Python loop over individual sampled ray
pixels. A Python loop over fixed-size vectorized batches is acceptable.

Bound temporary matrix dimensions by `batch_size * local_bbox_largest_axis`,
not total boundary count times largest axis. Choose a conservative constant and
document it in code. Do not expose it as request configuration and do not add a
dependency.

Add a deterministic test seam or direct internal test proving every actual
batch respects the fixed limit and that results are identical across a forced
small batch and the production batch. Also prove embedding the identical local
candidate in a much larger source canvas does not change local geometry other
than translated source coordinates and does not create source-sized scratch
arrays. Do not use fragile RSS or wall-clock unit assertions for this proof.

### 4. Correct response-schema bounds

- Raw radial min/max/mean are nonnegative finite diagnostics before policy
  clamping. Remove the false 512 upper bound or replace it only with a bound
  derived from the admitted image/configuration contract that remains correct
  under operator-configurable image dimensions.
- Effective radial min/max/mean and scalar effective radius remain bounded by
  the validated effective context policy.
- Review `external_boundary_pixel_count` similarly. Do not retain an arbitrary
  response-only maximum capable of rejecting a legal admitted mask; either
  derive it from admission limits or use a nonnegative integer with normal
  serialization bounds.
- Add the exact raw-600 regression and a service-level L3 response validation
  test proving successful structured serialization. Invalid/nonfinite metadata
  must still be rejected.
- Keep capabilities and documentation truthful about which fields are raw and
  which are post-clamp.

### 5. Make adjustment metadata truthful

`geometry_adjustment` must report the highest-precedence actual adjustment. The
precedence remains zero-context, radial scaling, contour disabled, contour
reduced, crop shifted, none. Compare the final crop against the unshifted
candidate-centered nominal crop, not against a helper result that has already
performed the edge shift. Preserve `none` only when no adjustment occurred.

Add focused cases for each reachable adjustment and precedence combinations.
An edge shift must be observable as `crop_shifted` when no higher-precedence
adjustment applies. This is metadata-only; do not change pixels or crop choice.

### 6. Complete deterministic tests

In addition to preserving all existing tests, add focused production-path tests
that prove:

1. policy omitted and explicit `reject` have identical rendered bytes,
   rejection records, and containment behavior across every existing
   single-image candidate-view fixture;
2. fallback-enabled feasible results remain identical across those fixtures;
3. horizontal, vertical, and rotated elongated masks are each forced through
   fallback and show the required chord-proportional result;
4. concave, fragmented, centroid-in-gap, disconnected, hole, border, corner,
   thin, one-pixel, and high-boundary masks are deterministic and preserve the
   complete raw mask without a rectangular bridge;
5. cross-gap accumulation affects the later positive segment exactly as
   specified;
6. contour reduction precedes disable, and disable precedes radial scaling;
   zero-context still yields one valid image;
7. the full clipped support and contour are inside the crop; width and height
   independently obey the multiplier/source caps;
8. fallback-enabled valid non-empty candidates never retain
   `crop_cannot_contain_support_and_contour`;
9. JSON and ZIP responses validate and expose identical new record values at
   verbosity 3, while lower verbosities retain their documented bounded
   omission behavior;
10. unknown policy remains HTTP 400 `invalid_config`; omitted/default/reject and
    capabilities remain consistent;
11. alternating fallback/reject/fallback requests prove no request-state leak;
12. debug artifact admission/omission does not change composition or
    verification counts and preserves existing artifact budgets/pagination;
13. a deterministic fake/recording clock proves `stage.blip3_composition`
    contains composition work, excludes model QA and artifact planning, and
    reconciles with the BLIP3 stage within documented rounding tolerance.

Tests may use generated NumPy masks and mocked model processors. Do not add a
photograph, tune a model, or require GPU/model downloads in CI.

### 7. Strengthen the non-CI benchmark

Retain a standalone benchmark outside normal unit-test wall-clock assertions.
Its approximately 122-candidate corpus must deterministically mix horizontal,
vertical, rotated, concave, fragmented, centroid-gap, and high-boundary masks,
including dimensions approaching the configured 199-by-199 reference bound.
Report each repeated total, median, maximum, batch size, Python/NumPy versions,
CPU model, and host. The documented target is total geometry below one second
per 122 candidates on a normal CPU. Judge the median and disclose the maximum;
do not pass merely because one minimum sample is below one second. Keep this a
reported qualification, not a flaky CI wall-clock gate.

## Documentation

Update only documentation affected by these corrections. State explicitly:

- geometry scratch is tight-bbox/local-window bounded and rays are processed in
  fixed-size batches;
- raw radial diagnostics are pre-clamp and may exceed
  `max_context_pixels`, while effective diagnostics cannot;
- adjustment precedence and `crop_shifted` semantics;
- the benchmark corpus/method and non-CI status;
- existing opt-in/default-reject migration and artifact behavior remain
  unchanged.

Run the maintained documentation checker. Do not make unrelated editorial or
architecture changes.

## Exact live regression on the literal implementation head

Use the already preserved exact fixture and configurations from 023-a after
verifying their hashes. The image must remain 1280 by 720 with SHA-256
`a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.
Do not reduce prompt arrays or change thresholds/questions.

Before the single authorized restart, commit and push the complete 023-b
implementation and record its literal SHA. Re-verify physical GPU0 UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` (RTX 3090, 24576 MiB), every GPU
process, private listener/auth, service unit, environment digest, `/dev/shm`,
and port ownership. Restart only `zap-it-lan.service`, once, using the existing
launcher/environment. Verify the new PID start time is later than the literal
implementation commit and that the process cwd/code checkout is that exact
commit. Leave the newest keyed private-LAN service running.

Run the exact fallback request once per required artifact page under the
existing global pagination contract. The same inference may therefore be
repeated for pages 1, 2, and 3; do not alter stage selection to fabricate a
single-page result. Preserve mode-0700 tmpfs evidence and mode-0600 request/
response files.

Require and report:

- HTTP 200; CLIP prompt counts 32/15/15/20/15, total 97;
- 205 SAM2, 137 after geometry, 137 CLIP-scored, 122 routed after cap;
- 122 BLIP3 verified and zero containment rejections;
- exactly the prior twelve IDs use radial fallback and the other 110 use
  Euclidean geometry;
- all 110 old/new encoded PNG and decoded RGB hashes still match the immutable
  baseline evidence;
- exact candidate IDs, strategies, adjustments, all response-schema records,
  contact sheet, final labelled visualization, hashes, and bounded manual visual
  findings;
- composition, model-verification, BLIP3-stage, and total timings, with timing
  semantics reconciled;
- no unauthorized GPU/process/network/auth/key/model/config mutation.

Do not require an exact final semantic count. Do not retry or tune after a model
answer. If the literal-head live request fails, preserve sanitized evidence and
report the exact failure without widening scope.

## Verification and CI

Run focused tests first, the strengthened benchmark, and then all canonical
checks:

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
JSON/ZIP schema smokes, `git diff --check`, and every required GitHub CI/CodeQL
check on both the literal implementation SHA and final report head. No skipped,
pending, failed, or missing check is a pass. Public CI must not download or run
models.

## Scope, safety, and non-goals

- Expected changes are limited to the existing Objective 023 geometry/
  compositor, response schema/metadata assembly, focused tests and benchmark,
  affected docs, and OAP transcript/report.
- No new dependency, SAM2/CLIP/BLIP3 semantic tuning, prompt/question change,
  filter/routing/final-label/renderer change, artifact budget/pagination change,
  service network/auth/key change, model revision/dtype/residency change,
  operator limit increase, host driver/CUDA/firewall/VPN mutation, release, or
  tag.
- No request-controlled path and no image/YAML/prompt/answer content in Git,
  logs, metrics, or filenames. Request data remains RAM or `/dev/shm` and
  self-cleaning.
- Use only assigned physical GPU0 with the exact UUID above, exposed as logical
  `cuda:0`. Protect every other device and unrelated process. One service
  process, worker, and request at a time.

## Acceptance and publication

Success requires the unchanged accepted geometry semantics, candidate-local
scratch, bounded ray batches, correct raw response schema, truthful adjustment
and timing metadata, the complete deterministic proof matrix, representative
sub-second reported benchmark, full package/CI green, and exact literal-head
live regression preserving 205/137/137/122/122 plus all 110 compatible hashes.

Commit and push the corrective implementation plus unchanged published 023-b
order/active transcript to PR #87. After all implementation-head checks and
literal-head live evidence pass, publish one immutable
`oap/reports/023-b-report.md`. The final report-only SELF commit must have the
023-b implementation SHA as first parent and change only that report. Push and
independently verify remote head, parent, exact report bytes, PR identity, and
all report-head checks before sending the exact response FIFO `OK`.

The report must include changed files, migration note (none beyond 023-a's
explicit opt-in field), exact focused/full test and benchmark commands/results,
CI URLs/SHAs, the raw-600 schema regression, batch/local-window resource proof,
all adjustment cases, implementation/service chronology, exact live counts and
ID sets, 110 hash comparison, contact-sheet/final-image paths and hashes,
timings/hardware/service/GPU/security facts, `Critical register action: NONE`,
limitations, and the strongest reason not to merge with an evidence-based
answer.
