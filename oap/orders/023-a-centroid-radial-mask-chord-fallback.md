# OAP Work Order 023-a — centroid-radial mask-chord fallback

## Objective

Add the expert-directed, explicitly opt-in
`centroid_radial_mask_chord` fallback to the existing one-image BLIP3
`single_dilated_blur` compositor. Preserve the present Euclidean-largest-axis
composition byte-for-byte whenever it already fits. Invoke the new geometry only
after the existing path raises its reviewed containment rejection, and ensure a
valid non-empty routed candidate is no longer lost solely because requested
support or contour cannot fit the independently bounded crop.

This is a new numeric objective and one new PR. It is a geometry, observability,
schema, documentation, deterministic-test, and exact live-regression objective.
It must not tune or refactor SAM2, geometry filtering, CLIP prompts/scoring/
routing, BLIP3 questions/answers/generation, final filtering, visualization, GPU
residency, artifact budgets, or service exposure.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Verified default branch: `main` at
  `515d5200e43feb0fa8b48c0762157491487dac3b` locally and remotely, with a clean
  local worktree.
- That SHA is the accepted merge of Objective 022 PR #78. Its main-branch CI run
  `33434381009` and CodeQL run `33434381088` are successful; every required
  check on the SHA is successful.
- Open PRs are Dependabot-only PRs #79–#86. No product objective PR is open.
- Current `oap/active` is the already merged immutable `022-c` round.
- Create branch `oap/023-a-centroid-radial-mask-chord-fallback` from the exact
  verified main SHA and exactly one PR titled
  `Objective 023: centroid-radial mask-chord fallback`.
- Do not amend an old branch/PR, merge, enable auto-merge, or alter Dependabot
  PRs.

Refresh remote main/open PRs/checks, local worktree, active/order state,
`CRITICAL.md`, the service, listener, environment-file mode/digest, tmpfs,
fixture hash, and every GPU/process fact before mutation. Stop only on a genuine
authority or safety contradiction.

## Independently reproduced baseline

Strategic ran exactly one authenticated pre-fallback request against the newest
service using the required repository image and required algorithm configuration
with the fallback field omitted. Preserve and verify this evidence before using
it:

- evidence directory:
  `/dev/shm/slaif-zap-it-geometry-review.E3GeIl`, mode 0700;
- baseline config: `baseline.yaml`, mode 0600, 4,112 bytes, SHA-256
  `128c65dbe2cd9c41bd66b5c1bdc3f98fee668e668eb4476894e5543bf482a048`;
- response: `result.zip`, mode 0600, 2,585,950 bytes, SHA-256
  `ce35534fccd36a0ed05b759d7aec40d872932fefe89b228267663d6000f20d3a`;
- exact image: `demos/tomato/2022-07-22-16-25-44-48.jpg`, 1280 by 720,
  SHA-256
  `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`;
- HTTP 200; prompt counts 32/15/15/20/15, total 97;
- SAM2 candidates 205, after geometry 137, CLIP scored 137, initially routed
  122, routed after cap 122, BLIP3 verified 110;
- exactly 122 `blip3_candidate_views` records: 110 rendered and 12 rejected;
- every rejection reason is
  `crop_cannot_contain_support_and_contour`, and the rejected source IDs are
  `6, 11, 20, 105, 113, 120, 124, 139, 142, 154, 167, 178`;
- the 110 rendered source IDs exactly equal the 110 model-input debug records;
  no warning, model exception, question cap, or artifact-budget omission caused
  the 12 missing verifications;
- final count was 23 in this fresh run. Final semantic count is not a fixed
  acceptance value.

The current implementation calculates one largest-axis Euclidean radius,
independently clamps both crop endpoints, raises when any support/contour pixel
is outside, and catches that exception by skipping QA. The stated containment
mechanism and 12-candidate delta are therefore verified. The geometry itself is
an expert-directed requirement for this objective; do not replace it with a
uniform-radius shrink, anisotropic ellipse, local-normal offset, or another
design.

The exact fallback config prepared by strategic is
`/dev/shm/slaif-zap-it-geometry-review.E3GeIl/fallback.yaml`, mode 0600,
SHA-256
`0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`.
It is exactly the supplied 97-prompt configuration: compared with the committed
`tests/fixtures/configs/ripe-tomato-multiprompt.yaml`, it adds `min_area: 300`,
uses `min_width: 20` and `min_height: 20`, enables BLIP3 rule debug, selects
`[blip3, visualization]` page 1 size 48, and adds only:

```yaml
candidate_views:
  blip3:
    infeasible_geometry_policy: centroid_radial_mask_chord
```

All other request values, including the complete prompt arrays, thresholds,
question, mappings, and visualization, must remain unchanged.

## Existing-path compatibility gate

Add the request field under `candidate_views.blip3` with exactly two values:

```yaml
infeasible_geometry_policy: reject
infeasible_geometry_policy: centroid_radial_mask_chord
```

The default is `reject`. Omission and explicit `reject` must retain the current
behavior and bytes.

For every candidate, processing order is mandatory:

1. Call the current Euclidean-largest-axis `single_dilated_blur` geometry and
   composition path first.
2. If it succeeds, do not recompute or substitute geometry. Preserve its crop,
   support, contour, blur, resize, final pixels, scalar radius values, and all
   pre-existing metadata values exactly.
3. If it raises the existing
   `crop_cannot_contain_support_and_contour` rejection and policy is `reject`,
   preserve the existing rejection record and skip behavior exactly.
4. If and only if it raises that exact rejection and policy is
   `centroid_radial_mask_chord`, invoke the fallback below.
5. Do not catch or reinterpret invalid image/mask/configuration, allocation,
   model, artifact, cancellation, deadline, or other failures as fallback
   eligibility.

Existing feasible candidates under fallback-enabled configuration must be
pixel-identical to the omitted/reject baseline. New additive metadata may be
present, but no existing field's value may change. Model and debug input remain
one identical single image per candidate/question.

## Exact centroid-radial mask-chord geometry

Implement the fallback as one pure, deterministic, request-local, testable
geometry function operating from the complete boolean SAM2 mask, source shape,
source candidate ID, and validated configuration. It must not inspect RGB values
to decide geometry and must not mutate inputs or shared state.

### Mask, centroid, contours, and ordering

1. Calculate the tight inclusive bbox `B=(x0,y0,x1,y1)` over every positive
   pixel. Width and height are inclusive and `L=max(W,H)`.
2. Calculate the whole-mask area centroid in source pixel-center coordinates:
   `cx = mean(all positive x coordinates)` and
   `cy = mean(all positive y coordinates)`, using deterministic float64
   accumulation. Do not use a bbox center, per-component centroid, contour
   centroid, RGB crop, or semantic label.
3. Treat 8-connected foreground as a component. Extract an ordered external
   boundary walk for every component. Internal hole contours are not expansion
   seeds. Thin/self-touching digital contours may require repeated samples in a
   boundary walk; remove only consecutive duplicate samples and the duplicated
   terminal closing point. Process each external contour independently, while
   every contour uses the one whole-mask centroid.
4. Normalize contour enumeration deterministically: sort contours by their
   lexicographically smallest `(y,x)` sample, rotate each closed walk to that
   sample, and use one documented orientation/tie rule consistently. Include
   the last-to-first adjacent pair for a closed contour with at least two
   samples. A one-sample contour has one spoke and no non-degenerate adjacent
   quadrilateral.
5. `external_boundary_pixel_count` is the total normalized boundary-walk sample
   count after the normalization above. It is bounded scalar metadata; never
   serialize the boundary or per-ray arrays.

The implementation may use an already pinned compiled image primitive or a
bounded NumPy/Pillow implementation, but must add no dependency. CPU tests must
exercise the actual production geometry, not a materially different mock.

### Inward chord sampling

For each ordered boundary sample `p=(px,py)`:

1. When `p != c`, define the inward vector as `c-p`, continue the same ray
   through `c`, and calculate its first intersection with the opposite edge of
   inclusive bbox `B` in continuous pixel-center coordinates.
2. Convert the possibly fractional bbox intersection to an integer endpoint by
   half-up rounding each non-integral non-negative source coordinate
   (`floor(value+0.5)`) and clamp it to `B`.
3. Rasterize the inclusive integer line from `p` to that endpoint with one
   documented all-octant Bresenham-equivalent rule. Fix tie behavior in code and
   tests; the same endpoints must always yield the same ordered pixels.
4. Count every positive mask pixel on the entire rasterized line, including
   later positive runs after any zero-valued gap, until the bbox endpoint.
   This cross-gap accumulation is intentional for fragmented candidates. Call
   the positive count `D(p)`.
5. Calculate
   `raw_distance(p)=ceil(context_fraction * D(p))`, then
   `bounded_distance(p)=min(max(raw_distance(p), min_context_pixels),
   max_context_pixels)`.
6. Do not use a tangent, local normal, distance transform, ellipse, component
   centroid, or substitute chord definition.

Sample rays in bounded vectorized batches or through an existing compiled
primitive. A Python loop over contours/batches or compiled line draws is
acceptable; a Python loop over every pixel of every sampled ray is not.

### Outward support and documented degenerate convention

1. For `p != c`, the outward direction is exactly `(p-c)/norm(p-c)`. Calculate
   the continuous endpoint `p + distance * outward_unit`, half-up-round each
   non-negative source coordinate, and rasterize the inclusive spoke with the
   same Bresenham rule.
2. For the mathematically degenerate `p == c` case, use this explicit convention:
   the inward chord consists only of `p`, so `D(p)=1`; the outward unit vector is
   fixed to positive source-x `(1,0)`. Source-boundary clipping may leave only
   `p`. This rule is a deterministic definition for the otherwise undefined
   vector and must not cause rejection.
3. Clip spokes/endpoints only at the source-image boundary. Preserve the
   complete original mask regardless of clipping.
4. For each adjacent normalized boundary pair, fill the integer quadrilateral
   `(p_i, p_next, endpoint_next, endpoint_i)` with one documented deterministic
   inclusive polygon-rasterization rule. Union every quadrilateral, every
   spoke, and the original mask. Process separate contours independently.
5. This union is the requested radial support. Do not convert it to a bbox or
   add a rectangular bridge.

The whole-mask centroid may lie outside the foreground or in a gap, and a ray
may cross gaps/components. Those outcomes are part of the directed algorithm,
not reasons to recenter, replace, or reject it.

## Contour, crop, and containment adjustments

Build the radial support first. Build the existing exterior contour around that
support with the existing exact Euclidean primitive and configured color. The
contour is strictly outside support.

Nominal crop width and height are independently
`ceil(crop_extent_multiplier * W/H)`, each capped by its source dimension. Start
from the current centered half-open placement rule. Unlike the existing first
attempt, fallback crop placement retains the complete nominal dimension and
shifts its origin within `[0, source_dimension-nominal_dimension]` so it contains
the required bbox when possible. For each axis, choose the valid origin closest
to the centered origin, with the lower origin on a tie. Never shrink merely
because the centered crop crosses an image edge.

Apply adjustments in this exact order:

1. Try requested radial support plus requested effective contour width and a
   shifted full-size crop.
2. If contour prevents containment, decrement its effective width one source
   pixel at a time through width 1, recomputing the exact exterior contour each
   time.
3. If still infeasible, disable only the contour. This fallback adjustment may
   go below the request's normal contour minimum and must be reported rather
   than silently represented as the requested value.
4. If radial support itself does not fit, apply one common scale to every
   bounded radial distance. Represent the scale deterministically as an integer
   millionth `q/1_000_000` in `[0,1]`; choose the greatest representable value
   that fits. Calculate each scaled integer distance as
   `floor(q * bounded_distance / 1_000_000)`. Use a monotone integer search,
   rebuild the exact rounded spokes/quadrilaterals, and verify containment.
   Decrement the selected fixed-point value if an exact post-rasterization check
   disproves containment. Do not apply independent per-ray scaling.
5. If no positive scaled context fits, use scale zero: support is the complete
   raw mask, contour is disabled, and the remaining crop scene is blurred.
6. A valid non-empty fallback candidate must always return one composition.
   It must never raise `crop_cannot_contain_support_and_contour` solely because
   requested context or contour does not fit.

Work on the tight candidate window expanded by the maximum required radial and
contour margin and clipped to source bounds. Do not repeatedly allocate or
transform complete source-sized arrays. Returned masks/arrays remain crop-local
under the existing resource law.

For the final composite, preserve source RGB bytes exactly inside final radial
support, paint the final exterior contour with configured RGB, use the existing
Pillow Gaussian blur for every other crop pixel, and retain the existing BLIP3
resize rules. Send exactly one composed image to BLIP3.

## Metadata and timing

Preserve all existing fields and add bounded typed fields to both the L3
composition record and applicable exact BLIP3 input records:

- `infeasible_geometry_policy`: `reject|centroid_radial_mask_chord`;
- `geometry_strategy_used`:
  `euclidean_largest_axis|centroid_radial_mask_chord_fallback`;
- `mask_centroid_xy`: two finite floats or null for an untouched Euclidean path;
- `external_boundary_pixel_count`: non-negative integer or null;
- raw radial distance min/max/mean;
- effective radial distance min/max/mean;
- `effective_radial_scale`: finite `0..1` or null;
- `geometry_adjustment`:
  `none|crop_shifted|contour_reduced|contour_disabled|radial_context_scaled|zero_context_fallback`.

For fallback candidates, raw statistics use `raw_distance(p)` before min/max;
effective statistics use the final scaled integer distances actually rasterized.
Means are finite arithmetic means. Existing scalar
`raw_context_radius`/`effective_context_radius` report the corresponding maximum
radial distance for fallback compatibility. Existing feasible Euclidean
candidates retain their exact current scalar values and use strategy
`euclidean_largest_axis`, policy from the request, adjustment `none`, and null
radial-only fields.

When several fallback adjustments occur, `geometry_adjustment` uses this
precedence: `zero_context_fallback`, `radial_context_scaled`,
`contour_disabled`, `contour_reduced`, `crop_shifted`, `none`. The effective
contour width, scale, bboxes, and radial statistics retain the rest of the
evidence.

Instrument actual timings rather than leaving composition at zero:

- `stage.blip3` remains total BLIP3-stage wall time;
- `stage.blip3_composition` is accumulated wall time around the actual one-per-
  candidate composition attempt, including fallback work and candidate-local
  rejection work, but excluding artifact-only preflight and QA generation;
- `stage.blip3_verification` is accumulated time in actual BLIP3 QA calls;
- values are finite non-negative milliseconds and exposed only through the
  existing L3 timing map/ZIP parity contract.

Do not count debug PNG encoding as model verification. Report composition,
model, and total timings separately in live evidence.

## Configuration, schemas, capabilities, and documentation

Update consistently:

- `CandidateViewConfig` defaults, strict validation, request-local serialization,
  and A/B/A isolation;
- public hostile-YAML field allowlist/validation;
- effective candidate-view response schema;
- BLIP3 composition and exact-input record schemas;
- authenticated static capabilities, advertising both values and default
  `reject`;
- maintained configuration, API, algorithms, architecture, core, output-parity,
  runtime/runbook/datasheet, testing, and other directly affected docs;
- generated OpenAPI/capability tests and documentation checker expectations.

Unknown/non-string policy values return HTTP 400 `invalid_config` or the current
documented unsupported-field convention without inference. Do not silently
ignore or clamp them. The old rejection reason remains valid and documented for
policy `reject`.

Document the fixed-point scaling/rounding, contour/scale adjustment precedence,
external-contour connectivity/order, degenerate centroid convention, scalar
radius compatibility semantics, timing boundaries, default compatibility, and
the fact that the fallback may intentionally cross zero-valued gaps when
measuring a fragmented whole-mask chord.

## Deterministic CPU and API tests

Add generated-array tests, with no external photo/model/download/CUDA dependency,
covering at least:

1. Omitted policy and explicit `reject` are byte-identical for every existing
   candidate-view fixture and preserve the existing rejection.
2. Fallback-enabled candidates that already fit are pixel-, crop-, support-,
   contour-, resize-, and existing-metadata-value-identical to `reject`.
3. A 20 by 100 rectangle gives short-axis context near 20 percent of its short
   radial chord and long-axis context near 20 percent of its long chord; a
   100 by 20 rectangle gives the exact transposed result.
4. A rotated elongated mask fits the independent crop limits.
5. Concave masks are deterministic and retain every raw-mask pixel.
6. A fragmented mask proves sampling continues after zero gaps and counts later
   positive pixels before the bbox endpoint.
7. A whole-mask centroid in a gap does not fail or substitute another centroid.
8. A one-pixel mask and any boundary sample coincident with the centroid exercise
   the exact positive-x convention.
9. Masks touching every edge and corner render without negative coordinates,
   wraparound, silent crop shrink, or source loss.
10. Multiple disconnected components, diagonal/corner connections, thin
    contours, and masks with holes have deterministic external contour handling;
    hole boundaries are not seeds.
11. Contour reduction is attempted before disabling; disabling occurs before
    radial scaling.
12. One common scale is used; its fixed-point search and integer distance
    rounding are deterministic and exact containment is rechecked.
13. Zero-context fallback returns one valid image containing the complete raw
    mask with blurred surroundings.
14. Final crop dimensions do not exceed the multiplier on either raw bbox axis
    unless capped by the smaller source image, and retain their nominal size at
    boundaries.
15. Final crop contains all clipped support and contour.
16. A valid non-empty mask cannot produce the containment rejection when the
    new policy is enabled.
17. Source pixels under support remain byte-identical, contour pixels are only
    outside support and have exact configured color, and all other final crop
    pixels equal the existing Gaussian-blurred source result.
18. Exactly one final image is passed to each mocked BLIP3 call and its decoded
    debug PNG is byte-identical to that model input.
19. Metadata statistics, strategy, adjustment precedence, compatibility scalar
    radii, IDs, bboxes, and source/model dimensions are exact and schema-valid.
20. Composition and QA timers are independently nonzero under instrumented
    deterministic fake clocks, reconcile within the total stage timer, and do
    not include artifact-only planning.
21. Capabilities/default/effective config/JSON/ZIP/OpenAPI cover both policies;
    invalid values are precise HTTP 400 errors; L0-L2 omission and L3 parity
    remain correct.
22. Request-local A/B/A tests alternate `reject` and fallback without resident
    holder replacement or geometry state leakage.
23. Artifact count/size/selection/omission behavior remains unchanged except
    that fallback candidates now offer real artifacts rather than nonexistent
    rejected ones.

Preserve all existing tests. Do not weaken, delete, rename away, or mock around
the current Euclidean, CLIP, BLIP3, artifact, schema, package, and service
regressions.

## Non-flaky benchmark

Add a manually invoked benchmark outside ordinary unit-test timing assertions.
It must build a deterministic representative workload of approximately 122
non-empty masks bounded to 199 by 199, exercise the production centroid-radial
geometry (not a simplified surrogate), verify deterministic digest/output, and
report total geometry time, per-candidate summary, Python/NumPy/Pillow and any
compiled-primitive versions, CPU model, core count, and platform. Do not make
normal CI pass/fail on wall time.

Run it on the live host CPU and require measured total centroid-radial geometry
below one second for that workload. If it misses, optimize the bounded batch or
compiled operations without changing geometry. Report exact command, hardware,
repeat count, warmup policy, minimum/median/maximum, and result digest.

## Baseline and live artifact proof

The exact diagnostic selector uses one global pipeline-ordered page. With BLIP3
debug enabled, page 1 contains the first 48 BLIP3 artifacts and the final
visualization is selection-excluded, not budget-omitted. Preserve this existing
contract; do not reorder or redesign artifact pagination in this objective.

Evidence is therefore collected through algorithm-identical diagnostic pages:

### Before mutation/restart

1. Verify strategic's page-1 baseline ZIP and manifest exactly as above.
2. Before changing or restarting the running old service, make exactly two
   additional authenticated requests using the baseline config and image,
   changing only `diagnostic_artifacts.page` to 2 and then 3. Store responses
   in a new mode-0700 tmpfs evidence directory with mode-0600 files. No retry.
3. Require the same 205/137/137/122/110/12 counts and rejected-ID set in all
   three baseline pages. Combine delivered debug artifacts by
   `source_candidate_id`, not shifting question index, and require exactly 110
   unique old BLIP3 views.
4. Record SHA-256 of both encoded PNG and decoded contiguous RGB bytes for every
   old rendered source candidate. Preserve the bounded hash map in tmpfs, not
   Git.

### After implementation-head CI is fully green

1. Reconcile every host/GPU/service/key/tmpfs fact again, then perform exactly
   one controlled restart of only `zap-it-lan.service`. Do not change its unit,
   environment, key, private address, port, model/cache/residency, deadline, or
   budgets. Wait for a stable new PID, zero restarts, ready service, and the
   assigned physical GPU only.
2. Run the exact fallback config hash
   `0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`
   at verbosity 3 ZIP page 1. Then make exactly two diagnostic-only derivative
   requests changing only page to 2 and 3. Do not change prompts, thresholds,
   routing, question, mappings, candidate view values, or visualization.
3. Read the bearer only into process memory, supply it without argv/log/report
   exposure, unset it immediately, and never copy it to evidence.
4. Require HTTP 200 for all three pages; exact image/config hashes; prompt counts
   32/15/15/20/15 total 97; SAM2 205; after geometry 137; CLIP scored 137;
   initially routed and routed after cap 122; BLIP3 verified 122; zero
   containment rejections; and 122 unique one-image BLIP3 debug inputs across
   pages.
5. Require the same 110 old-rendered source IDs to use
   `euclidean_largest_axis` and have byte-identical decoded RGB and encoded PNG
   hashes before versus after. Require exactly the prior 12 rejected IDs to use
   `centroid_radial_mask_chord_fallback`; no other ID may do so.
6. Page 3 must deliver the final fixed member
   `visualization/stream-0001.png` with logical
   `visualization_id=final-labelled-ripe-tomatoes`, descriptor/manifest
   hash-size parity, and a schema-valid labelled PNG. This supplemental page is
   required because global pagination intentionally preserves existing stage
   order.
7. Build a fixed operator-named, mode-0600 tmpfs contact sheet from the exact 12
   fallback model-input PNGs, labelled only with numeric source candidate IDs.
   Do not place client prompt/question/label text into its filename. Visually
   inspect all 12 views, the contact sheet, and final labelled image. Report
   obvious support loss, clipping, contour defects, unexpected rectangular
   bridges, fragmented/merged masks, obvious final false positives/misses, and
   any fallback candidate that was not actually sent once to BLIP3.
8. Report final count without requiring an exact value. Report exact composition,
   QA-model, total BLIP3, SAM2, CLIP, geometry, and overall request timings for
   every page; distinguish cold/warm effects and do not average away failures.
9. Validate JSON/ZIP response schemas, fixed safe names, artifact ledgers,
   requested/effective config, radial metadata, prompt/score classes, final
   labels/bounds, and no request workspace residue.
10. Recheck private auth/docs/metrics boundaries, listener, PID/restarts,
    assigned GPU UUID/process, environment digest, tmpfs cleanup, and leave the
    newest private keyed service running.

If a live request fails, preserve sanitized evidence and stop without retry,
threshold/prompt mutation, limit increase, another restart, or unrelated fix.

## Verification and CI

Run focused geometry/compositor/config/schema/capability/artifact/engine/API
tests, the benchmark, then all canonical checks:

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
check on the literal implementation SHA. Skipped/pending/missing is not pass.
No GPU/model download is allowed in public CI.

## Scope, safety, and non-goals

- Expected paths are narrowly in `modules/verifier/blip3.py`, the pure
  candidate-view geometry module if separated, `src/core/engine.py`, candidate
  config/service validation/schemas/capabilities/envelope metadata, focused
  tests/benchmark, maintained docs, and OAP transcript.
- No SAM2 proposal/filter change, CLIP crop/prompt/score/router change, BLIP3
  question/fixed instruction/answer mapping/generation change, final filter,
  renderer, visualization, artifact pagination/budget, object/response/deadline
  limit, model/revision/dtype/residency, dependency, service network/auth/key,
  host CUDA/driver/firewall/VPN, or unrelated process change.
- No request-controlled path, raw image/YAML/prompt/answer in Git/logs/metrics,
  persistent request data, public bind, release, or tag.
- Use only assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, RTX 3090 24576 MiB, exposed as
  logical `cuda:0`. Protect every unassigned device and unrelated workload.
- One service process, worker, and inference at a time. Keep all evidence in
  explicitly preserved mode-0700 tmpfs directories and product request
  workspaces self-cleaning.

## Acceptance and publication

Success requires the expert geometry exactly as ordered, byte-identical existing
feasible views, default reject compatibility, no containment loss for the 12
fallback candidates, all deterministic/API tests, sub-second non-flaky geometry
benchmark, full package/CI green, exact 205/137/137/122/122 live counts, 110
before/after view hashes, exactly 12 fallback views/contact sheet, final labelled
visualization, truthful timings, and newest safe service left running.

Commit implementation and the unchanged published `023-a` order/active
transcript, push, and create the one PR. After all implementation-head checks
and live evidence pass, capture the literal implementation SHA and publish one
immutable `oap/reports/023-a-report.md` using the report template. The final
report-only SELF commit must have the implementation SHA as first parent and
change only that report. Push, verify remote head/parent/exact report bytes and
all report-head checks, send exact response FIFO `OK`, mutate nothing further,
and exit.

The report must include changed files, policy migration notes, every test and
benchmark command/result, CI URLs/SHAs, baseline/fallback hashes, complete stage
counts, rejected/fallback ID sets, all 110 hash-comparison result, contact-sheet
and labelled-image tmpfs paths/hashes, bounded visual findings, timing/hardware,
service/GPU/security facts, `Critical register action: NONE`, limitations, and
the strongest reason not to merge with its evidence-based answer.
