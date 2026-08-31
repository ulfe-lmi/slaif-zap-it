# OAP Work Order 019-a — Single-image BLIP3 candidate view

## Objective

Replace BLIP3's target/context two-panel verification image with one deterministic,
request-local, mask-derived image per candidate. The one image must preserve the
exact source scene inside an exact Euclidean dilation of the SAM2 mask, draw a
thin exterior contour, and retain the surrounding scene only through bounded
Gaussian blur. Keep every task-specific BLIP3 question exactly client-provided;
change only generic image preparation and the exact fixed service instruction.

This is a new numeric objective and one new PR. It supersedes the BLIP3 view
portion of Objectives 013 and 017 without changing their immutable orders or
reports. CLIP's existing mask-isolated `mask_dilated` view remains unchanged.

## Verified GitHub and OAP state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Default branch: `main` at
  `4acff3a8f7717a08481b86338453d09e754c1e86`.
- That commit is the merge of Objective 018 PR #74; its post-merge CI run
  `33284462391` and CodeQL run `33284462467` are successful.
- Open pull requests: none.
- Branch protection API reports no configured branch protection; OAP's complete
  green-check and independent-review gate remains mandatory.
- Current selector `oap/active` is the already-merged immutable `018-b` round.
- Create branch `oap/019-a-single-image-blip3-candidate-view` from the verified
  `main` SHA and exactly one PR titled `Objective 019: single-image BLIP3 candidate view`.
- Do not amend an old branch/PR, merge, or enable auto-merge.

## Reconciled current behavior

Current `main` builds a shared tight mask view, then BLIP3 creates a resized
target-only left panel, a zero-filled/dimmed context right panel, and a four-pixel
black divider. The fixed instruction explicitly refers to those sides. The API
schema exposes BLIP3's old `mask_dilated`, `outside_fill`, `context_intensity`,
and `contour_width` fields, and current documentation repeats the paired design.

The new contract removes that BLIP3 design. A bounding box determines only the
bounded crop; the exact SAM2 mask and its exact Euclidean dilations determine
support and contour. No second pane, divider, duplicated candidate, black/zero
fill, neutral fill, dimmed ring, or untouched rectangular context may reach
BLIP3 in the new mode.

## Exact configuration contract

Keep `candidate_views.clip` byte-for-byte compatible. Replace only the effective
`candidate_views.blip3` surface with these request-local fields and defaults:

```yaml
candidate_views:
  blip3:
    mode: single_dilated_blur
    context_fraction: 0.20
    min_context_pixels: 0
    max_context_pixels: 64
    crop_extent_multiplier: 2.0
    blur_sigma_fraction: 0.15
    contour_enabled: true
    contour_fraction: 0.02
    contour_min_pixels: 1
    contour_max_pixels: 3
    contour_rgb: [255, 224, 0]
```

Validate without coercing explicit values:

| Field | Exact accepted type/range |
| --- | --- |
| `mode` | string literal `single_dilated_blur` |
| `context_fraction` | finite non-boolean number, `0.0..0.5` |
| `min_context_pixels` | integer, `0..256` |
| `max_context_pixels` | integer, `0..512`, and not below the minimum |
| `crop_extent_multiplier` | finite non-boolean number, `1.0..2.0` |
| `blur_sigma_fraction` | finite non-boolean number, `0.0..0.5` |
| `contour_enabled` | strict boolean |
| `contour_fraction` | finite non-boolean number, `0.0..0.25` |
| `contour_min_pixels` | integer, `1..3` |
| `contour_max_pixels` | integer, `1..3`, and not below the minimum |
| `contour_rgb` | list of exactly three strict integers, each `0..255` |

Reject null, bool-as-number, nonfinite, unknown, wrong-type, out-of-range, and
inverted-bound values with the existing structured `invalid_config` or
`unsupported_field` conventions. Never clamp an invalid request field. The
runtime formulas below intentionally clamp calculated effective values.

The old BLIP3 `mode: mask_dilated`, `outside_fill`, `context_intensity`, and
`contour_width` request fields are no longer accepted. Reject them explicitly;
do not ignore, translate, or silently preserve the paired behavior. They remain
valid only under `candidate_views.clip` where applicable to CLIP. Do not change
the top-level BLIP3 rule names or their bounded `question`, `trueresult`,
`falseresult`, `newcategory`, and `debug` semantics.

Expose the exact BLIP3 fields, defaults, types, bounds, effective formulas,
containment/rejection policy, and single-image debug rule through the authenticated
static capabilities endpoint. Effective BLIP3 policy at L0-L3 must contain only
the new BLIP3 field set plus `applied`; CLIP metadata stays unchanged. Preserve
request-local A/B/A isolation and resident model-holder identity.

## Exact source-space composition

Implement one pure compositor with an unambiguous name such as
`compose_single_blip3_view(image_rgb, mask, source_candidate_id, config)`.
It must not mutate its RGB image, mask, or configuration.

Use source coordinates and these exact rules:

1. Require a non-empty source-shaped boolean mask and RGB `uint8` image.
2. Compute the tight raw-mask bbox as inclusive coordinates
   `(x0, y0, x1_inclusive, y1_inclusive)`. Set
   `W = x1_inclusive - x0 + 1`, `H = y1_inclusive - y0 + 1`, and
   `L = max(W, H)`.
3. Compute `raw_context_radius = ceil(context_fraction * L)` and
   `effective_context_radius = min(max(raw_context_radius,
   min_context_pixels), max_context_pixels)`.
4. Produce support `D` by exact squared-Euclidean disk dilation of `M` with
   that integer radius, clipped only at source-image boundaries. Dilation must
   operate on the mask, preserve disconnected components, holes until reached,
   and never fill the bbox. Reuse the reviewed exact Euclidean primitive when
   sound; do not substitute square/Chebyshev, OpenCV-version-dependent, or bbox
   expansion semantics.
5. If `contour_enabled`, compute
   `raw_contour_width = ceil(contour_fraction * L)` and
   `effective_contour_width = min(max(raw_contour_width,
   contour_min_pixels), contour_max_pixels)`. Compute the contour as
   `exact_euclidean_dilate(D, effective_contour_width) & ~D`, clipped only at
   source boundaries. Thus every contour pixel is strictly outside D. If
   disabled, the effective width is zero and the contour is empty.
6. Compute the inclusive bbox center as
   `cx = (x0 + x1_inclusive) / 2`, `cy = (y0 + y1_inclusive) / 2`.
   Nominal crop dimensions are
   `ceil(crop_extent_multiplier * W)` and
   `ceil(crop_extent_multiplier * H)`; the defaults therefore give exactly
   `2*W` by `2*H`. Place that integer crop deterministically around the bbox
   center, using the lower coordinate on a half-pixel tie, then clamp each
   half-open endpoint to the source boundaries. Do not shift the crop to invent
   extra scene after an endpoint is clamped, do not exceed the nominal size,
   and do not apply any minimum source-space crop size.
7. Before reading/blur/model work, require that the clamped crop contains every
   in-source pixel of `D | contour`. A support or contour pixel outside the crop
   is a candidate-local rejection; it must never be clipped silently.
8. Copy the original RGB crop and compute a deterministic Gaussian-blurred copy
   with Pillow `ImageFilter.GaussianBlur`, using
   `effective_blur_sigma = min(max(blur_sigma_fraction * L, 2.0), 20.0)`.
   Then restore `source_crop[D_crop]` byte-for-byte and paint only
   `contour_crop` with the validated RGB triplet. The result therefore has
   unmodified source pixels throughout D, contour strictly outside D, and
   blurred source-scene context everywhere else. Applying blur broadly and then
   restoring D/contour is allowed; the final per-pixel result must obey this rule.
9. Resize only the fully composed image for BLIP3. Preserve the current bounded
   deterministic no-letterbox policy: bilinear RGB resize, 256-pixel target
   short side when smaller, 768-pixel maximum long side, and no upscale that
   would violate the long-side cap. Never enlarge the source-space crop, add a
   background, duplicate a pane, or add a divider. Keep the pre-resize composite
   available within the immutable composition result for source-coordinate tests;
   the final resized RGB/PIL image is the sole model/debug input.

Represent bboxes without ambiguity in code and metadata: raw-mask and support
bboxes are inclusive under names ending `_xyxy_inclusive`; the array-slice crop
bbox is half-open under a name ending `_xyxy_exclusive`. Record both source
composite and final dimensions. Arrays returned from the pure compositor should
be contiguous and immutable where current candidate-view conventions require it.

The broad blurred scene is intentional. Do not apply zero, neutral, or black
fill; do not dim/desaturate the context; do not expose a second untouched crop.

## Candidate-local containment rejection

Add a dedicated internal candidate-view rejection type/reason. If the bounded
crop cannot fully contain the in-source support plus contour:

- do not call BLIP3 QA for that candidate;
- do not emit a debug artifact for it;
- do not mark it negative or otherwise mutate its CLIP label/score/answer;
- continue safely with other candidates;
- append one bounded deterministic L3 diagnostic for that candidate with reason
  `crop_cannot_contain_support_and_contour`; and
- add no raw image, mask, prompt, path, label, or other content to diagnostics,
  logs, errors, or metrics.

The bounded L3 candidate-view composition records must contain one record per
candidate for which at least one effective BLIP3 rule would otherwise run, not
one duplicate per question. Add a clearly named field such as
`blip3_candidate_views`; do not overload the existing debug-artifact
one-to-one contract. Each record contains:

- `source_candidate_id` and zero-based `filtered_index`;
- `status: rendered|rejected` and the fixed diagnostic reason or null;
- `render_mode: single_dilated_blur`;
- raw-mask bbox, support bbox when computed, and crop bbox with the coordinate
  conventions above;
- raw/effective context radius;
- raw/effective contour width;
- effective blur sigma;
- source-composite and final model-input dimensions.

Keep the list bounded by existing BLIP3 question/candidate admission. Serialize
it only at L3 in JSON and ZIP manifest with exact parity. Lower levels omit it.
Existing fixed source candidate IDs and filtered indices must remain stable.

## BLIP3 model and debug seam

Compose the image exactly once for each candidate that has one or more applicable
BLIP3 questions, and reuse the identical immutable final image for every question
about that candidate. Each QA invocation receives exactly one PIL image and the
query; there is no image list containing two images and no visual duplication.

Replace the fixed instruction bytes with exactly:

```text
The unblurred region inside the yellow boundary is the selected candidate. The blurred surroundings are context only. Answer exactly Yes or No.
```

Keep each bounded client question byte-for-byte unchanged and before the fixed
instruction using the existing safe delimiter convention. Do not add a target
name, inferred class, rule name, label, answer hint, left/right wording, or any
other task-specific service prose. Existing question count, length, and token
limits remain enforced and model identity, revision, cache, device, dtype,
residency, destinations, and generation parameters remain operator-controlled.

At L3 with an effective rule's `debug: true`, retain the existing fixed safe
artifact name
`blip3-verification-CANDIDATE-####-QUESTION-####.png`. Decode the stored lossless
PNG in tests and require it to be byte-identical to the exact final RGB array/PIL
image given to QA for that question. A candidate with multiple debug questions
may retain the existing per-question artifact contract, but every artifact and
QA call must contain the same one-view candidate bytes; no artifact path derives
from client text. Preserve artifact-count, per-item, total, JSON/base64, ZIP, and
response-size pre-admission. Recalculate exact single-image bytes rather than the
old doubled width plus divider, and ensure a containment-rejected candidate
contributes no nonexistent artifact.

Compatibility aliases may remain only if needed by trusted internal callers,
but they must return the new one-image composition and must not expose or create
paired/divider semantics. Product code, current docs, public schemas, and tests
must use truthful single-image names.

## Required generated-array unit tests

Do not use external photographs, model downloads, CUDA, or image generation for
the deterministic acceptance gate. Generate nonuniform RGB arrays and boolean
masks in the tests. Expected geometry must use small visibly independent
brute-force Euclidean oracles rather than the production optimized transform.

Cover at least:

1. **Normal mask:** exact inclusive raw bbox, `L`, radius formula, exact D,
   exterior contour, default exact `2W x 2H` nominal crop, blur sigma, source
   composite, final dimensions, and repeat-byte determinism.
2. **Single image/no duplication:** use unique asymmetric markers and assert
   one candidate view only, no side-by-side copy, no divider/constant band, no
   black/zero/neutral fill, and no second untouched crop.
3. **Pixel authority:** every source-composite pixel under D equals the source
   byte-for-byte; every exact configured contour-color pixel is outside D; all
   remaining pixels equal an independently produced Pillow Gaussian-blurred
   crop. Assert D and contour are disjoint and their union is fully inside crop.
4. **Merged/large mask:** support and crop remain mask/dilation-derived and
   original pixels throughout D are unchanged.
5. **Fragmented mask:** all components and reached support remain visible, while
   no rectangular bridge or unblurred bbox hole is introduced.
6. **Image edges/corners:** each source edge and corner clamps safely, has no
   negative/wrapped coordinates, and contains all in-source D/contour when
   accepted.
7. **Extreme aspect/pathological containment:** include accepted and rejected
   cases. The rejected case must produce the exact diagnostic, zero QA calls,
   zero debug artifacts, unchanged label/score, and continued processing of a
   following valid candidate.
8. **Contour controls:** enabled/disabled, calculated width below/inside/above
   configured bounds, custom valid RGB, and strict exterior-only placement.
9. **Configuration:** every default/boundary/equality case, A/B/A isolation,
   min/max inversion, strict bool/integer/RGB types, nonfinite values, unknown
   fields, old BLIP3 fields/mode rejection, and unchanged CLIP behavior.
10. **Query:** exact client question preservation, exact fixed instruction,
    absence of every left/right/pane/target-only legacy phrase, and unchanged
    question/token limits.
11. **Model/debug identity:** capture the literal PIL/RGB image at the real QA
    seam and compare it to the compositor and decoded PNG. Multiple questions
    prove one composition per candidate and identical one-image bytes per call.
12. **Inputs/identity:** immutable source image/mask/config, one-based original
    SAM2 candidate ID and zero-based filtered index across filtering, BLIP3,
    diagnostics, debug artifacts, objects, JSON, and ZIP.

## API, integration, resource, and regression tests

- Validate effective defaults and explicit settings through real service YAML
  parsing, Pydantic response schemas, authenticated capabilities, JSON, and ZIP.
- Exercise L0-L3. Effective policy is present at every level; detailed
  composition/containment records and debug artifacts occur only at L3.
- Prove JSON/ZIP manifest equality and debug descriptor/hash/size/payload parity.
- Prove exact artifact pre-admission and zero model calls when the new smaller
  one-image artifact would exceed any item/count/total/response bound.
- Prove two consecutive requests and A/B/A with different BLIP3 view settings do
  not leak config, arrays, diagnostics, or holder state and do not reload the
  resident BLIP3 model.
- Keep genuine candidates processable under mocked/fake QA; semantic accuracy is
  not the deterministic gate.
- Replace paired-view assertions without deleting unrelated CLIP/mask-isolation,
  BLIP rule, resource, source-identity, service-level, or legacy CLI regressions.
- Run the full CPU/offline suite; no test may require GPU or network.

## Documentation and architecture refresh

Update the current contract comprehensively in the same implementation commit:

- `ARCHITECTURE.md`, `README.md`, `TESTING.md`;
- `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`; and
- any other current document found by a final repository-wide search.

Document the exact defaults/ranges/formulas, inclusive/half-open bbox conventions,
source crop and model-resize sequence, Gaussian implementation and sigma clamp,
candidate-local rejection semantics, fixed instruction, L3 metadata/debug
identity, resource bounds, and the fact that this is pixel-boundary evidence—not
semantic accuracy. Remove every current claim that BLIP3 receives a pair, left/
right panes, a divider, zero fill, dimmed context, or the old BLIP3 mode/fields.
Historical documents and immutable OAP orders/reports remain untouched and may
truthfully describe superseded evidence.

Do not change `zap-it.v1` solely for this bounded evolving service-contract
update. Capabilities are the authoritative discoverable field/default/range
surface.

## Scope and non-goals

Expected product paths include `src/core/mask_views.py`,
`modules/verifier/blip3.py`, `src/core/engine.py`, `src/core/results.py`,
`src/core/config.py`, `src/service/yaml_input.py`, `src/service/schemas.py`,
`src/service/capabilities.py`, `src/service/envelope.py`, focused tests, current
docs, `oap/active`, this order, and the final report. Exact paths may differ if
the report explains a smaller coherent design.

Do not change SAM2 generation, post-filtering, CLIP's candidate view or model
logic, task-specific BLIP3 questions/rules, final label semantics, model identity/
revision, generation limits, device/residency selection, dependencies/lockfiles,
auth/network policy, service unit/environment/key, artifact naming, response
limits, visualization, geometry, YOLO, release/tag/package publication, Git
history, or `CRITICAL.md`.

Do not add SciPy/OpenCV or another dependency for dilation/blur. Use the existing
exact Euclidean implementation and Pillow already in the declared CPU package.
Do not use image generation; generated arrays are more exact and redistributable
for this acceptance gate.

## Verification and evidence

Coding must run and report:

- focused compositor, BLIP3, candidate-view, API/schema/capability, resource,
  source-identity, and documentation tests;
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`;
- Ruff format/check, compileall, documentation integrity, and `git diff --check`;
- wheel/sdist build, direct and sdist-built wheel verification/comparison,
  outside-checkout installed smoke, archive/tracked-tree secret scans,
  `twine check`, and systemd-unit verification;
- all seven implementation-head CI/CodeQL checks; and
- all seven final report-head CI/CodeQL checks.

Record exact pass/fail/skip/warning counts and coverage. GPU/model/download/live
inference evidence is not required for the pure renderer gate and must be marked
`SKIPPED` rather than passed. Preserve any honest canonical GPU skip.

The strongest reason not to merge is that a visually plausible blurred image
could still be produced by bbox-based or post-debug-only logic while a different
or paired image reaches BLIP3. Answer it with independent source-coordinate
Euclidean/crop/blur oracles, asymmetric marker arrays, literal one-image QA
capture, decoded-debug byte equality, zero-call rejection/resource tests, and
end-to-end API metadata/artifact parity.

## Live service preservation

This implementation order does not authorize stopping, restarting, reloading,
reconfiguring, or sending inference to `zap-it-lan.service`. Strategic will
decide the post-merge refresh after independent review. Read-only start/final
checks must preserve:

- enabled, active/running service PID `528963`, `NRestarts=0`, start timestamp
  `2026-08-30 01:28:56 CEST` unless an external event changed it;
- one listener `10.8.132.76:17891`, health/readiness 200, unauthenticated
  capabilities 401, and disabled docs/OpenAPI;
- exact assigned physical GPU0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090,
  24,576 MiB, driver `610.43.02`; at order publication the only compute process
  is service PID `528963` using 13,408 MiB;
- mode-0700 empty `/dev/shm/slaif-zap-it`; and
- mode-0600 operator environment with SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.

Never print, log, copy, rotate, or mutate the API key. Do not touch firewall,
VPN/network, port, unit, driver/CUDA, GPU state, another process/device/service,
or global credentials. No raw image, YAML, question, answer, label, artifact, or
path belongs in logs/metrics/OAP evidence.

## Acceptance criteria

1. Every applicable BLIP3 candidate is composed once into one image; no pane,
   divider, duplicate, black/zero fill, dimmed ring, or untouched rectangular
   context reaches QA.
2. Exact source pixels under D are preserved in the source composite; the exact
   contour is outside D; every other local-crop pixel is the deterministic
   Gaussian-blurred source scene.
3. Default geometry uses exact Euclidean `R=ceil(0.20*L)`, a 1..3-pixel exterior
   contour from `ceil(0.02*L)`, and a nominal exact `2W x 2H` centered crop.
4. A crop that cannot contain support/contour causes a candidate-local skip with
   exact bounded L3 diagnostics, no QA/debug call, and no label mutation.
5. BLIP3 receives the exact single image represented by the debug PNG; task
   questions stay client-provided and the fixed instruction is byte-exact.
6. The request schema/capabilities/effective manifest expose only the new BLIP3
   settings with strict validation and A/B/A request isolation; CLIP is unchanged.
7. L3 records exact IDs, radii, widths, bboxes, mode, sigma, dimensions, status,
   and diagnostic; JSON/ZIP and artifact hash/size/data agree.
8. Existing question/generation, artifact/response, auth/privacy, residency,
   legacy CLI, and final result behavior remain bounded and compatible where not
   explicitly superseded.
9. Generated-array normal/merged/fragmented/edge/extreme-aspect tests, literal
   model-input tests, full CPU/package/static/docs checks, and all final-head
   GitHub checks pass.
10. Current documentation contains no stale paired-view contract and agrees with
    live capabilities to be activated after merge; the running service remains
    unchanged during coding.

## Deferred human adjudication

- Decision: NONE

## Publication and report contract

- Publish this exact order and selector `019-a` in the implementation/control
  commit. Record the order SHA-256.
- Push all product/test/docs/control work to the exact branch and create the one
  PR. Do not merge or enable auto-merge.
- After implementation-head evidence and checks are fully green, publish exactly
  one immutable `oap/reports/019-a-report.md` as a report-only commit whose
  publication field is literal `SELF` and whose parent is the implementation
  SHA. Push it, verify the remote topology/bytes, and require the final-head
  checks green before signaling.
- The report must map every requirement and test, list the exact diff, schema and
  migration behavior, geometry/blur implementation, diagnostic/metadata model,
  docs search, resource admission, CI, service-preservation evidence, limitations,
  strongest reason not to merge, and critical action `NONE`.
- Status is COMPLETE only if every criterion is met. Otherwise report PARTIAL
  honestly without weakening a test or changing scope after the report commit.
- Coding must not merge, release, restart/reconfigure the service, send live
  inference, mutate host/GPU/network/credentials, or print the key.
