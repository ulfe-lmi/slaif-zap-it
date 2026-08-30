# OAP Work Order 018-a — Close the exact mask-view acceptance matrix

## Objective

Close the remaining deterministic evidence gaps against the human-authored
mask-isolated candidate-view goal after Objective 017 merged. This is a narrow
test/provenance objective: add the exact generated-array, seam-identity,
request-isolation, response-level and identity-through-ordering assertions that
the Objective-017 order required but its merged tests did not all express
literally.

Do not weaken, reinterpret or replace the existing Objective-017 behavior. The
shared mask-derived visibility boundary remains authoritative: target-only shows
source pixels only from `M`; context shows source pixels only from the exact
Euclidean dilation `D`; a rectangular bbox is storage/resize geometry only.

## Verified starting state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Default branch: `main`.
- Exact base SHA: `03def697373f2ae83d03494315aa96c800f0bcdf`.
- Base commit is merge PR #73, Objective 017 mask-isolated candidate views.
- PR #73 is merged; its immutable report head is the merge's second parent.
- No pull request is currently open.
- Create one new branch `oap/018-a-close-mask-view-acceptance-matrix` from the
  exact base SHA and one new PR titled
  `Objective 018: close mask-view acceptance matrix` against `main`.
- One numeric objective remains one PR. Coding must not merge or enable
  auto-merge.

## Why a new objective is required

The merged implementation and live evidence establish the intended pixel
boundary, and current-head plus post-merge CI/CodeQL are green. Strategic's final
audit against the authoritative goal nevertheless found that several exact test
instructions were represented only by smaller or adjacent cases:

1. The primary bbox-hole test uses a 32x36 array and a solid distractor, not the
   explicitly required 512x512 nonrectangular-mask/high-contrast-striped
   rectangular-leakage fixture.
2. Disconnected and border tests establish important shape facts but do not
   jointly assert all original-source visibility/nonvisibility conditions with
   uniquely identifiable source pixels.
3. Tiny-mask tests prove resize support reapplication but do not state every
   source-space-crop/non-enlargement assertion together.
4. The CLIP seam test captures the argument to `classify_single`, not literally
   the `images=` argument received by a mocked CLIP processor while the real
   `classify_single` path executes.
5. Effective candidate-view response policy is proved at L3, but not explicitly
   at each L0-L3 level for applied false/true cases.
6. Offline API request-local isolation does not yet perform one A/B/A sequence
   varying both CLIP and BLIP3 context fractions on stable resident holders.
7. Source ID and filtered index are covered in separate tests, but one focused
   test does not carry them across an actual post-SAM2 removal, semantic stages,
   final ordering, visualization input, debug records and public objects.

These are evidence gaps, not permission to claim the implementation is wrong.
Tests must remain independent of semantic model accuracy and external photos.

## Scope

### 1. Exact 512x512 rectangular-leakage fixture

Add a self-contained test that creates exactly one 512x512 RGB `uint8` source
array and a nonrectangular boolean `M` whose tight bbox contains an empty region.
Place a deterministic high-contrast striped distractor wholly outside `M` but
inside `bbox(M)`. Choose configuration so the distractor is also outside `D`.

Assert all of the following directly:

- source pixels under the distractor are nonzero and uniquely recognizable;
- none of those source pixels appears in target-only;
- none appears in context when outside `D`;
- every target-only pixel outside the cropped `M` is exactly zero;
- every context pixel outside the cropped `D` is exactly zero;
- pixels inside `M` are byte-identical to the corresponding source pixels;
- bboxes/radius agree with the documented formula; and
- a repeat produces identical arrays, masks, bboxes, metadata and lossless PNG
  SHA-256 values.

The test must fail against a bbox-filled or untouched-rectangle implementation.

### 2. Dilation, holes, disconnected components and borders

Add or strengthen generated-array tests so each original condition is explicit:

- marker exactly within the Euclidean radius appears only in context; a marker
  outside it appears in neither output; neither marker appears target-only;
- fraction zero, ceil behavior and min/max overrides report exact raw/effective
  radius values;
- a U/ring hole distractor stays zero before true dilation reaches it and becomes
  eligible only when `D` actually reaches it;
- two separated components both preserve their uniquely colored source pixels;
  the intervening space stays zero unless reached by true dilation and no
  rectangular bridge is introduced;
- masks touch every edge and every corner, plus at least one disconnected
  border case; use nonzero uniquely patterned source pixels to prove no negative
  coordinate, wraparound or additional-source-pixel leakage, not merely valid
  shapes; and
- output dimensions and bboxes remain valid and clipped.

Use an independent definition/oracle where appropriate; do not restate the
production algorithm as the sole oracle.

### 3. Tiny-mask and contour contract

For a mask smaller than the BLIP3 model-input minimum, assert in one focused
case that:

- `M` and `D` are built in source space;
- the source crop is exactly the tight crop around `D` and is not expanded to
  the model minimum;
- neutralization occurs before enlargement;
- masks use the documented deterministic nearest-neighbor mapping;
- RGB uses the documented bilinear mapping;
- target-only remains zero outside the resized target mask;
- context remains zero outside the resized support mask; and
- the actual paired input contains no prohibited source pixel.

Cover BLIP3 contour width zero and a positive width. Prove contour pixels are
only in `D - M`, never overwrite target pixels, have the documented color,
placement and deterministic width, and repeated output is byte-identical.

### 4. Literal CLIP processor seam and BLIP3 QA seam

Exercise the real `_ClipFilter.classify_single` path with bounded fake
torch/model/text embeddings and a mocked processor that records its exact
`images=` value. Do not stop at replacing `classify_single` itself. Assert:

- the processor receives byte-for-byte the shared builder's context RGB;
- pixels outside `D` are zero;
- the old unmasked bbox/distractor is absent;
- the emitted debug PNG decodes byte-for-byte to the processor input; and
- the selected prompt decision is based on the fake complete similarity vector,
  preserving existing single-view semantics.

Retain the existing literal `_Blip3QA.answer` capture and strengthen it only as
needed to prove the exact PIL/array equals the safe shared-builder pair and the
lossless debug PNG. The pair must be built once per candidate and reused across
applicable questions.

No real model, model download, GPU or network is allowed in these tests.

### 5. One focused source-identity flow

Create a deterministic injected core/API flow with multiple source candidates
where at least one earlier candidate is removed by the post-SAM2 filter and final
area ordering differs from retained source order. Capture the `final_objects`
argument delivered to the labelled-visualization stage.

Assert one-based `source_candidate_id` and persistent zero-based
`filtered_index` across retained post-SAM2 masks, CLIP, BLIP3, candidate-view
debug records, captured final visualization objects, public JSON objects and ZIP
manifest objects. Assert final `instance_id` reflects final deterministic order
without renumbering source/filter identity. Artifact names must use source IDs,
not current list positions, and remain one-to-one with records.

Do not infer identity from a label or area alone; assert the numeric fields.

### 6. Response levels and request-local A/B/A

Using one injected service instance with stable fake resident CLIP and BLIP3
holders, perform generated-image A/B/A requests where A and B vary both stage
context fractions and A2 restores A exactly.

Assert:

- effective `candidate_views.clip` and `.blip3` values are present and correct at
  verbosity 0, 1, 2 and 3;
- `applied` is false when a stage is absent and true when configured;
- candidate-view input records/artifacts remain L3-only;
- A1/B/A2 model inputs and effective metadata change for B and restore exactly
  for A2;
- resident holder identity/initialization counts remain stable;
- no config or array from one request is mutated or retained into another;
- JSON descriptors and ZIP members remain exact PNG/name/hash/size matches; and
- public schemas continue to validate the responses.

The request sequence must not depend on a semantic label answer; use deterministic
fakes.

### 7. Documentation and provenance

Update current testing/provenance documentation only where necessary to state
the exact 512x512 striped fixture, literal CLIP processor seam, source-identity
flow, response-level matrix and offline A/B/A evidence. Do not rewrite historical
orders or reports. Do not claim semantic accuracy, recall or precision.

The immutable Objective-017 reports remain unchanged. The Objective-018 report
must disclose that this objective exists because strategic found an exact
acceptance-evidence gap only after merging PR #73.

## Implementation boundary

This round is expected to change tests and, if truthful documentation requires
it, current test/provenance docs plus OAP control/report files only. Do not change
runtime/product behavior, schemas, defaults, model adapters, algorithms,
dependencies or service configuration merely to make a test convenient.

If an exact required test exposes a runtime/product defect, preserve the failing
reproduction, do not weaken it, do not make an unreviewed broad fix, and return a
PARTIAL report describing the defect and the smallest likely correction. A later
018-b order will decide and authorize the runtime correction and any restart.

## Non-goals

- No CLIP two-view weighted embedding enhancement.
- No neutral/blurred fill implementation; zero remains the sole public fill.
- No semantic-model accuracy benchmark or external photograph.
- No SAM2 parameter/profile/model/residency change.
- No renderer, post-filter, geometry, YOLO, identity-PNG or raw-visualization
  behavior change.
- No dependency, lockfile, model, weight, cache, environment, service unit,
  network, firewall, driver, CUDA, GPU-selection or credential change.
- No release, tag, package publish, public/WAN exposure, history rewrite or
  destructive operation.

## Verification

### Focused and canonical CPU evidence

Run the focused candidate-view, BLIP3, core, visualization, schema and service
tests affected by the change. Then run the canonical full CPU/offline suite with
coverage. Record exact pass/skip/warning counts and coverage. The one honest GPU
test remains skipped in CPU CI; no model downloads or network may be required.

Run every repository static/package gate expected on current `main`:

- Ruff format check and lint;
- compileall;
- documentation integrity;
- `git diff --check`;
- wheel and sdist build;
- release artifact verification for both artifacts;
- install the sdist-built wheel outside the checkout and smoke the service;
- compare wheel member manifests/digests;
- archive and tracked-tree secret scans against the existing reviewed baseline;
- `twine check`; and
- `systemd-analyze verify` for the tracked unit.

### CI/checks

Push the implementation/control commit and wait for every current-head CI and
CodeQL check: static/build, release audit, tests on Python 3.10/3.11/3.12,
CodeQL workflow analysis and CodeQL check. None may be pending, missing, failed,
cancelled or skipped unexpectedly. Publish the immutable report-only SELF child
and require the same complete green set on that final PR head.

### Live service preservation

Do not restart, stop, reload, reconfigure or replace `zap-it-lan.service` in this
round. Do not send inference requests. The already-qualified Objective-017
runtime must remain enabled, active and ready while CPU/CI evidence runs.

At start and final report, verify read-only:

- unit PID `528963` unless an external failure changed it, `NRestarts=0`, enabled,
  active and ready;
- exactly one listener at `10.8.132.76:17891` and one assigned-GPU compute
  process;
- physical GPU0 is UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090,
  24576 MiB, driver `610.43.02`; application uses logical `cuda:0` only;
- `/dev/shm/slaif-zap-it` remains mode 0700 and empty;
- mode-0600 operator environment digest remains
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
- `/healthz` and `/readyz` return 200;
- unauthenticated protected endpoints remain denied and docs/OpenAPI remain 404;
  and
- no credential, prompt, answer, raw image/config, traceback, OOM or sensitive
  path is emitted into logs/report.

The key may be read by the existing service process and a bounded local auth
probe, but it must never be printed, copied into a command literal, report,
artifact, log, Git history or process arguments. Do not rotate, replace or
otherwise mutate it.

If the service changes PID, restart count or readiness for reasons outside this
order, disclose the fact and investigate read-only; do not restart it.

## Security, resource and scope constraints

- Preserve the private-LAN/key boundary. No public/WAN exposure.
- Uploaded/client strings never influence artifact paths.
- Tests use generated arrays and bounded fake processors only.
- No persistent request data; no repository output directory use.
- No mutation of assigned/unassigned GPU processes or unrelated services.
- Do not edit `CRITICAL.md`; its sole entry CRIT-0001 has a human `ACCEPTED`
  disposition and does not gate this objective.
- No new deferred human adjudication is warranted for this routine reversible
  evidence correction.

## Acceptance criteria

1. The exact 512x512 striped rectangular-leakage test proves zero outside `M`/`D`
   and byte identity inside `M`.
2. Generated-array tests explicitly prove dilation markers/formula, bbox holes,
   disconnected components, all borders/corners, tiny-mask resize order and
   positive/zero-width contour behavior.
3. A mocked CLIP processor receives the exact builder context input, and mocked
   BLIP3 QA receives the exact safe pair; debug PNG decodes match model inputs.
4. One focused filtering/reordering flow proves source ID and filtered index
   through both semantic stages, visualization, debug records, JSON and ZIP.
5. Effective policy is proved at L0-L3 and an offline stable-holder A/B/A varies
   both stages without request-state leakage.
6. No Objective-017 runtime behavior, dependency, schema/default, service or
   environment is changed unless a failing test causes a transparent PARTIAL
   return for a later strategic decision.
7. Focused/full CPU, package/static, final-head CI and CodeQL evidence is green.
8. The private-LAN service remains on the same ready PID/process/listener with
   unchanged key-file digest and empty RAM workspace; no restart or inference is
   performed.

The strongest reason not to merge is that tests added after implementation can
be tautological or give false comfort while missing the real model seam. Answer
this with uniquely identifiable prohibited source pixels, independent expected
masks/mappings, literal processor/QA captures, lossless decode equality and
end-to-end numeric identity assertions—not helper self-comparison alone.

## Deferred human adjudication

- Decision: NONE

## Publication and report contract

- Include this exact order and `oap/active` in the first implementation/control
  commit; record the order SHA-256 in the commit message/body or report.
- One implementation/control commit may contain the bounded tests/docs/control
  work. Any correction commits must remain on the same Objective-018 PR.
- After all implementation-head checks and required evidence are complete,
  publish exactly one immutable `oap/reports/018-a-report.md` as a report-only
  SELF child commit. That commit changes only the report path.
- Report exact base/branch/PR/implementation/report SHAs and topology; file/diff
  scope; each acceptance result; focused/full/static/package/CI evidence;
  service-preservation facts; limitations; strongest reason not to merge; and
  critical-register action `NONE`.
- Status is COMPLETE only if every ordered criterion passes exactly. Otherwise
  report PARTIAL with the narrow defect/gap and do not claim acceptance.
- Coding must not merge, accept, enable auto-merge, tag, release, publish,
  restart the service or print the key.
