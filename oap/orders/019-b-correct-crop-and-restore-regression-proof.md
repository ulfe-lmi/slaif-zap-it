# OAP Work Order 019-b — Correct crop centering and restore regression proof

## Objective

Amend Objective-019 PR #75 with the bounded corrections required by strategic
review of the actual `019-a` implementation, tests, report, and final-head CI.
Keep the one-image BLIP3 design, but correct its source crop placement, avoid
retaining redundant full-source masks, restore/adapt the acceptance evidence
deleted from Objective 018, and expose the RGB-array capability bounds in a
machine-readable form. Do not change the client-provided BLIP3 questions or the
exact fixed instruction.

## Verified GitHub state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Base: `main` at `4acff3a8f7717a08481b86338453d09e754c1e86`.
- Existing PR: #75, `Objective 019: single-image BLIP3 candidate view`.
- Branch: `oap/019-a-single-image-blip3-candidate-view`.
- 019-a implementation SHA:
  `573f5ed4d92e1b988baf325c65140b334eeed9ee`.
- 019-a immutable report-only head:
  `7e24eba6941cabdc5d877ad2b6d3b511fd163282`, whose sole path is
  `oap/reports/019-a-report.md` and whose parent is the implementation SHA.
- PR is open, non-draft, clean/mergeable. All seven implementation-head and all
  seven report-head CI/CodeQL checks are successful.
- Amend this exact branch and PR. Do not create another PR, merge, or enable
  auto-merge. Preserve the 019-a order/report bytes unchanged.

## Strategic review findings

### 1. Even-sized crops are shifted one pixel left/up

The implementation calculates each start as
`floor(center - nominal_size / 2)`. Pixel coordinates identify pixel centers,
and an inclusive bbox center is `(first + last) / 2`; therefore an even nominal
crop must use `floor(center - (nominal_size - 1) / 2)`. For example, raw x
coordinates `10..19` have center `14.5`; a 20-pixel crop is exactly centered at
`5..24` (half-open `5..25`), while 019-a chooses `4..23` (half-open `4..24`).
That is not centered and can falsely reject a candidate whose support/contour
fits on the right but not after the artificial left shift.

The normal 019-a test asserts only crop dimensions, not exact crop coordinates,
so the defect passes.

### 2. Objective-018 acceptance coverage was deleted

The merged base had 791 passing tests plus one honest GPU skip. The 019-a report
has only 760 passing tests plus the same skip. The three heavily rewritten test
files delete 31 collected cases and remove named proofs including:

- the exact 512-by-512 striped rectangular-leakage fixture;
- bounded radius-512 exact-dilation resource regression;
- literal real CLIP `classify_single` processor-input capture;
- fixed CLIP debug artifact/source-ID tests;
- CLIP request-local A/B/A behavior and fail-if-reinitialized CLIP/BLIP3 holder
  construction seams;
- full source-candidate/filtered-index flow through removal, CLIP, BLIP3, final
  ordering/visualization, objects, JSON, ZIP, and debug records;
- broad CLIP validation/default/inclusive-endpoint, holes, disconnected, border,
  immutable-input, and marker regressions; and
- prior BLIP invalid-input, rule, fixed-name, and positive/hard-negative seams
  that should have been adapted to the new one-image renderer where still
  applicable.

The 019-a order explicitly prohibited deleting unrelated CLIP/mask-isolation,
resource, source-identity, service-level, and legacy regressions. A smaller
replacement test file and green CI do not answer that gate. The 019-a report's
claim that these remain green is therefore overbroad.

### 3. Composition retains three unnecessary full-source boolean arrays

`Blip3VerificationComposition` retains `source_mask`, `source_support_mask`, and
`source_contour` in addition to crop-local masks. At the 64-megapixel input
limit, those three boolean arrays retain roughly 192 MB per live composition,
despite the model/debug seam needing only the crop-local arrays and bbox metadata.
The geometry calculation may use bounded temporary source masks, but the returned
composition must not keep redundant full-source copies after the crop is built.

### 4. `contour_rgb` capability limits are prose-only and two descriptions are false

Capabilities report `contour_rgb` only as `type: array`; exact length three and
strict integer channel range 0..255 exist only in a note. The user required
fields/defaults/limits through capabilities. Add optional array-item constraint
fields to the generic capability schema so clients can discover these limits
programmatically without changing other fields' serialized shape.

Also correct these claims everywhere they occur:

- D is restored from source bytes; the contour is painted with configured RGB,
  not “restored byte-for-byte”.
- A decoded lossless debug PNG has RGB pixels identical to the model-input
  array; encoded PNG bytes do not equal raw RGB/image bytes.

These findings are ordinary correctness, resource, test, and documentation
defects. They do not meet the CRITICAL threshold.

## Required corrections

### 1. Exact centered crop arithmetic

For each axis use the already-calculated inclusive bbox center and nominal
integer crop size:

```text
unclamped_start = floor(center - (nominal_size - 1) / 2)
unclamped_end = unclamped_start + nominal_size
start = clamp(unclamped_start, 0, source_size)
end = clamp(unclamped_end, 0, source_size)
```

Keep the existing independent endpoint clamp/no-shift rule. Do not enlarge the
source crop or exceed the nominal size. Continue to reject if any in-source
support/contour pixel lies outside the final half-open crop.

Add visibly independent test-owned arithmetic for:

- odd and even raw bbox widths and heights;
- odd and even nominal sizes, including a non-integer multiplier whose product
  is ceiled;
- interior and each boundary-clamped placement; and
- an asymmetric case where 019-a's one-pixel shift rejects but the exact centered
  crop contains D plus contour and is accepted.

Assert the literal half-open crop bbox, not only dimensions. Do not use the
production crop helper to derive expected values.

### 2. Restore and adapt the complete regression matrix

Use base commit `4acff3a8f7717a08481b86338453d09e754c1e86` as the source of
the deleted Objective-018 tests. Restore unchanged every proof that does not
depend on the superseded BLIP3 pair. Adapt rather than discard every proof whose
BLIP3 assertions changed. It is acceptable to split or rename tests, but the
019-b report must include a table mapping every removed base test name to its
restored name or its explicit one-image replacement and state what it proves.

At minimum the final test suite must again prove all of the following:

1. Exact 512x512 nonrectangular-mask/striped-distractor leakage and repeatability.
2. Exact Euclidean dilation against an independent brute-force oracle, including
   radius 512 bounded local-window resource behavior.
3. CLIP bbox-storage-only semantics, exact M/D zero-fill visibility, holes,
   disconnected components, border/corner clipping, markers, default and
   inclusive validation boundaries, immutable inputs, and deterministic bytes.
4. Literal real CLIP `classify_single` processor `images=` capture equal to the
   shared builder output; fixed debug name and source candidate ID.
5. Tiny-mask source-space-before-resize proof with independent nearest mapping
   and applicable one-image BLIP3 composition/resize expectations. No prohibited
   source crop enlargement.
6. Exact BLIP3 one-image QA/debug identity, fixed one-based candidate/question
   IDs, all input validation, any/label rule behavior, positive and same-crop
   hard-negative pixel-isolation seams adapted to blurred surroundings, and no
   paired/divider assertions.
7. Full source candidate identity and zero-based filtered index across
   post-SAM2 removal, CLIP, candidate-local BLIP3 rejection/success, final label
   filter/order, labelled visualization, composition/debug records, JSON objects,
   and ZIP manifest.
8. L0-L3 API response policy, JSON/ZIP artifact hash/size/payload parity, exact
   one-to-one debug records, and the one-per-candidate composition record.
9. A/B/A request-local CLIP and BLIP3 settings with exact model-input restoration,
   stable holder IDs, and fail-if-invoked guards at the actual CLIP `initialize`
   and BLIP3 `_Blip3QA` holder-construction seams. A hard-coded count is not proof.
10. Pre-model resource admission for count, single-item, total, and response
    limits, including zero QA/debug calls and zero artifact budget for a
    containment-rejected candidate.
11. Explicit asymmetric no-pane/no-duplicate/no-divider evidence: unique source
    markers occur once in the single model input, exact dimensions are not the
    old doubled width, and there is no injected constant band/fill.
12. Accepted and rejected extreme-aspect masks plus normal, merged, fragmented,
    and edge/corner masks with nonzero D and enabled contour.
13. Strict config validation across every field/type/nonfinite/boundary/equality/
    inversion/unknown/legacy case and exact authenticated capability/effective
    values.

The full collected CPU suite must contain at least the base's 791 passing tests
plus the one honest GPU skip, without dummy, duplicate, or assertion-free cases.
If parameterization changes the numeric collection count, preserve at least 791
meaningful passing cases and explain the exact delta. Do not weaken or delete an
unrelated test to satisfy this count.

### 3. Return only crop-local masks

Remove `source_mask`, `source_support_mask`, and `source_contour` from the
returned composition and all exports/callers/tests. The returned object may keep:

- final model-input RGB/PIL image;
- source composite crop;
- crop-local raw mask, support mask, and contour;
- bounded scalar/bbox/dimension metadata.

Do not retain any boolean array with source-image shape after composition unless
the crop itself equals the source. Ensure geometry temporaries become unreachable
after return. Add a generated large-source/small-candidate test that inspects all
retained ndarray shapes/nbytes and proves retained mask storage is crop-bounded,
while source-composite/model/debug correctness stays unchanged. Do not add a new
dependency or a flaky process-RSS threshold.

### 4. Machine-readable RGB capability limits and truthful wording

Extend `CapabilityField` with optional, omitted-by-default array constraints,
using clear names such as:

- `min_items: 3`;
- `max_items: 3`;
- `item_type: integer`;
- `item_minimum: 0`;
- `item_maximum: 255`.

Populate them for `candidate_views.blip3.fields.contour_rgb`. Keep all unrelated
capability field serializations stable by excluding absent values. Add Pydantic/
capability tests that assert the exact machine-readable object and strict YAML
validation.

Correct capabilities and all current docs to say support D is source-restored,
the exterior contour is painted, and decoded PNG RGB pixels equal the exact model
input. Do not rewrite immutable orders/reports or history documents. Run the
repository-wide stale-language search again.

## Scope and non-goals

Expected correction paths are `modules/verifier/blip3.py`,
`src/service/capabilities.py`, focused schemas/docs only if needed,
`tests/test_mask_views.py`, `tests/test_candidate_view_api.py`,
`tests/test_verifier_blip3.py`, other restored focused tests where their base
proof lived, `TESTING.md`, relevant current contract docs, `oap/active`, this
order, and the new report. Preserve every 019-a product behavior not named above.

Do not change SAM2, post-filtering, CLIP product behavior, client question/rule
semantics, the exact fixed instruction, model/generation/residency/device policy,
artifact names/limits, auth/network/service settings, dependencies, release,
Git history, or `CRITICAL.md`.

Do not use image generation, external photographs, GPU, model downloads, or live
inference. Generated arrays and fakes are exact for this correction.

## Verification

- Run the exact centered-crop, retained-array, independent dilation/resize,
  no-duplicate, extreme-aspect, CLIP literal processor/debug, source identity,
  A/B/A fail-if-reinitialize, config/capability, rejection/resource, JSON/ZIP,
  and adapted positive/hard-negative focused cases.
- Report a removed-base-test-to-final-test mapping table and exact collected test
  counts before (`791 passed, 1 skipped`) and after 019-b.
- Run the full canonical CPU/offline suite with coverage and exact counts.
- Run Ruff format/check, compileall, docs integrity, `git diff --check`, full
  package/release/secret/sdist-built/outside-tree/twine/systemd checks.
- Require all seven implementation-head and all seven final report-head GitHub
  checks successful.

The strongest reason not to merge is that new green tests replaced rather than
preserved prior acceptance evidence and failed to assert the exact crop coordinate
where the runtime is wrong. Answer it with the literal base-test mapping,
independent odd/even crop bboxes, restored semantic seams, fail-if-reinitialized
holders, and crop-bounded retained arrays—not with another broad “all tests pass”
claim.

## Live service preservation

Do not restart, stop, reload, reconfigure, or send inference to
`zap-it-lan.service`. Read-only start/final checks must preserve the service as
enabled and active/running on PID `528963`, `NRestarts=0`, timestamp
`2026-08-30 01:28:56 CEST`, listener `10.8.132.76:17891`, health/readiness 200,
unauthenticated capabilities 401, docs/OpenAPI 404, and its only assigned-GPU
process.

The assigned card remains physical index 0, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090,
24,576 MiB, driver `610.43.02`; the service process currently reports 13,408 MiB.
Preserve empty mode-0700 `/dev/shm/slaif-zap-it` and the mode-0600 environment
digest `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.

Never print/read/copy/mutate the API key. Do not touch another service/process,
GPU/device, driver, firewall, VPN/network, port, host config, or credential.

## Acceptance criteria

1. Every crop uses the exact inclusive-pixel-center formula and independent
   odd/even/boundary/asymmetric tests assert its half-open bbox.
2. The false one-pixel containment rejection is eliminated without shifting or
   silently clipping true support/contour.
3. The full suite restores at least 791 meaningful passing CPU cases plus the
   honest GPU skip and maps every deleted base proof to unchanged/adapted evidence.
4. Literal CLIP processor/debug, 512 leakage, radius-512, tiny mask, source
   identity, BLIP rule/isolation, API parity, resource, A/B/A and fail-holder
   seams are all present and green.
5. The one-image composition retains no redundant source-shaped boolean copies;
   retained mask storage is crop-bounded.
6. `contour_rgb` length/type/channel limits are machine-readable in capabilities,
   and docs/capabilities describe painted contour and decoded-PNG identity
   truthfully.
7. Client questions and exact fixed instruction remain unchanged; no paired
   design returns.
8. Full static/package/docs/release/CI/CodeQL evidence is green and the private
   service remains unchanged.

## Deferred human adjudication

- Decision: NONE

## Publication and report contract

- Include this exact order and selector `019-b` in the correction implementation/
  control commit on PR #75; record its SHA-256.
- Push all non-report corrections, wait for all seven implementation-head checks,
  then publish exactly one immutable `oap/reports/019-b-report.md` report-only
  SELF child whose parent is the literal correction implementation SHA.
- The report must include exact findings/fixes, crop examples/formula, retained
  array inventory, capabilities object, every test mapping and count, diff scope,
  docs/stale search, package/CI, service preservation, limitations, strongest
  reason not to merge, and critical action `NONE`.
- Require the same seven checks green on the final report head before signaling.
- Status is COMPLETE only if every criterion passes. Otherwise report PARTIAL;
  do not weaken tests or mutate after the report commit.
- Coding must not merge, accept, release, restart/reconfigure the service, send
  inference, mutate host/GPU/network/credentials, or print the key.
