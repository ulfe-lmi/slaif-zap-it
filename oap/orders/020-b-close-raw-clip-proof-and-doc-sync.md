# OAP Work Order 020-b — Close raw-CLIP proof and documentation synchronization

## Authority and exact repository state

- Continue numeric Objective 020 on the existing PR #76 and branch
  `oap/020-a-domain-neutral-clip-routing-pipeline`; do not create another PR.
- Reconcile from published report head
  `cecaaf1d4e2447147fbdea86fccf8e6eab27525d`, whose implementation parent is
  `3f75035fca95d68b7cfc054aace174ffb64353ec` and whose base remains remote
  `main` at `cc325d5d97acefe7624aecfe9fa157dbf37ce600`.
- Preserve the immutable 020-a order and report. Publish this exact 020-b order,
  update `oap/active` to `020-b`, implement one bounded correction commit, then
  publish a new report-only `OAP report 020-b (SELF)` commit whose only changed
  path is `oap/reports/020-b-report.md` and whose parent is the correction head.
- Do not merge, enable auto-merge, restart/reconfigure the service, perform live
  inference, touch a GPU/model cache, read/print/copy the bearer, mutate host
  networking, or modify `CRITICAL.md`.
- Critical-register action: NONE.

## Why continuation is required

PR #76 is green, but independent strategic review found acceptance proof and
documentation gaps that make the strongest 020-a merge claim unproven:

1. `test_real_clip_classify_single_receives_literal_processor_context_view`
   still selects trusted legacy `mask_dilated` through `_clip_config()` and
   compares the processor/debug image to `build_mask_views`. It therefore does
   not prove that the new API/default `raw_bbox_crop` crosses the actual CLIP
   processor boundary unchanged. The report's processor-seam claim is too broad.
2. The focused router matrix omits explicit uncertain-winner and clear-negative
   branches. It also does not directly prove canonical natural-language prompt
   bytes are embedded while machine identifiers are not substituted or split.
3. `StageStatus(name="clip")` is assembled after BLIP3 has overwritten
   `masked_after_clip`, so its detail can report the routed/post-BLIP count rather
   than the number CLIP actually scored.
4. generated `PostFilterDiagnostics` schema omits the canonical
   `removed_by_min_area`, `removed_by_max_area`, width/height/aspect and
   border-touching aggregate fields that runtime emits.
5. documentation is not fully synchronized. In particular,
   `docs/ALGORITHMS.md` still describes one-or-more prompts and top-label CLIP
   classification as the service contract, describes the old BLIP outline task,
   and `docs/CONFIG.md` later says false matches are always relabelled
   `negative`, contradicting configurable `falsecategory`. Search all current
   docs for equivalent stale wording, not only those exact lines.

## Required implementation corrections

### A. Preserve the actual post-CLIP count

- Immediately after CLIP returns, retain an unambiguous request-local value for
  the number of candidates actually scored by CLIP.
- Use that value for `candidate_counts.after_clip`, `clip_scored`, and the CLIP
  stage-status detail. Later routing/BLIP3 mutation must not alter it.
- Keep all existing routing, BLIP3 and final counts independently accurate.
- Add an integration assertion with at least one clear negative so the CLIP
  stage detail/count remain the pre-routing count while routed/verified/final
  counts reflect later losses.

### B. Make canonical dry-run routing coherent

- The trusted dry-run path must not silently discard every candidate merely
  because `_DryRunClipFilter` omitted `clip_scores` or emitted an identifier not
  present in the configured canonical label map.
- Give the dry-run adapter the request's effective label order and emit a finite,
  deterministic complete score mapping for every configured label, a winner,
  score and prompt consistent with that mapping. It must require no torch/model.
- Preserve historical dry-run behavior for legacy configurations where feasible,
  but the canonical API/core contract must route deterministically and retain
  stable source IDs.
- Add a CPU test covering canonical CLIP+routing+BLIP dry-run flow without model
  initialization.

## Required acceptance tests

### C. Exact raw CLIP model-boundary identity

Add or convert a test that uses the actual `_ClipFilter.filter_masks` and
`classify_single_scores` processor seam with `mode: raw_bbox_crop`.

The generated scene must include a nonrectangular or holed mask plus distinctive
source pixels outside the mask but inside both the tight bbox and padded crop.
Assert all of the following:

- expected radius is the documented half-up formula with min/max bounds;
- expected half-open crop bbox is independently source-clamped;
- every RGB pixel received by the mocked CLIP processor equals the corresponding
  source slice byte-for-byte, including mask holes and surroundings;
- no fill, dimming, blur, contour, alpha, silhouette, second pane or resizing is
  introduced before the processor;
- the image-array stored by the bounded sink and the decoded lossless debug PNG
  are byte-identical to the processor input;
- fixed artifact name/source ID/filtered index and crop metadata agree;
- the candidate receives every configured per-label cosine score in
  configuration order, plus the deterministic winner; and
- source image and mask remain unchanged.

Keep a clearly named separate legacy masked-mode test if compatibility proof is
still useful; do not let it masquerade as the raw service/default proof.

### D. Prompt and routing branch proof

- Exercise canonical prompt encoding through the mocked text processor. Use a
  machine identifier containing an underscore and a natural-language value
  containing punctuation/comma or a newline. Assert the processor receives the
  exact configured value as one prompt and never receives the identifier as
  replacement prose or splits the canonical value.
- Add direct deterministic router cases for:
  - target top-1;
  - target in top-k;
  - target within inclusive score margin;
  - target meeting inclusive minimum score;
  - an explicit uncertain winning label;
  - a clear negative that is not routed;
  - deterministic cap behavior and stable source-ID tie break.
- Assert all complete score vectors and exact primary/matched/cap reasons survive
  the L3 JSON and ZIP manifest, including the clear negative that does not reach
  BLIP3.
- Add hostile validator cases for routing field types/ranges, missing/orphan
  BLIP rules, unsupported legacy API forms and safe identifier/prompt bounds.
  Check precise `invalid_config` versus `unsupported_field`, not a set of either,
  wherever the contract already fixes the distinction.

### E. Generated schema closure for Objective 020 fields

- Extend `PostFilterDiagnostics` with every canonical runtime aggregate:
  `removed_by_empty_mask`, min/max area, min/max width, min/max height,
  min/max aspect ratio, and border touching. Retain the legacy aggregate names.
- Ensure `CompletionResponse.model_validate` preserves/rejects rather than
  silently hiding contradictions. Add schema/OpenAPI assertions for the new
  fields, complete CLIP scores, routing decision/reason, BLIP configured/effective
  question and answer mapping, stable candidate IDs, counts and timings.
- Do not implement Objective 021 artifact truncation or resource alternatives in
  this amendment. It remains honest for the docs to state optional-artifact
  overflow is currently inference-fatal.

## Documentation correction

Perform a repository-wide review of current user/operator documentation and
align it to the implemented canonical service contract:

- service CLIP labels are exactly one natural-language value per safe machine
  identifier; trusted CLI alone may retain multi-prompt/flattened-key behavior;
- CLIP scores every surviving geometry candidate using the raw unmodified bbox
  crop, returns the complete cosine vector, and permissive routing—not top-1
  classification alone—decides BLIP admission;
- BLIP3 uses the exact current generic fixed instruction after the delimited
  client question, and exact normalized result mapping selects configured
  `newcategory` or `falsecategory`; unmatched answers conservatively select
  `falsecategory`;
- canonical and legacy geometry diagnostics/schema names are accurate;
- debug identity claims name the test that actually crosses the raw processor
  seam; and
- Objective 021 limitations remain explicit.

At minimum reconcile `README.md`, `ARCHITECTURE.md`, `TESTING.md`,
`docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
`docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, and
`docs/SERVICE-DATASHEET.md`. Avoid duplicating a second conflicting narrative.

## Non-goals and preservation

- Do not change the accepted raw-crop pixel algorithm, BLIP3 compositor pixels,
  fixed generic instruction, SAM2 generator fields/caps/profiles, model identities,
  residency, renderer pixels, YOLO contract, response-size behavior, dependencies,
  service/network/auth/unit policy, or tracked example semantics except where a
  demonstrable validation/doc correction is necessary.
- Do not delete or weaken historical proof. Base floor remains at least 794
  meaningful passed tests plus the one honest opt-in GPU skip; this amendment
  must increase focused proof.
- Preserve all unrelated local/user files, including ignored presets; do not
  modify them in this round.

## Verification and report burden

Before reporting COMPLETE:

1. run the focused raw CLIP processor/debug identity, prompt, router, dry-run,
   geometry-schema, API JSON/ZIP and documentation tests;
2. run the complete default suite and coverage suite;
3. run format, lint, compile, documentation, diff-check, package/build, release
   manifest parity, twine, systemd verification, secret scans and both isolated
   install smokes required by repository law;
4. push the correction implementation and require all seven checks green on its
   exact SHA;
5. publish the immutable 020-b SELF report, push it, and require all seven checks
   green on the report head before FIFO signaling; and
6. report exact changed files, migration/backward-compatibility behavior, test
   commands/results, implementation/report SHAs, PR state, strongest remaining
   reason not to merge, and honest limitations. Do not claim live model accuracy.
