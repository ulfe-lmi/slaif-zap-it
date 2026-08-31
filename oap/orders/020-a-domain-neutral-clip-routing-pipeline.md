# OAP Work Order 020-a — Domain-neutral CLIP routing pipeline

## Objective

Implement the first dependency-complete half of the human-requested domain-neutral
candidate-pipeline redesign:

```text
preprocess
 -> request-local high-recall SAM2 proposals
 -> optional geometry impossibility filter
 -> untouched rectangular CLIP2 crops
 -> complete CLIP2 label-score vectors
 -> permissive deterministic routing
 -> one mask-derived contextual BLIP3 image
 -> request-authored semantic answer mapping
 -> final label filtering/order/render/serialization
```

SAM2 remains a proposal generator, CLIP2 becomes a routing-recall stage rather
than the final semantic authority, and BLIP3 owns exact semantic verification.
This order must deliver a coherent usable pipeline and migrate all shipped YAML
examples. Objective 021 will follow only after this PR is accepted and merged;
it will separate optional artifact delivery from inference success, add artifact
selection/pagination, enrich structured SAM2 capacity errors with admissible
alternatives, and close the exhaustive generated-documentation matrix. Do not
silently implement a partial version of those deferred resource semantics here.

This is one new numeric objective and one new PR. It intentionally supersedes
the CLIP candidate-view and top-1 flow from Objectives 017–019 while preserving
their immutable orders/reports and the accepted single-image BLIP3 compositor.

## Verified GitHub, OAP, and host state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Default branch: `main` at
  `cc325d5d97acefe7624aecfe9fa157dbf37ce600`, merge commit for Objective 019
  PR #75.
- Post-merge CI run `33382963462` and CodeQL run `33382963531` are successful.
- Open pull requests: none; local `main` equals `origin/main` and is clean.
- Current immutable selector is the merged `019-b` round.
- Create branch `oap/020-a-domain-neutral-clip-routing-pipeline` from the exact
  verified main SHA and one PR titled
  `Objective 020: domain-neutral CLIP routing pipeline`.
- Do not amend an old branch/PR, merge, or enable auto-merge.
- The authenticated private-LAN service is enabled, active/running and ready on
  host `hinton2`, PID `607106`, `NRestarts=0`, start timestamp
  `2026-08-31 12:36:38 CEST`, with one listener `10.8.132.76:17891`.
- The sole assigned GPU is physical index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. At publication the sole
  compute process is the service PID and uses approximately 12,010 MiB.
- `/dev/shm` has approximately 11 GiB free; `/dev/shm/slaif-zap-it` is mode 0700.
  The operator environment is mode 0600 with SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.

## Reconciled current behavior

Current main already has the important SAM2 request-local generator surface,
profiles, requested/effective/source metadata, prompt/prediction estimates,
candidate count, timing, warnings and fixed resident model. It also has the
accepted one-image `single_dilated_blur` BLIP3 compositor.

The contract gaps are material:

- CLIP receives `mask_dilated` zero-filled/dimmed mask views, not an untouched
  rectangular source crop.
- CLIP chooses one best prompt and returns only its label/score; it has no
  complete per-label score vector or independent routing policy.
- top-level label identifiers and natural-language prompt values are not cleanly
  separated in every legacy form.
- every post-geometry candidate flows into rule selection based primarily on the
  winning CLIP label; there is no top-k/margin/threshold/uncertainty router.
- geometry supports only three maximum aliases and rejection records omit the
  bbox and configured limit for area failures.
- BLIP3 does not serialize the configured/effective question, normalized answer,
  result mapping and final label for every verification.
- shipped examples contain legacy CLIP padding/`label ...` forms, very large
  synonym lists, operator-cap-exceeding SAM2 settings, and some trusted-only
  sections rejected by the service.

Do not weaken the already accepted BLIP3 pixel-composition invariants while
repairing these seams.

## Exact CLIP2 candidate-view contract

### New default and API surface

The effective default for `candidate_views.clip` becomes:

```yaml
candidate_views:
  clip:
    mode: raw_bbox_crop
    context_fraction: 0.10
    min_context_pixels: 0
    max_context_pixels: 64
```

Validate without coercing explicit values:

| Field | Accepted API value |
| --- | --- |
| `mode` | exact string `raw_bbox_crop` |
| `context_fraction` | finite non-boolean number `0.0..0.5` |
| `min_context_pixels` | strict integer `0..256` |
| `max_context_pixels` | strict integer `0..512`, not below the minimum |

API YAML must reject `mask_dilated`, `outside_fill`, `context_intensity`,
`contour_width`, unknown fields, nulls, bool-as-number, nonfinite values and
invalid bounds. The old masked CLIP mode may remain only as an explicitly named
trusted-CLI compatibility mode when a trusted config literally selects
`mode: mask_dilated`; it must never be a default, API mode, fallback or silent
translation. Current public code/docs/tests use `raw_bbox_crop`. Document the
legacy opt-in and report whether it was retained; do not delete the pure old
builder if doing so would break the constitutional legacy-CLI gate.

### Pure raw crop builder

Add a CLIP-specific pure builder separate from the BLIP3 compositor, with a name
such as `build_raw_clip_crop(image_rgb, mask, source_candidate_id, config)`.
It must not mutate inputs and must return an immutable/contiguous result plus
deterministic metadata.

For a non-empty source-shaped boolean SAM2 mask:

1. Compute its tight inclusive bbox `(x0,y0,x1,y1)`, width `W=x1-x0+1`, height
   `H=y1-y0+1`, and `L=max(W,H)`.
2. Compute `raw_radius = floor(context_fraction * L + 0.5)` (deterministic
   nonnegative half-up rounding; do not use Python banker rounding).
3. Compute `effective_radius = min(max(raw_radius,min_context_pixels),
   max_context_pixels)`.
4. Expand the inclusive bbox by the radius, clamp it to source boundaries, and
   represent the resulting array slice as an explicitly half-open
   `crop_bbox_xyxy_exclusive`.
5. Copy that complete rectangular RGB region directly from the original image.
   Every output pixel must equal the corresponding source pixel byte-for-byte.
6. Pass this raw RGB crop directly as `images=` to the fixed CLIP processor. Only
   the processor's standard fixed resize/normalization/tensor conversion may
   follow. No mask, zero/fill/background, alpha, blur, dimming, desaturation,
   overlay, contour, silhouette, cut-out, divider, duplication, second pane or
   second mask image may alter or accompany it.

At L3 with `clip.debug: true`, store the exact pre-normalization crop under
`clip-candidate-view-CANDIDATE-####.png`. Decoding that PNG must reproduce the
literal array passed as the processor `images=` input. Artifact names never use
labels, prompts, answers or client filenames.

Standardize CLIP input metadata to contain: stage, one-based stable
`source_candidate_id`, zero-based `filtered_index`, source dimensions, inclusive
mask bbox, half-open crop bbox, crop dimensions, pre-normalization model-input
dimensions, raw/effective context radius, artifact name/status when debug is
requested, and null/not-applicable dilation and contour fields. Preserve the
existing BLIP3 fixed artifact name and truthful stage-specific metadata; do not
reuse either stage's transformed image in the other.

## CLIP2 routing concepts and scoring

### Machine identifiers versus embedded language

The API canonical CLIP mapping is:

```yaml
clip:
  labels:
    target_candidate: a target object or a visually similar object
    foliage: plant foliage, leaves, and stems
    structure: structural material or equipment
    text_graphics: text, signs, or graphic markings
    background: background surface or scenery
```

- Keys are machine-safe routing identifiers matching
  `^[A-Za-z][A-Za-z0-9_-]{0,63}$`, unique by YAML mapping semantics, maximum 32.
- Values are non-empty natural-language strings, maximum 512 Unicode codepoints
  after ordinary YAML decoding. The complete value is one CLIP text prompt.
- Embed exactly the value. Never synthesize prose from the identifier, replace
  underscores, or embed the key in place of the value.
- Literal prompt text remains request data: never log it, use it in metric
  labels/artifact names, or echo it outside the effective sanitized config
  policy already documented.
- API YAML rejects `label "name"`, nested `clip.clip`, `clip.padding`, non-string
  values and unknown CLIP fields. Trusted CLI may retain legacy parsing only as
  explicit compatibility; report it and ensure the new API never selects it.

Do not implement semantic policing of whether prose is broad enough. Migrate
repository examples to short broad concepts and document that subtype/state/
quality/count decisions belong to BLIP3.

### Complete deterministic score vectors

For every candidate reaching CLIP:

- compute the similarity for every configured routing label;
- store an ordered mapping in configuration order under `clip_scores`;
- select the winning label by descending score with configuration order as the
  deterministic tie-breaker;
- keep compatibility `clip_label` and `clip_score` as the winner and score;
- if a trusted legacy label contains multiple prompts, aggregate that label by
  maximum prompt similarity, but the API canonical one-value/one-prompt form
  requires no aggregation;
- expose scores as finite cosine similarities in `[-1,1]`, not as undocumented
  softmax probabilities; and
- never return only the winner in the new diagnostics.

Use actual complete vectors in tests. Do not create correct-looking metadata
from values different from those used for routing.

### Request-local permissive router

Add top-level API-safe `clip_routing` to the algorithmic config surface:

```yaml
clip_routing:
  route_to_blip3:
    labels: [target_candidate]
    top_k: 2
    score_margin_from_best: 0.03
    minimum_target_score: null
    uncertain_labels: []
    max_candidates: null
```

Exact validation:

- `route_to_blip3` is the sole child; reject unknown fields.
- `labels`: required non-empty list of 1..32 unique identifiers present in
  `clip.labels`.
- `top_k`: null (disabled) or strict integer `1..number_of_clip_labels`.
- `score_margin_from_best`: null or finite non-boolean number `0.0..2.0`.
- `minimum_target_score`: null or finite non-boolean number `-1.0..1.0`.
- `uncertain_labels`: list of 0..32 unique identifiers present in `clip.labels`
  and disjoint from target labels.
- `max_candidates`: null or strict integer `1..256`.

For each candidate, choose the target routing label with the greatest score,
breaking ties by the order in `labels`. A candidate is initially routed when any
of these independently enabled/implicit conditions holds:

1. the chosen target is the winning/top-1 label;
2. the chosen target appears within configured `top_k`;
3. `best_score - chosen_target_score <= score_margin_from_best`;
4. `chosen_target_score >= minimum_target_score`; or
5. the winning label is in `uncertain_labels`.

This is OR logic. Null disables only that optional condition. Use exact inclusive
comparisons at equality. Record one deterministic primary reason with precedence:

```text
target_top_1
target_in_top_k
target_within_score_margin
target_exceeded_minimum_score
explicitly_uncertain
clear_negative
```

Also record all matched conditions so tuning evidence is not lost. When
`max_candidates` is set, rank initially routed candidates by descending chosen
target score then ascending source candidate ID; retain the first N. Remaining
plausible candidates get final reason `max_candidate_limit`, retain their
pre-limit reason/matched conditions, and never disappear silently.

In the service API, `clip_routing` is required whenever both `clip` and `blip3`
are configured, forbidden without either stage, and request-local. A trusted CLI
config without it may retain the exact previous rule-selection behavior as a
documented compatibility mode, but the service must not silently use that mode.
Two requests and A/B/A must prove no routing/config/vector/holder state leakage.

## BLIP3 rule selection, prompt, and result evidence

Keep the accepted `single_dilated_blur` image compositor and exact generic fixed
instruction. Do not add domain-specific fixed text or change client questions.
For the service's new route, the chosen target routing label selects the BLIP3
rule even when that target was top-k/margin/threshold plausible rather than the
CLIP winner.

The canonical service rule remains close to the existing YAML and adds an
explicit false label:

```yaml
blip3:
  target_candidate:
    question: Is the selected candidate the requested object?
    trueresult: Yes
    falseresult: No
    newcategory: requested_object
    falsecategory: negative
    debug: false
```

For every target routing label, require one same-named rule with non-empty
bounded string `question`, `trueresult`, `falseresult`, `newcategory`, and
`falsecategory`, plus optional strict boolean `debug`. Reject missing routing
rules, orphan rules, `any,<score>` rules, model/device/generation fields and
unknown children in API YAML. Trusted CLI may retain old `any` and implicit
`negative` behavior only as explicit documented compatibility.

Compose the effective question with the existing safe delimiters and exact
generic instruction. Preserve configured question bytes inside the delimiters.
Normalize answer and configured result tokens deterministically with Unicode
NFKC, surrounding whitespace removal, case folding, and removal only of terminal
ASCII `.`, `!`, `?`, `,`, `;`, `:` punctuation. Compare normalized values by
exact equality—never substring matching. A true match selects `newcategory`; a
false match selects `falsecategory`; an unmatched answer also selects the
request-configured `falsecategory` conservatively but is recorded as
`unmatched_answer`, not falsely described as an explicit No.

For every actual question, attach one bounded structured record containing:

- source candidate ID, filtered index and one-based question ID;
- routing target label and routing reason;
- `configured_question` and complete `effective_question`;
- raw answer and normalized answer;
- normalized true/false result tokens and configured true/false labels;
- mapping outcome `true_match|false_match|unmatched_answer`;
- input artifact name/status when requested; and
- final label.

The request already bounds question length/count and generated tokens; preserve
those limits. Prompt/answer records are response content only, never logs or
metrics. Final `ObjectRecord` at L2/L3 contains its complete CLIP score vector,
winner, routing decision/reason, and BLIP3 verification record(s). L3 additionally
contains stage-wide bounded candidate diagnostics for candidates pruned before
the final object list. Preserve `blip3_answer` and `clip_label` compatibility
fields as truthful last-answer/final-label aliases.

Exactly one contextualized image reaches each QA call. It may be composed once
and safely reused for multiple applicable questions in trusted legacy mode; the
new service route asks exactly its one selected target rule. The debug PNG must
remain decoded-pixel-identical to the QA image.

## Optional geometry impossibility filtering

Replace the three implicit huge-number limits at the normalized/API boundary
with optional canonical request fields under `postsam2processing`:

```yaml
postsam2processing:
  min_area: null
  max_area: null
  min_width: null
  max_width: null
  min_height: null
  max_height: null
  min_aspect_ratio: null
  max_aspect_ratio: null
  allow_border_touching: true
  debug: false
```

Validation is strict and noncoercing:

- area null or integer `0..64_000_000`;
- width/height null or integer `0..32768`;
- aspect ratio null or finite non-boolean number `0.0..1000.0`;
- minimum must not exceed corresponding maximum;
- `allow_border_touching` and `debug` are strict booleans;
- unknown fields are rejected.

Use inclusive bbox width/height and `width/height` aspect ratio. A candidate
touches the border when any inclusive bbox endpoint equals the corresponding
source boundary. Null disables a rule. Preserve trusted/API compatibility for
`maxsize`, `max_w`, and `max_h` as deprecated aliases for `max_area`,
`max_width`, and `max_height`; reject canonical/alias conflicts and expose a
migration warning. Do not silently invent limits.

Evaluate every non-empty candidate sufficiently to report its bbox even when an
area rule rejects it. Deterministic first-reason precedence is:

```text
empty_mask
min_area
max_area
min_width
max_width
min_height
max_height
min_aspect_ratio
max_aspect_ratio
border_touching
```

For every rejected candidate return at L3: stable one-based source candidate ID,
nullable inclusive bbox (`null` only for an empty mask), area, bbox width/height,
aspect ratio when defined, border-touching boolean when defined, fixed rejection
reason, configured limit field and value. Aggregate counts and bounded records
must reconcile exactly; preserve the existing 256-record cap and report the
truncated count rather than silently losing it. Empty SAM2 results must enter
this diagnostic path instead of disappearing before geometry evaluation.

Update the injectable filter boundary coherently; preserve compatibility wrappers
for trusted callers. Adapt old area-first tests explicitly and map every removed
or renamed test in the report. Do not merely delete the old regression matrix.

## Result, execution-order, schema, and timing contract

The engine's observable sequence must be:

```text
preprocessing
SAM2 proposal generation
geometry filtering
raw CLIP crop construction
CLIP scoring
permissive routing
single contextual BLIP3 view construction
BLIP3 verification
final label filtering
ordering
visualization
serialization
```

Keep the CLIP and BLIP builders separate. Add/retain candidate counts sufficient
to identify loss at each stage: raw SAM2 generated, non-empty/geometry evaluated,
after geometry, CLIP scored, initially routed, routed after deterministic cap,
BLIP3 verified, after final label filter, and final objects. Preserve existing
count aliases where compatible and document them; never relabel old values
dishonestly.

Add explicit stage timings for geometry, CLIP crop/scoring, CLIP routing, BLIP3
composition/verification, final filtering, ordering, visualization and existing
SAM2/preprocessing. Fakes may supply deterministic zero/fixture timings; runtime
timings remain nonnegative and are L3 evidence, not accuracy metrics.

At L3 add a typed bounded `clip_routing_diagnostics` collection for every
post-geometry candidate with full ordered score vector, winner, chosen target,
rank, target score, best-score delta, route boolean, matched conditions, primary
reason, cap outcome, IDs and crop metadata. At L2/L3 every final object carries
the equivalent relevant fields. Geometry-rejected candidates remain in
post-filter diagnostics. BLIP3 records include complete question/answer mapping.
JSON and ZIP manifest must agree exactly.

The authenticated capabilities endpoint must expose the new CLIP view, CLIP
label, routing, geometry and BLIP rule fields with type/default/range/units,
execution stage and interaction notes. Pydantic/OpenAPI response schemas must
type all new records and closed reason enums. Preserve L0/L1 boundedness: complete
candidate diagnostics remain L3, while effective candidate-view/routing policy
may remain in the always-present service metadata. L2 final objects must carry
their own semantic evidence.

SAM2 response metadata in this PR must retain requested/effective/source values,
profile, estimates, actual count, time and warnings. Add explicit operator-limit
metadata for every capped field and a per-field provenance object containing
`source`, `operator_limit` (nullable) and `operator_limit_applied: false` for
accepted requests. Never clamp an excessive request; an operator limit causes
`resource_limit`, so no accepted effective value may falsely claim a clamp.
Objective 021 owns detailed causing-field/admissible-alternative error bodies.

## Shipped configuration migration

Migrate every `configs/*.yaml` file, not just one showcase:

- every file must pass the exact hostile parser with deployed default
  `ServiceSettings` and operator caps at verbosity 0 and 3;
- use accepted SAM2 profile/explicit settings; current 64x/crop-2 files and the
  soccer 160x/crop-4 example exceed deployed estimate/field caps and must be
  changed honestly, never silently clamped;
- replace padding/legacy label syntax and huge synonym lists with bounded natural
  routing phrases and concrete broad negatives;
- add explicit `raw_bbox_crop`, `clip_routing`, matching BLIP3 rules and explicit
  answer-to-label mappings where BLIP3 is used;
- use domain-specific routing distinctions only when they genuinely select
  different BLIP3 questions/final labels;
- convert `geometry` and `blip2` to explicitly ignored trusted/batch-only
  sections or remove/migrate them; never pretend the service executes them;
- keep client-authored task questions substantively intact where still
  applicable, while removing model/device/path controls forbidden by API;
- update final visualization label whitelists/class mappings to terminal BLIP3
  labels; and
- add one parameterized CI test enumerating every shipped YAML example and
  applying the same parser/caps as the deployed service. A new unvalidated
  example must make CI fail.

The migration notes must state all compatibility breaks and the old-to-new YAML
shape. Do not claim these general examples establish semantic accuracy.

## Required deterministic tests

Use generated arrays, fakes and literal processor/QA captures. No external image,
image generation, model download, network or CUDA is needed.

### CLIP crop and processor seam

- every crop pixel equals its exact source coordinate;
- independent half-up radius oracle, min/max clamping and inclusive/half-open
  bbox arithmetic;
- every source edge/corner and odd/even bbox size;
- no mask/fill/black background/alpha/blur/dim/overlay/contour/silhouette/
  duplication/divider;
- nonrectangular, holed and disconnected masks influence only the tight bbox;
- input arrays/config are not mutated; repeat bytes are deterministic;
- real `classify_single` with fake fixed processor proves literal `images=` is
  the builder crop; decoded debug PNG is identical; and
- API rejects masked/legacy modes while trusted explicit legacy behavior, if
  retained, never becomes a default.

### Labels, scores, and routing

- exact natural values—not identifiers or underscore substitutions—reach the
  text processor;
- every label receives a finite score in config order and winner ties are stable;
- top-1, top-k, exact equality at top-k/margin/minimum thresholds, uncertain,
  clear-negative and combined-condition paths;
- deterministic maximum-candidate ranking and explicit capped-out reason;
- multiple target labels choose the correct BLIP rule even when a negative is
  top-1;
- all score vectors/reasons survive objects and L3 JSON/ZIP diagnostics;
- request A/B/A changes prompts/routing without reinitializing CLIP or leaking
  scores/state; and
- invalid/missing/orphan routing/label/rule configurations produce precise
  structured errors.

### Geometry and stable identity

- each optional min/max area/width/height/aspect/border rule, equality retention,
  precedence and disabled-null behavior;
- bbox/area/limit facts for every rejection, including empty nullable bbox;
- border/corner/disconnected candidates without drift;
- legacy aliases and canonical conflicts;
- more than 256 rejections reconcile truncation;
- one-based original source ID and zero-based filtered index remain stable across
  geometry, CLIP, routing, BLIP3, final filtering, ordering, labelled
  visualization, objects and JSON/ZIP.

### BLIP3 evidence

- accepted Objective-019 one-image dilation/source/blur/contour/crop matrix stays
  green;
- route-selected rule receives exactly one contextual image and exact configured
  question plus generic instruction;
- normalization/exact match for true, false and unmatched answers;
- complete effective prompt, raw/normalized answer, request mapping and final
  label are serialized;
- no hidden domain wording or duplicated semantic question;
- debug image is the literal QA image and names remain fixed; and
- CLIP raw crop and BLIP contextual image are provably different builders and
  arrays.

### API/config/docs/regression

- L0–L3 policy/level gating and typed schemas;
- all shipped configs validate at verbosity 0/3 with deployed caps;
- requested/effective SAM2 metadata and operator-limit provenance;
- candidate counts and timings for every stage;
- invalid types, bool-as-number, nonfinite, unknown and range/inversion errors;
- JSON/ZIP parity, artifact hash/size identity where artifacts fit;
- existing CLI, renderer, SAM2, service/auth/privacy/residency and artifact-limit
  regressions remain green; and
- no meaningful base test is deleted. The corrected base is 794 passed plus one
  honest opt-in GPU skip; final suite must have at least 794 meaningful passes
  and map every removed/renamed affected test in the report.

## Documentation and migration

Update current architecture and user/operator documents in the same PR,
including `ARCHITECTURE.md`, `README.md`, `TESTING.md`, `docs/ALGORITHMS.md`,
`docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`,
`docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`, shipped config comments, and any
other current file found by repository-wide stale-contract search.

Document the responsibility split, exact raw-crop/radius/bbox semantics, prompt
value behavior, cosine score units, routing OR logic/reason precedence/cap
ranking, geometry fields/precedence/records, BLIP prompt and exact mapping,
execution order, IDs, counts/timings, capabilities/schema fields, verbosity
gating and trusted legacy behavior. Include explicit migration examples for old
masked CLIP views, padding/`label ...`, top-1/`any,<score>` BLIP rules, implicit
negative mapping and old geometry aliases.

State clearly that semantic accuracy/recall is not proved by deterministic CPU
tests. Also state that Objective 021 will change the currently inference-fatal
optional artifact overflow into structured truncation; do not falsely document
that behavior as completed in this PR.

Do not edit immutable historical OAP orders/reports.

## Scope and non-goals

Expected product paths include `modules/classifier/clip.py`,
`modules/verifier/blip3.py`, `src/postprocessing.py`, `src/core/config.py`,
`src/core/mask_views.py`, `src/core/engine.py`, `src/core/results.py`,
`src/service/yaml_input.py`, `src/service/schemas.py`,
`src/service/capabilities.py`, `src/service/envelope.py`, `src/batch.py`, all
`configs/*.yaml`, focused tests/current docs, `oap/active`, this order and the
final report. Exact smaller coherent paths are allowed with report explanation.

Do not change model identities/revisions/weights, devices, dtype, caches,
residency, generation limits, auth/network/unit/key, service port, concurrency,
dependencies/lockfiles, SAM2 generator implementation, BLIP3 pixel compositor,
visual renderer pixels, YOLO format, release state, Git history or
`CRITICAL.md`.

Do not implement artifact truncation/pagination/download or detailed SAM2
resource alternatives here; Objective 021 owns them. Preserve current limits
and truthful errors until then. Do not restart/reconfigure or send inference to
the live service during coding. Do not use image generation; exact synthetic
arrays are the stronger test fixture.

## Verification and evidence

Coding must run and report:

- focused raw-crop, CLIP processor/vector/routing, geometry, BLIP prompt/mapping,
  source-ID, shipped-config, API/schema/capability, JSON/ZIP and docs tests;
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`;
- Ruff format/check, compileall, documentation integrity and `git diff --check`;
- wheel/sdist build, direct and sdist-built wheel verification/comparison,
  outside-checkout installed smoke, archive/tracked-tree secret scans,
  `twine check`, and systemd-unit verification;
- all seven implementation-head CI/CodeQL checks; and
- all seven final report-head CI/CodeQL checks.

Record exact counts, coverage, warnings and the one honest GPU skip. No GPU/model
download/live inference is authorized. Preserve the running service and perform
read-only before/after health/readiness/listener/PID/GPU/shm/env-digest checks.

The strongest reason not to merge is that a permissive-looking routing manifest
could be fabricated while a masked crop or only the top-1 prompt actually drives
the model, or plausible candidates could disappear between routing and BLIP3.
Answer it with independent source-coordinate crop oracles, literal CLIP processor
capture, complete-vector fake logits, branch-complete router tests, exact BLIP QA
capture, all-candidate loss records and end-to-end stable-ID JSON/ZIP parity.

## Live service preservation

This order does not authorize stopping, restarting, reloading, reconfiguring or
sending inference to `zap-it-lan.service`. Read-only checks must preserve the
verified service/GPU/listener/shm/environment facts above. Never print, log,
copy, rotate or mutate the API key. Do not touch firewall/VPN/network, port,
unit, driver/CUDA, GPU state, another process/device/service, or global
credentials. No raw image, YAML, prompt, answer, label, artifact or path belongs
in logs/metrics/OAP evidence.

## Acceptance criteria

1. The API's CLIP processor always receives the complete unmodified
   `raw_bbox_crop`; masked modes are never silent/API-selected.
2. Natural prompt values are embedded, every per-label score is returned, and
   the router implements deterministic top-1/top-k/margin/minimum/uncertainty/
   maximum-count behavior with exact reasons.
3. Every geometry rejection is optional/configured, carries source ID, bbox,
   area, rule and limit, and no candidate disappears silently.
4. Every routed candidate selects the correct request-authored BLIP3 rule and
   receives one accepted contextual image; complete effective question,
   answer, mapping and final-label evidence is returned.
5. Stable IDs, candidate counts and timings identify the exact stage/reason for
   every loss through final serialization.
6. CLIP and BLIP image constructors remain separate and their debug PNGs equal
   literal pre-normalization model inputs.
7. SAM2 request/effective/source/estimate/count/time/warning behavior remains
   request-local and gains truthful operator-limit provenance without clamping.
8. Every shipped YAML config passes deployed service validation/caps and uses
   the documented new routing contract.
9. Current schema/capabilities/docs match code; migration and retained trusted
   compatibility behavior are explicit.
10. Full CPU/package/static/docs checks and both implementation/report-head CI
    matrices are green, with no deleted proof or live-service mutation.

## Deferred human adjudication

- Decision: NONE

## Publication and report contract

- Publish this exact order and selector `020-a` in the implementation/control
  commit and record its SHA-256.
- Push all non-report work to the exact new branch and create exactly one PR.
  Never merge or enable auto-merge.
- After implementation-head checks are green, publish exactly one immutable
  `oap/reports/020-a-report.md` as a report-only commit with literal
  `Report publication commit: SELF`; its sole parent is the implementation SHA
  and its sole changed path is the report.
- Push and independently verify remote head/topology/bytes, then require all
  report-head checks green before signaling the response FIFO.
- The report must list changed files, exact configuration migration, retained
  compatibility, score/routing/geometry/prompt contracts, shipped-config
  validation, commands/results/coverage/CI, service preservation, limitations,
  strongest reason not to merge and critical action `NONE`.
- COMPLETE is allowed only when every criterion is met. Otherwise publish a
  truthful PARTIAL report without weakening tests or mutating after SELF.
- Coding must not merge, release, deploy, restart/reconfigure the service, run
  live inference, mutate host/GPU/network/credentials, or print the key.
