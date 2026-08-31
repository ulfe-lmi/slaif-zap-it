# OAP Work Order 022-a — Canonical CLIP multi-prompt classes and exact live proof

## Objective

Correct the canonical public API so one machine-safe CLIP semantic class may own
one prompt string or an ordered non-empty array of independent prompt strings.
Every array item is encoded separately; per-candidate prompt similarities are
aggregated to one score per semantic class by deterministic maximum; routing
continues to operate only on those semantic-class scores. Reject invalid or
overlong prompts as structured HTTP 400 `invalid_config` before model inference,
never as HTTP 500. Synchronize schema, capabilities and maintained documentation,
including the fact that `falsecategory` is required for canonical routed BLIP3
rules. Qualify the exact human-supplied 97-prompt tomato request on the private-LAN
service and leave the corrected service enabled, active and ready.

This is Objective 022 and creates exactly one new PR. Do not merge or enable
auto-merge; strategic owns review and merge.

## Reconciled starting state

- Remote repository is `ulfe-lmi/slaif-zap-it`; default branch `main` is exactly
  `d341a3c4ba47b71d10d70682771b315041dcbcb8`, the merged PR #77 head. Post-merge
  CI run `33414404088` and CodeQL run `33414403999` are successful. There are no
  open PRs. `main` is not protected by GitHub branch-protection settings, so the
  order must enforce the full seven-check merge discipline itself.
- Local coding checkout is clean on `main` at that same SHA. Active selector is
  merged round `021-b`. Publish this exact order and set `oap/active` to `022-a`
  in the implementation history.
- Confirmed defect in `modules/classifier/clip.py`: canonical scalar values are
  intentionally represented as a one-element list; canonical public validation
  accepts only a scalar string; the lower-level class-map parser happens to
  accept lists but public requests cannot reach it. A comma/newline-packed
  506-character value therefore remains one text sequence. `_encode_text_prompts`
  invokes the CLIP processor without an explicit tokenizer-length preflight or
  bounded defensive truncation. The pinned OpenAI CLIP ViT-B/32 text context is
  77 tokens, so an excessive prompt can escape request validation and become the
  generic HTTP 500 `inference_failure` mapping.
- `classify_single_scores` already computes a maximum prompt similarity for each
  class, but exposes only the winning prompt for the overall winning class. The
  public L3 contract lacks effective per-class prompt counts and per-class
  winning-prompt indices.
- Canonical routed BLIP3 validation requires non-empty `trueresult`,
  `falseresult`, `newcategory` and `falsecategory`, while the capabilities field
  catalog does not mark `falsecategory` required. Runtime behavior is the chosen
  authority: advertise all four mapping fields as required for a canonical
  routed rule; do not make `falsecategory` optional.
- The repository fixture
  `demos/tomato/2022-07-22-16-25-44-48.jpg` is a baseline 1280x720 JPEG, 358454
  bytes, with verified SHA-256
  `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.
- Host is `hinton2`. Existing `zap-it-lan.service` is enabled, active and ready,
  PID `672344`, `NRestarts=0`, started `2026-08-31 18:35:31 CEST`, and listens
  only at `10.8.132.76:17891`. It runs the merged `d341a3c4...` working tree and
  must remain unchanged throughout ordinary implementation and CI.
- The active operator-assigned device is physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB. Driver is `610.43.02`; the service environment has
  PyTorch `2.5.1+cu124` and CUDA `12.4`. The existing service is the sole compute
  process. At reconciliation it used about 10868 MiB with about 13233 MiB free.
- `/dev/shm` is a 12-GiB tmpfs with about 11 GiB free;
  `/dev/shm/slaif-zap-it` is mode 0700 and empty. The environment file is mode
  0600 and has SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  Never print, copy, rotate, replace or commit its credential value.

## Canonical label grammar and normalization

Under the public canonical `clip.labels` mapping, each safe class identifier may
map to exactly one of:

1. one string, which is exactly one prompt; or
2. one ordered non-empty YAML sequence of strings, each item an independent
   prompt associated with that same semantic class.

Canonical scalar strings are never split. Commas and internal newlines remain
literal content in the single prompt. Arrays are never joined or concatenated.
Trim only leading/trailing Unicode whitespace from each canonical prompt before
duplicate detection, token validation, hashing/effective-config serialization
and embedding; preserve all internal text, punctuation, commas and newlines.
Expose/serialize the same normalized prompt bytes that are encoded so validation
and inference cannot disagree.

Preserve YAML insertion order for classes and list order for prompts. Reject
duplicates within one class after trimming; do not silently deduplicate. The
error must identify the duplicate prompt index and the first equal prompt index,
without echoing prompt text. The same text may appear in different semantic
classes because those classes intentionally have independent routing meaning.

The canonical public limits are:

- 1..32 semantic classes (existing bound);
- 1..64 prompts per class;
- 1..256 prompts total across all classes;
- every prompt must be a string;
- non-empty after trimming;
- at most 512 Unicode codepoints after trimming; and
- at most 77 tokens under the exact pinned
  `openai/clip-vit-base-patch32` tokenizer, including its special tokens.

Strictly reject wrong container/item types, booleans, nulls, empty sequences,
empty/whitespace-only strings, excessive class/total counts, excessive
characters, excessive tokens and duplicates. Do not coerce non-strings. The
first invalid prompt in deterministic class/list order owns the error.

Legacy trusted CLI behavior is a separate compatibility boundary: existing
flattened `label <name>` keys and comma/newline splitting of legacy scalar prompt
lists may remain exactly as found. They must never be selected silently by the
canonical HTTP API or weaken canonical validation.

## Exact tokenizer-aware rejection before inference

Separate cheap structural normalization from model-aware token validation, but
make the deployed service execute both before SAM2 or any CLIP image/text model
inference:

1. `parse_hostile_config` owns type, emptiness, class/per-class/total count,
   character, duplicate and normalization validation and returns normalized
   canonical labels plus deterministic prompt-count metadata.
2. Wire a production request-preflight validator to the already resident CLIP
   processor/tokenizer. It must count every normalized prompt with the exact
   pinned tokenizer, with special tokens, against the fixed limit 77. It must run
   after readiness is established and as the first serialized/gated model-aware
   request operation, before calling the core runner/SAM2. It must not load or
   reload model weights, mutate the tokenizer/model, access the network or create
   a second resident holder. Tests may inject a deterministic tokenizer double.
3. Define a narrow typed prompt-validation exception/details seam. The live
   adapter/service maps only prompt-input validation failures to HTTP 400
   `invalid_config`; unrelated tokenizer/model corruption remains an honest
   internal/inference failure. No recognized prompt type/count/character/token/
   duplicate violation may reach generic HTTP 500.
4. In `_ClipFilter._encode_text_prompts`, repeat exact length validation as
   defense in depth immediately before processor/model use. Then pass explicit
   bounded tokenizer arguments (`max_length=77` and defensive truncation only
   after proof that every prompt already fits), or an equivalently fail-closed
   bounded call. Assert/test that accepted prompt IDs are unchanged and never
   silently truncated. Convert any detected excessive input to the same typed
   prompt-validation error.
5. At startup or first safe validation, verify the pinned tokenizer/model text
   context agrees with 77. A mismatched/corrupt operator asset is a readiness or
   internal model problem, not a client `invalid_config`; do not silently adopt a
   different limit.

Extend the typed error schema with bounded sanitized CLIP prompt details. For an
offending item, include `class_identifier`, zero-based `prompt_index`, a stable
reason, relevant measured count (`measured_character_count`,
`measured_token_count`, per-class count or total count), and `allowed_limit`.
Duplicate errors additionally identify `first_prompt_index`; type errors identify
the safe actual type name. A total-count overflow identifies the first item that
crosses 256. Never echo prompt text, raw YAML, tokenizer IDs, filesystem/model
paths or internals. Add OpenAPI/capabilities documentation for these 400 details.

## Scoring, aggregation and routing evidence

Flatten normalized prompts only for batch text encoding while retaining an
immutable `(class_identifier, zero-based prompt_index)` mapping. For every image
candidate:

- compute cosine similarity against every individual prompt;
- compute each semantic-class score as the maximum of that class's prompt
  similarities;
- on an equal score, choose the lowest prompt index deterministically;
- return exactly one finite score per semantic class, in configured class order;
- choose the overall winning class using existing deterministic configured-order
  tie-breaking;
- route only with the semantic-class score vector, never synthetic prompt IDs;
  and
- retain the overall winning prompt text/index and every class's winning prompt
  index as diagnostic evidence.

Preserve the historical public `classify_single` result and trusted callers.
Introduce a bounded detailed result/helper or compatibility wrapper rather than
silently breaking the existing tuple API. Do not place class identifiers or
prompt text in artifact filenames. Stable candidate IDs, raw CLIP crop identity,
BLIP3 selection and existing routing-reason semantics must remain unchanged.

At verbosity 3 expose a typed, bounded `service.clip_prompts` (exact name may be
adjusted only if an existing convention clearly requires it) containing class
prompt counts in configured order, total effective prompt count, tokenizer limit
77 and duplicate policy `reject`. Each L3 CLIP routing diagnostic must include
the winning prompt index for every class score when scores exist, plus the overall
winning class's prompt index and normalized prompt text. The existing complete
per-class `clip_scores` vector remains canonical. At verbosity below 3 do not add
this diagnostic payload solely for the change; normal object/score compatibility
remains as documented.

Dry-run/fake behavior must produce deterministic correctly shaped prompt counts,
class score vectors and prompt-index evidence without claiming semantic model
accuracy.

## Schema, capabilities and documentation

Update the canonical field inventory, Pydantic response/error schemas,
capabilities response and generated application OpenAPI so
`clip.labels.<identifier>` visibly advertises the union:

```text
string | array[string]
```

Document scalar indivisibility, array-item embedding, normalization, 64-per-class
and 256-total bounds, 512-character and exact 77-token bounds, duplicate rejection,
per-class maximum aggregation, deterministic ties, prompt diagnostics, routing
semantics and structured error details. Keep the canonical accepted-leaf inventory
exactly synchronized with capabilities; the dynamic leaf path itself does not
multiply per array item.

Mark `question`, `trueresult`, `falseresult`, `newcategory` and `falsecategory`
required for every canonical BLIP3 rule targeted by `clip_routing`. Where the
capabilities catalog cannot express conditional requirement in one boolean,
mark the canonical dynamic fields required and state the condition explicitly.
Retain and document any trusted legacy non-routed behavior separately.

Perform a detailed stale-claim refresh of `README.md`, `ARCHITECTURE.md`,
`TESTING.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/ALGORITHMS.md`,
`docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md` and
`docs/SERVICE-DATASHEET.md`. Change only documents actually affected, but search
all maintained documents for one-prompt-only claims, comma-splitting ambiguity,
prompt-score routing, missing token-limit/error behavior and nullable/optional
`falsecategory`. Generated schemas and live behavior are the final contract; add
tests that prevent those sources from drifting.

## Required deterministic automated tests

Use self-contained CPU arrays and fake/tokenizer doubles; public CI must not
download models or require CUDA.

Add tests proving all of the following:

1. A canonical class accepts one scalar prompt or an ordered list of 1, 2 and 64
   prompt strings; two classes can reuse equal prompt text.
2. Commas and internal newlines in one canonical scalar remain one exact prompt.
3. A list yields separate processor text inputs in exact order; it is never
   concatenated.
4. Prompt-to-class/index mapping is exact; class score is the maximum prompt
   score; equal-score ties retain the lowest prompt index; one score per semantic
   class is returned.
5. Routing top-1/top-k/margin/minimum/uncertain logic consumes semantic-class
   scores only and never a prompt identifier. Stable source candidate IDs survive.
6. L3 prompt summary reports exact per-class and total counts; routing diagnostics
   report class winning indices and the overall winning prompt/index; verbosity
   gating and JSON/ZIP parity are exact and bounded.
7. Empty list, >64 prompts/class, >256 total, non-string item, whitespace-only
   item, >512 characters and trimmed duplicate all return `invalid_config` with
   the ordered sanitized detail fields. Boundary equality passes.
8. A tokenizer double returning 77 tokens passes unchanged; 78 returns HTTP 400
   `invalid_config` before engine/SAM2 counters change. The error has class,
   zero-based item index, measured 78 and limit 77. The same excessive input at
   the model-adapter defense seam raises the typed client-input error, not a
   positional-embedding/internal exception, and no silent truncation occurs.
9. A processor/model double observes every array item as an independent prompt,
   bounded tokenizer options and unchanged accepted token IDs. Resident A/B/A
   label updates with different prompt-list lengths have no state leakage or
   model-weight reload.
10. Scalar canonical compatibility, trusted legacy scalar splitting, historical
    `classify_single`, raw bbox crops, complete class vectors, routing, single
    BLIP3 image, labelled renderer, artifact limits and SAM2 request-local tests
    remain green.
11. Capabilities and generated OpenAPI visibly encode string-or-array,
    per-class/total/token limits, L3 prompt evidence, typed prompt errors and
    required canonical `falsecategory`; runtime/schema/capabilities inventories
    agree.
12. The exact 97-prompt configuration in the appendix parses and reaches the
    fake engine successfully. Assert prompt counts 32/15/15/20/15 and total 97,
    exactly five class-score keys and routed target `ripe_tomato`.

## CPU/static/package/CI gate before service mutation

Keep the current PID active and ready while implementing. Run and report:

- focused classifier, canonical routing, hostile validator, API, envelope,
  schema, capabilities/OpenAPI, live-runtime adapter and exact-config tests;
- all existing core/service/SAM2/candidate-view/BLIP3/renderer/artifact tests;
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing` with the
  existing coverage threshold;
- Ruff format/check, compileall, documentation checker and `git diff --check`;
- wheel/sdist build, release-artifact verification and wheel parity, tracked-tree
  and archive secret scans, `twine check`, systemd verification, and isolated
  direct-wheel and sdist-built-wheel smoke tests; and
- all seven current-head GitHub CI/CodeQL checks, present and successful on the
  exact implementation/control head before any restart or live inference.

No new dependency is expected or authorized. The exact tokenizer is already an
operator-owned component of the pinned GPU runtime; CPU tests use injected
doubles. If implementation proves a dependency change unavoidable, stop and
report rather than modifying it.

## Exact bounded live private-LAN qualification

Only after every CPU/static/package/current-head CI gate above succeeds, this
order authorizes exactly one controlled restart of only `zap-it-lan.service` to
load the 022-a implementation. Before restart independently reverify exact
physical index/UUID/PCI/name/VRAM/process ownership, driver/CUDA/PyTorch,
`/dev/shm`, one listener/unit/port, environment mode/digest and capacity without
printing or copying the key. Do not alter the unit, environment, credential,
deadline, network, firewall, VPN, driver, model identities/revisions/caches or
residency strategy. Start no second model process.

Wait through the normal bounded cold load without speculative restart. After
readiness returns, require health/readiness 200, missing/wrong inference
credentials 401, authenticated capabilities/metrics 200 and private docs/OpenAPI
404. Verify live capabilities contain the exact new prompt and `falsecategory`
contract without sensitive disclosure.

Use the existing inference bearer only ephemerally in local process memory with
shell tracing disabled. Never print it, put it in arguments visible to process
list, copy it to another file, include it in output/report/logs or retain it after
the requests. The human suggested `/tmp`, but `/tmp` is ext4 on this host and the
product constitution requires RAM-backed request data. Therefore write the exact
appendix YAML and response only under a unique mode-0700 directory beneath
`/dev/shm` (outside the service's own workspace), with mode-0600 files. This is
the sole deliberate path adaptation; request YAML bytes and algorithm values must
otherwise be exact. Leave the final ZIP and extracted labelled PNG in that tmpfs
directory through strategic review so it can be visually inspected and linked
to the human; clean intermediate files and never commit them.

Before the accepted request, send the same authenticated request with exactly one
positive prompt replaced by a deterministic synthetic >77-token value. Require
HTTP 400 `invalid_config`, exact sanitized class/index/token/limit details, no
SAM2/CLIP/BLIP counters or GPU peak attributable to inference, stable PID and no
500. Do not put that synthetic prompt in the report.

Then submit exactly one verbosity-3 ZIP request using the verified repository
image and appendix configuration, adapted only to
`http://10.8.132.76:17891/v1/completions` and the protected credential source.
Require HTTP 200, no `inference_failure`, and:

- `service.clip_prompts` counts: `ripe_tomato=32`, `foliage=15`,
  `stem_or_vine=15`, `greenhouse_structure=20`, `background=15`, total `97`;
- every candidate's complete CLIP vector has exactly those five semantic-class
  keys, never prompt IDs;
- routing target is `ripe_tomato`, and only candidates whose chosen routing
  target is `ripe_tomato` reach that BLIP3 rule;
- every final object is labelled `ripe_tomato`, has inclusive bbox width and
  height each from 11 through 199 pixels, and retains its stable source ID;
- the ZIP manifest/member hashes and sizes agree and it contains the exact
  `final-labelled-ripe-tomatoes` annotated-labelled PNG; no prompt appears in a
  filename; and
- final image/debug selection and artifact ledger remain within existing bounds.

Record sanitized stage counts and timings: raw SAM2 proposals, geometry retained/
rejected by rule, CLIP scored, initially routed, routed after cap, BLIP3 submitted,
BLIP3 accepted/rejected, final retained, and BLIP3 candidate-view containment
rejections. State exact HTTP/status/ZIP/PNG sizes and hashes, stable PID/listener,
peak/current GPU memory, RSS, workspace cleanup and journal result without raw
answers, prompts, image bytes or credentials.

Visually inspect the source and extracted final labelled PNG. Report the number
of prominent ripe tomatoes visible to a human and, separately, obvious missed
ripe tomatoes, false-positive final masks, fragmented final masks and merged
multi-object final masks. This is an observational result, not a semantic model
accuracy guarantee. The request returning 200 alone is not acceptance. Preserve
the labelled PNG for strategic independent inspection and report its tmpfs path,
SHA-256, dimensions and byte size without committing it.

Recheck health/readiness/auth, exact one listener, sole assigned-GPU service
process, `NRestarts=0` for the new PID, unchanged environment digest/mode,
sanitized recent journal and empty `/dev/shm/slaif-zap-it`. Leave the corrected
service enabled, active and ready at `10.8.132.76:17891`.

If startup or the exact request fails, do not silently alter the config, extend
the deadline, weaken validation, repeat inference or spend a second restart.
Leave the service in the safest reachable ready state and publish an honest
PARTIAL/FAILED report with the exact sanitized failure. Strategic will decide a
same-PR 022-b correction. One recognized prompt violation returning 500, one
config mutation, or missing visual/count evidence blocks completion.

## Exact live configuration appendix

Use these YAML bytes, apart from the final newline:

```yaml
preprocessing:
  roi: false
  resize: 1.0
  debug: false

mask_generator:
  points_per_side: 32
  points_per_batch: 8
  pred_iou_thresh: 0.70
  stability_score_thresh: 0.70
  min_mask_region_area: 100
  crop_n_layers: 2
  crop_n_points_downscale_factor: 2
  crop_overlap_ratio: 0.40
  box_nms_thresh: 0.30
  multimask_output: true
  debug: false

postsam2processing:
  min_width: 11
  max_width: 199
  min_height: 11
  max_height: 199
  allow_border_touching: true
  debug: true

candidate_views:
  clip:
    mode: raw_bbox_crop
    context_fraction: 0.10
    min_context_pixels: 0
    max_context_pixels: 64
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

clip:
  labels:
    ripe_tomato:
      - tomato
      - ripe tomato
      - red tomato
      - ripe red tomato
      - mature tomato
      - tomato fruit
      - ripe tomato fruit
      - red tomato fruit
      - tomato on vine
      - vine tomato
      - hanging tomato
      - tomato on stem
      - tomato on plant
      - greenhouse tomato
      - plum tomato
      - ripe plum tomato
      - red plum tomato
      - oval tomato
      - oblong tomato
      - elongated tomato
      - glossy tomato
      - smooth tomato
      - bright red tomato
      - dark red tomato
      - shaded tomato
      - hidden tomato
      - occluded tomato
      - single tomato
      - individual tomato
      - harvest tomato
      - fully ripe tomato
      - mature red fruit

    foliage:
      - plant leaf
      - green leaf
      - dark green leaf
      - tomato leaf
      - plant foliage
      - dense foliage
      - leafy canopy
      - leaf cluster
      - overlapping leaves
      - leaf underside
      - leaf edge
      - backlit leaf
      - shadowed leaf
      - green vegetation
      - leafy branch

    stem_or_vine:
      - plant stem
      - green stem
      - brown stem
      - thick stem
      - thin stem
      - cut stem
      - plant stalk
      - tomato vine
      - vine segment
      - plant branch
      - stem junction
      - leaf petiole
      - bare fruit stem
      - diagonal vine
      - woody stem

    greenhouse_structure:
      - metal pipe
      - gray pipe
      - horizontal pipe
      - support rail
      - greenhouse frame
      - metal frame
      - support post
      - structural beam
      - trellis wire
      - support wire
      - hanging cable
      - support string
      - red rope
      - irrigation tube
      - black hose
      - greenhouse bench
      - growing channel
      - metal bracket
      - support clip
      - crop container

    background:
      - plant label
      - white printed tag
      - plastic grow bag
      - greenhouse floor
      - concrete floor
      - greenhouse walkway
      - greenhouse wall
      - greenhouse glass
      - glass reflection
      - bright window
      - sky through glass
      - blurred crop row
      - distant equipment
      - dark shadow
      - empty background

  debug: false

clip_routing:
  route_to_blip3:
    labels: [ripe_tomato]
    top_k: 2
    score_margin_from_best: 0.04

blip3:
  ripe_tomato:
    question: |
      Does the selected region represent one individual ripe tomato fruit with
      predominantly red, mature-looking skin? Accept a partially occluded fruit
      if it is clearly one tomato. Reject unripe or partly ripened fruit,
      foliage, stems, equipment, background, ambiguous fragments, and masks
      containing more than one tomato.
    trueresult: |-
      yes
    falseresult: |-
      no
    newcategory: ripe_tomato
    falsecategory: negative
    debug: false

visualization:
  labels: [ripe_tomato]
  alpha: 0.55
  blip3:
    - id: final-labelled-ripe-tomatoes
      renderer: annotated-labelled
      alpha: 0.55
      show_confidence: true

diagnostic_artifacts:
  stages: [visualization]
  page: 1
  page_size: 48
```

## Non-goals and protected boundaries

- No CLIP model/revision/tokenizer replacement, prompt-quality tuning, score
  calibration, semantic-threshold change or claim that 97 prompts improve
  accuracy. Implement transport/validation/aggregation evidence only.
- No SAM2, raw crop, routing-policy, BLIP3 question/image-composition, renderer,
  artifact-budget, residency, device or model-generation change beyond plumbing
  the required prompt evidence and exact test.
- Do not split canonical scalar strings, concatenate canonical arrays, silently
  truncate/deduplicate/clamp/ignore values, or convert identifiers into prose.
- No new endpoint, persistent result store, external download, model/cache path,
  service unit/environment/key, port/network/firewall/VPN, driver/CUDA, system
  package, unrelated process/device or dependency mutation.
- No tag, package publication, release, merge, auto-merge or public exposure.
- Do not commit generated ZIP/PNG, request YAML, credentials, model weights,
  caches or test outputs. The repository fixture remains unchanged.

## Acceptance criteria

1. Canonical scalar and ordered-array prompt forms validate exactly; commas and
   newlines in a scalar remain one prompt; all bounds and structured errors are
   enforced without prompt-related HTTP 500.
2. The exact pinned tokenizer rejects token count 78 before SAM2/inference and
   accepts 77 unchanged; the adapter has tested fail-closed defense with no silent
   truncation.
3. Separate prompt embeddings aggregate to one max score per semantic class with
   deterministic tie behavior; routing uses only five semantic classes in the
   exact configuration.
4. L3 contains exact 32/15/15/20/15 and 97 prompt accounting, complete class
   vectors, per-class winning prompt indices and overall winning prompt evidence,
   all typed/bounded and JSON/ZIP consistent.
5. Schema/capabilities/OpenAPI advertise string-or-array, exact limits/errors and
   required canonical `falsecategory`; maintained docs and runtime agree.
6. Every deterministic regression, full suite, coverage/static/package/security
   gate and all seven exact-head checks pass.
7. The verified exact image plus exact 97-prompt YAML returns HTTP 200 through the
   deployed API, produces the final labelled visualization and supplies all
   required stage/count/hash/visual inspection evidence. The deliberate >77-token
   live negative returns 400 before inference.
8. One corrected PID is left enabled/active/ready on private
   `10.8.132.76:17891`, with sole assigned RTX 3090 use, zero automatic restarts,
   unchanged key/environment digest, empty service workspace and no sensitive
   leakage.

The strongest reason not to merge is that CPU/fake tests could prove the array
shape while production tokenization still concatenates, truncates or validates
too late, and a successful HTTP response could conceal routing by prompt rather
than semantic class. Answer it with exact processor-call/token-ID tests, typed
pre-SAM2 78-token rejection, complete five-class vectors, live 97-count metadata,
one successful exact request, and independent final-image/count inspection.

## GitHub publication and immutable report contract

Create branch `oap/022-a-canonical-clip-multiprompt` from exact remote main
`d341a3c4ba47b71d10d70682771b315041dcbcb8` and exactly one PR targeting `main`.
Include the unchanged published order and `oap/active=022-a` with implementation.
Push all non-report work, obtain all required checks, perform authorized live
qualification, and capture the literal implementation SHA. Then create exactly
one immutable `oap/reports/022-a-report.md` from the report template, commit only
that report as final child with title `OAP report 022-a (SELF)`, push, verify its
parent equals the implementation SHA and remote commit changes one path, require
the same seven checks green again, mutate nothing further and send FIFO `OK`.

The report must include exact PR/base/head/starting/implementation/SELF SHAs,
commit topology, changed paths/diff bounds, every criterion, exact commands and
results, full and focused counts/coverage, all check names/run URLs/statuses,
prompt normalization/limit/tokenizer/preflight/adapter design, structured error
examples without prompt text, score aggregation/tie/routing proof, typed schema/
capabilities/OpenAPI/docs inventory, falsecategory reconciliation, compatibility,
and exact live service/GPU/resource/restart evidence. Include sanitized stage
counts, visual-observation counts and the final PNG tmpfs path/hash/dimensions/
size, but no image bytes, full YAML, raw answers, prompts, credentials, token IDs,
host-private paths beyond the deliberately returned tmpfs artifact, or model
internals. State limitations and the strongest remaining reason not to merge.

## Deferred human adjudication

- Decision: NONE

This is a bounded, reversible API validation/scoring correction using the
already pinned local model and explicitly authorized private-LAN fixture test; it
does not meet the `CRITICAL.md` five-condition threshold.
