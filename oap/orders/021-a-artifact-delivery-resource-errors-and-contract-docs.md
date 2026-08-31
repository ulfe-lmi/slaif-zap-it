# OAP Work Order 021-a — Non-fatal diagnostic delivery and complete API evidence

## Objective

Complete the remaining domain-neutral ZAP-IT pipeline acceptance surface after
the accepted Objective 020 implementation:

1. optional diagnostic/visualization artifact overflow must never replace a
   successful inference with HTTP 413;
2. clients must be able to select bounded diagnostic stages, candidate IDs and
   an artifact page without controlling paths or destinations;
3. SAM2 capacity rejection must return sanitized structured estimates, causes,
   limits and admissible alternatives;
4. capabilities, generated OpenAPI schemas and maintained documentation must
   exhaustively describe the request and response evidence added in Objectives
   015–020; and
5. every repository-owned example configuration must pass the same hostile
   request validator and default deployed operator limits used by the API.

This is the second and final dependency-complete part of the human-requested
domain-neutral pipeline redesign. Preserve the accepted execution order and
pixel contracts from Objective 020. In particular, CLIP receives the complete
unmodified rectangular source crop, BLIP3 receives exactly one contextualized
mask-derived image, and their constructors remain separate.

## Verified GitHub, OAP, and host state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote default branch: `main` at merge SHA
  `f2d58f7512af41751cb647bcd502d767a007f199`, Objective 020 PR #76.
- Post-merge CI run `33403791631` and CodeQL run `33403791614` both completed
  successfully on that exact SHA. There are no open pull requests.
- Create branch `oap/021-a-nonfatal-artifacts-and-contract-docs` from that exact
  remote main and one PR titled
  `Objective 021: non-fatal artifacts and complete API contract`.
- The coding worktree is clean but is still checked out on the merged Objective
  020 branch; fetch and create the new branch from the exact verified remote
  main. Do not reuse or amend PR #76.
- Current immutable selector is merged round `020-c`. Publish this exact order
  and set `oap/active` to `021-a` in the implementation history.
- Host is `hinton2`. The authenticated private-LAN service is enabled,
  active/running and listening only on `10.8.132.76:17891`, PID `607106`,
  `NRestarts=0`, start timestamp `2026-08-31 12:36:38 CEST`; `/healthz` and
  `/readyz` return 200. It must remain running and unchanged for this objective.
- The assigned device is physical GPU0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB. The sole compute process is the existing service
  and uses approximately 12,010 MiB. `/dev/shm` has approximately 11 GiB free
  and `/dev/shm/slaif-zap-it` is mode 0700.

## Reconciled current defects

Current main still treats optional L3 bytes as inference-fatal in several
independent places:

- `BoundedMemoryArtifactSink.ensure_capacity()` and `_commit()` raise
  `ArtifactSinkError` when debug count/single/total raw-byte budgets overflow;
- the core preflights CLIP and BLIP3 debug images before those stages and aborts
  model execution when the sink refuses them;
- the service reserves raw visualization bytes and translates sink refusal into
  `response_too_large`; and
- envelope collection rejects artifact count, per-artifact size and total raw
  artifact size instead of omitting optional bytes.

The inference result therefore disappears precisely when diagnostic output is
most useful. Current `resource_limit` errors also expose only a short message,
and the capabilities schema leaves several configuration and response surfaces
as stringly `Dict[str, Any]` descriptions without the requested field metadata.

## Exact diagnostic-artifact request contract

Add this API-safe top-level configuration section:

```yaml
diagnostic_artifacts:
  stages: [sam2, clip, blip3, visualization]
  candidate_ids: null
  page: 1
  page_size: 48
```

It is a request-local delivery selection, not an algorithm/model setting.
Validate strictly without coercion:

- the section is a mapping with only the four named fields;
- `stages` is a unique list of 1..4 exact enum values `sam2`, `clip`, `blip3`,
  `visualization`; omitted means all four;
- `candidate_ids` is null or a unique ascending-or-unsorted input list of 1..256
  strict positive integer source candidate IDs; preserve the requested list in
  requested metadata but normalize the effective filter to ascending order;
- `page` is a strict integer `1..65535`, default 1;
- `page_size` is a strict integer `1..48`, default 48; and
- reject unknown fields, nulls except `candidate_ids`, bool-as-int, duplicate
  values and invalid types/ranges as `invalid_config` or `unsupported_field`
  consistently with the existing validator taxonomy.

At verbosity below 3 this section remains valid but no optional L3 artifact is
delivered; emit one bounded warning and report it as not applied. It must never
enable a stage's debug flag. Existing stage debug flags remain the authority for
whether that stage produces eligible diagnostic artifacts. `stages` can only
narrow enabled stages.

Selection is applied in deterministic pipeline/insertion order to optional L3
artifacts. `candidate_ids` filters candidate-specific CLIP and BLIP3 input PNGs
by stable one-based `source_candidate_id`; it does not reinterpret aggregate
SAM2 contact sheets or final visualization streams. `page`/`page_size` paginate
the resulting eligible artifact sequence after stage and candidate filtering.
Artifacts excluded by stage, candidate or page selection are reported as
`not_selected_stage`, `not_selected_candidate` or `not_selected_page` and do not
count as budget truncation. No request value may become a filename, directory,
path, destination, media type or renderer identifier.

Expose the exact fields/defaults/ranges/semantics and the operator response
budgets through `/v1/capabilities`. Preserve safe fixed artifact names.

## Non-fatal optional artifact admission

Refactor optional in-memory artifact collection into an explicit deterministic
admission ledger. Invalid logical names, corrupt in-memory records and internal
encoding failures remain real errors; merely exceeding an artifact count,
single-artifact raw-byte, aggregate raw-byte or response-byte budget is an
omission, not an exception and not an inference failure.

Required behavior:

- never skip or abort SAM2, CLIP scoring/routing, BLIP3 verification, final
  labeling, ordering or structured serialization merely because an optional
  artifact cannot be retained;
- remove inference-fatal debug `ensure_capacity` preflights. They may become
  non-mutating estimates/admission hints, but their result must not gate model
  work;
- admit optional artifacts greedily in stable pipeline/name order after the
  request selection. An artifact is delivered only when its fixed name is
  unique and its actual/raw representation fits all remaining operator limits;
  otherwise record exactly one omission reason and continue;
- use the same admitted artifact set for JSON and ZIP semantics. Delivered PNG
  bytes, hashes and sizes must be exact and deterministic. A debug PNG that is
  delivered must decode byte-identically to the pre-normalization image actually
  passed to CLIP/BLIP3; never substitute an approximation;
- keep `candidate_view_inputs` and other per-candidate diagnostic metadata even
  when its PNG is not delivered. Set truthful status values such as `stored`,
  `not_selected`, `omitted_count_limit`, `omitted_single_size_limit`,
  `omitted_raw_total_limit` or `omitted_response_limit`;
- account for optional final visualization PNGs and raw-SAM2 PNGs under the same
  non-fatal policy instead of reserving enough room up front or raising 413;
- do not silently clamp a user selection and do not silently discard an
  artifact; and
- do not weaken fixed maximum candidate/object/question counts or create
  unbounded metadata. Cap the public omission ledger at 576 entries, which
  covers 256 CLIP views + 256 canonical BLIP3 views + the bounded SAM2 and
  visualization surfaces. If an internal compatibility path could exceed that,
  aggregate the excess in a typed count and warning without client text.

Add typed `service.artifact_delivery` metadata at L3 with at least:

- requested and effective selection plus `applied`;
- operator budgets: response artifact count, debug sink artifact count,
  per-artifact raw bytes, total raw artifact bytes and total response bytes;
- eligible, selected, delivered, selection-excluded, budget-omitted and
  unreported-overflow counts;
- deterministic estimated raw and base64/ZIP encoded byte totals, clearly
  labeled as estimates, plus actual delivered raw/encoded totals;
- `truncated`, true only when an eligible selected optional artifact is omitted
  by an operator budget;
- delivered fixed names; and
- typed omitted entries with fixed name, stage, optional source candidate ID and
  question ID, estimated raw bytes and an enum reason.

Metadata arithmetic must reconcile. Artifact descriptor hashes/sizes and ZIP
manifest hashes/sizes must match the exact delivered bytes. An artifact excluded
only by client selection does not set `truncated`; an eligible selected artifact
excluded by any operator artifact/response budget does.

The core may carry a request-local selection/admission interface, but the legacy
filesystem sink must retain its trusted CLI behavior and must not acquire service
response budgets. Preserve old public sink methods where practical or provide a
bounded compatibility seam with direct tests.

## Hard limits that remain hard

This objective separates optional byte delivery from successful inference; it
does not promise an unbounded response. The following may still return their
existing structured failure when the essential contract itself cannot fit:

- multipart/config/image input bounds and request deadline;
- fixed object/candidate/question limits;
- inability to create the required verbosity-1 uint16 identity PNG without
  identity loss;
- configured mask-RLE run limits;
- JSON/ZIP structure/manifest itself exceeding the final response cap after all
  optional artifacts have been omitted; and
- invalid artifact names or genuine internal serialization corruption.

Document this distinction precisely. Before the final hard response-size check,
drop admitted optional artifacts from the tail in reverse admission order until
the envelope/ZIP fits, recording `omitted_response_limit`. Only if the essential
document still does not fit may `response_too_large` remain. JSON and ZIP need
not have byte-identical envelopes, but must expose equivalent selection,
truncation, delivered/omitted identity and hashes for their respective payload.

## Structured SAM2 `resource_limit` evidence

Extend `ServiceError` and the frozen error envelope with an optional sanitized
typed `details` object. Existing errors without details retain the existing
three fields exactly. Do not expose raw YAML, image content, host paths, device
details, environment values, model cache facts or credentials.

For every SAM2 operator-cap rejection, return HTTP 413/code `resource_limit` and
details containing:

- `limit_kind`: `field`, `estimated_prompt_count` or
  `estimated_mask_prediction_count`;
- requested/effective SAM2 scalar values and selected profile;
- the complete deterministic `estimated_prompt_count` and
  `estimated_mask_prediction_count` for the rejected effective configuration;
- applicable public operator limits;
- `causing_values`, limited to the direct over-cap field and/or the work-driving
  fields `points_per_side`, `crop_n_layers`,
  `crop_n_points_downscale_factor`, and `multimask_output`;
- one or more distinct `admissible_alternatives`, each containing a complete
  request-safe `mask_generator` mapping and its two estimates; and
- a bounded generic warning when relevant.

Generate alternatives deterministically from public defaults/profiles and, if
necessary, a conservative minimum-work configuration. Validate every proposed
alternative with the same intrinsic validation and the same current
`ServiceSettings.sam2_operator_caps` as the rejected request. Never offer an
alternative that the deployed validator would reject, silently clamp the
original request, reload a model, or expose operator-only controls. Test direct
field overflow, prompt overflow and multimask-driven prediction overflow under
custom low-cap settings.

SAM2 successful metadata continues to distinguish requested and effective
values. For every field, retain `source: explicit|profile|default`, publish the
applicable operator limit when one exists, and publish
`operator_limit_applied: false`. Because capacity policy rejects rather than
clamps, no successful effective value may falsely claim an `operator_limit`
source. Document that exact interaction.

## Complete capabilities and generated response schemas

Make `/v1/capabilities` the machine-readable source of truth and ensure its
Pydantic models appear fully in generated OpenAPI. Extend each public field
descriptor to include, where applicable:

- exact type and nullability;
- default;
- minimum/maximum or allowed values;
- units;
- execution stage;
- concise behavior/description;
- profile interaction; and
- operator-limit interaction/name.

Do not leave the new canonical fixed configuration surfaces as prose-only
`Dict[str, Any]`. Use typed capability models for SAM2, preprocessing, geometry,
CLIP labels, CLIP routing, BLIP3 rules, CLIP/BLIP3 candidate views,
visualization/renderers and diagnostic-artifact selection. Dynamic label/rule
keys may use one explicitly documented wildcard entry, e.g.
`clip.labels.<identifier>` and `blip3.<routing_label>.<field>`.

Create one canonical inventory of all service-accepted leaf paths and a test
that compares it to the capabilities field inventory. At minimum it must cover:

- `alpha` and every accepted `preprocessing` field;
- every safe `mask_generator` scalar plus `profile` and `debug`;
- every canonical `postsam2processing` field and documented legacy alias;
- `clip.debug`, identifier/prompt values, and every routing policy field;
- every BLIP3 rule field;
- every CLIP and BLIP3 candidate-view field;
- every visualization entry field and safe renderer enum; and
- every new `diagnostic_artifacts` field.

Keep backward-compatible capability keys where existing clients may rely on
them, but make their values derive from the canonical typed inventory so they
cannot drift.

Replace response `Dict[str, Any]` seams with named typed models wherever needed
so generated OpenAPI and model validation explicitly document:

- stable source candidate IDs and filtered indices;
- all CLIP scores, winning label/prompt, routing decision, matched conditions,
  exact primary reason and cap outcome;
- configured/effective BLIP3 question, raw/normalized answer, mapping outcome
  and final label;
- candidate counts after every canonical stage;
- requested/effective SAM2 configuration, source/provenance, estimates, actual
  candidates, timing and warnings;
- stage timings/statuses and request warnings;
- candidate-view input geometry and artifact status; and
- the full artifact-selection/truncation ledger.

Add field descriptions and model validators for boundedness/arithmetic. Keep
dynamic per-label score maps typed as finite float maps and preserve label order
in serialized output. Add the optional resource-limit error details model to
the OpenAPI 413 schema without changing normal error codes.

## Repository examples and presets

Treat the four product configurations under `configs/` as service examples.
Each file must pass `parse_hostile_config(..., verbosity=3,
settings=ServiceSettings())`, `CoreConfig.from_mapping`, the same SAM2 capacity
validation used by the deployed API, and response-schema construction with
synthetic/dry-run data. Do not exempt a shipped example as "trusted CLI only".
Batch-only `images`, `video` and `export_yolo_det` keys may continue to be
ignored with their documented warnings, but every algorithmic value must be
accepted and within default operator limits.

Add one CI test enumerating exactly `configs/*.yaml` so a newly shipped example
cannot bypass the gate. Preserve the legacy batch CLI and its batch-only fields.

## Required automated proof

Use self-contained CPU arrays and the fake/dry-run engine; do not require a live
model or external photograph.

Artifact tests must prove:

- tiny max artifact count, per-artifact size, total raw size and final response
  size omit optional L3 PNGs while returning HTTP 200 structured inference for
  JSON and ZIP;
- object records, full CLIP vectors/routing, BLIP3 evidence, candidate counts,
  timings and warnings survive omissions;
- selection by stage, stable candidate ID and page is request-local,
  deterministic, validates strictly and has no A/B/A state leakage;
- excluded versus budget-omitted accounting and `truncated` semantics reconcile;
- fixed safe names only, no client text/path influence, exact delivered hashes
  and sizes, JSON/ZIP manifest parity, and deterministic repeated output apart
  from existing documented request/time fields;
- decoded delivered CLIP/BLIP3 PNGs equal the actual mocked processor/QA input
  arrays; omitted debug records still identify the exact model input and truthful
  status;
- optional raw SAM2 and visualization overflow is non-fatal; and
- an essential document that cannot fit still produces `response_too_large`, so
  the new behavior is not an uncontrolled limit bypass.

Resource/schema/docs tests must prove:

- every SAM2 capacity path returns the typed sanitized detail fields and at
  least one same-validator admissible alternative;
- no alternative includes a forbidden operator field and the rejected request
  is not mutated;
- all accepted configuration leaf fields exactly match the typed capabilities
  inventory with type/default/range/units/stage/profile/operator-limit metadata;
- generated OpenAPI references the named capabilities, response,
  artifact-delivery and resource-error models and contains descriptions for all
  fixed fields;
- all four shipped product configs pass the deployed validator/capacity gate;
- existing CLIP raw-crop, complete-vector/routing, BLIP3 single-image,
  geometry-diagnostic, labelled renderer, SAM2 request-locality and no-state-
  leakage tests remain green; and
- unsupported renderers, unsafe visualization IDs and hostile/path/device/model
  configuration remain rejected.

## Documentation refresh and migration notes

Perform a line-by-line contract refresh of at least `docs/API.md`,
`docs/CONFIG.md`, `docs/ALGORITHMS.md`, `docs/CORE.md`,
`docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`, the root
README/config examples, and generated OpenAPI/capabilities documentation.

Remove every statement that optional L3 raw/debug/visualization overflow is
preflight-rejected or necessarily returns 413. Document selection, pagination,
omission reasons, truncation arithmetic, the retained essential-payload hard
limits, structured SAM2 alternatives, exact CLIP/BLIP3 roles and image contracts,
complete routing/verification response evidence, defaults/ranges/units/stages,
profile precedence and operator-limit rejection. The live API and documents
must agree; add targeted stale-phrase/contract tests rather than relying only on
manual prose review.

The report must include a concise migration section:

- no change to the Objective 020 default `raw_bbox_crop` CLIP or
  `single_dilated_blur` BLIP3 modes;
- `diagnostic_artifacts` is optional and only narrows enabled L3 debug delivery;
- old requests that overflowed optional artifact budgets now succeed with
  `service.artifact_delivery.truncated=true` and omissions;
- essential response overflow can still return 413;
- legacy trusted-CLI masked CLIP mode and batch-only fields retained exactly as
  found unless a direct compatibility defect requires a narrowly documented
  fix; and
- existing error envelopes are unchanged except that applicable
  `resource_limit` errors add optional sanitized `details`.

## Non-goals

- No semantic model-accuracy benchmark, prompt taxonomy redesign, new
  image-generation fixture, model/weight/revision/dependency change, model
  reload, residency change or GPU qualification.
- No network, firewall, port, service unit, bearer-key, operator environment,
  cache or artifact-destination change.
- No separate artifact download endpoint or persistent request/result storage;
  selection plus pagination satisfies this objective without introducing
  retention or authorization risk.
- No removal of the legacy batch CLI or constitutional SAM2→CLIP→optional
  BLIP3→geometry/visualization/YOLO capability.
- Do not claim semantic recall/precision improvement from deterministic CPU
  contract tests.

## Security, privacy, resource, and protected-host constraints

- Uploaded YAML remains hostile. It cannot control paths, filenames,
  destinations, models, revisions, checkpoints, devices, GPU selection, dtype,
  cache, network, code/imports, point-grid uploads or generation settings.
- Never log or place prompts, labels, questions, answers, uploaded filenames or
  other request text in artifact names or metric labels. All artifact names stay
  fixed tokenized identifiers.
- Keep request data in RAM or the existing private `/dev/shm` policy; no
  persistent request images/config/artifacts.
- Do not read, print, copy, rotate or modify the private-LAN bearer key or its
  environment file. Do not include credential material or operator paths in
  commits, reports, logs or error details.
- Do not stop/restart/reload the service or touch the assigned/unassigned GPU.
  CPU tests and ordinary CI only. Preserve the running accepted build for the
  user until strategic authorizes final deployment after merge.
- No system package, driver, CUDA, firewall, network, systemd, service-account or
  host mutation. No dependency updates unless an unavoidable implementation
  blocker is proved and reported before mutation.

## Verification and merge evidence

Run and report exact results for:

- focused artifact sink/envelope/service JSON+ZIP, resource-limit,
  capabilities/OpenAPI, shipped-config, core engine, candidate-view, routing,
  BLIP3 and renderer tests;
- full default suite and full coverage suite with the existing threshold;
- formatter check, lint, compileall, documentation checker and `git diff
  --check`;
- wheel+sdist build, release-artifact parity/audit, tracked-tree secret scan,
  twine checks, systemd unit verification, and isolated direct-wheel and
  sdist-built-wheel smoke tests; and
- all seven required GitHub checks on the exact implementation head, followed
  by all seven again on the report-only SELF head.

No live GPU inference is required or authorized. The report must state that
semantic accuracy was not measured and give the strongest remaining reason not
to merge.

## GitHub publication and immutable report contract

- One numeric objective, one new branch and one new PR. Do not merge or enable
  auto-merge.
- Make one bounded implementation/order/test/docs commit (additional bounded
  correction commits are permitted only if CI requires them before reporting),
  push it, open the unique PR, and require all seven checks green on its exact
  implementation SHA.
- Then create exactly one report-only commit titled
  `OAP report 021-a (SELF)` changing only `oap/reports/021-a-report.md`. Its
  parent must be the reviewed implementation head. Push it and require all seven
  checks green again before signaling the response FIFO.
- The immutable report must state: PR URL/base/head; implementation SHA; SELF
  parent/head; exact changed files; behavior and migration notes; commands and
  results; coverage; all check names/results; example inventory; JSON/ZIP
  artifact-budget proof; structured alternative proof; documentation inventory;
  compatibility retained; service/GPU non-mutation; limitations; and strongest
  reason not to merge.
- Preserve all prior orders/reports and `CRITICAL.md` bytes.

## Deferred human adjudication

- Decision: NONE
