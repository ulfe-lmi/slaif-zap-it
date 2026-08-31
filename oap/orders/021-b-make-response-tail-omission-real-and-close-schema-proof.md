# OAP Work Order 021-b — Make response-tail omission real and close schema proof

## Authority and exact state

- Continue Objective 021 on existing PR #77 and branch
  `oap/021-a-nonfatal-artifacts-and-contract-docs`; create no new PR.
- Start from the published report-only head
  `ebf4fc4f64646a7afb7af590e9afa12310e1614b`, whose SELF parent/corrected
  implementation head is `653520dab50bad2b6c5a5bc588c178d2a149643d`.
  Remote `main` remains `f2d58f7512af41751cb647bcd502d767a007f199`.
- All seven checks are green on both the corrected 021-a implementation and
  report heads, but green CI is insufficient because independent diff review
  found the concrete defects below.
- Publish this exact order, set `oap/active` to `021-b`, make bounded
  implementation/test/docs corrections, then one report-only commit titled
  `OAP report 021-b (SELF)` changing only `oap/reports/021-b-report.md`.
- Do not merge/auto-merge, restart/reload the service, use the GPU, run live
  inference, change dependencies, network, key, host policy or `CRITICAL.md`.

## Merge-blocking defect: response fitting does not remove payload bytes

`ArtifactDeliveryLedger.drop_last_for_response()` changes the last admitted
entry to `omitted_response_limit`, but `_PreparedResponse.artifacts` remains
unchanged. `_refresh_prepared()` regenerates descriptors for every unchanged
artifact, `_json_size_upper_bound()` adds every unchanged payload again, and ZIP
assembly writes every unchanged payload again. Consequently the response does
not shrink. The loop eventually exhausts ledger entries and still returns
`response_too_large`; the documented non-fatal final response-byte behavior is
not implemented.

Correct this at one authoritative seam:

- after each response-limit omission, derive a new immutable artifact tuple
  containing the required `identity-mask.png` plus only optional artifacts whose
  ledger status remains `stored`;
- regenerate descriptors, hashes, byte totals and artifact-delivery metadata
  from exactly that tuple;
- JSON upper-bound calculation and ZIP assembly must use that reduced tuple;
- never omit the required identity artifact through the optional ledger;
- update CLIP/BLIP3 debug status records to `omitted_response_limit` while
  preserving structured candidate evidence; and
- terminate deterministically: return success as soon as the reduced envelope
  fits, or return `response_too_large` only when the essential document still
  cannot fit.

Add direct builder and API tests for JSON and ZIP. Compute deterministic
essential-only and with-optional sizes from synthetic data, choose a cap between
them, and prove: HTTP/builder success, optional bytes absent, identity retained,
`truncated=true`, correct omitted reason/status/counts, exact delivered
hashes/sizes, and ZIP manifest/member parity. Add a cap below the essential-only
size proving the retained hard 413. A raw-byte omission test is not a substitute
for this response-byte pressure test.

## Bounded-ledger correction

`ArtifactDeliveryLedger._record()` currently runs while a new entry still has
its default `stored` status. The selection/budget decision is assigned only
afterward, so the advertised 576-entry omission bound is never applied.

- Decide the final status before bounded public recording, or otherwise apply
  the bound to the final status.
- Track unreported selection exclusions separately from unreported budget
  omissions so `eligible_count`, `selected_count`, `selection_excluded_count`,
  `budget_omitted_count`, `truncated` and the public omitted-list arithmetic
  remain exact.
- Never let more than 576 omission records enter the public model; delivered
  names stay bounded by operator artifact limits plus the one identity artifact.
- Repeated fixed-name offers must be idempotent only when their stage/media/
  identity and size facts agree. Reject contradictory duplicate offers as an
  internal artifact error rather than permitting a size/budget bypass.
- Add a synthetic test offering more than 576 excluded/omitted artifacts in a
  mixed pattern. Validate the resulting `ArtifactDeliveryMetadata`, exact
  category counts, overflow counts, warning, bounded list and determinism.

## Strict selector correction

The validator and response schema currently accept candidate ID 257 and larger,
although the documented/requested range is 1..256. They also call `set()` before
verifying list member scalar types, so a nested list/mapping can escape as an
unhandled `TypeError` instead of a precise `invalid_config`.

- Enforce every `diagnostic_artifacts.candidate_ids` value as a strict integer
  from 1 through 256 in hostile validation, normalized metadata and Pydantic
  response validation.
- Validate element types before uniqueness for both `stages` and
  `candidate_ids`; all nested/unhashable/malformed values must produce the
  established sanitized 400 `invalid_config`, never 500.
- Add boundary, bool, 0, 257, duplicate, nested list and nested mapping tests.
  Preserve requested order and sorted effective IDs for valid input, with A/B/A
  request-local state proof.

## Generated OpenAPI and response-schema closure

The 021-a runtime capabilities response contains an 80-entry dictionary, but
OpenAPI only sees `Dict[str, CapabilityField]`; it cannot enumerate the accepted
paths. The response schema also leaves stage statuses, candidate counts,
timings, provenance and effective routing as broad `Dict[str, Any]` seams even
though Objective 021 explicitly required those surfaces to be documented.

Close this without breaking existing JSON keys:

- retain the existing capabilities dictionaries for compatibility, but add a
  typed ordered field catalog whose `path` property uses an OpenAPI-visible enum
  containing every canonical `service_config_leaf_paths()` value exactly once;
- each catalog record references the typed `CapabilityField` descriptor and
  states required/null/default semantics. Every record must have type, stage and
  non-empty description; units/profile/operator-limit metadata is required when
  applicable. Generate both dictionary and catalog from one source so drift is
  impossible;
- add named response models for stage status, canonical candidate counts,
  provenance and effective CLIP routing configuration. Type timing maps as
  finite non-negative millisecond values and document their dynamic stage-key
  convention. Runtime keys and compatibility behavior remain unchanged;
- point `ServiceMetadata` fields to those named types so OpenAPI contains real
  `$ref` evidence rather than only property names with free-form objects;
- preserve dynamic per-label CLIP score maps and sanitized runtime metadata as
  bounded typed maps where truly dynamic; and
- add an OpenAPI test proving the catalog path enum equals the validator
  inventory, every catalog descriptor satisfies its metadata obligations, and
  the named response models are referenced by `ServiceMetadata`.

Do not claim that a runtime map alone makes every accepted field visible in
generated OpenAPI. Reconcile API/config documentation if the corrected schema
shape or semantics require it.

## Preservation and verification

- Preserve the accepted raw CLIP crop, complete cosine vectors/permissive
  routing, single contextual BLIP3 image, exact question mapping, geometry
  diagnostics, SAM2 request-local generator/config/provenance, labelled renderer,
  fixed safe names, legacy CLI boundary and shipped configs.
- Do not weaken essential object/RLE/identity/deadline/input/resource limits or
  silently clamp any request.
- Run focused 021 artifact/envelope/API/validator/capabilities/OpenAPI/schema
  tests; all existing core/service/candidate-view/routing/BLIP3/renderer/SAM2 and
  shipped-config tests; full default and coverage suites; format, lint,
  compileall, docs, diff check, package/release parity, scans, twine, systemd and
  isolated direct/sdist install smokes.
- Require all seven checks green on the exact corrected implementation head,
  then all seven again on the 021-b report-only SELF head before FIFO signaling.
- Report exact changed files/SHAs/commands/results, corrected JSON and ZIP size
  thresholds/results, >576 ledger arithmetic, invalid selector matrix, OpenAPI
  enum/model proof, docs changes, retained compatibility, service/GPU
  non-mutation and strongest remaining reason not to merge.

## Deferred human adjudication

- Decision: NONE
