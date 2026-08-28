# OAP Work Order — 014-a — Post-filter rejection diagnostics

## Objective

Make the post-SAM2 area/bounding-box filter explain its decisions at service
verbosity 3. Add deterministic aggregate counts for candidates removed by
`maxsize`, `max_w`, `max_h`, empty masks, and candidates retained. Add a bounded
numeric-only rejection list so an operator can distinguish a SAM2 recall miss
from a candidate that SAM2 produced and the configured post-filter removed.

This is one new Objective-014 branch and PR. It completes the final requested
fix. Do not redesign the labelled renderer or BLIP3 verification merged in
Objectives 012 and 013.

## Verified starting state

- Remote `main` is `2e8c67997c2480cf66f5c87a1e19afba4c6d368f`, the merge of
  Objective-013 PR #69. Its post-merge CI run `33199006177` and CodeQL run
  `33199006173` are successful. GitHub has no open PR.
- Required branch: `oap/014-a-post-filter-rejection-diagnostics`; create exactly
  one PR titled `Objective 014: post-filter rejection diagnostics` against
  `main`.
- Coding checkout is on the clean Objective-013 branch at report-only SELF
  `c916158c92ec7bfc98934788a86efd6865662bca`. Fetch and branch from exact remote
  main while preserving the atomically published next order/active transcript;
  do not replay or amend PR #69.
- `src/postprocessing.py::filter_by_area_bbox` currently returns only the kept
  list. It evaluates `area > maxsize` first, skips empty masks, then retains a
  candidate only when both inclusive bbox width and height are within `max_w`
  and `max_h`. It emits only a coarse numeric log.
- `src/core/engine.py` assigns every non-empty remapped SAM2 candidate a numeric
  `_source_index`, records `sam2_candidates` and `after_area_bbox`, and discards
  rejected mask dictionaries after the filter. `PipelineResult` has no
  post-filter diagnostic field.
- At L3, `src/service/envelope.py` exposes existing `candidate_counts` but no
  rejection reasons. L0-L2 do not expose stage diagnostics. JSON and ZIP use the
  same prepared document, and `zap-it.v1` permits additive level-gated fields.
- No operator roof/panel photograph is available in the repository or strategic
  workspace. The exact two-wide-candidate roof scenario is therefore a
  programmatic CPU regression and must not be described as a real-image model
  benchmark.
- `zap-it-lan.service` is enabled, active and ready on exact
  `10.8.132.76:17891`, MainPID `402706`, `NRestarts=0`, with one listener. Its
  mode-0600 environment is unchanged; never print, rotate, commit, log or report
  either API key.
- Host `hinton2`; assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The service is the only
  compute process and used 13,524 MiB in the strategic snapshot. The application
  sees only logical `cuda:0`.
- `/dev/shm` is an approximately 12-GiB tmpfs and the private service request
  root is empty. No second model process or listener is required.

## Required diagnostic contract

### 1. One deterministic filter decision per candidate

1. Refactor the canonical area/bbox evaluation narrowly enough that filtering
   and diagnostics use the same comparisons and cannot drift. Preserve the
   existing public `filter_by_area_bbox(masks, post_maxsize, max_w, max_h,
   verbosity=..., log_print_func=...) -> list` behavior for every existing
   caller. An optional diagnostics collector or a separate detailed helper is
   acceptable; existing positional calls and list return values must not break.
2. Every input candidate receives exactly one terminal outcome using this fixed,
   mutually exclusive precedence:

   ```text
   area > maxsize             -> maxsize
   otherwise empty mask       -> empty_mask
   otherwise bbox width > max_w -> max_w
   otherwise bbox height > max_h -> max_h
   otherwise                  -> retained
   ```

   A candidate violating width and height is counted only under `max_w`; one
   violating area and another dimension is counted only under `maxsize`.
   Threshold equality is retained for that criterion, matching current law.
3. Bbox width and height are inclusive pixel extents (`max - min + 1`) over the
   exact remapped segmentation mask. `area_px` is the numeric area used by the
   existing filter decision. Diagnostics must not substitute an image crop,
   approximate bbox, label, prompt or later instance ID.
4. Preserve input order and mask dictionaries. Do not mutate segmentation,
   scores, labels or source indices. Filtering results for all existing inputs
   must remain byte/identity compatible apart from the new sidecar metadata.
5. Aggregate invariants must be checked by tests:

   ```text
   evaluated == retained
              + removed_by_maxsize
              + removed_empty_mask
              + removed_by_max_w
              + removed_by_max_h
   retained == len(filter_result)
   ```

### 2. Bounded numeric-only rejection records

1. Define one fixed implementation constant
   `MAX_POST_FILTER_REJECTION_RECORDS = 256`. Record only rejected candidates,
   in original filter-input order, up to that cap. Aggregate counts always cover
   all evaluated candidates even when records are truncated.
2. Each rejection record has exactly these public values:

   ```json
   {
     "source_index": 7,
     "reason": "max_w",
     "area_px": 1812,
     "bbox_width_px": 151,
     "bbox_height_px": 12
   }
   ```

   `source_index` is the non-negative integer `_source_index` assigned by the
   core, with the deterministic input ordinal used only as a defensive fallback
   for trusted legacy callers lacking a valid source index. Empty masks use
   numeric width and height `0`.
3. Never expose mask pixels/RLE, crops, coordinates, image content, prompts,
   labels, answers, filenames, config text or arbitrary candidate dictionary
   fields in these records. Never use any record value in an artifact path,
   metric label or unbounded log.
4. Add `rejections_truncated`, equal to the number of rejected candidates not
   represented after the fixed cap. An empty/no-rejection result uses an empty
   list and zero truncation, never a missing or ambiguous sentinel.

### 3. Core and L3 wire schema

1. Add a deterministic `post_filter_diagnostics` mapping to `PipelineResult`
   without making existing trusted/fake constructors incompatible. The canonical
   engine populates it from the exact post-SAM2 filter pass. Detailed rejection
   record construction may be skipped below verbosity 3; it must not cause
   lower-level visualization, artifact or serialization work.
2. At service verbosity 3 only, add this sibling of `candidate_counts`:

   ```json
   "post_filter_diagnostics": {
     "limits": {"maxsize": 100000, "max_w": 100, "max_h": 80},
     "evaluated": 4,
     "removed_by_maxsize": 1,
     "removed_empty_mask": 0,
     "removed_by_max_w": 2,
     "removed_by_max_h": 0,
     "retained": 1,
     "reason_precedence": ["maxsize", "empty_mask", "max_w", "max_h"],
     "rejections": [],
     "rejections_truncated": 0
   }
   ```

   The example's list is abbreviated; actual records follow the exact contract
   above. Limits are the effective numeric thresholds actually used.
3. Add explicit Pydantic/OpenAPI models for the diagnostic and rejection record
   instead of leaving the new public contract as an undocumented arbitrary
   mapping. Reasons are a closed literal set. Counts, limits, dimensions and
   source indices are numeric and non-negative where applicable.
4. The entire field is absent at L0, L1 and L2. Preserve the current meanings
   and keys of `candidate_counts`; in particular,
   `candidate_counts.sam2_candidates == post_filter_diagnostics.evaluated` and
   `candidate_counts.after_area_bbox == post_filter_diagnostics.retained` in the
   canonical path.
5. JSON and ZIP `manifest.json` must carry byte-equivalent diagnostic values for
   the same result. Repeated equal inputs/config must produce identical
   diagnostics. Wall-clock timings and request IDs remain outside this
   determinism claim.
6. Keep schema version `zap-it.v1`: this is an additive optional L3 field, not a
   reinterpretation of an existing field. Do not add an artifact or persist
   diagnostics separately.

## Required CPU/API tests

1. Direct filter tests cover retained, `maxsize`, empty, `max_w` and `max_h`
   outcomes; exact-threshold retention; and candidates violating multiple
   limits to prove the stated reason precedence.
2. Preserve and explicitly test the legacy list return, object identity/order,
   existing positional/keyword calls, and bounded content-free logging.
3. Add the required roof-failure regression: at least two deterministic masks
   representing apparently missing panel-array candidates exceed `max_w` while
   satisfying area and height. Assert both were present in `evaluated`, both are
   absent from the kept list, both count under `removed_by_max_w`, and their
   rejection records contain the correct source IDs and measured widths. This
   is filter evidence, not a SAM2/model-accuracy claim.
4. Prove inclusive bbox dimensions and remapped source indices for border-
   touching and disconnected masks. Candidate text or extra dictionary fields
   must never enter diagnostics.
5. Exercise 257 or more rejected candidates: exactly 256 ordered records,
   correct full aggregate count and exact `rejections_truncated`; repeated runs
   produce equal mappings.
6. Core tests prove aggregate invariants, candidate-count cross-checks, no change
   to final object identity/order, and no diagnostics contamination from CLIP,
   BLIP3 or the later keep-label filter.
7. Service unit/API/schema tests prove the exact L3 shape, closed reason values,
   absence at L0-L2, JSON/ZIP manifest parity, deterministic values, bounded
   serialized size, and unchanged existing response fields.
8. Existing labelled-renderer, BLIP3 mask-aware, hostile-YAML, artifact-budget,
   auth, cancellation, metrics, package and legacy CLI tests remain green.

Run and report the canonical CPU suite with coverage, focused postprocessing/
core/envelope/API/schema tests, Ruff format/check, compileall, documentation
checker, shell syntax where changed, wheel/sdist build, artifact audit,
tracked-tree and built-artifact secret scans, `twine check`, and
`git diff --check`. Public CI must not use CUDA or download models. All required
current-head CI and CodeQL checks must be present and successful.

## Bounded live private-LAN qualification

Keep the current service enabled and active during ordinary implementation.
After the implementation head is committed and all CPU/static checks pass, one
controlled restart of only `zap-it-lan.service` is authorized so the running
process uses Objective-014 code. Before restart independently re-verify exact
assigned physical index/UUID/PCI/name/VRAM/process ownership, unit/listener,
`/dev/shm`, and environment-file digest without reading or reporting a key. Do
not start a second model process or touch drivers, firewall, routes, VPN or any
unrelated unit.

After readiness returns:

1. Prove missing/wrong inference keys still return 401, authenticated readiness/
   metrics work, and docs/OpenAPI remain 404.
2. Use the already-authorized local ignored goat fixture/config, transformed in
   memory only. Remove optional CLIP/BLIP3/visualization work for this diagnostic
   probe; disable debug; retain the supported SAM2 profile; set effective
   `maxsize` and `max_h` to large bounded values and `max_w: 0`. Every non-empty
   candidate must therefore fail the width criterion without depending on model
   semantics. Do not copy the image, config or labels into Git/OAP/chat.
3. Send authenticated L3 JSON and ZIP plus repeated L3 JSON requests. Require at
   least one SAM2 candidate, zero retained, `removed_by_max_w == evaluated`, all
   other removal counts zero, aggregate invariants, source-indexed rejection
   records whose measured widths exceed zero, exact candidate-count cross-checks,
   JSON/ZIP diagnostic parity, and repeat determinism. Also send one L2 JSON
   request and prove `post_filter_diagnostics` is absent.
4. Prove unchanged post-restart PID/listener across requests, only the assigned
   GPU process, bounded response/resource metrics, sanitized journal, empty
   request workspace, preserved mode-0600 environment and unchanged key digest.
   Leave the unit enabled, active and ready on `10.8.132.76:17891`.

Disclose every failed live attempt and corrective action. A missing field,
non-reconciling count, unexpected reason, unsafe record, parity mismatch,
service/GPU drift or request residue is not acceptance.

## Documentation and provenance

Update `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`,
`docs/SERVICE-DATASHEET.md`, and any schema snapshot/current document affected.
Document the exact reason precedence, strict/inclusive threshold behavior,
aggregate invariant, numeric-only 256-record cap/truncation, L3-only field,
candidate-count relationships, and the roof-test interpretation. Remove or
correct any statement that says only before/after counts are available.

Do not change model identities/revisions/licenses, dependencies, hardware or
residency claims. The diagnostics explain configured filtering; they do not
prove SAM2 recall, model accuracy or that a rejected mask was semantically
correct.

## Non-goals

- no renderer, label-placement, BLIP3 composer/instruction, CLIP, SAM2 model or
  mask-generation change;
- no change to filter thresholds, comparisons, order, final-object ordering,
  candidate-count meanings, verbosity 0-2 response, schema version or artifact
  budgets;
- no mask/crop/RLE/coordinates/user text in rejection diagnostics, logs or
  metrics; no diagnostic artifact or persistent request data;
- no GPU sharing/MPS/MIG, second process/service, public/WAN bind, TLS/gateway/
  firewall/VPN/network change, key rotation/disclosure, release/tag/upload or
  unrelated cleanup;
- no rewrite of prior immutable orders/reports and no merge by coding.

## Acceptance and report contract

Acceptance requires all requirements above: exact mutually exclusive counts and
precedence; useful bounded source-indexed numeric records; the two-wide-candidate
roof regression; unchanged filter output and lower verbosity; deterministic
L3 JSON/ZIP schema; green CPU/CI/CodeQL; and satisfactory real private-LAN
integration with `max_w` removals visible.

The strongest reason not to accept is that diagnostics could drift from the
actual short-circuit filter or misleadingly double-count candidates that violate
multiple limits. Answer it by sharing one decision evaluator, fixing and testing
the precedence, enforcing the aggregate equality and kept-length cross-check,
and checking the canonical `candidate_counts` relationships in CPU and live
responses.

Push all implementation and exact active/order bytes before reporting. Record a
literal 40-hex implementation SHA. Then create exactly
`oap/reports/014-a-report.md`, commit only that report as the final `SELF` child,
push, verify remote parent/one-path topology and bytes, send exactly one response
FIFO `OK`, perform no later mutation, and exit. Coding never merges.

## Deferred human adjudication

- Decision: NONE
