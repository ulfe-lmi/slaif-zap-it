# OAP Work Order — 017-b — Close candidate-view contract and live proof

## Objective

Amend Objective-017 PR #73 to close the explicitly PARTIAL live result and the
independent strategic-review findings before merge. Activate and qualify the
request-local CLIP debug correction, make runtime filenames exactly match the
documented public templates, declare candidate views honestly in the top-level
capabilities/OpenAPI schema, eliminate the still-exported unsafe untouched-crop
compositor, prevent contour/context interpolation from altering right-side
target pixels, make radius-512 dilation time/memory bounded, split debug
resource admission at the correct pre-CLIP and post-CLIP seams, and add the
missing API/schema/A-B-A/resource evidence.

Keep the current service enabled, active and ready while making and verifying
these corrections. After the corrected implementation head, focused tests,
canonical CPU/static gates and current-head CI/CodeQL are green, this round
authorizes one controlled restart of only `zap-it-lan.service`, followed by the
complete CLIP+BLIP3 live qualification. Leave the corrected newest service
enabled, active and ready.

## GitHub state

- Numeric objective / round: `017-b`.
- Mode: `AMEND_EXISTING_PR`.
- Repository/default base and verified SHA:
  `ulfe-lmi/slaif-zap-it`, `main`,
  `645c8604f9c189e1367e6e27a4ce8298c109482a`.
- Required existing branch: `oap/017-a-mask-isolated-candidate-views`.
- Existing PR: #73, `Objective 017: mask-isolated candidate views`, open,
  non-draft, mergeable, against `main`:
  `https://github.com/ulfe-lmi/slaif-zap-it/pull/73`.
- Starting PR/report head:
  `c7e530de18e659633048d9d317bb5e7cd3eca0d8`, the report-only SELF child of
  implementation head `d0ad70e7b978b7c314db596245d061cb42e6c390`.
- Do not create a second PR or branch, rewrite history, amend the immutable
  017-a order/report, merge, or enable auto-merge. Commit the exact 017-b
  selector/order with the corrections on the existing branch.
- All seven required checks passed on the 017-a report head: CI run
  `33279573147`, CodeQL workflow run `33279573182`, and CodeQL check
  `99172397242`. Green prior checks do not answer the findings below.

## Verified review findings and current live state

1. `oap/reports/017-a-report.md` honestly reports PARTIAL. The one authorized
   restart loaded implementation `46e05cb...`; live requests found that the
   resident CLIP holder retained its startup debug flag. Commit `d0ad70e...`
   passes request `clip.debug` explicitly and is CI-green, but it was written
   after the restart and is not loaded in the running process.
2. Code emits `clip-candidate-view-0008.png` and
   `blip3-verification-0008-0001.png`, while the order, README, API, CORE,
   CONFIG, datasheet and capabilities claim literal templates
   `clip-candidate-view-CANDIDATE-0008.png` and
   `blip3-verification-CANDIDATE-0008-QUESTION-0001.png`. Focused tests assert
   the wrong runtime names, and the 017-a report repeats the documented names
   rather than the actual names.
3. `CapabilitiesResponse` does not declare a top-level `candidate_views` field.
   `build_capabilities` injects it as a Pydantic extra and also nests the policy
   under unrelated `raw_sam2_debug`. Thus runtime JSON, the typed model and
   OpenAPI are not one truthful contract.
4. The production BLIP3 filter uses the shared safe builder, but exported
   `compose_verification_image` still implements the superseded minimum-128
   padded untouched-left/context-right rectangle. Many old tests preserve that
   unsafe behavior. A second safe helper exists under another name. This leaves
   a callable product seam contradicting “replace the untouched-context pair.”
5. `compose_candidate_view_pair` bilinearly resizes a source-space context image
   that already contains contour/context pixels, then masks only outside `D`.
   It does not restore right-side pixels inside the resized target from the
   left target-only result before drawing the resized contour. Boundary
   interpolation can therefore mix contour/context into target pixels.
6. `_circular_dilate` runs over the full image for every candidate and caches
   a full-size horizontal result for each distinct disk row radius. At public
   `max_context_pixels: 512`, a 1672x941 mask can retain hundreds of ~1.57-MB
   boolean arrays and repeat hundreds of full-frame cumulative sums. This is
   neither time nor memory bounded in proportion to the input alone and was not
   tested at the advertised maximum.
7. `_candidate_view_debug_capacity` runs once before CLIP and tries to predict
   BLIP3 rules using pre-CLIP `clip_label`/`clip_score`. Label rules and some
   threshold rules become applicable only after CLIP. The method can reserve
   zero BLIP artifacts, then discover overflow only after expensive BLIP3 QA.
8. Only `tests/test_mask_views.py` was added. It has nine test functions (16
   parameterized cases) and no new full API JSON/ZIP manifest, capabilities
   OpenAPI, final object identity/reordering, two-phase admission or resident
   CLIP A/B/A test. Validation endpoints/unknowns/nonfinite cases are also
   incomplete. Current `TESTING.md` and `docs/OUTPUT-PARITY.md` retain old BLIP
   debug-name claims despite the report's stale-claim assertion.

Current service: `zap-it-lan.service` is enabled, active/running and ready at
exact `10.8.132.76:17891`, PID `513853`, `NRestarts=0`, one listener and one
assigned-GPU process. It runs pre-correction implementation `46e05cb...` and
must not be claimed as PR-head code. The mode-0600 environment digest remains
`bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
never print or report credential values.

Assigned host/device facts remain physical GPU index `0`, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
GeForce RTX 3090, 24,576 MiB, driver `610.43.02`, PyTorch `2.5.1+cu124`, CUDA
build 12.4, application logical `cuda:0`. The process is the sole compute
process. `/dev/shm/slaif-zap-it` is mode 0700 and empty. No new deferred human
adjudication dilemma exists.

## Required corrections

### 1. Exact fixed names and identity

Use these exact service-safe names everywhere—runtime sink, input records,
capabilities, schemas/tests, JSON/ZIP manifests and current documentation:

```text
clip-candidate-view-CANDIDATE-0008.png
blip3-verification-CANDIDATE-0008-QUESTION-0003.png
```

The literal tokens `CANDIDATE-` and `QUESTION-` are required. Candidate and
question numbers remain one-based, four-digit minimum formatting; filtered
index remains zero-based manifest metadata and never enters the public service
filename. User prompts, labels, rules, answers, frame/input/request names and
paths remain excluded. Trusted CLI names may prefix a sanitized trusted frame
stem but must retain the unambiguous candidate/question tokens and never use
rule/prompt text.

Add fixed regex/pattern validation to `CandidateViewInputRecord.artifact_name`
or an equivalent typed validator so a stage/name mismatch, missing token,
wrong ID or unsafe segment cannot validate. Prove every input record's name is
the exact artifact descriptor/member name and matches its numeric fields.

### 2. One declared capabilities/OpenAPI contract

Add `candidate_views: CandidateViewsCapability` as a normal required field on
`CapabilitiesResponse`. Construct it directly. Remove `ConfigDict(extra="allow")`,
the post-construction `model_copy`/`copy` injection and
`RawSam2DebugPolicy.candidate_views`. Raw SAM2 debug and semantic candidate views
are independent top-level policies.

Require runtime `/v1/capabilities`, `CapabilitiesResponse` validation and
`CapabilitiesResponse.model_json_schema()`/application OpenAPI to expose the
same top-level fields, defaults, ranges, zero-only fill, formulas, ID bases and
exact name templates. No duplicate nested policy, dynamic extra or undocumented
field is allowed. Capabilities remain authenticated, static and free of paths,
credentials and physical host/GPU topology.

### 3. Remove the unsafe compositor seam

There must be exactly one behavior for public BLIP3 verification composition.
Make exported `compose_verification_image` and
`compose_blip3_verification_image` call the safe shared mask-view builder and
safe pair composer (or make one a direct alias of the other). Delete the old
untouched rectangular implementation, minimum-128 source crop behavior and
obsolete padding/darkening constants/helpers when no longer used. Update all
old verifier tests to assert target-only left, mask-dilated right and zero
outside `M`/`D` rather than preserving an untouched crop.

No production, public helper, debug path or legacy CLI semantic model input may
construct/show an untouched rectangular crop. Compatibility covers callable
names and deterministic safe output, not the unsafe pixels.

### 4. Preserve target pixels through BLIP3 resize

After bilinear RGB resize and nearest target/support mask resize, explicitly
make right-side pixels inside the resized target equal the corresponding
left-side target-only pixels before drawing any contour. Apply zero outside
support and draw contour only in `support minus target`. The binding model-input
invariants are:

```text
left[~target_mask] == 0
right[~support_mask] == 0
right[target_mask] == left[target_mask]
contour & target_mask == false
contour & ~support_mask == false
```

Add high-contrast boundary/diagonal/tiny-mask tests that would detect bilinear
bleed from yellow contour or dimmed context. The exact debug PNG must remain
byte-identical to the QA image.

### 5. Bounded exact circular dilation

Replace the full-frame, radius-proportional cache with an exact Euclidean-disk
dilation whose transient memory is a constant number of arrays proportional to
the relevant source/local window, not `O(radius * image_area)`, and whose work
does not repeat a full-frame cumulative sum for hundreds of radii. Crop a
source-space work window by expanding the tight target bbox by the effective
radius and clipping it to image bounds before dilation; calculate the final
context bbox only from the resulting exact `D`. A linear-time squared Euclidean
distance transform or an equivalently bounded exact algorithm is acceptable.

Keep the public maximum 512 and exact inclusive integer disk definition
`D[p] = any M[q] where squared_distance(p,q) <= radius^2`. Preserve disconnected
components, holes until reached, border clipping and radius zero. Do not add
SciPy, OpenCV or another base dependency.

Tests must compare the optimized result to an independent brute-force oracle
over deterministic random small masks/radii, plus hole/disconnected/border
cases. Add a subprocess performance/resource regression at 1672x941 with
effective radius 512: it must complete within 30 seconds in the canonical CPU
environment and stay below 512 MiB maximum RSS, with valid exact output and no
cached radius-count collection of image-sized arrays. Record actual time/RSS in
the report; this is resource safety, not a broad benchmark claim.

### 6. Correct two-phase resource admission

Split candidate-view debug admission:

1. after post-SAM2 filtering and before CLIP, compute/reserve only the exact
   CLIP debug artifacts that request can emit;
2. after CLIP labels/scores exist and before any BLIP3 QA/model call, compute
   the exact applicable debug-question artifacts from the actual retained
   masks/rules and current sink contents.

Do not predict label/threshold rules from missing pre-CLIP values. Do not
double-count or pretend to reserve without committing. Existing dynamic sink,
encoded-response and deadline enforcement remains active. A count, per-item or
total-byte insufficiency known at either seam must raise the existing sanitized
`response_too_large` mapping before that stage's model call and emit no partial
artifact set from that stage.

Tests must include a label-specific debug rule that only becomes applicable
after mocked CLIP, and a negative-threshold `any` rule whose actual negative
CLIP score becomes applicable. Force count, per-item and total-byte failures;
assert zero BLIP3 QA calls and no BLIP3 partial artifacts. Also prove a CLIP
debug overflow prevents CLIP processor/model calls.

### 7. Close configuration, API and request-local evidence

Expand generated/offline tests to cover:

- omitted/top/child defaults and valid min/max endpoints;
- null/non-mapping top and children, every unknown stage/field, CLIP contour,
  bool-as-number/integer, `.nan`/`.inf`, every below/above range,
  `min > max`, unsupported mode/fill and `clip.padding`;
- strict boolean `clip.debug`, including lower-verbosity stripping, so a string
  or integer cannot activate work;
- resident CLIP holder A/B/A where request debug/config A emits A, B emits B or
  no debug as configured, then A is restored, without holder/model
  reinitialization or mutable state leakage;
- final object identity after post-filter removal, CLIP/BLIP mutation, final
  label filtering and deterministic reorder: `source_candidate_id`, persistent
  zero-based `filtered_index`, independent `instance_id`;
- L0-L3 effective candidate-view manifest/applied status; L3-only bounded input
  records; Pydantic validation and OpenAPI; authenticated static capabilities;
- one fake/injected API request returning both exact CLIP and BLIP3 PNG inputs,
  final objects, one-to-one input records, fixed names and JSON/ZIP manifest/
  member media type, byte size and SHA-256 parity;
- two consecutive/repeated API configs with different fractions and no state
  leakage; and
- coexistence with raw-SAM2 and labelled visualization artifact accounting.

Tests may add focused files instead of concentrating all assertions in
`test_mask_views.py`. Use only generated arrays and mocked processors for the
deterministic gates. Preserve the full existing suite and update obsolete old-
compositor/name assertions rather than leaving contradictory compatibility
tests.

### 8. Documentation truth

Correct all current docs to the implemented literal names and top-level
capability location. At minimum remove the stale old BLIP name claims in
`TESTING.md` and `docs/OUTPUT-PARITY.md`, then search every current document for
old numeric-only candidate/question filenames, untouched-context semantics,
minimum-128 semantic source crops and nested-under-raw-SAM2 capability claims.
Historical material may remain only when clearly historical.

Documentation must state the two-phase admission seam, bounded dilation
implementation/maximum, right-target equality after resize, safe helper
semantics, pixel-isolation versus semantic-accuracy scope and the completed
live qualification. Run the documentation checker.

## Non-goals

- No SAM2 setting/profile/model change, semantic accuracy benchmark, solar
  fixture/polygon work or recall/precision claim.
- No CLIP scoring/two-view weighting, BLIP decision parsing, prompt/question
  behavior, model identity/revision, residency, device/dtype/cache or generation
  limit change.
- No new fill mode, dependency, renderer, artifact budget increase, schema
  version bump, endpoint, client path/destination or persistent request data.
- No change to raw-SAM2 rendering, labelled final rendering, post-filter
  diagnostics, YOLO, identity PNG or geometry beyond exact compatibility tests.
- No release/tag/publish, network/firewall/VPN, system driver/CUDA or unrelated
  service/process/device mutation.

## CPU/static verification before restart

- Run focused pure-builder/dilation, CLIP, BLIP3, core/resource, validator,
  envelope/schema/capabilities/OpenAPI/API and legacy tests covering every
  correction and exact negative seam above.
- Run `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`
  and report pass/skip/count/coverage honestly.
- Run Ruff format/check, compileall, documentation checker, affected shell
  syntax/systemd verification, `git diff --check`, wheel/sdist build, release
  artifact verification, archive/tracked-tree secret scans against the existing
  baseline and `twine check`.
- Push the corrected implementation/control head and require all seven
  current-head CI/CodeQL checks present and successful before restart. Keep PID
  `513853` (or its unchanged naturally running successor only if an external
  failure occurs and is disclosed) active/ready during this phase. Do not
  restart speculatively or modify the unit/env.

## Complete bounded live qualification

After every gate above passes, this 017-b order authorizes exactly one
controlled restart of only `zap-it-lan.service` to load the corrected 017-b
implementation. Before restart reverify index/UUID/PCI/name/VRAM/process,
driver/CUDA/PyTorch, `/dev/shm`, listener/unit/port, environment mode/digest and
capacity without printing a key. Start no second model process.

After readiness returns:

1. Require health/readiness 200, missing/wrong inference credentials 401,
   authenticated capabilities/metrics 200, docs/OpenAPI 404. Verify the live
   capabilities JSON has one declared top-level candidate-view policy, exact
   names/ranges/defaults and no nested raw-SAM2 duplicate or sensitive fact.
2. Reuse one already-authorized ignored local fixture only in RAM or the
   mode-0700 `/dev/shm` workspace. Send one bounded L3 ZIP request that executes
   SAM2, CLIP and at least one BLIP3 question, enables both CLIP and BLIP3 debug,
   keeps tested candidates serializable and explicitly selects zero-fill
   candidate views. It must return final objects and at least one exact debug
   PNG from each stage in the same response.
3. Verify literal fixed names, one-based source/question IDs, zero-based
   filtered indices, one-to-one records, JSON/ZIP manifest/member media type,
   size/SHA and lossless decode. Reconstruct masks/support from object RLE and
   metadata and prove the exact model inputs: CLIP outside `D` zero; BLIP left
   outside target zero; right outside support zero; right target pixels equal
   left; contour outside target and inside support. Do not print/persist image
   content.
4. Run A/B/A on the same PID with different CLIP and BLIP3 fractions and debug
   enabled. Require both stages' effective values/radii/input hashes change and
   restore A/B/A, with stable model initialization/process/listener. Semantic
   labels/answers are not an acceptance gate.
5. Record bounded latencies, response/artifact/candidate counts and sizes, RSS,
   GPU peak/free memory and workspace cleanup. Require no failed HTTP request,
   OOM, timeout, second process/restart, credential/path leak or residue.
6. Recheck health/readiness/auth/capabilities/metrics, one listener/GPU process,
   `NRestarts=0` for the new PID, sanitized journal, unchanged environment
   digest and empty workspace. Leave the corrected newest service enabled,
   active and ready at `10.8.132.76:17891`.

Disclose any failure or retry. If the full restart/readiness/request/A-B-A/
cleanup sequence cannot be completed, leave the service in its safest ready
state and report PARTIAL; do not claim completion or silently spend a second
restart.

## Acceptance criteria

1. Runtime, typed records, capabilities, OpenAPI, docs, tests and ZIP bytes use
   the exact literal candidate/question filename templates.
2. Candidate views are one declared top-level typed capability, independent of
   raw-SAM2 debug, with no dynamic extras.
3. No exported or production BLIP3 composition helper shows an untouched
   rectangle; right target pixels equal left target pixels after resize.
4. Exact circular dilation is correct and completes the radius-512
   representative resource test within ordered time/RSS bounds.
5. CLIP admission occurs before CLIP and exact post-CLIP BLIP admission occurs
   before QA; applicable rule/resource negative tests prove zero forbidden model
   calls/partial artifacts.
6. Strict configuration, final identity, L0-L3 manifest, API JSON/ZIP,
   OpenAPI/capabilities and resident A/B/A tests close the 017-a evidence gaps.
7. All focused/canonical/static/package checks and all current-head CI/CodeQL
   checks pass with an exact bounded diff and honest docs.
8. One live request returns final output plus exact CLIP and BLIP3 inputs, and
   corrected-head live A/B/A passes for both stages on one stable process.
9. The corrected PR-head service is left enabled, active and ready with one
   assigned-GPU process, unchanged credential digest and empty workspace.

The strongest reason not to merge remains that debug/model-input identity can
look correct while a different resident request setting, interpolation path or
late resource decision reaches the actual model. Answer it with direct
processor/QA byte-identity seams, right-target equality, two-phase zero-call
resource negatives and corrected-head live CLIP+BLIP A/B/A—not documentation or
visual inspection.

## Documentation/provenance

Update only the current docs required for truthful correction and live result.
Do not change model/revision/license/dependency claims without evidence. Keep
`zap-it.v1`; this remains an additive contract on an unpublished candidate.
Record exact changed documents and stale-search patterns/results.

## Security/resource/protected-host constraints

- All Objective-017 security, privacy, fixed-name, uploaded-config, RAM/
  `/dev/shm`, response/artifact/question/token/deadline and one-process laws
  remain binding.
- Uploaded values never control paths, filenames, models, weights, revision,
  cache, device, dtype, network, code, point grids, residency or destinations.
- Do not expose keys, source pixels, YAML/prompts/questions, operator paths,
  exception strings or physical topology in errors/logs/Git/report/capabilities.
- Use only physical GPU0/assigned UUID above, exposed alone as logical
  `cuda:0`; fail closed, no fallback/heuristic. Touch no other resource.
- No unit/env/firewall/route/VPN/driver/system CUDA/unrelated mutation. Coding
  never merges, releases, tags, publishes or enables auto-merge.

## Deferred human adjudication

- Decision: NONE

## GitHub publication

Amend only PR #73 on the existing branch. Commit the exact `oap/active` and
immutable `oap/orders/017-b-close-candidate-view-contract-and-live-proof.md`
with the bounded correction implementation/tests/docs. Push all non-report
state and record a literal 40-hex implementation SHA.

Create exactly `oap/reports/017-b-report.md` as the final report-only SELF child.
It must change only that report, have the implementation SHA as sole parent, be
pushed, have all seven checks green, and be remotely verified before one exact
response FIFO `OK`. Make no later mutation. Coding does not merge.

## Required immutable report evidence

- Exact 017-b/PR/base/branch/start/implementation/report SHAs, commit topology,
  changed paths and bounded diff; current-head check run/job URLs/statuses.
- Finding-by-finding correction evidence, exact test names/commands/counts,
  radius-512 time/RSS, full suite/coverage/static/docs/build/package/secret scan.
- Runtime/capability/OpenAPI/name parity and independent stale-search result.
- Before/after service/GPU/listener/RSS/readiness/auth/env-digest/workspace facts,
  restart count and any failed attempt; live request/A-B-A timing/count/hash/
  pixel-invariant summaries without content or credentials.
- Explicit proof service PID loaded the 017-b corrected behavior, including
  request-local CLIP debug true/false and exact-name evidence.
- Deferred human adjudication `NONE`, CRITICAL unchanged, no model/dependency/
  credential/protected-resource/unrelated change.
- Strongest reason not to merge and evidence answering it; honest blockers.
  COMPLETE only if the corrected newest service is left enabled/active/ready.
