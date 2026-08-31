# OAP Work Order 022-b — close runtime boundaries and exact live proof

## Objective and decision

Continue Objective 022 in the existing PR. Do not create another PR. Correct
the two independently verified runtime gaps in 022-a, add the missing exact
configuration regression proof, and complete the exact authenticated tomato
qualification with a labelled result image and stage-by-stage evidence.

Decision: the BLIP3 question count is an operator-owned per-request workload
limit, not a request-YAML control. Replace the hard-coded 32 with an immutable
startup setting whose default is 256 and whose accepted operator range is
1..256. This matches the existing maximum final-object bound and permits at
most one canonical routed BLIP3 rule per routed candidate. It remains bounded,
reported in capabilities/runtime documentation, and may be lowered by the
operator. Do not expose it to uploaded YAML and do not silently drop routed
candidates. Legacy configurations capable of scheduling more than the
effective limit must still fail before BLIP3 generation with a structured
`resource_limit` error.

## Deferred human adjudication

- Decision: NONE

## Reconciled authoritative state

- Repository: `ulfe-lmi/slaif-zap-it`.
- PR: #78, open, base `main`, branch
  `oap/022-a-canonical-clip-multiprompt`.
- Accepted base: `d341a3c4ba47b71d10d70682771b315041dcbcb8`.
- 022-a implementation: `beb4830035696d50c1c248d850940e44e67f744e`.
- Immutable partial report/PR head:
  `b2d105bd507bdc2bfb06bcb068be057576301e8a`.
- All seven required checks are green on that report head. Do not rewrite,
  squash, amend, or delete either existing commit or the immutable report.
- The 022-a live service is enabled, active and ready as PID 685637 with
  `NRestarts=0`, on private `10.8.132.76:17891`, using only operator-assigned
  physical GPU 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`.
- Exact image remains
  `demos/tomato/2022-07-22-16-25-44-48.jpg`, 1280x720, 358454 bytes, SHA-256
  `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.
- The 022-a negative live request correctly returned HTTP 400
  `invalid_config`, measured token count 80, allowed limit 77, before engine
  execution.
- The one 022-a exact positive request reached the BLIP3 stage and returned a
  122-byte HTTP 413 `response_too_large` envelope after 16.077 seconds. Source
  reconciliation identifies the exact 38-character adapter message as
  `BLIP3 question resource limit exceeded`; the resident loader hard-codes
  `max_questions=32`, and the verifier rejects the planned routed question
  count before generation. This is a workload-limit classification/configuration
  gap, not response serialization or artifact overflow.
- `/dev/shm` is the request-artifact medium. The private bearer key and service
  environment remain operator secrets and must never be printed, copied,
  rotated, committed, placed in argv, or included in reports.

Before mutation, independently refresh the PR head/checks, service PID/state,
listener, all GPU index/UUID/PCI/name/VRAM/process facts, `/dev/shm`, environment
file mode/digest, exact fixture identity, and current `CRITICAL.md`. Stop on a
contradiction affecting authority or safety. Ordinary implementation findings
do not justify changing objective or PR.

## Required corrections

### 1. Typed prompt errors must never escape as HTTP 500

`modules/classifier/clip.py` deliberately performs tokenizer validation again
inside the model adapter as defense in depth. `live_engine_callable` currently
does not translate `ClipPromptValidationError`, so a bypass, race, or future
preflight mismatch could reach the generic application exception handler and
become HTTP 500.

- At the resident adapter/service boundary, translate
  `ClipPromptValidationError` to `ServiceError(code="invalid_config")` with the
  same bounded sanitized details used by the preflight path.
- Preserve the original typed exception as the cause; do not echo prompt text,
  tokenizer internals, model paths, or stack traces.
- Do not convert unrelated model/runtime exceptions to client errors.
- Add a service-level test that deliberately bypasses or disagrees with the
  outer preflight, makes the adapter defense reject, and proves HTTP 400
  `invalid_config`, exact bounded details, and no HTTP 500. Also prove an
  unrelated exception retains its existing 500 behavior.

### 2. Operator-owned BLIP3 question capacity

- Add an immutable `ServiceSettings` field for the per-request BLIP3 question
  limit, default 256, operator environment key
  `SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS`, accepted range 1..256.
- Parse and validate it at startup. Invalid, zero, negative, non-integer, or
  over-256 operator values must fail closed before model loading/listening.
- Pass the exact effective setting into the resident loader/holder. Never read
  it from request YAML and never mutate it between requests.
- Keep `max_new_tokens=32` operator-controlled/fixed exactly as today; do not
  expand generated text, model identity, device, paths, or any request control.
- Expose the effective question limit in authenticated capabilities and
  relevant runtime metadata without exposing environment contents.
- If a request plans more questions than the effective setting, return HTTP
  413 `resource_limit`, not `response_too_large`, before BLIP3 generation. The
  sanitized details must include at least `planned_questions`,
  `allowed_limit`, the controlling field name, and bounded admissible
  alternatives (for example a deterministic CLIP routing candidate cap or
  stricter routing parameters). Do not expose prompts, answers, image content,
  host data, or candidate pixels.
- No candidate may be silently truncated, clamped, reordered, or skipped to fit
  the limit.
- Canonical routing permits one chosen BLIP3 rule per routed candidate. Prove
  that up to 256 routed candidates can be planned under the default and 257 is
  rejected by the existing object/question bounds without actually running a
  model. Legacy multi-rule scheduling remains bounded by the same total
  question limit.

### 3. Exact configuration regression fixture

Add a repository-owned test fixture containing byte-for-byte the complete YAML
from the immutable 022-a order appendix and the human request. A natural path
is `tests/fixtures/configs/ripe-tomato-multiprompt.yaml`. This public synthetic
configuration is deliberate test data; it must not contain credentials,
addresses, paths, or generated answers.

The automated test must load the exact fixture through the public hostile-YAML
parser and fake/API path, not reconstruct only its prompt counts. Prove:

- `ripe_tomato=32`, `foliage=15`, `stem_or_vine=15`,
  `greenhouse_structure=20`, `background=15`, total 97;
- precisely five semantic class-score keys in configured order;
- routing target exactly `ripe_tomato`;
- all arrays remain independent prompt embeddings and scalar comma/newline
  behavior from 022-a remains covered;
- canonical BLIP3 `falsecategory` is present and accepted;
- every shipped example/fixture passes the same public schema and current
  operator-limit validation expected for its stage.

Do not make tests depend on `/tmp`, `/dev/shm`, a live service, network, model
downloads, or the photograph. Keep the exact photograph hash/dimension proof in
the live qualification only.

### 4. Documentation and schemas

Synchronize every affected maintained source, generated OpenAPI/capability
description, README/runbook/datasheet/config/API/algorithm/testing material.
Document:

- canonical label values are `string | ordered non-empty array[string]`;
- each array item is separately tokenized/embedded and per-class max aggregation
  chooses the score, with deterministic lowest-index tie evidence;
- scalar commas/newlines are not separators;
- structural/token limits and HTTP 400 error details;
- adapter validation is defense in depth and remains a client error;
- `falsecategory` is required for canonical routed BLIP3 rules;
- BLIP3 question capacity is startup/operator-owned, default/max 256, unit
  questions/request, stage BLIP3 planning, never uploaded YAML;
- exceeding it is typed `resource_limit`; response assembly overflow alone is
  `response_too_large`;
- the limit interacts with permissive routing and any explicit deterministic
  routing candidate cap without silent truncation.

Do not claim the exact live test passed until the evidence below exists.

## Required deterministic verification

At minimum run and report:

1. Focused Objective 022 and new boundary/capacity tests.
2. All adjacent YAML/API/schema/capability/CLIP/routing/BLIP3/runtime/settings
   tests affected by the change.
3. Full CPU suite with coverage, recording exact pass/skip totals and coverage.
4. Ruff format/check, compileall, documentation checker, diff checks.
5. Wheel/sdist build, audit, tracked-tree scan, twine check, member comparison,
   and isolated installed-package JSON/ZIP smoke parity.
6. Exact test fixture count/order/route assertions.
7. A no-model test proving 32 and 256 are accepted as configured operator
   capacities, invalid operator values fail closed, and planned over-limit
   workloads yield the structured `resource_limit` envelope.
8. The adapter-defense HTTP-400 test described above.

No external download or second model process is authorized.

## Live qualification

Only after all CPU/package gates pass and implementation CI is fully green:

1. Reconcile all device/service/secret/tmpfs/fixture facts again. Preserve
   every unassigned device and unrelated workload.
2. Perform exactly one controlled restart of `zap-it-lan.service` to load the
   corrected implementation/default operator limit. Do not alter the unit,
   environment file, bearer key, host firewall/network, driver, CUDA, model
   cache, port, or request deadline. Confirm PID changed once, `NRestarts=0`,
   readiness, the private listener, assigned UUID, and no fallback.
3. Re-run the already-proven over-77-token negative request once. It must return
   HTTP 400 `invalid_config` before engine work with safe class/index/count/limit
   details and unchanged GPU allocation.
4. In a newly mode-0700 `mktemp -d` beneath `/dev/shm`, materialize the exact
   immutable 022-a appendix YAML as `ripe-tomato-multiprompt.yaml`. Verify its
   parsed prompt counts and digest. Do not use ext4 `/tmp` for request data.
5. Read the existing key silently into a shell variable, submit exactly one
   authenticated verbosity-3 ZIP request against the private service using the
   exact repository image and exact YAML, write the body in the tmpfs directory,
   immediately unset the variable, and record HTTP status/elapsed time without
   printing secrets or request text.
6. The request must return HTTP 200. If it does not, capture the sanitized
   error code/details and stop live attempts: no retry, no hidden clamp, no YAML
   mutation, no second restart.
7. Validate the ZIP safely before extraction (fixed safe member names, no
   traversal/symlink/duplicates, count/size budgets). Verify manifest/member
   hashes and sizes, exact five-class CLIP score vectors, prompt counts
   32/15/15/20/15 and total 97, route target `ripe_tomato`, source candidate IDs,
   BLIP3 question/effective prompt/answer/final-label evidence, final labels and
   11..199 bounding-box dimensions, stage timings/counts, warnings, omitted
   artifacts, and candidate-view rejection count.
8. Require the ZIP to contain the final
   `final-labelled-ripe-tomatoes` labelled PNG. Preserve the ZIP and that PNG in
   the mode-0700 tmpfs result directory for strategic review. Report absolute
   paths, SHA-256 and byte sizes; never derive filenames from prompts.
9. Independently inspect the source photograph and final labelled image. Report
   bounded factual observations: prominent ripe tomatoes visible to a human;
   SAM2 proposals; geometry survivors; CLIP-routed candidates; BLIP3 verified
   and accepted; final objects; obvious misses; false positives; fragmented
   masks; merged multi-object masks; and candidates rejected because the BLIP3
   contextual crop could not contain support/contour. Distinguish manifest
   counts from human visual judgment. Do not claim semantic perfection merely
   from HTTP 200.
10. Recheck service readiness/auth boundaries/listener/PID/restarts/GPU/processes,
    `/dev/shm`, environment digest and absence of request data from persistent
    service storage. Leave the corrected private keyed service running.

The requested visual judgement may use deterministic local image inspection;
AI image generation is neither needed nor authorized because the exact real
fixture already exists.

## Security, privacy, and non-goals

- Keep one image plus hostile YAML, RAM/tmpfs-only request handling, bounded
  response behavior, stable candidate identity, deterministic outputs, and
  service-safe fixed artifact names.
- Uploaded YAML cannot control model/device/cache/path/network/auth/workload
  ceilings or artifact destinations.
- Never print or persist the bearer key, raw answers, uploaded YAML, image bytes,
  or prompt text in OAP reports/journal/GitHub metadata. The committed public
  fixture is the sole deliberate prompt-text exception.
- No dependency/model/revision/precision/residency/renderer/geometry/routing
  semantic change beyond this order.
- No service environment mutation is required: the safe default question limit
  is supplied by code. Do not raise response sizes, object count, SAM2 limits,
  timeouts, or artifact budgets to force the live result.
- No merge, auto-merge, release, public bind, firewall, driver, or unrelated
  service mutation. Coding never merges.

## Acceptance and publication

This round succeeds only when all of the following are true:

- both prompt-validation paths produce typed HTTP 400, never prompt-caused 500;
- the 1..256 operator-owned BLIP3 capacity is validated, documented and applied
  without request state leakage or silent pruning;
- the exact full YAML fixture is covered through the public parse/API seam;
- all required local/package/CI gates pass on the exact implementation head;
- one corrected live restart and one exact positive request produce HTTP 200,
  a valid ZIP, complete stage evidence and the final labelled PNG;
- independent visual observations are honestly reported;
- the private keyed service remains running and ready on only the assigned GPU;
- scope/security/non-goals hold and no critical-register append is needed.

Commit implementation/tests/docs as one normal 022-b implementation commit on
the existing branch and push it to PR #78. Wait for every required check on the
exact implementation head; none may be missing, pending, failed or cancelled.
Then append immutable `oap/reports/022-b-report.md` as a report-only SELF commit
whose sole changed path is that report and whose parent is the implementation
head. Push it, wait for every required report-head check, verify remote head and
report-only shape, send exactly `OK` on the response FIFO, make no later
mutation, and exit. The report must include changed files, migration notes,
commands/results, exact SHAs/check URLs, live counts, visual findings, tmpfs ZIP
and PNG paths/hashes/sizes, retained compatibility, strongest reason not to
merge and its answer, and all safety confirmations.
