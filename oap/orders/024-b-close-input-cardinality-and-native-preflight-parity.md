# OAP Work Order 024-b — close input cardinality and native preflight parity

## Objective

Amend Objective 024 PR #88 in place. Preserve the implemented thin,
stateless, non-streaming `POST /v1/responses` facade, official-SDK behavior,
public projection, canonical annotated PNG, documentation, and all native
`POST /v1/completions` behavior while correcting two independently verified
merge-gate defects:

1. missing Responses image/configuration inputs do not currently produce their
   required distinct typed errors; and
2. the shared inference extraction makes both HTTP surfaces execute the host
   resource admission preflight twice, whereas the pre-024 native route
   executed it once.

This is a narrow same-PR correction. Do not redesign either API, alter the
vision pipeline, change inference semantics, change the public projection or
renderer, tune models/configuration, or modify `slaif-api-gateway`.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote `main` remains
  `32812032781c5d7daf54d5b7586b3c01d3270c48`.
- Amend only open PR #88, branch
  `oap/024-a-openai-responses-compatible-facade`, titled
  `Objective 024: OpenAI Responses-compatible facade`.
- The reviewed remote head is immutable 024-a report-only SELF commit
  `e1d80512252262048b4409ad7b54bf20d53b3739`; its first parent is final 024-a
  implementation commit `3e6e44aed3a4a2dc0708a8032aa4e244fb4ddd89`.
- All seven CI/CodeQL checks on that head are green, the worktree is clean, and
  GitHub reports the PR mergeable/CLEAN. Green CI is insufficient because the
  defects below are visible in the reviewed implementation and its tests.
- Continue on the same branch and PR. Commit the published 024-b order/active
  transcript and corrective product changes after the immutable 024-a report.
  Do not modify or replace `oap/reports/024-a-report.md`.
- Do not merge, enable auto-merge, rewrite history, amend existing commits, or
  touch another PR.

Refresh the current branch, PR, `CRITICAL.md`, listener/service revision,
assigned GPU identity/process state, `/dev/shm`, and official SDK pin before
mutation. Preserve the running service until the corrected implementation head
is ready to qualify; then restart it so the newest product commit is loaded and
leave it running on the existing authorized private-LAN listener.

## Independent review evidence

### Finding 1 — required missing-input errors are unreachable

`src/service/responses.py::parse_responses_request` rejects every content list
whose length is not exactly two before it counts image/file parts. A request
containing only the valid `input_file` therefore returns generic
`invalid_config`, not `missing_image`; a request containing only the valid
`input_image` returns the same generic code, not `missing_config`. The later
dedicated `missing_image` and `missing_config` branches cannot be reached for
those ordinary missing-part cases.

The current cardinality regression is falsely weak: it parameterizes a
one-file case and duplicate image/file cases but asserts only HTTP 400 and the
generic error type, never the exact code. The Objective 024 contract requires
missing image, duplicate image, missing config, and duplicate config to remain
distinguishable by stable code/status/param before inference.

### Finding 2 — shared extraction duplicated resource admission

Before Objective 024, `/v1/completions` called
`check_request_resources(settings_local)` once, before content-length and body
processing. The current implementation retains that call and adds another
unconditional call at the start of `run_shared_inference`. `/v1/responses`
likewise calls the preflight before reading JSON and again through the helper.

The check reads live host RAM and tmpfs capacity. Calling it twice is not merely
an internal pure calculation: host state can change between observations and a
second decision can change the native API outcome after its request body was
accepted. This violates the explicit requirement that `/v1/completions`
behavior remain unchanged and needlessly duplicates admission work.

## Required corrective implementation

### 1. Make input cardinality errors exact and reachable

- Keep the accepted request contract exactly one user message containing
  exactly one `input_image` and exactly one `input_file`. Do not accept any new
  input shape.
- For a list containing a valid file but no image, return HTTP 400 with code
  `missing_image` and a bounded content-path `param`.
- For a list containing a valid image but no file, return HTTP 400 with code
  `missing_config` and a bounded content-path `param`.
- For multiple images, return HTTP 400 `duplicate_image`; for multiple config
  files, return HTTP 400 `duplicate_config`.
- An empty content list must deterministically report a missing-part code; use
  `missing_image` as the documented first-required-part precedence. A list
  containing unsupported part types must still return the existing explicit
  unsupported-field/source failure rather than being misreported as missing.
- Inspect/count only within the already bounded JSON body. Do not decode image
  or YAML bytes and do not enter inference until cardinality and exact part keys
  pass.
- Preserve strict rejection of extra messages, fields, content parts, URLs,
  file IDs, unsafe filenames, MIME mismatches, state, streaming, and tools.
- Add sanitized `_ERROR_MESSAGES` entries for all four cardinality codes so the
  OpenAI-shaped message is specific without echoing request data.
- Keep the generated OpenAPI schema strict at exactly two content parts. The
  runtime may inspect an invalid-length list only to classify its rejection;
  it must not create schema/runtime acceptance drift.

Add table-driven route tests that assert, for each missing/duplicate case:
exact HTTP status, exact error code, `invalid_request_error`, bounded non-null
`param`, canonical four-field error shape, and zero engine calls. Include the
empty-list precedence and at least one unsupported-part case. Add a direct
invalid-YAML route regression proving HTTP 400 `invalid_config`, OpenAI error
shape, and zero engine calls.

### 2. Restore exactly-once resource admission

- Each `/v1/completions` and `/v1/responses` request must call
  `check_request_resources` exactly once.
- Preserve the pre-body order on both routes: authentication where applicable,
  resource admission, content-length/body bounds, parsing, decode/config
  validation, readiness, shared gate, inference, serialization.
- The shared inference helper must not repeat a preflight already performed by
  its route. Use an explicit, hard-to-misuse helper contract or parameter;
  avoid hidden mutable state.
- Preserve every existing failure code/status/envelope for native
  `/v1/completions`. Do not move its sole resource check later in the request.
- Remove any newly dead helper left by the extraction, such as an unused
  `remaining_budget`, only if this is a direct consequence of the correction.

Add focused monkeypatched tests proving one and only one resource-admission
call for a successful native completion and a successful Responses request.
Also prove a failing first resource admission prevents body parsing/inference
and retains the correct native versus OpenAI error envelope. Existing private
completion byte/parity fixtures must remain unchanged.

### 3. Preserve the accepted Objective 024 contract

Do not change:

- the exact successful Responses shell, including `tools: []`, omitted usage,
  unique protocol IDs/timestamps, deterministic `zap-it.public.v1` output text,
  and optional standard `image_generation_call`;
- official `openai==3.7.0` request serialization and response parsing;
- the shared typed engine, gate, one worker, model residency, YAML validator,
  renderer/PNG encoder, public/private evidence boundary, metrics dimensions,
  authentication, bounds, or gateway-dependency statement;
- any `/v1/completions` request/response/error/artifact semantics;
- existing live network/GPU/key policy.

Do not delete or weaken a test to pass this order. Make only directly affected
documentation corrections if the exact missing/duplicate precedence needs to
be stated; otherwise avoid editorial churn.

## Required verification

Use the existing environments without downloading models or changing operator
dependencies.

At minimum run and report:

1. focused Objective 024/cardinality/resource-preflight tests;
2. the complete CPU/fake suite;
3. maintained formatting, lint, documentation, build, release-artifact,
   twine, secret, banned-file, and tracked-sensitive-name gates;
4. all required GitHub CI/CodeQL checks on the final implementation commit;
5. official SDK local qualification against the restarted live service; and
6. an authenticated native `/v1/completions` smoke after restart.

For the live proof, independently record the literal implementation commit,
service PID/start time/restart count, exact listener, health/readiness/auth
behavior, assigned physical GPU0 index/UUID/name/VRAM and process ownership,
unassigned-device preservation, `/dev/shm` mount/capacity/permissions, bounded
request-root contents, SDK response/object/image-call/PNG facts, and native
completion status/object/artifact facts. Do not print or persist the bearer.
Retain only bounded content-free evidence. Leave the newest corrected service
running.

## Acceptance criteria

1. Missing image, missing config, duplicate image, and duplicate config each
   return their exact stable OpenAI-shaped HTTP 400 code before inference.
2. Empty content and unsupported parts have deterministic documented rejection
   precedence; accepted cardinality remains exactly one image plus one YAML.
3. Invalid YAML is a typed HTTP 400 `invalid_config`, never HTTP 500.
4. Each HTTP surface performs resource admission exactly once and at its
   original/pre-body boundary.
5. Native completion request/response/error/artifact behavior remains unchanged.
6. The official SDK still obtains deterministic public JSON and exactly one
   decodable canonical annotated PNG when requested.
7. No-tool behavior, projection fields, renderer bytes, IDs, authentication,
   bounds, statelessness, private-artifact exclusion, and gateway non-goals are
   unchanged.
8. The complete suite and all required checks are green on the literal final
   implementation commit.
9. The corrected live service is running on the authorized private-LAN endpoint
   and the report contains bounded evidence sufficient for independent review.

## Report and SELF contract

Write immutable `oap/reports/024-b-report.md`. It must identify PR #88, base,
024-b implementation commit, all commits added this round, changed files and
bounded diff, exact tests/checks with results, the four cardinality cases,
exactly-once preflight evidence, private parity evidence, official SDK/live
qualification, listener/GPU/service/tmpfs facts, documentation impact,
security/privacy/resource review, strongest reason not to merge and answer,
and `Deferred human adjudication: NONE`.

After product changes, tests, push, live qualification, and report content are
complete, commit only `oap/reports/024-b-report.md` as final SELF. The SELF
commit's first parent must be the reviewed 024-b implementation head. Push it,
wait for all required checks on SELF, then signal the response FIFO with exact
payload `OK`. Coding must not merge.
