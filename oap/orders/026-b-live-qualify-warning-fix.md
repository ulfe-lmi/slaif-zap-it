# OAP Work Order 026-b — live-qualify the warning fix

## Objective

Amend Objective 026 PR #90 in place without changing product or test code.
Preserve the accepted one-line sanitizer fix and all green 026-a evidence, then
complete the sole missing acceptance gate: run the corrected private-LAN
service, obtain one successful bounded `POST /v1/responses` result containing
both exact unsplit warning strings, prove native preservation, and leave the
corrected service healthy, ready, and running.

The 026-a live request returned HTTP 503 immediately after readiness. The live
log independently showed private-LAN traffic around the same interval, and the
service uses a deliberate one-active-request, zero-queue gate whose
`service_busy` result is HTTP 503 with `Retry-After`. A single 503 did not test
the sanitizer. The 026-a order required rollback after that failed request, so
coding correctly restored merged main. This continuation permits bounded,
code-aware retries for only the transient `service_busy`/`not_ready` cases.

Do not alter source, tests, thresholds, service admission, retry policy, or
configuration to make live qualification pass.

## Deferred human adjudication

- Decision: NONE

This is a normal live-qualification retry after a safe transient admission
failure. No material architecture or trust-boundary decision is unresolved.

## Authoritative state and PR mode

- Remote `main` remains
  `90c4b4923e4924dcffed185a0bf54ffeea5f7eb4`.
- Continue only PR #90, `Objective 026: fix Responses warning sanitization`, on
  branch `oap/026-a-fix-responses-warning-sanitization`.
- Current remote PR head is immutable 026-a report-only SELF commit
  `bcb3292b83355e6f896e78a5cefe95100bdbbed7`; its only parent is accepted
  implementation commit `074179841fca59bb8468d4faa89ee3cd78e921b0`.
- All seven implementation-head and final-SELF CI/CodeQL checks are successful;
  GitHub reports PR #90 MERGEABLE/CLEAN.
- The accepted implementation diff remains exactly one separator change in
  `src/service/responses.py`, focused additions in
  `tests/test_objective_024.py`, and the OAP transcript. Focused tests passed
  44/44; the full suite passed 975 with one explicit GPU-marker skip and 82.87%
  coverage.
- Commit the 026-b active/order transcript after the immutable 026-a report.
  Make no product, test, documentation, dependency, deployment, or prior OAP
  file change.
- Do not merge, enable auto-merge, rewrite history, amend commits, create a new
  PR, publish a release/tag/package, or touch another repository.

Read the coding constitution, communication contract, exact 026-a order/report,
this order, current branch diff, service/gate code relevant to the 503, and
current `CRITICAL.md` before action.

## Required no-change verification

Before service mutation, prove and report:

1. remote/main/head and exact PR identity above;
2. `bcb3292…` changes only `oap/reports/026-a-report.md` relative to
   `0741798…`, with the correct parent;
3. current product/test bytes match accepted implementation `0741798…`;
4. `git diff --check origin/main...HEAD` passes;
5. `.venv/bin/pytest -q tests/test_objective_024.py` still passes 44/44; and
6. there is no need to rerun a redundant local full suite because 026-a ran it
   after the final product bytes, both implementation and SELF CI matrices are
   green, and 026-b changes no product/test bytes. Final 026-b CI must still run
   and pass normally.

If any product/test byte differs from `0741798…`, stop and report rather than
silently modifying it under this qualification-only round.

## Live-service procedure

### 1. Reconcile and restart exactly once

The safe rollback service currently runs merged-main product code as PID
`853653`, started `Thu Sep 3 00:05:19 2026`, on exact listener
`10.8.132.76:17891`, health/readiness 200/200. It is the only process on
assigned physical GPU index 0, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
GeForce RTX 3090, 24576 MiB, driver `610.43.02`, using about 10084 MiB.
Process visibility remains `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`, with the expected UUID pinned.
`/dev/shm` is a 12 GiB tmpfs with about 9.7 GiB free.

Immediately re-verify all those facts, every GPU/process, exact listener,
environment, service state, and `/dev/shm`. Protect every unassigned device and
unrelated workload. Use only the repository launcher and existing operator
environment to restart once from current PR head, whose product bytes equal
`0741798…`. Do not print the bearer or change firewall, routes, VPN, drivers,
CUDA, model cache, credentials, gateway, or system-wide state.

Wait for complete readiness, including resident model loading; do not treat
health-only or readiness 503 as ready and do not submit inference during load.

### 2. One successful request, with bounded admission retries

Construct one small synthetic image in memory and one safe inline YAML config
that simultaneously contains:

```yaml
postsam2processing:
  debug: true
diagnostic_artifacts:
  stages: [sam2]
```

plus the existing valid minimal CLIP/task content needed by the live service.
Use the canonical JSON Responses input shape, no tool declaration, and the
operator-supplied bearer without printing or retaining it.

The acceptance target is one successful request. If an attempt returns 503:

- parse only the bounded safe OpenAI error `code` and `Retry-After` header in
  memory;
- if the code is `service_busy`, honor at least `Retry-After` (and never retry
  more frequently than once per five seconds), then retry the same request;
- if the code is `not_ready`, return to readiness polling before retrying;
- for any other code/status, stop and follow rollback; and
- bound the entire retry window to ten minutes and at most 60 attempts. Do not
  queue, parallelize, flood, change queue depth, kill/interfere with the other
  client, or restart again during this window.

For each failed attempt retain/report only attempt number, HTTP status, safe
error code, and retry delay. Do not retain or report request content, raw error
body, bearer, image, YAML, prompt, or model output.

On HTTP 200, decode the assistant `output_text` in memory and require all of:

- `schema_version == "zap-it.public.v1"`;
- top-level `warnings` contains the exact complete entries
  `debug flag postsam2processing.debug ignored at verbosity below 3` and
  `diagnostic_artifacts selection is valid but not applied below verbosity 3`;
- neither character-spaced equivalent is present;
- no-tool metadata remains `tools: []`, `tool_choice: "none"`,
  `parallel_tool_calls: false`, and no `image_generation_call` exists; and
- no persistence of the response or request content occurs.

Retain/report only HTTP status, schema-version equality, exact-warning equality
booleans, character-spaced-absence booleans, tool metadata, bounded counts/
timing, and safe process/resource facts. Quoting the two already-public expected
warning constants is permitted; do not retain the complete projection.

### 3. Native and post-run checks

After the successful Responses proof, run one bounded authenticated native
`POST /v1/completions` smoke using an established repository helper and report
only status/shape/count/size facts. Require HTTP 200 and the unchanged native
contract.

Then require health/readiness 200/200, unauthenticated Responses 401, exact
listener ownership, only the assigned GPU/process, expected environment, bounded
`/dev/shm`, and a log-hygiene scan finding no bearer, data URL, input filename,
raw request/response content, or traceback.

Leave the corrected service running. If readiness, bounded retries, warning
equality, native smoke, or post-run safety checks fail, restore merged main
`90c4b49…` with the repository launcher, leave that service healthy/ready, and
report FAILED. Do not improvise code changes in 026-b.

## Preservation and non-goals

- No change to `_bounded_warning` beyond already accepted `0741798…` bytes.
- No source, tests, docs, config, schema, endpoint, model, inference, warning,
  renderer, object, Responses metadata, artifact, bounds, auth, logging,
  deployment, dependency, or gateway changes.
- No test expectation weakening, prompt/threshold tuning, queue-depth change,
  extra worker, public exposure, or external client interference.
- Do not modify activated 026-a order/report or prior history.

## Acceptance criteria

1. PR product/test bytes remain exactly those accepted in `0741798…`.
2. Focused tests remain 44/44; all final 026-b CI/CodeQL checks pass.
3. A corrected live Responses request returns 200 within the bounded retry
   policy and contains both exact complete warnings without character spacing.
4. No-tool Responses metadata/output remains unchanged.
5. Native `/v1/completions` returns 200 with its unchanged structure.
6. The exact authorized private-LAN/GPU/service/resource/security invariants hold.
7. The corrected service, not rollback main, is left healthy, ready, and running.

## Report and SELF contract

Write immutable `oap/reports/026-b-report.md`. Identify PR/base/head and
implementation/SELF lineage, prove no product/test change this round, carry
forward exact 026-a focused/full evidence, report focused rerun, all CI URLs/
statuses, pre/post service and GPU facts, restart/readiness facts, each bounded
503 retry as allowed, successful exact-warning/no-spacing/tool-metadata proof,
native smoke, log hygiene, final running commit/process/listener, strongest
reason not to merge and its answer, and `Deferred human adjudication: NONE`.

After qualification, verification, push, and report content are complete,
commit only `oap/reports/026-b-report.md` as final SELF. Its first parent must be
the 026-b transcript/live-evidence implementation commit, which itself follows
immutable 026-a SELF. Push, wait for every required final-SELF check to complete
successfully, then signal the response FIFO with exact `OK`. Coding must not
merge.
