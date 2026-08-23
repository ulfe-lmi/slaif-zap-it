# OAP Work Order — 005-c — Final deadline and zero-stream corrections

Objective `005-c`, narrow continuation of numeric Objective 005 on existing PR
#49. Fix two strategic-review-proven edge bugs in the otherwise accepted
`005-b` implementation: false L3 raw-artifact rejection when zero visualization
streams are configured, and success metrics recorded before the final request
deadline check. Preserve all prior code/evidence, including the completed
nonredistributable local goats run. Do not rewrite earlier artifacts.

## Authoritative state

- Existing branch: `oap/005-a-full-output-parity-hardening-and-evidence`.
- Sole PR: #49, open/mergeable/clean, base `main`, unchanged title.
- Current PR/local/remote head:
  `4222e13ce2fade6d668b0c66745cf40be06756f4`, immutable report-only
  `005-b` commit.
- Its first parent is implementation
  `c4f786452ceabd1f5028efcadb178e608e684db5`; only changed path is
  `oap/reports/005-b-report.md`. All six checks are SUCCESS.
- Required new report: `oap/reports/005-c-report.md`; every prior order/report
  is immutable.

## Finding 1 — zero visualization streams

`check_visualization_raw_budget()` computes `reserved=0` when no annotated
streams exist, but still compares `height*width*3` against the per-artifact
limit. Thus an L3 request with no visualization can return
`response_too_large` solely because a hypothetical stream would be large.

- Return zero immediately when stream count is zero; perform no per-stream raw
  check and reserve no debug bytes.
- Continue enforcing exact per-stream/total bounds when one or more supported
  streams are configured.
- Add CPU tests using dimensions whose hypothetical RGB stream exceeds the
  limit: L3/no-stream must reach the engine and succeed; the same request with
  one annotated stream must reject before the engine; L0–L2 behavior unchanged.

## Finding 2 — deadline metrics ordering

Both JSON and ZIP success branches call `metrics.observe_success()` and observe
request duration before their last `check_deadline()`. If the final deadline is
crossed there, the exception handler also records `timeout`, double-counting one
request as success and timeout.

- Complete every final deadline check before constructing/recording successful
  completion counters/histograms and before marking `metrics_recorded=True`.
- A timed-out serialization records only `timeout` (and appropriate duration),
  never success/completion/response/object/artifact success metrics.
- A successful response records exactly one success/completion.
- Apply identically to JSON and ZIP.
- Add deterministic TestClient tests using the operator-only serialization
  delay or a monotonic clock seam: scrape metrics and prove timeout count 1,
  success/completion 0 for the rejected request; subsequent success increments
  success/completion exactly once with timeout unchanged.

## Verification

- Focused resource/app/metrics tests above for JSON and ZIP.
- Complete canonical pytest/coverage, Ruff format/lint, compile, shell syntax,
  wheel build/import, diff/secret/large-artifact checks.
- One normal live synthetic L3 JSON+ZIP request on freshly verified physical
  GPU1 plus one safe serialization-timeout/metrics probe and recovery. Do not
  rerun or expose goats assets; inherit the complete `005-b` goats evidence.
- Final service stopped, ports free, GPU1 idle, shared memory empty, GPU0/PID
  66522 untouched.
- All six GitHub checks SUCCESS on implementation and report heads.

## Scope/non-goals

- Expected behavior changes: `src/service/resources.py`, `src/service/app.py`
  and focused tests/docs only as needed, plus exact order/active transcript.
- No API limit/format, RLE, model, visualization capability, metrics schema,
  goats evidence, CLI, dependency, security or deployment-scope change.
- No new branch/PR/title; no merge by coding.

## Acceptance/publication

1. Zero-stream L3 cannot be rejected for hypothetical visualization bytes.
2. Configured streams retain exact early raw-allocation rejection.
3. Serialization timeout and success metrics are mutually exclusive and exact
   for JSON and ZIP, including recovery.
4. PR #49 and all prior Objective-005 behavior/evidence remain green.
5. Commit non-report corrections first; capture literal SHA; final immutable
   `oap/reports/005-c-report.md` SELF commit changes only that path and has the
   implementation SHA parent. Verify remote bytes/topology and all checks
   before FIFO response.

## Deferred human adjudication

- Decision: `NONE`

These are ordinary low-risk correctness bugs. The human fixture-rights decision
and Objective-006 nonredistribution gate remain unchanged.

## Coding response

Send exact FIFO `OK` only after corrected PR/report/CI and final stopped host
verification. Coding never merges.
