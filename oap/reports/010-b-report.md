# OAP Coding-Agent Report — 010-b

## Status

PASSED

## GitHub identity

- Objective/round: `010-b`; amended only PR #66.
- Base: `main` at `5da3851347c2031bea11012fc554140ba7894cc2`.
- Implementation head: `010b080fe8c569705a6009eb605b967bdb68c530`.
- Report publication commit: SELF.
- No merge, auto-merge, release, or second PR was performed by implementation work.

## Implemented correction

- Preserved the 010-a explicit model-management API and lifecycle controller.
- Replaced unload's transient model-to-CPU move with direct isolated-holder graph
  destruction, bounded CUDA cleanup, garbage collection, and best-effort glibc
  heap trimming. This avoids retaining a CPU model copy and bounded cold-cycle RSS.
- Expanded the sanitized lifecycle harness to mechanically verify credentials,
  invalid requests, transition visibility, real inference, drain rejection,
  idempotency, two-cycle memory release, PID/listener continuity, metrics, logs,
  and cleanup.
- Replaced the harness's backtracking metric-label regex after GitHub CodeQL
  correctly reported it as high severity. The replacement is a tested linear
  parser; the subsequent CodeQL result is successful.
- Corrected the repository coding-wrapper prompt to use only the active-order
  assigned physical index and UUID rather than a stale fixed GPU0/GPU1 rule.

## Real RTX 3090 evidence

Assigned physical index 0 matched UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090
24,576 MiB. The service used logical `cuda:0`, one PID (`333860`) and one
loopback listener through both cycles.

- Cold: health 200, ready/completion 503, index `UNAVAILABLE`, initialization 0.
- Authentication/policy: 9 missing/wrong/cross-credential requests returned 401;
  malformed/unknown/oversized/query/wrong-model requests failed without allocation.
- Cycle 1: `LOADING -> READY`; load 200; 10,095,121,920 Torch bytes allocated;
  real combined L3 inference 200 with 8 objects, 8 bounded answers, digest
  `ca8cdc96b8671f54a74b2d4905d59e91dd035a026a1e2c5cde4c5791eed154b7`.
- Drain: active inference completed 200; new inference returned 503 `not_ready`;
  unload exposed `UNLOADING` and returned 200 only after drain.
- First cold proof: 8,519,680 bytes allocated, 20,971,520 reserved; 99.92% and
  99.80% of loaded deltas released; idempotent unload 200.
- Cycle 2: initialization count exactly 2; repeated real inference returned the
  identical digest; unload returned 200 with the same 8/20 MiB cold bounds and
  no monotonic GPU or host-RSS growth beyond tolerance.
- Final cleanup: service stopped normally; port free; zero target/non-target
  compute processes; physical GPU returned to 15 MiB; shared-memory root empty;
  log sanitation passed. A first disclosed attempt failed only the cold-cycle
  RSS tolerance and motivated the heap-trim correction; the complete rerun passed.

## Verification

- Canonical CPU suite: `413 passed, 1 skipped`, 77.23% coverage. The skip is the
  separately opt-in GPU integration test; the ordered real lifecycle harness passed.
- Focused model-control suite after CodeQL correction: `10 passed`.
- Ruff format/check, compileall, documentation (27 files), shell syntax, wheel/
  sdist build, tracked-tree and artifact secret scans, artifact verification,
  and `git diff --check`: PASSED.
- All seven implementation-head GitHub checks: PASSED, including CodeQL after
  the rejected regex was replaced.

## Scope and safety

No cross-process lease/handoff, LAN/public bind, system service activation,
firewall/network/driver mutation, model substitution, release, credential or raw
request logging, or critical-register mutation was performed. Objective 011
remains necessary for any cooperative multi-process ownership claim.

Deferred human adjudication: NONE.

The strongest reason not to accept was that CPU fakes could hide retained Torch
references, host-memory growth, or a readiness race. The repeated real sequence
answered that concern with one unchanged PID/listener, concurrent drain evidence,
identical semantic results, measured 8/20 MiB Torch cold bounds, 99.8%+ release,
and complete post-stop resource cleanup.
