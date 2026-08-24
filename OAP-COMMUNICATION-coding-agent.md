# OAP communication protocol — coding OpenCode

**Role:** execute one active bounded order, publish one correct GitHub PR state,
publish one immutable report, signal, exit. Never plan roadmap, merge, release,
accept, or choose the next ID.

## Truth and paths

```text
GitHub=remote software truth
OAP active/orders/reports=orchestration truth
local checkout=recoverable execution state
FIFO OK=synchronization only

REPO=$HOME/opencode-work/slaif-zap-it
STRATEGIC=$HOME/opencode-supervision/slaif-zap-it
CONTROL=$STRATEGIC/control.fifo
RESPONSE=$STRATEGIC/response.fifo
```

Verify paths/FIFO types from runtime. Strategic writes control/order/active and
reads response/report. Coding does the inverse. Wire payload exactly ASCII `OK`
(hex `4f4b`), no newline/ID/status. Wrapper consumes control before starting a
fresh `opencode run`; do not read control again inside that run.

## Selection and PR law

Read `oap/active`; require `^[0-9]{3}-[a-z]$` and exactly one matching immutable
`oap/orders/<ID>-*.md`. Never infer newest/highest/mtime.

- `NNN-a`: create one new branch and exactly one new PR from verified remote base.
- `NNN-b..z`: amend exact existing branch/PR; **NO NEW PR**.
- coding never invents ID or adjacent work; never merges/auto-merges/closes.

## Preflight

Before mutation: read constitution/protocol/compact architecture/security/testing
and exact order; fetch/reconcile GitHub default branch, current objective PR,
head/checks; inspect working tree and preserve unrelated work; verify relevant
GPU/service state without changing protected resources; identify exact scope,
non-goals, acceptance, tests, docs and report evidence.

## Deferred human adjudication

`CRITICAL.md` is not a routine report sink. Coding may append only when the exact
active order contains a strategic decision `APPEND CRIT-NNNN` and exact entry
content satisfying the register threshold. Use `oap/bin/append_critical.py`;
verify the diff adds one new terminal section and changes no earlier bytes.
Commit the append with implementation work before capturing the implementation
SHA. Never autonomously create, rewrite, delete, reorder, close, or human-approve
an entry.

When no append is ordered, the normal report states `Critical register action:
NONE`. A candidate may be reported only for a material unresolved dilemma that
plausibly meets all five register conditions; normal bugs, limitations, failed
tests, design preferences and low-risk reversible choices are not candidates.

## Round

1. Resolve active/order after wrapper signal.
2. Reconcile GitHub/local/protected-host state.
3. Implement only scope; install safe repo-local tools yourself.
4. Run exact required tests; fix safe in-scope failures.
5. If and only if ordered, append exact strategic-authored `CRIT-NNNN` and
   verify append-only integrity.
6. Commit/push implementation, any ordered critical append, and exact unchanged
   active/order transcript.
7. Create (`a`) or amend (`b..z`) exact PR; inspect current-head CI.
8. Push all non-report work; capture literal 40-hex implementation head.
9. Atomically create one report with:

```text
Implementation head SHA: <literal>
Report publication commit: SELF
```

10. Stage only report; commit as final child; parent=implementation head.
11. Push; verify remote PR head, parent, one-path report commit and exact bytes.
12. Perform no later mutation/push; send response `OK`; terminate.

A truthful `PARTIAL|BLOCKED|FAILED` report also signals. `OK` never means
accepted. Activated orders/reports are immutable.

## Required report

Use `oap/templates/REPORT-TEMPLATE.md`. Include exact repository/PR/base/head,
starting SHA, implementation SHA, SELF, commits/files, each criterion and evidence,
all commands with exact status, CI check names/SHA/state, docs, dependencies,
GPU physical/visible mapping when relevant, ports/services, memory/`/dev/shm`,
secrets/scope confirmations, `Critical register action: NONE | APPENDED
CRIT-NNNN | CANDIDATE REPORTED`, limitations and factual follow-up.

Never include raw images/YAML, credentials, environment values, private paths
unneeded for operation, model weights, customer data, or fabricated readiness.

## Crash recovery

Waiting is normal. Crash before report: restart, read active, inspect branch/PR/
working tree, and resume only unresolved turn after strategic re-signal. Existing
final report is immutable; never overwrite/replay. If report push succeeded but
FIFO did not, preserve state and let strategy reconcile. GitHub/OAP truth beats
conversation memory.

## Invariants

1. One active ID; one order/report; no inference.
2. `a` one PR; `b..z` same PR.
3. All non-report claims remote before report.
4. SELF first parent equals literal implementation SHA.
5. Final round commit changes only report.
6. Coding never merges/accepts/advances.
7. Required skipped/pending/missing is not pass.
8. The active-order-assigned physical GPU alone is exposed as logical `cuda:0`;
   every unassigned device and unrelated service/process is protected.
9. Request content/secrets never enter logs/OAP artifacts.
10. CRITICAL entries are rare, strategic-authored and append-only; coding does not
    use them to avoid ordinary engineering judgment.
