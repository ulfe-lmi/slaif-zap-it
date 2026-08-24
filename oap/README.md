# OAP transcript

`orders/` and `reports/` are immutable versioned orchestration artifacts.
`active` identifies the most recently published round; it is not a completion
signal. FIFOs and strategic drafts live outside Git under the strategic
workspace.

Wire payload is exactly bytes `OK`. `NNN-a` creates one PR; `NNN-b..z` amend it.
Coding never merges. Strategic independently reviews and alone merges.

`CRITICAL.md` is separate from the per-turn transcript: it is a rare append-only
Deferred Human Adjudication Register. Strategic must decide provisionally and may
order an append only when all five materiality conditions hold. It is not updated
on ordinary turns. Open entries gate the stated deployment/release boundary, not
routine development.

## Local two-process operation

The coding checkout and strategic workspace are separate:

```text
CODING=$HOME/opencode-work/slaif-zap-it
STRATEGIC=$HOME/opencode-supervision/slaif-zap-it
```

Runtime configuration is private mode 0600 under the strategic workspace.
Coding blocks on the control FIFO outside the coding agent, executes exactly one
active order, publishes an immutable report, and sends the exact response bytes
`OK`. FIFO acknowledgment is synchronization only; it is never acceptance.

Recovery reconciles `active`, immutable reports, GitHub, the local worktree, and
live host state. Restart only a dead wrapper. Re-signal the same unresolved
order after a process failure; do not invent a letter solely for a crash.

## Deferred human adjudication

Human Work Preloading records intent, architecture, sequence, and acceptance
before execution. Strategic owns ordinary decisions. A `CRITICAL.md` entry is
allowed only when all five register conditions hold; it is not a TODO, bug list,
or substitute for technical judgment.

Strategic authors exact entry bytes and coding appends them only when an active
order explicitly requires it. Agents never edit or close prior entries. Only a
human disposition can mark the applicable gate accepted.

Inspect orchestration consistency with:

```bash
python oap/bin/check_state.py \
  --repo-root "$CODING" \
  --strategic-home "$STRATEGIC"
```
