# OAP transcript

`orders/` and `reports/` are versioned append-only orchestration artifacts.
`active` is absent until strategy publishes the first finalized order. FIFOs and
strategic drafts live outside Git under `~/opencode-supervision/slaif-zap-it`.

Wire payload is exactly bytes `OK`. `NNN-a` creates one PR; `NNN-b..z` amend it.
Coding never merges. Strategic independently reviews and alone merges.

`CRITICAL.md` is separate from the per-turn transcript: it is a rare append-only
Deferred Human Adjudication Register. Strategic must decide provisionally and may
order an append only when all five materiality conditions hold. It is not updated
on ordinary turns. Open entries gate the stated deployment/release boundary, not
routine development.
