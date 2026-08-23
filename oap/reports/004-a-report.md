# OAP Coding-Agent Report — 004-a

## Work order
- Identifier: `004-a`; objective 004; round mode `CREATE_NEW_PR` expected.
- Order file: `oap/orders/004-a-loopback-service-activation.md` (read in full).
- Deferred human adjudication decision in order: `NONE`.

## Status
**BLOCKED** — pre-implementation orchestration collision: two coding agents were
executing this same round concurrently in the same checkout. This report is the
incident evidence required by the block/failure protocol; it claims **no**
implementation acceptance and opens **no** objective PR.

## Executive summary

The wrapper consumed exactly one control OK and launched this agent
(`opencode run --title "OAP coding 004-a"`, PID 210885, started 18:29 CEST),
which created the required branch `oap/004-a-loopback-service-activation` from
verified base and began implementation. At 18:42 a second coding process for
the same round started in the same working directory (`codex exec` "OAP CODING
agent ... active OAP round is 004-a ... recovering the SAME round", PID 212559,
launched via `/synology/homes/janezp/opencode-supervision/slaif-zap-it/
codex-coding-supervisor.sh`, whose mtime is also 18:42). Both writers then
mutated the shared uncommitted tree simultaneously:

- The second writer overwrote files authored by this stream (e.g.
  `src/runtime/live_service.py`, `scripts/serve_local.sh`,
  `deploy/zap-it-local.service`, `tests/test_live_service_units.py`) while they
  were in use, and edited additional shared sources
  (`src/service/app.py`, `src/service/envelope.py`, `src/runtime/strategy.py`,
  `modules/segmenter/sam2.py`, `README.md`, `.gitignore`,
  `requirements-gpu-cu124.lock`).
- This stream observed the second writer perform live GPU1 activation cycles:
  a listener appeared on `127.0.0.1:23654`, GPU1 memory rose to ~1241–1849 MiB,
  then the listener disappeared and GPU1 returned to 6 MiB (observed repeatedly
  between 18:56 and 18:58 CEST).

Exclusive execution became impossible without destroying the other stream's
work (forbidden). Per instruction ("on block/failure publish truthful evidence
according to protocol"), this stream stopped all further mutation, preserves
both work streams untouched in the working tree, publishes this report as the
only commit of its branch, and terminates after signaling.

## Authoritative GitHub state
- Repository: `ulfe-lmi/slaif-zap-it`.
- Base (start) SHA: `1a4272d60c52cc045f57f2842652485efdb7a55c`
  (= verified remote `main` head at branch creation; prerequisite PR #47 merged).
- Implementation head SHA: NONE EXISTS — no implementation commit was published
  by either writer as of this report's parent state (local and remote branch
  absent/`main`-equal, verified via `git log` and `ls-remote` immediately before
  publication).
- Report publication commit: SELF (single-file commit of this report path on
  branch `oap/004-a-loopback-service-activation`; parent = base SHA above).
- New PR: **NO** (no exclusive verifiable implementation exists to publish);
  amended existing: NO; coding merge: NO.

## Changes/files

Authored by this stream before standing down (subsequently partially rewritten
by the concurrent writer; content provenance in the current tree is mixed and
MUST NOT be trusted without audit):

- `src/runtime/live_service.py` (original design: strict loopback config,
  preflight, resident registry, engine adapter, readiness composition, main)
- `modules/classifier/clip.py` (resident-label resync: `_class_map_from`,
  `_rebuild_prompt_index`, `_encode_text_prompts`, `update_labels`, run() hook)
- `scripts/serve_local.py`, `scripts/serve_local.sh`,
  `scripts/serve_local_stop.sh`, `deploy/zap-it-local.service`
- `tests/test_live_service_units.py`, `scripts/smoke_local_service.py`

Observed in tree, authored by the concurrent writer:
`docs/RUNBOOK.md`, `tests/test_live_runtime.py`, plus modifications listed in
the summary above. `oap/active` + `oap/orders/004-a-*.md` are strategic's
transcript, unchanged by both streams.

## Acceptance evidence
All fourteen acceptance criteria: **NOT RUN** by this stream — blocked before
any exclusive implementable state existed. No acceptance claim is made for the
concurrent writer's uncommitted work.

## Verification
- `git fetch && git status/log/ls-remote` reconciliation: PASSED (base verified;
  no premature remote branch/PR from either writer at publication time)
- `nvidia-smi` all-GPU snapshots (start/during/close): PASSED — GPU0 byte-stable
  at 2161 MiB (unrelated PID 66522) across every observation; GPU1 idle 6 MiB at
  start; transient allocations (~1241/1849 MiB) observed only from the other
  stream's activity; GPU1 back to 6 MiB at final check
- `ss -tln` port scans (17891/23654): PASSED — 17891 unused throughout this
  stream's checks; 23654 transiently LISTENING (other stream), free again at
  final check
- `pytest tests/test_live_service_units.py -q` at ~18:50: 2 FAILED / 29 PASSED —
  the two failures were caused by the concurrent rewrite of
  `src/runtime/live_service.py` mid-run (its stricter config validation vs this
  stream's tests); result voided as mixed-provenance, treated as NOT RUN
- Canonical CPU suite/Ruff/build: NOT RUN (blocked)
- Live service E2E/concurrency/restart/cleanup layers: NOT RUN (blocked)
- CI/CodeQL on any head: NOT RUN (no implementation push)

## CI/checks
None triggered by this stream except the report push itself. No PR exists.

## GPU/service/resource evidence
- Physical mapping re-verified live: index 0 UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182` (PROTECTED, unrelated PID 66522);
  index 1 UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` (target).
- This stream never initialized CUDA, never loaded models, never bound a port,
  and left no `/dev/shm` residue (only the other stream's `runtime/` dir was
  observed under `/dev/shm/slaif-zap-it`).
- No firewall, VPN, driver/CUDA, systemd, unrelated service, or global
  credential change occurred.

## Documentation/provenance
No documentation claims are made by this report beyond incident facts. The
working tree currently contains documentation authored by the concurrent
writer; its accuracy is unaudited here.

## Deferred human adjudication
- Critical register action: **NONE**
- No CRITICAL entry was created, edited, or appended. The collision itself is
  an orchestration fault to be resolved by strategy/human, not a register
  threshold matter; safe continuation (letting the surviving stream proceed)
  remained possible throughout, which is why this is a BLOCKED incident report
  rather than a register candidate.

## Safety/scope confirmations
- Physical GPU0 protected and byte-stable throughout every observation.
- No request data, secrets, model weights, or raw YAML/images enter this
  report; process command lines are quoted only as far as needed for identity.
- Never merged; never closed another's work; nothing reset or cleaned for
  convenience — both streams' uncommitted files are preserved exactly as found.
- Exactly one commit (this report) is published by this stream.

## Limitations/blockers
BLOCKER: dual-writer split-brain on round 004-a. Strategy must decide:
1. which stream (if either) may complete the round, or whether to relaunch
   cleanly after quarantining both trees;
2. mandatory full provenance audit of ANY content later committed from this
   working tree, since authorship interleaves at file level (and within-file
   risk cannot be excluded);
3. the supervisor defect: a recovery coder was spawned without checking for a
   live coding-agent process already holding the round.

## Factual strategic follow-up
- Reconcile the two OK-capable processes: this stream sends its terminal signal
  together with this report; the concurrent writer may attempt its own signal
  later — FIFO lifecycle needs strategy-side handling.
- Consider a liveness/lock guard in the launch/supervision path before the next
  control OK is issued.
- The order itself remains executable as written once single-writer ownership
  is restored; no requirement changes are implied by this incident.
