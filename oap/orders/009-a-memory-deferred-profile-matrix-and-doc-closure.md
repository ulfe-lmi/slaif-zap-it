# OAP Work Order — 009-a — Close the memory-deferred profile matrix and current truth

## Objective

Create the sole Objective-009 PR from current remote `main`. Complete the audit
of work historically held back by the 11-GB GPU: run one explicit real
all-resident service matrix covering every supported SAM2/CLIP/BLIP3 profile on
the assigned RTX 3090, reconcile every current document that still describes
the >=24-GB profile as pending or sequential-only, and add a bounded regression
guard against reintroducing those obsolete claims.

This objective closes evidence/current-truth integration for the already merged
Objective-007/008 capability. It does not invent a new model, geometry stage,
panoptic renderer, deployment topology or release authority.

## GitHub and orchestration state

- Numeric objective / round: `009 / 009-a`; mode `CREATE_NEW_PR`.
- Repository/default branch: `ulfe-lmi/slaif-zap-it`, `main` at
  `b1d8c5dbc9392002ab52b3b0b744582a073ebf75`.
- Local `main` is clean and equals `origin/main`; no PR or issue is open.
- Objective 008 PR #64 is merged as the exact current-main squash. Its final
  implementation/report and post-merge CI/CodeQL checks are successful.
- Required branch: `oap/009-a-memory-deferred-profile-matrix-and-doc-closure`.
- Required PR title: `Objective 009: close memory-deferred profile evidence`.
- Required report: `oap/reports/009-a-report.md` as a final SELF child.
- Existing orders/reports are immutable. `CRITICAL.md` has no open entry;
  CRIT-0001 remains human-accepted and byte-identical.

## Reconciled audit findings

The historical memory-held work is bounded and identifiable:

1. Objective 003 blocked BLIP3 and profiles requiring it on the 11-GB card
   before allocation.
2. Objective 007 qualified pinned FP16 BLIP3 and the sequential low-card
   lifecycle.
3. Objective 008 qualified simultaneous FP16 SAM2+CLIP+BLIP3 residency on the
   assigned 24,576-MiB RTX 3090, including combined/no-CLIP BLIP requests,
   repeated goat calls, zero transitions, recovery and cleanup.

Current code declares four supported service profiles:

```text
sam2
sam2_clip
sam2_blip3
sam2_clip_blip3
```

Evidence exists across prior rounds, but no single live all-resident matrix
names and proves all four profiles together. Current documentation also remains
internally contradictory. At minimum `CHANGELOG.md`, `RELEASE_NOTES.md`,
`THIRD_PARTY_NOTICES.md`, `docs/CONFIG.md`, `docs/RUNBOOK.md`,
`docs/SERVICE-DATASHEET.md`, `docs/OUTPUT-PARITY.md`, and
`docs/RELEASE-GATE-INVENTORY.md` still contain pre-008 pending,
sequential-only, old-host-only or incomplete qualification claims. Preserve
historical OAP reports and `docs/history/`; correct only current truth.

Geometry/panoptic, gateway/container/LAN/public exposure, commercial model use,
tracked-media rights and final release are not GPU-memory deferrals. They remain
separate scope/gates and must not be folded into this objective.

## Fresh assigned-host evidence

At activation on 2026-08-24, hinton2 exposes exactly one GPU:

```text
physical index: 0
UUID: GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
PCI: 00000000:0B:00.0
name/capacity: NVIDIA GeForce RTX 3090 / 24576 MiB
used/free: 15 / 24109 MiB
compute processes: none
driver / Torch / CUDA runtime: 610.43.02 / 2.5.1+cu124 / 12.4
```

`/dev/shm` is a 12-GiB tmpfs; the mode-0700 service root exists and is empty.
No ZAP-IT service/listener is running and port 17891 is currently unused. The
pinned local model cache/revisions qualified by Objectives 007/008 remain the
only permitted model assets. Reverify all facts immediately before live work;
these observations are not a reservation.

## Mandatory implementation and evidence

### 1. Explicit four-profile live matrix

Extend the existing sanitized local service smoke tooling, or add one focused
operator harness, so it can drive and verify the four supported profiles without
embedding raw response bodies or private inputs in evidence. Reuse generated
128x128 RGB input and API-safe in-memory YAML only.

Run the exact interleaved sequence:

```text
sam2,
sam2_clip,
sam2_blip3,
sam2_clip_blip3,
sam2_clip_blip3,
sam2_blip3,
sam2_clip,
sam2
```

For every call require authenticated L3 JSON HTTP 200, runtime strategy
`sam2_clip_blip3_gpu_resident`, logical `cuda:0`, three pinned runtime model
identities, and residency transition count zero. Verify exact stage semantics:

- `sam2`: SAM2 executes; CLIP and BLIP3 are `not_configured`;
- `sam2_clip`: SAM2 and CLIP execute; BLIP3 is `not_configured`;
- `sam2_blip3`: SAM2 and BLIP3 execute, CLIP is `not_configured`, and at least
  one bounded BLIP3 answer is produced;
- `sam2_clip_blip3`: all three execute and at least one bounded BLIP3 answer is
  produced through a valid bounded rule.

The two occurrences of each profile must have the same sanitized stage/result
shape and content-free semantic digest. Floating scores and timings need not be
identical. Capture per-call and per-profile first/minimum/maximum/median timing,
objects/answers/stage counts, Torch current/peak allocated/reserved/free, host
RSS, transition count and content-free digest. Peak reserved memory must remain
strictly below 22,118.4 MiB (90% of physical capacity), with no monotonic
GPU/host growth, reload, CPU migration, fallback or persistence.

The harness must fail closed on a wrong profile/stage status, missing answer,
wrong strategy/device/model count, nonzero transition, memory-gate breach,
request residue or malformed response. Add CPU/fake tests for the matrix
classifier/validator and failure cases; do not rely only on manual report prose.

### 2. Current-document reconciliation

Update all current authoritative documents to say, consistently and precisely:

- `<24576 MiB` uses the live-qualified sequential stage-boundary lifecycle on
  the historical 11-GB RTX 2080 Ti;
- `>=24576 MiB` uses the live-qualified all-resident lifecycle on the assigned
  24,576-MiB RTX 3090;
- both expose only logical `cuda:0` after an explicit operator index+UUID pin;
- all four supported profiles have real all-resident matrix evidence after
  this objective;
- the all-resident measurements are bounded local research evidence, not an
  SLA, accuracy claim, commercial license clearance or external deployment;
- geometry/panoptic and deployment/release gates remain unsupported/separate
  for reasons other than GPU memory.

At minimum reconcile every stale file named in the audit findings. Also inspect
all other current root/docs Markdown for equivalent contradictions; absence from
the minimum list is not permission to leave a false claim. Retain the detailed
historical 007-b and 008 measurements rather than replacing them.

Extend `scripts/check_documentation.py` and its tests with narrow current-doc
patterns that reject the obsolete claims found by this audit, including
"all-resident qualification remains separate", "pending a separate live
qualification", "only live-qualified BLIP3 residency", and the old
"implemented/fake-tested but not live-qualified" statement. Do not scan or
rewrite immutable OAP reports or historical docs as current truth.

### 3. Completion classification

Add a concise current-document statement that the GPU-memory deferrals from
Objectives 003/007 are closed by Objectives 007–009, with links to the runtime
evidence and immutable reports. State explicitly that remaining unsupported or
release-gated items are not being represented as memory-blocked work.

## Live execution law

Use only:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
logical application device=cuda:0
```

Run full CPU/static/package checks before live work. Start one authenticated
loopback process/worker/request on a freshly verified unused port. No second
worker/process, concurrent inference or other card is authorized. Stop after
the matrix and require port free, no ZAP-IT/compute process, assigned GPU back
to the fresh baseline, and `/dev/shm/slaif-zap-it` empty.

## Scope and non-goals

Expected product diff is limited to sanitized profile-matrix smoke tooling,
focused CPU tests, current documentation and its integrity checker, plus exact
009 transcript. No production inference/model behavior change is expected; if
a genuine profile bug is found, fix only the smallest in-scope defect and report
every failed live attempt honestly.

Non-goals:

- no geometry/panoptic activation, Detectron2 install, streaming, gateway,
  container, LAN/public bind, TLS, auth-policy expansion or multi-worker work;
- no model/revision/dtype/residency-threshold change, quantization, offload,
  substitution, download or request-selected operator setting;
- no tag, package upload, release, systemd activation, driver/CUDA/firewall/VPN
  mutation, new CRITICAL entry, history rewrite or unrelated process mutation;
- no private goat fixture use is needed; no raw image/YAML/prompt/answer/body,
  API key, cache path or model weight may enter Git, logs or OAP evidence;
- no rewrite of any prior order/report or `docs/history/` artifact.

## Acceptance and verification

1. Exactly one new Objective-009 branch/PR exists from `b1d8c5d`, with bounded
   scope, exact active/order transcript and a final report-only SELF child.
2. CPU tests prove the matrix harness recognizes all four profile/stage shapes,
   repeatability and every fail-closed condition named above.
3. The exact eight-call live matrix passes on the assigned RTX 3090 with correct
   stage statuses, non-empty BLIP3 answers where required, zero transitions,
   repeatable content-free semantics and peak reserved below 22,118.4 MiB.
4. Final live cleanup proves GPU/process/port/shared-memory baseline and no
   unrelated resource change.
5. Every current pre-008 memory-deferral contradiction is corrected; the docs
   checker and tests prevent the audited stale phrases from returning while
   leaving historical evidence immutable.
6. Documentation distinguishes completed GPU-memory work from geometry,
   panoptic, licensing, media, deployment and release gates without overclaim.
7. Canonical CPU suite/coverage, focused harness/docs tests, Ruff, compile,
   shell syntax, documentation integrity, package build/install/artifact scans,
   tracked-tree secret equality and every implementation/report-head CI/CodeQL
   check are successful. Required skipped/pending/failed/missing is not pass.
8. Final report includes literal implementation SHA/SELF, exact commands and
   statuses, eight-call sanitized table, resource calculations, doc audit,
   scope/immutability/cleanup proof and the completion classification.

## Deferred human adjudication

- Decision: `NONE`

The assigned GPU is already explicitly authorized and the remaining changes
are local, reversible evidence/documentation work. Existing model-license and
release gates are preserved rather than decided here.

## Publication/report contract

- Create only the required Objective-009 PR; coding never merges/auto-merges.
- Commit/push all non-report work, record the literal implementation SHA and
  require all current checks successful.
- Publish only `oap/reports/009-a-report.md` in the final SELF child and verify
  its first parent, one-path diff and remote bytes.
- After report publication, rerun the tracked-tree release helper using its
  default secret-baseline path so report prose cannot introduce an unreviewed
  finding. Require all seven report-head checks successful before signal.
- Explicitly answer the strongest reason not to accept: Objectives 007/008 may
  have qualified the models while leaving a supported profile unexercised or
  current docs misleading. Answer with the exact real four-profile matrix,
  fail-closed tests, exhaustive current-doc scan and immutable historical
  evidence—not inference from a combined-profile success.
- Send exact FIFO `OK` only after final remote topology/check verification.
