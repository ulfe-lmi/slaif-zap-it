# OAP Work Order — 008-a — Qualify RTX 3090 all-resident BLIP3 profile

## Objective

Create the sole Objective-008 PR from current remote `main`. Qualify the already
implemented `sam2_clip_blip3_gpu_resident` profile on the newly operator-assigned
24,576-MiB RTX 3090, prove that SAM2, CLIP and pinned FP16 BLIP3 remain
simultaneously resident without request-time swaps, and replace the repository's
honest "pending >=24 GB qualification" claims with exact bounded evidence.

This objective also makes the strict launcher honor an explicit, validated
operator-owned physical GPU index so the assigned physical index 0 can be used.
The request boundary still cannot select a device, UUID, model, revision, dtype
or residency strategy. This is local research qualification only, not external
deployment, commercial BLIP3 authorization or release.

## Sequencing and GitHub state

- Numeric objective / round: `008 / 008-a`.
- Mode: `CREATE_NEW_PR` from `main` at
  `bdc9aad62a813d7830b4b6920de03fb106f3f886`.
- Repository/default branch: `ulfe-lmi/slaif-zap-it`, `main`.
- There are no open PRs. Local `main` is clean and exactly equals `origin/main`.
- Objective 007 PR #62 is already `MERGED`; rewritten remote main contains its
  accepted sequential implementation/report as commit `9f48962`.
- Current-main CI is green: `static (format, lint, build)`,
  `release (artifact audit)`, Python 3.10/3.11/3.12, and `Analyze (python)` all
  completed successfully at `bdc9aad`. No branch protection is configured.
- The former planned `007-c` cannot amend the merged Objective-007 PR. This
  order preserves its deferred outcome as a new PR-sized Objective 008 rather
  than violating one-numeric-objective/one-PR and immutable transcript law.
- Required branch: `oap/008-a-rtx3090-all-resident-qualification`.
- Required PR title: `Objective 008: qualify RTX 3090 all-resident BLIP3`.
- Required new immutable report: `oap/reports/008-a-report.md`.
- Prior orders/reports are immutable. `CRITICAL.md` has one entry,
  `CRIT-0001`, whose latest human disposition is `ACCEPTED`; no open critical
  gate applies to this local qualification.

## Human authority and reconciled live host facts

The human stated on 2026-08-24: "now you have RTX 3090 at your disposal!".
Strategic independently verified that the currently assigned host is `hinton2`
and exposes exactly one NVIDIA device:

```text
physical index: 0
UUID: GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
PCI: 00000000:0B:00.0
name: NVIDIA GeForce RTX 3090
physical total/used/free: 24576 / 15 / 24109 MiB
compute processes: none
driver: 610.43.02
```

The human assignment explicitly authorizes this card despite the earlier
host-specific GPU0 prohibition and supersedes that old index assignment for
this objective only. It does not authorize any unobserved card, unrelated
process, Hinton vLLM service or alternate host resource. Reverify the complete
device list, UUID, PCI, name, physical capacity, used/free memory and compute
processes immediately before every live phase; stop if the assigned card is
occupied, missing or mismatched.

The repository GPU environment presently reports Python 3.12.3, Torch
2.5.1+cu124, CUDA runtime 12.4, one visible RTX 3090 and 24,123.5 MiB Torch
usable memory. Host RAM is 22 GiB with about 20 GiB available at reconnaissance.
`/dev/shm` is a 12-GiB tmpfs with about 12 GiB free; the service root did not
exist. The operator Hugging Face cache is about 22 GiB and contains exactly the
approved snapshot revisions:

- SAM2 `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`;
- CLIP `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`;
- BLIP3/XGen-MM `1d91d356d3b6fbc141140edf490b39890417af44`.

No loopback ZAP-IT listener is running. Port 17891 was unused at reconnaissance,
but it is only a candidate and must be freshly rechecked. The coding wrapper is
the only OAP worker and is waiting on the control FIFO.

## Mandatory implementation scope

### 1. Bounded operator-selected physical index support

- Remove the strict service/launcher assumption that the only legal physical
  index is 1. `SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX` must be an explicit
  non-negative decimal integer in strict live operation, and the launcher must
  set `CUDA_VISIBLE_DEVICES` to exactly that same value before Python imports
  CUDA libraries.
- Keep the expected UUID mandatory and cross-check physical `nvidia-smi`
  index/UUID/PCI/name/capacity/process evidence against the single masked
  logical `cuda:0`. Never select another card automatically and never fall back
  after a mismatch or occupancy finding.
- The index is operator-owned startup configuration only. Uploaded YAML, HTTP
  fields, model config and request headers must not influence it.
- Preserve the old index-1 deployment as a documented valid configuration; do
  not silently change an unspecified index to the new host. Prefer requiring an
  explicit index in the shell launcher; any retained Python default must remain
  compatibility-only and must not weaken strict launcher checks.
- Update unit/integration tests, launcher help/comments, environment template,
  runbook, architecture/agent compact law and current service documentation so
  they distinguish the historical maelstrom1 physical-GPU1 qualification from
  this explicitly assigned hinton2 physical-GPU0 qualification. Do not retain a
  false universal claim that GPU0 is always forbidden; the invariant is now the
  exact operator-assigned index+UUID with all other devices/processes protected.

### 2. All-resident invariants and observability

- Physical total capacity `>=24576 MiB` must still automatically select
  `sam2_clip_blip3_gpu_resident`; `<24576 MiB` must retain the accepted
  sequential profile. Do not add a request or environment residency override.
- In all-resident mode, pinned FP16 SAM2, CLIP and BLIP3 load onto logical
  `cuda:0` and remain there from readiness until shutdown. No `to_blip3`,
  `restore`, CPU migration, empty-cache lifecycle or request-time model reload
  may occur.
- Readiness must remain false until all three pinned holders are successfully
  resident and UUID/capacity checks pass. OOM, loader failure, capacity mismatch
  or missing pinned snapshot must fail honestly without downgrade.
- Ensure existing fixed-label metrics/runtime metadata can prove the selected
  all-resident strategy, all model identities, zero residency transitions and
  bounded CUDA/host memory without revealing paths, request content or secrets.
  Add only focused instrumentation or harness support if current evidence is
  insufficient.
- Strengthen CPU/fake tests for index 0 and index 1 strict launch, invalid/
  inconsistent index-mask rejection, wrong UUID, occupancy, the exact 24,575/
  24,576-MiB boundary, all-resident no-transition behavior, load/readiness
  failure and sequential behavior preservation.

## Live qualification sequence

Use only the currently assigned card and an explicit mask:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
logical application device=cuda:0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Run complete CPU/static/package verification before live GPU work. Live phases
must be serialized. Snapshot the full device/process state before and after
each phase and clean up even on failure.

### Phase A — simultaneous-residency hard gate

- From the pinned local-only cache, construct FP16 SAM2, CLIP and BLIP3 using
  the production all-resident loader and prove exact identities/revisions.
- Reach a state where all three holders are simultaneously on logical
  `cuda:0`; run bounded generated 128x128 SAM2, CLIP and BLIP3 inference through
  the real stage chain with at most 32 generated tokens.
- Capture sanitized model-load time, stage timings, Torch current/peak
  allocated and reserved memory, CUDA free/total, host available/RSS and
  `nvidia-smi` used/free evidence.
- Hard gate: non-empty bounded BLIP3 answer evidence, no OOM/fallback/reload,
  peak reserved memory strictly below 90% of physical 24,576 MiB (22,118.4
  MiB), and all three holders still resident after inference. If this gate
  fails, stop later live phases and report `PARTIAL` or `FAILED`; do not move
  the boundary, quantize, offload, substitute models or force sequential mode.

### Phase B — real loopback service matrix

- Start exactly one authenticated loopback service process/worker/request on a
  freshly verified unused port with the explicit index-0/UUID pin and offline
  cache. Prove readiness reports `sam2_clip_blip3_gpu_resident`, logical
  `cuda:0`, all pinned models and no path/free-memory leakage.
- Run a generated no-BLIP control and a generated real BLIP3 L3 request. Both
  must return HTTP 200; the latter must report real `blip3=executed` with
  bounded answer metadata. No residency transition counter/event may change.
- Exercise one operator-only injected inference failure and one client timeout/
  cancellation/drain path. The next request must succeed, readiness must remain
  honest, the three holders must remain resident and no request content may be
  logged or persisted.
- Record startup/load latency, per-stage/end-to-end timings, current/peak CUDA
  memory, host RSS, model-initialization metrics and zero-transition evidence.

### Phase C — exact local goat regression

- Use the ignored local `goats1.jpg`, `goats2.jpg` and `goats2.yaml` only
  through the existing safe harness. Crop both 5568x4176 images to 2784x2088
  central-50% arrays in memory and never persist a derivative.
- Run exactly ten BLIP3-enabled L3 JSON requests in
  `A,B,A,B,A,B,A,B,A,B` order. Require HTTP 200 and real BLIP3 execution for
  every call with zero residency transitions and all three models retained on
  GPU after every response.
- Report per-request end-to-end/SAM2/CLIP/BLIP3 timings, first/minimum/median/
  nearest-rank-p95/maximum by image and aggregate, CUDA current/peak allocated/
  reserved/free, host RSS/available, object/answer counts and content-free
  semantic digests.
- Compare only facts actually preserved from accepted 007-b evidence: stable
  class mapping/normalized YOLO digest, object and answer counts, executed
  stage status and per-image repeatability. Do not fabricate answer-content or
  floating-score parity where the privacy-preserving predecessor report did
  not retain a comparable baseline.
- Require no monotonic GPU/host growth, no OOM/fallback/reload, no persistence,
  and peak reserved memory below the same 90%-of-physical-capacity gate.

After evidence, stop the service and require the chosen port free, no ZAP-IT
process, no compute process on the assigned GPU, GPU memory back to the fresh
preflight baseline, and `/dev/shm/slaif-zap-it` absent or empty. Do not disturb
the OAP coding wrapper or unrelated listeners/processes.

## Scope and non-goals

Expected product diff is limited to strict operator-index portability, focused
tests/harness evidence, all-resident qualification documentation and exact 008
OAP transcript. Fix ordinary in-scope compatibility defects encountered while
reaching the production all-resident path, but report every live attempt.

Non-goals:

- no Objective 009, second PR, merge/auto-merge, tag, package upload, GitHub
  release, external deployment, LAN bind, gateway, TLS, firewall, VPN, systemd,
  driver/CUDA/global credential or unrelated process mutation;
- no quantization, model substitution, CPU inference/offload, partial-layer
  offload, forced residency selector, threshold change or silent downgrade;
- no API/schema/model-ID/revision/dtype change, multi-worker/concurrent GPU
  operation or persistent request data;
- no use of another GPU/host or any card/process not exactly authorized above;
- no goat bytes, crops, raw YAML, prompts, answers, response bodies, API key,
  model weights or cache paths committed or placed in OAP evidence;
- no claim of commercial BLIP3 permission, production SLA, public exposure,
  accuracy, release readiness or universal performance.

## Acceptance and verification

1. Exactly one Objective-008 branch/PR exists from `bdc9aad`; diff is bounded,
   prior transcript/register bytes are unchanged, and the final report-only
   commit has the literal implementation SHA as first parent.
2. Strict launcher and runtime accept explicit operator index 0 or 1, bind the
   same value to `CUDA_VISIBLE_DEVICES`, require UUID agreement and fail closed
   on missing/invalid/inconsistent/occupied device evidence. Requests cannot
   influence device selection.
3. Physical 24,576-MiB capacity selects all-resident automatically, while the
   24,575-MiB/sequential behavior and accepted low-card tests remain intact.
4. Phase A proves real simultaneous pinned FP16 residency and inference below
   the strict 22,118.4-MiB reserved-memory gate with no fallback or movement.
5. Phase B proves authenticated one-worker loopback readiness, no-BLIP and real
   BLIP3 success, injected failure and client-abort recovery, zero transitions,
   bounded resources and no persistence.
6. All ten alternating goat requests succeed with real BLIP3 execution,
   repeatable sanitized semantics, zero transitions, stable resources and
   comparable 007-b facts reported honestly.
7. Documentation records exact RTX 3090 index/UUID/PCI/driver/Torch/CUDA,
   startup/latency/memory evidence, low-card versus all-resident latency
   comparison, host-specific assignment history and remaining license/release
   limitations without replacing measured 007-b evidence.
8. Full canonical CPU suite/coverage, focused runtime/residency/API/harness
   tests, Ruff format/lint, compile, shell syntax, documentation checks,
   package build/install/artifact/secret scans and every current PR-head
   CI/CodeQL check are successful. Required skipped, pending, failed or missing
   evidence is not pass.
9. Final cleanup proves the port, `/dev/shm`, process and assigned-GPU baseline;
   unrelated host resources remain unchanged.

## Deferred human adjudication

- Decision: `NONE`

The operator explicitly assigned the card, the work remains local/reversible,
and no release/external/security boundary is crossed. BLIP3's existing
non-commercial license limitation remains a documented release/deployment gate,
not a new unresolved design dilemma.

## GitHub publication and report contract

- Create only the required branch and one PR. Never merge or enable auto-merge.
- Commit/push all implementation, tests, docs, live-evidence summaries and exact
  unchanged active/order transcript before the report. Record the literal
  implementation SHA and wait for every required implementation-head check.
- Publish only `oap/reports/008-a-report.md` in the final `SELF` commit; it must
  change exactly that path and have the implementation SHA as first parent.
- The report must include exact base/head/PR topology; file/commit scope; every
  criterion; all CPU/live commands with exact status; every live attempt;
  device/process snapshots; model revisions; Phase A/B/C timing and memory
  tables; 90% calculations; zero-transition proof; semantic comparison;
  cleanup; docs/dependencies; current CI/checks; CRITICAL immutability; and all
  limitations/non-goals.
- Explicitly answer the strongest reason not to accept: the implementation was
  previously only fake-tested at the exact 24-GiB threshold, the new card is
  physical index 0 contrary to old host-specific guards, and 24,576 MiB
  marketed capacity leaves little evidence margin. Answer only with fail-closed
  index/UUID tests plus real simultaneous/repeated memory and recovery evidence.
- Send exact FIFO `OK` only after the immutable report and remote topology have
  been verified. Coding never merges or starts another objective.
