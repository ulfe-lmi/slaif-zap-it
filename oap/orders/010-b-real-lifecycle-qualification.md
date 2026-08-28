# OAP Work Order — 010-b — Complete real explicit-lifecycle qualification

## Objective

Amend only Objective-010 PR #66. The 010-a implementation/API/CPU evidence is
useful but its immutable report is `PARTIAL` because a stale external wrapper
sentence was incorrectly treated as prohibiting the explicitly assigned RTX
3090 physical index 0. That wrapper has been corrected to the merged
operator-assigned index+UUID law. Now run the required real cold/load/infer/
drain/unload/reload/infer/unload sequence, fix any ordinary in-scope lifecycle
defect it exposes, and close Objective 010 only with measured memory release.

Do not create another PR or merge. Preserve the immutable 010-a order/report.

## Authority and reconciled state

- Human instruction explicitly made the RTX 3090 available for this work.
- Current root/strategic law says the active order may authorize any exact
  operator-assigned index+UUID; every unassigned device remains protected.
- This order explicitly authorizes hinton2 physical index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, and no other GPU.
- The local Luna wrapper prompt now repeats that exact general law. Its former
  hard-coded GPU0/GPU1 wording is superseded and must not block this round.
- Numeric objective/round: `010 / 010-b`; mode `AMEND_EXISTING_PR`.
- Sole PR: #66, `Objective 010: explicit model-control API`, branch
  `oap/010-a-explicit-model-control-api`, base `main` at
  `5da3851347c2031bea11012fc554140ba7894cc2`.
- Current report head:
  `5d0bb1af4ca59825fa9a4effaf13710a7e82c556`; first parent is implementation
  `3319939313dee8e3f65cdc7f72058a41d68e5888`; SELF changes only
  `oap/reports/010-a-report.md`.
- All seven implementation/report-head checks are successful. The report is
  honestly `PARTIAL`; it is not acceptance.
- Existing implementation includes the management subset, lifecycle controller,
  gate pause/drain, repeated registry lifecycle, metrics/docs/tests and a seventh
  reviewed secret-baseline identifier finding.
- Required new report: `oap/reports/010-b-report.md`.

## Fresh live preflight

At activation the assigned host still exposes exactly one device: index 0,
UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
`00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24,576 MiB, driver 610.43.02,
15 MiB used / 24,109 MiB free and no compute process. `/dev/shm` is a 12-GiB
tmpfs; `/dev/shm/slaif-zap-it` is mode 0700 and empty. No ZAP-IT listener is
running; ports 17891 and 23654 are free at activation. Reverify immediately
before every live attempt.

Use only:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
SLAIF_ZAP_IT_MODEL_CONTROL_MODE=explicit
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
logical application device=cuda:0
```

Use distinct generated inference and model-control bearer values only through
environment variables. Never print, put on a command line, log, report or
persist them.

## Mandatory review/correction before live work

1. Update the repository-owned generic coding-wrapper prompt source
   `oap/bin/launch-coding-agent.sh` from hard-coded GPU0/GPU1 wording to the
   merged active-order-assigned index+UUID law. Do not alter initial historical
   orders.
2. Re-review the new registry/controller/gate implementation against the 010-a
   order. In particular, ensure unload does not retain models on CPU and does
   not require a full transient GPU-to-host copy that can exhaust this 22-GiB
   host. Prefer dropping the isolated holder graph and collecting it directly;
   if a holder-specific move is essential, prove bounded host memory.
3. Expand the sanitized live harness so evidence is mechanically checked rather
   than assembled only from ad-hoc curls. It must cover control/inference auth
   separation, concurrent `LOADING`/`UNLOADING` index visibility, real combined
   inference, drain behavior, idempotency, PID/listener continuity, two-cycle
   semantic repeatability, metrics and memory-release gates. It may emit only
   states/statuses/timings/counts/digests/resources.
4. Run full CPU/static/package/secret checks after any correction and before GPU
   allocation. Preserve default `none` compatibility and all current checks.

## Required live sequence

Start one explicit-mode authenticated service on a freshly verified unused
loopback port. Record one immutable service PID and listener identity. Without
restarting that process:

### A. Cold control plane

- `/healthz` 200; `/readyz` 503; completion 503 `not_ready`.
- Properly authenticated repository index returns one `UNAVAILABLE` entry;
  `ready:true` returns an empty list.
- Missing/wrong/inference bearer requests to index/load/unload return 401 in the
  repository error shape and cause no allocation/state change.
- Wrong model name, query parameter, malformed/oversized/non-object body and
  any config/device/path/revision parameter fail before allocation.
- Record cold Torch allocated/reserved, `nvidia-smi` process/context memory,
  host RSS and initialization count 0.

### B. First load and real inference

- Invoke load asynchronously from the harness; while it is outstanding, prove
  health 200 and index `LOADING` through a second authenticated request.
- Load returns 200 with an empty body only after index `READY`, `/readyz` 200,
  model-loaded gauge 1 and initialization count 1.
- A second load is 200 no-op and does not change initialization count/memory.
- Send a generated combined SAM2+CLIP+BLIP3 L3 request; require HTTP 200, all
  stages executed, non-empty bounded answers and a content-free semantic digest.
- Record loaded Torch current/peak allocated/reserved/free, physical GPU memory,
  host RSS and load/inference latency.

### C. Drain and first unload

- Start one operator-delayed real combined inference and prove it is active.
- Invoke unload concurrently. It must atomically expose `UNLOADING`, reject a
  new/queued inference as `not_ready`, leave health 200, allow the original call
  to complete, and return 200 only after drain and model-memory release.
- The unload duration must be consistent with waiting for the active call; no
  request/model state may leak.
- After unload: index `UNAVAILABLE`, ready/completion 503, loaded gauge 0,
  Torch allocated and reserved each <=64 MiB, at least 90% of the loaded model
  memory delta released, and PID/listener unchanged. Record retained CUDA-context
  `nvidia-smi` memory honestly without calling it zero.
- A second unload is a 200 no-op with stable counters/memory.

### D. Second cycle and cleanup

- Observe `LOADING` on a second load; reach READY with initialization count 2.
- Idempotent ready load remains no-op.
- Repeat the real combined inference; require the same semantic shape/digest as
  cycle 1 and no cross-cycle request state.
- Unload again and require the same cold Torch/context/RSS bounds within a
  documented tolerance, with no monotonic GPU/host growth.
- Keep the same service PID and listener through all A–D phases.

Finally stop normally and require port free, no ZAP-IT/compute process, physical
GPU back to the fresh 15-MiB baseline, shared-memory root empty, no credential or
request content in logs, and no unrelated resource change.

Report every failed attempt and ordinary correction. CUDA OOM, inability to free
the model delta, PID restart, hidden reload, inference race or auth separation
failure is not pass and must not be deferred to Objective 011.

## Scope and non-goals

Expected amendment: focused lifecycle/gate/registry/API/settings/metrics fixes if
live evidence requires them, expanded sanitized smoke harness/tests/docs, generic
wrapper-prompt source, exact 010-b transcript. No new dependency is expected.

Non-goals remain unchanged: no cooperative cross-process lease, second service,
other model software, MPS/MIG/Kubernetes/broker, generic repository, V2 tensor
inference, LAN/public exposure, systemd activation, driver/CUDA mutation,
quantization/offload/substitution, release or arbitrary model/config/device.

## Acceptance and verification

1. PR #66 remains the sole Objective-010 PR with same title/base/branch and a
   bounded 010-b implementation plus final SELF report.
2. Every 010-a CPU/API/default-mode/standards/security criterion remains green;
   the generic wrapper source no longer contains stale fixed-index law.
3. The exact real A–D sequence passes in one PID/listener with two successful
   loads, two real repeated inferences and two measured unloads.
4. Atomic drain and credential separation are proven under real concurrency;
   no inference client can mutate model state.
5. Both unloads prove Torch allocated/reserved <=64 MiB and >=90% model-delta
   release; cold context is bounded/measured and final process stop restores the
   physical baseline.
6. Objective 011 remains necessary and unstarted for cooperative multi-process
   ownership; no such claim appears in this PR.
7. Canonical CPU/coverage, focused lifecycle tests, Ruff, compile, shell syntax,
   docs, package/install/artifact/secret scans and all implementation/report-head
   CI/CodeQL checks pass.
8. Prior transcript/CRITICAL bytes remain unchanged; final host cleanup passes.

## Deferred human adjudication

- Decision: `NONE`

The human and maintained governance explicitly authorize this exact assigned
card. The blocker was stale local wrapper wording, not missing authority or a
new consequential dilemma.

## Publication/report contract

- Amend only PR #66; never merge/auto-merge or create a second PR.
- Push all non-report work, record literal implementation SHA and require all
  seven checks successful.
- Publish only `oap/reports/010-b-report.md` in final SELF; verify parent,
  one-path diff, remote bytes, default-path secret scan and all final checks.
- Report exact timeline, endpoint statuses, auth negatives, PID/listener,
  initialization/idempotency, inference digests, two-cycle Torch/physical/RSS
  table, drain duration, cleanup and every limitation.
- Explicitly answer the strongest reason not to accept: CPU fakes may conceal
  PyTorch references/context memory or a real readiness/admission race. Answer
  only with the unchanged-PID concurrent lifecycle and repeated measured unload
  evidence, not projections.
- Send exact FIFO `OK` only after immutable report and final remote verification.
