# OAP Work Order — 007-b — Correct stage residency and complete sequential BLIP3 qualification

## Objective

Amend the sole Objective-007 PR #62. Treat the `007-a` constructor-keyword
failure as an ordinary remediable implementation defect per the human's explicit
instruction, “this is not a blocker, continue!” Correct the already identified
sequential stage-order defect, rerun the pinned BLIP3-only FP16 gate with the
corrected loader, and—only after that gate passes—complete the real low-card
loopback service and alternating five-per-image goat benchmark originally
ordered in `007-a`.

This continuation does not qualify or merge the >=24-GB all-resident profile.
That closure moves to future `007-c` on the same PR after an exclusive qualifying
card exists. Coding must keep PR #62 open and unmerged.

## GitHub state

- Numeric objective / round: `007 / 007-b`.
- Mode: `AMEND_EXISTING_PR`.
- Repository/default base: `ulfe-lmi/slaif-zap-it`, `main` at
  `85af4fd562fdf128709a0160bc36884a585e1a5e`.
- Sole branch/PR: `oap/007-a-adaptive-blip3-residency`, PR #62,
  `Objective 007: adaptive BLIP3 model residency`.
- Verified current PR/report head:
  `233e5def21aacf2283a5a5c5ced9e00452c361c5`, report-only `007-a`; its
  first parent is implementation `f60db64e4a22b23995e45a5720d69103d9615512`.
- PR #62 is open, non-draft and mergeable. All exact report-head checks are
  SUCCESS: static, release artifact audit, Python 3.10/3.11/3.12, Analyze and
  CodeQL. No check is pending, failed or missing.
- Required new immutable report: `oap/reports/007-b-report.md`. Prior order and
  report bytes are immutable. `CRITICAL.md` remains byte-identical at Git object
  ID `29c2366359fb5a05e151fe546bcf6330477f60ee`.

## Reconciled findings and mandatory corrections

### 1. The first live failure is remediable, not a capacity verdict

`007-a` proved the correct physical GPU1 mask and local pinned cache, but the
remote constructor rejected `dtype=` before allocating model weights. The
implementation changed this call to standard Transformers `torch_dtype=` and
fixed service dtype to FP16, but did not rerun it. The human explicitly overrode
the stop interpretation. Resume progressively and fix further ordinary API/
loader integration defects in this round rather than treating them as capacity
disqualifiers.

The hard disqualifier remains actual evidence that the corrected pinned FP16
model cannot complete bounded inference on the cleared 11-GB GPU1: CUDA OOM,
intrinsic unsupported FP16 execution, generation failure after successful
construction that is not a correctable integration bug, or peak reserved VRAM
above 95% of CUDA-visible total. If that hard gate occurs, stop without
quantization, CPU inference/offload, substitution or disabling BLIP3.

### 2. Strategic review found the swap at the wrong stage boundary

Current `ResidentRegistry.execute()` moves SAM2 and CLIP to CPU and BLIP3 to GPU
**before** calling the whole `run_single_image` pipeline. Therefore a real BLIP3
request would attempt SAM2 and CLIP after their models left GPU, contrary to the
approved sequence and likely with CPU/CUDA device mismatch. The existing fake
test asserts only transition order around a runner that never executes stages,
so it does not detect this defect.

Correct the lifecycle so:

1. SAM2 and optional CLIP execute on logical `cuda:0` with their resident models;
2. only after CLIP completes and immediately before the first BLIP3 operation,
   SAM2+CLIP move to CPU, CUDA is synchronized/cleared, available memory is
   checked, and BLIP3 moves to GPU;
3. BLIP3 executes on GPU over the already produced masks/labels;
4. BLIP3 returns to CPU and SAM2+CLIP return to GPU in mandatory `finally` before
   the engine returns or propagates an error.

Implement this through an explicit internal stage hook/wrapper or equivalent
typed lifecycle boundary. Do not split the public HTTP endpoint, persist
intermediate masks, rerun SAM2/CLIP, or move baseline models before their stages.
The normal no-BLIP path and >=24-GB path perform no swap.

### 3. Strengthen evidence before live service

- Add CPU/fake stage-aware tests whose fake SAM2 and CLIP assert their holders
  are GPU-resident when called, whose fake BLIP3 asserts baseline holders are on
  CPU and BLIP3 is on GPU, and whose post-call assertions prove baseline restore.
- Cover SAM2+BLIP3 without CLIP, SAM2+CLIP+BLIP3, no-BLIP, all-resident,
  pre-transition SAM2/CLIP failure, BLIP3 failure, timeout/cancellation drain and
  restoration failure. A pre-transition failure must not invent a BLIP3 move;
  every post-transition outcome restores or terminally fails readiness.
- Verify the corrected `torch_dtype=` path reaches a fake remote constructor
  without passing deprecated `dtype=`, uses pinned FP16, local-only snapshots,
  and keeps uploaded dtype/model/revision controls rejected.
- Preserve actual candidate-question preflight (not merely rule-count bounds),
  32 generated tokens, request-local rule state, fixed-label metrics and legacy
  CLI semantics.
- Re-run full CPU/static/package/release verification before any GPU work.

## Live qualification sequence

Freshly reverify before each phase. At activation, maelstrom1 physical GPU1 is
still RTX 2080 Ti, 11264 MiB, PCI `00000000:00:0C.0`, UUID
`GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, 6 MiB used / 10815 MiB free; GPU0
still has unrelated PID 66522 using 2492 MiB. These are observations, not
permanent truth.

Use only:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
logical cuda:0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### Phase A — corrected BLIP3-only hard gate

- Initialize the pinned holder in host RAM with FP16 and local-files-only, move
  it to GPU1, and prove the constructor, tokenizer/processor and model identity/
  revision without downloading.
- Run a bounded generated 128x128 image with a yes/no question and 32 generated
  token cap. Synchronize and capture load time, inference time, Torch
  allocated/reserved peaks, CUDA free/total and sanitized nvidia-smi snapshots.
- Require a non-empty bounded answer, no OOM, no intrinsic FP16 error, peak
  reserved `<= 0.95 * CUDA-visible total`, and full cleanup back to the observed
  baseline. Ordinary correctable library-argument/adapter mistakes may be fixed
  and retried in this same round; report every attempt honestly.

### Phase B — real sequential service and no-BLIP control

- Start exactly one authenticated loopback service on a freshly verified free
  port with pinned GPU1/cache and the automatic `<24576 MiB` sequential mode.
  Startup must become ready only after SAM2+CLIP are GPU-resident and BLIP3 is
  initialized/retained in host RAM.
- Prove runtime metadata reports `sam2_clip_gpu_blip3_cpu_swap` and all pinned
  models, with no path/free-memory leakage.
- First send a generated no-BLIP request and prove no residency transition and
  preserved supported baseline latency/output.
- Send a generated BLIP3 request and prove stage-aware residency, response
  success, real `blip3` executed status/answer metadata, transition/restore
  metrics, and SAM2+CLIP GPU baseline restored before the response completes.
- Exercise one BLIP3 inference failure or bounded test injection and one
  timeout/cancellation path without content logging; the worker drains and
  restores correctly or readiness fails terminally as designed.

### Phase C — required goat benchmark

- Use the ignored local `goats1.jpg`, `goats2.jpg` and `goats2.yaml` only through
  the updated local harness. Both images are 5568x4176 and must be cropped to
  2784x2088 central-50% arrays in memory; never persist derivatives.
- Preserve the safe CLIP and nested BLIP3 academic rules while stripping model,
  dtype, revision, path/device/network and other operator controls.
- Run exactly ten BLIP3-enabled L2 JSON calls in order
  `A,B,A,B,A,B,A,B,A,B`. Require HTTP 200 and executed BLIP3 stage for every
  call. If the fixed 120-second deadline is inadequate, add/use only the
  documented operator-owned BLIP3 deadline setting; never make timeout
  client-selectable or weaken non-BLIP deadlines.
- Record sanitized startup/model initialization; per-request end-to-end, SAM2,
  CLIP, transition-to-BLIP3, BLIP3 and restore timing; first/minimum/median/
  nearest-rank-p95/maximum per image and aggregate; peak allocated/reserved/free
  GPU memory; host RAM; object/answer counts and content-free semantic digests.
- Require per-image repeatability, no monotonic GPU/host growth, BLIP3 retained
  in host RAM, baseline GPU residency after every call, zero request persistence,
  and no raw YAML/labels/prompts/answers/body/fixture bytes in evidence.

Stop the service after evidence. Require port free, GPU1 back to baseline/no
ZAP-IT process, `/dev/shm/slaif-zap-it` empty, and GPU0/unrelated services
unchanged.

## Scope and non-goals

Expected scope is a focused correction to runtime/core stage lifecycle, tests,
benchmark/runtime evidence and the docs/timing tables already introduced in
`007-a`, plus exact `007-b` OAP transcript. Correct further ordinary loader API
compatibility defects encountered before the hard resource verdict.

Non-goals:

- no merge, Objective 008, second PR, or >=24-GB live qualification;
- no quantization, model substitution, CPU-only inference, partial layer
  offload, disabled BLIP3 or silent downgrade;
- no Hinton/GPU0/vLLM/LAN/firewall/driver/systemd/gateway/release mutation;
- no request schema/model ID/schema version change and no persistent request
  content;
- no goat bytes, crops, raw configuration or response content committed or
  reported.

## Acceptance and verification

1. PR #62 remains the unique open Objective-007 PR with unchanged base/branch/
   title and contains a bounded `007-b` implementation commit plus final
   report-only child.
2. Stage-aware tests and actual service evidence prove SAM2/CLIP run before the
   swap on GPU, BLIP3 runs after the swap on GPU, and restoration precedes
   return. The flawed pre-runner swap is removed.
3. Corrected pinned FP16 BLIP3-only Phase A passes the 95% hard gate and cleanup.
4. Sequential startup/readiness, no-BLIP control, real BLIP request, failure/
   timeout recovery and fixed-label metrics all pass on physical GPU1.
5. All ten alternating cropped-goat requests pass with real BLIP3 execution,
   stable sanitized semantics, complete timing/resource table and no persistence.
6. Docs replace `007-a` blocked placeholders with exact measured 11-GB startup,
   memory and latency evidence while retaining honest pending `007-c` warm-all
   status.
7. Full canonical CPU suite/coverage, focused adaptive/BLIP/API/harness tests,
   Ruff, compile, shell syntax, package build/install/artifact/secret checks and
   every GitHub CI/CodeQL check are successful on implementation and report
   heads.
8. Prior orders/reports and CRITICAL are byte-identical; deferred human
   adjudication remains NONE.

## Deferred human adjudication

- Decision: `NONE`

## GitHub publication and report

- Amend only PR #62/its existing branch. Do not change title/base or create a PR.
- Commit/push all non-report changes first and record the literal implementation
  SHA. Wait for all required checks.
- Create only `oap/reports/007-b-report.md` in the final SELF commit; it must
  change only that path and have the implementation SHA as first parent. Push,
  verify remote bytes/topology and wait for every report-head check.
- Report exact correction diff, all loader attempts, CPU/live commands, phase
  results, ten-run sanitized table/statistics, peak/headroom calculations,
  startup/transition metrics, cleanup/GPU0 proof, CRITICAL immutability, current
  PR state and the remaining exclusive >=24-GB prerequisite for `007-c`.
- Explicitly answer the strongest reason not to accept this round: `007-a`'s
  fake lifecycle tests missed a stage-order defect and no prior real BLIP3 load
  succeeded. Answer only with new stage-aware tests and complete real repeated
  evidence—not projections.
- Coding never merges, starts `007-c`, or signals another order. Send exact FIFO
  `OK` only after immutable report verification, then exit.
