# OAP Work Order — 007-a — Implement adaptive BLIP3 residency and qualify the sequential profile

## Objective

Create the sole Objective-007 PR and implement the approved adaptive model
residency policy without changing the public endpoint shape or legacy CLI
behavior. Cards below 24576 MiB must support BLIP3 by explicitly swapping
SAM2+CLIP and BLIP3 between GPU and host RAM; they must no longer reject BLIP3.
Cards at or above 24576 MiB must select the all-three-resident code path, whose
real qualification is deliberately deferred to `007-b` on this same PR.

In this round, independently prove that the pinned BLIP3 model loads and infers
alone on maelstrom1's cleared 11-GB physical GPU1, then run the two local
academic goat images as in-memory central-50% crops five times each in alternating
order with the real sequential service path. If BLIP3-alone execution fails or
cannot meet the ordered memory bound, report the disqualifying result and stop;
do not substitute, quantize, offload inference to CPU, disable BLIP3 or weaken
the acceptance gate.

This workorder is complete when its implementation, tests, low-card live
qualification, benchmark evidence and immutable report are published. The PR
must remain open and unmerged for `007-b`; coding must not simulate or claim
>=24-GB qualification.

## GitHub state

- Numeric objective / round: `007 / 007-a`.
- Mode: `CREATE_NEW_PR` — exactly one new Objective-007 PR.
- Repository/default base: `ulfe-lmi/slaif-zap-it`, `main`.
- Verified base SHA: `85af4fd562fdf128709a0160bc36884a585e1a5e`,
  human CRIT-0001 acceptance from PR #61.
- Exact-base GitHub CI run `32675037693` and CodeQL run `32675037679` are
  completed SUCCESS. No PR is open at activation.
- Required branch: `oap/007-a-adaptive-blip3-residency`.
- Required PR title: `Objective 007: adaptive BLIP3 model residency`.
- Create the branch from the exact verified remote base, not from the local
  `human-adjudication/crit-0001-accepted` branch tip. Preserve ignored local goat
  files and unrelated local state. Coding never merges, closes or enables
  auto-merge.
- Required future continuation remains `007-b` on this same branch/PR. Do not
  create Objective 008 or a second PR.

## Verified current state

- Existing live policy is hard-coded as
  `sam2_clip_resident_blip3_rejected`; supported profiles are only `sam2`,
  `clip`, and `sam2_clip`. The service registry loads SAM2+CLIP once and always
  passes `blip3_state=None`.
- BLIP3 currently combines the reusable QA/model and per-request label/rule
  mapping in one `_Blip3Filter`, so safe residency requires separating reusable
  model ownership from fresh request rules.
- Pinned models remain:
  - SAM2 `facebook/sam2-hiera-large` at
    `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`;
  - CLIP `openai/clip-vit-base-patch32` at
    `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`;
  - BLIP3 `Salesforce/xgen-mm-phi3-mini-instruct-r-v1` at
    `1d91d356d3b6fbc141140edf490b39890417af44`.
- The cached BLIP3 checkpoint has 4,589,362,243 FP32 parameters across four
  safetensor shards (18,357,535,724 bytes). FP16 resident weights are about
  8.55 GiB before runtime overhead. The prior 10,505-MiB projection was never
  loaded because it crossed the former conservative 90% 11-GB-card guard.
- BLIP3 ordinarily receives a SAM-mask bounding-box patch, minimum 128x128. A
  frame-spanning mask can still create a near-frame CPU patch, but the pinned
  image processor maps any-resolution input to a finite 378-pixel tile grid.
  Preserve this scientific behavior; the central crop bounds SAM2/input cost.
- At 2026-08-24 04:41 CEST, maelstrom1 physical GPU1 is RTX 2080 Ti,
  `11264 MiB`, PCI `00000000:00:0C.0`, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, `6 MiB` used and `10815 MiB`
  free. Physical GPU0 is UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, has unrelated PID 66522 using
  2492 MiB, and is untouchable.
- Host runtime is driver `580.178.04`, PyTorch `2.5.1+cu124`, CUDA runtime
  `12.4`; host `MemAvailable` is about 47 GiB and `/dev/shm` has about 26 GiB
  free. Recheck all facts immediately before live work.
- Local ignored fixtures exist: both JPEGs are 5568x4176 and their exact
  central crops are 2784x2088; `configs/goats2.yaml` contains CLIP and BLIP3
  academic rules. CRIT-0001's latest disposition is human `ACCEPTED`; existing
  package/current-tip exclusions remain defense in depth.
- No ZAP-IT listener is active. Loopback ports must still be freshly selected.
  The LAN listener `10.8.132.72:8000` and both Hinton vLLM services/cards are
  unrelated and immutable.

## Scope and required design

### 1. Operator-only automatic strategy selection

- Replace the rejected-BLIP3 invariant with exactly two named residency modes:
  `sam2_clip_gpu_blip3_cpu_swap` and `sam2_clip_blip3_gpu_resident`.
- Select from the pinned physical device's actual total capacity:
  `<24576 MiB` selects sequential swap; `>=24576 MiB` selects all resident.
  Use a masked, UUID-matched physical `nvidia-smi` capacity observation and
  cross-check CUDA/Torch device facts. Do not classify a marketed 24-GB card as
  low-memory merely because CUDA reserves a small driver region.
- The selector is automatic and operator-owned. Uploaded YAML and multipart
  fields cannot choose, override or reveal a model path, revision, dtype,
  physical index or residency mode. Do not add a request-level selector.
- Preserve the physical-GPU1 launch mask and exact UUID pin. Refuse startup if
  the selected GPU is occupied by an unrelated compute process or the
  index/UUID/PCI/capacity observations disagree. Do not treat occupied free
  memory as a reason to select the smaller strategy.
- Runtime metadata/readiness and fixed-label metrics must identify the selected
  mode and pinned model set honestly without paths, prompts, answers, raw free
  memory, credentials or unrelated process details.

### 2. Residency registry and transition lifecycle

- Generalize the one-process registry to own SAM2, CLIP and BLIP3 model holders,
  their current device residency, sanitized transition state/failure category,
  load/transition timings and the selected strategy.
- Pin service BLIP3 to FP16 on both GPU classes. Uploaded `dtype` is forbidden.
  Keep all downloads disabled (`local_files_only=True` or equivalent) and use
  only the approved cache/revisions.
- Sequential startup must fully initialize BLIP3 in host RAM and SAM2+CLIP on
  logical `cuda:0` before readiness becomes true. This makes request latency
  predictable and proves all required cached assets at startup. Host-resident
  BLIP3 is reused; do not reload the 18-GB checkpoint per request.
- A request without BLIP3 uses resident SAM2+CLIP exactly as today and performs
  no transition.
- For a BLIP3 request, under the existing single-request gate:
  1. run SAM2 and optional CLIP on GPU;
  2. move every reusable SAM2/CLIP tensor to CPU and eliminate all obsolete GPU
     references/caches;
  3. synchronize, collect/empty the CUDA allocator and prove sufficient memory;
  4. move the already initialized pinned BLIP3 holder to logical `cuda:0`;
  5. run bounded BLIP3 verification over the request's masks;
  6. move BLIP3 back to CPU, restore SAM2+CLIP to logical `cuda:0`, synchronize,
     and only then return the response.
- Restoration is mandatory in `finally` for success, module failure, timeout and
  cancellation. The API already drains a timed-out worker before releasing its
  gate; preserve that property. If transition or restoration fails, sanitize the
  client error, mark registry/readiness failed, and require operator restart.
- The >=24-GB path loads all three pinned model holders on GPU at startup and
  uses no request-time transition. Implement and CPU/fake-test it now, but do not
  claim it live-qualified in `007-a`.
- Keep one worker/request. State transitions must be serialized, idempotent and
  safe across repeated requests and shutdown; no second CUDA model copy, stale
  request rules, or fork after CUDA.

### 3. BLIP3 reusable model and request isolation

- Separate the pinned `_Blip3QA`/processor/tokenizer/model holder from mutable
  label rules. Add a request-rule update/use path analogous in intent to
  resident CLIP prompt updates, but do not persist questions, answers, masks or
  request-derived tensors after the call.
- Service YAML may contain only nested BLIP3 verification rules (`question`,
  `trueresult`, `falseresult`, `newcategory`, bounded debug flag). Continue to
  reject `model_name`, `revision`, `dtype`, tokenizer/processor controls,
  devices, paths, URLs, cache/download and remote-code controls anywhere in an
  upload. The local goat harness must preserve safe BLIP3 rules while removing
  those operator/model keys.
- Bound the service path to at most 32 BLIP3 questions per request and at most
  32 generated tokens per question. Limits are fixed operator/service policy,
  never uploaded controls. Reject an over-limit candidate set before the first
  BLIP3 generation with an existing safe bounded-resource error or a narrowly
  added stable error code; do not process an arbitrary prefix silently.
- Legacy CLI keeps its established configuration and generation defaults. Do
  not make service limits alter legacy behavior.
- Preserve the established per-mask bounding-box patch and pinned finite
  any-resolution preprocessing. Do not add lossy crop resizing in this round.

### 4. Public service behavior and observability

- Keep the multipart `/v1/completions` fields, `zap-it-1`, `zap-it.v1`, response
  levels and JSON/ZIP structure unchanged. BLIP3 configurations become supported
  instead of receiving `unsupported_profile`; existing L2/L3
  `blip3_answer`/stage metadata carries real results.
- Supported runtime profiles must include all actual SAM2-rooted combinations:
  `sam2`, `sam2_clip`, `sam2_blip3`, and `sam2_clip_blip3`. Do not claim a
  standalone pipeline that skips mandatory SAM2.
- Add bounded fixed-label counters/histograms for model initialization and
  transition-to-BLIP3/restore durations and failures. No per-label, prompt,
  request ID or client-controlled metric label.
- Preserve the ordinary request deadline initially. If a real sequential goat
  request cannot finish within 120 seconds, record the observed facts and use an
  operator-only BLIP3 deadline setting with a conservative documented default;
  do not weaken non-BLIP deadlines or allow clients to set timeouts.
- `/readyz` must remain honest through loading and terminal failure. A successful
  transition within the sole active request need not make health fail, but no
  readiness response may claim baseline restoration before it is true.

### 5. Local-only benchmark harness

- Extend the existing opt-in harness rather than creating a path that commits or
  copies fixtures. Independently safe-load `goats2.yaml`, allowlist the safe
  algorithmic sections including BLIP3 rules, strip operator/model controls,
  crop both JPEGs to their central 50% in memory, and never persist a derivative.
- Add a benchmark mode that sends exactly ten real BLIP3-enabled requests in
  this order: `A,B,A,B,A,B,A,B,A,B`. Each image therefore runs five times.
  Use one response level/format consistently for latency comparison; prefer L2
  JSON for user-representative metadata without L3 visualization/ZIP cost.
- Collect sanitized end-to-end latency plus server-side SAM2, CLIP, BLIP3,
  transition-to-BLIP3 and restoration timings; startup/model-initialization time;
  peak allocated/reserved GPU memory; host RSS/available-memory deltas; result
  counts and non-content semantic digests sufficient to prove repeatability.
- Report first, minimum, median, nearest-rank p95 and maximum by image and across
  all ten requests. Explicitly distinguish startup, first request and steady
  request latency so users know what to expect.
- Never print or persist source YAML, labels, prompts, BLIP answers, raw response
  bodies, fixture/crop bytes, derivatives or bearer keys. CPU tests use generated
  images/config only. The real benchmark remains local and opt-in.

## Non-goals

- No merge in `007-a`; no Objective 008 or second PR.
- No use or mutation of any Hinton host/card/vLLM process; no GPU0 allocation,
  signal, reset or workload inspection beyond sanitized process evidence.
- No quantization, alternate BLIP/VLM, CPU-only inference, Accelerate layer
  offload, lower precision, silent fallback or disabled BLIP3.
- No 24-GB qualification claim, external/LAN/public deployment, gateway change,
  systemd installation, driver/firewall/network mutation, tag/package/release or
  production/customer data.
- No request-controlled model/runtime/resource policy and no persistent request
  data.
- No goat fixture bytes or derivatives added back to Git or release artifacts.

## Acceptance criteria

1. One PR/branch from exact base implements the bounded design; `007-a` report is
   its final report-only commit and the PR remains open for `007-b`.
2. Boundary tests prove 24575 MiB selects sequential and 24576 MiB selects all
   resident using physical capacity evidence; UUID/process mismatch and occupied
   target fail closed.
3. CPU/fake tests prove exact sequential transition order, no-swap requests,
   host-state reuse, all-resident no-transition behavior, idempotency, rule
   isolation, fixed limits and mandatory restoration on success/error/timeout/
   cancellation. A restoration failure leaves readiness false.
4. API tests replace the old unconditional BLIP3 rejection with successful fake
   BLIP3 requests in both strategies, preserve hostile YAML/model-control
   rejection, and verify stable safe errors/metadata/metrics.
5. Legacy CLI/regression tests remain green; CPU CI performs no CUDA use, model
   download or real-fixture access.
6. Fresh live preflight proves the exact maelstrom1 GPU topology and protected
   resources. GPU1 alone is made visible as logical `cuda:0`; GPU0 and unrelated
   listener/process state remain unchanged.
7. Pinned FP16 BLIP3 loads from local cache and completes at least one bounded
   synthetic inference alone after SAM2+CLIP leave GPU. Peak reserved VRAM must
   not exceed 95% of CUDA-visible total. OOM, unsupported FP16 operation,
   generation failure or excess peak is `BLOCKED` and ends the round without a
   substitute design.
8. The real sequential loopback service becomes ready with BLIP3 host-resident,
   ordinary no-BLIP smoke passes without transition, and all ten alternating
   central-crop goat requests return valid successful responses with BLIP3 stage
   executed. No request remains timed out or partially restored.
9. Across repeated runs, per-image sanitized semantic results are stable, GPU/
   host allocation does not grow monotonically, SAM2+CLIP is restored after each
   request, GPU1 returns to expected baseline after stop, port is free and
   `/dev/shm/slaif-zap-it` is empty.
10. Documentation states exact supported behavior, capacity boundary, host-RAM
    cost, startup and per-image timing distributions, memory peaks, current
    11-GB evidence, unqualified >=24-GB status, operator troubleshooting and the
    pending `007-b` requirement. Remove obsolete claims that BLIP3 is unsupported
    or always rejected, but do not claim warm-all qualification.
11. Full canonical CPU suite/coverage, focused tests, Ruff format/lint, compile,
    shell syntax, package/release artifact verification and every GitHub CI/
    CodeQL check are SUCCESS on implementation and report heads.

## Verification and live evidence

### CPU/static

- Run the repository's full canonical CPU suite with coverage and all focused
  runtime/service/BLIP3/harness tests.
- Run Ruff format check and lint, compileall, shell syntax, package build/install
  smoke, artifact verifier/scanner and `git diff --check`.
- Add generated tests for fixture derivation and exact ten-request A/B ordering;
  no real fixture is used in public CI.

### GPU1/live

Before every live phase, capture sanitized all-GPU and compute-process snapshots,
driver/CUDA/PyTorch versions, physical index/UUID/PCI/name/total/free memory,
host `MemAvailable`, `/dev/shm`, listeners and chosen loopback port. Never touch
GPU0.

Use:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
logical device cuda:0
expected UUID GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
```

Run progressively: pinned BLIP3-only synthetic load/inference and cleanup;
sequential-registry synthetic transitions; authenticated loopback service
readiness/no-BLIP smoke; then the ten-request real goat benchmark. Abort further
live work after the first disqualifying BLIP3 load/inference or memory result.
Use no download flag and no network model retrieval.

Final evidence must include exact commands/exit states, capacity selection,
model load/transition timing, per-image benchmark table, GPU peak allocation/
reservation/free margin, host memory, process snapshots, GPU0 equality,
zero-persistence and final cleanup. Content-bearing output stays local and out of
the report.

### CI/checks

Push implementation before the report, wait for all required current checks and
record each exact check name/conclusion. After the report-only commit, require
checks again. Green CI is necessary but not sufficient.

## Documentation/provenance

Update runtime/GPU/runbook/datasheet/config/API/output-parity/release-gate text as
needed so it consistently describes:

- automatic `<24576` sequential versus `>=24576` all-resident selection;
- real 11-GB sequential evidence and its host-RAM/latency tradeoff;
- all-resident implementation present but not qualified until `007-b`;
- pinned FP16 identities/revisions/licenses and local-cache-only loading;
- BLIP3 per-object patches and finite processor tiling;
- request limits, transition failure/restart behavior and benchmark method;
- no change to loopback/security/no-persistence/GPU0 rules.

Do not claim general accuracy, commercial model clearance, production SLA or
24-GB performance. CRIT-0001 remains accepted and byte-identical.

## Security/resource/protected-host constraints

- Treat image/YAML as hostile. Preserve safe YAML composition limits and reject
  every model/path/device/network/code control.
- Model transitions and metrics must not log request content or cache paths.
- One process/worker/request, bounded queue and bounded BLIP questions/tokens.
- No request persistence; temporary artifacts only under the validated
  `/dev/shm` workspace with cleanup.
- Never touch physical GPU0, PID 66522 or successor unrelated workload, LAN
  listener `10.8.132.72:8000`, Hinton hosts/GPUs/vLLM endpoints, firewall,
  drivers, system services or repository settings.
- Preserve ignored local goat assets exactly. Never stage them; verify release
  artifacts continue excluding them.

## Deferred human adjudication

- Decision: `NONE`
- `CRITICAL.md` must remain byte-identical. This reversible measured runtime
  policy does not meet the five-entry threshold.

## GitHub publication

- Create exactly branch `oap/007-a-adaptive-blip3-residency` from verified remote
  main and exactly one PR titled `Objective 007: adaptive BLIP3 model residency`.
- The PR description must state that `007-a` qualifies only the sequential
  11-GB profile and the PR intentionally remains open pending `007-b` on an
  exclusive >=24-GB GPU.
- Commit/push all non-report changes first. Capture the literal 40-hex
  implementation SHA and wait for every check.
- Then create only `oap/reports/007-a-report.md` in a final SELF commit whose
  first parent is that implementation SHA. Prior orders/reports and CRITICAL are
  immutable. Push and wait for all report-head checks.
- Coding never merges/closes the PR, signals another objective, tags, releases or
  starts `007-b`.

## Required immutable report evidence

Use the repository report template and include:

- status and exact base/branch/PR/head/implementation/report topology;
- bounded diff and every changed path grouped by behavior;
- exact CPU/static/API/fake/legacy/package commands and outcomes;
- every GitHub check name/status/conclusion on implementation and report heads;
- fresh device/process/host/shm/port facts before and after live work;
- BLIP3-only dtype/load/inference/memory result and the 95% calculation;
- startup/no-BLIP/sequential-transition evidence;
- the sanitized ten-request A/B benchmark table and aggregate statistics;
- state isolation, restoration, memory-stability, zero-persistence and GPU0
  proofs;
- honest unqualified >=24-GB limitation and exact `007-b` prerequisite;
- `Deferred human adjudication: NONE` and byte-identical CRITICAL proof;
- explicit strongest reason not to accept `007-a`: first-ever near-capacity
  BLIP3 execution and swap complexity on the 11-GB card; answer it with measured
  headroom, repeated restoration, failure-path tests and preserved PR hold.

Send exact FIFO bytes `OK` only after the immutable report is remotely verified,
then exit. FIFO acknowledgment is not acceptance or merge authorization.
