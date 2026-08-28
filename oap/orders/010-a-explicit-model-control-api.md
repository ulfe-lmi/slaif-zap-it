# OAP Work Order — 010-a — Standard explicit model-control API and lifecycle

## Objective

Create the sole Objective-010 PR from current remote `main`. Add a privileged,
standard-aligned explicit model-control plane that keeps the HTTP process live
while the fixed ZAP-IT model is unloaded, loads all pinned holders on demand,
drains in-flight inference before unloading, releases model GPU memory, and can
load/infer/unload repeatedly without restarting the process.

Use **load**, **unload**, and repository **index** terminology and the
KServe/Triton Model Repository Extension paths. Do not expose fill, empty,
warmup or cooldown commands. This objective proves one-process lifecycle
correctness. Cooperative cross-process ownership/handoff is reserved for
Objective 011 and must not be claimed here.

## Standards decision

Primary references:

- KServe/Triton Model Repository Extension:
  `POST /v2/repository/index`,
  `POST /v2/repository/models/{model_name}/load`, and
  `POST /v2/repository/models/{model_name}/unload`;
- Triton explicit model-control mode: the server remains running while models
  start unloaded and later load/unload through management requests;
- KServe multi-model serving: load, unload and model health are the integration
  vocabulary.

ZAP-IT implements the fixed-model **management extension subset**, not KServe V2
tensor inference. `/v1/completions` remains the only inference contract. Server
metadata/docs must say this explicitly so the new `/v2` management surface does
not imply full V2 inference compatibility.

Load/unload requests are synchronous: a successful load returns only after the
model is ready; a successful unload returns only after new inference is blocked,
existing inference has drained, holders are released and CUDA allocator memory
meets the cold bound. The process, listener and `/healthz` remain live throughout.

## GitHub and OAP state

- Numeric objective/round: `010 / 010-a`; mode `CREATE_NEW_PR`.
- Repository/default branch: `ulfe-lmi/slaif-zap-it`, `main` at
  `5da3851347c2031bea11012fc554140ba7894cc2`.
- Local `main` is clean and equal to `origin/main`; no PR or issue is open.
- Objective 009 PR #65 is merged as current main. Post-merge CI and CodeQL are
  successful; active/report `009-b` are resolved orchestration history.
- Required branch: `oap/010-a-explicit-model-control-api`.
- Required PR title: `Objective 010: explicit model-control API`.
- Required report: `oap/reports/010-a-report.md` as final SELF child.
- Prior orders/reports and `CRITICAL.md` are immutable. CRIT-0001 remains
  accepted; no open critical gate applies to this local reversible work.

## Reconciled current implementation

- `ResidentRegistry` is one-shot: background load at process start, no unload,
  no repeated load, and shutdown only joins its initial loader thread.
- `InferenceGate` serializes inference but cannot pause admission or drain queued
  and active requests for lifecycle mutation.
- `/healthz`, `/readyz`, `/metrics`, and `/v1/completions` exist; no model-control
  surface or lifecycle index exists.
- Startup imports CUDA, verifies one exact index+UUID, selects capacity strategy,
  creates the registry and always starts background loading.
- Current default behavior is qualified and must remain backward-compatible.

## Mandatory API contract

### Operator mode and credentials

Add immutable operator settings:

```text
SLAIF_ZAP_IT_MODEL_CONTROL_MODE=none|explicit   # default none
SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY=<separate bearer secret>
SLAIF_ZAP_IT_MODEL_CONTROL_DRAIN_SECONDS=<positive bounded timeout>
SLAIF_ZAP_IT_MODEL_CONTROL_OPERATION_SECONDS=<positive bounded timeout>
```

- `none` preserves current behavior: background startup load; load/unload
  mutation requests fail honestly because explicit control is disabled.
- `explicit` starts the listener/process with the model `UNAVAILABLE` and no
  background model load. `/healthz` is 200, `/readyz` is 503 until load.
- Explicit mode MUST fail startup if the dedicated model-control key is absent,
  empty or equal to the inference API key. Management authorization uses
  constant-time comparison and is required even on loopback.
- The inference bearer key never authorizes model control; the control key never
  authorizes inference. Neither key is logged, echoed, placed in metrics, CLI
  arguments, OAP evidence or error details.
- Uploaded YAML/multipart fields cannot select control mode, lifecycle state,
  model identity, operation timeout or credential.

If the new environment-variable identifier creates a secret-scanner false
positive, assess it exactly and add only a narrowly reviewed baseline tuple in
the same PR; do not weaken scanners or add broad exclusions.

### Management subset

Implement:

```text
GET  /v2
POST /v2/repository/index
POST /v2/repository/models/zap-it-1/load
POST /v2/repository/models/zap-it-1/unload
```

- `/v2` returns bounded static server metadata and advertises
  `model_repository`; it clearly identifies this as a management subset.
- Repository index requires the control credential. Accept only a small JSON
  object with optional boolean `ready`; reject malformed bodies and unknown
  fields. With `ready:true`, omit the fixed model unless it is `READY`.
- Index returns one fixed-model entry when applicable:
  `name=zap-it-1`, optional fixed package/schema version, lifecycle `state`, and
  sanitized `reason`. Use uppercase `UNAVAILABLE|LOADING|READY|UNLOADING`;
  a failed load is `UNAVAILABLE` with a generic failure reason.
- Load/unload path model names other than exact `zap-it-1` fail before any
  lifecycle/resource action. Percent-encoding or odd path forms must not bypass
  the fixed-name check.
- Bodies for load/unload may be empty or `{}` only. Reject every parameter,
  config/file/model override, path, URL, revision, device or unknown member.
- Successful load/unload returns HTTP 200 with the standard empty response body.
  Errors return the repository-extension `{"error": "<sanitized>"}` shape and
  correct HTTP status; no stack/path/model internals.
- Fixed immutable load is idempotent: load while `READY` is a 200 no-op with no
  reload/initialization count; unload while `UNAVAILABLE` is a 200 no-op.
- Concurrent/conflicting lifecycle operations are serialized or rejected
  deterministically. Never allocate a second registry/model set.
- OpenAPI/docs must identify these endpoints as privileged local management,
  not inference and not full KServe V2 compliance.

## Lifecycle and concurrency law

Introduce a typed, testable lifecycle controller/state machine, separate from
HTTP transport. Required stable states:

```text
UNAVAILABLE -> LOADING -> READY -> UNLOADING -> UNAVAILABLE
```

- Load failures return to `UNAVAILABLE` and may be retried; record only a
  sanitized error category.
- `READY` is the only inference-admitting state. `/readyz` and completions derive
  from the same authoritative lifecycle/registry verdict; no split-brain window.
- Lifecycle mutation executes outside the event loop in one bounded control
  executor/thread. Health, index and error responses remain responsive during
  multi-minute model load.
- Before unload, atomically stop new inference admissions and reject queued/new
  inference as `not_ready`; drain the already active synchronous inference
  without cancellation. Only then release holders.
- If drain times out, models remain loaded/ready and admission resumes; do not
  partially unload or report success.
- Load keeps inference paused until every required pinned holder is valid and
  ready. An operation timeout/failure leaves inference unavailable but the
  process/control API usable for retry.
- Client disconnect/cancellation of a management HTTP request must not abandon
  a background transition mid-state; finish/drain to a stable state and report
  sanitized evidence.
- Shutdown blocks new inference/control work, drains the executor, unloads
  safely if possible and never deadlocks.

Extend `InferenceGate` (or an equivalent single authority) with race-free pause,
queued-request rejection and active-drain semantics. Cover the exact readiness-
check/admission race, queue depth >0, simultaneous load/unload, timeout and
cancellation cases with deterministic CPU tests.

## Real unload requirements

Make `ResidentRegistry` reusable across at least two complete cycles. Unload must:

- make model states unreachable and unavailable before successful return;
- handle all-resident and sequential holder structures without persisting them
  to disk or keeping hidden global references;
- drop holders, run bounded garbage collection, synchronize CUDA where safe,
  release allocator cache and collect IPC cache where supported;
- update readiness, state, timings and fixed-label metrics honestly;
- verify logical Torch allocated and reserved model memory is each <=64 MiB
  after unload on the assigned host. Record `nvidia-smi` cold-context memory
  separately; a live process may retain a small CUDA context, but it must release
  at least 90% of the measured loaded-model GPU delta and leave enough capacity
  for another service model;
- fail unload (state unavailable/failed, no misleading 200) if the ordered
  memory proof cannot be met. Do not release future ownership/lease state on an
  incomplete unload.

No CUDA context destruction claim is allowed unless measured. No process restart,
worker subprocess, CPU-resident model retention, quantization or model
substitution is required in this objective.

## Metrics, readiness and observability

Add finite-cardinality metrics for model loaded state, lifecycle state,
operation/outcome counts, operation duration and drain duration. Allowed labels
are fixed operation/state/outcome enums only. Reset/update GPU allocated,
reserved, peak and free gauges after load and unload; never emit keys, paths,
request IDs, model-control callers or arbitrary model names.

`/healthz` must remain 200 in every stable/transition state while the process is
healthy. `/readyz` is 200 only in `READY`; cold/loading/unloading/load-failed
states return sanitized 503. The fixed repository index remains queryable with
the control key in every state.

## CPU/API verification

Add exhaustive CPU/fake coverage without CUDA/model downloads for:

- settings/default mode/explicit key separation and invalid values;
- metadata/index schemas, `ready` filtering, body bounds and fixed model name;
- correct control/inference credential separation and constant-time auth path;
- disabled-mode behavior and default startup-load compatibility;
- every legal/illegal state transition, idempotency, retry after load failure,
  concurrent/conflicting calls and one loader at a time;
- inference pause/drain, active completion, queued/new rejection, drain timeout
  rollback, readiness/admission race and operation cancellation;
- unload releasing fake holders exactly once, cleanup error handling, repeated
  load/unload cycles and sanitized failure reason;
- metrics fixed labels/state and absence of secrets/high-cardinality values;
- OpenAPI routes and documentation compatibility caveat;
- shutdown in cold/loading/ready/unloading/failure states.

Preserve every existing CPU/API/profile/CLI test and default `none` behavior.

## Live RTX-3090 qualification

Freshly reverify exact card/process/driver/Torch/cache/host-RAM/`/dev/shm`/port
facts before every live phase. At activation hinton2 has only physical index 0,
UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
`00000000:0B:00.0`, RTX 3090 24,576 MiB, driver 610.43.02, 15 MiB used / 24,109
MiB free and no compute process. Use only:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
SLAIF_ZAP_IT_MODEL_CONTROL_MODE=explicit
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Use two distinct synthetic bearer secrets in the live environment but never
print/report them. Start one process/worker/request on a freshly verified unused
loopback port and prove, without process/listener restart:

1. Cold start: same PID, health 200, ready 503, completion 503, authenticated
   index `UNAVAILABLE`, no model initialization and cold GPU/Torch/RSS snapshot.
2. Auth/contract: missing, wrong and inference-only credentials cannot index or
   mutate; wrong model/body/parameters fail before allocation.
3. First load: observe `LOADING` from a concurrent authenticated index request;
   health stays 200; load returns 200 only after index/readiness are `READY`.
4. Real combined L3 inference returns 200 with all three stages, non-empty
   bounded BLIP3 answers and a content-free semantic digest. Capture loaded GPU,
   Torch and RSS evidence.
5. Drain unload: start one operator-delayed real inference, then unload. Prove
   new/queued inference is rejected, the existing call completes, index exposes
   `UNLOADING`, unload returns only after drain+memory release, and the PID/
   listener/health never change.
6. Cold proof: index `UNAVAILABLE`, ready/completion 503, Torch allocated and
   reserved <=64 MiB, >=90% loaded delta released, no request residue. A second
   unload is an idempotent 200.
7. Second load: observe `LOADING`, return to `READY`, initialization count
   increments exactly once, an already-ready load is a no-op, and the same real
   inference semantic shape/digest succeeds.
8. Second unload returns to the same bounded cold memory within tolerance,
   showing no monotonic GPU/host growth across two cycles.

Then stop normally and require port free, no ZAP-IT/compute process, physical GPU
back to the fresh 15-MiB baseline and `/dev/shm/slaif-zap-it` empty. Report every
failed attempt. Do not use private goat fixtures.

## Scope and non-goals

Expected diff: lifecycle/registry/gate/service/settings/metrics modules, focused
tests and synthetic live harness support, launcher/env/systemd templates,
architecture/API/runbook/security/testing/datasheet docs and exact 010
transcript. No dependency should be needed beyond the standard library/current
stack.

Non-goals:

- no cross-process GPU lease, second live service, other software, GPU scheduler,
  MPS, MIG, Kubernetes, broker or claim that two processes can load safely yet;
- no `/v2/models/*/infer`, generic model repository, upload/download/config/file
  override, arbitrary model name/revision/device or request-selected policy;
- no LAN/public exposure, gateway, TLS, systemd activation, firewall/VPN/driver/
  CUDA/global credential mutation, tag/package upload or release;
- no quantization, CPU/offloaded retained model, worker subprocess, threshold or
  scientific-pipeline change;
- no raw request/body/prompt/answer, bearer secret, cache path, model weight or
  private fixture in Git/logs/OAP evidence.

## Acceptance and verification

1. Exactly one Objective-010 branch/PR exists from `5da3851`, with bounded diff,
   exact active/order and final report-only SELF topology.
2. Management paths/vocabulary/schemas follow the repository-extension subset;
   docs cite primary specifications and disclaim full V2 inference compliance.
3. Dedicated control authentication and fixed-model/body policy prevent
   inference clients or uploaded data from changing lifecycle/GPU state.
4. CPU tests prove the complete lifecycle/concurrency/drain/failure/idempotency/
   shutdown contract and default startup-loaded compatibility.
5. Real cold-load-infer-drain-unload-reload-infer-unload passes in one unchanged
   PID/listener with the exact memory-release and repeatability evidence above.
6. No cross-process ownership claim is made; Objective 011 remains the explicit
   dependency for safe multi-service handoff.
7. Canonical CPU suite/coverage, focused API/lifecycle tests, Ruff, compile,
   shell syntax, docs integrity, package build/install/artifact/secret scans and
   all implementation/report-head CI/CodeQL checks pass.
8. Final cleanup and protected-resource evidence pass; prior transcript and
   CRITICAL bytes remain unchanged.

## Deferred human adjudication

- Decision: `NONE`

The human requested lifecycle control, primary standards resolve terminology,
and a dedicated loopback-only credential plus conservative drain/unload semantics
are least-privilege reversible engineering decisions. Cross-process arbitration
is explicitly deferred until it has its own evidence-gated objective.

## Publication/report contract

- Create only the required Objective-010 PR; coding never merges/auto-merges.
- Push all non-report work, record literal implementation SHA and require every
  current check successful.
- Publish only `oap/reports/010-a-report.md` in the final SELF child; verify
  parent, one-path diff and remote bytes. Use the release helper's default secret
  baseline path after report publication and require all seven final checks.
- Report exact endpoint bodies/statuses, lifecycle timeline, PID/port continuity,
  auth negatives, two-cycle timing/memory tables, drain evidence, metrics,
  cleanup, docs/standards caveat and every limitation.
- Explicitly answer the strongest reason not to accept: an unload endpoint can
  race inference, expose a denial-of-service control to inference clients, or
  report success while PyTorch still owns model VRAM. Answer with separate
  credentials, atomic pause/drain, authoritative state, repeated real memory
  proof and no cross-process overclaim.
- Send exact FIFO `OK` only after immutable report and final remote checks.
