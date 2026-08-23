# OAP Work Order — 004-a — Loopback service activation on physical GPU1

Objective `004-a`. Turn the proven API contract and measured GPU runtime into a
real local service on the target host. Activate exactly one ZAP-IT service
process bound only to `127.0.0.1` on a freshly verified unused port, expose only
physical GPU1 through visibility masking, pin/check the expected GPU UUID,
enforce the supported model profile/resource strategy selected in Objective 003,
and demonstrate health, readiness, all response levels, cleanup, bounded
concurrency, restart and rollback under real inference. No LAN/public exposure
and no GPU0 use.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `004 / 004-a`
- Mode: `CREATE_NEW_PR`
- Objective 003 merged on remote `main`: squash merge commit
  `1a4272d60c52cc045f57f2842652485efdb7a55c` (PR #47); post-merge CI+CodeQL
  green — prerequisite satisfied.
- Verified default branch/base SHA: `main` @
  `1a4272d60c52cc045f57f2842652485efdb7a55c` (verify live again immediately
  before branch creation).
- Required branch: `oap/004-a-loopback-service-activation`
- Required PR title: `Objective 004-a: loopback service activation on physical
  GPU1 with live E2E evidence`
- Existing objective-004 PR: none expected (confirm none open at round start).
- Objective-003 tested GPU1 UUID:
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`; supported strategy
  `sam2_clip_resident_blip3_rejected` with profiles `sam2`, `clip`, `sam2_clip`
  (measured combined peak ~3.66 GiB allocated / ~5.53 GiB reserved against
  ~10.8 GiB free).
- Objective-003 selected candidate loopback port: `127.0.0.1:17891` — strategic
  re-verified it UNUSED immediately before this publication; it MUST be
  re-verified live again at activation time. Selection is not a reservation.

## Verified current state (strategic live inspection at publication; re-verify at execution)

- All GPUs/processes: index 0 UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`,
  PCI 00000000:00:08.0, RTX 2080 Ti, 11264 MiB total, 2161 MiB used solely by
  unrelated PID 66522 — PROTECTED throughout. Index 1 UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI 00000000:00:0C.0, RTX
  2080 Ti, 11264 MiB total, 6 MiB used / 10815 MiB free — TARGET, currently
  free of compute processes.
- Expected mapping: with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`, the process sees exactly one device as logical
  `cuda:0` whose UUID must equal the pinned value above (proven by Objective
  003 qualification; guard code exists in `src/runtime/device.py`).
- Repo/runtime environment and approved model revisions: pip venv
  `.venv-gpu` per committed `requirements-gpu-cu124.lock` (Torch 2.5.1+cu124,
  torchvision 0.20.1, torchaudio 2.5.1, Transformers 4.41.1, Accelerate
  0.32.1, SAM2 source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`);
  pinned HF revisions recorded in `src/runtime/models.py` and
  THIRD_PARTY_NOTICES.md; weights already present in the operator cache
  (~22 GiB) — do NOT re-download; never commit weights.
- Supported service profile and margin: resident SAM2+CLIP; BLIP3 profiles are
  configuration-rejected before load (Objective 003 hard-stop law stands).
- `/dev/shm`: tmpfs 27G empty; `src/runtime/shm.py` enforces mode-0700 dirs /
  mode-0600 files under `/dev/shm/slaif-zap-it`.
- Listener scan proving selected port unused: `ss -tln` shows no 17891 binding;
  existing unrelated listeners include 127.0.0.1:32995, cups 631, DNS sockets
  and unrelated LAN workload 10.8.132.72:8000 — all untouchable.
- Current API limits/auth/concurrency defaults (frozen in Objective 002):
  multipart cardinality enforcement, image ≤20 MiB, config ≤256 KiB, decoded
  pixels ≤64 MP, response ≤256 MiB, deadline 120 s, queue depth default 0 with
  HTTP 503 `service_busy` + Retry-After, optional bearer auth via operator env
  `SLAIF_ZAP_IT_API_KEY` (loopback default: no key), stable sanitized error
  envelope, `/healthz` + `/readyz` with injected readiness provider.
- Operator launch mechanism already present or preferred: none installed yet.
  Frozen decision below.

## Required runtime invariants

Service environment must include at least:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
SLAIF_ZAP_IT_HOST=127.0.0.1
SLAIF_ZAP_IT_PORT=<freshly verified unused port>
SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it
```

Exact variable names follow the implemented package (`SLAIF_ZAP_IT_*` policy);
semantics must stay unambiguous. Inside the service only logical cuda:0 is
used. One Uvicorn worker, one process, one active inference slot, no reload.

## Frozen strategic decisions (binding)

- Launcher: a repo-owned operator script (e.g. `scripts/serve_local.sh` plus
  `scripts/serve_local_stop.sh`) managing exactly one background Uvicorn
  process with pidfile/logfile under a private runtime dir is the PRIMARY
  tested mechanism. A user-level systemd UNIT FILE is provided as an optional
  template under `deploy/` with install/start/stop documented — tests MUST NOT
  require installing/enabling it. No Docker this objective.
- End-of-round final state: the service is STOPPED, the port free, GPU1 back
  near-idle, `/dev/shm/slaif-zap-it` clean. Nothing runs unattended after the
  round; the human/operator starts the service deliberately via the runbook.
  Restart tests happen inside the round, then shut down again.
- Port selection: prefer 17891 after fresh verification; fallback 23654 then
  any verified-unused port in 20000–40000 chosen live. Record actual choice +
  verification evidence in the report and runbook notes.
- Service profile: start ONLY `sam2_clip` resident registry; readiness reports
  supported profiles; requests selecting BLIP3-dependent configs receive the
  stable unsupported/not-ready rejection defined in Objective 002/003 code.

## Scope

1. Finalize operator configuration separating deployment settings from request
   YAML (bind/port, expected UUID, pinned model IDs/revisions/cache root,
   limits, auth, temp root, queue/busy behavior, logging policy); secrets stay
   outside Git with restrictive permissions if any are introduced.
2. Fail-closed startup/device guard before model init: validate visible count,
   logical cuda:0 identity, expected physical UUID; wrong/missing device means
   startup failure/not-ready, never silent GPU0/CPU fallback in strict mode.
3. Implement the frozen launcher(s) with exact environment, logs, pidfile,
   start/stop/status semantics and safe re-entry (stale pidfile handling).
4. Bind loopback only; assert via `ss` that no wildcard/LAN listener exists.
5. Load only the supported profile; enforce the resource strategy in readiness/
   request validation exactly as measured.
6. Verify one worker/process owns the models; verify GPU residency belongs to
   that single PID.
7. Real health/readiness transitions observed live (not-ready during load,
   ready after registry+shm checks, honest failure states).
8. Real E2E L0–L3 JSON and ZIP calls with small redistributable fixtures and
   representative safe YAML; validate YOLO text, uint16 identity PNG dims/dtype/
   ids, object metadata, bounded full artifacts against CPU-contract invariants.
   Never require BLIP3.
9. Capture latency, peak/end GPU1 memory, response sizes and `/dev/shm` residue
   per profile; assert no request-derived persistence in repo/cwd/disk.
10. Live concurrency/busy proof with overlapping real requests (one inference
    at a time; deterministic 503 behavior; rejected requests allocate nothing).
11. Failure/cancel paths live: invalid input, safely injectable inference error,
    timeout/cancel, response-too-large — no model-state corruption, no residue.
12. Bounded repeated-request stability across supported profiles; record
    VRAM/RSS trend and confirm no cross-request leakage; no soak overclaims.
13. Graceful stop/restart of ONLY the ZAP-IT-owned process; verify port release,
    GPU1 cleanup, then restart → ready → E2E → final controlled shutdown.
14. Documented rollback returning host to pre-004 state (no data/GPU0/firewall/
    global-runtime mutation).
15. Log-safety inspection: identifiers/timings/counts/status only.
16. Operator runbook: start/stop/status/logs/health/readiness/E2E smoke/
    rollback with exact GPU and port checks; document the port as runtime
    selection, not universal.

## Non-goals

- no LAN/public exposure, TLS, reverse proxy or firewall changes;
- no physical GPU0 use;
- no multi-worker/multi-GPU scale-out;
- no persistent request/result storage;
- no job queue/background processing;
- no scientific model change unless separately justified from 003 evidence;
- no broad Docker/Compose/gateway/release work;
- no production/customer data;
- no service left running after the round (see frozen decisions).

## Acceptance criteria

1. One documented local service instance runs on only
   `127.0.0.1:<verified-port>` DURING tests; nothing remains listening after.
2. Startup/readiness verifies exact expected physical GPU1 UUID; process sees
   exactly one CUDA device as logical cuda:0.
3. GPU0 process/memory shows no ZAP-IT allocation before/during/after.
4. Exactly one worker/process owns the model profile; at most one concurrent
   inference, proven live.
5. `/healthz` and `/readyz` show distinct, observed real transitions.
6. Real E2E succeeds for every supported verbosity/format/profile; output
   invariants match the CPU contract; unsupported stages fail honestly/stably.
7. Request artifacts memory or service-owned `/dev/shm`; success/failure/
   timeout/cancel leave zero per-request residue.
8. Overlapping live requests show deterministic bounded busy behavior.
9. Repeated requests within the documented window show no obvious unbounded
   VRAM/RSS growth or state leakage.
10. Graceful stop/restart works; rollback documented and demonstrated where
    non-destructive; final state stopped/clean per frozen decision.
11. No firewall/VPN/global OpenCode/driver/CUDA/unrelated-service change occurs.
12. CPU CI/CodeQL green; live service evidence reported separately, honestly.
13. Runbook reproduces operation without exposing secrets.
14. Correct one-PR/report-only SELF contract satisfied; coding never merges.

## Required verification

- Fresh listener scan + port selection evidence (before/at/after) — reported
- All-GPU/process snapshots before/start/during-E2E/after-stop — reported
- Service environment + in-process device UUID/count proof — reported
- Process/worker count and model-residency ownership — reported
- Health/readiness transition log — reported
- Real E2E L0/L1/L2/L3 JSON (+ZIP as supported) results — reported
- Identity PNG/YOLO/object invariant checks on REAL outputs — reported
- Concurrency/busy live test — reported
- Timeout/cancel/failure/cleanup live tests — reported
- Repeated-run VRAM/RSS/residue measurements — reported
- Graceful restart demonstration + final shutdown + rollback commands — reported
- Log-content safety inspection — reported
- Canonical CPU suite/Ruff/build/CI/CodeQL regression — PASSED (six GitHub
  checks SUCCESS on head; suite green; coverage gate held)
- Read-only GPU snapshots included in report; GPU0 byte-stable throughout

## Documentation and provenance

Update operator runbook (docs/RUNBOOK.md or equivalent), README navigation,
tested service profile, runtime variables, endpoint description, model
revision/resource limits, health/readiness/auth policy, known unsupported
stages and rollback. Port is a runtime selection. Do not claim external
deployment readiness — LAN/public exposure stays behind later human gates.

## Security/resource constraints

Authorized mutations: repo-owned/user-level ZAP-IT runtime artifacts,
`/dev/shm/slaif-zap-it`, the verified loopback listener during tests, and an
OPTIONAL user-level systemd unit file shipped uninstalled. Do not touch GPU0,
other services/processes, firewall/VPN, system NVIDIA/CUDA, global provider
credentials or production data. Any secret lives privately outside Git with
restrictive permissions and is never printed or committed.

## Deferred human adjudication

- Decision: `NONE`

Launcher mechanics, exact local port, measured service limits and the
stopped-at-round-end policy are routine reversible operator decisions fully
inside already-authorized local scope. Loopback-only operation exposes no
external boundary; LAN/public exposure and release remain behind explicit
human gates in later objectives. Do not create a critical entry merely because
VRAM is tight — support only the measured profile honestly. If implementation
exposes a genuine five-condition dilemma (e.g. a forced scientific-model swap),
report it as a candidate instead of deciding silently and continue all
unambiguous safe scope. Coding may not invent the entry.

## GitHub publication and report

Create exactly one branch `oap/004-a-loopback-service-activation` from remote
`main` @ `1a4272d60c52cc045f57f2842652485efdb7a55c` and exactly one PR titled
as specified. Version code/config templates/runbook — never secrets, weights
or transient host logs. Push all implementation before the final report-only
SELF commit (literal implementation SHA parent, single report path, `Report
publication commit: SELF`); exercise/fix in-scope CI; never merge; send
response OK only after remote head/parent/bytes verification. Report exact
listener scans, GPU isolation evidence, process/worker facts, model profile,
E2E outputs, concurrency, resource/residue measurements, restart/rollback,
final stopped-state proof, CI status and limitations.
