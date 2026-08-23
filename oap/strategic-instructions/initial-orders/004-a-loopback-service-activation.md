# OAP Work Order — 004-a — Loopback service activation on physical GPU1

> DRAFT UNTIL Objective 003 is merged and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** preloaded human engineering intent. Strategic may refine the concrete launcher/operator mechanism after Objective 003, but must preserve loopback-only exposure, physical-GPU1 isolation, one-worker/one-inference operation, rollback and complete live E2E evidence.

## Objective

Turn the proven API contract and measured GPU runtime into a real local service on
the target host. Activate exactly one ZAP-IT service process bound only to
`127.0.0.1` on a freshly verified unused port, expose only physical GPU1 through
visibility masking, pin/check the expected GPU UUID, enforce the supported model
profile/resource strategy selected in Objective 003, and demonstrate health,
readiness, all response levels, cleanup, bounded concurrency, restart and rollback
under real inference. No LAN/public exposure and no GPU0 use.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `004 / 004-a`
- Mode: `CREATE_NEW_PR`
- Objective 003 merged on remote `main`, merge SHA/checks: VERIFY:
- Verified default branch/base SHA: VERIFY:
- Required branch/PR title: VERIFY:
- Existing objective-004 PR: N/A after strategic confirms none: VERIFY:
- Objective-003 tested GPU1 UUID and supported profile: VERIFY:
- Objective-003 selected candidate loopback port: VERIFY:

## Verified current state

Immediately before activation, replace with live evidence:

- all GPU/device/process state, especially GPU0 protected workload and GPU1 free capacity: VERIFY:
- exact expected GPU1 UUID and visible-device mapping: VERIFY:
- repo/runtime environment and approved model revisions: VERIFY:
- supported service profile and measured peak VRAM margin: VERIFY:
- `/dev/shm` capacity/root/permissions: VERIFY:
- listener scan proving selected port is still unused: VERIFY:
- current API limits/auth/concurrency defaults: VERIFY:
- operator launch mechanism already present or preferred for this host: VERIFY:

The port must be checked again; Objective 003 selection is not a reservation.

## Required runtime invariants

Service environment must include at least:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
SLAIF_ZAP_IT_EXPECTED_GPU_UUID=<verified physical GPU1 UUID>
SLAIF_ZAP_IT_HOST=127.0.0.1
SLAIF_ZAP_IT_PORT=<freshly verified unused port>
SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it
```

Exact variable names may follow the implemented package, but semantics must be
unambiguous. Inside the service, only logical `cuda:0` is used. One Uvicorn worker,
one process, one active inference slot. No auto-reload in service operation.

## Scope

1. **Finalize operator configuration.** Separate deployment/operator settings from
   request YAML: bind address/port, expected GPU UUID, model IDs/revisions/cache,
   service limits, auth setting, temp root, queue/busy behavior and logging policy.
   Secrets remain outside Git.
2. **Implement a fail-closed startup/device guard.** Before model initialization,
   validate visible-device count, logical cuda:0 identity and expected physical
   GPU1 UUID. Wrong/missing device means not-ready/startup failure, never silent
   GPU0/CPU fallback in strict service mode.
3. **Choose a maintainable local launcher.** Provide a documented host-native
   launch command/script and, if appropriate on the target machine, a user-level
   systemd unit. Prefer user-level service ownership over system-wide mutation.
   Do not introduce Docker merely for appearance if native GPU operation is the
   proven path; container polish can remain Objective 006.
4. **Bind loopback only.** Start on `127.0.0.1:<verified-port>`. Assert with `ss`
   that no `0.0.0.0`, LAN or IPv6 wildcard listener exists unless explicitly and
   separately justified. Do not alter firewall/VPN because loopback is sufficient.
5. **Load only the supported model profile.** Apply the Objective-003 resource
   strategy exactly. Unsupported BLIP3/full configurations must receive a stable
   rejection/not-ready behavior, not trigger opportunistic loads that risk OOM.
6. **Use one worker and one active inference.** Verify actual process count and GPU
   model residency. Do not use multiple Uvicorn workers or forking after CUDA.
7. **Real health/readiness semantics.** `/healthz` succeeds once the process is
   functioning. `/readyz` becomes success only after operator config, expected
   device, model registry and shared-memory requirements are ready. During model
   load/failure it must remain not-ready.
8. **Real E2E L0–L3 calls.** Use small redistributable fixtures and representative
   safe YAML configs. Exercise JSON and ZIP where applicable, validate YOLO,
   identity PNG, object metadata and bounded full artifacts against invariants.
   Do not require an optional stage that Objective 003 declared unsupported.
9. **Resource and residue evidence.** For each E2E profile capture latency, peak/
   end GPU1 memory, host memory where useful, response sizes and `/dev/shm`
   cleanup. Assert no request images/config/results persist in repo/cwd/disk.
10. **Concurrency/busy live test.** Submit overlapping requests and prove only one
    inference runs at a time. Verify configured bounded queue/rejection and that a
    rejected request does not allocate model/request state unexpectedly.
11. **Failure/cancel tests.** Exercise invalid input, inference error where safely
    injectable, timeout/cancel and response-too-large behavior without corrupting
    model state or leaving shared-memory residue.
12. **Repeated-request stability.** Run a bounded series across supported profiles
    to detect obvious VRAM/RSS growth, cross-request object/artifact leakage or
    stale config/model state. Record evidence; do not overclaim production soak.
13. **Graceful shutdown/restart.** Stop only the ZAP-IT-owned service, verify port
    and GPU1 request/model process cleanup as designed, then restart and prove
    readiness/E2E again. Never stop unrelated GPU processes/services.
14. **Rollback.** Document exact commands to stop/disable/remove any user service
    and return host state to pre-004 condition without touching data, GPU0,
    firewall or global runtime.
15. **Operational logs.** Validate logs contain only safe identifiers/timings/
    counts/status and do not expose raw image/YAML, secrets, host paths, prompts or
    customer content.
16. **Operator runbook.** Add start/stop/status/log/health/readiness/E2E/rollback
    instructions with exact GPU and port checks.

## Non-goals

- no LAN/public exposure, TLS, reverse proxy or firewall changes;
- no physical GPU0 use;
- no multi-worker/multi-GPU scale-out;
- no persistent request/result storage;
- no job queue/background processing;
- no scientific model change unless separately justified from 003 evidence;
- no broad Docker/Compose/gateway/release work;
- no production/customer data.

## Acceptance criteria

1. One documented local service instance runs on only `127.0.0.1:<verified-port>`.
2. Startup/readiness verifies exact expected physical GPU1 UUID and the process
   sees only one CUDA device as logical cuda:0.
3. GPU0 process/memory state shows no ZAP-IT allocation before/during/after tests.
4. Exactly one service worker/process owns the supported model profile and at most
   one inference executes at a time.
5. `/healthz` and `/readyz` have distinct, observed real behavior.
6. Real inference E2E succeeds for every supported verbosity/format/profile and
   output invariants match the CPU contract. Unsupported stages fail honestly.
7. Request artifacts are memory or service-owned `/dev/shm`; success/failure/
   timeout/cancel leave no per-request residue.
8. Overlapping live requests exhibit deterministic bounded busy/queue behavior.
9. Repeated requests show no obvious unbounded VRAM/RSS or state leakage within the
   documented test duration.
10. Graceful stop/restart works and rollback returns the host to a known state.
11. No firewall/VPN/global OpenCode/driver/CUDA/unrelated service change occurs.
12. CPU CI/CodeQL remains green and live service evidence is reported separately.
13. Runbook accurately reproduces operation without exposing secrets.
14. Correct one-PR/report-only SELF contract is satisfied; coding never merges.

## Required verification

- fresh listener scan and port selection: VERIFY:
- all-GPU/process snapshot before/start/requests/stop: VERIFY:
- service environment/device UUID proof: VERIFY:
- process/worker count: VERIFY:
- health/readiness transitions: VERIFY:
- real E2E L0/L1/L2/L3 JSON and ZIP as supported: VERIFY:
- identity PNG/YOLO/object invariants on real output: VERIFY:
- concurrency/busy test: VERIFY:
- timeout/cancel/failure/cleanup test: VERIFY:
- repeated-run VRAM/RSS/residue measurements: VERIFY:
- graceful restart and exact rollback: VERIFY:
- log-content safety inspection: VERIFY:
- CPU/static/CI/CodeQL regression: VERIFY:

## Documentation and provenance

Update operator runbook, tested service profile, runtime variables, local endpoint,
model revision/resource limits, health/readiness, auth policy, known unsupported
stages and rollback. Do not document the chosen port as universal; it is a runtime
selection. Do not claim external deployment readiness.

## Security/resource constraints

Authorized mutations are limited to repo-owned/user-level ZAP-IT runtime artifacts,
`/dev/shm/slaif-zap-it`, the verified loopback listener and optional user-owned
systemd unit needed for this service. Do not touch GPU0, other services/processes,
firewall/VPN, system NVIDIA/CUDA, global provider credentials or production data.
Any service secret is stored privately with restrictive permissions and never
printed/committed.

## Deferred human adjudication

- Decision: `NONE`
- Native launcher versus user-systemd, exact local port and measured service limits
  are routine strategic/operator decisions.
- Do not create a critical entry merely because GPU memory is tight; support only
  the measured profile honestly. Use `CRITICAL.md` only if all five conditions are
  genuinely met.

## GitHub publication and report

Create one objective-004 branch/PR. Version code/config templates/runbook, not
secrets or host-specific transient logs. Push all implementation before the final
report-only SELF commit. Report exact listener, GPU isolation, process/worker,
model profile, E2E outputs, concurrency, resource/residue, restart/rollback, CI and
limitations. Coding never merges.