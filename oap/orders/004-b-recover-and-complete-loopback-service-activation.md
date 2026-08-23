# OAP Work Order — 004-b — Recover and complete loopback service activation

Objective `004-b`, continuation of numeric Objective 004. Recover the preserved
post-collision implementation candidate under exclusive single-writer ownership,
audit every retained byte against the merged Objective-003 base and the immutable
`004-a` order/report, complete the physical-GPU1 loopback service activation,
publish the one still-missing Objective-004 PR, and provide fresh complete CPU,
CI and live GPU/service evidence. Do not replay `004-a`, rewrite its report, or
discard recoverable work.

## Authoritative recovery and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Numeric objective / round: `004 / 004-b`.
- Mode: `AMEND_EXISTING_OBJECTIVE_BRANCH_AND_CREATE_MISSING_SINGLE_PR`.
- Remote `main`: `1a4272d60c52cc045f57f2842652485efdb7a55c`, the
  accepted Objective-003 squash merge from PR #47; post-merge CI and CodeQL are
  green.
- Existing branch: `oap/004-a-loopback-service-activation`, local and remote at
  `336374e293968d8a0d86dc92b25d53305c95d795`.
- Commit `336374e...` is an immutable report-only `004-a` BLOCKED incident
  report whose parent is exactly remote main. It contains no implementation and
  opened no PR. Preserve it byte-for-byte.
- Existing Objective-004 PR: NONE, verified through GitHub immediately before
  publication. Therefore this `b` round stays on the existing branch and creates
  the one and only numeric-Objective-004 PR. This is not a second PR and does not
  authorize a new branch.
- Required PR title: `Objective 004: loopback service activation on physical
  GPU1 with recovered live E2E evidence`.
- Required new report: `oap/reports/004-b-report.md`; the existing
  `oap/reports/004-a-report.md` is immutable and must not be edited, replaced,
  renamed or deleted.

## Preserved local work and provenance law

The branch worktree intentionally contains a restored, unstaged Objective-004
candidate. It is not accepted implementation merely because it exists.

Tracked changes currently comprise `.gitignore`, `README.md`,
`modules/segmenter/sam2.py`, `requirements-gpu-cu124.lock`,
`src/runtime/strategy.py`, `src/service/app.py`, `src/service/envelope.py`, and
the strategic `oap/active` transition. Untracked candidate assets comprise:

```text
deploy/service.env.example
deploy/zap-it-local.service
docs/RUNBOOK.md
oap/orders/004-a-loopback-service-activation.md
scripts/serve_local.py
scripts/serve_local.sh
scripts/serve_local_stop.sh
scripts/smoke_local_service.py
src/runtime/live_service.py
tests/test_live_runtime.py
tests/test_live_service_units.py
```

Strategic quarantine at
`/synology/homes/janezp/opencode-supervision/slaif-zap-it/quarantine/004-a-split-brain/`
preserves the collision snapshot and additional candidate edits, including CLIP
resident-label and package-export changes that are not currently applied. Treat
the quarantine as read-only recovery evidence. Reuse only code whose behavior,
authorship relevance and tests are independently understood; do not blindly
apply the full patch or replace current files wholesale.

The previous Codex session recorded a provisional 296-pass/1-GPU-skip CPU run
and a live one-process readiness demonstration, but it crashed during the full
smoke. Those observations guide recovery only. They are stale/mixed-provenance
evidence and count as NOT RUN for this round until freshly reproduced.

Before any edit or live launch:

1. Reconcile the current branch, remote branch, absent PR, both immutable orders,
   the `004-a` report and every dirty/untracked path.
2. Confirm no other coding process is writing this checkout.
3. Review the complete current diff and compare relevant archived/build copies
   when necessary to identify partial or overwritten seams.
4. Preserve all recoverable content until a reviewed replacement is committed;
   never reset, clean, delete, or overwrite it for convenience.
5. Record material provenance choices and discarded candidate behavior in the
   `004-b` report without exposing private session content.

## Fresh host state at strategic publication

- Physical GPU0: RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI
  `00000000:00:08.0`, 11264 MiB total, 2161 MiB used by the unrelated protected
  compute PID 66522. Never allocate, stop, reset or otherwise touch it.
- Physical GPU1 target: RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI
  `00000000:00:0C.0`, 11264 MiB total, 6 MiB used / 10815 MiB free, no compute
  process.
- Driver 580.178.04; repo-owned `.venv-gpu` remains present with Torch
  2.5.1+cu124, the Objective-003 pinned model stack, FastAPI 0.141.1,
  python-multipart 0.0.32 and Uvicorn 0.52.4.
- Pinned SAM2, CLIP and BLIP3 model caches remain present at the exact
  Objective-003 revisions. Do not re-download or commit weights. BLIP3 remains
  rejected before load under the measured strategy.
- `/dev/shm` is a 27-GiB tmpfs; `/dev/shm/slaif-zap-it` is mode 0700 and empty.
- No ZAP-IT service or user systemd unit is active. Ports 17891 and 23654 are
  free by live `ss`. Re-verify immediately before every activation; a prior
  observation is not a reservation.

## Binding runtime decisions

- Launch environment:

  ```text
  CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=1
  SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
  SLAIF_ZAP_IT_HOST=127.0.0.1
  SLAIF_ZAP_IT_PORT=<freshly verified unused port>
  SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it
  ```

- Only logical `cuda:0` may exist inside the service. Strict startup fails
  closed on UUID/count/device mismatch; no CPU or GPU0 fallback.
- One Uvicorn process, one worker, one resident registry and one active
  inference. Queue depth remains zero with deterministic HTTP 503
  `service_busy` and `Retry-After`.
- Resident strategy remains `sam2_clip_resident_blip3_rejected`. Start the
  measured SAM2+CLIP registry only; request YAML cannot change models,
  revisions, cache, device, strategy, network or service settings.
- Primary operator mechanism remains the repo-owned launcher. The user-systemd
  unit is an optional shipped-uninstalled template; do not install or enable it
  during this round.
- Prefer port 17891 after fresh verification, then 23654, then a verified-unused
  20000–40000 fallback. Bind IPv4 loopback only.
- Final round state is STOPPED: no listener, no ZAP-IT process, GPU1 near its
  initial idle baseline and no per-request/runtime residue under the shared-
  memory root. Nothing is left running unattended.

## Required implementation and audit scope

1. Audit and complete operator-only configuration for bind/port, UUID/device,
   model/cache/revisions, resource strategy, limits, authentication, temp root,
   queue/busy semantics and safe logging. Secrets remain outside Git.
2. Audit the live engine/registry adapter. Reuse resident SAM2 and CLIP weights
   while keeping request configuration, prompt embeddings, masks, results and
   artifacts isolated. Resolve any missing CLIP label-refresh/export wiring only
   after examining the archived candidate and proving it with focused tests.
3. Complete fail-closed device/startup/readiness composition before unsafe model
   use. Normalize the masked PyTorch UUID correctly without trusting unmasked
   `nvidia-smi` output as the visible-process view.
4. Complete launcher start/stop/status/log/restart behavior, executable modes,
   stale-PID protection, PID command-line ownership checks, verified port
   selection, detached lifetime, restrictive runtime permissions and cleanup.
   Stop only the checkout-owned PID; never use broad kill patterns.
5. Bind exclusively to `127.0.0.1`; reject wildcard/LAN/IPv6-wildcard operator
   configuration and prove the actual listener with `ss`.
6. Preserve Objective-002 transport/security semantics while wiring the real
   engine: hostile multipart/YAML/image bounds, optional bearer auth, stable
   sanitized errors, response limits, deadline and concurrency gate.
7. Ensure timeout or client cancellation cannot release the single-inference
   gate while a synchronous CUDA call still runs. Return/document honest bounded
   semantics and leave the registry usable with no request residue.
8. Keep request bytes, decoded arrays, results and artifacts in RAM. Any required
   path compatibility uses unique mode-0700 directories/mode-0600 files below
   the configured RAM-backed root with unconditional cleanup.
9. Complete the smoke/evidence harness without customer/private inputs. It must
   validate real L0–L3 JSON and ZIP invariants, unsupported BLIP3 behavior,
   bounded overlapping requests, failures, deadlines/cancellation, response-size
   rejection, repeated requests, restart and final rollback.
10. Complete the runbook and deployment templates so they reproduce the tested
    mechanism and clearly state loopback-only/non-production limitations,
    current supported profile, live port selection, GPU checks, logs, cleanup,
    rollback and optional-uninstalled systemd status.
11. Preserve the legacy CLI and Objective-000–003 behavior. No unrelated
    refactor or scientific threshold/default/model change.

## Non-goals and protected boundaries

- no replay/rewrite of activated `004-a` artifacts;
- no new branch, duplicate PR or numeric objective;
- no LAN/public exposure, TLS, reverse proxy, gateway, firewall or VPN change;
- no physical GPU0 use or unrelated process/service mutation;
- no multi-worker, multi-GPU, background job queue or persistent request data;
- no BLIP3 enablement, scientific-model substitution or request-triggered
  download/remote code;
- no installed/enabled systemd unit, Docker/Compose or release packaging;
- no production/customer data, credentials, private raw bodies or weights;
- no merge by coding.

## Required fresh verification

### Recovery/provenance

- exact starting branch/head/upstream/status and absent-PR proof;
- inventory of every initial dirty/untracked path and disposition in the final
  implementation;
- review of the archived-only CLIP/export candidates and explicit keep/reject
  rationale;
- proof that only one coding writer operated after this activation.

### CPU/static/package

- focused runtime/launcher/service/smoke tests;
- complete canonical CPU suite with exact pass/fail/skip counts and coverage;
- Ruff format and lint;
- shell syntax for every shipped shell asset;
- wheel build and package/import checks;
- secret/raw-content/large-artifact and diff-scope inspection.

CPU tests and CI must not load CUDA or download models. A GPU-marker skip is
reported as SKIPPED, never counted as a pass.

### Live GPU1/service

Capture sanitized before/start/loading/ready/request/failure/stop/restart/final
evidence:

- all-GPU UUID/PCI/name/VRAM and compute-process snapshots proving GPU0 remains
  byte-stable and every ZAP-IT allocation belongs only to the one GPU1 PID;
- fresh selected-port proof, one IPv4 loopback listener, one PID/process/worker,
  and no wildcard listener;
- `/healthz` process health distinct from `/readyz` registry/device/shm
  readiness, including a genuine observed not-ready/loading or injected
  fail-closed transition rather than fabricated timing;
- in-process visible device count/logical index/name/UUID proof;
- real L0/L1/L2/L3 JSON plus ZIP for the supported resident service profile,
  including normalized five-field YOLO, uint16 original-dimension identity PNG,
  bijective object IDs, produced metadata, bounded full artifacts and runtime
  provenance;
- stable pre-inference `unsupported_profile` rejection for BLIP3 configuration;
- overlapping real requests proving at most one inference and deterministic
  second-request 503 with no rejected-request model allocation;
- invalid input, safe injected inference failure, deadline/timeout, cancellation
  and response-too-large behavior with registry recovery and zero request
  residue;
- bounded repeated requests with latency, response bytes, process RSS, GPU1
  allocated/reserved/end memory and state-isolation observations;
- graceful stop, port/GPU cleanup, restart to readiness, one post-restart E2E,
  then final stop and rollback proof;
- log-content inspection excluding raw image/YAML, prompts/answers, filenames,
  headers/keys, model-cache/host paths and stack traces.

If any required live layer cannot be completed safely, report it truthfully and
do not claim Objective 004 complete.

### GitHub/OAP publication

- Commit all valid implementation, both immutable order transcripts and current
  `oap/active` before the report.
- Push the existing branch and create exactly one PR with the required title;
  never create a replacement branch.
- Require every expected CI and CodeQL check on the current implementation and
  final report head to be present and SUCCESS; pending/missing/skipped/failing is
  not green.
- Capture the literal implementation SHA after all non-report work is remote.
- Create exactly one new immutable `oap/reports/004-b-report.md` containing the
  literal SHA and `Report publication commit: SELF`.
- Final report commit changes only that report path; its first parent equals the
  implementation SHA. Push and verify remote PR head, parent, changed path and
  exact report bytes before signaling response `OK`.
- Do not modify or republish `oap/reports/004-a-report.md`.

## Acceptance criteria

1. Recovered work is audited rather than discarded or blindly trusted; all
   committed behavior is attributable, reviewed and freshly tested.
2. Exactly one Objective-004 PR exists on the existing branch, based on the
   accepted Objective-003 main SHA.
3. One service process/worker runs only on freshly verified
   `127.0.0.1:<port>` during tests and sees only physical GPU1 as logical
   `cuda:0` with the exact pinned UUID.
4. Physical GPU0 remains unchanged and no unrelated process/service is touched.
5. Health/readiness, resident profile, one-inference/busy behavior and
   fail-closed startup are observed honestly.
6. Real inference satisfies every supported L0–L3 JSON/ZIP invariant; BLIP3 is
   rejected before load.
7. Success, failure, timeout, cancellation, busy rejection, repeated requests,
   restart and final stop leave no request/runtime residue or corrupted state.
8. Resource measurements show bounded behavior for the tested window without
   production-soak claims.
9. Launcher, optional unit template and runbook are secure, reproducible and
   accurately bounded to loopback local research operation.
10. Full CPU/static/package verification and all expected GitHub CI/CodeQL
    checks are green on the immutable report head.
11. Final host state is stopped/clean with both candidate listener state and all
    GPU processes explicitly reported.
12. `004-a` report/order immutability, one-PR law, report-only SELF parent and
    coding-never-merges law all hold.

## Deferred human adjudication

- Decision: `NONE`

Recovery provenance, launcher mechanics, local loopback port, supported
resident profile and bounded service limits are reversible engineering decisions
already resolved by architecture and measured evidence. No CRITICAL threshold is
met. LAN/public deployment, commercial/release licensing and production/customer
use remain outside this objective and behind later explicit gates.

## Coding response

Coding sends the exact FIFO `OK` only after the remote implementation, sole PR,
fresh evidence, immutable `004-b` SELF report and final stopped host state are
all verified. A truthful BLOCKED/FAILED/PARTIAL report also signals, but no status
word is placed on the FIFO. Coding never merges.
