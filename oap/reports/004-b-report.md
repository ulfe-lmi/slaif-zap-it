# OAP Coding-Agent Report — 004-b

## Work order

- Identifier: `004-b`; numeric objective `004`; mode
  `AMEND_EXISTING_OBJECTIVE_BRANCH_AND_CREATE_MISSING_SINGLE_PR`.
- Order: `oap/orders/004-b-recover-and-complete-loopback-service-activation.md`.
- Existing branch retained: `oap/004-a-loopback-service-activation`.
- Starting branch/head: `oap/004-a-loopback-service-activation` at
  `336374e293968d8a0d86dc92b25d53305c95d795`, tracking the same remote head.
- Verified base: remote `main` at
  `1a4272d60c52cc045f57f2842652485efdb7a55c`.

## Status

COMPLETE

## Executive summary

Recovered and audited the preserved post-collision candidate, completed the
single-process loopback activation on freshly verified physical GPU1, and
published the one Objective-004 PR. The implementation now provides strict
operator configuration, fail-closed device/readiness checks, resident SAM2+CLIP
runtime wiring, secure launcher lifecycle, real JSON/ZIP L0–L3 smoke coverage,
bounded busy/deadline/cancellation/failure behavior, runbook/deployment
templates, and final stopped-state cleanup.

Two candidate defects were found and repaired during fresh evidence collection:
the smoke YAML interpreted `resize: 512` as a multiplicative 65,536× upscale,
and larger-area identity rasterization could hide a fully occluded object ID.
The smoke fixture now uses native scale, and the service projection reserves a
deterministic source pixel when necessary while retaining the complete source
masks. A read-only log audit also found and eliminated a torchvision warning
that exposed a host package path by making decoded request arrays writable.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- PR: [#48](https://github.com/ulfe-lmi/slaif-zap-it/pull/48), OPEN, non-draft.
- Required title: `Objective 004: loopback service activation on physical GPU1
  with recovered live E2E evidence`.
- Base SHA: `1a4272d60c52cc045f57f2842652485efdb7a55c`.
- Starting SHA: `336374e293968d8a0d86dc92b25d53305c95d795`.
- Implementation head SHA: `fea4319c9bfd4845c347a89ac46ef48fcd43fa97`.
- Report publication commit: SELF
- New PR: YES — the sole numeric-Objective-004 PR; amended existing: NO.
- Coding merge: NO.
- Remote implementation branch was verified at the implementation SHA before
  report creation. `oap/reports/004-a-report.md` was not edited, replaced,
  renamed, deleted, or included in the implementation commit.

## Changes/files

- Operator runtime and lifecycle: `src/runtime/live_service.py`,
  `src/runtime/strategy.py`, `src/runtime/__init__.py`,
  `scripts/serve_local.py`, `scripts/serve_local.sh`,
  `scripts/serve_local_stop.sh`, and the optional uninstalled
  `deploy/zap-it-local.service`.
- Configuration and operator documentation: `deploy/service.env.example`,
  `docs/RUNBOOK.md`, `README.md`, `.gitignore`, and the GPU lock additions for
  FastAPI, multipart parsing, and Uvicorn.
- Resident engine integration: pinned local-only SAM2/CLIP loading,
  resident CLIP label refresh, strict request/runtime separation, timeout gate
  draining, runtime provenance, and writable request-local image buffers in
  `modules/classifier/clip.py`, `modules/segmenter/sam2.py`,
  `src/service/app.py`, `src/service/envelope.py`, and
  `src/service/image_input.py`.
- Identity/service contract: service-level bijective uint16 identity IDs with
  deterministic occlusion reservation, documented in `docs/API.md` and
  `docs/CORE.md`, with focused renderer and decoder regressions.
- Live evidence tooling and tests: `scripts/smoke_local_service.py`,
  `tests/test_live_runtime.py`, `tests/test_live_service_units.py`,
  `tests/test_core_renderers.py`, and `tests/test_service_units.py`.
- Governance transcript: the exact immutable `004-a` and `004-b` order files
  and the current `oap/active` selector were committed unchanged as required.

## Acceptance evidence

1. Recovery audit: PASSED. The complete initial dirty/untracked inventory was
   reconciled before edits. The candidate and quarantine were treated as mixed
   provenance; no wholesale quarantine patch was applied.
2. One PR/base: PASSED. PR #48 is the only Objective-004 PR, on the retained
   branch and based on accepted Objective-003 `main`.
3. GPU1 service: PASSED. The final live process was one Python/Uvicorn process,
   one worker, PID 261868, on `127.0.0.1:17891`; the process reported one
   visible device as logical `cuda:0` with the pinned GPU1 UUID.
4. GPU0 protection: PASSED. GPU0 stayed at 2161 MiB with protected compute PID
   66522 before, during, and after activation. No process on GPU0 was touched.
5. Health/readiness: PASSED. Fresh starts returned `/healthz` 200 while the
   registry was loading and `/readyz` 503 `not_ready`, followed by `/readyz`
   200 with the resident strategy after loading. A wrong-UUID launch exited
   before model load with no listener and GPU1 at its 6 MiB baseline.
6. Supported E2E: PASSED. The final current-head smoke run passed 10 cases:
   L0–L3 JSON and ZIP, BLIP3 pre-load rejection, and three repeat calls. Real
   responses had normalized five-field YOLO, 8 lines, original 128×128
   uint16 identity masks, bijective IDs 1..8, 8 L2 objects, produced SAM2 and
   CLIP fields, and bounded L3 stage/timing/provenance metadata.
7. Failure and cleanup: PASSED. Invalid image/YAML, injected inference
   failure, deadline timeout, client cancellation, response-size rejection,
   busy rejection, and repeated requests were exercised. Final stop removed
   the service listener and runtime PID/log files; the shared-memory root was
   mode 0700 and empty.
8. Bounded resources: PASSED for the tested window. Final repeated calls had
   one distinct YOLO output and 433.7–437.1 ms latency; the live PID RSS was
   approximately 1,984,604 KiB and GPU1 was approximately 5,740 MiB used while
   resident. After stop GPU1 returned to 6 MiB. No soak or leak-proof claim is
   made.
9. Operator mechanism: PASSED. Start/stop/status/log/restart, stale-PID
   ownership checks, verified port selection, mode-0700/0600 runtime files,
   loopback-only binding, rollback, and optional-uninstalled systemd status are
   documented and tested.
10. CPU/static/CI: PASSED. The complete CPU suite, coverage gate, Ruff,
    shell syntax, compile, wheel build, and all required GitHub checks passed.
11. Final host state: PASSED. No ZAP-IT listener or process remained; GPU1 was
    at baseline and GPU0 remained protected; no request residue remained under
    `/dev/shm/slaif-zap-it`.
12. Governance: PASSED. The 004-a report/order remained immutable, one PR was
    created on the existing numeric-objective branch, implementation preceded
    the report, and coding did not merge.

## Verification

- `git fetch origin --prune`: PASSED — remote reconciliation completed before
  mutation.
- `git status --short --branch`, `git branch -vv`, remote SHA inspection, and
  `gh pr list --repo ulfe-lmi/slaif-zap-it --state all --search
  'head:oap/004-a-loopback-service-activation'`: PASSED — starting head and
  upstream matched `336374e...`; no PR existed before publication.
- Mandatory contract/order reads and immutable 004-a order/report inspection:
  PASSED — all required files were read completely; `CRITICAL.md` was not
  required by the order and was not read or modified.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED —
  `300 passed, 1 skipped`, coverage `75.21%`; the single skip was the explicit
  opt-in GPU integration marker. Two known deprecation warnings were emitted.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python -m compileall -q src scripts/serve_local.py
  scripts/smoke_local_service.py`: PASSED.
- `git diff --check` and staged diff check: PASSED.
- `.venv/bin/python -m build --wheel`: PASSED — wheel built; only upstream setuptools
  license-metadata deprecation warnings were emitted.
- `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free
  --format=csv,noheader` and the compute-process query: PASSED — GPU0/GPU1
  identity and ownership evidence captured before, during, and after live use.
- `ss -H -ltn`: PASSED — the ZAP-IT listener was only IPv4 loopback
  `127.0.0.1:17891`; no wildcard or LAN ZAP-IT listener existed.
- `scripts/serve_local.sh start`, repeated `curl /healthz` and `curl
  /readyz`, and `scripts/serve_local.sh stop`: PASSED — loading, ready,
  restart, graceful stop, and final cleanup were observed.
- `timeout 900 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels
  0 1 2 3 --formats json zip --repeat 3`: PASSED — 10 current-head cases.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels 0
  --formats json --skip-blip3-check --busy --invalid`: FAILED only for the
  no-delay busy overlap; the invalid-image and invalid-YAML cases passed. The
  required busy test was then rerun with the operator-only delay hook:
  PASSED — follower 503 `service_busy`, `Retry-After: 5`, one active
  inference.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels
  --formats json --skip-blip3-check --failure`: PASSED — expected 500
  `inference_failure` under the failure-injection environment.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels
  --formats json --skip-blip3-check --timeout`: PASSED — expected 504
  `timeout`; the underlying synchronous call drained before gate release.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels
  --formats json --skip-blip3-check --cancel`: PASSED under delay-only
  injection — client socket closed and the recovery request returned 200.
- `timeout 300 python scripts/smoke_local_service.py --port 17891 --levels
  --formats json --skip-blip3-check --response-too-large`: PASSED — expected
  413 `response_too_large` under the operator cap.
- Wrong expected UUID launch through `scripts/serve_local.sh start`: PASSED as
  a negative test — exit code 1, sanitized device-guard failure, no model
  listener, no GPU1 allocation.
- Final log-content scan for raw labels, filenames, auth headers/keys, host
  paths, cache paths, traceback text, and request data: PASSED — no matches in
  the final service log.
- Final `find /dev/shm/slaif-zap-it -mindepth 1 -maxdepth 3`: PASSED — no
  request/runtime children after stop.

## CI/checks

All required checks were present and SUCCESS on implementation head
`fea4319c9bfd4845c347a89ac46ef48fcd43fa97` before report creation:

- `static (format, lint, build)`: SUCCESS.
- `tests (py3.10)`: SUCCESS.
- `tests (py3.11)`: SUCCESS.
- `tests (py3.12)`: SUCCESS.
- `Analyze (python)` / CodeQL: SUCCESS.

No GPU runner was required or claimed by CI. The final report commit is
documentation-only; its same required checks were verified after publication
before the FIFO signal, with no implementation mutation or follow-up commit.

## GPU/service/resource evidence

- Physical GPU0: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`, 11264
  MiB; protected PID 66522 used approximately 2161 MiB throughout.
- Physical GPU1: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`, 11264
  MiB; idle baseline 6 MiB, approximately 5740 MiB used by the single ZAP-IT
  PID while resident, and 6 MiB after stop.
- In-process runtime provenance reported physical index 1, logical `cuda:0`,
  visible count 1, the pinned UUID, and the resident strategy
  `sam2_clip_resident_blip3_rejected` with supported profiles `sam2`, `clip`,
  and `sam2_clip`. BLIP3 was rejected before load.
- Service binding: `127.0.0.1:17891`, freshly verified unused before starts;
  final listener scan showed no ZAP-IT listener. One Uvicorn worker and one
  active inference slot were used.
- Runtime files were mode 0700 for the private runtime directory and 0600 for
  PID/log files. Request content stayed in memory; no request workspace was
  created. The final shared-memory root remained mode 0700 and empty.
- Model identity remained pinned to the Objective-003-approved SAM2 and CLIP
  revisions, loaded local-only. No weights, credentials, customer data, or
  persistent request artifacts entered the repository.
- Final logs contained only sanitized process/device/status/timing/access
  facts. The optional user-systemd unit was shipped uninstalled and was not
  enabled.

## Documentation/provenance

The runbook, README navigation, service environment template, optional unit,
identity overlap policy, supported resident profile, resource limits, cleanup,
rollback, and loopback/non-production limitations were updated in the same
implementation commit.

The initial worktree contained the recovered tracked candidate changes in
`.gitignore`, `README.md`, `modules/classifier/clip.py`,
`modules/segmenter/sam2.py`, `oap/active`, `requirements-gpu-cu124.lock`,
`src/runtime/__init__.py`, `src/runtime/strategy.py`, `src/service/app.py`, and
`src/service/envelope.py`; untracked candidate/transcript paths were
`deploy/service.env.example`, `deploy/zap-it-local.service`, `docs/RUNBOOK.md`,
both `oap/orders/004-*.md` files, the four launcher/smoke scripts,
`src/runtime/live_service.py`, and the two live test files. Every listed path
was audited and either committed in the implementation or deliberately
excluded as generated/ignored data. No unrelated checkout change was reset or
cleaned.

The strategic quarantine was read-only recovery evidence. The archived CLIP
resident-label/local-cache behavior was retained only after focused unit tests
and real label-bearing requests passed. The quarantine-only service package
docstring/export candidate was rejected because it added no required activation
behavior and the existing public exports already passed the CPU contract. No
private collision-session material, model cache, or raw request was copied.

## Deferred human adjudication

- Critical register action: NONE
- The finalized order explicitly selected `NONE`; no critical register bytes
  were read, appended, edited, or otherwise mutated.

## Safety/scope confirmations

- No merge, auto-merge, close, release, LAN/public exposure, TLS, reverse proxy,
  firewall, VPN, Docker, installed systemd unit, or unrelated service change.
- No GPU0 allocation, process stop, reset, reconfiguration, or memory theft.
- No request-selected device/model/revision/cache/download/remote-code/path or
  service-setting capability.
- No broad kill pattern was used; stop signaled only the checkout-owned PID
  after command-line ownership validation.
- No credentials, raw inputs, prompts/answers, customer data, model weights,
  or persistent request artifacts entered GitHub/OAP evidence.

## Limitations/blockers

- This remains a loopback-only, operator-controlled, non-production service.
- The supported resident profile is SAM2+CLIP; BLIP3 remains rejected before
  load under the measured GPU1 strategy.
- One process, one worker, one active inference, and queue depth zero are the
  tested bounded profile. Client cancellation cannot interrupt a synchronous
  CUDA call; the call is drained before the gate is released.
- Resource and stability evidence is bounded to the redistributable 128×128
  fixture and the measured test window. It is not a production soak, SLA, or
  leak-proof claim.

## Factual strategic follow-up

Strategic review/acceptance of PR #48 is the next governance action. Coding has
not merged or selected a subsequent order. The final host is stopped and ready
for strategic reconciliation.
