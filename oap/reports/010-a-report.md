# OAP Coding-Agent Report — 010-a

## Work order

- Identifier/order/objective/PR mode: `010-a` / Objective 010 / `CREATE_NEW_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Required branch: `oap/010-a-explicit-model-control-api`
- Required PR: [#66](https://github.com/ulfe-lmi/slaif-zap-it/pull/66)

## Status

PARTIAL

## Executive summary

Implemented and pushed the fixed-model explicit model-control subset. The
service now supports separate operator settings and credentials, standard
repository `index`/`load`/`unload` paths, typed lifecycle states, authoritative
inference pause/drain admission, reusable registry cycles, CUDA cleanup and
cold-memory proof hooks, finite-cardinality metrics, docs, packaging support,
and CPU/fake coverage.

The required real RTX-3090 lifecycle qualification was `NOT RUN`: the active
order assigns physical GPU0, while this execution instruction explicitly
protects physical GPU0. No GPU allocation, CUDA context, live service, listener,
or live model process was started. Therefore this report does not claim real
load/infer/unload memory evidence or full Objective-010 acceptance.

## Authoritative GitHub state

- Base `main`: `5da3851347c2031bea11012fc554140ba7894cc2`
- Starting SHA: `5da3851347c2031bea11012fc554140ba7894cc2`
- Implementation head SHA: `3319939313dee8e3f65cdc7f72058a41d68e5888`
- Report publication commit: SELF
- Branch head verified on `origin`: `3319939313dee8e3f65cdc7f72058a41d68e5888`
- PR state: OPEN, base `main`, head `oap/010-a-explicit-model-control-api`
- New PR: yes, exactly one Objective-010 PR (#66)
- Amended existing PR: no
- Coding merge/auto-merge: NO

## Changes/files

- `src/service/model_control.py`: lifecycle controller, state machine,
  serialized control executor, timeout/cancellation settlement and readiness.
- `src/service/gate.py`, `src/service/app.py`: atomic pause/drain admission and
  fixed `/v2` management routes with sanitized repository errors.
- `src/runtime/live_service.py`: reusable registry load/unload cycles, holder
  release, GC/CUDA cleanup, logical allocated/reserved memory proof and explicit
  startup-mode wiring.
- `src/service/settings.py`, `src/service/metrics.py`, `src/service/errors.py`:
  immutable mode/key/timeouts, finite lifecycle metrics and error taxonomy.
- `tests/test_model_control.py` plus existing suites: CPU/fake settings,
  authentication, schemas, transitions, idempotency, retry, drain, fake memory,
  two cycles, OpenAPI and default-mode coverage.
- `scripts/smoke_model_control.py`, launcher/env/systemd templates, package
  manifest/release verifier, architecture/API/runbook/security/testing/
  datasheet docs.
- Exact unchanged active/order transcript included in the implementation head:
  `oap/active` SHA-256
  `84a40e92f26d70aad62897ef54cb24c7174991849fd0de6867f08dc8589b4252`;
  order SHA-256
  `ea49e4662b7f7685ca0fe16a90d1cd3b57de0e44466500489abf1e810d984b91`.

## Acceptance evidence

1. **One branch/PR from current main:** `PASSED` — PR #66 is the sole open
   Objective-010 PR, based on `5da3851`; implementation head is pushed.
2. **Repository-extension subset and vocabulary:** `PASSED` — `/v2` metadata
   explicitly advertises a management subset; index/load/unload use the fixed
   `zap-it-1` name and docs link the primary Triton Model Repository Extension
   and KServe V2 protocol while disclaiming V2 tensor inference.
3. **Credential and request policy:** `PASSED` — explicit mode requires a
   separate control credential; constant-time bearer comparison, fixed model
   name, bounded JSON bodies, no query parameters, no request-selected device,
   revision, path or lifecycle setting; inference and control credentials are
   independently tested.
4. **CPU lifecycle/concurrency/default compatibility:** `PASSED` — 412 tests
   passed in the final canonical run, including fake repeated load/unload,
   failure retry, idempotency, pause/drain, drain-timeout rollback, readiness
   admission, metrics/OpenAPI, and default `none` behavior. One GPU-marked test
   was honestly skipped because no live GPU phase was authorized.
5. **Real unchanged-PID lifecycle and memory proof:** `NOT RUN` — physical
   GPU0 protection prevented the order-assigned live phase. CPU fake holders
   did prove release to zero in the focused test, but this is not Torch or
   `nvidia-smi` evidence.
6. **No cross-process ownership claim:** `PASSED` — implementation and docs
   make no cross-process handoff or multi-service safety claim.
7. **Static/package/CI verification:** `PASSED` — local checks and all current
   PR CI/CodeQL checks are successful at implementation SHA.
8. **Cleanup/protected resources/transcript:** `PARTIAL` — no live service,
   listener, model process, CUDA allocation or GPU mutation was started and
   prior critical-register bytes were unchanged; live post-cycle cleanup was
   not applicable because the live phase did not run.

The strongest reason not to accept this round is the unproven real unload
boundary: an unload endpoint could otherwise race inference, be reachable by
the inference credential, or report success while PyTorch still owns VRAM. The
implementation addresses those failure modes with separate credentials,
atomic gate pause/drain, one authoritative lifecycle state, repeated fake
memory-proof tests, and no cross-process overclaim. Real assigned-GPU evidence
is still required before acceptance.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 412 passed, 1 honest GPU skip, 77.23% total coverage.
- `.venv/bin/pytest -q tests/test_model_control.py`:
  `PASSED` — focused lifecycle/API tests.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 documents.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  `PASSED` — seven reviewed baseline findings, no additions/removals.
- `bash -n` for every `scripts/*.sh` and `deploy/*.sh`: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- Release artifact secret scan and `verify_release_artifacts.py`: `PASSED`.
- `git diff --check`: `PASSED`.
- Live assigned-GPU qualification and `scripts/smoke_model_control.py` against
  a real service: `NOT RUN` — physical GPU0 protected by execution instruction.

## CI/checks

All checks below completed `SUCCESS` for
`3319939313dee8e3f65cdc7f72058a41d68e5888`:

- `static (format, lint, build)`
- `tests (py3.10)`
- `tests (py3.11)`
- `tests (py3.12)`
- `release (artifact audit)`
- `Analyze (python)` / CodeQL

## GPU/service/resource evidence

- Order-assigned physical target: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`; intended logical mapping would be
  only `cuda:0` after masking. `NOT RUN` — no CUDA environment was launched.
- Physical GPU0: no process, CUDA context, allocation, reset or service
  mutation was initiated by this round. GPU1 and all unrelated workloads were
  not used or modified.
- Host/port/process: no live port was selected or opened; no external Uvicorn
  process was started. CPU tests used in-process fake/TestClient transports.
- GPU/RSS/Torch memory: real evidence `NOT RUN`; fake holder cleanup reached
  zero in the CPU test. No request `/dev/shm` workspace was created by a live
  service in this round.

## Documentation/provenance

The API and architecture docs identify the `/v2` surface as a fixed-model
KServe/Triton Model Repository Extension management subset, not KServe V2 tensor
inference. The runbook documents cold startup, authenticated load, drain/unload,
repeatability and the sanitized operator smoke helper. Model IDs/revisions and
operator paths remain startup policy; weights and request content are not
committed or reported.

## Deferred human adjudication

- Critical register action: NONE
- No `CRITICAL.md` append, edit, closure or human disposition was performed.

## Safety/scope confirmations

- Exactly active round `010-a` was executed; no adjacent order was searched for
  or executed.
- No merge, auto-merge, release, deployment, systemd activation, firewall,
  network, CUDA/driver, global credential, unrelated service or unrelated
  process mutation was performed.
- No bearer credential, raw request/image/YAML, model weight, cache path,
  prompt, answer, customer data or private fixture entered code, logs, PR or
  OAP evidence.
- The local worktree is clean and equals the pushed implementation head before
  report publication.

## Limitations/blockers

The real cold-load-infer-drain-unload-reload-infer-unload sequence, PID/listener
continuity, Torch allocated/reserved `<=64 MiB` proof, 90% loaded-delta release,
real BLIP3 semantic result, and physical GPU cleanup snapshots remain
`NOT RUN` because the user instruction protects the order-assigned GPU0. This
round is therefore a truthful partial implementation/evidence report and makes
no deployment or acceptance claim.

## Factual strategic follow-up

Reconcile the explicit GPU-protection constraint with the order-assigned live
qualification authority before any real GPU phase. Until that is resolved, use
the pushed CPU/API implementation and PR #66 evidence only as partial Objective
010 evidence.
