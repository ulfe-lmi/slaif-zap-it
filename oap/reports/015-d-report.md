# OAP Coding-Agent Report — 015-d

## Work order

- Identifier/order/objective/PR mode: `015-d` — sanitize the path-bearing TIMM
  startup warning; complete Objective 015 by amending the existing PR.

## Status

COMPLETE

## Executive summary

Installed one narrowly scoped Python warning filter at the live-service
bootstrap boundary. It matches only `FutureWarning`, the stable prefix
`Importing from timm.models.layers is deprecated`, and the exact
`timm.models.layers` warning module. The filter is installed at the start of
`main()`, before service settings, Torch, or resident-model imports.

Added CPU-only warning API tests. They suppress the reviewed TIMM warning while
leaving the same message from another module and an unrelated warning from the
TIMM module observable. The tests use `warnings.catch_warnings`, restore global
warning state, and import neither TIMM nor a model.

After the implementation checks and green implementation-head CI, exactly one
controlled restart of `zap-it-lan.service` was performed. Readiness reached
HTTP 200 after 95 bounded HTTP 503 samples over 190 seconds. The new boot
journal contains no TIMM warning, absolute filename, `FutureWarning`, error,
credential, environment-name, request-content, or path-shaped cache/checkpoint/
repository record. No completion request was accepted and no inference was
run. The private-LAN service remains ready on its existing endpoint.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/71`
- PR state: OPEN, non-draft, MERGEABLE, `mergeStateStatus=CLEAN`.
- Base: `main` at `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Branch: `oap/015-a-request-local-sam2-configuration`.
- Starting report-only head SHA: `3b26d2454a790f501381b7b8d4c289537b18e06a`.
- Implementation head SHA: `792dfad5ef14320fee6fac72bbeb24d1da3478a7`.
- Report publication commit: SELF.
- New PR: no. Amended existing PR: yes. Coding merge/auto-merge: NO.

## Changes/files

Implementation commit `792dfad5ef14320fee6fac72bbeb24d1da3478a7` contains only:

- `src/runtime/live_service.py`: bootstrap warning filter and early `main()`
  installation.
- `tests/test_live_service_units.py`: focused warning-scope/state-restoration
  test.
- `oap/active`: exact active selector `015-d`.
- `oap/orders/015-d-sanitize-path-bearing-startup-warning.md`: exact immutable
  active order transcript.

No SAM2 behavior, request behavior, model identity/residency, authentication,
network, artifact, dependency version, documentation, systemd, shell, or GPU
policy change was made.

## Acceptance evidence

1. **Narrow filter — PASSED.** The filter is `ignore` only for category
   `FutureWarning`, message prefix `Importing from timm.models.layers is
   deprecated`, and module `timm.models.layers`.
2. **Warning scope — PASSED.** The test proves the exact message from another
   module remains visible and an unrelated `FutureWarning` from the TIMM module
   remains visible. No global `FutureWarning` suppression, stderr redirection,
   logger disabling, or journal filtering was added.
3. **Bootstrap placement — PASSED.** `main()` installs the filter before
   `ServiceSettings`, Torch, and the resident loader/import chain. The filter is
   outside core inference and uploaded-config handling.
4. **CPU-only focused coverage — PASSED.** The test emulates Python's warning
   API with a TIMM module and installed filename, without TIMM/model imports or
   GPU access. `warnings.catch_warnings` restores warning state.
5. **No forbidden product changes — PASSED.** The implementation commit's four
   paths are limited to the two runtime/test files and the exact OAP selector/
   order transcript; no dependency or model/config contract changed.
6. **Single live activation — PASSED.** Exactly one
   `systemctl --user restart zap-it-lan.service` was issued. New PID `449821`
   reached ready after 95 HTTP 503 samples in 190 seconds, remained stable for
   a two-second repeat sample, and no corrective restart or inference occurred.
7. **Sanitized new boot — PASSED.** The aggregate journal scan for the boot
   beginning `2026-08-29 01:08:48 CEST` found 112 records and zero TIMM exact
   warnings, TIMM filenames, `FutureWarning`s, tracebacks/severe errors, error
   words, auth material, environment names, request markers, absolute paths,
   cache paths, checkpoint paths, repository paths, or fixed tokenizer records.
8. **Service/auth/counter gate — PASSED.** Authenticated health, readiness and
   metrics returned 200; missing and wrong completion credentials returned 401;
   active inference was `0`, inference duration count/sum were `0/0`, successful
   completions were zero, and residency transition count was zero.

## Verification

- `.venv/bin/pytest -q tests/test_live_service_units.py -k startup_warning_filter`:
  PASSED — 1 passed, 50 deselected.
- `.venv/bin/pytest -q tests/test_live_runtime.py tests/test_live_service_units.py tests/test_sam2_configuration.py tests/test_segmenter_sam2.py`:
  PASSED — 283 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 689 passed, 1 intentional GPU skip; 79.60% total coverage.
- `.venv/bin/ruff format --check .`: PASSED — 144 files formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED — wheel and sdist policy audit.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  PASSED — no unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  PASSED — exact seven reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED — wheel and sdist metadata.
- Systemd/shell file checks: NOT RUN — no systemd or shell files changed. The
  live unit, listener, process, GPU, shared-memory, auth and journal checks were
  run as required above.

## CI/checks

All seven required checks passed on implementation SHA
`792dfad5ef14320fee6fac72bbeb24d1da3478a7`:

- `Analyze (python)`: PASSED / GitHub `pass`.
- `CodeQL`: PASSED / GitHub `pass`.
- `release (artifact audit)`: PASSED / GitHub `pass`.
- `static (format, lint, build)`: PASSED / GitHub `pass`.
- `tests (py3.10)`: PASSED / GitHub `pass`.
- `tests (py3.11)`: PASSED / GitHub `pass`.
- `tests (py3.12)`: PASSED / GitHub `pass`.

## GPU/service/resource evidence

- Assigned physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB. The process-visible device is logical `cuda:0`.
- Before restart: service PID `443516`; after restart: service PID `449821`.
  The final sample showed only PID `449821` on the assigned GPU, with 10107 MiB
  used and 14017 MiB free; the process-reported use was 10084 MiB.
- Final unit: enabled/active/running, `NRestarts=0`; exactly one listener at
  `10.8.132.76:17891`; PID remained `449821` after two seconds.
- `/dev/shm/slaif-zap-it`: mode 0700 and empty of request-workspace entries.
  The operator environment remained mode 0600 with unchanged digest
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- No unassigned GPU, unrelated process/service, port, route, firewall, VPN,
  driver, global environment, credential, request file, image, YAML, or model
  asset was changed or entered into evidence.

## Documentation/provenance

No documentation or dependency changes were needed. The implementation keeps
startup failures and unrelated warnings visible and changes only the formatting
exposure of the already-reviewed TIMM deprecation. The order's CPU/offline
verification and live private-LAN activation were both completed without model
or request-content evidence.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Same branch and PR #71 only; no new PR, merge, release, tag, or auto-merge.
- Exact active order `015-d` was committed unchanged with the implementation.
- Exactly one controlled restart of only `zap-it-lan.service` was performed.
- The service was left enabled, active, ready, and listening on its existing
  private-LAN endpoint.

## Limitations/blockers

None for the ordered scope. No live inference was run, as required by the
order; inference counters therefore remain at zero.

## Factual strategic follow-up

None. PR #71 is ready for strategic review; coding has not merged or selected a
subsequent order.
