# OAP Coding-Agent Report — 007-a

## Work order

- Identifier/order/objective/PR mode: `007-a` / Objective 007 / `CREATE_NEW_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Verified base: `main` at `85af4fd562fdf128709a0160bc36884a585e1a5e`
- Branch: `oap/007-a-adaptive-blip3-residency`
- PR: [#62](https://github.com/ulfe-lmi/slaif-zap-it/pull/62), open and unmerged

## Status

BLOCKED

## Executive summary

The bounded adaptive residency implementation, CPU/fake coverage, hostile YAML
policy, local benchmark mode, documentation and operator evidence wiring are
published on the sole Objective-007 PR. The first mandatory live BLIP3-alone
qualification on the freshly verified physical GPU1 was blocked before model
initialization: the pinned remote BLIP3 constructor rejected the existing
`dtype=` keyword. The loader was corrected in the implementation to use the
standard `torch_dtype=` argument, but the order requires live work to stop after
the first disqualifying BLIP3 load result; it was not retried. Therefore this
round does not claim BLIP3 load/inference, sequential service readiness, or the
ten-request goat benchmark.

## Authoritative GitHub state

- Implementation head SHA: `f60db64e4a22b23995e45a5720d69103d9615512`
- Report publication commit: SELF
- Starting SHA: `85af4fd562fdf128709a0160bc36884a585e1a5e`
- New PR: yes; amended existing: no; coding merge: NO
- Required branch and title are exact; PR #62 remains open for `007-b`.

## Changes/files

- Residency/device policy: `src/runtime/strategy.py`, `src/runtime/device.py`,
  `src/runtime/live_service.py`, `src/runtime/__init__.py`.
- Reusable BLIP3 holder, FP16/local-cache loading, request-rule isolation and
  fixed service limits: `modules/verifier/blip3.py`,
  `modules/verifier/__init__.py`, `src/core/engine.py`,
  `src/service/yaml_input.py`.
- Fixed-label residency metrics: `src/service/metrics.py`.
- Local benchmark/smoke behavior: `scripts/smoke_local_goats.py`,
  `scripts/smoke_local_service.py`.
- CPU/fake/API regression coverage: `tests/test_adaptive_residency.py`,
  `tests/test_live_runtime.py`, `tests/test_live_service_units.py`,
  `tests/test_release_candidate.py`, `tests/test_runtime_units.py`,
  `tests/test_service_api.py`.
- Operator/API/runtime documentation and environment example: `docs/API.md`,
  `docs/CONFIG.md`, `docs/GPU-RUNTIME.md`, `docs/OUTPUT-PARITY.md`,
  `docs/RELEASE-GATE-INVENTORY.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md`, `docs/runtime.md`,
  `deploy/service.env.example`.
- Exact orchestration transcript: `oap/active` = `007-a` and
  `oap/orders/007-a-adaptive-blip3-residency.md`.
- No goat fixture bytes, derivatives, model weights, credentials or request
  data were added.

## Acceptance evidence

1. `PASSED` — One implementation commit/branch/PR from the exact verified base;
   the report is this final report-only child and the PR remains open.
2. `PASSED` — CPU tests prove 24575 MiB selects
   `sam2_clip_gpu_blip3_cpu_swap` and 24576 MiB selects
   `sam2_clip_blip3_gpu_resident`; UUID mismatch and occupied-target tests fail
   closed.
3. `PASSED` — CPU/fake tests prove exact swap/restore ordering, no-BLIP and
   all-resident no-transition behavior, reusable-holder reuse, request-rule
   isolation, fixed 32/32 limits and terminal readiness failure on restoration
   failure.
4. `PASSED` — Fake API tests accept nested BLIP3 profiles and preserve hostile
   model/path/device/runtime-control rejection.
5. `PASSED` — Legacy/regression CPU suite remains green and does not download
   models or use CUDA.
6. `PASSED` — Fresh live preflight verified the pinned physical topology and
   one-device logical masking for the attempted gate; GPU0 and unrelated
   listener/process state were not changed.
7. `BLOCKED` — The pinned FP16 BLIP3-only load failed before model
   initialization due to the existing remote-constructor `dtype=` incompatibility.
   No substitute, quantization, CPU inference, offload, disabled BLIP3 or
   weakened memory gate was used.
8. `NOT RUN` — Sequential service readiness, no-BLIP live smoke and ten goat
   requests were not started after criterion 7 blocked.
9. `PASSED` for the attempted phase / `NOT RUN` for repeated qualification —
   GPU1 returned to its observed 6 MiB baseline, no ZAP-IT process or listener
   was started, and `/dev/shm/slaif-zap-it` remained empty. Repeated service
   stability and restoration measurements require a future live qualification.
10. `PARTIAL` — Documentation describes the capacity boundary, host-RAM swap,
    fixed limits, pinned identities, cleanup/failure policy and unqualified
    >=24-GB status; measured startup/latency/memory distributions are absent
    because the ordered live gate blocked first.
11. `PASSED` on implementation head — all required GitHub CPU/static/release/
    CodeQL checks passed. Report-head checks are required again after this SELF
    commit and are recorded below after publication.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 366 passed, 1 intentional GPU skip, 77.63% total coverage.
- `.venv/bin/pytest -q tests/test_adaptive_residency.py tests/test_verifier_blip3.py tests/test_service_api.py tests/test_runtime_units.py tests/test_live_service_units.py`:
  `PASSED` — 115 passed.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n`:
  `PASSED`.
- `git diff --check`: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED` — wheel and sdist
  built; setuptools emitted only existing metadata deprecation warnings.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  `PASSED` — wheel/sdist member and manifest audit.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  `PASSED` — no unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  `PASSED` — exactly the five pre-existing baseline findings.
- `.venv/bin/twine check dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`:
  `PASSED` with declared repo-local Twine 7.0.0.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-gpu/bin/python` with the inline synthetic BLIP3-only holder/load/inference harness:
  `BLOCKED` — one visible logical device and pinned identity were proven, then
  the cached pinned model load raised `TypeError` because the remote constructor
  rejected `dtype=` before initialization. The in-scope code correction is
  `torch_dtype=`; the live gate was not retried.
- Final targeted GPU1 `nvidia-smi`, listener, host-memory and `/dev/shm` checks:
  `PASSED` for cleanup evidence; no service was started.

## CI/checks

Implementation head `f60db64e4a22b23995e45a5720d69103d9615512`:

- `static (format, lint, build)`: `PASSED`.
- `Analyze (python)`: `PASSED`.
- `CodeQL`: `PASSED`.
- `release (artifact audit)`: `PASSED`.
- `tests (py3.10)`: `PASSED`.
- `tests (py3.11)`: `PASSED`.
- `tests (py3.12)`: `PASSED`.

Report-head checks are pending at the moment of report construction and must
be inspected after the SELF commit; no report-head result is claimed here
before that verification.

## GPU/service/resource evidence

- Host: `maelstrom1`.
- Fresh physical GPU1: NVIDIA GeForce RTX 2080 Ti, 11264 MiB, PCI
  `00000000:00:0C.0`, UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`.
- Fresh all-GPU snapshot also showed GPU0 as an NVIDIA GeForce RTX 2080 Ti,
  UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, 11264 MiB, with the
  unrelated existing Python compute PID 66522; GPU0 was not allocated,
  signaled, reset or inspected beyond sanitized evidence.
- Pre-attempt GPU1: 6 MiB used / 10815 MiB free. Post-attempt GPU1: 6 MiB
  used / 10815 MiB free, with no compute-app row.
- Masked BLIP3 probe: one visible CUDA device, logical `cuda:0`, same target
  UUID (Torch omitted the `GPU-` prefix), Torch `2.5.1+cu124`, CUDA runtime
  `12.4`, visible Torch total `10820.9 MiB`.
- BLIP3 peak reserved/95% calculation: `NOT RUN`; the model failed before
  initialization and no allocation was retained. The ordered gate therefore
  remains blocked rather than being inferred from a projection.
- Host `MemAvailable`: about 49.2 GiB before and after. `/dev/shm` had about
  26.2 GiB available after the attempt; `/dev/shm/slaif-zap-it` was empty.
- No ZAP-IT listener or service process was started. The existing LAN listener
  `10.8.132.72:8000` remained unrelated and untouched; no new loopback port was
  selected.
- No request image/config/result was persisted; no goat crop or response body
  was printed or committed.

## Documentation/provenance

The operator strategy, supported profiles, FP16/local-cache-only BLIP3 identity,
per-mask patch behavior, 32/32 service limits, transition failure/restart
behavior, benchmark method, GPU0 protection, loopback binding and pending
007-b qualification status are documented in the changed runtime/API/runbook/
datasheet/parity files. The three approved model IDs/revisions remain in the
existing pinned model specification; weights and caches remain operator assets.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was not modified. Base and working-tree SHA-256 object IDs are
  both `29c2366359fb5a05e151fe546bcf6330477f60ee`; `git diff` is empty.

## Safety/scope confirmations

- Physical GPU0 and PID 66522 were preserved.
- No other GPU workload, Hinton service, LAN listener, firewall, driver, CUDA
  installation, system service, global credential or network setting was
  changed.
- No merge, auto-merge, release, tag, external deployment, gateway mutation or
  second PR was performed.
- The exact active selector is `007-a`; no adjacent order was read or executed.
- The final implementation commit includes the unchanged strategic-authored
  007-a order and the active selector; prior OAP reports and CRITICAL remain
  immutable.

## Limitations/blockers

The ordered live gate is blocked by the pinned remote model constructor keyword
compatibility failure. Because the order requires stopping after the first
disqualifying BLIP3-alone load/inference result, there is no valid 95%-reserved
VRAM measurement, real sequential startup qualification, ten-request benchmark,
or repeated restoration/memory-growth evidence in this round. The code now uses
`torch_dtype=` for the pinned FP16 holder, but that correction is not live proof.

## Factual strategic follow-up

The next qualification must independently re-verify physical GPU1, load the
pinned cached FP16 BLIP3 holder with the corrected constructor path, complete the
ordered 95%-reserved-memory and bounded inference gate, and only then perform
the sequential service/benchmark evidence. The >=24-GB all-resident path remains
unqualified and requires the exclusive >=24-GB qualification reserved for
`007-b` on this same PR.
