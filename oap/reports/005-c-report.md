# OAP Coding-Agent Report — 005-c

## Work order

- Identifier/order/objective/PR mode: `005-c` / Objective 005 / `AMEND_EXISTING_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Required PR: [#49](https://github.com/ulfe-lmi/slaif-zap-it/pull/49)

## Status

COMPLETE

## Executive summary

Fixed the two ordered Objective-005 edge cases without changing the API,
visualization capability, metrics schema, model profile, or prior evidence. L3
now skips hypothetical raw visualization reservation when zero annotated streams
are configured. JSON and ZIP success paths perform their final deadline check
before recording success metrics, while timeout handling records the request and
serialization durations without recording success/object/artifact metrics.

## Authoritative GitHub state

- Base: remote `main` at `22e827eaab15a5eb3299a6b5bfd156eb96c68946`
- Starting PR/report head SHA: `4222e13ce2fade6d668b0c66745cf40be06756f4`
- Implementation head SHA: `47c3a9c6d47d96e29599c73d4f70cd11e0fe4a00`
- PR #49: OPEN, mergeable, clean, base `main`, exact existing title and branch
- Head branch: `oap/005-a-full-output-parity-hardening-and-evidence`
- New PR: no; amended existing PR: yes; coding merge: NO
- Report publication commit: SELF

## Changes/files

Implementation commit `47c3a9c…` contains the exact `005-c` active selector and
order transcript plus:

- `src/service/resources.py`: immediate zero-stream return; configured streams
  retain per-stream and total raw-byte checks.
- `src/service/app.py`: final JSON/ZIP deadline checks precede success metrics;
  serialization duration is recorded once for successful and timed-out
  serialization paths.
- `tests/test_parity_hardening.py`: L3 zero-stream/configured-stream regressions,
  explicit L0-L2 no-preflight coverage, and deterministic JSON/ZIP metric
  exclusivity/recovery checks.
- `oap/active` and
  `oap/orders/005-c-final-deadline-and-zero-stream-corrections.md`: exact
  round transcript and active order.

No dependency, API limit/format, model, geometry, visualization capability,
goats evidence, CLI, security, deployment, or documentation behavior was
changed. No new fixture, model weight, credential, or generated artifact was
added.

## Acceptance evidence

1. **Zero-stream L3 budget:** PASSED — an 8×6 L3 request with no configured
   visualization streams reached the fake engine and returned 200 even when the
   hypothetical RGB stream exceeded the single/total raw cap. The same request
   with one annotated stream returned `413 response_too_large` before the engine
   (`engine.calls == []`).
2. **Configured-stream and lower-level behavior:** PASSED — existing exact
   boundary/two-stream budget tests remain green; parameterized L0, L1 and L2
   requests with visualization configuration continue to bypass the L3 raw
   preflight.
3. **Deadline metrics:** PASSED — JSON and ZIP TestClient cases timed out during
   the operator-only serialization delay with timeout count 1, success and
   completion counts 0, and response/object/artifact success histogram counts 0.
   Serialization duration count was 1. After removing only the test delay, the
   same app returned 200; timeout remained 1 and success/completion/
   response/object/artifact counts each became exactly 1.
4. **Live service:** PASSED — fresh physical-GPU1 synthetic L3 JSON and ZIP smoke
   requests returned 200 with matching eight-object semantics, uint16 identity
   masks, RLE/object fields and deterministic YOLO content. The corrected live
   operator probe (`5 s` request deadline, `6 s` serialization delay) returned
   `504 timeout` with timeout 1, success/completion/response/object/artifact
   counts 0 and serialization-duration count 1. A clean normal restart returned
   synthetic L3 JSON 200 with success count 1. The prior nonredistributable goats
   run was not rerun or exposed; its accepted `005-b` evidence remains
   inherited and untouched.
5. **Remote topology:** PASSED — PR #49 remains the sole Objective-005 PR; the
   implementation is pushed at the literal implementation SHA, and all prior
   report/order artifacts remain immutable.

## Verification

- `pytest -q tests/test_parity_hardening.py`: BLOCKED — `pytest` is not on the
  shell PATH; no test executed by that command.
- `.venv/bin/pytest -q tests/test_parity_hardening.py`: PASSED — focused
  Objective-005 parity/resource/metrics tests passed before the final full-suite
  run; the final full suite includes the completed parameterized cases.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — `343 passed, 1 skipped`, `76.69%` total coverage; the skip is the
  explicit opt-in physical-GPU1 pytest marker.
- `.venv/bin/ruff format --check .`: PASSED — 122 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src tests scripts`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python -m build --wheel`: PASSED — `zap_it-0.1.0-py3-none-any.whl`.
- Wheel import probe for `src.service.app` and `src.service.resources`: PASSED.
- `git diff --check`: PASSED.
- Changed-diff secret scan: PASSED — no private-key, token, bearer-key or
  password assignment pattern.
- Changed implementation large/derived-artifact scan: PASSED — no changed file
  exceeded 5 MiB and no fixture/output path or derivative was added.
- Fresh GPU1 synthetic smoke and corrected serialization-timeout/recovery probe:
  PASSED — sanitized statuses, counters and response facts recorded above.

## CI/checks

At implementation SHA `47c3a9c6d47d96e29599c73d4f70cd11e0fe4a00`, all six required
GitHub checks were SUCCESS:

| Check | State |
|---|---|
| `static (format, lint, build)` | SUCCESS |
| `tests (py3.10)` | SUCCESS |
| `tests (py3.11)` | SUCCESS |
| `tests (py3.12)` | SUCCESS |
| `Analyze (python)` | SUCCESS |
| `CodeQL` | SUCCESS |

## GPU/service/resource evidence

- Fresh live verification recorded physical GPU1 as NVIDIA GeForce RTX 2080 Ti,
  PCI `00000000:00:0C.0`, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, 11,264 MiB. Service launch used
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1`; application
  device was logical `cuda:0`.
- Physical GPU0 remained the separate RTX 2080 Ti at PCI `00000000:00:08.0`,
  UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`. Protected PID 66522 remained
  its compute process at 2,492 MiB in the before/after snapshots; no ZAP-IT
  process was allocated on GPU0.
- GPU1 was 6 MiB before live launch and 6 MiB in the final stopped snapshot;
  no ZAP-IT compute process remained. Current-round peak GPU1 allocation was
  not separately sampled; accepted Objective-005-b peak evidence remains
  unchanged.
- Live endpoint was loopback `127.0.0.1:17891`, freshly free before launch and
  free after final stop. Final service process scan was empty and
  `/dev/shm/slaif-zap-it` had no request/runtime residue.

## Documentation/provenance

The existing API, runbook and service-datasheet statements already describe
per-configured-stream reservation and L0-L2 no-render behavior; no documentation
change was needed for this narrow correction. No model identity, license,
runtime pin, fixture, prompt, response, credential or private input entered the
report.

## Deferred human adjudication

- Critical register action: NONE
- The finalized order explicitly resolves `NONE`; no critical-register append,
  edit, close or disposition was performed.

## Safety/scope confirmations

- No merge or auto-merge was performed; PR #49 was amended only.
- No API contract, metrics schema, model, dependency, geometry, panoptic,
  BLIP3, CLI, persistence, LAN exposure, firewall/VPN, systemd, CUDA/driver or
  unrelated service/process change was made.
- Physical GPU0 and PID 66522 were not touched.
- No goats asset was rerun or exposed, and no derived academic fixture entered
  the repository, package, report or generated evidence.
- No request image, YAML, raw response, credential or model weight entered OAP
  evidence.

## Limitations/blockers

The one missing PATH-level `pytest` executable was resolved by the repository
`.venv` and did not block canonical verification. The inherited goats run
remains bounded nonredistributable local academic evidence, not an accuracy
benchmark, SLA, soak test, production-readiness claim, or rights clearance.

## Factual strategic follow-up

Strategic review/acceptance and any merge decision remain outside coding scope.
Objective 006 remains responsible for packaging, integration, rights and release
gates. Coding does not select another order.

Implementation head SHA: 47c3a9c6d47d96e29599c73d4f70cd11e0fe4a00
Report publication commit: SELF
