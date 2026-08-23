# OAP Coding-Agent Report — 005-b

## Work order

- Identifier/order/objective/PR mode: `005-b` / Objective 005 / `AMEND_EXISTING_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Branch: `oap/005-a-full-output-parity-hardening-and-evidence`
- Required PR: [#49](https://github.com/ulfe-lmi/slaif-zap-it/pull/49)

## Status

COMPLETE

## Executive summary

Objective 005-b closes the ordered serialization, RLE, visualization-allocation,
live-auth/resource-recovery and local academic E2E gaps on the existing
Objective-005 PR. RLE now uses fixed-size NumPy transition chunks without a
second full-size flatten, and one absolute monotonic request deadline covers
inference through RLE, artifact preparation, base64 JSON expansion and ZIP
assembly. L3 annotated streams are raw-memory preflighted before engine
execution and their reservation is deducted from the debug sink budget.

The two local academic goat fixtures were exercised only as in-memory central
50% crops using a sanitized in-memory API-safe config. They are
**NONREDISTRIBUTABLE — local academic E2E only; excluded from packages/release
fixtures**. No source bytes, crops, raw config, prompts, labels or responses
were committed or placed in evidence.

## Authoritative GitHub state

- Base: remote `main` at `22e827eaab15a5eb3299a6b5bfd156eb96c68946`
- Starting checkout/PR head SHA: `17f67bd6eedc527c239e93326bc9afc4b1d43daa`
- Implementation head SHA: `c4f786452ceabd1f5028efcadb178e608e684db5`
- PR #49: OPEN, existing exact title/base `main`, head at implementation SHA
- New PR: no; amended existing PR: yes; coding merge: NO
- Report publication commit: SELF

## Changes/files

Implementation commit `c4f7864…` contains the exact active `005-b` order
transcript and selector plus:

- `src/service/rle.py`: fixed-size column-major NumPy transition detection,
  compatibility-preserving counts, run limits and serialization deadline checks.
- `src/service/envelope.py` and `src/service/app.py`: request-scoped absolute
  deadline propagation/checks through all JSON/ZIP serialization loops and stable
  `504 timeout` mapping.
- `src/service/resources.py` and `src/core/sinks.py`: exact L3 annotated RGB
  raw-allocation preflight and reserved debug-sink budget accounting, including
  a zero remaining debug budget boundary.
- `src/service/settings.py`: private operator-only serialization-delay injection
  for bounded live deadline evidence; it is not request-selectable.
- `tests/test_parity_hardening.py` and `tests/test_service_units.py`: chunked
  RLE, large uniform/checkerboard, deadline recovery, visualization preflight,
  resource recovery and settings coverage.
- `docs/API.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md` and
  `docs/SERVICE-DATASHEET.md`: deadline/resource behavior and the prominent
  nonredistributable fixture/release constraint.

No goats derivatives, evidence payloads, package data, model weights or
credentials were added.

## Acceptance evidence

1. **Nonredistributable semantic E2E:** PASSED — aliases `goats-A` and `goats-B`
   were cropped in memory from the exact central box to `2784x2088`; both ran
   L2 JSON and L3 JSON/ZIP. Five-field YOLO, identity IDs, object counts, exact
   RLE dimensions/areas and the annotated artifact were validated. Config digest
   was `99e93c1a0a35e919cf8cd88685b0ca7c001012f286c3325746e4ad8e37ed1921`.
   Crop digests were `goats-A=e8be6f2221cfccdafd41ca2d32065045a6abea6ccedde4229413fd3463ebc7ec`
   and `goats-B=1f865d0f657ff03d39daf223ba22647c1fc4a6000046627f1134042b27ddad29`.
2. **Fixture rights/release constraint:** PASSED — parity matrix and datasheet
   prominently state `NONREDISTRIBUTABLE — local academic E2E only; excluded
   from packages/release fixtures`; Objective 006 is explicitly required to
   review and exclude the existing assets unless rights clearance changes.
3. **RLE/deadline:** PASSED — prior RLE bytes were matched on random masks;
   chunk boundaries, large uniform/checkerboard masks, run limits, exact
   round-trips and an expired-deadline-then-success path passed. Live operator
   RLE-limit rejection returned `413 response_too_large`; live serialization
   deadline rejection returned `504 timeout`; each recovered after restart.
4. **Visualization raw allocation:** PASSED — a two-stream boundary test
   rejected before the engine with no calls, exact `height*width*3` bytes were
   reserved per stream, and the equality boundary accepted with zero debug
   bytes remaining. L0-L2 retained no-render behavior.
5. **Auth/resource/privacy recovery:** PASSED — missing/wrong bearer returned
   `401` for both completions and metrics; the correct temporary process-env key
   succeeded and was never printed or persisted. Simulated host and shm floors
   returned `507 insufficient_memory` and `507 insufficient_shm`; RLE and
   artifact limits returned sanitized `413`; all were followed by normal
   readiness/success recovery. Metrics and log scans contained finite counters
   only and no key, auth header, request ID label, paths or raw fixture/config
   content.
6. **A/B/A isolation:** PASSED — L3 semantic hash for `goats-A` was
   `f96c8e1ff7f4` before and after `goats-B`; B used an independent semantic
   result (`34ce8c35cf33` at L3), proving no request-state leakage.
7. **Inherited parity/CLI/API:** PASSED — full CPU suite, focused service/API
   tests, package checks and remote CI remained green; no geometry, panoptic,
   BLIP3 or other scientific capability was activated.
8. **PR/topology/host:** PASSED — PR #49 remains the sole Objective-005 PR; the
   final report is the only intended child of the implementation head, and the
   final host is stopped with GPU1 idle, port free and shared memory clean.

### Sanitized local goat E2E table

| Alias/case | Status | Response bytes | Latency ms | Objects | Semantic hash prefix |
|---|---:|---:|---:|---:|---|
| goats-A L2 JSON | 200 | 20,724 | 2,399.0 | 4 | `12f55839c671` |
| goats-A L3 JSON | 200 | 7,210,635 | 4,353.3 | 4 | `f96c8e1ff7f4` |
| goats-A L3 ZIP | 200 | 5,388,981 | 4,233.6 | 4 | `f96c8e1ff7f4` |
| goats-B L2 JSON | 200 | 15,785 | 1,388.1 | 0 | `eb83f188afcd` |
| goats-B L3 JSON | 200 | 6,203,499 | 3,435.5 | 0 | `34ce8c35cf33` |
| goats-B L3 ZIP | 200 | 4,642,157 | 3,405.3 | 0 | `34ce8c35cf33` |

The post-restart synthetic L3 JSON contract returned `200`, 11,151 bytes and
eight objects. A post-restart goats-A L3 JSON request returned `200`, 7,219,989
bytes and nine objects under a separately sanitized config derivation.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — `338 passed, 1 skipped`, `76.74%` total coverage; the skip is the
  explicit opt-in physical-GPU1 pytest marker.
- `.venv/bin/pytest -q tests/test_parity_hardening.py tests/test_service_api.py tests/test_service_units.py`:
  PASSED — `127 passed`.
- `.venv/bin/ruff format --check .`: PASSED — 122 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src tests scripts`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python -m build --wheel`: PASSED — wheel built successfully.
- Wheel import probe for `src.service.rle`/`src.service.app`: PASSED.
- Deterministic TestClient OpenAPI probe: PASSED — two schemas byte-identical,
  `/v1/completions` present.
- Changed-diff secret scan and nonredistributable artifact scan: PASSED — no
  private-key/token pattern and no goat derivative/config path added.
- Live GPU1 goats/API/resource/auth/log/metrics probes: PASSED — sanitized
  central-crop E2E matrix, A/B/A, auth, resource floors, RLE/artifact limits,
  serialization timeout, recovery, restart and cleanup all completed.

## CI/checks

At implementation SHA `c4f786452ceabd1f5028efcadb178e608e684db5`:

| Check | State |
|---|---|
| `static (format, lint, build)` | SUCCESS |
| `tests (py3.10)` | SUCCESS |
| `tests (py3.11)` | SUCCESS |
| `tests (py3.12)` | SUCCESS |
| `Analyze (python)` | SUCCESS |
| `CodeQL` | SUCCESS |

## GPU/service/resource evidence

- Fresh live snapshots recorded physical GPU1 as NVIDIA GeForce RTX 2080 Ti,
  PCI `00000000:00:0C.0`, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, 11,264 MiB; driver
  `580.178.04`, CUDA `12.4`, Torch `2.5.1+cu124`. The service launch used
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1`; application
  device was logical `cuda:0`, with one Uvicorn worker and one inference slot.
- Protected physical GPU0 was the separate RTX 2080 Ti at PCI
  `00000000:00:08.0`, UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`.
  Protected PID `66522` remained its only compute process, with 2,492 MiB in
  the final snapshot; no ZAP-IT allocation reached GPU0.
- During the goats matrix, GPU1 used approximately 8,501–8,503 MiB. The
  post-restart goats-A process snapshot was `VmRSS=2,334,364 KiB` and
  `VmHWM=2,532,200 KiB`; the client-side peak RSS observed by the probe was
  613 MiB. Final GPU1 use returned to 6 MiB.
- The only live endpoint was IPv4 loopback `127.0.0.1:17891`. Final state was
  service stopped, port free, no service compute process, and zero children
  under `/dev/shm/slaif-zap-it`. Request bytes, crops, configs and responses
  remained in memory.

## Documentation/provenance

Updated API, parity, runbook and service datasheet documentation with the
serialization deadline, raw visualization reservation, operator test hook and
nonredistributable academic-fixture constraint. Model identities, runtime pins,
GPU facts and service policy remain inherited from accepted earlier objectives.
No model weights, credentials, source fixture bytes, prompts, labels or customer
data were committed or placed in OAP evidence.

## Deferred human adjudication

- Critical register action: NONE
- The finalized order explicitly resolves `NONE`; no critical-register append,
  edit, close or disposition was performed.

## Safety/scope confirmations

- Geometry, panoptic and BLIP3 live enablement were not added.
- No LAN/public exposure, gateway, firewall/VPN, TLS, Docker, systemd,
  multi-worker service, model substitution, persistence or customer data.
- Physical GPU0, PID 66522, system CUDA/driver, unrelated services/ports and
  global credentials were not modified.
- PR #49 was amended only; no merge or auto-merge action was performed.
- Existing academic assets remain tracked only as local nonredistributable
  source fixtures; no derived artifact or package/release fixture was added.

## Limitations/blockers

The goats run is bounded local academic regression evidence, not an accuracy
benchmark, golden fixture, SLA, soak test, production-readiness claim or rights
clearance. The qualified live profile remains resident SAM2+CLIP, loopback-only,
single-process and serialized. Packaging, gateway integration, licensing review
and final release remain Objective-006/human-gated work.

## Factual strategic follow-up

Strategic review/acceptance and any merge decision remain outside coding scope.
Objective 006 must inspect and exclude the existing nonredistributable academic
assets from distributable artifacts unless human rights clearance changes. The
next action remains governed by the OAP process; coding does not select another
order.

Implementation head SHA: c4f786452ceabd1f5028efcadb178e608e684db5
Report publication commit: SELF
