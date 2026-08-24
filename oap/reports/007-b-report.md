# OAP Coding-Agent Report — 007-b

## Work order

- Identifier: `007-b`
- Numeric objective: `007`, correct stage residency and complete sequential
  BLIP3 qualification
- Mode: `AMEND_EXISTING_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#62](https://github.com/ulfe-lmi/slaif-zap-it/pull/62), existing branch
  `oap/007-a-adaptive-blip3-residency`

## Status

COMPLETE

## Executive summary

Corrected the sequential lifecycle defect identified in 007-a. SAM2 and CLIP
now run on logical `cuda:0` before an explicit engine BLIP3 stage boundary;
only that boundary swaps SAM2+CLIP to CPU, moves BLIP3 to GPU, runs BLIP3, and
restores the baseline in a mandatory context-manager `finally` before the
pipeline returns or propagates an error. Added stage-aware CPU/fake coverage,
driver-capacity tolerance coverage, fixed-loader coverage, peak-memory metrics,
and a sanitized ten-call benchmark harness.

The corrected pinned FP16 BLIP3-only gate passed on physical GPU1. The real
11-GB sequential loopback service passed startup/readiness, no-BLIP control,
BLIP3 execution, failure injection, client-abort restoration, and the exact
alternating ten-call goat benchmark. The exclusive >=24-GB all-resident
qualification remains pending for future 007-c.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR URL: https://github.com/ulfe-lmi/slaif-zap-it/pull/62
- State at implementation publication: `OPEN`, non-draft, `MERGEABLE`
- Base: `main` at `85af4fd562fdf128709a0160bc36884a585e1a5e`
- Branch: `oap/007-a-adaptive-blip3-residency`
- Starting PR/report head SHA: `233e5def21aacf2283a5a5c5ced9e00452c361c5`
- Implementation head SHA: `3267abb0acf582ce478f4e90a8ad748cadc45439`
- Report publication commit: SELF
- New PR: no
- Amended existing PR: yes
- Coding merge/auto-merge: NO

## Changes/files

- `src/core/engine.py`: optional context-managed BLIP3 stage hook around the
  actual BLIP3 call.
- `src/runtime/live_service.py`: sequential swap/restore lifecycle moved from
  around the whole runner to the BLIP3 stage boundary, with terminal restore
  failure handling retained.
- `src/runtime/device.py`: bounded downward tolerance for the real driver
  difference between physical `nvidia-smi` capacity and masked CUDA usable
  capacity; physical capacity still selects the strategy.
- `src/service/metrics.py`, `src/service/app.py`: per-inference logical CUDA
  peak allocated/reserved/free and host RSS evidence with fixed labels.
- `tests/test_adaptive_residency.py`: stage-aware SAM2/CLIP/BLIP3 residency,
  no-BLIP/all-resident, pre-transition failure, BLIP3 failure/timeout,
  restoration failure, request-local rules, capacity tolerance, and
  `torch_dtype`/local-loader tests.
- `scripts/smoke_local_goats.py`: exact A/B benchmark stage, transition,
  memory, RSS, repeatability, semantic-digest, and request-workspace evidence;
  service-owned runtime files are excluded from request-persistence accounting.
- Current runtime/API/runbook/datasheet/output/release documentation records
  measured 007-b evidence and pending 007-c scope.
- `oap/active` is exactly `007-b`; the exact matching order is committed.
- No prior order/report or `CRITICAL.md` bytes were changed.

## Acceptance evidence

1. **Existing PR and bounded implementation:** `PASSED`. PR #62 remains the
   sole open Objective-007 PR with unchanged title, base and branch. The remote
   branch points to implementation SHA
   `3267abb0acf582ce478f4e90a8ad748cadc45439` before this report child.

2. **Correct stage order and restoration:** `PASSED`. CPU/fake stage-aware
   tests observe SAM2 and CLIP on GPU, BLIP3 on GPU only inside the hook after
   baseline holders move to CPU, and baseline holders restored before return.
   Tests cover SAM2+BLIP3 without CLIP, SAM2+CLIP+BLIP3, no-BLIP, all-resident,
   pre-transition failure, BLIP3 failure, timeout-like failure and terminal
   restoration failure.

3. **Corrected pinned FP16 BLIP3 gate:** `PASSED`. Offline local-only Phase A
   constructed the pinned `Salesforce/xgen-mm-phi3-mini-instruct-r-v1`
   revision `1d91d356d3b6fbc141140edf490b39890417af44` in host RAM, used FP16,
   moved it to GPU1, and completed a bounded 128x128 yes/no inference. Host
   load was 176.983 s, GPU move 2.176 s, inference 2.286 s, answer length 4
   and non-empty. Peak was 9,327.9 MiB allocated / 9,532.0 MiB reserved from
   10,820.9 MiB CUDA-visible total: 88.09%, below the 95% gate. No OOM or
   intrinsic FP16 failure occurred. External process-exit cleanup returned
   GPU1 to 6 MiB used.

4. **Real sequential service:** `PASSED`. One authenticated loopback process
   on freshly verified `127.0.0.1:17891` became ready with
   `sam2_clip_gpu_blip3_cpu_swap`; readiness became `200` within the observed
   6.1-second startup probe window. A generated no-BLIP L3 request returned
   200 with `blip3=not_configured` and no residency-transition counter change.
   A generated real BLIP3 L3 request returned 200 with `blip3=executed`, eight
   bounded answer fields and successful `to_blip3`/`restore` counters. A
   client-aborted BLIP3 request returned a client timeout while the worker
   drained; the restore counter reached success before a subsequent no-BLIP
   request returned 200. A fresh operator-only failure injection returned
   sanitized HTTP 500 `inference_failure`.

5. **Exact goat benchmark:** `PASSED`. The updated local-only harness used
   in-memory central-50% crops of both 5568x4176 images (2784x2088 each), with
   no derivative persistence, and sent exactly ten L3 JSON calls in
   `A,B,A,B,A,B,A,B,A,B` order. Every response was HTTP 200 and every L3 stage
   reported `blip3=executed`; A and B semantic digests were repeatable.

   | # | Image | Latency ms | SAM2 / CLIP / BLIP3 ms | To-BLIP3 / restore s | Peak alloc / reserved MiB | Free MiB | Objects / answers |
   |---:|:---:|---:|---:|---:|---:|---:|---:|
   | 1 | A | 11412.7 | 1127.471 / 100.726 / 6636.364 | 2.324 / 4.026 | 9465.8 / 9662.0 | 8930.2 | 0 / 0 |
   | 2 | B | 10193.8 | 401.540 / 55.906 / 6261.553 | 2.307 / 3.927 | 8902.5 / 9052.0 | 8930.2 | 0 / 0 |
   | 3 | A | 11273.9 | 1098.506 / 94.580 / 6537.982 | 2.303 / 3.949 | 9465.8 / 9662.0 | 8930.2 | 0 / 0 |
   | 4 | B | 10284.4 | 397.470 / 55.012 / 6341.419 | 2.305 / 4.007 | 8902.5 / 9052.0 | 8930.2 | 0 / 0 |
   | 5 | A | 11484.1 | 1116.722 / 94.368 / 6726.721 | 2.330 / 4.109 | 9465.8 / 9662.0 | 8930.2 | 0 / 0 |
   | 6 | B | 10389.7 | 408.402 / 55.154 / 6429.286 | 2.329 / 4.072 | 8902.5 / 9052.0 | 8930.2 | 0 / 0 |
   | 7 | A | 11345.6 | 1114.698 / 95.880 / 6604.880 | 2.304 / 4.014 | 9465.8 / 9662.0 | 8930.2 | 0 / 0 |
   | 8 | B | 10186.7 | 420.312 / 55.604 / 6218.817 | 2.321 / 3.869 | 8902.5 / 9052.0 | 8930.2 | 0 / 0 |
   | 9 | A | 11302.9 | 1082.628 / 94.211 / 6594.054 | 2.305 / 4.002 | 9465.8 / 9662.0 | 8930.2 | 0 / 0 |
   | 10 | B | 10150.0 | 413.027 / 55.800 / 6192.563 | 2.309 / 3.856 | 8902.5 / 9052.0 | 8930.2 | 0 / 0 |

   Per-image latency statistics were A first/minimum/median/p95/max
   `11412.7/11273.9/11345.6/11484.1/11484.1 ms`, B
   `10193.8/10150.0/10193.8/10389.7/10389.7 ms`, and aggregate
   `11412.7/10150.0/11273.9/11484.1/11484.1 ms`. The stable YOLO semantic
   digest prefix was `e3b0c44298fc1c14`; answer count was zero for this
   academic configuration, with no answer content printed or persisted. Host
   RSS high-water reached 16,003.6 MiB by call 7 and remained flat thereafter;
   this bounded sequence showed no unbounded growth.

6. **Cleanup and protected resources:** `PASSED`. After the final service stop,
   GPU1 was 6 MiB used / 10,815 MiB free, port 17891 had no listener, and
   `/dev/shm/slaif-zap-it` had no entries. GPU0 remained the unrelated
   `GPU-4c129e25-8e59-eee4-b49c-56c40e294182` with PID 66522 and 2,492 MiB
   process usage; no ZAP-IT allocation or unrelated process was changed.

7. **Strongest reason not to accept without new evidence:** 007-a's fake
   lifecycle test allowed a whole-pipeline runner that never executed stages,
   so it missed the pre-runner swap, and no real BLIP3 load had succeeded. This
   round answers that risk with engine-level stage-aware fakes, corrected
   `torch_dtype` loader coverage, a real offline FP16 gate, real service
   `blip3=executed` metadata, repeated transitions/restores and the exact
   alternating benchmark above.

## Verification

- `.venv/bin/python -m coverage run --branch -m pytest -q`: **PASSED** — 372
  passed, 1 honest GPU integration skip; 78% total branch coverage.
- `.venv/bin/python -m pytest -q tests/test_adaptive_residency.py tests/test_verifier_blip3.py tests/test_service_api.py`: **PASSED** — 65 passed.
- `.venv/bin/ruff format --check .`: **PASSED**.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules tests scripts ...`: **PASSED**.
- `bash -n scripts/*.sh`: **PASSED**.
- `git diff --check`: **PASSED**.
- `.venv/bin/python -m build --no-isolation`: **PASSED** — wheel and sdist.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree dist/*.whl dist/*.tar.gz`: **PASSED** — 0 unexpected archive findings and exactly 5 tracked baseline findings.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: **PASSED**.
- `.venv/bin/twine check dist/*`: **PASSED**.
- Clean external-venv installed wheel JSON/ZIP smoke: **PASSED** — package
  version and `zap-it-service` entry point verified without checkout imports.
- Offline masked Phase A BLIP3-only qualification: **PASSED** — hard gate and
  external cleanup evidence above.
- Authenticated Phase B service controls and exact ten-call harness: **PASSED**.

## CI/checks

All required checks passed for implementation SHA
`3267abb0acf582ce478f4e90a8ad748cadc45439` before report publication:

- `static (format, lint, build)`: **PASSED** — CI run `32687531576`.
- `release (artifact audit)`: **PASSED** — CI run `32687531576`.
- `tests (py3.10)`: **PASSED** — CI run `32687531576`.
- `tests (py3.11)`: **PASSED** — CI run `32687531576`.
- `tests (py3.12)`: **PASSED** — CI run `32687531576`.
- `Analyze (python)`: **PASSED** — CodeQL run `32687531627`.
- `CodeQL`: **PASSED** — check run `97315357237`.

The final SELF report head was rechecked with `gh pr checks 62` before signal;
the same named CI/CodeQL checks were successful and PR #62 remained open and
unmerged.

## GPU/service/resource evidence

- Physical GPU1: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`,
  physical total 11,264 MiB; masked application device `cuda:0`, Torch usable
  total 10,820.9 MiB.
- Physical GPU0: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, unrelated PID 66522; untouched.
- Launch mask for every live phase: `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`; model access was local-only with offline Hub and
  Transformers settings.
- Service: one authenticated Uvicorn process, one worker, one inference slot,
  loopback only at port 17891; stopped after evidence.
- Phase A peak: 9,327.9 allocated / 9,532.0 reserved MiB; service benchmark
  peak range: 8,902.5–9,465.8 allocated / 9,052.0–9,662.0 reserved MiB.
- Host memory: Phase A isolated process maximum RSS 12,584.4 MiB; service
  benchmark high-water RSS 16,003.6 MiB.
- `/dev/shm/slaif-zap-it`: mode-0700 operator root, zero request files during
  the final benchmark accounting, empty after service stop.
- No raw image, YAML, prompt, answer, response body, credential, model weight,
  crop or derivative entered the repository, OAP report or service log.

## Documentation/provenance

Updated runtime, GPU, API, config, output-parity, service-datasheet, runbook
and release-gate documentation with measured 007-b facts, fixed-capacity
cross-check behavior, metric evidence, benchmark semantics and the exclusive
007-c all-resident prerequisite. Pinned model identities/revisions and the
existing academic/non-commercial provenance remain unchanged. No runtime
dependency was added.

## Deferred human adjudication

- Critical register action: **NONE**
- `CRITICAL.md` remained byte-identical at object ID
  `29c2366359fb5a05e151fe546bcf6330477f60ee`.

## Safety/scope confirmations

- Only the existing Objective-007 PR/branch was amended; no second PR was
  created and no merge or auto-merge was enabled.
- Physical GPU0, PID 66522, unrelated services/listeners, firewall, VPN,
  systemd, drivers, CUDA installation and global credentials were not changed.
- No quantization, CPU BLIP3 inference, model substitution, offload, request
  selector, schema change, persistent request workspace or LAN exposure was
  introduced.
- Prior OAP orders/reports remain immutable; the active selector remains exactly
  `007-b`.

## Limitations/blockers

- The >=24-GB all-resident profile is implemented and CPU/fake-tested but was
  not live-qualified in this round; that is the exclusive future 007-c gate.
- The goat sequence is a state/isolation/resource regression, not an accuracy
  benchmark. Its academic configuration produced zero objects and zero answer
  fields while still executing the real BLIP3 stage on every call.
- This evidence does not authorize public exposure, production deployment,
  customer data, commercial BLIP3 use, release, or merge.

## Factual strategic follow-up

Strategic review may adjudicate the open PR and, if separately ordered, prepare
007-c for an exclusive >=24-GB card. Coding does not choose, start or merge
that follow-up.
