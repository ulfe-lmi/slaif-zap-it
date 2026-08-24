# OAP Coding-Agent Report — 008-a

## Work order

- Identifier/order/objective/PR mode: `008-a` / Objective 008 / `CREATE_NEW_PR`
- Objective: qualify the operator-assigned RTX 3090 all-resident SAM2/CLIP/BLIP3 profile and portable strict physical-GPU index handling.
- Required branch: `oap/008-a-rtx3090-all-resident-qualification`

## Status

COMPLETE

## Executive summary

Objective 008 is implemented and live-qualified on the explicitly assigned
hinton2 physical GPU index 0. The production loader selected
`sam2_clip_blip3_gpu_resident` from 24576 MiB physical capacity, loaded the
pinned FP16 SAM2, CLIP, and BLIP3 holders on logical `cuda:0`, produced seven
bounded real BLIP3 answers in Phase A, and recorded zero residency transitions.
Peak Phase-A Torch reservation was 11912.0 MiB, strictly below the 22118.4 MiB
90%-of-physical-capacity gate.

The authenticated loopback service passed readiness, injected-failure recovery,
real BLIP3 response, client-close/drain recovery, and the exact ten-call local
academic A/B sequence. The final PR-head CI and CodeQL checks are green. The PR
is open and unmerged.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#64](https://github.com/ulfe-lmi/slaif-zap-it/pull/64)
- PR title: `Objective 008: qualify RTX 3090 all-resident BLIP3`
- PR state: `OPEN`, non-draft, merge state `CLEAN`
- Base: `main` at starting SHA `bdc9aad62a813d7830b4b6920de03fb106f3f886`
- Head branch: `oap/008-a-rtx3090-all-resident-qualification`
- Implementation head SHA: `927279f6803e53fb466badb0df3a364acf1f1b14`
- Implementation commit parent: `bdc9aad62a813d7830b4b6920de03fb106f3f886`
- Report publication commit: SELF
- New PR: yes; amended existing PR: no; coding merge/auto-merge: NO
- Remote branch head was verified equal to the implementation SHA before report publication.

## Changes/files

Implementation commit `927279f6803e53fb466badb0df3a364acf1f1b14` contains the
bounded 008-a scope:

- Operator index portability and fail-closed capacity/name/UUID readiness in
  `src/runtime/device.py`, `src/runtime/strategy.py`,
  `src/runtime/live_service.py`, and `src/runtime/__init__.py`.
- FP16 all-resident SAM2/CLIP/BLIP3 loading, SAM2 autocast compatibility,
  exact logical-device handling, one-shot operator test injection, and a fixed
  zero-transition metric.
- Launcher, entrypoint, environment template, systemd template, architecture,
  agent compact law, installer, README, API, runbook, and runtime evidence
  documentation updates.
- Focused runtime, residency, boundary, launcher, integration, and goat-harness
  tests in `tests/` and `scripts/smoke_local_goats.py`.
- Exact unchanged active selector `oap/active` and exact active order
  `oap/orders/008-a-rtx3090-all-resident-qualification.md`.
- No CRITICAL register file change and no model/cache/fixture bytes.

The final report path is the only path changed by the SELF commit.

## Acceptance evidence

1. **PR topology and bounded diff — PASSED.** One new Objective-008 PR (#64)
   exists from the specified `main` base; the implementation commit has the
   exact active selector/order transcript; the report is published as its sole
   final-child path.

2. **Strict operator index and device binding — PASSED.** CPU tests cover
   explicit indices 0 and 1, decimal parsing, missing/invalid/negative values,
   inconsistent `CUDA_VISIBLE_DEVICES`, wrong UUID, wrong visible count,
   occupancy, model-name mismatch, and capacity mismatch. The launcher requires
   `SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX`, canonicalizes only decimal leading zeros,
   and exports that same value as `CUDA_VISIBLE_DEVICES` before service Python
   starts. Requests and YAML have no device-selection path.

3. **Capacity boundary and sequential preservation — PASSED.** `24575 MiB`
   selects `sam2_clip_gpu_blip3_cpu_swap`; `24576 MiB` selects
   `sam2_clip_blip3_gpu_resident`. Existing sequential transition tests and
   accepted low-card documentation remain intact.

4. **Phase A simultaneous pinned residency — PASSED after one documented
   harness-config attempt.** The corrected offline production-loader run used
   the exact assigned device and pinned revisions, generated 128x128 input, and
   exercised real SAM2, CLIP, and BLIP3. It returned seven objects and seven
   bounded answers with zero transition events.

   | Measure | Result |
   | --- | ---: |
   | Loader / chain time | 182.395 s / 10.979 s |
   | Physical capacity / strict ceiling | 24576 / 22118.4 MiB |
   | Torch load allocated / reserved | 9627.5 / 9784.0 MiB |
   | Torch inference current allocated / reserved | 9635.6 / 11912.0 MiB |
   | Torch inference peak allocated / reserved | 11188.8 / 11912.0 MiB |
   | CUDA free after inference | 11864.8 MiB |
   | Host maximum RSS | 12793.5 MiB |
   | Objects / answers | 7 / 7 |
   | Residency transitions | 0 |
   | Holders proven on logical device | `cuda:0`, true |

   The first Phase-A attempt loaded the same resident profile and stayed below
   the memory gate, but its generated rule did not match the CLIP labels and
   therefore produced zero answers. It was recorded as `FAILED`, exited without
   fallback, and returned the GPU to baseline. The corrected `any,1.0` attempt
   was the passing gate. No OOM, downgrade, reload, CPU migration, or
   request-time movement occurred.

5. **Phase B loopback matrix — PASSED.** One authenticated loopback worker on
   port 17891 returned readiness HTTP 200 with the all-resident strategy,
   logical `cuda:0`, and all three model identities without path leakage. A
   one-shot operator failure returned HTTP 500 `inference_failure`; the next
   no-BLIP request returned HTTP 200. A corrected real-BLIP3 L3 request returned
   HTTP 200 with `blip3=executed` and eight bounded answers. A separately
   restarted one-shot client-close/drain attempt closed its authenticated
   socket, waited eight seconds, and the next request returned HTTP 200 in
   13583.1 ms. The process remained ready and the transition counter stayed 0.

6. **Phase C exact local regression — PASSED.** The safe harness cropped the
   ignored 5568x4176 fixtures in memory to 2784x2088 and sent exactly
   `A,B,A,B,A,B,A,B,A,B` as ten authenticated L3 JSON requests. All ten were
   HTTP 200, all reported `blip3=executed`, all had zero transitions, all
   reported three runtime model identities, and A/B semantic digests were
   repeatable. Object and answer counts were 0/0 on every call, matching the
   accepted 007-b academic baseline; no answer text was retained.

   | Image | E2E first / min / median / nearest-rank p95 / max (ms) | SAM2 range (ms) | CLIP range (ms) | BLIP3 range (ms) |
   | --- | ---: | ---: | ---: | ---: |
   | A | 4218.5 / 4170.0 / 4182.6 / 4218.5 / 4218.5 | 795.259–857.730 | 103.035–106.367 | 253.620–266.017 |
   | B | 3096.9 / 3096.9 / 3105.8 / 3318.7 / 3318.7 | 150.526–150.754 | 60.678–63.618 | 28.763–30.676 |
   | Aggregate | 4218.5 / 3096.9 / 4170.0 / 4218.5 / 4218.5 | — | — | — |

   Each sample recorded 11189.0 MiB peak allocated, 13200.0 MiB peak reserved,
   10576.8 MiB sampled free, and 12752.2 MiB maximum RSS. The stable YOLO
   digest prefix was `e3b0c44298fc1c14`; zero persistence was proven before and
   after the sequence.

7. **Documentation/provenance — PASSED.** Current docs record the exact hinton2
   index/UUID/PCI/model/capacity/driver/Torch/CUDA facts, pinned revisions,
   startup and phase timings, memory ceiling calculation, zero-transition
   proof, low-card comparison, host-assignment history, cleanup, and the
   existing BLIP3 CC-BY-NC/non-commercial release limitation. No accuracy,
   commercial authorization, production SLA, public exposure, or release claim
   was added.

8. **Verification and CI — PASSED.** See the exact statuses below. The
   canonical CPU suite’s one skipped item is the explicit opt-in GPU module;
   that module was then run on the assigned card and passed separately.

9. **Final cleanup and non-interference — PASSED.** After service stop, port
   17891 was free, no ZAP-IT process or compute row remained, the assigned card
   was 15 MiB used / 24109 MiB free, `/dev/shm/slaif-zap-it` was empty, and no
   unrelated device/process/service was changed.

The strongest reason not to accept was addressed directly: the prior threshold
was only fake-tested, old guards rejected physical index 0, and marketed 24576
MiB leaves little margin. The new strict index/UUID/capacity tests plus real
simultaneous and repeated pinned inference measured 11912.0 MiB peak reserved
(53.85% of marketed capacity), zero movements, and recovery after failure and
client close; the live result is still limited to this assigned local host.

## Verification

- `.venv/bin/pytest -q tests/test_runtime_units.py tests/test_live_runtime.py tests/test_live_service_units.py tests/test_adaptive_residency.py`: `PASSED` — focused suite during implementation, 83 passed before final additions.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: `PASSED` — 388 passed, 1 intentional opt-in GPU marker skipped, 77.90% total coverage.
- `ZAP_IT_RUN_GPU=1 SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 SLAIF_ZAP_IT_EXPECTED_GPU_UUID=<assigned> .venv-gpu/bin/python -m pytest -q -m gpu tests/test_gpu_integration.py`: `PASSED` — 1 passed in 9.10 s.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/ruff format --check .`: `PASSED` — 134 files already formatted.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 20 current documents.
- `.venv/bin/detect-secrets scan --baseline .secrets.baseline`: `PASSED`; tracked-tree enforcement found exactly the existing five reviewed findings and the baseline bytes remained unchanged.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree`: `PASSED` — 5 known tracked findings, no additions/removals.
- `.venv/bin/python scripts/scan_release_artifacts.py --baseline .secrets.baseline dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: `PASSED` — both archives audited with no unexpected findings.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED` — wheel and sdist built; upstream setuptools deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: `PASSED` — 63 wheel members and 146 sdist members.
- Temporary no-dependency wheel install/import/boundary probe: `PASSED` — installed package imported, index-0 mask helper worked, and 24575/24576 strategy boundary was verified; temporary `/dev/shm` install directory was removed.
- `git diff --check`: `PASSED`.

## CI/checks

All checks below were successful on implementation head
`927279f6803e53fb466badb0df3a364acf1f1b14`:

- `static (format, lint, build)`: `SUCCESS`
- `release (artifact audit)`: `SUCCESS`
- `tests (py3.10)`: `SUCCESS`
- `tests (py3.11)`: `SUCCESS`
- `tests (py3.12)`: `SUCCESS`
- `Analyze (python)`: `SUCCESS`
- `CodeQL`: `SUCCESS`

## GPU/service/resource evidence

- Assigned physical device: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver 610.43.02.
- Masked application view: exactly one visible device, logical `cuda:0`,
  matching UUID, Torch 2.5.1+cu124, CUDA runtime 12.4, 24123.5 MiB usable.
- Pre/post live snapshots: 15 MiB used / 24109 MiB free and no compute rows.
  During the service, only the owned service PID appeared on the assigned
  device. The complete host inventory exposed no second device; no unassigned
  GPU/process was touched.
- Phase-A peak: 11188.8 MiB allocated / 11912.0 MiB reserved; Phase-C peak:
  11189.0 / 13200.0 MiB. All are below 22118.4 MiB.
- Phase-B final content-free metrics: transition count 0, current/peak
  allocation 9635.6/11188.8 MiB, current/peak reservation 13200.0/13200.0
  MiB, free 10576.8 MiB, maximum RSS 12752.2 MiB.
- Host preflight: 22904 MiB RAM, approximately 21009 MiB available; final
  cleanup snapshot reported 21116 MiB available. `/dev/shm` was a 12-GiB
  tmpfs and was empty after stop.
- Service: loopback only at `127.0.0.1:17891`, one process, one Uvicorn worker,
  one active inference. Port was free after stop; no persistent request data,
  log content, or cache path was observed.

## Documentation/provenance

No dependency or model revision changed. The three approved revisions remain
the operator-pinned SAM2, CLIP, and BLIP3 identities in `src/runtime/models.py`.
The loader adds no download path and live processes used offline mode. Docs and
launcher examples explicitly distinguish the historical maelstrom1 physical
GPU1 qualification from this hinton2 physical GPU0 assignment; index 1 remains
a valid explicit operator configuration.

## Deferred human adjudication

- Critical register action: NONE
- The active order’s deferred-adjudication decision is `NONE`; no CRITICAL
  bytes were appended or changed. Existing `CRIT-0001` remains outside this
  local qualification’s gate.

## Safety/scope confirmations

- No merge, auto-merge, tag, release upload, external deployment, LAN bind,
  gateway/TLS/firewall/VPN/systemd activation, driver/CUDA change, global
  credential change, or unrelated service/process mutation occurred.
- No physical device other than the explicitly assigned index 0 was selected;
  no GPU reset, process kill, memory stealing, or fallback occurred.
- No request image, YAML, crop, prompt, answer text, response body, API key,
  model weight, cache path, or customer data entered Git or the OAP report.
- No report-only prose was included in the implementation commit. The final
  SELF commit changes only `oap/reports/008-a-report.md`.

## Limitations/blockers

- This is bounded local research evidence on the assigned hinton2 RTX 3090,
  not a production deployment or universal performance claim.
- BLIP3 remains subject to its existing CC-BY-NC/non-commercial limitation and
  release/deployment gate. No commercial authorization is inferred.
- The goat configuration intentionally produced zero objects/answers; only
  stage execution, sanitized counts, stable digests, timing, memory, and
  state-isolation facts are compared because the accepted predecessor did not
  retain answer-content or floating-score parity data.
- The first Phase-A harness configuration attempt failed its non-empty-answer
  gate because its rule did not match the generated labels; the corrected
  production-path attempt passed all hard gates and both attempts cleaned up.

## Factual strategic follow-up

PR #64 remains open and unmerged for strategic review/acceptance. No deployment
or next-order action was started by coding.

Implementation head SHA: 927279f6803e53fb466badb0df3a364acf1f1b14
Report publication commit: SELF
