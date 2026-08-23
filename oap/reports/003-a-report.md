# OAP Coding-Agent Report — 003-a

## Work order

- Identifier: `003-a` (numeric objective `003`, round `003-a`)
- Order: `oap/orders/003-a-gpu1-runtime-qualification.md`
- PR mode: `CREATE_NEW_PR`
- Exact `oap/active` transcript and the selected order were carried in the
  implementation commit.

## Status

COMPLETE

## Executive summary

Qualified the actual physical GPU1 runtime on `maelstrom1` with a repo-owned
CPython 3.12 pip environment, exact cu124 Torch/model dependency pins,
revision/license provenance, an audit of BLIP3 remote code, a fail-closed
physical-device guard, operator-owned resource profiles, `/dev/shm` workspace
and loopback-port helpers, and explicit opt-in live GPU coverage.

Live evidence supports resident SAM2+CLIP on the 11 GiB target. The approved
BLIP3 checkpoint was conservatively rejected before loading because its
predicted peak exceeded the 90% budget. No alternate scientific model was
substituted. No service was started or activated.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/47 — OPEN, not merged,
  merge state CLEAN
- Required title: `Objective 003-a: physical GPU1 runtime qualification, model
  revision/license pinning, measured resource strategy`
- Base: `main` @ `b3bf252ca7a37c00a75276c4d5bac176316655e0`
- Branch: `oap/003-a-gpu1-runtime-qualification`
- Starting SHA: `b3bf252ca7a37c00a75276c4d5bac176316655e0`
- Implementation head SHA: `10bb5778786d27ffcf2ab1b7e7dc13690e5e62c7`
- Report publication commit: SELF
- New PR: yes (#47); existing PR amended: no; coding merge: NO

## Changes/files

Implementation commit `10bb5778786d27ffcf2ab1b7e7dc13690e5e62c7` contains:

- `requirements-gpu-cu124.lock`, `scripts/qualify_gpu_runtime.py`, and the
  `src/runtime/` device, model, readiness, strategy, shared-memory and port
  helpers;
- lazy optional `detectron2` loading in `modules/visualizer.py`, pinned
  revision-aware SAM2/CLIP/BLIP3 loaders, and API runtime-profile rejection;
- `tests/test_runtime_units.py`, `tests/test_gpu_integration.py`, and the
  live-test harness changes in `tests/conftest.py`;
- `docs/runtime.md`, updated installation/provenance/security/navigation
  documentation, and the stale GPU-capacity wording correction in
  `ARCHITECTURE.md`;
- `oap/active` changed to `003-a` and the exact selected order transcript
  `oap/orders/003-a-gpu1-runtime-qualification.md`.

## Acceptance evidence

1. **Physical GPU1 identity and masking — PASSED.** Live PyTorch evidence
   showed exactly one visible device as logical `cuda:0`: physical UUID
   `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI
   `00000000:00:0C.0`, NVIDIA GeForce RTX 2080 Ti, 11264 MiB. The strict
   guard requires `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`,
   one visible device, and the pinned UUID.

2. **GPU0 protection — PASSED.** The qualification runner sampled all GPUs and
   compute processes before/after each stage, each supported combination and
   the full run. GPU0 remained at 2161 MiB with only the pre-existing
   unrelated Python process; no ZAP-IT process or allocation appeared there.
   The final post-test snapshot was byte-equivalent on GPU0. GPU1 returned to
   6 MiB used / 10815 MiB free after the process exited.

3. **Reproducible real runtime — PASSED.** `requirements-gpu-cu124.lock`
   installs CPython 3.12-compatible Torch 2.5.1/cu124, TorchVision 0.20.1,
   TorchAudio 2.5.1, SAM2 source, Transformers 4.41.1 and the pinned support
   stack. The masked import smoke passed for the real stack. `detectron2` is
   optional and lazy.

4. **Model identity, license and remote-code provenance — PASSED.** The
   approved Hugging Face revisions are recorded in `src/runtime/models.py`,
   `THIRD_PARTY_NOTICES.md` and [docs/runtime.md](../../docs/runtime.md):
   SAM2 `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251` (Apache-2.0), CLIP
   `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (research card with no SPDX
   field and deployed use out of scope), and BLIP3
   `1d91d356d3b6fbc141140edf490b39890417af44` (CC-BY-NC-4.0). The audited
   BLIP3 remote-code file hashes and operator-only `trust_remote_code=True`
   boundary are documented; request YAML cannot select models, revisions,
   cache roots, devices or remote code.

5. **Individual stage evidence — PASSED/BLOCKED as designed.** The bounded
   128x128 generated RGB fixture produced the following live measurements:

   | Profile | Status | Predicted peak | Load | Inference repeats (ms) | Peak allocated/reserved | Cleanup allocated/reserved |
   | --- | --- | ---: | ---: | --- | ---: | ---: |
   | SAM2 | `PASSED` | 1285 MiB | 2.798 s | 668.94, 375.86, 363.12 | 3079.2 / 4902.0 MiB | 8.1 / 20.0 MiB |
   | CLIP | `PASSED` | 866 MiB | 1.432 s | 12.68, 10.28, 9.90 | 588.9 / 632.0 MiB | 8.1 / 20.0 MiB |
   | BLIP3 | `BLOCKED` | 10505 MiB | not loaded | not run | hard stop before load | no model allocation |
   | SAM2 + CLIP resident | `PASSED` | 2151 MiB | 3.208 s | 434.73, 417.44, 434.17 | 3656.5 / 5530.0 MiB | 8.1 / 20.0 MiB |

   The Torch-reported total was 10820.9 MiB and the 90% ceiling was about
   9738.8 MiB. BLIP3's pinned shard set is 18,357,535,724 bytes and was
   rejected under the conservative bfloat16-plus-overhead prediction without
   an OOM attempt.

6. **Supported profiles and stability — PASSED.** The operator strategy is
   `sam2_clip_resident_blip3_rejected`; supported profiles are `sam2`, `clip`
   and `sam2_clip`. Three serial runs per supported profile completed with
   stable shapes: SAM2 returned 7 masks of 128x128 each run, CLIP returned one
   `red` label, and the combined profile returned four labels in the same
   `green, red, red, red` shape. BLIP3-only and combined BLIP3 profiles are
   rejected before engine execution by the operator policy.

7. **Fail-closed device guard — PASSED.** CPU unit tests inject wrong UUID,
   wrong visible-device count, missing launch mask, explicit CPU mode and
   readiness/registry states. Strict mode never silently falls back to CPU or
   another GPU.

8. **Shared memory — PASSED.** `/dev/shm/slaif-zap-it` was created and verified
   mode 0700 with 26964.1 MiB free. Opaque request directories are mode 0700,
   atomic files are mode 0600, symlinks/traversal are rejected, and cleanup
   leaves no child residue after the live run.

9. **Candidate loopback port — PASSED.** `127.0.0.1:17891` was selected after
   a live `ss` listener scan and transient bind check. The socket was closed
   immediately; final scans showed no listener or reservation. No service was
   started.

10. **Opt-in GPU test isolation — PASSED.** The `gpu`-marked module skips at
    collection without `ZAP_IT_RUN_GPU=1`, uses a mode-0600 RAM-backed lock,
    sets/checks the mask before importing Torch, and checks GPU0 compute-app
    evidence. The final live run passed 1 test in 3.04 seconds.

11. **CPU CI/CodeQL — PASSED.** Local CPU, packaging and lint checks passed;
    all six required GitHub checks were SUCCESS on the implementation SHA.
    Live GPU evidence is reported separately and is not represented as
    GitHub-hosted CI.

12. **PR/report contract — PASSED.** The required branch and one exact PR were
    created from the verified merged-002 base. Non-report work is remote on the
    implementation SHA; this report is being published as the single final
    child with `Report publication commit: SELF`. No merge or auto-merge was
    performed.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 259 passed, 1 intentional module-level GPU skip, 3 existing
  warnings, 74.98% total coverage.
- `.venv/bin/pytest -q tests/test_runtime_units.py tests/test_visualizer.py`:
  `PASSED` — 18 focused tests.
- `ZAP_IT_RUN_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
  SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
  SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it .venv-gpu/bin/python -m pytest
  -q -m gpu tests/test_gpu_integration.py`: `PASSED` — 1 live test in 3.04 s.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
  .venv-gpu/bin/python scripts/qualify_gpu_runtime.py`: `PASSED` — masked
  import smoke, stage/combined measurements, repeated runs and snapshots.
- `.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download`:
  `PASSED` — the three approved snapshots were downloaded at their pinned
  revisions into the operator cache; no weights entered Git.
- `uv pip check --python .venv-gpu/bin/python`: `PASSED` — all installed
  packages compatible.
- `.venv-gpu/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED` — all checks passed.
- `.venv/bin/ruff format --check .`: `PASSED` — all files formatted.
- `.venv/bin/python -m build --wheel`: `PASSED` — isolated wheel build produced
  `zap_it-0.1.0-py3-none-any.whl` including `src.runtime`.
- `git diff --check` on implementation paths excluding the exact order
  transcript: `PASSED`. The carried strategic-authored order has one trailing
  space on its original line 210; it was preserved byte-for-byte as required
  by the OAP transcript law.

## CI/checks

All checks below were SUCCESS, COMPLETED on implementation head
`10bb5778786d27ffcf2ab1b7e7dc13690e5e62c7`:

- `static (format, lint, build)` — SUCCESS
- `tests (py3.10)` — SUCCESS
- `tests (py3.11)` — SUCCESS
- `tests (py3.12)` — SUCCESS
- `Analyze (python)` — SUCCESS
- `CodeQL` — SUCCESS

The PR remained OPEN and CLEAN after the checks; no report content was part of
this implementation head.

## GPU/service/resource evidence

- Host: `maelstrom1`, Ubuntu 24.04.4 LTS, kernel `6.8.0-138-generic`.
- Driver: NVIDIA 580.178.04; system nvcc 13.3 was not used for the cu124
  wheels. Python 3.12.3; Torch 2.5.1+cu124; TorchVision 0.20.1+cu124;
  TorchAudio 2.5.1+cu124; Transformers 4.41.1; Accelerate 0.32.1; Hugging
  Face Hub 0.24.6; Pillow 10.4.0; NumPy 1.26.4; SAM2 source commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Physical GPU0 final snapshot: UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`, RTX
  2080 Ti, 11264 MiB total, 2161 MiB used; only the pre-existing unrelated
  PID 66522 Python compute process at 2152 MiB.
- Physical GPU1 final snapshot: UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`, RTX
  2080 Ti, 11264 MiB total, 6 MiB used / 10815 MiB free after cleanup.
- `/dev/shm/slaif-zap-it`: mode 0700, empty after qualification; no request
  data, model weights, credentials or generated dump was committed.
- Candidate service port: `127.0.0.1:17891`, verified unused and not left
  listening. No Uvicorn process, systemd unit, firewall, driver, CUDA system
  installation, or unrelated service was changed.

## Documentation/provenance

- Runtime record: [docs/runtime.md](../../docs/runtime.md)
- Installation and reproduction: [INSTALL.md](../../INSTALL.md)
- Notices and model terms: [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
- Short GPU isolation law: [docs/GPU-RUNTIME.md](../../docs/GPU-RUNTIME.md)
- Source model cards: [SAM2](https://huggingface.co/facebook/sam2-hiera-large),
  [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [XGen-MM/BLIP3](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1)
- The BLIP3 audit records SHA-256 hashes for the pinned
  `configuration_xgenmm.py`, `modeling_xgenmm.py`,
  `image_processing_blip_3.py`, `vlm.py` and `utils.py` files. The service
  boundary rejects request-controlled model/revision/cache/device/remote-code
  settings.

## Deferred human adjudication

- Critical register action: NONE
- Order decision `NONE` was honored; no critical entry was appended and no
  candidate meeting the five-condition threshold was encountered.

## Safety/scope confirmations

- No physical GPU0 allocation, process termination, reset, reconfiguration or
  unrelated workload mutation occurred.
- No persistent service, LAN/public exposure, systemd activation, firewall/VPN
  change, system CUDA/driver change, or global provider/configuration change
  occurred.
- Model weights stayed in the operator cache and were not committed or
  redistributed. No raw image/YAML, customer data, credentials, tokens or
  private cache paths were entered in the report.
- Client YAML cannot select model IDs/revisions, devices, filesystem paths,
  URLs, commands, cache roots, credentials or resource strategy. Strict mode
  does not spill to GPU0 or silently fall back to CPU.

## Limitations/blockers

- BLIP3 and profiles requiring it remain unsupported on this measured GPU; the
  result is a deliberate configuration rejection, not a claim of BLIP3 service
  readiness. A future resource/model change requires a new governed order.
- Repeated runs are bounded stability checks, not a production soak or accuracy
  evaluation. The generated fixture and labels establish shape/resource
  behavior only.
- Detectron2 remains an optional dependency for the legacy panoptic renderer;
  it was not installed in the qualified runtime.
- Commercial use, redistribution and deployed use of the pinned research model
  set remain behind the applicable human release gate.
- Objective 004 still owns live loopback service activation, readiness wiring
  with the loaded model registry, E2E requests, restart and rollback.

## Factual strategic follow-up

- Use the documented strategy and freshly re-verify GPU1 UUID/process/free
  memory before Objective 004 activation.
- Keep BLIP3 profiles rejected unless a later measured order establishes a safe
  resource strategy without changing the approved scientific identity silently.
- Resolve the carried strategic order's original trailing-space byte only if
  strategic governance explicitly permits changing the immutable transcript;
  it was intentionally not altered in this round.
