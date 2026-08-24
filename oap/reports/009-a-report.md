# OAP Coding-Agent Report — 009-a

## Work order

- Identifier/order/objective/PR mode: `009-a` / Objective 009 / `CREATE_NEW_PR`
- Objective: close the memory-deferred all-resident profile matrix and current-document truth.

## Status

COMPLETE

## Executive summary

The exact authenticated eight-call all-resident matrix passed on the assigned
RTX 3090. All four supported profiles were exercised twice in the required
interleaved order, with the expected stage semantics, eight bounded BLIP3
answers for each BLIP3 profile, repeatable content-free semantic digests, zero
residency transitions, one model-registry initialization, and no request
residue. Peak Torch reserved memory was 11,912.0 MiB, strictly below the
22,118.4-MiB 90% ceiling of the 24,576-MiB physical card.

Current root and `docs/*.md` documentation now consistently distinguishes the
live-qualified sequential lifecycle below 24,576 MiB from the live-qualified
all-resident lifecycle at or above 24,576 MiB. The narrow documentation guard
rejects the four audited obsolete claims. Historical OAP reports and
`docs/history/` were not changed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#65](https://github.com/ulfe-lmi/slaif-zap-it/pull/65)
- PR state: `OPEN`; required title: `Objective 009: close memory-deferred profile evidence`
- Base: `main` at starting SHA `b1d8c5dbc9392002ab52b3b0b744582a073ebf75`
- Required branch: `oap/009-a-memory-deferred-profile-matrix-and-doc-closure`
- Starting SHA: `b1d8c5dbc9392002ab52b3b0b744582a073ebf75`
- Implementation head SHA: `c54eb75d9741f67081cb93e50102194662cb5667`
- Report publication commit: SELF
- New PR: yes; amended existing PR: no; coding merge: NO

## Changes/files

- Added `scripts/profile_matrix.py`, a content-free authenticated operator
  harness using generated 128x128 RGB data and in-memory API-safe YAML.
- Added `tests/test_profile_matrix.py` with all four profile shapes, repeatable
  digest, resource, memory, model/device, stage, transition and residue failure
  cases.
- Extended `scripts/check_documentation.py` and
  `tests/test_documentation.py` with the four narrow obsolete-claim guards.
- Reconciled current `README.md`, `ARCHITECTURE.md`, `INSTALL.md`,
  `CHANGELOG.md`, `RELEASE_NOTES.md`, `THIRD_PARTY_NOTICES.md`, `SECURITY.md`,
  `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/README.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RELEASE-GATE-INVENTORY.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md`, and `docs/runtime.md`.
- Added the matrix harness to the source-distribution allowlist.
- Committed the activated `oap/active` marker and exact 009-a order transcript.
- No prior order/report or historical document was modified.

## Acceptance evidence

1. One Objective-009 branch and PR were created from the verified current
   `main`; no adjacent PR or order was created.
2. Focused CPU/fake tests passed: `15 passed`. The validator fails closed on
   wrong status/stage, missing BLIP3 answer, wrong strategy/device/model count,
   nonzero transition, memory-ceiling breach, malformed response, monotonic
   resource growth, and shared-memory residue.
3. The exact required sequence passed:

   ```text
   sam2, sam2_clip, sam2_blip3, sam2_clip_blip3,
   sam2_clip_blip3, sam2_blip3, sam2_clip, sam2
   ```

   The first harness invocation was `FAILED` as a validator-shape check
   because the service also reports the `ordering` stage. It produced no
   acceptance evidence; the validator was corrected in scope and the exact
   sequence was rerun successfully.

   | Call | Profile | HTTP | Latency ms | Objects | Answers | Peak reserved MiB | Free MiB | RSS MiB | Transitions | Digest prefix |
   |---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
   | 1 | `sam2` | 200 | 201.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | 0 | `4b8febc645a4` |
   | 2 | `sam2_clip` | 200 | 807.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | 0 | `a8a6e0b5e943` |
   | 3 | `sam2_blip3` | 200 | 2299.4 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | 0 | `fcbf3d1e4761` |
   | 4 | `sam2_clip_blip3` | 200 | 1652.6 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | 0 | `c1ed971ded2b` |
   | 5 | `sam2_clip_blip3` | 200 | 1652.1 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | 0 | `c1ed971ded2b` |
   | 6 | `sam2_blip3` | 200 | 1624.8 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | 0 | `fcbf3d1e4761` |
   | 7 | `sam2_clip` | 200 | 177.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | 0 | `a8a6e0b5e943` |
   | 8 | `sam2` | 200 | 192.4 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | 0 | `4b8febc645a4` |

   Per-profile timing statistics were first/minimum/maximum/median in ms:

   - `sam2`: `201.2 / 192.4 / 201.2 / 196.8`; objects `8 / 8 / 8 / 8`;
     answers `0 / 0 / 0 / 0`; digest
     `4b8febc645a4b0a838e1dcb50d4fb07f9140d9bffd2c96b22239a9dcf85f8e5a`.
   - `sam2_clip`: `807.2 / 177.2 / 807.2 / 492.2`; objects `8 / 8 / 8 / 8`;
     answers `0 / 0 / 0 / 0`; digest
     `a8a6e0b5e9433b1319cf9fbb3e8c19e738ccf4f515fea0a74ed215020a819671`.
   - `sam2_blip3`: `2299.4 / 1624.8 / 2299.4 / 1962.1`; objects `8 / 8 / 8 / 8`;
     answers `8 / 8 / 8 / 8`; digest
     `fcbf3d1e4761a2ef5f58818820e87cc106f90609715f6511e78743b50214e302`.
   - `sam2_clip_blip3`: `1652.6 / 1652.1 / 1652.6 / 1652.3`; objects
     `8 / 8 / 8 / 8`; answers `8 / 8 / 8 / 8`; digest
     `c1ed971ded2bdb581c4c646bfc9d29e43fc5e0b2d6929c565e708e7c41849be8`.

4. Assigned-host cleanup passed: physical index 0, UUID
   `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
   GeForce RTX 3090, 24,576 MiB; driver `610.43.02`, Torch `2.5.1+cu124`,
   CUDA runtime `12.4`. The masked process used logical `cuda:0`. The fresh
   15-MiB/no-compute-process baseline was restored; port `127.0.0.1:17891` was
   free; no ZAP-IT process remained; and `/dev/shm/slaif-zap-it` was empty.
5. The current-document scan covered all root and current `docs/*.md` files;
   the minimum audited set is included in the changed-file list. No obsolete
   memory-deferral pattern remains, while detailed 007-b/008 evidence remains
   intact.
6. The completion classification explicitly keeps geometry/panoptic,
   licensing, tracked-media, gateway/deployment, and final-release gates
   separate from closed GPU-memory work.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 402 passed, 1 intentional module-level GPU skip; total coverage
  77.90%.
- `.venv/bin/pytest -q tests/test_documentation.py tests/test_profile_matrix.py`:
  `PASSED` — 15 passed.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 20 current
  documents.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  `PASSED` — wheel and sdist allowlist/member checks.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  `PASSED` — archive scans clean.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  `PASSED` — six reviewed baseline findings matched exactly.
- `git diff --check`: `PASSED`.
- First live harness invocation: `FAILED` — fail-closed stage-shape validator
  caught the unlisted `ordering` stage; no result was accepted.
- Corrected live matrix command:
  `.venv-gpu/bin/python scripts/profile_matrix.py --port 17891`: `PASSED` —
  exact eight-call matrix and resource/cleanup invariants.

## CI/checks

All seven required checks passed on implementation head
`c54eb75d9741f67081cb93e50102194662cb5667`:

- `Analyze (python)`: `PASSED`
- `CodeQL`: `PASSED`
- `release (artifact audit)`: `PASSED`
- `static (format, lint, build)`: `PASSED`
- `tests (py3.10)`: `PASSED`
- `tests (py3.11)`: `PASSED`
- `tests (py3.12)`: `PASSED`

The report-only child will receive a fresh equivalent check set; it must be
verified before signaling.

## GPU/service/resource evidence

- Authorized physical target: index 0 / UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` / PCI `00000000:0B:00.0` /
  GeForce RTX 3090 / 24,576 MiB.
- Launch mask: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`,
  operator physical index `0`; application logical device `cuda:0`.
- One loopback service process, one Uvicorn worker, one inference request at a
  time, authenticated L3 JSON calls only.
- Torch current/peak allocated: `9635.6 / 11188.8 MiB`; current/peak reserved:
  `11912.0 / 11912.0 MiB`; free: `11864.8 MiB`; host maximum RSS:
  `12752.4 MiB`; transitions: `0` on every call.
- Physical 90% ceiling: `22118.4 MiB`; observed peak: `11912.0 MiB`
  (`53.85%`). No monotonic GPU/host growth, reload, CPU migration, fallback,
  or request persistence was observed.
- Port `127.0.0.1:17891` was freshly verified free before launch and free after
  stop. The shared-memory root was mode `0700`, had only the launcher's
  runtime entry while active, and was empty after stop. No unrelated process or
  device was changed.

## Documentation/provenance

Current docs state that `<24576 MiB` uses the live-qualified sequential
stage-boundary lifecycle on the historical 11-GB RTX 2080 Ti, while
`>=24576 MiB` uses the live-qualified all-resident lifecycle on the assigned
24,576-MiB RTX 3090. Both expose only logical `cuda:0` after an explicit
operator index/UUID pin. Objectives 007–009 close the GPU-memory deferrals and
provide all-four-profile evidence. The measurements are bounded local research
evidence, not an SLA, accuracy claim, commercial-license clearance, or external
deployment. Geometry/panoptic and deployment/release gates remain separate.

The current-document checker rejects the audited phrases equivalent to
all-resident qualification remaining separate, pending separate qualification,
low-card-only BLIP3 qualification, and fake-tested-but-not-live-qualified
all-resident behavior. Historical OAP reports and `docs/history/` remain
immutable. Pinned model identities and revisions were not changed.

## Deferred human adjudication

- Critical register action: NONE
- No new register entry was created or requested.

## Safety/scope confirmations

- No model, model revision, dtype, residency threshold, geometry stage, panoptic
  renderer, deployment topology, release setting, driver, CUDA installation,
  firewall, VPN, unrelated service, or unrelated process was changed.
- No request image, YAML, prompt, answer text, response body, model cache path,
  or bearer value entered Git, logs, or this report.
- The generated fixture was used in memory only. No private goat fixture was
  needed. Request data was not persisted.
- The activated order was `009-a`; no adjacent order was read or executed.
- No merge or auto-merge was performed.

## Limitations/blockers

This is bounded local research evidence on the assigned host, not a production
SLA, accuracy qualification, commercial-use decision, public deployment, or
release authorization. Public CI remains CPU-only and intentionally skips live
GPU integration without the explicit operator environment. Geometry/panoptic,
licensing, media, gateway, deployment, and final-release work remains outside
this order.

## Factual strategic follow-up

PR #65 is open at the verified report-head topology for strategic review and
acceptance. Coding has not merged it or selected a subsequent order.
