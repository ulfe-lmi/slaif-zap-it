# OAP Coding-Agent Report — 018-a

## Work order

- Identifier/order/objective/PR mode: `018-a`, close the exact mask-view
  acceptance matrix, create one new PR.
- Active selector: `018-a`.
- Immutable order SHA-256 in implementation commit:
  `746c943d1ab66c984d7ab4cf9f04e69cdd830575db1d494f5828d4c64fbf5326`.

## Status

COMPLETE

## Executive summary

Implemented the narrow Objective-018 test/provenance closure requested after
Objective-017 merged. Strategic's final audit found exact acceptance-evidence
gaps only after PR #73 merged; this round closes those gaps without changing
runtime behavior. Generated CPU/fake tests now state the 512x512 striped
rectangular-leakage boundary, independent dilation and shape cases, tiny-mask
resize/contour order, literal CLIP processor and BLIP3 QA inputs, source/filter
identity through final visualization and public JSON/ZIP objects, and the full
L0-L3 request-local A/B/A matrix.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- PR: [#74 — Objective 018: close mask-view acceptance matrix](https://github.com/ulfe-lmi/slaif-zap-it/pull/74).
- PR state: open, non-draft, base `main`; coding did not merge or accept it.
- Required branch: `oap/018-a-close-mask-view-acceptance-matrix`.
- Verified exact base SHA: `03def697373f2ae83d03494315aa96c800f0bcdf`.
- Starting local execution SHA before the exact-base branch:
  `a1d75f86ed49c1893f1700ff4f06ff90712e716e`.
Implementation head SHA: 74ab1d8dc685f2ea1ce8d9c0e38bb6fac9de5184
- Implementation parent: `03def697373f2ae83d03494315aa96c800f0bcdf`.
Report publication commit: SELF
- New PR: yes. Existing PR amended: no. Coding merge/acceptance: NO.
- The remote branch was verified at the implementation SHA before report
  publication.

## Changes/files

The implementation/control commit contains five paths: `TESTING.md`,
`oap/active`, the exact active order, `tests/test_mask_views.py`, and
`tests/test_candidate_view_api.py`. It adds no runtime, schema, dependency,
model, environment, service, GPU, credential, or CRITICAL-register change.
The order and selector are included unchanged in the implementation/control
commit.

## Acceptance evidence

1. **Exact 512x512 rectangular-leakage fixture — PASSED.** One generated RGB
   uint8 512x512 source contains a uniquely recognizable high-contrast striped
   distractor inside a nonrectangular tight bbox but outside `M` and `D`. The
   test directly asserts prohibited-pixel absence, zero fill outside `M`/`D`,
   byte identity under `M`, exact bbox/radius arithmetic, repeated arrays,
   metadata, and lossless PNG SHA-256 equality.

2. **Dilation, holes, components, borders, and contour contract — PASSED.**
   Generated tests assert exact Euclidean markers, ceil/fraction-zero and
   min/max radius values, ring-hole reachability, disconnected component
   preservation without a rectangular bridge, uniquely patterned edge/corner
   pixels, clipping, source-space tiny-mask crop, nearest-neighbor mask
   reapplication, bilinear RGB mapping, and zero/positive contour width.

3. **Literal CLIP/BLIP3 seams — PASSED.** The real
   `_ClipFilter.classify_single` path runs against bounded fake torch/model
   objects while a fake processor captures its literal `images=` array. The
   BLIP3 QA capture is compared with the shared safe paired image. Debug PNGs
   decode byte-for-byte to the captured model inputs, and the fake CLIP
   complete similarity vector selects the expected prompt.

4. **Source identity flow — PASSED.** An injected core/API flow removes source
   candidate 1 by the post-SAM2 bbox filter, retains source candidates 2 and 3,
   and reverses their retained order by area. One-based source IDs and
   zero-based filtered indices persist through CLIP, BLIP3, candidate-view
   records, final labelled visualization input, public JSON objects, and ZIP
   manifest objects. Fixed artifact names use source IDs and remain one-to-one
   with records.

5. **Response levels and request-local A/B/A — PASSED.** One injected service
   instance validates effective CLIP/BLIP3 policies and `applied` false/true at
   L0-L3. Generated A/B/A requests vary both context fractions; B changes both
   model inputs and effective metadata, A2 restores A exactly, candidate-view
   records/artifacts remain L3-only, JSON/ZIP hashes and sizes agree, and
   resident fake holder identity/initialization counts remain stable.

6. **No product change — PASSED.** The diff is limited to tests, current
   testing documentation, and required OAP control/order transcript. No
   Objective-017 runtime behavior, schema/default, dependency, service,
   environment, model, GPU, or credential behavior changed.

7. **CPU/static/package/CI gates — PASSED.** The full offline CPU suite,
   focused suite, package/static checks, release audits, and all seven current
   implementation-head CI/CodeQL checks passed. The one GPU integration marker
   remained honestly skipped because this order authorizes no live GPU work.

8. **Service preservation — PASSED.** The already-running private-LAN service
   remained enabled, active, ready, on the same PID/listener and one assigned
   GPU process. No restart, reload, reconfiguration, or inference request was
   performed.

## Verification

- `.venv/bin/pytest -q tests/test_mask_views.py tests/test_candidate_view_api.py`:
  **PASSED** — 55 passed, 1 warning.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  **PASSED** — 791 passed, 1 honest GPU test skipped, 81.52% total coverage,
  2 warnings.
- `.venv/bin/ruff format --check .`: **PASSED** — 149 files already formatted.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules scripts tests`: **PASSED**.
- `.venv/bin/python scripts/check_documentation.py`: **PASSED** — 27 current
  documents.
- `git diff --check`: **PASSED**.
- `.venv/bin/python -m build --wheel --sdist`: **PASSED** — wheel and sdist
  built; setuptools emitted existing deprecation warnings only.
- `scripts/verify_release_artifacts.py` on direct wheel and sdist: **PASSED** —
  67 wheel members and 160 sdist members.
- sdist-built wheel verification and
  `scripts/verify_release_artifacts.py --compare-wheels`: **PASSED** — equal
  67-member manifests and bytes.
- `scripts/scan_release_artifacts.py` on wheel/sdist/rebuilt wheel:
  **PASSED** — zero unexpected findings.
- Tracked-tree secret baseline scan: **PASSED** — exactly seven existing
  reviewed findings, with no addition/removal/path change.
- `.venv/bin/python -m twine check` on wheel, sdist, and rebuilt wheel:
  **PASSED**.
- `systemd-analyze verify deploy/zap-it-local.service`: **PASSED**.
- Outside-checkout installation of the sdist-built wheel and
  `scripts/smoke_installed_package.py`: **PASSED** — console script plus JSON
  and ZIP fake-service smoke reported package version `0.1.0` from site-packages.
- The final read-only service log scan found no credential, prompt, answer,
  traceback, OOM, or sensitive-path matches.

## CI/checks

All seven required checks were successful on implementation SHA
`74ab1d8dc685f2ea1ce8d9c0e38bb6fac9de5184`:

- `static (format, lint, build)`: **PASSED**, CI run `33283039183`, job
  `99181338372`.
- `release (artifact audit)`: **PASSED**, CI run `33283039183`, job
  `99181338466`.
- `tests (py3.10)`: **PASSED**, CI run `33283039183`, job `99181338439`.
- `tests (py3.11)`: **PASSED**, CI run `33283039183`, job `99181338519`.
- `tests (py3.12)`: **PASSED**, CI run `33283039183`, job `99181338490`.
- `Analyze (python)`: **PASSED**, CodeQL run `33283039179`, job
  `99181338285`.
- `CodeQL`: **PASSED**, check run `99181439613`.

The report-only child changes no product code; the same complete check set was
required to be inspected again on its final PR head before signaling.

## GPU/service/resource evidence

No live GPU phase was authorized or performed. Read-only final verification
found the exact assigned physical device unchanged: index `0`, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The existing service process
had `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, expected physical
index `0`, and used logical `cuda:0`; the assigned-device query showed exactly
one compute process, PID `528963`, with 11,178 MiB reported used. No unassigned
device or unrelated process was touched.

The user unit remained enabled/active/running with PID `528963` and
`NRestarts=0`. The sole listener remained `10.8.132.76:17891`. Read-only
endpoint checks returned `/healthz` 200, `/readyz` 200, unauthenticated
`/v1/capabilities` 401, and `/docs`/`/openapi.json` 404. The service request
workspace remained mode 0700 and empty. The mode-0600 operator environment
digest remained
`bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.

## Documentation/provenance

`TESTING.md` now records the exact 512x512 striped fixture, literal CLIP
processor seam, BLIP3 pair capture, source-identity flow, L0-L3 policy matrix,
and offline stable-holder A/B/A evidence. Historical Objective-017 orders and
reports were not rewritten. No semantic accuracy, recall, or precision claim
is made.

## Deferred human adjudication

- Critical register action: NONE.
- No CRITICAL entry was read, appended, edited, reordered, or closed.

## Safety/scope confirmations

- Only generated arrays and bounded CPU/fake holders were used by the new tests;
  no real model, model download, network, or live inference was used.
- No request image/config/result was persisted by the tests or service probes;
  no repository output path was used for request data.
- No service restart/reload/reconfiguration, GPU mutation, firewall/network
  change, credential change, dependency change, model/cache change, merge,
  auto-merge, tag, release, or publication occurred.
- The PR remains open for strategic review. Coding did not accept, merge, or
  advance the objective.

## Limitations/blockers

The evidence is deterministic boundary/provenance evidence and does not
measure semantic-model accuracy. The CPU GPU integration marker is skipped
honestly; this order grants no live GPU qualification. The final report child
must remain the only post-implementation file change.

## Factual strategic follow-up

PR #74 is open for strategic review at implementation head
`74ab1d8dc685f2ea1ce8d9c0e38bb6fac9de5184`; it was not merged or accepted by
this coding round. The final child of that implementation commit is this
immutable report-only SELF commit.
