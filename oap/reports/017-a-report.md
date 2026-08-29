# OAP Coding-Agent Report — 017-a

## Work order

- Identifier/order/objective/PR mode: `017-a` — mask-isolated candidate views
  for CLIP and BLIP3 / `CREATE_NEW_PR`.

## Status

PARTIAL

## Executive summary

Implemented the shared pure CPU mask-view builder, strict request-local
`candidate_views` configuration, CLIP and BLIP3 mask-isolated model inputs,
lossless exact-input debug artifacts, stable candidate/question identity,
typed L3 provenance, capabilities, resource admission, tests and documentation.

All local CPU/static gates and current-head CI/CodeQL pass. The bounded live
qualification is partial: the one authorized service restart loaded
implementation commit `46e05cb...`; BLIP3 exact-input and A/B/A evidence passed,
but request-level CLIP debug was not active in that resident CLIP holder. The
request-local CLIP debug correction is implementation commit `d0ad70e...` and
is pushed and CI-green, but the order authorizes only one restart, so the live
service was not restarted again. It remains enabled, active and ready on the
pre-correction live process.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/73`
- PR state: OPEN, non-draft; coding did not merge or enable auto-merge.
- Base branch/SHA: `main` /
  `645c8604f9c189e1367e6e27a4ce8298c109482a`.
- Branch: `oap/017-a-mask-isolated-candidate-views`.
- Starting SHA: `645c8604f9c189e1367e6e27a4ce8298c109482a`.
- Implementation commits: `46e05cb1ac6c8b68919f747157daf98f63039615`,
  followed by `d0ad70e7b978b7c314db596245d061cb42e6c390`.
- Implementation head SHA: `d0ad70e7b978b7c314db596245d061cb42e6c390`
- Report publication commit: SELF
- New PR: yes; amended existing PR: no; coding merge/auto-merge: NO.

## Changes/files

Implementation/control state contains the exact activated `oap/active` and
immutable `oap/orders/017-a-mask-isolated-candidate-views.md` transcript,
`src/core/mask_views.py`, normalized core/config/result/sink/engine exports and
identity propagation, CLIP and BLIP3 adapter integration, service validation,
schemas, envelope, capabilities and fake-engine compatibility, plus generated
array tests.

Documentation was refreshed in `README.md`, `ARCHITECTURE.md`, `TESTING.md`,
`CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/API.md`, `docs/CONFIG.md`,
`docs/ALGORITHMS.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`
and `docs/SERVICE-DATASHEET.md`. No `CRITICAL.md`, model pin, credential,
dependency or protected-host configuration changed.

The implementation head changes 32 paths across the two implementation commits:
the files named above, `modules/classifier/clip.py`,
`modules/verifier/{__init__.py,blip3.py}`, `src/{__init__.py,batch.py}`,
`src/core/{__init__.py,config.py,engine.py,mask_views.py,results.py,sinks.py}`,
`src/service/{app.py,capabilities.py,envelope.py,fake_engine.py,schemas.py,yaml_input.py}`,
and `tests/test_mask_views.py`, together with the OAP selector/order.

## Acceptance evidence

1. **Shared builder and pixel boundary — PASSED locally.** Generated-array
   tests cover nonrectangular masks, bbox holes, disconnected components,
   borders/corners, exact Euclidean dilation, zero-fill invariants, source
   immutability, deterministic floor-rounded context intensity, contour
   placement and bounded dimensions. The builder owns immutable RGB/mask arrays
   and never reads models, files, network, environment or global request state.

2. **CLIP isolation — PASSED in CPU seam tests; LIVE INCOMPLETE.** CPU tests
   capture the builder-derived context view and fixed PNG, including one-based
   source naming. The resident live process was initialized before the
   request-local debug correction, so its L3 request emitted no CLIP debug
   artifact; `d0ad70e...` now passes request `debug` explicitly without
   mutating resident model state.

3. **BLIP3 isolation — PASSED in CPU and bounded live evidence.** The actual
   QA path builds one shared result per candidate, places target-only pixels on
   the left and bounded zero-filled/dimmed dilated context on the right, uses
   bilinear RGB plus nearest-neighbor support reapplication, and passes the
   exact paired image retained by the PNG debug artifact. Live reconstruction
   from returned object RLE and manifest metadata was byte-identical for the
   emitted BLIP3 inputs.

4. **Configuration and request locality — PASSED locally and PARTIAL live.**
   Strict typed defaults, ranges, bool/null/nonfinite/type rejection,
   `min <= max`, stage-specific contour policy, zero-only fill, unsupported
   mode/fill, and service `clip.padding` rejection are covered. Effective
   values and applied status appear at every response level. The corrected
   CPU path is request-local. Live BLIP3 A/B/A requests changed effective
   radii/input hashes and returned stable A/B/A values; CLIP input A/B/A was not
   live-observed because of the pre-correction holder behavior.

5. **Identity and names — PASSED locally; BLIP3 PASSED live.** `_source_index`
   remains zero-based, `source_candidate_id` is one-based, and
   `_filtered_index` is assigned immediately after the post-SAM2 filter and
   survives final ordering/object serialization. Fixed names use only numeric
   source/question IDs: `clip-candidate-view-CANDIDATE-####.png` and
   `blip3-verification-CANDIDATE-####-QUESTION-####.png`.

6. **Manifest/capabilities/artifact parity — PASSED locally; BLIP3 PASSED
   live.** Pydantic schemas, JSON/ZIP manifests, exact PNG media/hash/size
   descriptors, effective candidate-view capabilities, no-path/no-secret/no-GPU
   topology disclosure, and one-for-one L3 input records are covered. The live
   L3 ZIP had HTTP 200, two final candidates, three artifacts including two
   candidate-view PNGs, two matching input records, exact hash/size checks and
   fixed safe names. Those two candidate-view artifacts were BLIP3 artifacts;
   CLIP artifact coverage is the live blocker above.

7. **Resource and compatibility behavior — PASSED locally.** Candidate-view
   debug count/per-item/total raw-byte admission occurs before CLIP/BLIP3 model
   work; dynamic sink and encoded JSON/ZIP/deadline limits remain active. Full
   CPU regressions preserve SAM2, post-filter, raw visualization, labelled
   rendering, response, auth, lifecycle, legacy and package behavior.

8. **Ordered final qualification — INCOMPLETE.** The required newest-service
   condition cannot be claimed because the corrected implementation could not
   be loaded without a second restart, which is outside the one-restart host
   authorization in this round.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 742 passed, 1 explicit GPU test skipped, 80.30% total coverage
  against a 64% threshold. The skip is `tests/test_gpu_integration.py:20`
  because its opt-in live-GPU marker was not enabled by this order.
- `.venv/bin/pytest -q tests/test_mask_views.py`: PASSED — 16 tests.
- `.venv/bin/pytest -q tests/test_classifier_clip.py tests/test_verifier_blip3.py
  tests/test_core_engine.py`: PASSED — 47 tests.
- `.venv/bin/pytest -q` before the final adapter correction: PASSED — 740
  passed, 1 skipped; after the correction, the canonical suite above passed.
- `.venv/bin/ruff format --check .`: PASSED — 148 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh scripts/serve_local.py
  scripts/smoke_local_service.py`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED before both implementation pushes.
- `.venv/bin/python -m build --wheel --sdist`: PASSED; existing setuptools
  license-metadata deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED — wheel/sdist allowlists and manifests verified.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz
  --baseline .secrets.baseline`: PASSED — zero unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: PASSED — exactly seven unchanged reviewed
  baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED — wheel and sdist.
- A supported GitHub query was used after one exploratory `gh pr view` command
  was rejected because this installed `gh` does not support requested field
  `baseRefOid`; the failed query caused no mutation.
- One A/B/A summary harness initially failed with an internal `NameError` after
  a completed HTTP request; it was explicitly disclosed and corrected. The
  corrected bounded sequence had three HTTP 200 responses, no service failure
  or restart, and stable A/B/A BLIP3 effective manifests/pixels.

## CI/checks

All required checks passed on implementation head
`d0ad70e7b978b7c314db596245d061cb42e6c390`:

- `static (format, lint, build)` — PASSED — CI run
  `33279364192`, job `99171768573`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364192/job/99171768573
- `tests (py3.10)` — PASSED — CI run `33279364192`, job `99171768616`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364192/job/99171768616
- `tests (py3.11)` — PASSED — CI run `33279364192`, job `99171768658`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364192/job/99171768658
- `tests (py3.12)` — PASSED — CI run `33279364192`, job `99171768671`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364192/job/99171768671
- `release (artifact audit)` — PASSED — CI run `33279364192`, job
  `99171768495`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364192/job/99171768495
- `Analyze (python)` — PASSED — CodeQL workflow run `33279364197`, job
  `99171768476`:
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33279364197/job/99171768476
- `CodeQL` — PASSED — check run `99171872444`:
  https://github.com/ulfe-lmi/slaif-zap-it/runs/99171872444

The report-only child is required to be checked separately after publication;
the live qualification remains PARTIAL regardless of those check results.

## GPU/service/resource evidence

- The active order assigned only physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The process exposed this
  card as logical `cuda:0` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`. The live host reported no other GPU/device to
  touch; no unassigned resource was mutated.
- Before the authorized restart: PID `498617`, `NRestarts=0`, active/running,
  one listener at `10.8.132.76:17891`, health/readiness 200, and one assigned
  GPU compute process using approximately 13,436 MiB. The environment digest
  was unchanged at
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- The one authorized restart occurred once. After load: PID `513853`,
  `NRestarts=0`, active/running, health/readiness 200, one listener, one
  assigned-GPU compute process, final GPU use approximately 10,821 MiB
  (process query approximately 10,798 MiB), and current RSS `4,433,552 KiB`.
  Startup readiness was delayed while the resident registry loaded; it then
  returned 200. No OOM, failed HTTP request, second process or second restart
  occurred. The startup registry was initialized once in the sole service
  process; no separate initialization counter is exposed by the service.
- Authenticated live policy checks passed before and after the request set:
  health/readiness 200, missing/wrong key 401, capabilities/metrics 200 and
  docs/OpenAPI 404. The final authenticated capabilities response included the
  candidate-view policy and disclosed no credential, operator path or GPU
  topology.
- The first bounded live L3 ZIP was HTTP 200, 8,984 bytes, two final
  candidates, three artifacts, two candidate-view artifacts and two exact
  records. BLIP3 reconstruction was byte-identical; CLIP debug was absent in
  that pre-correction process. Corrected A/B/A BLIP3 ZIPs were 3,988 / 7,893 /
  3,988 bytes with request latencies approximately 591.4 / 505.5 / 499.2 ms;
  all were HTTP 200, had two candidates and two candidate-view records, and
  changed then restored effective radius/input values without state leakage.
- `/dev/shm/slaif-zap-it` remained mode `0700` and empty after every checked
  request; its observed free capacity was approximately 11,452 MiB. Request
  image/config/results stayed in memory and no raw content was written to Git,
  OAP, logs or the report.
- Final post-qualification checks remained health/readiness 200, authenticated
  capabilities/metrics 200, docs/OpenAPI 404, one listener, PID `513853`,
  `NRestarts=0`, unchanged environment digest and empty request workspace.

## Documentation/provenance

The current documentation was searched and refreshed for stale untouched-context,
rectangular semantic-crop, old candidate/question filename/index and omitted
candidate-view capability claims. The documentation checker passed all 27
current documents. Exact formulas/defaults/limits, zero-only fill, contour and
resize order, public ID bases, fixed debug names, response levels, debug
triggers, resource limits, `clip.padding` migration, deterministic scope and
pixel-isolation versus semantic-accuracy limits are documented.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Exactly active order `017-a` was executed; no adjacent order was selected.
- Exactly one new PR, #73, was created for numeric Objective 017. No merge,
  auto-merge, release/tag/publish, history rewrite, firewall/network/VPN,
  driver/CUDA, unrelated unit, unrelated process or unassigned GPU mutation
  occurred.
- The running private-LAN service was restarted exactly once as authorized and
  remains enabled, active and ready. The corrected implementation was not
  loaded into that process because a second restart was not authorized.
- Credentials, model identities/revisions, caches, weights, `CRITICAL.md`,
  unassigned resources and unrelated state were not changed or disclosed.
- All implementation state was pushed before this report. The final report
  publication commit is intended to change only this report and to have the
  implementation head as its sole parent.

## Limitations/blockers

The live acceptance gate is incomplete solely because request-level CLIP debug
was fixed after the only authorized restart. Consequently no COMPLETE or
newest-code live-readiness claim is made. A future governed round may reload
the pushed implementation and repeat the bounded CLIP/BLIP3 L3 and A/B/A
qualification; this report does not authorize that action.

## Factual strategic follow-up

PR #73 is open with the corrected implementation and its required checks green.
Strategic review must account for the explicit PARTIAL status and decide the
next governed action. Coding has not merged, accepted, released, enabled
auto-merge, selected another order or performed any post-report mutation.
