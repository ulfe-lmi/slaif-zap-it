# OAP Coding-Agent Report — 016-a

## Work order

- Identifier/order/objective/PR mode: `016-a` — bounded raw SAM2
  visualizations; create one new Objective-016 PR.

## Status

COMPLETE

## Executive summary

Implemented the bounded, deterministic L3 raw-SAM2 visualization contract.
Service-safe `mask_generator.debug: true` now renders source-indexed candidate
contact sheets and all-candidate union, overlap and uncovered diagnostics in
fixed PNG names. The typed JSON/ZIP manifest reports exact source-resolution
coverage and bounded overlap facts, including candidates omitted from the
96-candidate display ceiling. Capabilities, schemas, admission checks,
documentation and CPU/API coverage were extended. Legacy CLI rectangular JPEG
debug patches, lower verbosity behavior, SAM2 generation/filtering,
model identity/residency, CLIP/BLIP3 behavior, identity PNG and response-limit
defaults remain unchanged.

The implementation passed the required CPU/static checks, current-head CI and
CodeQL, and the authorized repeated live qualification on the exact assigned
GPU. No CRITICAL register action was required.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/72`
- PR state: OPEN, non-draft, MERGEABLE; coding did not merge or enable
  auto-merge.
- Base: `main` at
  `8081152403657f5e737ab0b491e0b89f587209e1`.
- Branch: `oap/016-a-bounded-raw-sam2-visualizations`.
- Starting SHA: `8081152403657f5e737ab0b491e0b89f587209e1`.
- Implementation head SHA:
  `7f393710d53966941acc3adf7bde2f194180fb7e`.
- Report publication commit: SELF.
- New PR: yes. Amended existing PR: no. Coding merge/auto-merge: NO.

## Changes/files

Implementation commit
`7f393710d53966941acc3adf7bde2f194180fb7e` contains the exact active/order
transcript and these bounded implementation, test and documentation changes:

- `oap/active`
- `oap/orders/016-a-bounded-raw-sam2-visualizations.md`
- `src/core/raw_visualizations.py` and `src/core/__init__.py`
- `src/core/engine.py`
- `src/service/app.py`, `capabilities.py`, `envelope.py`, `fake_engine.py`,
  `resources.py`, and `schemas.py`
- `tests/test_raw_sam2_visualizations.py` and the capability contract updates
  in `tests/test_sam2_configuration.py`
- `README.md`, `TESTING.md`, and the applicable current API/config/core/
  algorithm/output-parity/runbook/datasheet/runtime documentation.

The report publication commit changes only `oap/reports/016-a-report.md`.
No prior OAP report or CRITICAL entry was edited.

## Acceptance evidence

1. **Pure deterministic renderer — PASSED.** A model-free NumPy/Pillow
   component accepts original RGB pixels and source-indexed non-empty masks,
   validates shape/source-index agreement, preserves disconnected and border
   masks, uses the fixed arithmetic candidate palette and does not mutate
   inputs. CPU tests repeat arrays and PNG hashes in the pinned environment.
2. **Separate bounded candidate display — PASSED.** Pages are fixed
   `960x1072` RGB sheets with three columns, four rows, 320x240 content,
   28-pixel label bars, exact source-order one-based IDs, fixed three-decimal
   score labels/`n/a`, clamped `ceil(10%)` context with a four-pixel minimum,
   bilinear RGB/mask-nearest resizing and 45% exact-mask overlay. Pagination
   covers 0/1/12/13/96/>96 candidates, emits no ninth page and records explicit
   truncation with one fixed aggregate warning.
3. **All-candidate coverage/overlap truth — PASSED.** The uint32 source
   overlap canvas includes candidates beyond the display ceiling. Union,
   overlap heatmap and uncovered images have fixed semantics; exact covered,
   uncovered, maximum-overlap and bounded 0..255-plus-overflow histogram facts
   reconcile to the source area. CPU tests cover overlaps, disconnected masks,
   deep overlap and inverse images.
4. **Typed JSON/ZIP contract — PASSED.** The optional
   `service.sam2.raw_visualization` child is emitted only at L3 with debug
   enabled, and its counts, IDs, page count, dimensions and artifact names are
   cross-consistent. JSON descriptors and ZIP members carry matching PNG media
   types, sizes and SHA-256 values. Repeated live requests produced identical
   raw metadata and identical raw PNG bytes.
5. **Static capabilities — PASSED.** Authenticated capabilities now describe
   the trigger, fixed names, ID base, layout, crop/score semantics, palette,
   diagnostic limits and truncation policy. The route remained static and did
   not consult readiness or acquire the inference gate.
6. **Admission and error bounds — PASSED.** Before readiness/gate/engine, the
   service reserves fixed diagnostic capacity using the exact formula
   `8 * 960 * 1072 * 3 + 3 * diagnostic_width * diagnostic_height * 3`,
   bounded at a maximum of 42,698,880 RGB bytes, plus the configured streams
   and response artifact slots. Count, per-artifact, total raw and response
   budget insufficiency returns sanitized `response_too_large`; the isolated
   CPU test returned HTTP 413 with zero engine calls. Existing sink, encoding,
   JSON, ZIP and response limits remain authoritative.
7. **Routing/parity — PASSED.** The service-safe path uses fixed PNG
   diagnostics; below L3 debug is stripped; trusted legacy CLI debug continues
   to use its historical rectangular JPEG names and format. No client text,
   frame name, prompt, label, path or destination enters a raw diagnostic name.
8. **Live qualification — PASSED.** After one authorized restart of only
   `zap-it-lan.service`, health/readiness, authentication, capabilities,
   metrics and docs policy passed. Two repeated in-memory bounded L3 ZIP
   requests returned HTTP 200 with 28 raw candidates, 28 represented
   candidates, three pages, six raw artifacts, 47,491 covered pixels, 1,661
   uncovered pixels and maximum overlap 5. Both PNG-byte and raw-metadata
   repeatability checks passed.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 706 passed, 1 explicit GPU skip, 79.92% total coverage; the skip
  is the opt-in GPU integration test without live-test enablement.
- `.venv/bin/pytest -q tests/test_raw_sam2_visualizations.py`: PASSED — 23
  focused renderer/API/resource tests.
- `.venv/bin/ruff format --check src tests`: PASSED — 85 files formatted.
- `.venv/bin/ruff check src tests`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED — wheel and sdist built;
  the new core module was included.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED — wheel/sdist release audit.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz
  --baseline .secrets.baseline`: PASSED — zero unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: PASSED — seven reviewed baseline findings,
  unchanged.
- `.venv/bin/python -m twine check dist/*`: PASSED — wheel and sdist.
- Required CPU, schema, hostile-YAML, lifecycle, timeout/cancel, artifact,
  auth, metrics, model-residency, BLIP3, post-filter, legacy CLI and package
  regressions: PASSED through the canonical suite.

Two live harness setup attempts were FAILED before service execution: one
heredoc syntax error and one unavailable `httpx` import in the GPU environment.
They made no request and changed no host state. A content-free diagnostic run
then passed, followed by the final qualification below; no failed inference
request was accepted as evidence.

## CI/checks

All required checks passed on implementation head
`7f393710d53966941acc3adf7bde2f194180fb7e`:

- `static (format, lint, build)`: PASSED — GitHub run `33222719125`.
- `tests (py3.10)`: PASSED — GitHub run `33222719125`.
- `tests (py3.11)`: PASSED — GitHub run `33222719125`.
- `tests (py3.12)`: PASSED — GitHub run `33222719125`.
- `release (artifact audit)`: PASSED — GitHub run `33222719125`.
- `Analyze (python)`: PASSED — GitHub run `33222719144`.
- `CodeQL`: PASSED — GitHub run `99020130139`.

## GPU/service/resource evidence

- Assigned physical target only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24,576 MiB, driver
  `610.43.02`; Torch qualification reported `2.5.1+cu124`, CUDA build
  `12.4`. The application saw one logical device, `cuda:0`.
- The one authorized restart changed the service PID from 449821 to 465389;
  the final unit was enabled, active/running, ready, and had `NRestarts=0`.
  Exactly one listener remained at `10.8.132.76:17891`, owned by PID 465389.
- The service metrics reported exactly one successful
  `component="registry", outcome="success"` initialization and zero
  residency transitions. The process/listener/registry remained continuous
  across both qualification requests.
- Final assigned-GPU snapshot: 10,595 MiB used, 13,529 MiB free; the sole
  compute process was PID 465389 using 10,572 MiB. Sampling during the two
  requests observed peak used 10,595 MiB and minimum free 13,529 MiB. Final
  service RSS was 3,993,548 KiB.
- `/dev/shm` remained a 12-GiB tmpfs with the configured workspace mode 0700
  and no request-workspace entries after requests.
- The mode-0600 operator environment digest remained
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  No credential value was printed, logged, committed or included here.
- Health/readiness were HTTP 200; missing/wrong inference credentials were
  HTTP 401; authenticated capabilities/metrics were HTTP 200; `/docs` and
  `/openapi.json` were HTTP 404. Capabilities did not disclose GPU topology.
- The live request used one already-authorized ignored local fixture, cropped
  and resized in memory; no input bytes, YAML, prompts, labels, answers or
  response body were persisted or printed. The journal scan found no bearer
  key or fixture filename.
- The isolated CPU/FastAPI resource request returned HTTP 413
  `response_too_large` before engine invocation, with zero fake-engine calls.

## Documentation/provenance

Updated the root README, API, configuration, core, algorithms, output-parity,
testing, runtime, runbook and service datasheet documentation with the exact
L3 trigger, overlap rationale, source-order IDs, page/crop/score semantics,
fixed names, colors, downscale and histogram facts, deterministic scope,
preflight formula and unchanged lower/legacy behavior. Model identities,
revisions, licenses, network/auth/key policy, GPU assignment, response default
limits, cache/offline mode, dtype, residency and accepted CRIT-0001 disposition
were not changed.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Exactly active order `016-a` was executed; no adjacent order was selected.
- Exactly one new PR (#72) was created for numeric Objective 016; no merge,
  auto-merge, release, tag, upload, key rotation, firewall/VPN/network change,
  unrelated service change or unassigned-GPU operation occurred.
- Only the exact order-assigned physical GPU 0/UUID was used for live work;
  unassigned GPUs and unrelated workloads remained protected.
- The active selector and exact order transcript were committed in the
  implementation head before this report. The final report-only commit will
  change only this report and will have the implementation head as its first
  parent.
- No request data was written to repository output directories, persistent
  disk or OAP evidence.

## Limitations/blockers

None for the ordered scope. Deterministic PNG byte identity is claimed only
for equal inputs in the same pinned Pillow/NumPy environment, not across
arbitrary encoder/library versions. The live counts are bounded qualification
evidence, not a segmentation-quality, solar-array recall/precision or
production-readiness claim.

## Factual strategic follow-up

PR #72 is open and ready for strategic review/acceptance. Coding has not
merged it, enabled auto-merge, selected a next order or authorized deployment.
