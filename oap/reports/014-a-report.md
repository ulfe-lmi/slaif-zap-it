# OAP Coding-Agent Report — 014-a

## Work order

- Identifier/order/objective/PR mode: `014-a` / post-filter rejection diagnostics / one new branch and PR
- Branch: `oap/014-a-post-filter-rejection-diagnostics`
- Base: `main`

## Status

COMPLETE

## Executive summary

Implemented deterministic post-SAM2 rejection diagnostics without changing the
legacy filter result. Area, empty-mask, width and height decisions now share one
strict short-circuit evaluator with inclusive pixel bbox extents. The core
retains aggregate counts and bounded numeric-only source-indexed rejection
records; the service exposes the sidecar only at L3 with explicit Pydantic and
OpenAPI models. JSON and ZIP carry equal diagnostic values, and the existing
pipeline, object ordering, lower verbosity levels, artifact budgets and CLI
boundary remain intact.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#70](https://github.com/ulfe-lmi/slaif-zap-it/pull/70)
- PR state: OPEN
- PR title: `Objective 014: post-filter rejection diagnostics`
- Base/head: `main` / `oap/014-a-post-filter-rejection-diagnostics`
- Starting checkout SHA: `c916158c92ec7bfc98934788a86efd6865662bca`
- Verified remote base SHA: `2e8c67997c2480cf66f5c87a1e19afba4c6d368f`
- Implementation head SHA: `83247d3c09f8058f068b5bcdcf121ccd6698b33d`
- Report publication commit: SELF
- New PR: yes; amended existing: no; coding merge: NO

## Changes/files

- Added one shared post-filter evaluator and `MAX_POST_FILTER_REJECTION_RECORDS = 256`.
- Added deterministic counts, precedence, inclusive bbox dimensions, source-index
  fallback and numeric-only rejection records to the filter sidecar.
- Added backward-compatible `PipelineResult.post_filter_diagnostics` and
  canonical engine population before CLIP/BLIP3/label filtering.
- Added L3-only response serialization and explicit
  `PostFilterLimits`, `PostFilterRejection` and `PostFilterDiagnostics` models.
- Updated the CPU fake engine, focused core/filter/API tests and all order-named
  documentation. The exact `oap/active` and 014 order bytes are included unchanged.

## Acceptance evidence

- Shared evaluator and precedence: `PASSED` — every candidate is classified
  exactly once as `maxsize`, `empty_mask`, `max_w`, `max_h` or retained; multiple
  violations use the required first reason and equality is retained.
- Aggregate and legacy behavior: `PASSED` — evaluated/retained/removal
  invariants, list return, object identity/order, positional/keyword calls and
  bounded content-free logging are covered.
- Numeric rejection records: `PASSED` — records contain exactly source index,
  closed reason, area and inclusive bbox dimensions; input order and the fixed
  256-record cap/truncation are tested without candidate text, labels, pixels,
  crops, coordinates or arbitrary fields.
- Roof regression: `PASSED` — two deterministic wide candidates satisfying area
  and height limits are both counted as `max_w` removals with measured width;
  this is programmatic CPU filter evidence, not a real-image or SAM2 accuracy
  benchmark.
- Core isolation: `PASSED` — remapped source indices and exact masks are used;
  diagnostics are created before CLIP, BLIP3 and the later keep-label filter,
  with canonical candidate-count cross-checks.
- L3 wire contract: `PASSED` — exact diagnostic keys, closed reason schema,
  non-negative numeric bounds, L0-L2 omission, deterministic values and JSON/
  ZIP manifest parity are covered.
- Existing behavior: `PASSED` — full CPU suite and public CI remain green;
  no model identity/revision/license, dependency, renderer, threshold, schema
  version or artifact budget was changed.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 465 passed, 1 skipped; 79% total coverage, required 64% reached. The skip is the explicit opt-in GPU marker.
- `.venv/bin/pytest -q tests/test_postprocessing.py tests/test_core_engine.py tests/test_service_api.py`: PASSED — 76 passed.
- `.venv/bin/ruff format --check .`: PASSED — 142 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED — no shell files changed.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED — 64 wheel members and 153 sdist members audited.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — no unexpected built-artifact findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exactly 7 reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `git diff --check`: PASSED.

## CI/checks

All checks below passed at implementation SHA
`83247d3c09f8058f068b5bcdcf121ccd6698b33d`:

- CI workflow run `33201207119`: `static (format, lint, build)` PASS;
  `tests (py3.10)` PASS; `tests (py3.11)` PASS; `tests (py3.12)` PASS;
  `release (artifact audit)` PASS.
- CodeQL workflow run `33201207106`: `Analyze (python)` PASS.
- CodeQL check `98951046859`: PASS.

## GPU/service/resource evidence

- The single authorized restart was only `zap-it-lan.service`. The bounded
  readiness wait initially ended `FAILED` at 180 seconds while the checkpoint
  loaded; readiness returned `200 ready` shortly afterward without a second
  restart or other corrective mutation.
- Final host/service state: `hinton2`, enabled and active, `NRestarts=0`, one
  listener at `10.8.132.76:17891`, final MainPID `411354`. The post-restart PID
  and listener stayed unchanged across all requests.
- Exact assigned device: physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; the process saw only
  logical `cuda:0`. No unassigned device or unrelated process was touched;
  the only compute process was the service.
- Auth/docs: `PASSED` — missing and wrong inference keys returned 401;
  authenticated readiness and metrics returned 200; `/docs` and
  `/openapi.json` returned 404.
- Diagnostic live probe: `PASSED` — authenticated L3 JSON/ZIP and three
  repeated L3 JSON requests returned 200; 25 evaluated candidates all had
  `max_w: 0` rejection, zero retained/other reasons, 25 ordered records,
  reconciled counts, source-indexed positive widths and JSON/ZIP parity. One
  authenticated L2 request omitted the field. Response sizes were 21,191 bytes
  for JSON and 2,319 bytes for ZIP.
- Final resource evidence: peak logical Torch allocated/reserved bytes were
  `16790213632`/`19142803456`; logical free bytes were `5788925952`; maximum
  host RSS was `13387403264`. The final GPU process used 18,580 MiB. The
  private request workspace had zero files/bytes after the probe.
- The first live probe after readiness failed with a connection reset while
  posting a body for the wrong-key case. The service remained PID `411354`,
  ready and listening; a read-only no-body check then proved both missing and
  wrong keys return 401, after which the authenticated diagnostic probe passed.
- The mode-0600 operator environment remained unchanged by digest and content;
  no key value was read into evidence, printed, logged, committed or reported.
  The service remained enabled, active and ready.

## Documentation/provenance

Updated `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`,
and `docs/SERVICE-DATASHEET.md` with precedence, strict/inclusive threshold
rules, aggregate invariants, source-indexed numeric-only 256-record cap,
L3-only exposure, candidate-count relationships and the programmatic roof-test
interpretation. No model, hardware, residency, dependency or license claim
changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only active order `014-a` was executed.
- The exact active transcript and order were included unchanged in the
  implementation commit; prior orders/reports were not rewritten.
- Implementation state was pushed before report creation; PR #70 is open and
  unmerged, and auto-merge was not enabled.
- No raw image, YAML, prompt, label, answer, credential, model weight, customer
  data or request artifact entered the repository or OAP evidence.
- No second model process, unrelated unit, GPU, driver, firewall, route, VPN,
  port or persistent request-data location was changed.

## Limitations/blockers

- The initial 180-second readiness window was insufficient for the resident
  checkpoint load, although the same process became ready shortly afterward;
  no blocker remained for the final bounded qualification.
- The live max-width scenario proves configured filter diagnostics and source
  accounting only. It is not a SAM2 recall, roof/panel semantic, or model
  accuracy benchmark.
- The GPU pytest marker remains skipped in the CPU suite; the order-authorized
  private-LAN qualification supplied live evidence.
- No blockers remain within this order. Coding does not merge or declare
  acceptance.

## Factual strategic follow-up

- Strategic review/acceptance of PR #70 remains outstanding under OAP
  governance. The PR is intentionally left OPEN and unmerged.
