# OAP Coding-Agent Report — 014-b

## Work order

- Identifier/order/objective/PR mode: `014-b` — restore the area-first
  post-filter short circuit; amend the existing Objective-014 PR #70.

## Status

COMPLETE

## Executive summary

Restored the terminal `area > post_maxsize` branch before segmentation access.
Oversized candidates now retain their exact numeric area and report `0/0` bbox
dimensions because bbox evaluation did not occur. Added direct non-access and
precedence regressions, updated the affected diagnostic documentation, pushed the
correction to PR #70, and completed the authorized private-LAN service closure.

## Authoritative GitHub state

- Repository: https://github.com/ulfe-lmi/slaif-zap-it
- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/70
- State: `OPEN`; base `main`; head branch
  `oap/014-a-post-filter-rejection-diagnostics`; mergeable; coding merge: `NO`.
- Starting SHA: `2013628383c11f6faadae1ec6b95f6374c63f2d4` (014-a report-only
  head); remote `main`: `2e8c67997c2480cf66f5c87a1e19afba4c6d368f`.
- Implementation head SHA: `5c4b35d53cd8f2588f05bff0894fb5574092f562`
- Report publication commit: SELF
- New PR: `NO`; amended existing PR #70: `YES`.

## Changes/files

- `src/postprocessing.py`: area-first terminal evaluation and explicit
  not-evaluated bbox sentinel.
- `tests/test_postprocessing.py`: segmentation non-access regression and updated
  maxsize precedence expectations.
- `README.md`, `ARCHITECTURE.md`, `TESTING.md`, and the affected diagnostic
  sections of `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, and `docs/SERVICE-DATASHEET.md`.
- Exact wrapper-supplied `oap/active` (`014-b`) and immutable
  `oap/orders/014-b-restore-area-first-filter-short-circuit.md` bytes were
  committed with the implementation.
- No prior order or report was rewritten; no `014-a` file was changed.

## Acceptance evidence

- `_evaluate_candidate` compares `area_value > post_maxsize` immediately after
  reading and converting `area`; the terminal return does not access,
  convert, index, iterate, or measure segmentation and uses numeric `0/0` bbox
  dimensions.
- The direct regression supplies a dict-like candidate with `area` but no
  segmentation and an accessor that raises if segmentation is requested. It
  succeeds, removes the candidate once under `removed_by_maxsize`, and records
  `area_px: 101`, `bbox_width_px: 0`, and `bbox_height_px: 0`.
- Existing multi-limit tests now require `0/0` for maxsize records while
  retaining measured inclusive dimensions for empty, width, height, and
  retained-path behavior. Legacy list return, identity/order, aggregate
  invariants, bounded numeric records, and service/core behavior remain green.
- Public documentation explicitly distinguishes maxsize `0/0` dimensions that
  were not evaluated from empty-mask `0/0` dimensions.

## Verification

- `.venv/bin/pytest -q tests/test_postprocessing.py`: PASSED — 9 tests.
- `.venv/bin/pytest -q tests/test_postprocessing.py tests/test_core_engine.py tests/test_service_api.py tests/test_service_units.py`: PASSED — 138 tests; one existing dependency deprecation warning.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 466 passed, 1 GPU test skipped honestly because `ZAP_IT_RUN_GPU=1` was not set, 79% total coverage against the 64% gate.
- `.venv/bin/ruff format --check .`: PASSED — 142 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — no unexpected artifact findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exact seven reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED.
- Shell syntax check: NOT RUN — no shell files changed by this order.

## CI/checks

All current-head checks on implementation SHA
`5c4b35d53cd8f2588f05bff0894fb5574092f562` are `PASSED`/`SUCCESS`:

- `static (format, lint, build)`
- `tests (py3.10)`
- `tests (py3.11)`
- `tests (py3.12)`
- `release (artifact audit)`
- `Analyze (python)`
- `CodeQL`

## GPU/service/resource evidence

- Authorized physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; process visibility was
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, logical device
  `cuda:0`.
- One authorized restart of user-manager `zap-it-lan.service` was performed.
  Final PID `416545` remained stable across all requests, `NRestarts=0`, the
  unit is enabled/active/running/ready, and exactly one listener is bound at
  `10.8.132.76:17891`.
- The first two bounded 45-second readiness polls after restart were FAILED at
  HTTP 503 while the resident model registry was still loading. No corrective
  mutation was made; polling continued. The next bounded poll reached HTTP 200
  after 14 attempts. The service then remained ready for qualification.
- Final live state showed only PID `416545` on the assigned GPU, 13,547 MiB
  device memory used and 13,524 MiB attributed to that process. The request
  workspace was empty before and after the probe; the recent journal passed a
  sanitized no-error/no-secret check.
- The mode-0600 operator environment remained unchanged, with the same verified
  digest `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  No key value was printed, logged, committed, or reported.
- Authenticated live checks: readiness `200`, metrics `200`; missing and wrong
  inference keys `401`; `/docs` and `/openapi.json` `404`.
- The in-memory goat-derived diagnostic probe used the authorized ignored
  fixture/config without persisting request content. Optional CLIP, BLIP3, and
  visualization work was removed; SAM2 produced 10 candidates. L3 JSON, repeat
  L3 JSON, L3 ZIP, and L2 JSON all returned `200`; diagnostics were deterministic
  and JSON/ZIP-equivalent: evaluated `10`, max-w removals `10`, retained `0`,
  maxsize/empty/max-h removals `0`, rejection records `10`, truncation `0`.
  L2 omitted `post_filter_diagnostics`.

## Documentation/provenance

The public contract now states that maxsize is decided before segmentation
access, carries exact area with zero bbox dimensions because dimensions were not
evaluated, and is distinct from the empty-mask zero sentinel. No model identity,
revision, license, dependency, hardware, renderer, BLIP3, service-contract, or
residency claim was changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only the exact active order 014-b was executed.
- The existing PR #70 was amended; no new PR was created, and no merge or
  auto-merge was performed.
- No unassigned GPU, unrelated process/unit, driver, firewall, route, VPN,
  global credential, or persistent request-data location was modified.
- No raw image/YAML, labels, prompts, answers, credentials, or model weights
  entered the report or OAP evidence.

## Limitations/blockers

- The canonical CPU suite's opt-in GPU test remained `SKIPPED`; the required
  authorized private-LAN qualification was performed separately on physical GPU
  0 and passed after normal cold-load latency.
- No additional model inference beyond the order-authorized SAM2 diagnostic
  probe was required.

## Factual strategic follow-up

PR #70 remains open and unmerged for strategic review/acceptance.
