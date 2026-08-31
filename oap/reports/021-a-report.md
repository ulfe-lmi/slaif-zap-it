# OAP Coding-Agent Report — 021-a

## Work order

- Identifier: `021-a`
- Objective: Objective 021 — non-fatal diagnostic delivery and complete API evidence
- PR mode: new numeric-objective PR
- Order authority: `oap/orders/021-a-artifact-delivery-resource-errors-and-contract-docs.md`

## Status

COMPLETE

## Executive summary

Optional L3 diagnostic and visualization artifacts now use a request-local,
deterministic admission ledger. Stage/candidate/page selection is strict and
request-local; count, single-artifact, aggregate-raw and response-byte misses
omit optional bytes with typed reasons while inference, complete CLIP/BLIP3
evidence, ordering and essential serialization continue. JSON and ZIP use the
same admitted set and exact hashes/sizes.

SAM2 operator-cap failures now return HTTP 413 `resource_limit` with sanitized
requested/effective values, exact estimates, causes, public limits and
same-validator admissible alternatives. Capabilities, OpenAPI models,
configuration inventory, shipped-example validation and the required contract
documentation were refreshed. No model, dependency, GPU, service, network,
credential or persistent-request-data behavior changed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#77](https://github.com/ulfe-lmi/slaif-zap-it/pull/77), `OPEN`
- PR title: `Objective 021: non-fatal artifacts and complete API contract`
- Base: `main` at `f2d58f7512af41751cb647bcd502d767a007f199`
- Branch: `oap/021-a-nonfatal-artifacts-and-contract-docs`
- Starting local/remote base SHA: `f2d58f7512af41751cb647bcd502d767a007f199`
- Implementation head SHA: `653520dab50bad2b6c5a5bc588c178d2a149643d`
- Implementation commits: `4712333584a424161f6174d1f6ab9fb3d9f81f99`,
  followed by bounded correction `653520dab50bad2b6c5a5bc588c178d2a149643d`
- Report publication commit: SELF
- SELF parent: `653520dab50bad2b6c5a5bc588c178d2a149643d`
- SELF head: the report-only commit containing this file; literal remote SHA
  and parent were verified after publication
- New PR: yes
- Amended existing PR: no
- Coding merge/auto-merge: NO

## Changes/files

The implementation history contains exactly these 30 paths:

- `README.md`
- `docs/ALGORITHMS.md`
- `docs/API.md`
- `docs/CONFIG.md`
- `docs/CORE.md`
- `docs/OUTPUT-PARITY.md`
- `docs/RUNBOOK.md`
- `docs/SERVICE-DATASHEET.md`
- `modules/classifier/clip.py`
- `modules/verifier/blip3.py`
- `oap/active`
- `oap/orders/021-a-artifact-delivery-resource-errors-and-contract-docs.md`
- `src/core/config.py`
- `src/core/engine.py`
- `src/core/sinks.py`
- `src/service/app.py`
- `src/service/artifacts.py`
- `src/service/capabilities.py`
- `src/service/envelope.py`
- `src/service/errors.py`
- `src/service/resources.py`
- `src/service/schemas.py`
- `src/service/yaml_input.py`
- `tests/test_core_engine.py`
- `tests/test_mask_views.py`
- `tests/test_objective_021.py`
- `tests/test_parity_hardening.py`
- `tests/test_raw_sam2_visualizations.py`
- `tests/test_real_yaml_config.py`
- `tests/test_sam2_configuration.py`

The implementation commit preserves prior orders/reports and does not modify
`CRITICAL.md`. This final child changes only `oap/reports/021-a-report.md`.

## Acceptance evidence

- `ArtifactDeliveryLedger` greedily admits fixed logical names after strict
  stage/candidate/page selection and records exactly one typed omission per
  excluded or budget-rejected artifact, with a bounded 576-entry public ledger.
- `BoundedMemoryArtifactSink` no longer gates CLIP/BLIP3/SAM2 work through
  `ensure_capacity`; optional sink overflow is non-fatal. Invalid names,
  corrupt records and genuine encoding failures remain errors.
- Final visualization and raw-SAM2 arrays use the same optional admission
  policy. Request-safe service visualization names are fixed stream tokens and
  never contain client visualization IDs.
- L3 `service.artifact_delivery` exposes requested/effective selection,
  applied state, all operator budgets, eligible/selected/delivered/excluded and
  omitted counts, estimated raw/base64/ZIP totals, actual delivered byte
  totals, truncation, delivered names, and typed omission records.
- JSON descriptors and ZIP manifests are built from the same admitted payloads;
  SHA-256 and size values match decoded JSON and ZIP bytes.
- Response-size fitting removes admitted optional artifacts from the tail in
  reverse order and updates candidate/BLIP3 artifact statuses. A response is
  still rejected only when its essential document cannot fit.
- `diagnostic_artifacts` accepts only the strict four-field mapping: unique
  stages, null or unique positive source IDs 1..256, page 1..65535 and page
  size 1..48. It never enables stage debug flags and does not select paths or
  destinations. Below L3 it remains valid with `applied: false` semantics.
- SAM2 field, prompt-estimate and multimask-prediction capacity paths return
  typed sanitized details and alternatives generated and checked against the
  same intrinsic validator and active operator caps. The rejected mapping is
  not mutated.
- The canonical capabilities inventory contains 80 accepted service leaf
  paths and is checked at runtime against the validator inventory. OpenAPI
  includes named capabilities, artifact-delivery, completion and resource
  error models with descriptions.
- The four tracked product examples under `configs/*.yaml` (`glasswool`,
  `icecream`, `soccer`, `tomato`) pass the service parser, default SAM2 caps and
  `CoreConfig`; the local checkout additionally contained two ignored goat
  examples, which also passed the same check.

### Migration notes

- No change to the Objective 020 default `raw_bbox_crop` CLIP or
  `single_dilated_blur` BLIP3 modes.
- `diagnostic_artifacts` is optional and only narrows enabled L3 debug delivery.
- Requests that previously overflowed optional artifact budgets now succeed
  with `service.artifact_delivery.truncated=true` and typed omissions.
- Essential response overflow can still return HTTP 413 `response_too_large`.
- Trusted-CLI masked CLIP mode and batch-only fields remain retained exactly as
  found.
- Existing error envelopes remain the three original fields except applicable
  SAM2 `resource_limit` errors, which add optional sanitized `details`.

## Verification

- `.venv/bin/pytest -q`: `PASSED` — 846 passed, 1 GPU integration test skipped
  honestly because live GPU execution was not authorized; 2 expected warnings.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 846 passed, 1 skipped; total coverage 81.34%, required threshold
  64%.
- Focused Objective 021 and service/core/artifact tests: `PASSED` — 182 passed,
  1 warning on the final correction head.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current
  documents; stale optional-artifact preflight/fatal claims removed.
- `git diff --check`: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED` — packaging emitted only
  existing setuptools deprecation warnings.
- `scripts/verify_release_artifacts.py` on wheel, sdist and sdist-built wheel:
  `PASSED`; wheel member manifests matched with no differences.
- `scripts/scan_release_artifacts.py` with `.secrets.baseline`: `PASSED` — no
  unexpected findings; tracked-tree scan preserved all 7 reviewed findings.
- `.venv/bin/python -m twine check dist/*`: `PASSED`.
- `systemd-analyze verify deploy/zap-it-local.service`: `PASSED`.
- Direct installed-wheel `smoke_installed_package.py`: `PASSED` — JSON and ZIP
  smoke, console script and site-packages import verified.
- Sdist-built-wheel isolated `smoke_installed_package.py`: `PASSED` — JSON and
  ZIP smoke, console script and site-packages import verified.
- An initial direct smoke invocation without the isolated venv on `PATH` was
  `FAILED` by the smoke harness; the CI-prescribed `PATH=<venv>/bin:$PATH`
  rerun was `PASSED` and is the authoritative result.

## CI/checks

All seven visible GitHub checks were `PASSED` on the exact implementation head
`653520dab50bad2b6c5a5bc588c178d2a149643d` before report publication:

- `static (format, lint, build)` — `SUCCESS`
- `tests (py3.10)` — `SUCCESS`
- `tests (py3.11)` — `SUCCESS`
- `tests (py3.12)` — `SUCCESS`
- `release (artifact audit)` — `SUCCESS`
- `Analyze (python)` — `SUCCESS`
- `CodeQL` — `SUCCESS`

The same seven check names were required and verified `PASSED` on the report-only
SELF head after publication, before FIFO signaling.

## GPU/service/resource evidence

- Physical GPU live phase: `SKIPPED` — the order explicitly requires CPU/fake
  proof and forbids live GPU inference. The assigned order context remains
  physical GPU0, UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, exposed as
  logical `cuda:0` only when the operator-controlled service runs.
- No GPU allocation, reset, process termination, service restart, port change,
  network/firewall change, cache change or operator environment mutation was
  performed. The accepted private-LAN service was left running and unchanged.
- CPU tests and fake service use generated in-memory images/configuration. No
  request image, YAML, prompt, answer, credential, customer data or model
  weight entered the report or repository.
- Package smoke scratch directories were under ephemeral shared memory and
  were cleaned. No request artifacts were written to repository output paths,
  cwd or persistent storage.
- No live semantic accuracy, recall, precision, GPU memory or deployment
  qualification was measured.

## Documentation/provenance

The refreshed documentation set is `README.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/ALGORITHMS.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`,
`docs/RUNBOOK.md`, and `docs/SERVICE-DATASHEET.md`. It documents strict
selection/pagination, omission arithmetic, retained essential limits,
structured SAM2 alternatives, complete CLIP/BLIP3 evidence, fixed names,
defaults/ranges/stages, profile precedence and operator-limit rejection.

The order and active selector were included unchanged in implementation history:
`oap/orders/021-a-artifact-delivery-resource-errors-and-contract-docs.md` and
`oap/active` containing `021-a`.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was not read or modified because the active order decision is
  `NONE` and no append/cross-reference was ordered.

## Safety/scope confirmations

- Only active order `021-a` was executed; no adjacent order was selected.
- Exactly one new numeric-objective PR (#77) was created; no merge or auto-merge
  was performed.
- Existing Objective 020 PR #76, prior OAP orders/reports, and the critical
  register were preserved.
- Legacy batch sink and CLI behavior remain the compatibility boundary; the
  service-only admission ledger is not used to impose API budgets on the
  filesystem sink.
- No credentials, raw request content, private inputs, model weights or
  unnecessary host evidence entered GitHub/OAP artifacts.

## Limitations/blockers

- Semantic model accuracy and real-image recall/precision remain unmeasured;
  all new proof is CPU/fake/contract evidence.
- No live GPU or running-service smoke was authorized in this order.
- The strongest remaining reason not to merge is that strategic/maintainer
  review and acceptance/merge authority remain outstanding despite green CI.

## Factual strategic follow-up

Review PR #77 at the report-only SELF head. The coding agent has completed its
bounded implementation, pushed all non-report state, verified the PR checks and
published this immutable report; no further coding mutation is authorized in
this round.
