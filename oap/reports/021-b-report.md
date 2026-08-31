# OAP Coding-Agent Report — 021-b

## Work order

- Identifier: `021-b`
- Objective: Objective 021 — make response-tail omission real and close schema proof
- PR mode: amend existing numeric-objective PR
- Order authority: `oap/orders/021-b-make-response-tail-omission-real-and-close-schema-proof.md`

## Status

COMPLETE

## Executive summary

PR #77 now removes optional artifact payloads from the prepared JSON/ZIP
response when the response-byte cap is reached, while retaining the required
identity mask and structured candidate evidence. The request-local artifact
ledger decides status before recording, keeps selection and budget overflow
arithmetic separate, bounds public omission records at 576, and rejects
contradictory duplicate artifact facts.

The diagnostic selector is strict for stages and source candidate IDs 1..256,
including hostile nested members. The capabilities document now exposes one
OpenAPI-enumerated ordered field catalog generated with the compatibility field
dictionaries, and L3 response metadata uses named typed models for statuses,
counts, timings, provenance and effective CLIP routing. No dependencies, model,
GPU, service, network, credential or CRITICAL-register behavior changed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#77](https://github.com/ulfe-lmi/slaif-zap-it/pull/77), `OPEN`
- PR title: `Objective 021: non-fatal artifacts and complete API contract`
- Base: `main` at `f2d58f7512af41751cb647bcd502d767a007f199`
- Branch: `oap/021-a-nonfatal-artifacts-and-contract-docs`
- Starting local/remote head: `ebf4fc4f64646a7afb7af590e9afa12310e1614b`
- Implementation head SHA: `00026cb63acd07cc61183bb2482aa64593445824`
- Implementation commit: `00026cb63acd07cc61183bb2482aa64593445824`
- Report publication commit: SELF
- SELF parent: `00026cb63acd07cc61183bb2482aa64593445824`
- New PR: no
- Amended existing PR: yes
- Coding merge/auto-merge: NO

## Changes/files

The implementation commit contains exactly these 12 paths:

- `docs/API.md`
- `docs/CONFIG.md`
- `docs/SERVICE-DATASHEET.md`
- `oap/active`
- `oap/orders/021-b-make-response-tail-omission-real-and-close-schema-proof.md`
- `src/service/artifacts.py`
- `src/service/capabilities.py`
- `src/service/envelope.py`
- `src/service/schemas.py`
- `src/service/yaml_input.py`
- `tests/test_objective_021.py`
- `tests/test_service_api.py`

The report-only SELF commit changes only `oap/reports/021-b-report.md`.

## Acceptance evidence

- Response fitting now rebuilds the immutable delivered artifact tuple after
  each response omission. JSON descriptors, hashes, byte totals and ZIP
  members use exactly the reduced tuple; `identity-mask.png` remains present.
  CLIP/BLIP3 nested artifact statuses become `omitted_response_limit` while
  candidate geometry and verification evidence remains.
- Synthetic direct-builder thresholds were deterministic: JSON essential
  4,014 bytes, with optional 91,666 bytes, cap 47,840 bytes, fitted 4,205
  bytes; ZIP essential 1,815 bytes, with optional 2,924 bytes, cap 2,369
  bytes, fitted 1,896 bytes. Both returned success with the optional member
  absent, the identity member retained, exact omission reason and descriptor
  parity. A cap below each essential size returned `response_too_large`.
- HTTP JSON and ZIP pressure tests prove the same behavior at the API boundary,
  including successful optional-tail omission and hard 413 essential overflow.
- Ledger admission now records final status before applying the 576 public
  omission bound. Mixed synthetic offers preserve exact eligible, selected,
  delivered, selection-excluded, budget-omitted and overflow arithmetic;
  omission output is bounded to 576 with one warning. Duplicate name offers
  with contradictory stage/media/identity/size facts and payload sizes are
  rejected as internal artifact errors; identical repeats are idempotent.
- Hostile selector validation accepts boundary IDs 1 and 256 and rejects 0,
  257, bool, duplicate, nested-list and nested-mapping members. Stage member
  types are checked before uniqueness. Response Pydantic validation has the
  same strict 1..256 item contract and preserves valid requested/effective
  ordering semantics.
- `configuration.field_catalog` contains all 80 validator leaf paths exactly
  once in sorted order. Each record has a typed `CapabilityField` descriptor,
  non-empty type/stage/description metadata, and explicit
  required/nullable/default semantics. OpenAPI exposes the complete path enum.
- `ServiceMetadata` references named `StageStatus`, `CandidateCounts`,
  `TimingMetadata`, `ProvenanceMetadata` and `ClipRoutingConfiguration` models.
  Timing values are bounded finite non-negative milliseconds; CLIP score and
  sanitized runtime maps remain typed and bounded.
- Existing raw CLIP crops, complete vectors/routing, BLIP3 views/questions,
  geometry diagnostics, SAM2 provenance, fixed safe artifact names, labelled
  rendering, legacy CLI boundary and shipped configurations remain covered by
  the existing suites.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 866 passed, 1 GPU integration test honestly skipped; total
  coverage 81.70%, required threshold 64%.
- `.venv/bin/pytest -q tests/test_objective_021.py tests/test_service_api.py tests/test_service_units.py`:
  `PASSED` — 143 passed, 1 warning.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current
  documents.
- `git diff --check`: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`:
  `PASSED` — wheel and sdist member manifests validated.
- `.venv/bin/python scripts/scan_release_artifacts.py --baseline .secrets.baseline --tracked-tree dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`:
  `PASSED` — 7 tracked baseline findings, no unexpected archive findings.
- `.venv/bin/python -m twine check dist/*`: `PASSED`.
- `systemd-analyze verify deploy/zap-it-local.service`: `PASSED`.
- Direct installed-wheel `smoke_installed_package.py`: `PASSED` — JSON/ZIP,
  console script and site-packages import.
- Sdist-built-wheel `smoke_installed_package.py`: `PASSED` — JSON/ZIP, console
  script and site-packages import.
- Exploratory `.venv/bin/python scripts/smoke_installed_package.py --help`:
  `FAILED` — this script has no help mode and executed from the checkout
  context; the prescribed isolated invocations above are the authoritative
  smoke results.

## CI/checks

All seven required checks were `PASSED` on the exact implementation head
`00026cb63acd07cc61183bb2482aa64593445824`:

- `static (format, lint, build)` — `SUCCESS`
- `tests (py3.10)` — `SUCCESS`
- `tests (py3.11)` — `SUCCESS`
- `tests (py3.12)` — `SUCCESS`
- `release (artifact audit)` — `SUCCESS`
- `Analyze (python)` — `SUCCESS`
- `CodeQL` — `SUCCESS`

The same seven check names were re-run on the report-only SELF head and verified
`SUCCESS` before FIFO signaling.

## GPU/service/resource evidence

- Live GPU phase: `SKIPPED` — this order explicitly forbids GPU use, live
  inference, service restart/reload and host mutation. No physical GPU was
  selected, exposed or allocated.
- No service, listener, port, firewall, network, driver, cache, process or
  unrelated workload was changed. No protected GPU or service was inspected by
  mutation.
- All proofs used CPU/fake engines and synthetic in-memory bytes. No request
  image, YAML, prompt, answer, credential, customer data or model weight was
  written to the report or repository.
- Disposable release-smoke environments were created under `/dev/shm` and
  cleaned after each valid run. No request artifacts were written to repository
  output paths or persistent request storage.

## Documentation/provenance

Updated `docs/API.md`, `docs/CONFIG.md` and `docs/SERVICE-DATASHEET.md` to
document reduced response tuples, essential identity retention, typed omission
arithmetic, strict candidate selection and the capabilities field catalog/
named response models. The 021-b order and `oap/active` selector were included
unchanged in the implementation commit.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was not read or modified because the active order decision is
  `NONE` and no append/cross-reference was ordered.

## Safety/scope confirmations

- Only active order `021-b` was executed; no adjacent order was selected.
- Exactly existing PR #77 was amended; no new PR, merge or auto-merge occurred.
- The Objective 021-a implementation/report, remote main, prior OAP transcript
  and critical register were preserved.
- No dependency, model, GPU, service, network, credential, host policy or
  persistent request-data behavior changed.
- No secrets, raw request content, private inputs, model weights or customer
  data entered GitHub/OAP artifacts.

## Limitations/blockers

- Semantic model accuracy and real-image recall/precision remain unmeasured;
  this order supplies CPU/fake contract evidence only.
- Live GPU qualification and running-service smoke were explicitly outside the
  order.
- The strongest remaining reason not to merge is that strategic/maintainer
  review and acceptance/merge authority remain outstanding despite green CI.

## Factual strategic follow-up

Review PR #77 at the report-only SELF head. The coding agent has completed the
bounded 021-b implementation, pushed all non-report state, verified the exact
implementation and SELF-head checks, published this immutable report and sent
the required response FIFO `OK`; no further coding mutation is authorized in
this round.
