# OAP Coding-Agent Report — 022-c

## Work order

- Identifier/order/objective/PR mode: `022-c` / preserve logical visualization
  ID with safe path / Objective 022 / amend existing PR.

## Status

COMPLETE

## Executive summary

Implemented the bounded logical `visualization_id` response metadata contract
while keeping service visualization members fixed and ordinal. A configured
stream such as `final-labelled-ripe-tomatoes` is now associated with
`visualization/stream-0001.png` in JSON descriptors, ZIP manifests, and omission
records without entering a path, ZIP member name, log message, or metric label.
The service rejects duplicate visualization IDs, restores the independent
32-rule BLIP3 YAML definition limit, and retains the separate operator-owned
1..256 planned-question capacity.

The exact live proof returned HTTP 200 with the required fixed member, logical
ID, hashes/sizes, 97-prompt accounting, five semantic score classes, routing
target, stage counts, final labels/bounds, and labelled PNG. CPU, package,
release, security, and all implementation-head CI checks passed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Pull request: [#78](https://github.com/ulfe-lmi/slaif-zap-it/pull/78), open,
  base `main`, branch `oap/022-a-canonical-clip-multiprompt`.
- Accepted base SHA: `d341a3c4ba47b71d10d70682771b315041dcbcb8`.
- Starting SHA: `464456edde622456d8fbc420d15a8bc0345d51fe`.
- Implementation head SHA: `c0276b436fb5d18a5291a876b45972a80b63d796`.
- Report publication commit: SELF
- New PR: no; amended existing PR #78: yes; coding merge: NO.

## Changes/files

One implementation commit (`c0276b4`, 25 paths) contains:

- artifact metadata and fixed service-safe naming: `src/core/sinks.py`,
  `src/service/artifacts.py`, `src/service/envelope.py`;
- typed public schema and capability disclosure:
  `src/service/schemas.py`, `src/service/capabilities.py`;
- YAML policy, duplicate-ID validation, fake/API rendering seam, and safe
  renderer logging: `src/service/yaml_input.py`, `src/service/fake_engine.py`,
  `modules/visualizer.py`;
- deterministic contract tests:
  `tests/test_labelled_visualization.py`, `tests/test_objective_022.py`,
  `tests/test_parity_hardening.py`, `tests/test_sam2_configuration.py`;
- maintained contracts and documentation: `README.md`, `ARCHITECTURE.md`,
  `TESTING.md`, `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`,
  `docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md`, `docs/runtime.md`;
- unchanged orchestration transcript: `oap/active` and
  `oap/orders/022-c-preserve-logical-visualization-id-with-safe-path.md`.

Migration/compatibility notes: service L3 visualization descriptors gain an
optional field; identity and non-visualization debug descriptors omit it.
Trusted non-service artifact names remain unchanged. The former
`MAX_BLIP3_QUESTIONS` symbol remains a source-compatible alias, while validation
uses `MAX_BLIP3_RULE_DEFINITIONS = 32`. The immutable operator
`SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS` default/range remains 256 / 1..256 planned
questions. No dependency, model, revision, precision, residency, response
budget, deadline, object limit, credential, network, or host policy changed.

## Acceptance evidence

1. **Logical visualization metadata and fixed names — PASSED.** Service-safe
   raw artifacts carry the validated configured ID separately from the fixed
   `visualization/stream-####.png` name. Multiple streams use deterministic
   ordinal assignment in pipeline order; changing an ID changes metadata but
   not member/path names or bytes. Duplicate IDs are rejected before inference;
   unsafe/path-like IDs remain rejected. Trusted non-service naming remains
   `visualization/<configured-id>.png` for compatibility.

2. **Descriptor/manifest/ledger contract — PASSED.** JSON descriptors and ZIP
   manifest descriptors share exact names, logical IDs, SHA-256 values, and
   byte sizes. Identity and candidate/debug artifacts omit `visualization_id`.
   A budget-omitted visualization retains its logical ID in the typed omission
   record. Pydantic schema tests enforce the same bounded ID pattern as the YAML
   validator.

3. **Independent BLIP3 limits — PASSED.** The service accepts 32 rule
   definitions and rejects 33 with `response_too_large` during request config
   validation. The operator-owned default 256 planned canonical candidate
   questions remains admitted, and planned excess remains typed
   `resource_limit` HTTP 413 before generation. Capabilities, schema/API
   surfaces, and maintained docs disclose the two independent units.

4. **Exact fake/API fixture — PASSED.** The committed fixture is 4,090 bytes,
   SHA-256
   `e89a149d2f3530d1b7a4cb919b3641f230fb51fbd6a0ee59805587d4db89cd3d`, with
   CLIP prompt counts `ripe_tomato=32`, `foliage=15`, `stem_or_vine=15`,
   `greenhouse_structure=20`, `background=15`, total 97, and one BLIP3 rule.
   The fake/API path proves the logical ID
   `final-labelled-ripe-tomatoes` maps to the fixed stream member.

5. **CPU/package/CI proof — PASSED.** The full CPU suite passed 891 tests with
   one explicit GPU skip and 82.00% total coverage. Focused visualization,
   envelope, schema, API, YAML, and limit tests passed. Direct and sdist-built
   wheel manifests matched; package scans, Twine, documentation, compilation,
   and isolated installed-wheel JSON/ZIP smokes passed.

6. **Exact live request — PASSED.** One authenticated verbosity-3 ZIP request
   returned HTTP 200; no retry or YAML mutation occurred. Evidence is preserved
   under `/dev/shm/slaif-zap-it-022c.eRDULP`:

   - ZIP: `result.zip`, 1,401,310 bytes, SHA-256
     `70a5f11fefa5277cd46919a37c369f8afb905325000087fd82c5f52aba72bb9f`;
   - service visualization member:
     `visualization/stream-0001.png`, 1,344,837 bytes, SHA-256
     `df74674dca77c563da589962461d3558eeeec1cf70349979f6593486cd3bf783`;
   - fixed review copy: `final-labelled-review.png`, mode 0600, same size and
     SHA-256 as the ZIP visualization member;
   - ZIP members were exactly `manifest.json`, `detections.yolo.txt`,
     `identity-mask.png`, and `visualization/stream-0001.png`; no traversal,
     duplicate, or symlink member was present;
   - the visualization descriptor associated the fixed member with
     `visualization_id=final-labelled-ripe-tomatoes`; identity-mask and
     candidate/debug descriptors omitted that field;
   - manifest hash/size checks and typed artifact/omission schema validation
     passed; artifact delivery was `eligible=31`, `selected=1`, `delivered=1`,
     `budget_omitted=0`, `selection_excluded=30`, `truncated=false`;
   - prompt counts were 32/15/15/20/15, total 97; semantic score classes were
     `background`, `foliage`, `greenhouse_structure`, `ripe_tomato`, and
     `stem_or_vine`; route target was `ripe_tomato`;
   - candidate counts were SAM2 205, geometry evaluated 205 with 33 rejected
     and 172 retained, CLIP scored 172, initially routed 156, routed after cap
     156, BLIP3 verified 136, and final 30;
   - all 30 final objects were labelled `ripe_tomato`. Inclusive bbox widths
     ranged from 13 through 103 pixels and inclusive bbox heights from 13
     through 114 pixels. These are inclusive dimensions, not exclusive spans.

   Bounded visual inspection found many larger hanging fruits visibly covered
   by coloured output masks, while several smaller/occluded fruit regions
   remained unlabelled; some masks also visibly overlapped adjacent fruit or
   covered stems/foliage/background. These are human visual observations, not
   recall, precision, or semantic-accuracy measurements.

## Verification

- `git fetch origin --prune`: PASSED — remote PR head reconciled before and
  after implementation.
- `.venv/bin/pytest -q tests/test_labelled_visualization.py tests/test_objective_022.py tests/test_parity_hardening.py tests/test_service_units.py tests/test_service_api.py tests/test_core_sinks.py`:
  PASSED — 199 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 891 passed, 1 explicit GPU skip, 82.00% total coverage.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `git diff --check` and staged diff check: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py` on direct wheel/sdist
  and sdist-built wheel: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py --compare-wheels`:
  PASSED — direct and sdist-built wheel member manifests matched.
- `.venv/bin/python scripts/scan_release_artifacts.py` archive scan: PASSED —
  no unexpected findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree`: PASSED —
  exactly 7 reviewed baseline findings, no additions/removals.
- `.venv/bin/python -m twine check` on direct and sdist-built artifacts: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- Isolated direct-wheel and sdist-built-wheel `smoke_installed_package.py`:
  PASSED — JSON/ZIP package-version parity, console script, and
  site-packages import.
- Exact fixture hash/count, fake/API mapping, limit independence, schema,
  omission, fixed-name, JSON/ZIP parity, and deterministic repeat tests:
  PASSED.
- One controlled service restart: PASSED — PID changed from 697088 to 708466;
  stable cold readiness was HTTP 200 with `NRestarts=0`.
- One exact live request and post-request validation: PASSED — HTTP 200,
  fixed member, logical ID, descriptor hashes/sizes, schema, stage counts,
  labels/bounds, and labelled PNG.
- Final private auth/docs boundary checks: PASSED — health/readiness 200,
  unauthenticated capabilities 401, `/metrics` 401, `/docs` 404, and
  `/openapi.json` 404.

## CI/checks

All required checks below ran on implementation SHA
`c0276b436fb5d18a5291a876b45972a80b63d796` and are `PASSED`:

- `static (format, lint, build)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432883001/job/99622254657).
- `tests (py3.10)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432883001/job/99622254778).
- `tests (py3.11)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432883001/job/99622254777).
- `tests (py3.12)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432883001/job/99622254858).
- `release (artifact audit)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432883001/job/99622254772).
- `Analyze (python)` — [CodeQL workflow job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33432882972/job/99622254399).
- `CodeQL` — [CodeQL check](https://github.com/ulfe-lmi/slaif-zap-it/runs/99622540176).

## GPU/service/resource evidence

- One controlled restart of only the user unit `zap-it-lan.service`: old PID
  697088 -> new PID 708466; `NRestarts=0`; unit remained enabled, active,
  running, and ready.
- Listener remained private and exact: `10.8.132.76:17891`; no public bind or
  port change.
- Assigned physical GPU only: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`. The process exposed it as
  logical `cuda:0` through the existing `CUDA_VISIBLE_DEVICES=0` policy.
- Sole compute process after live proof: PID 708466, service Python process,
  using 10,868 MiB of the assigned card; current GPU snapshot was 10,891 MiB
  used and 13,233 MiB free. No unassigned GPU or unrelated process was touched.
- User-unit memory snapshot after proof: 14,561,144,832 bytes current and
  19,958,501,376 bytes peak.
- `/dev/shm/slaif-zap-it`: mode 0700 and empty after request cleanup. Evidence
  directory `/dev/shm/slaif-zap-it-022c.eRDULP`: mode 0700; retained ZIP and
  review PNG: mode 0600. No request data was persisted outside this explicitly
  preserved tmpfs evidence.
- Environment file remained mode 0600 with unchanged SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- Exact live input hashes: image
  `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`;
  configuration
  `e89a149d2f3530d1b7a4cb919b3641f230fb51fbd6a0ee59805587d4db89cd3d`.

## Documentation/provenance

Maintained README, architecture, testing, API, configuration, algorithm, core,
output-parity, runbook, runtime, and service-datasheet documentation now
describes fixed service visualization names, logical-only `visualization_id`,
descriptor/manifest/omission parity, duplicate-ID rejection, the 32 uploaded
BLIP3 rule-definition ceiling, and the independent 256 planned-question
operator capacity. No model weights, credentials, raw request YAML/image,
answers, or customer data entered the report or GitHub metadata.

## Deferred human adjudication

- Critical register action: NONE.
- `CRITICAL.md` was read because the active order required the current-register
  refresh. No entry was appended or changed.

## Safety/scope confirmations

- Only active order `022-c` was executed; no adjacent order was selected.
- Existing PR #78 was amended; no new PR was created; no merge, auto-merge,
  release, tag, or public exposure was attempted.
- Only the exact order-assigned physical GPU 0 / UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` was used. No unassigned GPU,
  unrelated process/service, system CUDA/driver, firewall/VPN, credential,
  model cache, port configuration, or global agent configuration was mutated.
- Exactly one controlled restart and one authenticated live request were
  performed after implementation CI was green. There was no retry, restart
  again, config mutation, candidate clamp, or limit increase.
- The private keyed service was left newest, active, ready, and listening on
  the same private address. The bearer was read only into process memory,
  supplied without appearing in argv/logs/report, and unset immediately.
- Request data used RAM/tmpfs only; the preserved evidence directory is the
  explicit order-required operator evidence location.

## Limitations/blockers

No acceptance blocker remains for 022-c. The live labelled image is a bounded
pipeline/contract proof, not a semantic-quality or recall/precision benchmark;
the visual observations above intentionally do not claim model accuracy. The
service remains private and operator-keyed, and PR merge remains outside coding
authority.

## Factual strategic follow-up

Review the pushed implementation and preserved live evidence on PR #78. The
strongest reason not to merge is that the model output visibly includes some
missed, overlapping, and non-fruit masks; the answer is that semantic accuracy
is explicitly outside 022-c scope and non-goal, while the required fixed-name,
logical-ID, independent-limit, CPU/CI, and exact live contract evidence passed.
No next order is selected by coding.
