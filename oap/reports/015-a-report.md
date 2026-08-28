# OAP Coding-Agent Report — 015-a

## Work order

- Identifier/order/objective/PR mode: `015-a` — request-local SAM2
  configuration; new Objective-015 branch and PR.

## Status

COMPLETE

## Executive summary

Implemented strict request-local SAM2 generator configuration while keeping the
pinned model resident. The live holder now stores the SAM2 model only; each
accepted request resolves explicit/profile/default values, checks operator work
caps, and constructs one fresh fixed-mode generator without model reload,
device movement, dtype conversion or cache lookup. Added authenticated static
capabilities, typed `service.sam2` provenance/raw-count/timing metadata at all
verbosity levels, resource-limit errors, CPU/API coverage, and synchronized
operator documentation.

## Authoritative GitHub state

- Repository: https://github.com/ulfe-lmi/slaif-zap-it
- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/71
- State: `OPEN`; base `main`; head branch
  `oap/015-a-request-local-sam2-configuration`; coding merge: `NO`.
- Starting SHA / remote `main`:
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Implementation head SHA: `27aa21c39752dad6603df458b61141efd807fa04`
- Report publication commit: SELF
- New PR: `YES`; amended existing PR: `NO`; merge/auto-merge: `NO`.

## Changes/files

- `modules/segmenter/sam2.py` and `modules/segmenter/__init__.py`: explicit
  safe scalar/default/profile contract, exact prompt estimator, fixed
  `point_grids=None`/`output_mode=binary_mask` generator seam, model-only
  initialization and request-local generation.
- `src/service/yaml_input.py`, `src/service/settings.py`, and
  `src/service/errors.py`: strict YAML scalar/range validation, source
  resolution, operator caps/environment parsing, deterministic warnings and
  non-retryable `resource_limit` HTTP 413.
- `src/runtime/live_service.py`, `src/core/config.py`,
  `src/core/engine.py`, `src/core/results.py`, and
  `src/service/fake_engine.py`: resident model-only wiring and raw SAM2
  candidate/timing provenance without request-state write-back.
- `src/service/capabilities.py`, `src/service/app.py`,
  `src/service/schemas.py`, and `src/service/envelope.py`: authenticated
  static `/v1/capabilities`, explicit OpenAPI models, and `service.sam2` in
  every JSON/ZIP response level.
- `tests/test_sam2_configuration.py` plus updated runtime tests cover
  constructor exclusions, lifecycle isolation, strict types, profiles, exact
  estimates, caps, capabilities and JSON/ZIP metadata.
- Updated README, INSTALL, TESTING, architecture/core/API/configuration,
  output-parity, runtime, runbook, datasheet and deployment environment
  documentation.
- Exact wrapper-supplied `oap/active` (`015-a`) and immutable
  `oap/orders/015-a-request-local-sam2-configuration.md` bytes were committed
  with the implementation. No prior order/report was rewritten.

## Acceptance evidence

- The CPU constructor seam forwards only the 14 allowlisted SAM2 scalar
  fields. `profile` and `debug` are not constructor values; `point_grids` is
  always `None`, `output_mode` is always `binary_mask`, and arbitrary kwargs
  are not forwarded.
- The live resident loader calls model-only SAM2 initialization. CPU lifecycle
  tests construct distinct generators around one model identity in A/B/A order;
  the resident state has no request generator and no later request state leak.
- Server defaults and exact `fast`, `balanced`, and `quality` overrides are
  resolved with explicit > profile > default precedence. Sources are recorded
  independently for every effective scalar, including explicit values equal to
  inherited values.
- Strict integer/number/boolean handling rejects booleans-as-integers,
  integers-as-booleans, numeric strings, nulls, non-finite numbers, unknown
  profiles/fields and zero-point deep crop configurations. Hostile model,
  path, device, cache, network and credential controls remain `unsafe_config`.
- Admission computes the exact crop/grid prompt sum and the multimask
  prediction multiplier before readiness, gate acquisition, generator creation
  or inference. Field caps, estimate caps, cap equality and deterministic
  80%-warning behavior are covered.
- `/v1/capabilities` is an explicit Pydantic/OpenAPI response, requires the
  inference bearer, does not call readiness or acquire the inference gate, and
  describes fixed controls without disclosing credentials, sensitive paths,
  GPU topology, process IDs or request state.
- `service.sam2` is present at L0-L3 and in ZIP manifests. It contains the
  complete effective/source mappings, requested safe values, profile, exact
  estimates, raw pre-filter candidate count, three-decimal SAM2 duration and
  resource warnings. The raw count remains distinct from the existing L3
  post-remap non-empty candidate count.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 481 passed, 1 intentional GPU-marked test skipped, 79.25% total
  coverage against the 64% gate.
- `.venv/bin/pytest -q tests/test_sam2_configuration.py`: PASSED — 16 tests.
- `.venv/bin/pytest -q tests/test_segmenter_sam2.py tests/test_service_units.py tests/test_service_api.py tests/test_live_runtime.py tests/test_live_service_units.py tests/test_core_engine.py`:
  PASSED — 189 tests.
- `.venv/bin/ruff format --check .`: PASSED — 144 files formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `for script in scripts/*.sh; do bash -n "$script"; done`: PASSED.
- `systemd-analyze verify --user deploy/zap-it-local.service`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED — wheel and sdist member audits succeeded.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  PASSED — no unexpected built-artifact findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  PASSED — exact seven reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `git diff --check`: PASSED.

## CI/checks

All current-head checks for implementation SHA
`27aa21c39752dad6603df458b61141efd807fa04` are `PASSED`/`SUCCESS`:

- `static (format, lint, build)` — CI run `33214019140`.
- `tests (py3.10)` — CI run `33214019140`.
- `tests (py3.11)` — CI run `33214019140`.
- `tests (py3.12)` — CI run `33214019140`.
- `release (artifact audit)` — CI run `33214019140`.
- `Analyze (python)` — CodeQL run `33214019078`.
- `CodeQL` — completed successfully at current head.

## GPU/service/resource evidence

- Authorized physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; application device was
  logical `cuda:0` under the operator mask. No unassigned device or unrelated
  process was modified.
- Exactly one authorized restart of user-level `zap-it-lan.service` was
  performed after implementation/static checks. Cold model loading returned
  readiness from expected HTTP 503 to HTTP 200 after 38 bounded polls; health
  remained 200, and no corrective host mutation was made.
- The final service remained enabled/active/running/ready with PID `426972`,
  `NRestarts=0`, and exactly one listener at `10.8.132.76:17891`. The sole
  assigned-GPU compute process was that service PID.
- Authenticated live checks passed: missing and wrong inference credentials
  returned 401; capabilities, readiness and metrics returned 200; `/docs` and
  `/openapi.json` returned 404. Capabilities matched the ordered defaults,
  profiles and operator maxima without exposing sensitive operator values.
- Consecutive live A/B/A JSON requests on the stable process all returned 200.
  A used explicit 8/8/crop-0 with estimates 64/192; B selected `quality` and
  explicitly repeated points-per-side 32, yielding profile batch 32/crop 1
  and estimates 2048/6144; the final A restored A's exact sources/effective
  values and raw count. Raw candidates were `8, 7, 8`; latencies were
  `8479.3, 2661.2, 185.8` ms; response sizes were `1794, 1748, 1793` bytes.
  GPU monitor peak used memory was 12,951 MiB, minimum free memory 11,173 MiB,
  and peak sampled service RSS was 3,374.1 MiB.
- A separate same-settings crop comparison first returned successful but
  colliding raw counts `8, 8`, then a second bounded fixture returned `10, 10`.
  Following the ordered fallback, the authorized ignored `goats1.jpg` fixture
  was centrally cropped in memory; crop-0 versus crop-1 returned raw counts
  `25` versus `62`, with prompt estimates `64` versus `320`, latencies
  `2160.9` and `5059.9` ms, and response sizes `2495` and `3940` bytes.
- Invalid type and intrinsic-range requests returned `invalid_config` HTTP 400;
  the per-field cap and estimated-work cap requests returned
  `resource_limit` HTTP 413. The inference-duration metric increased only by
  the three accepted A/B/A calls, rejected requests did not enter inference,
  and the model-initialization count remained one across the sequence.
- The request workspace was empty after qualification. The mode-0600 operator
  environment remained unchanged with digest
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`; no key
  value was read into evidence, printed, logged, committed or reported. The
  recent journal contained no traceback and no credential.
- Two local qualification-harness corrections were required before the final
  crop probe: one missing `os` import and two incorrect helper-module import
  assumptions. They occurred before HTTP requests and caused no service or
  host mutation. The first two same-settings probes were successful HTTP
  comparisons whose equal raw counts were recorded as indeterminate, not
  acceptance.

## Documentation/provenance

Documentation now states the exact SAM2 defaults/profiles/ranges, source
precedence, startup caps and formulas, resource errors/warnings, request-local
generator lifecycle, raw-count/timing semantics, authenticated capabilities
contract and fixed operator controls. Model identities, revisions, licenses,
network/auth policy, residency strategy, artifact limits, BLIP3/CLIP behavior,
and the accepted CRIT-0001 disposition were not changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only active order `015-a` was executed.
- Exactly one new PR was created; no merge, auto-merge, release, tag, upload or
  next-order action was performed.
- Only the ordered user-level `zap-it-lan.service` was restarted. No unassigned
  GPU, unrelated process/unit, driver, firewall, route, VPN, global credential,
  or persistent request-data location was modified.
- No raw image/YAML, labels, prompts, answers, credentials or model weights
  entered the report or OAP evidence.

## Limitations/blockers

- The canonical CPU suite intentionally skipped the opt-in GPU-marked test;
  the required bounded private-LAN qualification ran separately on the exact
  assigned GPU and passed.
- The first two same-settings crop comparisons had exact count collisions and
  were not accepted as evidence; the ordered authorized fallback produced the
  decisive 25-versus-62 count difference.
- Live evidence is bounded local research/service evidence, not an SLA,
  semantic-accuracy claim, commercial-license clearance, public exposure or
  release acceptance.

## Factual strategic follow-up

PR #71 remains open and unmerged for strategic review/acceptance.
