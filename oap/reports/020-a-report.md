# OAP Coding-Agent Report — 020-a

## Work order

- Identifier/order/objective/PR mode: `020-a` / Domain-neutral CLIP routing pipeline / one new numeric-objective PR (`a`).
- Repository: `ulfe-lmi/slaif-zap-it`.
- Starting SHA: `cc325d5d97acefe7624aecfe9fa157dbf37ce600`.
- Active selector/order transcript SHA-256: `a4dc8eee08c7bc8cd1cf0d63f9848da3babe9c3cd795112fd4daf2bcbb7e5798`.

## Status

COMPLETE

## Executive summary

Implemented the domain-neutral semantic pipeline on the in-memory core and API
boundary. CLIP now receives a source-byte-exact rectangular `raw_bbox_crop`,
returns complete ordered cosine vectors, and feeds a deterministic permissive
router. Routed candidates select request-authored BLIP3 rules, receive the
existing single contextual image, and carry exact normalized answer mapping
evidence. Optional canonical geometry evaluates every candidate and records
bounded losses. All four redistributable shipped YAML examples were migrated;
the ignored local goat presets were also validated locally and remain excluded
by the existing release-artifact policy.

## Authoritative GitHub state

- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/76
- State: OPEN; merge commit: none; coding merge: NO; auto-merge: not enabled.
- Base: `main` at `cc325d5d97acefe7624aecfe9fa157dbf37ce600`.
- Head/implementation head SHA: `3f75035fca95d68b7cfc054aace174ffb64353ec`.
- Remote branch `origin/oap/020-a-domain-neutral-clip-routing-pipeline` equals
  the implementation head.
- Report publication commit: SELF

## Changes/files

- Core/API behavior: `src/core/mask_views.py`, `src/core/routing.py`,
  `src/core/config.py`, `src/core/engine.py`, `src/core/results.py`,
  `src/postprocessing.py`, `src/service/yaml_input.py`,
  `src/service/schemas.py`, `src/service/capabilities.py`,
  `src/service/envelope.py`, `src/service/app.py`,
  `src/service/fake_engine.py`, `src/batch.py`, and package exports.
- Model seams: `modules/classifier/clip.py`, `modules/verifier/blip3.py`,
  and `modules/verifier/__init__.py`.
- Shipped configuration migration: `configs/glasswool.yaml`,
  `configs/icecream.yaml`, `configs/soccer.yaml`, `configs/tomato.yaml`.
- Documentation: `ARCHITECTURE.md`, `README.md`, `TESTING.md`,
  `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, and
  `docs/SERVICE-DATASHEET.md`.
- Governance/test evidence: exact `oap/active` and order transcript,
  `tests/test_domain_neutral_clip_routing.py`, and explicit updates to the
  affected candidate-view, SAM2, and API contract tests. No historical OAP
  order/report or meaningful proof test was deleted.

## Acceptance evidence

1. **Raw CLIP input:** `build_raw_clip_crop` uses inclusive mask bbox and
   half-up `floor(fraction * L + 0.5)` radius, clamps a half-open source slice,
   copies contiguous immutable RGB bytes, and never applies mask/fill/blur or
   resize. The API rejects masked fields/mode; explicit trusted CLI
   `mask_dilated` compatibility remains separate.
2. **Labels/vectors/router:** canonical safe identifier-to-prompt validation
   bounds 32 labels and 512 Unicode codepoints. CLIP aggregates only explicit
   trusted legacy multi-prompts and emits all finite cosine similarities in
   configuration order. Router decisions implement top-1/top-k/margin/
   minimum/uncertainty OR logic, inclusive comparisons, fixed reason
   precedence, and source-ID tie-broken `max_candidates` capping.
3. **Geometry:** canonical optional area, inclusive width/height,
   aspect-ratio, and border rules retain equality, evaluate empty/non-empty
   candidates, preserve first-reason precedence, and report source ID, nullable
   bbox, area, dimensions, reason, configured limit, and bounded truncation.
   Legacy `maxsize`/`max_w`/`max_h` inputs remain explicit compatibility
   aliases with migration warnings and canonical/alias conflict rejection.
4. **BLIP3:** the selected routing target chooses the matching rule even when
   another label wins CLIP. The accepted `single_dilated_blur` compositor and
   fixed generic instruction are unchanged. Each canonical question gets one
   contextual image and a structured record containing configured/effective
   questions, raw/normalized answer, exact true/false mapping, outcome, and
   final label; unmatched answers conservatively use `falsecategory`.
5. **Stable pipeline evidence:** source candidate IDs, filtered indices,
   candidate counts, stage statuses, timings, CLIP diagnostics, geometry
   diagnostics, BLIP records, object records, JSON, and ZIP manifest retain the
   relevant stage loss and agree at the tested levels.
6. **SAM2/resource provenance:** existing request/effective/source/profile,
   estimate/count/time/warning behavior is retained. Accepted requests now
   carry operator-limit values and per-field provenance with
   `operator_limit_applied: false`; requests over caps are rejected rather than
   clamped.
7. **Configuration/docs:** all four tracked shipped examples pass the hostile
   parser at verbosity 0 and 3 with default ServiceSettings/caps, use
   `raw_bbox_crop`, canonical routing, matching BLIP3 rules, terminal labels,
   and no forbidden service controls.

## Verification

- `.venv/bin/pytest -q --disable-warnings`: PASSED — 816 passed, 1 honest
  opt-in GPU skip, 2 warnings.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 816 passed, 1 skip; total coverage 81.63%, required 64.0% gate.
- Focused raw-crop/routing/geometry/BLIP/config tests:
  PASSED — 22 new focused tests; affected service/candidate suites 144 passed.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `python scripts/check_documentation.py`: PASSED — 27 current documents.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `scripts/verify_release_artifacts.py` on direct wheel/sdist:
  PASSED — wheel 68 members, sdist 162 members.
- sdist extraction, sdist-built wheel, archive inspection, and
  `--compare-wheels`: PASSED — equal 68-member manifests.
- `.venv/bin/python -m twine check` on direct and sdist-built artifacts:
  PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: PASSED — exactly 7 reviewed findings.
- Archive secret scans on direct wheel, sdist, and sdist-built wheel:
  PASSED — 0 findings outside the reviewed baseline.
- Direct installed-package smoke outside the checkout: PASSED — JSON/ZIP and
  console script checks.
- Sdist-built installed-package smoke outside the checkout: PASSED — JSON/ZIP
  and console script checks.
- Warnings: 2 expected test deprecation warnings and expected setuptools
  metadata deprecation warnings during artifact builds; no test failure.

## CI/checks

Implementation head `3f75035fca95d68b7cfc054aace174ffb64353ec` had all seven
remote checks PASS before report creation:

- CI `static (format, lint, build)`: PASS, run `33396945140`.
- CI `tests (py3.10)`: PASS, run `33396945140`.
- CI `tests (py3.11)`: PASS, run `33396945140`.
- CI `tests (py3.12)`: PASS, run `33396945140`.
- CI `release (artifact audit)`: PASS, run `33396945140`.
- CodeQL `Analyze (python)`: PASS, run `33396945005`.
- CodeQL report check: PASS, check run `99503836384`.

The final report-head matrix is required and will be inspected at the pushed
SELF report head before FIFO signaling; no claim about that future head is
made in this pre-publication evidence.

## GPU/service/resource evidence

- No GPU phase, model download, live inference, service restart, reload,
  reconfiguration, or host mutation was authorized or performed.
- Read-only before/after service state: `zap-it-lan.service` user unit active/
  running, PID `607106`, `NRestarts=0`, start `2026-08-31 12:36:38 CEST`;
  health HTTP 200 and readiness HTTP 200.
- One listener remained `10.8.132.76:17891`; no other listener was changed.
- Assigned physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`; logical service mapping
  remains operator-masked `cuda:0`. Read-only process evidence remained Xorg
  graphics plus service PID compute at approximately 12010 MiB; no unassigned
  device was touched.
- `/dev/shm`: approximately 11 GiB available at the final check;
  `/dev/shm/slaif-zap-it` remained mode 0700 owned by `janezp:users`.
- Operator environment file was not printed or read into evidence; its verified
  mode-0600 SHA-256 remained `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- CPU/API tests found no persistent request files. Packaging checks used only
  generated release artifacts and did not alter the service workspace.

## Documentation/provenance

The architecture, API, configuration, core, algorithm, output-parity, runbook,
datasheet, README, and testing documents describe the raw-crop semantics,
prompt/value split, vector units, routing logic/reasons/cap, geometry records,
BLIP exact mapping, stage order, IDs/counts/timings, schema/capability fields,
verbosity gating, and retained trusted compatibility. They also state that
deterministic CPU/fake tests do not prove semantic accuracy/recall and that
Objective 021 owns structured optional-artifact truncation/pagination.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Exactly one active order `020-a`, one new branch, and one PR for numeric
  objective 020; no merge or auto-merge.
- Only the exact active-order selector/order transcript was published with the
  implementation. Historical OAP files remain unchanged.
- No model identity/revision/weight, device, dtype, cache, residency,
  generation limit, auth/network/unit/key, port, concurrency, dependency,
  SAM2 generator, BLIP3 compositor, renderer pixels, YOLO format, release
  state, CRITICAL register, or protected host resource was changed.
- No API key, raw image/YAML, prompt, answer, customer data, model asset, or
  unnecessary private path entered the OAP report.

## Limitations/blockers

- This order intentionally does not add live semantic-accuracy evidence; the
  one opt-in GPU test remains SKIPPED because live GPU work was forbidden.
- Objective 021 still owns optional artifact selection/pagination, admissible
  SAM2 alternatives, and changing artifact overflow from inference-fatal to
  structured truncation. Those behaviors remain unchanged here.
- External deployment, public/WAN exposure, TLS, and release acceptance remain
  outside this coding order.

## Factual strategic follow-up

Strategic may inspect PR #76 and the final report-head checks. The strongest
technical merge risk is a fabricated-looking routing manifest that does not
correspond to literal CLIP processor pixels, complete vectors, or candidates
that actually reach BLIP3. The independent generated-array crop oracle,
processor/QA seams, branch-complete router tests, stable-ID records, and JSON/
ZIP checks provide the bounded CPU evidence available under this no-GPU order;
they do not substitute for later semantic qualification.
