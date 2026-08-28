# OAP Coding-Agent Report — 013-a

## Work order

- Identifier/order/objective/PR mode: `013-a` / mask-aware BLIP3 verification / one new branch and PR
- Branch: `oap/013-a-mask-aware-blip3-verification`
- Base: `main`

## Status

COMPLETE

## Executive summary

Implemented deterministic mask-aware BLIP3 verification. Each executed question
now receives one cached RGB pair containing untouched context on the left and
an exact mask spotlight on the right, with fixed nearest-neighbor scaling,
exterior-only dimming, and a component-following yellow contour. The fixed
region-specific instruction follows the bounded client question. Verbosity-3
debug stores only the exact paired image passed to QA as a safe lossless PNG;
JSON and ZIP carry identical bytes and descriptors. Existing relabeling,
request-local rules, model holder, question/token bounds, and CLI compatibility
remain intact.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#69](https://github.com/ulfe-lmi/slaif-zap-it/pull/69)
- PR state: OPEN
- PR title: `Objective 013: mask-aware BLIP3 verification`
- Base/head: `main` / `oap/013-a-mask-aware-blip3-verification`
- Starting checkout SHA: `1f15646cd2d991547c8c29182aa9fdebbb9dedd8`
- Verified remote base SHA: `43fcfe99b47545b218a70338f02c01f69b35a29e`
- Implementation head SHA: `7e241ac59a5234648a884c3559dc8609144ee66d`
- Report publication commit: SELF
- New PR: yes; amended existing: no; coding merge: NO

## Changes/files

- Added reusable `Blip3VerificationComposition`, deterministic crop/scale
  metadata, paired-image composer, fixed query composer, and exact spotlight
  pixel/contour rules in `modules/verifier/blip3.py`.
- Passed the core artifact sink to BLIP3 only for effective rule debug and
  passed service-safe naming through the core.
- Sanitized all nested BLIP3 debug flags below L3 with one bounded aggregate
  warning in `src/service/yaml_input.py`.
- Replaced BLIP3 JPEG/answer-text debug output with fixed numeric PNG names for
  service and sanitized frame-stem/numeric PNG names for trusted CLI use.
- Added composer, pixel, positive/hard-negative, rule reuse, budget, safe-name,
  sink, YAML, and JSON/ZIP parity tests.
- Updated the active transcript and exact immutable order bytes, plus the
  order-named API/core/algorithm/operator documentation.

## Acceptance evidence

- Exact mask inputs: `PASSED` — RGB `uint8` image and boolean mask shape/type
  mismatches and empty masks fail explicitly.
- Crop and transform: `PASSED` — complete-mask bbox, symmetric bounded padding,
  128-pixel minimum desired extents, border/back-shift/span cases, positive
  dimensions, 256-short-side targeting, 768-long-side cap, and shared
  nearest-neighbor indices are tested.
- Paired pixels: `PASSED` — dimensions are `(scaled_height, 2*scaled_width+4, 3)`;
  selected right pixels equal the left; non-contour exterior pixels retain
  floor(`2*channel/5`); the four-pixel exterior square dilation ring is yellow
  and never overwrites selected pixels.
- Verification instruction/QA: `PASSED` — the bounded target is delimited and
  precedes the fixed final instruction; `any,<threshold>` and label rules pass
  the same cached PIL image/query seam; true/false/newcategory behavior remains
  covered.
- Roof/panel semantics: `PASSED` — injected QA inspecting only the spotlight
  retains a selected panel and rejects a roof mask when panel pixels remain
  elsewhere in the same context crop; the ordinary crop contains those
  elsewhere pixels, proving this is not a bbox-presence check.
- Debug artifacts: `PASSED` — only L3 effective debug emits fixed
  `blip3-verification-####-####.png` images; arrays equal the QA input, names
  exclude labels/questions/answers/path fragments, no answer-text duplicate is
  emitted, and trusted CLI names remain safe.
- JSON/ZIP/determinism: `PASSED` — repeated arrays/PNG encoding are identical;
  JSON and ZIP artifact bytes, sizes, and SHA-256 descriptors match.
- Existing behavior: `PASSED` — focused and complete CPU suites remain green;
  32-question preflight and 32-token service bounds remain enforced.

## Verification

- `.venv/bin/pytest -q tests/test_verifier_blip3.py tests/test_adaptive_residency.py tests/test_core_engine.py tests/test_service_units.py tests/test_service_api.py`: PASSED — 151 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 458 passed, 1 skipped; total coverage 78.74%, required 64.0% reached. The one skip is the opt-in GPU unit marker; live service qualification below ran separately.
- `.venv/bin/ruff format --check .`: PASSED — 142 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: PASSED — 64 wheel members and 153 sdist members audited.
- Direct wheel versus wheel rebuilt from the sdist with `scripts/verify_release_artifacts.py --compare-wheels`: PASSED — member names/bytes equal.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: PASSED — zero new findings; tracked baseline remains exactly seven reviewed findings.
- `.venv/bin/twine check dist/zap_it-0.1.0-py3-none-any.whl dist/zap_it-0.1.0.tar.gz`: PASSED.
- Shell syntax: NOT RUN — no shell files changed.

## CI/checks

All checks below are successful at implementation SHA
`7e241ac59a5234648a884c3559dc8609144ee66d`:

- CI run `33198366363`: `static (format, lint, build)` SUCCESS; `tests (py3.10)` SUCCESS; `tests (py3.11)` SUCCESS; `tests (py3.12)` SUCCESS; `release (artifact audit)` SUCCESS.
- CodeQL workflow run `33198366408`: `Analyze (python)` SUCCESS.
- CodeQL check `98941462604`: SUCCESS.

## GPU/service/resource evidence

- Preflight and final live snapshots matched the active order: physical GPU
  index `0`, UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24,576 MiB, driver
  `610.43.02`; the application saw only logical `cuda:0`.
- Only the assigned GPU process was present throughout qualification. The
  service remained one enabled process, one inference slot, and one listener
  at `10.8.132.76:17891`; PID `402706` and the listener stayed unchanged across
  the qualified requests. `NRestarts=0`.
- The one authorized restart was only `zap-it-lan.service`. It initially
  returned expected `503 not_ready` responses while the resident checkpoint
  loaded, then returned `200 ready` without a restart loop.
- Final live qualification: `PASSED` — L3 JSON 200, L3 ZIP 200, repeated L3
  JSON 200; one final object with one bounded BLIP3 answer and five fixed-name
  debug PNGs; JSON response sizes 18,805,076 and 18,805,077 bytes for the two
  JSON calls, ZIP 14,064,722 bytes; JSON/ZIP/repeat name and digest checks
  passed.
- Live artifact aggregate evidence: ordered-name digest
  `63d5de1f1a79abc69c5e19fda2e563123c7d94c20d0d2c57b09822973980d00a`;
  ordered-PNG-digest aggregate
  `4b06479876650f2f1d95b579393c451f2c58d2f0f13a80ac7f0be45951f232ed`.
- Live resource evidence: finite metrics; peak reserved logical Torch CUDA
  memory `13,841,203,200` bytes; host RSS maximum `13,389,393,920` bytes;
  `/dev/shm` request workspace `(0 files, 0 bytes)` after requests. The
  service's private root remained clean.
- Auth/docs: `PASSED` — missing key 401, wrong key 401, correct key readiness/
  metrics access 200, `/docs` 404, `/openapi.json` 404.
- Operator environment remained mode 0600 with unchanged pre/post file and key
  digests; no key value was printed, logged, committed, or included here.

Two live attempts were unsuccessful and are disclosed:

1. The first sanitized goat request returned HTTP 200 but selected no executed
   final answer/artifact because the chosen existing rule did not match an
   executed final candidate. Status: FAILED.
2. A corrective bounded `any` probe executed five questions and produced five
   paired artifacts, but its answer substrings relabelled all candidates away
   from the final object list. Status: FAILED.

The corrective action was an in-memory one-rule `any` mapping with debug
enabled and no relabel substrings. The final three-request qualification then
passed. No second service/process, unrelated unit, device, driver, firewall,
route, VPN, key, or persistent request-data location was changed.

## Documentation/provenance

Updated `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/ALGORITHMS.md`, `docs/CORE.md`,
`docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, and
`docs/SERVICE-DATASHEET.md` to describe the mask-aware transform, fixed
instruction, verbosity-3 names/limits, structured-answer independence, and
accuracy limitation. No model identity, revision, license, dtype, residency
claim, dependency, or remote asset changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only active order `013-a` was executed.
- The exact active transcript and order were carried into the implementation
  commit unchanged; prior orders/reports were not rewritten.
- Implementation and transcript state was pushed before report creation.
- No merge or auto-merge was enabled.
- No raw image, YAML, prompt, label, answer, credential, model weight, or
  customer data entered the repository report or OAP evidence.
- No filesystem request persistence was introduced; the live ignored fixture
  remained outside Git, OAP, and chat.

## Limitations/blockers

- The live goat request is pinned-model integration/resource/artifact evidence,
  not a solar-panel accuracy benchmark or a universal BLIP3 semantic-accuracy
  claim.
- The opt-in GPU pytest marker remains skipped in the CPU canonical suite; the
  order-authorized private-LAN qualification supplied the live GPU evidence.
- No blockers remain within this order. Coding does not merge or declare
  acceptance.

## Factual strategic follow-up

- Strategic review/acceptance of PR #69 remains outstanding under OAP
  governance. The PR is intentionally left OPEN and unmerged.
