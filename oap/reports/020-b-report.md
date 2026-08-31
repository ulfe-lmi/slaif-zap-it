# OAP Coding-Agent Report — 020-b

## Work order

- Identifier: `020-b`
- Objective: Objective 020 — close raw-CLIP proof and synchronize documentation
- PR mode: amend existing numeric-objective PR #76; no new PR
- Order authority: `oap/orders/020-b-close-raw-clip-proof-and-doc-sync.md`

## Status

COMPLETE

## Executive summary

The correction freezes the actual post-CLIP candidate count before routing and
BLIP3 mutation, makes canonical dry-run CLIP emit complete deterministic score
vectors and configured prompts without model work, and closes the generated
response schema for canonical geometry aggregates. Focused CPU proof now crosses
the actual raw `raw_bbox_crop` CLIP processor seam, covers prompt/routing and
clear-negative branches, verifies JSON/ZIP parity, and exercises canonical
dry-run routing. Current documentation was synchronized across the required
user/operator documents. No live inference, GPU, model cache, service,
networking, or CRITICAL register action was performed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#76](https://github.com/ulfe-lmi/slaif-zap-it/pull/76), `OPEN`
- PR title: `Objective 020: domain-neutral CLIP routing pipeline`
- Base: `main` at `cc325d5d97acefe7624aecfe9fa157dbf37ce600`
- Starting published report head: `cecaaf1d4e2447147fbdea86fccf8e6eab27525d`
- Starting implementation parent: `3f75035fca95d68b7cfc054aace174ffb64353ec`
- Branch: `oap/020-a-domain-neutral-clip-routing-pipeline`
- Implementation head SHA: `701397b17a14b85be0a7a07e8de3c6aec4b1bea6`
- Report publication commit: SELF
- New PR: no
- Amended existing PR: yes
- Coding merge: NO

## Changes/files

Implementation commit `701397b17a14b85be0a7a07e8de3c6aec4b1bea6` contains exactly
the following 21 paths:

- `ARCHITECTURE.md`
- `README.md`
- `TESTING.md`
- `docs/ALGORITHMS.md`
- `docs/API.md`
- `docs/CONFIG.md`
- `docs/CORE.md`
- `docs/OUTPUT-PARITY.md`
- `docs/RUNBOOK.md`
- `docs/SERVICE-DATASHEET.md`
- `modules/classifier/clip.py`
- `oap/active`
- `oap/orders/020-b-close-raw-clip-proof-and-doc-sync.md`
- `src/core/engine.py`
- `src/service/fake_engine.py`
- `src/service/schemas.py`
- `tests/test_candidate_view_api.py`
- `tests/test_core_engine.py`
- `tests/test_domain_neutral_clip_routing.py`
- `tests/test_mask_views.py`
- `tests/test_service_api.py`

Behavior and compatibility:

- `after_clip`, `clip_scored`, and CLIP stage detail use the count captured
  immediately after CLIP returns; routing, BLIP3, and final counts remain
  independent.
- Canonical dry-run CLIP uses the effective label order and one configured
  prompt per safe identifier, with finite complete vectors and deterministic
  winner/prompt values. Empty/noncanonical trusted dry-run behavior remains the
  historical `dryrun region N` behavior.
- `PostFilterDiagnostics` retains legacy aggregate spellings and adds all
  canonical runtime removal counters. Its generated model rejects unknown
  fields and contradictory aggregate accounting. `ServiceMetadata` now models
  the emitted package version, and candidate-view records retain effective view
  configuration.
- API canonical CLIP remains one natural-language prompt per safe identifier
  and untouched raw crop; trusted CLI multi-prompt, flattened-key, and masked
  compatibility behavior remains available where previously supported.
- Objective 021 artifact truncation/resource alternatives were not implemented;
  optional artifact overflow remains inference-fatal.

## Acceptance evidence

- A — `test_clip_count_is_frozen_before_clear_negative_routing_and_blip3` proves
  two candidates were scored while one clear negative was excluded from routing
  and BLIP3; the CLIP stage detail remains `2 -> 2`, and routed/verified/final
  counts are `1`.
- B — `test_canonical_dryrun_clip_routing_and_blip3_never_initialize_models`
  exercises canonical CLIP, routing, and BLIP3 dry-run flow with no model
  initialization and complete configured score vectors.
- C — `test_real_clip_classify_single_receives_literal_raw_bbox_processor_view`
  uses the actual `_ClipFilter.filter_masks` and `classify_single_scores` seam.
  It independently checks half-up radius, source-clamped half-open crop,
  holed-mask/surrounding source bytes, immutable inputs, fixed IDs/metadata,
  complete cosine scores, bounded sink pixels, and decoded lossless PNG parity.
  The old masked builder remains covered by its separately named legacy test.
- D — focused routing tests cover inclusive top-1, top-k, margin, minimum,
  explicit uncertain winner, clear negative, cap/tie-break behavior, exact
  natural-language prompt bytes, validator type/range and missing/orphan rule
  cases, and precise invalid/unsupported codes. The API proof preserves both
  complete vectors and exact reasons/cap outcomes in JSON and ZIP, including a
  clear negative that never reaches the BLIP3 seam.
- E — OpenAPI assertions cover canonical post-filter aggregate fields, complete
  routing score/reason fields, BLIP3 configured/effective answer mapping,
  package metadata, counts, and timings. `PostFilterDiagnostics` model tests
  verify field preservation and rejection of unknown/contradictory data.
- Documentation — current contract wording was reviewed and synchronized in
  the required README, architecture, API, configuration, algorithms, core,
  output-parity, runbook, service-datasheet, and testing documents. Stale
  top-label/one-or-more-prompt/implicit-negative/old-instruction claims were
  removed or explicitly scoped to trusted legacy CLI compatibility.

## Verification

- `.venv/bin/pytest -q tests/test_domain_neutral_clip_routing.py tests/test_classifier_clip.py tests/test_mask_views.py tests/test_candidate_view_api.py tests/test_core_engine.py tests/test_service_api.py`: PASSED — 169 passed, 1 warning.
- `.venv/bin/pytest -q`: PASSED — 831 passed, 1 skipped; the skip is the
  opt-in live GPU test because `ZAP_IT_RUN_GPU=1` was not enabled.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 831 passed, 1 skipped; total coverage 82.03%, required gate 64%.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `scripts/verify_release_artifacts.py` on direct wheel and sdist: PASSED.
- `scripts/verify_release_artifacts.py --compare-wheels` on direct and
  sdist-built wheels: PASSED — no member differences.
- `scripts/scan_release_artifacts.py` on wheel/sdist/sdist-built wheel:
  PASSED — no unexpected findings.
- `.venv/bin/python -m twine check` on direct and sdist-built distributions:
  PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  PASSED — all seven reviewed baseline findings preserved, no additions.
- Direct-wheel isolated `smoke_installed_package.py`: PASSED — JSON and ZIP
  package version `0.1.0`, installed module imported from site-packages.
- Sdist-built-wheel isolated `smoke_installed_package.py`: PASSED — JSON and
  ZIP package version `0.1.0`, installed module imported from site-packages.

## CI/checks

All seven required checks were successful on the exact implementation head
`701397b17a14b85be0a7a07e8de3c6aec4b1bea6` before report publication:

- `static (format, lint, build)`: PASSED / `SUCCESS`
- `Analyze (python)`: PASSED / `SUCCESS`
- `tests (py3.10)`: PASSED / `SUCCESS`
- `tests (py3.11)`: PASSED / `SUCCESS`
- `tests (py3.12)`: PASSED / `SUCCESS`
- `release (artifact audit)`: PASSED / `SUCCESS`
- `CodeQL`: PASSED / `SUCCESS`

The report-only SELF child is required to rerun the same seven checks; they
were verified on the final report head before FIFO signaling.

## GPU/service/resource evidence

- Physical GPU index/UUID and visible logical mapping: SKIPPED — this order
  expressly prohibited live GPU/model work.
- GPU0 and all other unassigned devices/processes: not touched; no allocation,
  reset, or process mutation performed.
- Service/port/process continuity: SKIPPED — no service restart, launch,
  reconfiguration, live inference, host networking, or port mutation was
  authorized.
- Model cache, `/dev/shm`, and request artifacts: no live request data or model
  cache was touched. Temporary release/build scratch was removed; no repository
  request-data residue remains.
- CPU/fake tests used generated in-memory arrays only and do not establish live
  semantic-model accuracy.

## Documentation/provenance

The canonical service contract is now documented as one natural-language prompt
per safe identifier, complete ordered CLIP cosine vectors on every surviving
geometry candidate, permissive routing for BLIP3 admission, exact normalized
answer mapping to configured `newcategory`/`falsecategory`, and separate raw
CLIP versus composed BLIP3 debug identity. Objective 021's overflow limitation
remains explicit. No dependency, model identity, revision, residency, or
service policy was changed.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was not modified.

## Safety/scope confirmations

- Only active order `020-b` was executed.
- Existing PR #76 and its numeric Objective 020 branch were amended; no second
  PR was created, and no merge or auto-merge was enabled.
- The immutable 020-a order/report were preserved.
- No bearer, credential, raw image, raw YAML, model weights, customer data, or
  unnecessary private host data entered this report.
- No unrelated user files were changed.

## Limitations/blockers

- No live model accuracy, recall, precision, GPU memory qualification, or
  deployment readiness is claimed.
- Optional artifact overflow is still inference-fatal until Objective 021.
- PR #76 remains open for strategic/maintainer review and authorized merge;
  coding did not merge it.

## Factual strategic follow-up

Review the amended PR #76 at the final report head and apply the separate
strategic merge/acceptance decision. The strongest remaining non-technical gate
is maintainer/strategic review and merge authority; no additional coding action
is authorized in this round.
