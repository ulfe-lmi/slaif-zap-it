# OAP Coding-Agent Report — 020-c

## Work order

- Identifier: `020-c`
- Objective: Objective 020 — bound canonical dry-run score vectors
- PR mode: amend existing numeric-objective PR #76; no new PR
- Order authority: `oap/orders/020-c-bound-canonical-dryrun-score-vectors.md`

## Status

COMPLETE

## Executive summary

Canonical dry-run CLIP scores now use the existing deterministic formula bounded
explicitly to the closed interval `[-1.0, 1.0]` before rounding. A direct CPU
boundary test covers 256 candidates and 32 configured canonical labels,
including ordered complete vectors, finite bounds, winner/score/prompt
consistency, and first/last `ClipRoutingDiagnostic` validation. Legacy
noncanonical dry-run behavior is unchanged. No model, GPU, service, dependency,
network, host, or CRITICAL register work was performed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#76](https://github.com/ulfe-lmi/slaif-zap-it/pull/76), `OPEN`
- PR title: `Objective 020: domain-neutral CLIP routing pipeline`
- Base: `main` at `cc325d5d97acefe7624aecfe9fa157dbf37ce600`
- Starting published report head: `510af456da522b2268b20cac6e0800e6d4dc4e50`
- Starting correction parent: `701397b17a14b85be0a7a07e8de3c6aec4b1bea6`
- Branch: `oap/020-a-domain-neutral-clip-routing-pipeline`
- Implementation head SHA: `d84cb9fd35c6ba6a8718d75a55cbdba5a79e62e0`
- Report publication commit: SELF
- New PR: no
- Amended existing PR: yes
- Coding merge: NO

## Changes/files

Implementation commit `d84cb9fd35c6ba6a8718d75a55cbdba5a79e62e0` contains exactly
these four paths:

- `modules/classifier/clip.py` — bound canonical simulated score values while
  preserving formula ordering, tie-breaking, prompts, and legacy behavior.
- `tests/test_domain_neutral_clip_routing.py` — add the 256-candidate/32-label
  boundary and schema-validation proof.
- `oap/active` — publish `020-c`.
- `oap/orders/020-c-bound-canonical-dryrun-score-vectors.md` — publish the exact
  active order.

No public documentation described the simulated formula or range, so no
documentation file changed. No dependency, schema, preset, service, or model
identity changed.

## Acceptance evidence

- Canonical dry-run score construction clamps every existing deterministic
  per-label value to `[-1.0, 1.0]`; no labels or candidates are omitted.
- `test_canonical_dryrun_score_vectors_are_bounded_at_maximum_size` exercises
  256 candidates and 32 labels, asserts configuration key order, finite bounded
  values, and winner/score/prompt agreement for every vector, and validates the
  first and last vectors with `ClipRoutingDiagnostic`.
- The retained `test_canonical_dryrun_clip_routing_and_blip3_never_initialize_models`
  continues to prove canonical routing and dry-run model non-initialization.
- Existing classifier, routing, schema, core, mask-view, candidate-view, and
  service tests continue to pass. No live semantic accuracy or recall claim is
  made.

## Verification

- `.venv/bin/pytest -q tests/test_domain_neutral_clip_routing.py tests/test_classifier_clip.py tests/test_mask_views.py tests/test_candidate_view_api.py tests/test_core_engine.py tests/test_service_api.py`: `PASSED` — 170 passed, 1 warning.
- `.venv/bin/pytest -q`: `PASSED` — 832 passed, 1 skipped; the skip is the
  opt-in live GPU test because `ZAP_IT_RUN_GPU=1` was not enabled.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: `PASSED` — 832 passed, 1 skipped; total coverage 82.04%, required gate 64%.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current documents.
- `git diff --check` on implementation/test changes: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py dist-from-sdist/*.whl`: `PASSED`.
- `.venv/bin/python scripts/verify_release_artifacts.py --compare-wheels dist/*.whl dist-from-sdist/*.whl`: `PASSED` — no member differences.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz dist-from-sdist/*.whl --baseline .secrets.baseline`: `PASSED` — no unexpected findings.
- `.venv/bin/python -m twine check dist/*`: `PASSED`.
- `.venv/bin/python -m twine check dist-from-sdist/*`: `PASSED`.
- `systemd-analyze verify deploy/zap-it-local.service`: `PASSED`.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: `PASSED` — all seven reviewed findings preserved.
- Direct-wheel isolated `smoke_installed_package.py`: `PASSED` — JSON and ZIP
  package version `0.1.0`, installed module from site-packages.
- Sdist-built-wheel isolated `smoke_installed_package.py`: `PASSED` — JSON and
  ZIP package version `0.1.0`, installed module from site-packages.

## CI/checks

All seven required checks were `PASSED` / `SUCCESS` on the exact implementation
head `d84cb9fd35c6ba6a8718d75a55cbdba5a79e62e0` before report publication:

- `static (format, lint, build)`
- `Analyze (python)`
- `tests (py3.10)`
- `tests (py3.11)`
- `tests (py3.12)`
- `release (artifact audit)`
- `CodeQL`

The same seven checks are required and were verified on the report-only SELF
head before FIFO signaling.

## GPU/service/resource evidence

- Physical GPU index/UUID and visible logical mapping: `SKIPPED` — this order
  expressly prohibited live GPU/model work.
- GPU devices and unrelated processes: untouched; no allocation, reset, or
  process mutation performed.
- Service/port/process continuity: `SKIPPED` — no service launch, restart,
  networking, port mutation, or live inference was authorized.
- Request data and model cache: no request image/YAML, customer data, model
  weights, or persistent request artifact was used. CPU tests used generated
  in-memory data; temporary build/smoke scratch was cleaned.

## Documentation/provenance

No documentation update was needed because current documents do not describe
the simulated score formula/range. The implementation changes only canonical
dry-run score bounding and its direct CPU proof. Repository dependency
declarations and model/service policy are unchanged.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was not modified.

## Safety/scope confirmations

- Only active order `020-c` was executed.
- Existing PR #76 and its Objective 020 branch were amended; no second PR was
  created, and no merge or auto-merge was enabled.
- The immutable prior orders/reports were preserved; the exact 020-c order and
  active marker were included in the implementation commit.
- No credentials, bearer keys, raw inputs, private data, model weights, or
  unnecessary host evidence entered this report.
- No unrelated user files were changed.

## Limitations/blockers

- No live model accuracy, recall, precision, GPU memory qualification, or
  deployment readiness is claimed.
- Optional artifact overflow remains inference-fatal until Objective 021.
- PR #76 remains open for strategic/maintainer review and authorized merge;
  coding did not merge it.

## Factual strategic follow-up

Review PR #76 at the report-only SELF head and apply the separate strategic
merge/acceptance decision. The strongest remaining reason not to merge is that
strategic/maintainer review and merge authority remain outstanding; all seven
required GitHub checks are green and no further coding action is authorized in
this round.
