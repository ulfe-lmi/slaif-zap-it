# OAP Coding-Agent Report — 016-c

## Work order

- Identifier/order/objective/PR mode: `016-c` — enforce every represented raw
  SAM2 candidate ID bound; amend Objective-016 PR #72.

## Status

COMPLETE

## Executive summary

Corrected the pure raw-SAM2 manifest validator so the one-based upper bound is
checked independently for every `represented_candidate_ids` entry, including a
singleton and the first entry of a longer sequence. The Pydantic schema already
reused this validator, so it now rejects the same inconsistency. Added focused
core and schema proofs for the rejected singleton, accepted singleton, and
accepted gapped IDs. No renderer, API, resource, model, GPU or service behavior
changed.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/72`
- PR state: OPEN, non-draft, MERGEABLE; coding did not merge or enable
  auto-merge.
- Base branch: `main` at
  `8081152403657f5e737ab0b491e0b89f587209e1`.
- Starting PR/report head SHA:
  `2f6b1364dff76cb589c2dd87fa56b88ce5f0ca19`.
- Branch: `oap/016-a-bounded-raw-sam2-visualizations`.
Implementation head SHA: `abd0e6e5b0731e3576fa5ddefa2822ce2b964e07`
Report publication commit: SELF
- New PR: no. Amended existing PR: yes. Coding merge/auto-merge: NO.

## Changes/files

Implementation commit `abd0e6e5b0731e3576fa5ddefa2822ce2b964e07` contains only:

- `oap/active`
- `oap/orders/016-c-enforce-all-represented-candidate-id-bounds.md`
- `src/core/raw_visualizations.py`
- `tests/test_raw_sam2_visualizations.py`

The source change separates the per-entry upper-bound check from the adjacent
pair ordering check. The exact activated selector and order transcript were
committed with the implementation as required. No product documentation,
public schema shape, capability payload, limits or dependency changed.

## Acceptance evidence

1. **Singleton upper bound — PASSED.** A valid rendered result with
   `raw_candidate_count == 1` and `represented_candidate_ids == [1]` was
   accepted. Tampering its sole ID to `[2]` was rejected by
   `validate_raw_sam2_manifest` with the fixed sanitized
   `raw SAM2 visualization manifest is inconsistent` `CoreError`.
2. **Schema seam — PASSED.** `RawVisualizationManifest` rejected the same
   singleton `[2]` tampering through its reused core validator, without
   exposing the invalid value or request data.
3. **Gapped IDs — PASSED.** A valid rendered result with raw count 3,
   represented IDs `[1, 3]`, and one omitted empty candidate remained accepted
   by both the pure validator and the Pydantic schema; IDs are not incorrectly
   required to be consecutive.
4. **Regression preservation — PASSED.** The complete focused Objective-016,
   API, core and legacy set remained green, including raw visualization
   rendering, JSON/ZIP parity, fixed names, pagination, limits, resource
   admission and legacy behavior.

## Verification

- `.venv/bin/pytest -q tests/test_raw_sam2_visualizations.py`: PASSED — 37
  focused raw renderer/API/resource/schema tests.
- `.venv/bin/pytest -q tests/test_raw_sam2_visualizations.py tests/test_sam2_configuration.py tests/test_service_api.py tests/test_service_units.py tests/test_core_engine.py tests/test_run_frame_pipeline.py`: PASSED — 389 focused Objective-016/API/core/legacy tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 726 passed, 1 explicit opt-in GPU test skipped, 80.08% total coverage. The skip was `tests/test_gpu_integration.py:20` because live GPU tests require `ZAP_IT_RUN_GPU=1`; this CPU-only order did not enable them.
- `.venv/bin/ruff format --check .`: PASSED — 146 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh scripts/serve_local.py scripts/smoke_local_service.py`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED before commit and after implementation commit.
- `.venv/bin/python -m build --wheel --sdist`: PASSED; setuptools emitted existing license-metadata deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED — wheel and sdist manifests verified.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — zero unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exactly seven reviewed baseline findings, unchanged.
- `.venv/bin/python -m twine check dist/*`: PASSED — wheel and sdist.
- `git status --short --branch`: PASSED — clean after the implementation commit.

One exploratory command, `gh pr view 72 --json number,state,baseRefName,baseRefOid,headRefName,headRefOid,mergeStateStatus,reviewDecision,statusCheckRollup,url`, was FAILED by the installed `gh` because `baseRefOid` is unsupported. It caused no mutation; the supported PR query was rerun successfully and verified the same open PR and head.

## CI/checks

All seven required checks are SUCCESS/PASSED on implementation SHA
`abd0e6e5b0731e3576fa5ddefa2822ce2b964e07`:

- `static (format, lint, build)` — PASSED, CI run `33225783599`, job
  `99029156540`.
- `tests (py3.10)` — PASSED, CI run `33225783599`, job `99029156451`.
- `tests (py3.11)` — PASSED, CI run `33225783599`, job `99029156561`.
- `tests (py3.12)` — PASSED, CI run `33225783599`, job `99029156580`.
- `release (artifact audit)` — PASSED, CI run `33225783599`, job
  `99029156566`.
- `Analyze (python)` — PASSED, workflow run `33225783593`, job
  `99029156312`.
- `CodeQL` — PASSED, check run `99029283796`.

The same seven required checks were reverified after publication on the final
report-only head: `static (format, lint, build)`, `tests (py3.10)`,
`tests (py3.11)`, `tests (py3.12)`, `release (artifact audit)`,
`Analyze (python)` and `CodeQL` — all PASSED.

## GPU/service/resource evidence

- This was a CPU-only validator correction. No GPU inference was run and
  `zap-it-lan.service` was not restarted, stopped, reloaded or reconfigured.
- Before and after the work, the user service was enabled/active/running with
  stable PID `476019`, `NRestarts=0`, `readyz=200`, and `healthz=200`.
- The private-LAN listener remained exactly one listener at
  `10.8.132.76:17891`; the assigned GPU had exactly one compute process before
  and after.
- The assigned target remained physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, NVIDIA GeForce RTX 3090,
  `24576 MiB`, driver `610.43.02`. The service environment retained
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`, exposing that
  physical card as logical `cuda:0`; no unassigned device was touched.
- The after-work assigned-card process snapshot was `11594 MiB` used by the
  single service compute process; service RSS was `4,556,148 KiB`.
- `/dev/shm/slaif-zap-it` remained mode `0700` with zero entries after the
  checks. No request data or credentials were printed, persisted or added to
  OAP evidence.

## Documentation/provenance

No documentation change was needed: the documented invariant already required
every represented candidate ID to be one-based and no greater than the raw
candidate count. No model identity/revision, license, security posture,
auth/network/device/cache policy, CRIT-0001 or dependency changed.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Exactly active order `016-c` was executed; no adjacent order was selected.
- Exactly PR #72 was amended for numeric Objective 016; no new PR, merge,
  auto-merge, release/tag, history rewrite, network/firewall/VPN change,
  unrelated service change or protected-process mutation occurred.
- All non-report implementation state was pushed before this report was
  prepared. The final report commit is intended to change only this report and
  to have the implementation SHA as its sole parent.

## Limitations/blockers

None for the ordered scope. The CPU suite's opt-in GPU test was not run, as
required for this CPU-only correction. The report does not claim segmentation
quality, production readiness or deployment authorization.

## Factual strategic follow-up

PR #72 remains open with the implementation checks green. Coding has not
merged it, enabled auto-merge, selected Objective 017, authorized deployment or
performed any post-report mutation.
