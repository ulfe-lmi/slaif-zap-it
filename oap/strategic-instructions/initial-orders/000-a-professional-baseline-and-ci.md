# OAP Work Order — 000-a — Professional baseline and CI

> DRAFT UNTIL strategic live/GitHub reconnaissance is complete. Do not publish.

## Objective

Bring the existing ZAP-IT repository to an honest professional baseline: preserve
and characterize current behavior; establish reproducible package/dev setup;
audit, repair and extend CPU tests; enable CI and CodeQL; refresh core repository
documentation/security/provenance. Do not implement the service API or mutate GPU
runtime in this objective.

## GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `000 / 000-a`
- Mode: `CREATE_NEW_PR`
- Verified default branch and 40-hex SHA: VERIFY:
- Required new branch name: VERIFY:
- Existing objective PR: N/A after strategic confirms none: VERIFY:
- Required PR title: VERIFY:

## Verified current state

Replace with evidence before activation:

- current repository/test/package/CI/docs state: VERIFY:
- exact existing CPU test command/result/count/failures: VERIFY:
- current Python/dependency constraints and safe CI version: VERIFY:
- current GitHub Actions/CodeQL/branch-protection facts: VERIFY:
- local dirty/unrelated work preservation plan: VERIFY:
- all GPUs/processes summarized read-only; objective uses no GPU: VERIFY:

## Scope

1. Inventory current public CLI, config grammar, modules, outputs, tests and docs;
   add a concise baseline record where useful.
2. Add modern `pyproject.toml`/package metadata and dev-test installation path
   compatible with current code, without breaking existing entry points.
3. Make existing CPU/mocked tests deterministic and runnable without CUDA, model
   download, network, private data or repository result mutation.
4. Fix genuine in-scope defects exposed by tests; do not replace behavior with
   mocks merely to pass.
5. Add focused missing tests for package/import/entrypoint/config and current pure
   pipeline boundaries proportionate to baseline gaps.
6. Add Ruff formatting/lint policy, pytest config and measured coverage report;
   set a defensible initial threshold/ratchet rather than an invented high number.
7. Add least-privilege GitHub CI for package/static/CPU tests and CodeQL. Pin
   standard action major versions; no secrets/GPU/model downloads.
8. Refresh README, installation, configuration/algorithm navigation and testing
   instructions to match verified commands/limitations.
9. Add/refresh `CONTRIBUTING.md`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md` and
   provenance/dependency/model notes while preserving the existing MIT license.
10. Audit generated/legacy artifacts such as `everything.txt`, `last_results`,
    compatibility shims and typo-named docs; remove/rename only with evidence and
    compatibility/migration explanation.

## Non-goals

- no `/v1/completions`, FastAPI, service schemas, Docker/systemd or API key;
- no in-memory core refactor beyond minimal package/test seams;
- no model download, real inference or GPU test;
- no CUDA/driver/environment/system package/firewall/port/service changes;
- no physical GPU0 or GPU1 allocation;
- no scientific algorithm/threshold/default change unless required to fix a
  clearly demonstrated bug and explicitly documented/tested;
- no model weights/results corpus committed;
- no adjacent roadmap implementation.

## Acceptance criteria

1. A clean clone can install the documented CPU/dev test environment using the
   chosen supported Python version without installing/downloading GPU models.
2. One canonical CPU command runs the complete existing+new CPU suite; all tests
   pass, count and duration reported; no network/GPU/model dependency.
3. Ruff format/check and package build/import/CLI smoke pass.
4. Coverage is measured, exclusions explained and an initial non-regressive gate
   enforced or a precise staged ratchet documented if immediate enforcement is
   technically unsafe.
5. GitHub CI runs on PR and main, least permissions, and executes the canonical
   static/package/CPU suite. CodeQL is enabled and green or exact repository-
   policy blocker reported.
6. Existing supported CLI/config behavior remains covered and documented; any
   intentional compatibility change is explicit.
7. README/install/config/testing/security/contributing/provenance/third-party docs
   are consistent with actual code and do not claim API or GPU readiness.
8. No large generated/model/cache/private artifact or secret enters the diff.
9. No GPU process, listener, system service, firewall, global OpenCode config or
   unrelated host state is changed.
10. Correct branch/one PR exists, required checks pass, coding agent never merges,
    and immutable report-only SELF child is remote head before response signal.

## Required verification

Strategic replaces commands after audit; minimum categories:

- clean CPU/dev install: VERIFY:
- package build/import/CLI smoke: VERIFY:
- Ruff format/check: VERIFY:
- full CPU pytest+coverage: VERIFY:
- no-network/no-CUDA assertion or equivalent test: VERIFY:
- docs/config/schema examples: VERIFY:
- secret/large-artifact scan: VERIFY:
- GitHub CI and CodeQL check names: VERIFY:
- read-only before/after GPU process snapshot proving no objective allocation:
  VERIFY:

## Security/resource constraints

Treat current host as shared. Routine repo-local environment setup is coding
agent work. Never use GPU, download model weights, modify global Conda/OpenCode,
system drivers/CUDA, other processes, ports, services, firewall/VPN or sudo
system state. Preserve unrelated working-tree files. Do not print credentials or
private provider configuration.

## Deferred human adjudication

- Decision: `NONE`
- Do not create a CRITICAL entry merely because modernization choices require
  judgment.
- If strategic reconnaissance exposes a genuinely material dilemma satisfying all
  five `CRITICAL.md` conditions, strategic must decide it before activation and
  replace this section with exact `APPEND CRIT-NNNN` bytes. Coding may not invent
  the entry.

## GitHub publication and report

Create exactly one new objective branch/PR from verified remote base. Push all
non-report work and exact activated order/active transcript. Inspect/fix in-scope
CI. Capture implementation SHA, then publish exactly one report-only final commit
with literal implementation SHA and `Report publication commit: SELF`; push and
verify parent/path/bytes/current PR head. Never merge. Report exact tests, CI,
coverage, docs, dependencies, files, safety evidence, skips/failures/limitations.
