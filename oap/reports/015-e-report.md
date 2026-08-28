# OAP Coding-Agent Report — 015-e

## Work order

- Identifier/order/objective/PR mode: `015-e` — reconcile the static tokenizer
  journal record; amend the existing Objective-015 PR.

## Status

COMPLETE

## Executive summary

Corrected one factual contradiction in the immutable 015-d report without
changing product code, tests, documentation, dependencies, configuration,
service state, model state, GPU state, or request behavior. The actual new-boot
journal contained one fixed tokenizer vocabulary notice. It was allowed by
015-d because it was static and contained no path, secret, environment name, or
request-controlled material. The 015-d report's statement that the scan found
zero fixed tokenizer records is superseded only by this report.

The previously established sanitized-journal findings remain true, and the
015-d code/live acceptance remains complete. No differently scoped scan was
rerun and no new sanitizer count was invented.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/71`
- PR state: OPEN, non-draft, MERGEABLE, `mergeStateStatus=CLEAN` at the
  implementation head; no merge or auto-merge performed.
- Base: `main` at `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Branch: `oap/015-a-request-local-sam2-configuration`.
- Starting report-only head SHA:
  `bb0b55a44b917ce0a10b8780855f2470667460c5`.
- Implementation head SHA:
  `04824300c31614511590e35819bebccb236df761`.
- Report publication commit: SELF.
- New PR: no. Amended existing PR: yes. Coding merge/auto-merge: NO.

## Changes/files

Implementation/control commit
`04824300c31614511590e35819bebccb236df761` contains exactly:

- `oap/active`: exact active selector `015-e`.
- `oap/orders/015-e-reconcile-static-tokenizer-journal-record.md`: the exact
  immutable active-order transcript.

The report publication commit changes only
`oap/reports/015-e-report.md`. No prior report was edited.

## Acceptance evidence

1. **Static tokenizer reconciliation — PASSED.** One fixed tokenizer
   vocabulary notice was present in the already observed new-boot journal. The
   notice was the allowed model-vocabulary message, not an API credential;
   “tokens” is ordinary vocabulary terminology here.
2. **Allowed-record basis — PASSED.** The notice was static and contained no
   path, secret, environment-variable name, request/image/YAML material, user
   label or prompt, cache/checkpoint/repository location, or customer content.
3. **Erroneous statement superseded narrowly — PASSED.** This report
   supersedes only 015-d acceptance item 7's claim of “zero fixed tokenizer
   records.” It does not rewrite or alter the immutable 015-d report.
4. **Preserved sanitized result — PASSED.** The exact TIMM warning, TIMM
   filename, `FutureWarning`, path-bearing records, auth/key material, request
   content, errors, and tracebacks remained absent in the already observed
   new-boot record. The 015-d code/live acceptance remains complete.
5. **Evidence integrity — PASSED.** No new sanitizer count was created by
   rerunning a differently scoped scan; this report reconciles the existing
   boot record only.

## Verification

- `git show --format=fuller --name-status --stat --oneline 04824300c31614511590e35819bebccb236df761`:
  PASSED — exactly `oap/active` and the 015-e order changed.
- `git diff --check HEAD^ HEAD`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `.venv/bin/pytest -q`: NOT RUN — explicitly unnecessary because no product
  or test bytes changed in this report-reconciliation order.
- `.venv/bin/python -m build --wheel --sdist`: NOT RUN — explicitly unnecessary
  because no product, dependency, or packaging bytes changed.
- `systemctl --user show zap-it-lan.service -p MainPID -p NRestarts -p ActiveState -p SubState -p UnitFileState -p ExecMainStatus`:
  PASSED — MainPID `449821`, `NRestarts=0`, enabled, active/running, exit
  status 0.
- `ss -H -ltnp 'sport = :17891'` and `ps -p 449821 ...`: PASSED — exactly one
  listener at `10.8.132.76:17891`, owned by PID `449821`.
- `curl ... /healthz` and `curl ... /readyz`: PASSED — both HTTP 200.
- `nvidia-smi` assigned-GPU and compute-process queries: PASSED — only the
  assigned GPU0 process was present.
- `find /dev/shm/slaif-zap-it -mindepth 1 -maxdepth 2 ...`: PASSED — no request
  workspace entries.
- Operator environment mode/digest check: PASSED — mode `0600`, unchanged
  digest `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- `gh pr checks 71` at implementation head: PASSED — all seven current-head
  checks completed successfully.

## CI/checks

All required implementation-head checks passed on
`04824300c31614511590e35819bebccb236df761`:

- `Analyze (python)`: PASSED / GitHub `pass`.
- `CodeQL`: PASSED / GitHub `pass`.
- `release (artifact audit)`: PASSED / GitHub `pass`.
- `static (format, lint, build)`: PASSED / GitHub `pass`.
- `tests (py3.10)`: PASSED / GitHub `pass`.
- `tests (py3.11)`: PASSED / GitHub `pass`.
- `tests (py3.12)`: PASSED / GitHub `pass`.

## GPU/service/resource evidence

- Assigned physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB. The process-visible device remains logical
  `cuda:0`.
- Final service state: enabled and active/running, PID `449821`,
  `NRestarts=0`, exactly one listener at `10.8.132.76:17891`; the sole compute
  process on the assigned GPU was PID `449821`.
- Assigned-GPU snapshot: 10107 MiB used and 14017 MiB free; process-reported
  use 10084 MiB.
- `/dev/shm/slaif-zap-it` remained mode `0700` and empty of request-workspace
  entries.
- No restart, completion request, inference, request mutation, key operation,
  unassigned-GPU operation, unrelated process/service operation, or network
  change occurred in 015-e. No key value was read or printed.

## Documentation/provenance

No product documentation or dependency update was needed. The report is an
append-only factual correction based on the already observed new-boot record.
The active selector and exact order transcript were committed before this
report, and the existing PR was amended rather than creating a new PR.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Same branch and PR #71 only; no new PR, merge, release, tag, or auto-merge.
- The non-report implementation/control commit contains only the exact active
  selector and immutable 015-e order.
- The final report-only commit will contain only this report and will have
  implementation head `04824300c31614511590e35819bebccb236df761` as its parent.
- No source, test, documentation, dependency, configuration, service, model,
  GPU, authentication, network, artifact, request, or prior-report bytes were
  changed.

## Limitations/blockers

None for the ordered scope. Product tests and builds were not rerun because
the order explicitly requires a report reconciliation only; normal GitHub
CI passed on the implementation/control head.

## Factual strategic follow-up

None. PR #71 is ready for strategic review; coding has not merged or selected a
subsequent order.
