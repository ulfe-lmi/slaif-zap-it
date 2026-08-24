# OAP Coding-Agent Report — 009-b

## Work order

- Identifier/order/objective/PR mode: `009-b` / Objective 009 / `AMEND_EXISTING_PR`
- Sole PR: [#65](https://github.com/ulfe-lmi/slaif-zap-it/pull/65)

## Status

COMPLETE

## Executive summary

Corrected the remaining current GPU-governance contradictions in the root
coding law, maintained strategic instructions, and service datasheet. The
current-document checker now scans those maintained current-law files and
rejects the audited physical-GPU1-only, universal-GPU0, and fixed launcher-index
claims while allowing explicit operator-assigned index+UUID wording. No runtime,
model, harness, service, GPU allocation, or historical transcript was changed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/65`; `OPEN`, mergeable
- Required title: `Objective 009: close memory-deferred profile evidence`
- Base: `main` at `b1d8c5dbc9392002ab52b3b0b744582a073ebf75`
- Required branch: `oap/009-a-memory-deferred-profile-matrix-and-doc-closure`
- Starting SHA: `f0eae1d91ef4b29903da74495139cbc7dce49f1c`
- Implementation head SHA: `f1ffe6cbe1c99b78f0b871c64f8ab20715fd4335`
- Report publication commit: SELF
- New PR: no; amended existing PR: yes; coding merge: NO

## Changes/files

- Updated root `AGENTS.md` and `OAP-COMMUNICATION-coding-agent.md` with the
  explicit active-order-assigned index+UUID invariant and protected-unassigned
  device law.
- Updated `docs/SERVICE-DATASHEET.md` and the five maintained files under
  `oap/strategic-instructions/` while retaining host-specific historical GPU1
  and qualified hinton2 GPU0 facts.
- Extended `scripts/check_documentation.py` and
  `tests/test_documentation.py` with maintained-current-law coverage and narrow
  stale-claim regression tests.
- Committed the exact `009-b` active marker and order transcript.
- This report is the only file in the publication commit.

## Acceptance evidence

1. **Current governance correction — PASSED.** The maintained invariant names
   one exact active-order operator-assigned physical index and UUID, masks only
   that card as logical `cuda:0`, fails closed on mismatch, forbids automatic
   fallback/request device selection, and protects every unassigned device and
   unrelated process.
2. **Targeted current-truth scan — PASSED.** The repository-wide targeted scan
   outside historical/immutable material found no stale current service-only,
   universal-GPU0, or fixed-index-1 launcher claim. Historical order/report and
   `docs/history/` content was not rewritten.
3. **Regression guard — PASSED.** Focused tests reject obsolete samples for
   physical GPU1-only service wording, universal GPU0 prohibition, and
   `CUDA_VISIBLE_DEVICES=1`; explicit operator-assigned wording passes.
4. **Scope/safety — PASSED.** No production source, runtime/model code,
   profile harness, dependency, workflow, `CRITICAL.md`, prior order/report,
   service/listener, model cache, request fixture, or protected process changed.
5. **Strongest reason not to accept — addressed.** Generalizing old GPU1 law
   could accidentally authorize arbitrary GPU use. The correction instead
   requires exact active-order index+UUID authority, fail-closed single-device
   masking, protected unassigned devices, narrow checker tests, and zero runtime
   behavior change.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 404 passed, 1 intentional GPU skip, 77.75% total coverage.
- `.venv/bin/pytest -q tests/test_documentation.py`: `PASSED` — 4 passed.
- `.venv/bin/ruff format --check .`: `PASSED` — 136 files already formatted.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current
  documents/current-law files scanned.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- Release artifact allowlist, sdist wheel rebuild/comparison, archive secret
  scan, Twine metadata check, and `systemd-analyze verify`:
  `PASSED`.
- Installed wheel service smoke: `PASSED`.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: `PASSED` — exactly 6 reviewed findings.
- `git diff --check`: `PASSED`.

## CI/checks

All required current-head checks passed on implementation SHA
`f1ffe6cbe1c99b78f0b871c64f8ab20715fd4335`:

- `static (format, lint, build)`: `PASSED` — CI run `32768068944`.
- `tests (py3.10)`: `PASSED` — CI run `32768068944`.
- `tests (py3.11)`: `PASSED` — CI run `32768068944`.
- `tests (py3.12)`: `PASSED` — CI run `32768068944`.
- `release (artifact audit)`: `PASSED` — CI run `32768068944`.
- `Analyze (python)`: `PASSED` — CI run `32768068955`.
- `CodeQL`: `PASSED` — check run `97562263951`.

## GPU/service/resource evidence

- No GPU or service phase was run; the active order expressly required no
  rerun. Read-only baseline: physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, `24576 MiB`, driver `610.43.02`.
- `nvidia-smi` reported no compute applications. No application logical-device
  mapping was exercised this round; the maintained contract is the selected
  physical index exposed as logical `cuda:0`.
- No ZAP-IT service/uvicorn listener was present. `127.0.0.1:17891` was not
  listening, and `/dev/shm/slaif-zap-it` was empty at final inspection.
- No model cache access, CUDA initialization, request data, fixture media, or
  persistent request artifact was used.

## Documentation/provenance

Current law now distinguishes host-specific historical measurements from the
active-order selection authority. `oap/strategic-instructions/initial-orders/`,
prior OAP orders/reports, `docs/history/`, `CRITICAL.md`, and runtime/model
evidence remain unchanged. The implementation commit contains only the bounded
governance/datasheet/checker/test/transcript paths listed above.

## Deferred human adjudication

- Critical register action: NONE
- No register entry was created, appended, or altered.

## Safety/scope confirmations

- PR #65 was amended; no new PR, merge, rebase, force-push, auto-merge, or
  adjacent order was created or executed.
- The exact active ID remained `009-b`; the exact order transcript was committed
  unchanged with the implementation.
- No credentials, raw images/YAML, prompts, answers, model weights, customer
  data, or sensitive host paths entered the report.

## Limitations/blockers

This round is documentation/governance closure only. It does not requalify GPU
memory, start the service, or authorize deployment, public exposure, customer
data, or release.

## Factual strategic follow-up

PR #65 remains open at the verified report-head topology for strategic review
and acceptance. Coding has not merged it or selected another order.
