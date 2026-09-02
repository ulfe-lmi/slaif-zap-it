# OAP Coding-Agent Report — 026-a

## Work order

- Identifier: `026-a-fix-responses-warning-sanitization`
- Objective: fix character-spaced warning strings in the Responses public projection
- Repository: `ulfe-lmi/slaif-zap-it`
- PR mode: new numeric objective / new PR

## Status

FAILED

## Executive summary

The ordered one-line sanitizer correction and focused CPU/fake regressions are
complete and all implementation-head CI/CodeQL checks pass. The authorized
live Responses proof returned HTTP 503 after readiness, so the live acceptance
criterion could not be established. In accordance with the order, the service
was rolled back to merged `origin/main` and left healthy and ready. No retry was
made and no raw request or response content was retained.

## Authoritative GitHub state

- PR: [#90](https://github.com/ulfe-lmi/slaif-zap-it/pull/90), `OPEN`
- Title: `Objective 026: fix Responses warning sanitization`
- Base: `main` at `90c4b4923e4924dcffed185a0bf54ffeea5f7eb4`
- Starting SHA: `90c4b4923e4924dcffed185a0bf54ffeea5f7eb4`
- Branch: `oap/026-a-fix-responses-warning-sanitization`
- Implementation head SHA: `074179841fca59bb8468d4faa89ee3cd78e921b0`
- Report publication commit: SELF
- New PR: YES; amended existing PR: NO; coding merge: NO

The implementation commit is the only non-report commit and is based directly
on `origin/main`. The report-only commit is created as its sole child.

## Changes/files

Implementation commit `0741798…` changes only the ordered product/test paths
plus the exact active/order transcript:

- `src/service/responses.py`: changed the `_bounded_warning` generator join
  separator from one ordinary space to the empty string. The preserved policy
  remains `str(value)`, code point boundary `ord(character) >= 32`, one-space
  replacement for lower code points, and final `[:256]` truncation.
- `tests/test_objective_024.py`: added the focused sanitizer, projection-path,
  and parameterized HTTP regressions.
- `oap/active`: activated `026-a`.
- `oap/orders/026-a-fix-responses-warning-sanitization.md`: exact immutable
  active order transcript.

Before:

```python
text = " ".join(character if ord(character) >= 32 else " " for character in text)
```

After:

```python
text = "".join(character if ord(character) >= 32 else " " for character in text)
```

The corrected helper remains used by both `sam2.resource_warnings` and the
top-level `warnings` projection paths. No warning origin, order, list bound,
schema, serializer, endpoint, native completion, model, inference, dependency,
or documentation path was changed.

## Acceptance evidence

1. **Sanitizer policy — PASSED (CPU).**
   `test_bounded_warning_preserves_printable_controls_and_limit` proves
   printable text and `str(value)` preservation, newline/tab/NUL/`0x1f`
   replacement by one ordinary space without inserted separator spaces, and
   exact first-256-character truncation.
2. **Both projection warning lists — PASSED (CPU).**
   `test_public_projection_sanitizes_top_level_and_sam2_resource_warnings`
   proves exact sanitized entries in top-level `warnings` and
   `sam2.resource_warnings`, with bounded metadata construction.
3. **HTTP warning regression — PASSED (fake engine).**
   The two cases of
   `test_responses_warning_projection_preserves_complete_config_warning` prove
   complete list-entry equality for the debug and diagnostic-artifact warning
   strings and reject their character-spaced versions.
4. **Responses preservation — PASSED (CPU/fake).** Existing Objective 024/025
   tests remain green for no-tool metadata, the typed image-generation tool and
   deterministic PNG, projection/schema/bounds/errors, and private
   `/v1/completions` behavior.
5. **Live corrected behavior — FAILED.** After the corrected service reached
   readiness 200, the one authorized bounded authenticated Responses request
   returned HTTP 503. Therefore the two live warning equalities and live
   no-tool metadata were not claimed. The request was not retried.
6. **Rollback safety — PASSED.** The required fallback to merged `origin/main`
   completed at `90c4b492…`; its service is healthy and ready with no listener
   abandonment.

## Verification

- `.venv/bin/pytest -q tests/test_objective_024.py`: **PASSED** — 44 passed;
  the four added cases are the two direct tests plus two parameterized HTTP
  cases named above.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  **PASSED** — 975 passed, 1 explicit GPU-marker skip, 82.87% total coverage
  against the maintained 64% gate. The skip is not counted as a pass.
- `.venv/bin/ruff format --check .`: **PASSED** — 162 files formatted.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules scripts tests`: **PASSED**.
- `.venv/bin/python scripts/check_documentation.py`: **PASSED** — 28 current
  documents; documents were unchanged.
- `.venv/bin/python -m build --wheel --sdist`: **PASSED** — existing
  setuptools license metadata deprecation warnings only.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  **PASSED** — wheel/sdist member safety verification.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz
  --baseline .secrets.baseline`: **PASSED** — zero archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: **PASSED** — seven reviewed baseline findings.
- sdist extraction followed by a wheel rebuild and
  `verify_release_artifacts.py --compare-wheels`:
  **PASSED** — no member differences (`[]`).
- `scan_release_artifacts.py` on the rebuilt wheel:
  **PASSED** — zero findings.
- `.venv/bin/python -m twine check dist/*` and the rebuilt wheel:
  **PASSED**.
- `git diff --check origin/main...HEAD`: **PASSED**.

## CI/checks

All seven required implementation-head checks passed on
`074179841fca59bb8468d4faa89ee3cd78e921b0`:

- [static (format, lint, build)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785386/job/100439264729): **PASSED**
- [release (artifact audit)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785386/job/100439264532): **PASSED**
- [tests (py3.10)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785386/job/100439264723): **PASSED**
- [tests (py3.11)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785386/job/100439264811): **PASSED**
- [tests (py3.12)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785386/job/100439264727): **PASSED**
- [Analyze (python)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33687785397/job/100439264187): **PASSED**
- [CodeQL](https://github.com/ulfe-lmi/slaif-zap-it/runs/100439465268): **PASSED**

Final SELF checks are intentionally inspected after publication. Their
post-publication result is not edited into this immutable report.

## GPU/service/resource evidence

- Authorized physical GPU only: index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`.
- Corrected implementation service: PID `852078`, exact listener
  `10.8.132.76:17891`, launch visibility `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`; application mapping is logical `cuda:0` and the
  expected UUID was pinned. Health was 200 and readiness reached 200 before
  the failed live request. The assigned-card process snapshot was the service
  alone at 1578 MiB.
- Live result: the one bounded authenticated Responses request returned HTTP
  503; no raw image, YAML, bearer, response body, prompt, or model output was
  retained. Native `/v1/completions` live smoke was `NOT RUN` after the failed
  required Responses proof; CPU/fake preservation coverage remained PASSED.
- Required rollback service: merged-main PID `853653`, started 2026-09-03
  00:05:19, exact listener `10.8.132.76:17891`, health/readiness 200/200,
  unauthenticated Responses 401. Its final assigned-card snapshot reported
  only PID `853653` at 10084 MiB (14017 MiB free). No unassigned GPU or
  unrelated process was touched.
- `/dev/shm` is a 12 GiB tmpfs with 9.7 GiB free; the scoped service workspace
  remained mode 700. No request data was persisted.

## Documentation/provenance

No documentation or dependency change was necessary: the documented warning
contract already described the intended unsplit text. The active/order
transcript is included in the implementation commit; prior OAP orders,
reports, and the critical register were not modified.

## Deferred human adjudication

- Critical register action: **NONE**
- The order explicitly specifies `Decision: NONE`; `CRITICAL.md` was read and
  no register bytes changed.

## Safety/scope confirmations

Only Objective 026-a was implemented. No merge, auto-merge, force-push, release,
tag, package publication, gateway change, network/firewall/VPN change, device
reconfiguration, credential change, model-cache change, unrelated service
mutation, or unassigned-GPU operation occurred. The one implementation restart
and mandatory main rollback used only the repository launcher and existing
operator environment.

## Limitations/blockers

The implementation is not live-qualified because its authorized Responses
request returned HTTP 503. The fallback merged-main service is healthy and
ready, but it does not run the corrected sanitizer. The live failure class was
not inferred from raw response content and no retry was performed. The PR
remains open and must not be treated as accepted or merge-ready from this
report.

## Factual strategic follow-up

The strongest reason not to merge autonomously is the failed live acceptance
proof: an HTTP 503 means the corrected service behavior was not demonstrated on
the assigned private-LAN runtime. The answer is to leave the safe merged-main
service running and require strategic review of the live 503 and a later
explicit qualification before acceptance. CPU evidence establishes the exact
one-line correction and regression coverage but cannot substitute for the
ordered live proof.
