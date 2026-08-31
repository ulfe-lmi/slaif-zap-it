# OAP Coding-Agent Report — 022-a

## Work order

- Identifier/order/objective/PR mode: `022-a` / canonical CLIP multi-prompt validation and exact live proof / Objective 022 / one new PR

## Status

PARTIAL

## Executive summary

Implemented the canonical CLIP prompt contract. A safe semantic class now
accepts one indivisible scalar prompt or an ordered non-empty array of
independent prompts. Structural limits and normalized duplicate handling are
enforced before inference; the pinned tokenizer is checked in a serialized
pre-inference seam; individual prompt similarities aggregate by semantic-class
maximum with deterministic lowest-index ties; routing consumes only semantic
class scores; and L3 prompt accounting/evidence, schemas, capabilities,
OpenAPI, documentation, and CPU/fake tests are synchronized.

The deliberate authenticated negative live request passed with HTTP 400
`invalid_config` before engine inference. The single authorized exact 97-prompt
live request reached inference but returned HTTP 413 `response_too_large` during
response assembly, so the required HTTP-200 ZIP, final PNG, stage-count and
visual-inspection evidence is unavailable. Per the order, no retry, config
mutation, or second restart was performed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#78](https://github.com/ulfe-lmi/slaif-zap-it/pull/78), open, targeting `main`
- Branch: `oap/022-a-canonical-clip-multiprompt`
- Base SHA: `d341a3c4ba47b71d10d70682771b315041dcbcb8`
- Starting SHA: `d341a3c4ba47b71d10d70682771b315041dcbcb8`
- Implementation head SHA: `beb4830035696d50c1c248d850940e44e67f744e`
- Report publication commit: SELF
- New PR: yes; amended existing: no; coding merge/auto-merge: NO

## Changes/files

Implementation commit `beb4830` changes 26 paths, including:

- canonical prompt policy and typed token-validation seam in `src/core/clip_prompts.py`;
- hostile YAML normalization/error details, resident CLIP adapter/preflight, semantic-class scoring and L3 result/envelope metadata;
- capability/OpenAPI/Pydantic error and response schemas, fake-engine evidence, and Objective 022 tests;
- synchronized maintained API/config/algorithm/core/architecture/parity/runbook/datasheet/README/testing documentation;
- exact published `oap/active=022-a` and unchanged 646-line active order transcript.

No dependency, model identity/revision, threshold, renderer policy, service
unit, environment, credential, port, network, driver, cache or unrelated path
was changed.

## Acceptance evidence

1. **Canonical scalar/array normalization: PASSED.** Scalars remain one prompt
   including commas and internal newlines; arrays remain ordered independent
   items; only boundary Unicode whitespace is trimmed; cross-class equal text is
   allowed; within-class trimmed duplicates are rejected without echoing text.

2. **Structural limits and sanitized errors: PASSED.** The parser enforces 32
   classes, 64 prompts/class, 256 total prompts, 512 Unicode codepoints and
   strict string/item/container types. Error details are bounded to safe class
   identifier/index, stable reason, measured count/type, duplicate first index,
   and allowed limit.

3. **Exact tokenizer preflight: PASSED in CPU/fake proof; live negative PASSED.**
   Accepted 77-token fake input is unchanged. The deliberate live input returned
   HTTP 400 `invalid_config`, class `ripe_tomato`, zero-based prompt index 0,
   measured token count 80 and allowed limit 77; no engine call occurred and
   GPU memory was unchanged at 10,107 MiB used / 14,017 MiB free before and
   after the negative request. The request was not logged with prompt text.

4. **Independent scoring/aggregation/routing: PASSED in CPU/fake proof.** Each
   processor text item is independent with bounded `max_length=77` and
   defensive truncation after preflight; accepted IDs are checked unchanged.
   Class scores are maximum individual-prompt similarities, ties choose the
   lowest prompt index, and routing vectors contain semantic-class keys only.
   The historical `classify_single` tuple remains compatible.

5. **L3/schema/capability contract: PASSED in CPU/fake proof.** L3 exposes
   `service.clip_prompts` with exact per-class/total counts, tokenizer limit 77
   and duplicate policy `reject`; routing diagnostics carry per-class and
   overall winning prompt indices/text. JSON/ZIP metadata paths share the same
   typed result. The dynamic field advertises `string_or_array` with
   `value_types=["string", "array"]`, 64-per-class and 256-total policy.

6. **Canonical routed BLIP3 fields: PASSED.** Capabilities marks
   `question`, `trueresult`, `falseresult`, `newcategory`, and `falsecategory`
   required for the canonical routed rule; runtime validation already required
   the same five fields.

7. **Exact 97-prompt CPU/fake shape: PASSED.** The appendix structure parsed
   with counts `ripe_tomato=32`, `foliage=15`, `stem_or_vine=15`,
   `greenhouse_structure=20`, `background=15`, total 97, five semantic score
   keys and routed target `ripe_tomato`.

8. **Exact live 97-prompt request: FAILED / BLOCKED.** The verified fixture and
   exact appendix request were submitted once over the private-LAN endpoint at
   verbosity 3 with ZIP format. The service returned HTTP 413
   `response_too_large` with a 122-byte sanitized error after 16.077 seconds of
   inference. No ZIP or labelled PNG was produced, so member hashes/sizes,
   stage counts, final object bounds, and independent visual observations are
   MISSING. The order forbids retrying this inference.

9. **Final service state: PASSED.** After the one authorized restart, PID 685637
   stayed active with `NRestarts=0`; health/readiness were 200; the listener
   remained exactly `10.8.132.76:17891`; the unit remained enabled/active/ready;
   the sole compute process remained on assigned physical GPU 0. Missing and
   wrong inference credentials returned 401; authenticated capabilities and
   metrics returned 200; private `/docs` and `/openapi.json` returned 404.

## Verification

- `git fetch origin --prune`: PASSED — remote `main` matched the ordered base.
- Pre-change focused suite: PASSED — 444 passed.
- `.venv/bin/pytest -q tests/test_objective_022.py`: PASSED — 13 passed.
- Focused Objective 022/adjacent suite: PASSED — 161 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 878 passed, 1 skipped, 81.76% total coverage; the one skip was the explicit opt-in GPU test.
- `.venv/bin/ruff format --check src modules tests`: PASSED.
- `.venv/bin/ruff check src modules tests`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED.
- `git diff --check` and staged diff check: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `scripts/verify_release_artifacts.py` wheel/sdist audit: PASSED.
- `scripts/scan_release_artifacts.py --tracked-tree`: PASSED — exactly 7 reviewed findings.
- `twine check dist/*`: PASSED.
- Direct wheel and sdist-built wheel member comparison: PASSED.
- Isolated direct-wheel and sdist-built-wheel `smoke_installed_package.py`: PASSED for both JSON and ZIP.
- Pre-restart GPU/service/tmpfs/fixture reconciliation: PASSED — exact assigned card, unit, environment digest, tmpfs and fixture identity.
- One authorized `systemctl --user restart zap-it-lan.service`: PASSED — new PID 685637, zero restarts.
- Bounded cold-load readiness wait: PASSED — stable PID, readiness 200.
- Authenticated negative live request: PASSED — HTTP 400 `invalid_config` before engine inference.
- Exact accepted live request: FAILED — HTTP 413 `response_too_large`; no retry performed.
- Final service/auth/listener/GPU/tmpfs recheck: PASSED except exact-200 artifact evidence, which is MISSING.

## CI/checks

All seven required checks passed on exact implementation head
`beb4830035696d50c1c248d850940e44e67f744e` before report publication:

- `static (format, lint, build)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598678/job/99594997445)
- `tests (py3.10)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598678/job/99594997790)
- `tests (py3.11)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598678/job/99594997982)
- `tests (py3.12)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598678/job/99594997716)
- `release (artifact audit)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598678/job/99594997784)
- `Analyze (python)` — PASSED — [run](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33424598145/job/99594995168)
- `CodeQL` — PASSED — [check](https://github.com/ulfe-lmi/slaif-zap-it/runs/99595239053)

The implementation head is the current PR head and all seven checks are
successful on that exact SHA. Report-only head checks are required after this
SELF commit and will be verified before signaling.

## GPU/service/resource evidence

- Assigned physical GPU only: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; visible to the process as
  logical `cuda:0`.
- No unassigned GPU was used or changed. The only compute process after live
  qualification was service PID 685637, with 10,658 MiB reported process GPU
  memory and current device snapshot 10,681 MiB used / 13,443 MiB free.
- Service: enabled, active, ready; one listener at `10.8.132.76:17891`; PID
  685637; `NRestarts=0`; one controlled restart only.
- `/dev/shm`: 12 GiB tmpfs with about 11 GiB free at final check;
  `/dev/shm/slaif-zap-it` remained mode 0700 and empty. The temporary request
  files created outside the service workspace were removed after the failed
  attempt; no final ZIP/PNG exists.
- Environment file remained mode 0600 with unchanged SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`; its
  credential was never printed, copied, rotated, committed, or placed in a
  process argument.
- Sanitized journal showed the expected health/readiness/auth/negative-request
  records and the final 413; no raw image, YAML, prompt, answer or credential
  was included in the report or OAP artifacts.

## Documentation/provenance

Updated only affected maintained documents: `README.md`, `ARCHITECTURE.md`,
`TESTING.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/ALGORITHMS.md`,
`docs/CORE.md`, `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, and
`docs/SERVICE-DATASHEET.md`. The effective config normalizes prompt bytes before
hashing/serialization and the implementation has no new dependency.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- No merge or auto-merge was enabled.
- Exactly one PR was created for numeric Objective 022.
- Exactly one service restart was performed, only for `zap-it-lan.service`.
- No second model process, dependency, unit, environment, credential, port,
  network, firewall, driver, cache, unassigned GPU or unrelated workload was
  changed.
- No request YAML, prompt text, raw answer, image bytes, model weights,
  credential or generated ZIP/PNG was committed.
- The exact fixture remained unchanged: 358454 bytes,
  SHA-256 `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.

## Limitations/blockers

The exact live request did not complete response serialization within the
existing response/artifact budget and therefore did not provide the required
HTTP-200 ZIP, final labelled PNG, stage counts, object-bound evidence or visual
inspection. The implementation and CPU/CI proof are complete, but live
Objective 022 acceptance is not.

## Factual strategic follow-up

Strategic review is required for the exact `response_too_large` live blocker.
The service is intentionally left corrected, enabled, active and ready on the
private-LAN endpoint. A same-PR 022-b correction, if chosen by strategic, must
follow the OAP order and re-establish the required current-head CI and live
evidence gates.
