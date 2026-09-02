# OAP Coding-Agent Report — 024-b

## Work order

- Identifier: `024-b-close-input-cardinality-and-native-preflight-parity`
- Objective: close Responses input-cardinality errors and restore exactly-once
  native/Responses resource admission
- PR mode: amend existing Objective 024 PR in place
- Repository: `ulfe-lmi/slaif-zap-it`

## Status

COMPLETE

## Executive summary

Amended PR #88 with the narrow Objective 024-b correction. Responses content
lists are now inspected within the bounded JSON body before payload decoding so
missing image/configuration and duplicate image/configuration cases return their
distinct stable OpenAI-shaped errors. Empty content uses the documented
`missing_image` precedence and unsupported content retains `unsupported_field`.

The shared inference helper now accepts an explicit private resource-admission
token produced by the route-owned pre-body check. Both `/v1/completions` and
`/v1/responses` therefore perform `check_request_resources` exactly once while
preserving the existing native pre-body boundary. A direct invalid-YAML route
regression, exact cardinality matrix, success call-count tests, and first-check
failure tests were added. No vision pipeline, model, renderer, public
projection, authentication policy, or gateway code was changed.

## Authoritative GitHub state

- PR: [#88](https://github.com/ulfe-lmi/slaif-zap-it/pull/88), `OPEN`
- Title: `Objective 024: OpenAI Responses-compatible facade`
- Base: `main` at `32812032781c5d7daf54d5b7586b3c01d3270c48`
- Starting SHA: `e1d80512252262048b4409ad7b54bf20d53b3739` (024-a report-only
  SELF head)
- Implementation commit added this round:
  `639a319041cfa7f72f8fa5d645d43f062d24bcb7`
- Implementation head SHA:
  `639a319041cfa7f72f8fa5d645d43f062d24bcb7`
- Report publication commit: SELF
- New PR: NO; amended existing PR: YES; coding merge: NO

The implementation commit has first parent
`e1d80512252262048b4409ad7b54bf20d53b3739`. The 024-a report was verified
byte-for-byte unchanged. The branch was pushed to PR #88 before live
qualification. The report-only child below is the final branch head after
publication.

## Changes/files

Implementation commit `639a319…` changes six paths, with 391 insertions and 47
deletions including the exact 215-line active order transcript:

- `src/service/responses.py`: remove the premature exact-length rejection and
  add sanitized messages for the four cardinality codes.
- `src/service/app.py`: add the explicit `_ResourceAdmission` token and
  route-owned `_admit_request_resources` contract, remove the shared duplicate
  preflight and dead `remaining_budget` helper, and pass the token on both
  routes.
- `tests/test_objective_024.py`: table-driven exact cardinality/precedence
  cases, invalid-YAML typed error, one-call success checks, and first-admission
  failure/envelope checks.
- `docs/RESPONSES-FACADE.md`: document missing/duplicate/empty/unsupported
  content precedence and the unchanged two-part accepted shape.
- `oap/active`: wrapper-published active round `024-b`.
- `oap/orders/024-b-close-input-cardinality-and-native-preflight-parity.md`:
  exact strategic-authored active order transcript.

No changes were made to `oap/reports/024-a-report.md`, the vision pipeline,
model configuration, renderer/PNG encoder, `slaif-api-gateway`, credentials,
or unrelated service code.

## Acceptance evidence

- The table-driven route test covers valid file only -> HTTP 400
  `missing_image`, valid image only -> HTTP 400 `missing_config`, empty content
  -> HTTP 400 `missing_image`, duplicate images -> HTTP 400 `duplicate_image`,
  and duplicate YAML files -> HTTP 400 `duplicate_config`. Each cardinality
  response has the canonical four-field OpenAI error body, type
  `invalid_request_error`, bounded non-null `input[0].content` param, and zero
  fake-engine calls. The unsupported-part case remains HTTP 400
  `unsupported_field` with its bounded content-type param.
- The direct invalid-YAML route case returns HTTP 400 `invalid_config` with the
  canonical OpenAI error shape and zero engine calls.
- `_ERROR_MESSAGES` now gives all four cardinality codes specific bounded
  messages without request content or filenames.
- The generated `ResponsesRequest` Pydantic schema remains unchanged with
  exactly two content parts (`min_length=2`, `max_length=2`); runtime
  classification still rejects every non-one-image/one-file shape.
- `test_each_http_surface_admits_resources_exactly_once` observed one
  monkeypatched admission call after a successful native request and one after
  a successful Responses request. The explicit token prevents the shared
  helper from calling the check again.
- `test_first_resource_admission_failure_precedes_body_parsing_on_both_surfaces`
  observed the failing first check before malformed native/Responses bodies:
  native retained its 507 native envelope and Responses retained its 507
  OpenAI-shaped `server_error` envelope; both made zero engine calls.
- Existing private completion byte/parity and full service regressions remained
  green. The authenticated native live smoke returned HTTP 200, object
  `text_completion`, L2 verbosity, eight objects, and the sole expected
  `identity-mask.png` artifact.

## Verification

- `.venv/bin/pytest -q tests/test_objective_024.py`: PASSED — 39 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 968 passed, 1 explicit GPU test skipped, 82.80% total coverage
  against the 64% gate. Two pre-existing warnings were reported.
- `.venv/bin/ruff format --check .`: PASSED — 162 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 28 current
  documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl
  dist/*.tar.gz`: PASSED.
- sdist extraction, wheel rebuild and
  `.venv/bin/python scripts/verify_release_artifacts.py --compare-wheels ...`:
  PASSED — no wheel member differences.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl
  dist/*.tar.gz dist-from-sdist/*.whl --baseline .secrets.baseline`: PASSED —
  0 new archive findings.
- `.venv/bin/python -m twine check dist/*` and the corresponding
  `dist-from-sdist/*` command: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree
  --baseline .secrets.baseline`: PASSED — exactly 7 reviewed baseline findings.
- `git diff --check`: PASSED; post-live worktree remained clean before report
  creation.

## CI/checks

All seven required checks passed on implementation head
`639a319041cfa7f72f8fa5d645d43f062d24bcb7`:

- [static (format, lint, build)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945239/job/100179317853): PASSED
- [release (artifact audit)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945239/job/100179317619): PASSED
- [tests (py3.10)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945239/job/100179317846): PASSED
- [tests (py3.11)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945239/job/100179317770): PASSED
- [tests (py3.12)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945239/job/100179317904): PASSED
- [Analyze (python)](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33608945224/job/100179317541): PASSED
- [CodeQL](https://github.com/ulfe-lmi/slaif-zap-it/runs/100179569864): PASSED

At the last implementation-head inspection GitHub reported PR state `OPEN`,
`MERGEABLE`, `CLEAN`.

## GPU/service/resource evidence

- Host: `hinton2`. The exact assigned physical device was GPU index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI bus
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24576 MiB, driver `610.43.02`.
- Launch mapping was preserved as `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`; the process exposed the selected card as logical
  `cuda:0`. `nvidia-smi` showed only the assigned service process on the
  device. No unassigned device or unrelated process was touched; no second
  device was present in the live host snapshot.
- The old validated service PID 805444 was gracefully stopped once after the
  implementation push. The newest implementation was then started once and
  left running as PID 815951, started `2026-09-02 10:32:15` local time. The
  listener remained exactly `10.8.132.76:17891` with PID 815951 ownership and
  no wildcard listener. `/healthz` and `/readyz` returned HTTP 200 after the
  pinned model load; readiness initially returned the honest loading 503.
- Final GPU sample: 10595 MiB used / 13529 MiB free; the service process used
  10572 MiB. The service process RSS sample was 3,956,120 KiB. The model load
  used the same operator environment and assigned UUID; no memory/process
  action was taken on another device.
- `/dev/shm` is a 12 GiB tmpfs with 1.6 GiB used and 9.7 GiB available at the
  final snapshot. `/dev/shm/slaif-zap-it` is mode 700, owned by `janezp:users`.
  Its three retained qualification summaries are each mode 600 and 389 bytes;
  the launcher PID/log files are mode 600. The summaries contain only bounded
  hashes/counts/statuses. No request image, YAML, result, bearer or request
  body was persisted.
- Official SDK live qualification against the restarted service passed using
  the pinned `openai==3.7.0` development dependency: typed object `response`,
  status `completed`, one decodable image-generation PNG call, 2 public
  objects, 2500 projection bytes and 997 PNG bytes. The bounded summary was
  stored with SHA-256 projection
  `a0324ac28ba64e2da5c70b9c18123069ad227b549ef5b98503f422b0cab315d6` and PNG
  `1848669e4a0816187975f600327352748fa6a6aa3efc3ff8208c638e043eb527`.
- The authenticated native live smoke passed with HTTP 200, object
  `text_completion`, verbosity 2, 8 objects, one `identity-mask.png`, and an
  8176-byte response. Unauthenticated probes returned 401 for both HTTP
  surfaces. A live Responses matrix passed all seven bounded cases: the four
  cardinality codes, empty/unsupported precedence, and invalid YAML.

## Documentation/provenance

`docs/RESPONSES-FACADE.md` now states the exact pre-decode cardinality
precedence. The code keeps the official SDK development pin at `openai==3.7.0`
and does not import that SDK at runtime. The active order and exact 024-b
transcript are included in the implementation commit; the 024-a report remains
immutable.

## Deferred human adjudication

- Critical register action: NONE
- Deferred human adjudication: NONE
- The active order specifies `Decision: NONE`; `CRITICAL.md` was read and no
  register bytes changed.

## Safety/scope confirmations

Only Objective 024-b was implemented. No model, inference parameter, holder,
residency, device, CUDA/driver, firewall/network/VPN, credential, gateway,
unrelated service/process, merge, tag, release or external deployment was
changed. The live service restart was limited to the already running scoped
ZAP-IT process and left the newest corrected implementation on the same
authorized private-LAN listener and assigned GPU.

## Limitations/blockers

The live proof establishes bounded local HTTP, SDK, parser, native parity and
renderer integration; it does not establish semantic accuracy, recall,
precision, commercial model licensing, public/WAN deployment safety or final
release readiness. Coding cannot merge or accept PR #88. The final SELF CI
checks must still be inspected before signaling.

## Factual strategic follow-up

The strongest reason not to merge autonomously is the coding role’s authority
boundary: it cannot accept or merge. The strongest technical risk was the
possibility that broad runtime cardinality inspection or shared extraction
could drift from the strict OpenAPI/native contract. This round closes that
risk with explicit bounded classification, stable typed errors, unchanged
two-part schema, an explicit admission token, full CPU/fake coverage, green
multi-version CI/CodeQL, live SDK/native qualification and the exact final
service/GPU/tmpfs evidence above. Strategic should independently inspect the
PR diff, implementation-to-SELF lineage and final SELF checks before any
acceptance decision.
