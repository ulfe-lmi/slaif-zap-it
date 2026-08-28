# OAP Coding-Agent Report — 015-c

## Work order

- Identifier/order/objective: `015-c` — prove non-finite rejection and activate
  the already-committed SAM2 `dtype` denial policy.
- PR mode: amend Objective-015 PR #71; same numeric objective and branch.

## Status

PARTIAL

## Executive summary

Added truthful CPU/API coverage for unquoted YAML floating-point `.nan`, `.inf`
and `-.inf` across all seven public SAM2 `number` fields. Each case proves that
PyYAML constructs a non-finite Python `float`, then proves sanitized
`invalid_config` rejection and no readiness/engine work. Quoted numeric-looking
strings remain separate string-type cases.

The already-committed `dtype -> unsafe_config` policy was activated with the one
authorized restart of `zap-it-lan.service`. The live `mask_generator.dtype:
float16` request returned HTTP 400 `unsafe_config`, with no model or inference
work. The service remains enabled, active, ready and on the assigned private-LAN
listener.

The required request/key/config journal scan passed, but a full sanitized-journal
claim is not made: one dependency `FutureWarning` contains a repository path and
one fixed tokenizer warning contains the word `tokens`. No bearer, raw YAML,
image, request filename or request-derived content was found. This leaves the
round PARTIAL under the host-path logging law; no second restart was authorized.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/71`
- PR state: OPEN, non-draft, MERGEABLE; base `main` at
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Branch: `oap/015-a-request-local-sam2-configuration`.
- Starting report-only head SHA: `36c8fe0561a064f11343a5ba9fe141739b784d9b`.
- Implementation head SHA: `1d00de1faa8cb1d84ed1e51b1c38abb2b046d333`.
- Report publication commit: SELF.
- New PR: no. Amended existing PR: yes. Coding merge/auto-merge: NO.

## Changes/files

- `tests/test_sam2_configuration.py`: separated quoted-string and actual YAML
  non-finite cases; added parser and HTTP coverage for 21 field/scalar pairs.
- `oap/active`: exact active selector `015-c`.
- `oap/orders/015-c-prove-nonfinite-and-activate-dtype-policy.md`: committed
  unchanged as the exact active order transcript.
- No production, renderer, documentation, systemd, shell, dependency, model,
  network, authentication or device-policy change was made.
- Prior reports, including `oap/reports/015-b-report.md`, were not edited.

## Acceptance evidence

1. **Actual non-finite YAML values — PASSED.** For every public SAM2 number
   field (`pred_iou_thresh`, `stability_score_thresh`,
   `stability_score_offset`, `mask_threshold`, `box_nms_thresh`,
   `crop_nms_thresh`, `crop_overlap_ratio`), each unquoted `.nan`, `.inf` and
   `-.inf` scalar was preconditioned as `type(value) is float` and
   `math.isfinite(value) is False`. Parser and HTTP tests require
   `invalid_config` and the public field name in the sanitized message.
2. **Quoted look-alikes — PASSED.** Quoted `NaN`, `.inf` and `-.inf` values
   were separately proven to remain strings and to exercise strict type
   rejection.
3. **Live dtype policy — PASSED.** After restart, one bounded authenticated
   completion request containing only the affected `dtype: float16` control
   returned HTTP 400 `unsafe_config`; the sanitized message named `dtype` and
   did not contain `float16`.
4. **No model/inference work — PASSED.** Across the live dtype probe, model
   initialization, successful completion, residency-transition and inference
   duration counters remained unchanged; active inference was zero. The final
   metrics showed one `unsafe_config` request and zero inference-duration calls.
5. **Auth/readiness — PASSED.** Authenticated readiness and metrics returned
   200; missing and wrong completion keys returned 401.
6. **015-b record correction — RECORDED HERE.** The 015-b non-finite claim was
   over-broad because quoted strings exercised the type branch; 015-c supplies
   actual float NaN/infinity evidence. The 015-b “no production policy
   changed/no restart needed” statement was incorrect because `dtype`
   classification changed, and 015-c activated it. The 015-b report remains
   immutable.

## Verification

- `.venv/bin/pytest -q tests/test_sam2_configuration.py`: PASSED — 222 passed.
- `.venv/bin/pytest -q tests/test_sam2_configuration.py tests/test_segmenter_sam2.py tests/test_live_runtime.py tests/test_live_service_units.py tests/test_service_units.py tests/test_service_api.py tests/test_core_config.py tests/test_core_engine.py`: PASSED — 373 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 646 passed, 1 intentional GPU skip; 79.57% total coverage.
- `.venv/bin/ruff format --check .`: PASSED — 144 files already formatted.
- `.venv/bin/ruff check .`: PASSED after the new parser test was given a unique name.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — no unexpected archive findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exact seven reviewed findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `git diff --check`: PASSED.
- Initial corrected-live harness attempt: FAILED before the authorized request
  because its local helper result was unpacked incorrectly; it performed no
  inference and no restart. The corrected bounded harness PASSED.
- Full journal sanitizer: FAILED — request/key/raw-config scan had no matches,
  but the dependency warning retained one repository path and the tokenizer
  warning retained one fixed token-related term. No request-derived material
  was present.

## CI/checks

All required checks on implementation SHA
`1d00de1faa8cb1d84ed1e51b1c38abb2b046d333` were present and successful:

- `static (format, lint, build)`: SUCCESS.
- `tests (py3.10)`: SUCCESS.
- `tests (py3.11)`: SUCCESS.
- `tests (py3.12)`: SUCCESS.
- `release (artifact audit)`: SUCCESS.
- `Analyze (python)`: SUCCESS.
- `CodeQL`: SUCCESS.

The final report-only SELF head was checked after publication; the same seven
required check names were present and green, with the remote PR head equal to
the SELF publication commit.

## GPU/service/resource evidence

- Exactly assigned physical GPU 0: UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24576 MiB; application-visible
  device is logical `cuda:0`. No unassigned device was touched.
- Pre-restart service PID was 426972. Exactly one authorized
  `systemctl --user restart zap-it-lan.service` produced stable PID 443516;
  the bounded readiness observation recorded 97 HTTP 503 samples during cold
  loading and HTTP 200 at 200 seconds.
- Final unit state: enabled/active/running, `NRestarts=0`; exactly one listener
  at `10.8.132.76:17891`; only PID 443516 was present on the assigned GPU.
- Final assigned-GPU sample: 10107 MiB used, 14017 MiB free; process-reported
  use 10084 MiB. These are observations, not capacity claims.
- Operator environment remained mode 0600 with unchanged SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- `/dev/shm/slaif-zap-it` remained mode 0700 with zero request-workspace files
  and zero request-workspace bytes; reported shared-memory availability was
  12008685568 bytes.
- Authenticated `/healthz`, `/readyz` and `/metrics` were each HTTP 200 after
  the probe. No accepted SAM2 inference was run.

## Documentation/provenance

The SAM2 configuration contract and existing service policy documentation were
unchanged because this round only corrected test evidence and activated an
already-committed denial policy. The implementation commit and exact active
order were pushed before report publication. No model weights, raw image/YAML,
credentials, customer data or private cache contents entered Git, the OAP
report or the journal scan output.

## Deferred human adjudication

- Critical register action: NONE.

## Safety/scope confirmations

- Same branch and PR #71 only; no new PR, merge, release, tag or auto-merge.
- No unassigned GPU, unrelated process/service, firewall, route, VPN, driver,
  port configuration, credential or persistent request data was touched.
- The private-LAN service was left running on its existing listener as ordered.

## Limitations/blockers

The dependency-generated FutureWarning containing a repository path means the
full journal-sanitization requirement is not cleanly satisfied. The single
authorized restart has already been consumed, so no additional activation or
service mutation was performed. The live dtype denial, auth behavior, model
counter invariants and request cleanup all passed.

## Factual strategic follow-up

Strategic review should decide whether the dependency warning requires a future
bounded logging-sanitization order. No production change is proposed or made in
015-c.
