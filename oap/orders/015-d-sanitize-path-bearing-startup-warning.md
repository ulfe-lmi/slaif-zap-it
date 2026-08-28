# OAP Work Order — 015-d — Sanitize the path-bearing startup warning

## Objective

Complete Objective 015 by preventing the exact third-party TIMM deprecation
warning from writing an absolute installation/repository path to the live
service journal, without broadly suppressing warnings. Restart once to activate
the narrow fix, prove the new boot journal is path/secret/request clean, and
leave the private-LAN service ready. This is the final same-PR corrective round;
do not change SAM2 behavior or repeat inference.

## Verified starting state and 015-c disposition

- Remote `main` is still
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Objective-015 PR #71 is open, cleanly mergeable, based on `main`, on branch
  `oap/015-a-request-local-sam2-configuration`, at report-only SELF head
  `3b26d2454a790f501381b7b8d4c289537b18e06a`. SELF changes only
  `oap/reports/015-c-report.md` and its parent is implementation SHA
  `1d00de1faa8cb1d84ed1e51b1c38abb2b046d333`. All seven current-head CI/CodeQL
  checks are successful.
- 015-c truthfully proves unquoted YAML NaN/positive-infinity/negative-infinity
  rejection for all seven number fields and successfully activated the
  `mask_generator.dtype -> unsafe_config` policy. Preserve it.
- 015-c is honestly `PARTIAL` only because the new boot journal contains the
  Python warning record for TIMM's exact deprecation message, whose standard
  formatting includes the absolute `...site-packages/timm/models/layers/__init__.py`
  filename. It contains no request data or credential, but path-bearing startup
  logs do not satisfy the ordered sanitized-journal gate.
- The separate fixed tokenizer message saying special tokens were added is not
  itself a secret leak: “token” is a normal model-vocabulary term. Do not treat
  a generic substring search for `token` as a credential scan. It may remain if
  it contains no bearer value, authorization header, environment variable name,
  host path, request bytes or user-controlled text.
- Live unit `zap-it-lan.service` is enabled, active and ready at
  `10.8.132.76:17891`, PID `443516`, `NRestarts=0`, one listener, and the sole
  compute process on assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`. Environment mode/digest remain
  0600 and `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  Never read or print a key value.

## Required implementation

1. Install a narrowly scoped warning filter before the import chain that emits
   the warning during resident-model startup. Match all of:
   - category `FutureWarning`;
   - the exact stable prefix/message `Importing from timm.models.layers is
     deprecated`;
   - TIMM's `timm.models.layers` module/filename origin where the Python warning
     API permits it.
2. Do not use `ignore::FutureWarning`, global stderr redirection, logger
   disabling, broad TIMM/Transformers suppression, shell output discarding, or
   journal filtering. Other warnings and startup failures must remain visible.
3. Keep the filter in the live-service bootstrap/runtime boundary, not in core
   inference or uploaded-config code. Add a concise comment explaining that it
   prevents Python's warning formatter from disclosing the absolute installed
   filename for this already-reviewed deprecation only.
4. Add focused tests that prove the exact warning is suppressed before the
   relevant import and an unrelated `FutureWarning` remains observable. Tests
   must restore warning state and must not require TIMM/model imports or GPU.
5. No SAM2 defaults/profiles/validation/capabilities/manifest behavior, model
   identity/residency, authentication, network, artifact or dependency version
   changes are allowed.

## Verification

Run focused bootstrap/live-runtime tests, the SAM2 contract tests, canonical CPU
coverage suite, Ruff format/check, compileall, documentation checker,
`git diff --check`, build/artifact/secret/twine audit, and systemd/shell checks
only if those files change. Public CI remains CPU/offline. Require every
CI/CodeQL check on both implementation and report-only SELF heads.

## Live activation and journal acceptance

After green implementation checks:

1. Reverify the exact assigned GPU/UUID/PCI/name/VRAM/process, unit/listener,
   `/dev/shm`, request workspace and unchanged environment mode/digest.
2. Perform exactly one controlled restart of only `zap-it-lan.service` and
   tolerate bounded cold-load readiness 503s without corrective restart.
3. Require final enabled/active/ready state, `NRestarts=0`, one new stable PID,
   one listener and only that PID on the assigned GPU.
4. Inspect only the new boot's journal. Require absence of the exact TIMM
   deprecation warning and its absolute filename, tracebacks/errors, bearer or
   authorization material, key/environment variable names, request image/YAML,
   user labels/prompts, cache/checkpoint/repository paths, and request residue.
   Report aggregate/pass evidence, not matched sensitive bytes.
5. A fixed dependency message may remain only when it is demonstrably static
   and contains no path, secret, environment name or request-controlled text;
   do not fail merely on the generic vocabulary word “tokens”.
6. Require authenticated health/readiness/metrics 200, missing/wrong completion
   credentials 401, unchanged inference counters, and zero accepted inference.
   Leave the service running on the same private-LAN endpoint.

## Scope/non-goals

- Same branch/PR #71; no new PR, merge, release, tag or next objective.
- No request-level behavior change, model load strategy change, extra process,
  worker or GPU inference.
- No unassigned GPU, driver, firewall, route, VPN, listener, key, global
  environment or unrelated service mutation.
- No broad warning suppression and no rewrite of prior reports/orders.

The strongest reason not to accept is that suppressing a path-bearing warning
could also hide real dependency failures. Answer it with exact category/message/
origin matching, an unrelated-warning survival test, visible startup/readiness
failures, and a clean new-boot journal.

## Publication/report contract

Amend exact PR #71 and branch `oap/015-a-request-local-sam2-configuration`.
Commit/push exact active/order bytes and the narrow implementation, capture the
literal implementation SHA, then add only `oap/reports/015-d-report.md` as the
report-only SELF child. The report must explicitly close the 015-c PARTIAL
journal limitation. Verify parent/one-path/remote bytes/current checks before
FIFO response. Coding never merges.

## Deferred human adjudication

- Decision: NONE
