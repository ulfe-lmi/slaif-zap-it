# OAP Work Order — 015-c — Prove non-finite rejection and activate dtype policy

## Objective

Amend Objective-015 PR #71 with two narrow corrections discovered during
strategic review of 015-b: make the non-finite-number tests exercise actual YAML
floating-point NaN/infinity values rather than quoted strings, and activate/
qualify the already-committed `dtype -> unsafe_config` production policy on the
live private-LAN service. Do not broaden implementation or repeat SAM2 inference.

## Verified starting state and rejection of 015-b acceptance

- Remote `main` remains
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- PR #71 remains the unique open Objective-015 PR on branch
  `oap/015-a-request-local-sam2-configuration`, base `main`, mergeable, current
  report-only head `36c8fe0561a064f11343a5ba9fe141739b784d9b`.
- SELF `36c8fe0561a064f11343a5ba9fe141739b784d9b` changes only
  `oap/reports/015-b-report.md` and has parent 015-b implementation SHA
  `8eb9e4f16070795ae54d4fae9a7807cb6ad67660`. All CI/CodeQL checks on the SELF
  head are successful.
- The 015-b implementation added `dtype` to `_FORBIDDEN_KEYS`, 180 focused
  SAM2 test cases, full manifest/capability/admission coverage, and corrected
  scalar-count documentation. Preserve those satisfactory changes.
- The 015-b report claims actual non-finite coverage, but the parameterization
  passes Python strings `"NaN"`, `".inf"`, and `"-.inf"` through `_yaml_literal`.
  That helper quotes every string, so PyYAML produces strings and validation
  fails on type before `math.isfinite` is exercised. The claim is not evidence
  of NaN/infinity rejection.
- The 015-b report also says no production policy changed and performs no
  restart. The implementation did change live API behavior for a request-level
  `mask_generator.dtype`: old live code classifies it as `unsupported_field`;
  committed code classifies this server-owned control as `unsafe_config`. The
  015-b order explicitly required a restart/minimal probe for a test-driven
  production behavior fix. The currently running service is still prior PID
  `426972`, started before 015-b, so the new policy is not active.
- The live service is otherwise enabled, active and ready with `NRestarts=0`,
  one listener at `10.8.132.76:17891`, and PID `426972` as the sole compute
  process on assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`. Its mode-0600 environment digest
  remains `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  Never read or print a key value.

## Required correction

### 1. Exercise real YAML non-finite numbers

Refactor the strict-type/non-finite parameterization so these are separate,
truthful categories:

- quoted numeric/non-finite-looking strings remain string-type rejection tests;
- for every public SAM2 `number` field, end-to-end hostile-YAML parsing must be
  tested with unquoted YAML `.nan`, `.inf`, and `-.inf` scalars (case-insensitive
  equivalents are acceptable only if PyYAML demonstrably constructs floats);
- assert before the service call, or in a focused parser precondition, that the
  raw YAML scalar is a Python float and `math.isfinite(value)` is false, so the
  test cannot silently regress into another string test;
- require `invalid_config` HTTP/validator semantics and the public field name in
  the sanitized message for each actual non-finite scalar.

Do not weaken `_validate_sam2_scalar`; it already has the correct finite-number
guard. Do not add NaN/infinity normalization, clamping or substitution.

### 2. Correct the immutable-report record

The 015-c report must explicitly state:

- the 015-b non-finite claim was over-broad because quoted strings exercised the
  type branch, and 015-c supplies actual float NaN/infinity evidence; and
- the 015-b “no production policy changed/no restart needed” statement was
  incorrect because `dtype` classification changed, and 015-c activated it.

Do not edit either prior report.

## Verification

Run the focused SAM2 test file, the affected hostile-YAML/service tests, the
canonical CPU coverage suite, Ruff format/check, compileall, documentation
checker, build/artifact/secret/twine audit, and `git diff --check`. No docs,
systemd or shell change is expected; run their specialized checks only if such
files change. Public CI remains CPU/offline. Require every CI/CodeQL check on the
implementation head and final report-only SELF head to be present and green.

## Required minimal live activation and probe

After the implementation/test head is committed and green:

1. Recheck exact assigned GPU index/UUID/PCI/name/VRAM/process ownership,
   environment mode/digest, `/dev/shm`, unit/listener and request workspace.
2. Perform exactly one controlled restart of only user unit
   `zap-it-lan.service`; do not start a second process or touch network/system
   configuration.
3. Tolerate and report bounded cold-load 503 readiness while models reload; do
   not perform a corrective restart. Require final enabled/active/ready state,
   `NRestarts=0`, one listener and only the new stable service PID on the exact
   assigned GPU.
4. Using the existing private inference bearer without printing it, send one
   bounded request whose safe image/config harness includes only the affected
   hostile control `mask_generator.dtype: float16`. Require HTTP 400
   `unsafe_config`, a sanitized message, and no readiness/gate/inference/model
   work as demonstrated by unchanged inference/model-load counters after
   readiness. Do not log or report raw image/YAML/key bytes.
5. Require authenticated readiness/metrics 200, missing/wrong completion key
   401, empty request workspace, unchanged environment digest/mode and sanitized
   journal. Leave the service running on `10.8.132.76:17891`.

No accepted SAM2 inference is required; 015-a already supplies the decisive
A/B/A and crop-layer evidence.

## Scope and non-goals

- Same branch and PR #71; no new PR, merge, release, tag or numeric objective.
- Production code should not change beyond an unexpected defect directly
  exposed by the corrected tests. No renderer/raw-visualization/aerial-fixture
  work.
- No model/revision/device/dtype/cache/residency/artifact/network/auth/key policy
  change beyond activating the already-committed `dtype` denial.
- No extra worker/process/concurrency, unassigned GPU use, firewall/VPN/route/
  driver mutation, credential disclosure or persistent request data.

The strongest reason not to merge is that a green test named “non-finite” can
still be false evidence, while the only production security fix in 015-b is not
actually running. Acceptance requires proof of real float NaN/±infinity and the
minimal live `dtype -> unsafe_config` activation on a clean stable service.

## Publication and report contract

Amend exact PR #71 and branch `oap/015-a-request-local-sam2-configuration`.
Commit/push exact `oap/active`, this immutable order and the narrow correction;
record the literal implementation SHA. Then create only
`oap/reports/015-c-report.md` as the report-only SELF child, verify parent,
one-path topology, remote bytes and current-head checks, send the FIFO response,
and exit. Coding never merges.

## Deferred human adjudication

- Decision: NONE
