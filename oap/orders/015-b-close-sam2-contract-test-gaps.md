# OAP Work Order — 015-b — Close SAM2 contract test and documentation gaps

## Objective

Amend Objective-015 PR #71 with the missing decisive CPU/API contract tests for
request-local SAM2 configuration, correct the inaccurate supported-scalar count,
and make the 015-b report explicitly supersede the over-broad test claims in the
immutable 015-a report. Preserve the already successful implementation and live
service unless a new test exposes a real defect; fix only such in-scope defects.

This is a same-PR continuation. Do not create another PR, merge, or begin raw
SAM2 visualization or aerial-solar qualification.

## Verified starting state and review decision

- Remote `main` remains
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- Existing PR #71, `Objective 015: request-local SAM2 configuration`, is open,
  mergeable, based on `main`, and headed by branch
  `oap/015-a-request-local-sam2-configuration` at report-only SELF commit
  `01760808c0f4a5549a313fb2b422e447da7ce674`. Its parent is the literal 015-a
  implementation SHA `27aa21c39752dad6603df458b61141efd807fa04`, and the SELF
  commit changes only `oap/reports/015-a-report.md`.
- All CI, release-audit, Analyze and CodeQL checks on current PR head
  `01760808c0f4a5549a313fb2b422e447da7ce674` are successful.
- The 015-a implementation is functionally promising and its bounded live
  evidence passed: one stable service PID; A/B/A effective configurations and
  raw counts `8, 7, 8`; exact prompt/prediction estimates; structured 400/413
  failures before inference; one model initialization; and a decisive crop-0
  versus crop-1 raw-count change `25` versus `62` on the authorized ignored
  fixture.
- Strategic review of the actual diff found that `tests/test_sam2_configuration.py`
  contains only 16 tests and does not substantiate several 015-a report claims:
  its lifecycle test is A/B rather than A/B/A; it does not count a model loader,
  prove three generator identities, or make crop layers return different fake
  proposal sets; it does not cover every field boundary/source/operator cap;
  it exercises the success manifest only at L0 rather than L0-L3; and it does
  not prove raw-versus-post-remap count semantics or the structured API
  `resource_limit` envelope. These are merge-gate gaps despite green CI.
- Public wording in `docs/API.md` says the effective/source mappings contain
  “all 14 safe generator scalars plus `use_m2m`”. The implementation exposes 14
  total safe scalars, including `use_m2m` (the user's 13 minimum scalars plus
  that additional pinned safe scalar). Correct all variants of this off-by-one
  wording.
- Live `zap-it-lan.service` is enabled, active, ready, `NRestarts=0`, with one
  listener at `10.8.132.76:17891`, PID `426972`, and the sole compute process on
  assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`. Preserve it. No restart is needed
  for tests/docs-only work.

## Required changes

### 1. Exact constructor and lifecycle contract

Extend focused CPU tests so they explicitly require the exact 14-field tuple,
not merely iteration over the implementation constant:

```text
points_per_side
points_per_batch
pred_iou_thresh
stability_score_thresh
stability_score_offset
mask_threshold
box_nms_thresh
crop_n_layers
crop_nms_thresh
crop_overlap_ratio
crop_n_points_downscale_factor
min_mask_region_area
use_m2m
multimask_output
```

Lock `profile` and `debug` as service-only, and `point_grids`, `output_mode`,
model/path/device/dtype/cache controls and arbitrary kwargs as excluded/fixed.
Use a factory spy to prove exact forwarding and fixed `point_grids=None`,
`output_mode="binary_mask"`.

Add a true counted A/B/A lifecycle test. It must:

1. create/load exactly one model object;
2. invoke settings A (8/crop-0), B (32/crop-1), then A again;
3. construct three distinct generator objects around the exact same model
   identity;
4. prove no generator is written into the resident holder at any point;
5. prove the final A receives A values, not B values;
6. count zero model rebuild/reload, cache/download, `.to()`, `.half()` or dtype
   conversion activity during the three request calls; and
7. make the deterministic fake generator return proposal sets that differ for
   crop layer 0 versus 1, so the test covers behavior as well as kwargs.

The pure unit may use spies/fakes and must not import or download heavy model
weights in public CI. Separately run a read-only `.venv-gpu` signature probe
against the installed pinned `SAM2AutomaticMaskGenerator` and report that every
public scalar exists while `point_grids`, `output_mode` and variadic kwargs stay
fixed/excluded. This probe loads no model and performs no inference.

### 2. Complete strict validation, profile and admission matrix

Use compact parameterization, without duplicative hundreds of cases, to cover:

- valid lower and upper boundaries for every integer/number field and both
  values for both booleans;
- below/above intrinsic range failures for every numeric field;
- boolean-as-integer/number, integer-as-boolean, numeric strings, explicit
  null, NaN and positive/negative infinity as applicable;
- unknown profile/key and representative unsafe model/checkpoint/path/device/
  dtype/cache/point-grid/output-mode controls with the correct distinction
  between `invalid_config`, `unsupported_field`, and `unsafe_config`;
- deepest-layer zero-point rejection;
- exact `fast`, `balanced`, `quality` partial overrides and all-field
  explicit > profile > default source resolution, including an explicit value
  equal to the inherited value;
- exact prompt formula over representative crop/downscale combinations and the
  one-versus-three multimask prediction multiplier;
- equality-at-cap acceptance and above-cap `resource_limit` rejection for
  points per side, points per batch, crop layers, min-mask area, total prompts
  and total predictions;
- each new service-settings environment variable, startup type/range failure,
  and the rule that operator caps cannot exceed intrinsic maxima; and
- deterministic 80%-threshold warnings and no silent clamping/substitution.

Test the public API failure path, not only the validator exception: malformed
intrinsic requests must return sanitized `invalid_config` HTTP 400, capacity
requests must return sanitized non-retryable `resource_limit` HTTP 413, and a
gate/engine spy must prove rejected requests do not acquire inference or call the
engine. No error body may echo YAML, paths, credentials or host/GPU internals.

### 3. Capabilities and manifest contract at every verbosity

Strengthen capabilities tests to require two byte-equivalent deterministic
authenticated responses, exact field/type/range/default/profile/operator-cap
content, source precedence and formulas, and the fixed-control exclusions.
Prove missing and wrong credentials return 401; no readiness provider, gate,
engine/model loader or mutable request state is consulted; no secret/path/GPU
UUID/process value leaks; and the explicit response schema is present in
OpenAPI when docs are enabled. Preserve the live private-LAN policy that docs
and OpenAPI are disabled there.

Parameterize successful completion tests over verbosity 0, 1, 2 and 3. For each
level require a complete typed `service.sam2` with all 14 effective/source
entries, normalized requested values, selected profile, exact estimates, raw
count, nonnegative three-decimal timing and warnings while preserving the
existing monotonic fields. Add a controlled core/API fake whose raw result
contains an empty mask so `actual_candidate_count` is demonstrably larger than
L3 `candidate_counts.sam2_candidates`; filtering/CLIP/BLIP3 must not change the
raw count. Require JSON and ZIP manifest equivalence for the same deterministic
outcome, response/artifact limits unchanged, and timing excluded only from byte
determinism claims.

### 4. Documentation and factual correction

Correct API/config/testing/datasheet wording to say **14 total safe generator
scalars, including `use_m2m`**, wherever a count is stated. Keep the explicit
field list authoritative. Do not weaken the service contract.

The 015-b report must explicitly state that its expanded tests supply evidence
that the immutable 015-a report claimed too broadly. Do not edit or delete the
015-a report. Update `TESTING.md` only so its assertions match tests that now
exist.

If the expanded tests expose an in-scope implementation defect, fix it narrowly,
document it, and rerun all affected verification. Do not refactor functioning
code merely to satisfy line coverage.

## Verification and CI

Run and report:

- focused SAM2 configuration/segmenter/live-runtime/service/API/core tests;
- the canonical CPU suite with coverage;
- Ruff format/check, compileall, documentation checker and `git diff --check`;
- systemd and shell syntax only if affected;
- wheel/sdist build, artifact audit, tracked-tree and built-artifact secret
  scans, and `twine check`;
- the read-only pinned GPU-environment constructor signature probe described
  above; and
- every required CI/CodeQL check on the final implementation head and then the
  final report-only SELF head.

Public CI remains CPU/offline and must not import/download models. Green CI is
necessary but the new assertions themselves must match this order.

## Live/service handling

Do not repeat GPU inference merely to restate 015-a's already decisive A/B/A and
crop-count evidence. Before reporting, verify read-only that the service still
has PID `426972`, `NRestarts=0`, one listener, ready state, the sole assigned-GPU
process, unchanged mode-0600 environment digest and empty request workspace.
Never print or read a key into evidence.

If and only if a test-driven production-code fix changes live behavior, one
additional controlled restart and the minimal affected live probe are
authorized after CPU/static checks. Disclose the reason and preserve the exact
GPU/private-LAN/key constraints. Tests/docs-only changes require no restart.

## Non-goals and safety

- no new PR, merge, release, tag or next numeric objective;
- no raw candidate contact sheets, candidate IDs, union/overlap/uncovered
  visualization, aerial-solar fixture/polygons, or semantic quality claim;
- no model/revision/checkpoint/config/device/dtype/cache/residency/network/auth/
  artifact policy change;
- no extra process/worker/concurrency, public bind, key disclosure/rotation,
  firewall/VPN/route/driver mutation or persistent request data; and
- no rewrite of activated orders or prior reports.

The strongest reason not to accept PR #71 remains hidden request-state coupling
or admission behavior that green happy-path tests miss. Answer it with the exact
three-generator/one-model A/B/A proof, complete validation and cap matrices,
pre-engine structured-error tests, all-verbosity/raw-count manifest tests, the
already successful stable-PID live evidence, and current-head CI.

## Publication and report contract

Amend exact branch `oap/015-a-request-local-sam2-configuration` and PR #71. Push
all implementation/test/doc state plus exact `oap/active` and this immutable
order, record a literal 40-hex implementation SHA, then create only
`oap/reports/015-b-report.md` as the final report-only SELF child. Verify its
parent, one-path topology, remote bytes and all current-head checks before the
FIFO response. Coding never merges.

## Deferred human adjudication

- Decision: NONE
