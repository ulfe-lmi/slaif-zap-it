# OAP Work Order — 015-a — Request-local SAM2 configuration

## Objective

Fix the HTTP contract so the pinned, resident SAM2 model can serve a fresh,
strictly validated `SAM2AutomaticMaskGenerator` configuration for every request.
Keep weights resident, never retain or mutate a request-configured generator,
never reload weights when generator scalars change, enforce documented operator
capacity limits before inference, expose authenticated capabilities, and report
complete SAM2 configuration provenance and execution facts in every response.

This is one new Objective-015 branch and PR. It establishes the configuration
and resource foundation for later raw-candidate visualization and aerial-solar
quality qualification; do not implement those later artifact/accuracy scopes in
this PR.

## Verified starting state

- Remote `main` is `1c6e42c28e3a4c29fff4c16be8311176ba07621a`, merge of
  Objective-014 PR #70. Post-merge CI run `33203847094` and CodeQL run
  `33203847118` are successful. GitHub has no open PR.
- Create branch `oap/015-a-request-local-sam2-configuration` from exact remote
  main and exactly one PR titled `Objective 015: request-local SAM2
  configuration` against `main`.
- The coding checkout is clean on the prior Objective-014 report head. Preserve
  the atomically published 015 order/active files, fetch, and branch from remote
  main; do not amend or replay PR #70.
- The live runtime pins the approved SAM2 model
  `facebook/sam2-hiera-large` at revision
  `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`, FP16, operator-selected logical
  `cuda:0`, local-files-only. `default_resident_loader()` currently stores one
  configured automatic mask generator at points 8/8, thresholds 0.5/0.5 and
  crop layer 0. `live_engine_callable()` rejects all non-debug request generator
  fields as `unsupported_field`.
- The installed pinned `SAM2AutomaticMaskGenerator` constructor supports these
  safe scalars: `points_per_side`, `points_per_batch`, `pred_iou_thresh`,
  `stability_score_thresh`, `stability_score_offset`, `mask_threshold`,
  `box_nms_thresh`, `crop_n_layers`, `crop_nms_thresh`,
  `crop_overlap_ratio`, `crop_n_points_downscale_factor`,
  `min_mask_region_area`, `use_m2m`, and `multimask_output`. It also accepts
  `point_grids`, `output_mode`, and arbitrary kwargs; those are not safe public
  request controls here.
- Upstream builds one point grid per layer with
  `int(points_per_side / downscale_factor**layer)` points per side. Crop layer
  `i` is used by `4**i` crops. The exact estimated prompt count is therefore
  the sum over layers `0..crop_n_layers` of
  `4**i * int(points_per_side / downscale_factor**i)**2`.
- The service has no capabilities endpoint and no `resource_limit` error. L3
  exposes a general SAM2 timing and post-remap candidate count; it does not
  expose the requested/effective/source configuration, raw generator candidate
  count, prompt estimate or resource warnings at every verbosity.
- Documentation incorrectly says request parameters are forwarded while the
  live runtime rejects them.
- Live service `zap-it-lan.service` is enabled, active and ready on exact
  `10.8.132.76:17891`, MainPID `416545`, `NRestarts=0`, with one listener and
  an empty request workspace. The mode-0600 environment digest is
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
  never read into evidence or disclose either key.
- Host `hinton2`; assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The service is the only
  compute process. `/dev/shm` is an approximately 12-GiB tmpfs with negligible
  use.

## Required architecture

### 1. Resident model, request-local generator

1. Refactor the SAM2 module through explicit, testable seams so the live
   resident holder contains the pinned model object, not a permanently
   configured `SAM2AutomaticMaskGenerator`. Preserve the trusted legacy CLI
   initialization behavior and its existing callers.
2. For every live request, construct exactly one fresh automatic mask generator
   around the already-resident model using that request's resolved effective
   scalar mapping. Generator predictor/image/request state must not be written
   back into the registry, model holder, `CoreConfig`, another request, or any
   module global.
3. Do not call a model builder, checkpoint loader, Hugging Face download/cache
   lookup, `.to()` residency move, or weight conversion because generator
   values changed. Keep the existing offline pinned identity, revision, dtype,
   cache and physical/logical device controls server-owned.
4. The request-local generator must continue using fixed `output_mode =
   "binary_mask"`, `point_grids = None`, and only the explicit constructor
   kwargs allowlisted below. Never forward arbitrary mappings or `**kwargs` from
   YAML.
5. Preserve the all-resident and sequential-registry abstractions, explicit
   model control lifecycle, one process/worker/request, CUDA autocast behavior,
   core mask remapping, candidate ordering and legacy batch path. Holder device
   inspection/movement/readiness must still recognize the model-only SAM2 state.

### 2. Safe scalar contract and strict types

Accept only `profile`, the 14 safe generator scalars listed below, the additional
safe pinned scalar `use_m2m`, and existing service debug flag `debug` under
`mask_generator`. Unknown generator keys return `unsupported_field`; the existing
hostile scan must still return `unsafe_config` for model/checkpoint/path/device/
cache/network/credential controls before any model work.

Strictly reject booleans where integers/numbers are required, integers where a
strict boolean is required, numeric strings, NaN/infinity, null explicit values,
and incompatible values. Invalid types or intrinsic ranges return
`invalid_config` with a bounded sanitized message naming only the field and
public constraint.

Intrinsic public ranges:

| Field | Strict type | Intrinsic range |
|---|---|---|
| `points_per_side` | integer | 1..1024 |
| `points_per_batch` | integer | 1..1024 |
| `pred_iou_thresh` | number | 0.0..1.0 |
| `stability_score_thresh` | number | 0.0..1.0 |
| `stability_score_offset` | number | 0.0..10.0 |
| `mask_threshold` | number | -32.0..32.0 |
| `box_nms_thresh` | number | 0.0..1.0 |
| `crop_n_layers` | integer | 0..8 |
| `crop_nms_thresh` | number | 0.0..1.0 |
| `crop_overlap_ratio` | number | 0.0..1.0 |
| `crop_n_points_downscale_factor` | integer | 1..32 |
| `min_mask_region_area` | integer | 0..64,000,000 |
| `use_m2m` | boolean | `true|false` |
| `multimask_output` | boolean | `true|false` |
| `debug` | boolean | `true|false` |

Require at least one point per side in every configured crop layer:
`int(points_per_side / downscale_factor**crop_n_layers) >= 1`; violation is
`invalid_config`. `points_per_side: null`, `point_grids`, `output_mode`,
constructor kwargs not explicitly listed, request model/revision/dtype/device,
and any artifact destination remain unsupported or unsafe as applicable. Never
silently clamp, coerce, ignore or replace a supplied scalar.

### 3. Defaults, profiles and source resolution

Keep the no-profile/no-explicit service behavior compatible with the current
live generator. Define these server defaults exactly:

```text
points_per_side = 8
points_per_batch = 8
pred_iou_thresh = 0.5
stability_score_thresh = 0.5
stability_score_offset = 1.0
mask_threshold = 0.0
box_nms_thresh = 0.7
crop_n_layers = 0
crop_nms_thresh = 0.7
crop_overlap_ratio = 512 / 1500
crop_n_points_downscale_factor = 1
min_mask_region_area = 0
use_m2m = false
multimask_output = true
```

Expose exactly three case-sensitive profiles. A profile supplies only these
overrides; every other scalar uses the server default:

```yaml
fast:
  points_per_side: 8
  points_per_batch: 8
  pred_iou_thresh: 0.5
  stability_score_thresh: 0.5
  crop_n_layers: 0

balanced:
  points_per_side: 16
  points_per_batch: 16
  pred_iou_thresh: 0.7
  stability_score_thresh: 0.8
  crop_n_layers: 0

quality:
  points_per_side: 32
  points_per_batch: 32
  pred_iou_thresh: 0.75
  stability_score_thresh: 0.85
  crop_n_layers: 1
  crop_n_points_downscale_factor: 2
  min_mask_region_area: 50
  multimask_output: true
```

`profile` must be one of `fast|balanced|quality`; invalid type/name returns
`invalid_config`. Resolution order is explicit request scalar, then selected
profile override, then server default. Record source `explicit`, `profile`, or
`default` independently for every effective scalar. Explicit values win even
when equal to the inherited value. Do not pass `profile` or `debug` to SAM2.

### 4. Operator capacity limits and admission

Add immutable `ServiceSettings` fields, environment parsing, startup validation,
deployment example and documentation for these operator-controlled defaults:

```text
SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_SIDE=64
SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_BATCH=64
SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS=2
SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_PROMPTS=8192
SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_MASK_PREDICTIONS=24576
SLAIF_ZAP_IT_SAM2_MAX_MIN_MASK_REGION_AREA=1000000
```

All must be positive integers except crop layers, which may be zero. Operator
limits may narrow but never expand the intrinsic ranges. Fail service settings
construction/startup on invalid operator values.

After profile/explicit resolution and before readiness, queue acquisition,
generator construction or inference:

1. reject any effective value above an operator field cap as `resource_limit`;
2. calculate exact estimated prompts using the pinned grid/crop formula above;
3. calculate estimated mask predictions as prompt count times `3` when
   `multimask_output` is true, otherwise times `1`;
4. reject either estimate above its operator cap as `resource_limit`;
5. never clamp or substitute a cheaper setting.

Add stable `resource_limit` to the error taxonomy with HTTP 413 and sanitized,
non-retryable semantics. Invalid intrinsic values remain `invalid_config` 400.
Emit a deterministic resource warning when either accepted estimate is at least
80% of its corresponding operator cap; otherwise use an empty list. Warnings
must not contain image/YAML content, paths, labels or host/GPU internals.

### 5. Authenticated capabilities endpoint

Add `GET /v1/capabilities`, protected by the ordinary inference bearer. It is a
read-only static/operator-policy endpoint and does not require model readiness or
acquire the inference gate. Missing/wrong keys return 401.

Return an explicit Pydantic/OpenAPI schema containing at least:

- service schema/model identifier;
- supported generator fields and strict public types;
- intrinsic ranges and current operator maxima;
- exact server defaults;
- exact profile override mappings;
- source precedence;
- prompt and mask-prediction estimation formulas in bounded descriptive form;
- fixed controls including model identity/revision, checkpoint/config paths,
  device/GPU, dtype, cache paths, residency, artifact destinations,
  `point_grids`, `output_mode=binary_mask`, and arbitrary kwargs.

Do not disclose credentials, environment values, cache/checkpoint paths, GPU
topology, process IDs or mutable request state. Live `/docs` and
`/openapi.json` remain disabled; the endpoint itself must remain available and
authenticated on the private LAN.

### 6. Per-response SAM2 manifest contract

At every verbosity level and in both JSON and ZIP `manifest.json`, add one typed
`service.sam2` object with exactly these semantic components:

```json
{
  "requested": {"profile": "quality", "points_per_side": 32},
  "effective": {"points_per_side": 32, "points_per_batch": 32},
  "sources": {"points_per_side": "explicit", "points_per_batch": "profile"},
  "selected_profile": "quality",
  "estimated_prompt_count": 2048,
  "estimated_mask_prediction_count": 6144,
  "actual_candidate_count": 123,
  "execution_time_ms": 456.789,
  "resource_warnings": []
}
```

The abbreviated example does not omit fields in the real effective/source
mappings: both contain all 14 required scalars plus `use_m2m`. `requested`
contains only the client-supplied profile and safe generator scalars, preserving
their normalized JSON scalar values; it never echoes debug, unknown/unsafe
fields, raw YAML or server controls. `selected_profile` is null when omitted.

`actual_candidate_count` is the raw count returned by the automatic mask
generator before empty-mask removal, remapping, post-filtering, CLIP or BLIP3.
It is distinct from existing `candidate_counts.sam2_candidates` at L3.
`execution_time_ms` is the measured `stage.sam2` duration including request-local
generator construction and generation. Round it to three decimals using the
same timing policy as L3. The config mappings, sources, estimates, counts and
warnings are deterministic for equal execution results; timing is observability
and excluded from byte-determinism claims. JSON and ZIP for the same outcome
must agree exactly.

Keep schema version `zap-it.v1` because this is additive. Preserve every existing
verbosity field, artifact budget, YOLO/identity/object/renderer/BLIP3/post-filter
contract and response-size check.

## Required CPU/API tests

1. Constructor-signature contract tests lock the pinned safe scalar allowlist and
   fixed exclusions. All accepted scalars reach a request-local generator with
   exact values; no unknown kwarg is forwarded.
2. Lifecycle tests use a counted model loader and generator factory: load once,
   run A/B/A with distinct settings, construct three distinct generators around
   the exact same model identity, and prove no generator/request state remains in
   the resident holder or leaks into the later A request.
3. Prove 8 -> 32 and crop 0 -> 1 values reach the factory without model load,
   move, dtype conversion, or cache/download calls. A deterministic fake must
   return different proposal sets when crop layers differ.
4. Cover every scalar's valid boundaries, exact strict types, NaN/infinity,
   booleans-as-integers, numeric strings/nulls, unknown profile/key, unsafe
   server-controlled fields, and the zero-points-at-deep-layer incompatibility.
5. Cover explicit > profile > default source resolution for all fields, including
   explicit values equal to inherited values and exact profile definitions.
6. Cover the exact prompt formula for crop layers/downscale, multimask estimate,
   equality-at-cap acceptance, per-field/operator-cap rejection, aggregate
   estimate rejection, stable `resource_limit` 413 envelope, no silent clamp,
   and 80% resource warnings.
7. Capabilities tests prove auth, exact fields/types/ranges/defaults/profiles/
   operator caps/fixed controls, deterministic response, no readiness/gate/model
   call, no secret/path/device/process leakage, and OpenAPI schema.
8. API tests prove `service.sam2` at L0-L3, raw vs post-remap candidate-count
   semantics, execution timing, JSON/ZIP parity, response bounds and unchanged
   existing fields. Fake-engine output must be honest and typed.
9. Preserve and run existing hostile-YAML, model-control lifecycle, resident
   strategy/device, timeout/cancellation, renderer, BLIP3, post-filter, artifact,
   auth, metrics, package and legacy CLI tests.

Run and report the canonical CPU suite with coverage, focused SAM2/config/live
runtime/service/API/schema tests, Ruff format/check, compileall, documentation
checker, systemd/shell syntax where affected, wheel/sdist build, release-artifact
audit, tracked-tree and built-artifact secret scans, `twine check`, and `git diff
--check`. Public CI must remain CPU/offline and must not download models. All
required current-head CI and CodeQL checks must be present and successful.

## Bounded live private-LAN qualification

Keep the existing service enabled and active during ordinary implementation.
After the implementation head is committed and CPU/static checks pass, one
controlled restart of only `zap-it-lan.service` is authorized so the process
loads the model-only SAM2 resident holder. Before restart independently recheck
the exact assigned GPU index/UUID/PCI/name/VRAM/process ownership, unit/listener,
`/dev/shm`, free capacity, and environment-file mode/digest without reading or
reporting a key. Do not start a second model process or touch drivers, firewall,
routes, VPN or unrelated units.

After readiness returns:

1. Require missing/wrong inference keys 401; authenticated capabilities,
   readiness and metrics 200; docs/OpenAPI 404. Compare capabilities against the
   ordered defaults/profiles/operator caps without exposing the key.
2. Use only an already-authorized ignored local fixture/config, crop/resize in
   memory to a bounded input, remove CLIP/BLIP3/visualization, disable debug and
   use verbosity 0 so the qualification measures SAM2 rather than response
   artifacts. Never copy fixture bytes/YAML/prompts into Git/OAP/chat.
3. Send consecutive A/B/A requests in the same stable process. A explicitly uses
   points 8/8 and crop 0. B selects `quality` and explicitly repeats
   `points_per_side: 32`, proving explicit-over-profile source, points 32,
   batch 32 and crop 1. Require exact requested/effective/source/estimate fields,
   raw actual counts and finite positive execution times; the final A must return
   A's exact configuration/provenance and deterministic count, with no B state.
4. Separately compare crop 0 and crop 1 on the same bounded image/settings and
   require the generated proposal set to change, demonstrated by a different raw
   candidate count. If an exact count collision occurs, use a second bounded
   authorized crop; do not weaken the requirement or claim semantic accuracy.
5. Send invalid-type and intrinsic-range requests and require `invalid_config`;
   send one per-field cap and one estimated-prompt cap violation and require
   `resource_limit` 413 before inference. Prove PID/GPU counters/timing do not
   indicate execution for rejected requests.
6. Prove the model was not reloaded across A/B/A: unchanged service PID/listener,
   no additional registry/startup model load event, same sole assigned-GPU
   process, and the CPU object-identity/counting proof. Record bounded GPU peak,
   host RSS, response sizes and latency for A/B/A.
7. Require sanitized journal, empty request workspace, unchanged mode-0600
   environment and digest, `NRestarts=0`, one listener, only assigned GPU use,
   and leave the unit enabled, active and ready on `10.8.132.76:17891`.

Disclose every failed live request, timeout, readiness delay, resource rejection
or corrective action. A state leak, model reload, silent clamp, mismatched
manifest, missing capability, wrong error code, unexpected process/GPU change or
request residue is not acceptance.

## Documentation and provenance

Update README, architecture, testing, configuration, API, core, output parity,
runbook, service datasheet, deployment environment example, error/capability
schema documentation and any current snapshots. Remove the false claim that
request SAM2 values are merely forwarded or fixed/rejected. Document exact
defaults, profiles, source precedence, strict ranges, operator env caps, prompt
formula, resource errors/warnings, request-local lifecycle, manifest raw-count
semantics, timing nondeterminism, capabilities auth and fixed controls.

Do not change or weaken model identities/revisions/licenses, device selection,
network/auth, cache/offline mode, dtype, BLIP3/CLIP residency, artifact limits or
the accepted CRIT-0001 disposition.

## Non-goals

- no raw candidate IDs/contact sheets, union/overlap/uncovered visualization or
  other new artifact in this PR;
- no aerial-solar image/polygon addition, profile accuracy threshold, panel
  coverage/precision claim or model benchmark in this PR;
- no weights/model/revision/config/checkpoint/dtype/device/cache/residency
  selection by clients; no point-grid upload, output-mode choice or arbitrary
  constructor kwargs;
- no new model process/worker/concurrency, public/WAN bind, TLS/gateway,
  firewall/VPN/network mutation, key rotation/disclosure, release/tag/upload or
  persistent request data;
- no change to post-SAM2 filtering, CLIP, BLIP3, final ordering, renderer output,
  lower-level artifact semantics or legacy CLI behavior except the necessary
  compatibility-preserving SAM2 seams.

## Acceptance and report contract

Acceptance requires all requirements above: one resident model with fresh
request-local generators; exact safe scalar/profile/source contract; proactive
prompt/prediction admission and structured errors; authenticated capabilities;
complete per-response SAM2 metadata; decisive no-leak/no-reload CPU evidence;
green full CPU/current-head CI/CodeQL; and satisfactory A/B/A live private-LAN
qualification on the exact assigned RTX 3090.

The strongest reason not to accept is that a nominally request-local generator
could still mutate shared registry/model state or cause hidden model rebuilds,
making consecutive users influence one another or exhausting GPU memory. Answer
it with an immutable model-only resident holder, a new generator object per call,
no write-back, strict single request, counted object-identity lifecycle tests,
A/B/A effective-output evidence, stable PID/load/GPU facts and bounded resource
admission before construction.

Push all implementation and exact active/order bytes before reporting. Record a
literal 40-hex implementation SHA. Then create exactly
`oap/reports/015-a-report.md`, commit only that report as the final report-only
SELF child, push, verify remote parent/one-path topology and bytes, send exactly
one response FIFO `OK`, perform no later mutation, and exit. Coding never merges.

## Deferred human adjudication

- Decision: NONE
