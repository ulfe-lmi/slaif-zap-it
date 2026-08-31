# OAP Work Order 022-c — preserve logical visualization ID with safe path

## Objective

Finish Objective 022 in existing PR #78. Resolve the 022-b report's naming
interpretation without allowing request-controlled text to determine artifact
paths: service visualization members remain fixed ordinal names, while the
validated configured visualization ID is preserved as bounded logical metadata
in JSON and ZIP manifests. Also restore the independent 32-rule YAML structural
limit that was unintentionally coupled to the new 256-question operator
workload limit.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and strategic adjudication

- PR #78 remains open on `oap/022-a-canonical-clip-multiprompt`.
- Current remote/report head is
  `464456edde622456d8fbc420d15a8bc0345d51fe`; its sole path is
  `oap/reports/022-b-report.md` and its parent is implementation
  `d3e5cb29768c964f378ede462182c6808ead6b78`.
- Every report-head check is green.
- 022-b live proof succeeded with HTTP 200 and a valid labelled PNG under the
  service-safe member `visualization/stream-0001.png`. The configured logical
  ID was `final-labelled-ripe-tomatoes`, but that ID is absent from the
  manifest.
- The human requirements explicitly say user-controlled text cannot affect
  artifact paths and prompts/labels must never be filenames. A visualization
  ID is request-controlled text even after syntax validation. Therefore do not
  rename the ZIP member from `visualization/stream-0001.png`, and do not follow
  the 022-b report suggestion to derive a member name from the configured ID.
- Interpret the prior order's “final `final-labelled-ripe-tomatoes` labelled
  PNG” criterion as requiring an unambiguous manifest association between the
  configured logical stream ID and the fixed safe member, not an ID-derived
  filesystem/ZIP path.
- The live service is enabled, active and ready as PID 697088 with
  `NRestarts=0`, private listener `10.8.132.76:17891`, and only assigned
  physical GPU0 UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`.
- Preserved 022-b evidence is under mode-0700
  `/dev/shm/slaif-zap-it-022b.LIzzBL`. Do not overwrite or delete it before
  strategic review.

Refresh GitHub, host, GPU/process, service/listener, `/dev/shm`, environment
mode/digest, exact fixture hash, and `CRITICAL.md` before mutation. Stop only on
an authority/safety contradiction.

## Required implementation

### 1. Logical ID metadata with invariant safe names

- Extend the internal raw-artifact representation and public artifact
  descriptor schema so each final visualization artifact can report the exact
  validated configured ID as a bounded optional `visualization_id` (or an
  equivalently explicit field name).
- For the exact config, the descriptor must associate:
  `name="visualization/stream-0001.png"` with
  `visualization_id="final-labelled-ripe-tomatoes"`.
- Identity masks and non-visualization debug artifacts must omit/null that
  field according to one documented schema convention. Do not invent IDs for
  them.
- Preserve the same logical ID in any visualization omission/ledger record so
  clients can identify a selected stream even when a budget omits its payload.
- JSON descriptors and ZIP `manifest.json` must have exact parity. The actual
  ZIP member remains the fixed ordinal `name` and hashes/sizes remain over the
  exact delivered bytes.
- Validate schema bounds consistently with the existing visualization-ID
  validator. Never place this ID, prompt text, class label, question, or answer
  into a filesystem path, ZIP member, temp name, log message, or metric label.
- Multiple streams must deterministically map logical IDs to stable ordinal
  member names without collisions or request-to-request state leakage.

### 2. Keep definition count and execution count separate

- Restore the existing hostile-YAML maximum of 32 BLIP3 rule definitions per
  request. Rename the internal constant if useful so it is unambiguously a
  rule-definition count rather than the operator total-question workload cap.
- Keep the new immutable operator `blip3_max_questions` default/range 1..256
  exactly as implemented. It counts planned candidate-question executions,
  not configured rule definitions.
- Capabilities, schema, API/config/runbook/datasheet documentation must describe
  these two independent units and stages. Do not expand any uploaded-rule,
  prompt, object, response, timeout, or artifact ceiling in this round.
- Add tests proving 32 rule definitions parse, 33 fail as request config, the
  default 256 planned canonical candidate-questions remain admitted, and over
  the operator question cap remains typed `resource_limit`.

### 3. Tests and documentation

Add deterministic tests proving:

- service-safe single and multiple visualization streams retain fixed ordinal
  artifact names while reporting their configured logical IDs;
- changing an ID changes only metadata, never the artifact member/path pattern;
- unsafe IDs remain rejected;
- duplicate/path-like/prompt-like values cannot collide with or escape fixed
  names;
- JSON/ZIP manifest parity, schema validation, omission-ledger propagation,
  artifact hashes/sizes, deterministic repeat output, and backwards-compatible
  trusted non-service behavior;
- the exact tomato fixture's logical ID maps to the fixed stream name under the
  fake/API path;
- rule-definition and planned-question limits are independent as above.

Synchronize all affected maintained docs and generated capability/OpenAPI
descriptions. Explicitly state that `visualization_id` is logical metadata and
never a path. Correct any 022-b prose that conflated 256 questions/request with
256 uploaded BLIP3 rules.

## Verification and live proof

Run the focused visualization/envelope/schema/API/Objective-022/runtime/YAML
tests, then the full CPU coverage suite; Ruff format/check; compileall;
documentation checker; diff checks; wheel/sdist build/audit/scan/Twine/member
comparison; isolated installed-wheel JSON/ZIP smokes; and all required CI on
the exact implementation head.

Only after implementation CI is fully green:

1. Reconcile all live safety facts again.
2. Perform exactly one controlled restart of `zap-it-lan.service`; do not alter
   its unit/environment/key/network/port/deadline/model/cache/budgets. Wait for
   cold readiness with a stable new PID and zero restarts.
3. In a new mode-0700 tmpfs evidence directory, use the exact committed tomato
   image and exact 97-prompt fixture YAML. Verify their hashes/counts.
4. Submit exactly one authenticated verbosity-3 ZIP request. Read the key only
   into process memory, never print/copy/commit/argv it, and unset immediately.
   No retry or YAML mutation is authorized.
5. Require HTTP 200, safe ZIP structure, fixed member
   `visualization/stream-0001.png`, descriptor
   `visualization_id=final-labelled-ripe-tomatoes`, exact hash/size parity,
   prompt counts 32/15/15/20/15 total 97, five semantic score classes, route
   target `ripe_tomato`, final labels/bounds, stage counts, and labelled PNG.
6. Extract a mode-0600 review copy with a fixed operator-chosen filename in the
   tmpfs evidence directory; this review filename is not a service artifact
   path. Preserve ZIP/PNG for strategic inspection.
7. Report stage counts and bounded human visual observations again, correcting
   numeric wording: distinguish inclusive bbox width/height, budget omissions
   from selection omissions, and manifest facts from visual judgment.
8. Recheck private auth/docs boundaries, listener, PID/restarts, assigned GPU,
   `/dev/shm`, unchanged environment digest, and empty service workspace. Leave
   the newest private keyed service running.

If the one live request fails, report exact sanitized evidence and stop; do not
retry, restart again, clamp candidates, change the exact YAML, or raise another
limit.

## Non-goals and safety

- No new dependency, model/revision/precision/residency, prompt/routing/
  verification/geometry/renderer semantics, response budget, deadline, object
  limit, service exposure, credential, host driver/CUDA, firewall/VPN, or
  unrelated process change.
- No prompt/label/question/answer/request-derived artifact paths.
- No persistent request data; use RAM/tmpfs only.
- No new PR, merge, auto-merge, release, or public bind. Coding never merges.

## Acceptance and publication

Success requires complete CPU/package/CI proof, independent rule/question
limits, fixed service artifact names plus explicit logical ID metadata in both
JSON and ZIP/omission contracts, one exact HTTP-200 live ZIP with the mapping
and labelled PNG, honest visual/count evidence, and a ready newest private
service.

Commit one 022-c implementation commit on the existing branch/PR. After all
implementation-head checks are green, perform the one live proof. Then publish
immutable `oap/reports/022-c-report.md` as a report-only SELF commit whose sole
path is that report and parent is the implementation head. Push it, wait until
every report-head check is green, verify remote head/parent/one-path shape, send
exactly `OK` on the response FIFO, make no later mutation, and exit. The report
must include changed files, migration/compatibility notes, commands/results,
SHAs/check URLs, exact live counts/hashes/paths, visual findings, strongest
reason not to merge and its answer, and all safety confirmations.
