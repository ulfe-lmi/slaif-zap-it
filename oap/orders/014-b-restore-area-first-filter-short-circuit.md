# OAP Work Order — 014-b — Restore area-first filter short-circuit

## Objective

Amend Objective-014 PR #70 to correct one strategic-review defect: the new
diagnostic evaluator reads and scans `mask["segmentation"]` before applying the
terminal `area > maxsize` decision. The pre-014 filter returned immediately for
an oversized candidate and therefore did not require or inspect segmentation.
Restore the promised area-first short circuit, preserve the legacy accepted
input domain, add the missing regression, and make the diagnostic contract
honest about dimensions that were not evaluated.

This is a narrow same-PR continuation. Do not redesign or extend diagnostics,
the renderer, BLIP3, models, service contract, or any other objective.

## Verified starting state

- Remote `main` remains `2e8c67997c2480cf66f5c87a1e19afba4c6d368f`, the
  accepted merge of Objective 013. The unique open PR is
  [#70](https://github.com/ulfe-lmi/slaif-zap-it/pull/70), titled
  `Objective 014: post-filter rejection diagnostics`, base `main`, head branch
  `oap/014-a-post-filter-rejection-diagnostics`.
- PR #70 head is report-only SELF
  `2013628383c11f6faadae1ec6b95f6374c63f2d4`; its parent is implementation
  `83247d3c09f8058f068b5bcdcf121ccd6698b33d`. Every required current-head CI
  and CodeQL check is present and successful, and the PR is mergeable, but it is
  intentionally unmerged.
- The 014-a report and live max-width qualification are accepted as factual
  evidence except for the short-circuit compatibility claim corrected here.
- In `src/postprocessing.py::_evaluate_candidate`, `mask["segmentation"]` and
  `np.where` execute before `if area_value > post_maxsize`. This contradicts the
  ordered terminal contract and changes legacy behavior for an oversized mask
  record whose segmentation is absent or otherwise must not be evaluated.
- Documentation says the evaluator is short-circuiting. Existing tests prove
  reason precedence but do not prove that later fields are untouched after an
  earlier terminal decision.
- `zap-it-lan.service` is enabled, active and ready on exact private-LAN
  `10.8.132.76:17891`, MainPID `411354`, `NRestarts=0`, with one listener. It is
  the only GPU compute process on the operator-assigned physical GPU 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`; application device remains
  logical `cuda:0`. The mode-0600 key environment must remain secret and
  unchanged.

## Required correction

1. Make `area > post_maxsize` a real first terminal branch. It must complete
   without accessing, converting, indexing, iterating, or measuring the
   candidate's segmentation value. Preserve the existing numeric area
   comparison and strict `>` rule.
2. For a `maxsize` terminal outcome, retain the exact numeric `area_px` and use
   numeric `0` for both bbox dimensions because bbox evaluation did not occur.
   This is a deliberate compatibility sentinel, not an assertion that the mask
   is empty. Document this narrowly wherever the public record semantics are
   described.
3. Only after the area branch fails may the evaluator require segmentation,
   detect an empty mask, and then measure inclusive bbox dimensions for the
   `max_w`, `max_h`, or retained outcomes. Preserve the exact precedence:

   ```text
   area > maxsize               -> maxsize, bbox dimensions 0/0 (not evaluated)
   otherwise empty segmentation -> empty_mask, dimensions 0/0
   otherwise width > max_w      -> max_w
   otherwise height > max_h     -> max_h
   otherwise                    -> retained
   ```

4. Preserve every other 014-a contract: one mutually exclusive outcome,
   aggregate invariants, input order and object identity, numeric-only records,
   256-record cap/truncation, L3-only exposure, JSON/ZIP parity, schema version,
   lower verbosity behavior, and legacy function signature/list return.
5. Do not weaken normal candidate validation or silently fabricate retained
   masks. The compatibility rule applies only after the oversized area decision
   is already terminal.

## Required tests and verification

1. Add a direct regression with an oversized mapping that has `area` but no
   `segmentation`. It must be removed without error, counted exactly once under
   `removed_by_maxsize`, and produce `area_px` plus `bbox_width_px == 0` and
   `bbox_height_px == 0`.
2. Prove true non-access, not merely absence handling: use a mapping/value whose
   segmentation access would raise, or another equally decisive sentinel, and
   require the oversized call to succeed without touching it.
3. Update the existing multiple-limit/precedence expectations for maxsize
   records to the documented unmeasured `0/0` dimensions. Retain all equality,
   empty, width, height, identity/order, bounded-record, determinism, core, API,
   OpenAPI, and JSON/ZIP tests.
4. Run the focused postprocessing/core/service tests and the complete canonical
   CPU suite with coverage. Also run Ruff format/check, compileall,
   documentation checker, package build/audit/scans/twine, and `git diff
   --check` as required by 014-a. All current-head PR CI and CodeQL checks must
   be present and successful.
5. Keep the diff limited to the correction, its direct tests, exact affected
   documentation, and immutable OAP order/active/report files. Do not rewrite
   `014-a` order or report.

## Service and resource closure

Keep the current service running throughout ordinary editing and CPU/CI work.
After the correction is committed and verified, one controlled restart of only
`zap-it-lan.service` is authorized so the live process uses the corrected PR
head. Before restart recheck exact GPU index/UUID/PCI/name/VRAM and process
ownership, listener, `/dev/shm`, unit state, and environment-file mode/digest
without reading or reporting any key.

After readiness returns, prove the new PID is stable, readiness is authenticated
and healthy, missing/wrong inference keys still return 401, exactly one listener
is bound at `10.8.132.76:17891`, only the assigned GPU is visible/used, the
request workspace is empty, and the environment mode/digest is unchanged. No
new model inference is required because the canonical live max-width diagnostic
path passed in 014-a and this correction changes only the earlier maxsize
compatibility branch. If implementation scope exceeds that branch or any prior
live invariant is disturbed, repeat the relevant bounded 014-a L3 JSON/ZIP
probe. Leave the service enabled, active and ready. Disclose every failed live
attempt and corrective action.

Never print, rotate, commit, log, or report a key. Do not touch the unassigned
GPU, drivers, firewall, VPN/routes, unrelated units, or persistent request data.

## Documentation and provenance

Correct the affected diagnostic documentation to state that maxsize is decided
before segmentation access and its rejection record carries zero bbox
dimensions because those dimensions were not evaluated. Empty masks also carry
zero dimensions for their distinct reason. Do not overclaim model accuracy or
change any renderer/BLIP3/model/hardware/license statement.

## Acceptance and report contract

Acceptance requires the actual code path—not only reason selection—to
short-circuit before segmentation for oversized candidates, decisive regression
coverage, unchanged remaining Objective-014 behavior, bounded scope, green full
CPU/current-head CI/CodeQL, and the corrected service left enabled, active and
ready on the authorized private LAN/GPU.

The strongest reason not to accept is that the fix could hide a malformed mask
by treating zero dimensions as a real measurement. Answer it by limiting the
sentinel to the already-terminal `maxsize` branch, naming it explicitly as
"not evaluated" in docs/tests, and retaining normal segmentation handling for
every non-oversized candidate.

Push the correction and exact `014-b` active/order bytes to the same PR branch.
Record a literal 40-hex new implementation SHA. Then create exactly
`oap/reports/014-b-report.md`, commit only that report as the final report-only
SELF child, push it, verify remote parent/one-path topology and bytes, send
exactly one response FIFO `OK`, perform no later mutation, and exit. Coding does
not merge.

## Deferred human adjudication

- Decision: NONE
