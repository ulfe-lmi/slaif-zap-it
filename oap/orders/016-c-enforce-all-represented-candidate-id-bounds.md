# OAP Work Order 016-c — enforce every represented-candidate ID bound

## Objective

Amend Objective-016 PR #72 to correct one independently reproduced omission in
the raw-SAM2 manifest validator: every entry of `represented_candidate_ids`,
including the first and only entry, must be one-based and less than or equal to
`raw_candidate_count`. Add direct core and Pydantic-schema regression proofs,
without changing any renderer, API, resource, model, GPU or service behavior.

## Verified starting state

- GitHub repository: `ulfe-lmi/slaif-zap-it`.
- Remote `main` is
  `8081152403657f5e737ab0b491e0b89f587209e1`.
- Continue the sole Objective-016 PR #72 on branch
  `oap/016-a-bounded-raw-sam2-visualizations`; do not create another PR.
- Current remote/report head is the immutable 016-b SELF commit
  `2f6b1364dff76cb589c2dd87fa56b88ce5f0ca19`, whose sole parent is
  implementation commit `8097166285db44020b92b7b9661688a619dd8994`.
- All seven required GitHub checks are successful on the current report head.
- The 016-b live qualification passed and the private-LAN user service is
  active/ready at the operator-assigned endpoint with stable PID `476019`,
  `NRestarts=0`, and exactly physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575` exposed as logical `cuda:0`.
- Independent review found that `validate_raw_sam2_manifest` checks the upper
  bound only through adjacent `(previous, current)` pairs. Therefore a valid
  one-candidate summary tampered from `[1]` to `[2]` while
  `raw_candidate_count == 1` is not rejected, contrary to 016-b's explicit
  invariant that every represented ID is `<= raw count`.

## PR mode and sequence

- Identifier: `016-c`.
- Amend PR #72 only; one numeric objective remains one PR.
- Commit the exact activated 016-c order and selector with the implementation.
- Do not merge, enable auto-merge, create a release/tag or select Objective 017.

## Required implementation

1. Correct the pure raw-manifest validator so the upper-bound condition applies
   independently to every represented candidate ID, including a singleton and
   the first element of a longer sequence.
2. Preserve the existing requirements that IDs are integer, one-based, unique,
   strictly increasing, ordered and compatible with gaps caused by omitted
   empty candidates.
3. Preserve the fixed sanitized `CoreError` failure contract. Do not expose the
   invalid value, request data or internal mappings in an error message.
4. Add an explicit core regression that starts from a valid one-candidate
   rendered result, tampers its sole represented ID above the raw count, and
   proves `validate_raw_sam2_manifest` rejects it.
5. Add or extend the schema seam proof so `RawVisualizationManifest` rejects the
   same singleton upper-bound inconsistency through its reused validator.
6. Retain positive proof that a valid singleton and a gapped, increasing ID set
   within the raw count are accepted. Do not incorrectly require IDs to be
   consecutive.

## Acceptance criteria

- A manifest with `raw_candidate_count == 1` and
  `represented_candidate_ids == [2]` is rejected by both the pure validator and
  the Pydantic response schema.
- A valid singleton `[1]` remains accepted.
- A valid increasing nonconsecutive sequence such as `[1, 3]` remains accepted
  when its raw/visualizable/omitted/represented/truncated arithmetic is
  internally consistent.
- All 016-a/016-b raw visualization, JSON/ZIP, fixed-name, heatmap, pagination,
  bounds, resource-admission and legacy behavior remains unchanged and green.
- The code diff is the smallest clear correction; no renderer output, public
  schema shape, capability payload, documentation claim or limits change.

## Verification and evidence

Run and report, at minimum:

1. focused raw-visualization tests including the new pure and schema cases;
2. the focused Objective-016/API/core regression set from 016-b;
3. the canonical full CPU suite with coverage;
4. formatting, lint, compile, documentation, shell/unit verification,
   `git diff --check`, build, release-artifact verification/scanning and twine;
5. all required GitHub CI and CodeQL checks on the implementation commit and
   again on the final report-only head.

The report must name the exact previously accepted-invalid singleton fixture and
show that both rejection seams now pass. Failed exploratory commands, if any,
must be reported honestly.

## Live service, GPU and resource constraints

- This is a CPU-only validator correction. Do not run GPU inference and do not
  restart, stop, reload or reconfigure `zap-it-lan.service`.
- Before and after the work, verify the current user service remains
  enabled/active/ready with the same PID, `NRestarts=0`, one private-LAN
  listener and one compute process on the exact assigned GPU UUID. Do not print
  or persist the API credential.
- Do not touch any unassigned GPU, unrelated process, driver, CUDA stack,
  firewall, VPN, network interface, system service, model/cache path or
  environment file.
- Preserve in-memory request handling, bounded artifact limits, fixed safe
  artifact names and the zero-entry request workspace.

## Documentation and provenance

No product documentation change is expected because the documented invariant is
already correct. If a code comment is changed, keep it narrowly factual. Do not
change model identity/revision, licenses, authorship, security posture or
CRIT-0001.

## Non-goals

- No new raw visualization feature, palette, layout, image transform or label
  format.
- No SAM2 parameter, profile, proposal-quality, BLIP3, CLIP, post-filter,
  geometry, YOLO or labelled-renderer work.
- No fixture/polygon work from pending Objective 017.
- No dependency, packaging, CI policy, deployment or network change.

## Publication and report contract

- Push one bounded implementation commit to PR #72.
- After implementation-head checks succeed, publish exactly
  `oap/reports/016-c-report.md` as a report-only SELF commit whose sole parent is
  the implementation commit and whose only changed path is that report.
- Wait for every required check on the report head to complete successfully.
- Report PR/base/head identities, exact commits and parentage, changed paths,
  tests/checks, negative and positive invariant evidence, service continuity,
  workspace/resource facts and strongest reason not to merge.
- Send exactly `OK` on the response FIFO only after the immutable report and
  report-head checks are verified. Coding must not merge.

## Deferred human adjudication

- Decision: NONE
