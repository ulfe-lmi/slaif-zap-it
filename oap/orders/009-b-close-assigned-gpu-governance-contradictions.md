# OAP Work Order — 009-b — Close assigned-GPU governance contradictions

## Objective

Amend only Objective-009 PR #65. Correct the remaining current statements that
still universally reserve physical GPU0 or hard-code physical GPU1, synchronize
the maintained strategic bootstrap law with the operator-assigned index+UUID
model already implemented and qualified, and add a narrow regression guard.

Do not change the accepted profile-matrix harness, live evidence, runtime/model
code or any immutable prior order/report. No GPU or service rerun is required.

## Reconciled state

- Mode: `AMEND_EXISTING_PR`; numeric objective/round `009 / 009-b`.
- Sole PR: #65, `Objective 009: close memory-deferred profile evidence`, branch
  `oap/009-a-memory-deferred-profile-matrix-and-doc-closure`, base `main` at
  `b1d8c5dbc9392002ab52b3b0b744582a073ebf75`.
- Current report head:
  `f0eae1d91ef4b29903da74495139cbc7dce49f1c`; first parent is implementation
  `c54eb75d9741f67081cb93e50102194662cb5667`; the SELF commit changes only
  `oap/reports/009-a-report.md`.
- All seven implementation and report-head checks are successful. The exact
  eight-call live matrix and cleanup passed; no live evidence is disputed.
- Strategic review found current contradictions that the 009-a four-phrase
  checker did not cover:
  - root `AGENTS.md` still summarizes the service as physical-GPU1-only;
  - root `OAP-COMMUNICATION-coding-agent.md` still states physical GPU0 is
    universally protected;
  - `docs/SERVICE-DATASHEET.md` still says physical GPU0 is never used despite
    naming the assigned all-resident GPU0 in the same hardware table;
  - the maintained files under `oap/strategic-instructions/` still present the
    2026-08-23 index-1/GPU0 prohibition as permanent current law rather than a
    historical assignment superseded by explicit operator authorization.
- Historical initial orders, immutable OAP transcripts and `docs/history/`
  correctly preserve the old host facts and must not be rewritten.

## Required correction

1. Replace current universal index claims with one invariant:
   the active order names an explicit operator-assigned physical index and UUID;
   the launcher exposes exactly that card as logical `cuda:0`; every unassigned
   device and unrelated process is protected. No automatic fallback or request
   selection is allowed.
2. Correct root `AGENTS.md`, root `OAP-COMMUNICATION-coding-agent.md`, and the
   service-datasheet protected-device row. Preserve the historical maelstrom1
   GPU1 and current hinton2 GPU0 measurements as host-specific evidence.
3. Synchronize the maintained current-law files under
   `oap/strategic-instructions/`: `AGENTS.md`, `ARCHITECTURE-for-agents.md`,
   `OAP-COMMUNICATION-strategic.md`, `strategic_model_init_material.md`, and
   `INITIAL-ROADMAP.md`. The roadmap may add a post-Objective-008 correction;
   do not rewrite its historical Objective-003 facts or any
   `initial-orders/*` draft.
4. Extend the documentation/current-truth checker so it also scans the root
   coding constitution/protocol and maintained strategic current-law files,
   while excluding historical initial-order drafts and immutable transcripts.
   Add narrow forbidden patterns for the audited universal claims, with tests
   proving both obsolete samples fail and the operator-assigned wording passes.
5. Preserve the exact six-entry secret baseline, every prior order/report,
   `CRITICAL.md`, runtime/model/harness/test evidence, dependencies and
   workflows. Expected implementation paths are limited to current governance/
   datasheet/checker/tests plus exact 009-b active/order transcript.

## Verification and acceptance

- A repository-wide targeted scan outside immutable/historical material finds
  no current statement that the service is physical-GPU1-only, that physical
  GPU0 is universally unavailable, or that the launcher must always use index
  1. Host-specific historical tables may still name their actual indices.
- Documentation/current-law integrity tests pass and demonstrate the new guard
  rejects the exact stale forms without rejecting correct historical context.
- Canonical CPU suite/coverage, focused documentation tests, Ruff, compile,
  documentation integrity, package build/install/artifact scans, tracked-tree
  secret equality and all seven implementation/report-head GitHub checks pass.
- No GPU allocation, cache/model access, service/listener or fixture use occurs.
  Read-only final evidence must show the assigned card at baseline, port free
  and shared-memory root empty.
- Final SELF report is `oap/reports/009-b-report.md`, changes only that path,
  has the literal implementation SHA as first parent, does not introduce a new
  secret finding, and records all exact statuses/topology.

## Scope and non-goals

No production source/runtime behavior, matrix harness, model/revision/dtype,
residency threshold, API, geometry/panoptic, deployment, release, license,
driver/CUDA/network/systemd or unrelated process change is authorized. Do not
create another PR/objective, merge, rebase, force-push, or edit 009-a/prior
reports.

## Deferred human adjudication

- Decision: `NONE`

The human explicitly assigned the RTX 3090 physical index 0 and Objective 008
already implemented/qualified the general operator-assigned law. This round
only removes contradictory current prose and adds regression coverage.

## Publication/report contract

- Amend only PR #65 and preserve its title/base/branch.
- Push all non-report work first, record literal implementation SHA and require
  all current checks successful.
- Publish one report-only SELF child, verify parent/one-path/remote bytes, run
  the tracked-tree release helper using its default baseline path, and require
  all seven report-head checks successful before signaling.
- Explicitly answer the strongest reason not to accept: generalizing old GPU1
  law could accidentally authorize arbitrary GPU use. Answer with explicit
  active-order index+UUID authority, fail-closed single-device masking,
  protected-unassigned-device wording, narrow checker tests and zero runtime
  change.
- Coding never merges or starts another order. Send exact FIFO `OK` only after
  final remote verification.
