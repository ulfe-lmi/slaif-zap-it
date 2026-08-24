# OAP Work Order — 008-b — Baseline the immutable report false positive

## Objective

Amend only Objective-008 PR #64. Resolve the report-head release-audit failure
caused by one reviewed false-positive finding in immutable
`oap/reports/008-a-report.md`, preserve all product/live evidence unchanged,
and publish a new immutable `008-b` report only after both implementation and
report heads are fully green.

This is transcript/release-governance remediation. Do not rerun GPU phases or
change product code, model behavior, documentation claims, the 008-a report, or
the accepted live evidence.

## Reconciled state

- Mode: `AMEND_EXISTING_PR`; numeric objective/round `008 / 008-b`.
- Repository/default base: `ulfe-lmi/slaif-zap-it`, `main` at starting base
  `bdc9aad62a813d7830b4b6920de03fb106f3f886`.
- Sole PR: #64, title `Objective 008: qualify RTX 3090 all-resident BLIP3`,
  branch `oap/008-a-rtx3090-all-resident-qualification`, open/non-draft.
- Current remote report head:
  `6c4420057bfb682e6a2bdda87efefafa0497af74`; its first parent is exact
  implementation `927279f6803e53fb466badb0df3a364acf1f1b14`, and it changes
  only `oap/reports/008-a-report.md`.
- All seven checks passed at implementation head `927279f`.
- At report head `6c44200`, static, Analyze, CodeQL and Python 3.10/3.11/3.12
  passed, but `release (artifact audit)` failed. The tracked-tree scan reported
  exactly `additions=1 removals=0`.
- Independent reproduction identifies the sole addition as a `Secret Keyword`
  finding at immutable report line 173. That line contains only a literal
  secret-scanner command and its truthful status; it contains no credential,
  token, key value or secret material. The finding is a reviewed false positive.
- The existing register contains exactly five reviewed findings. No existing
  finding may be removed, rehashed, reclassified or moved to hide this result.
- PR #64 is the only open PR. Local/remote branch and report bytes/topology are
  clean and equal.
- Live cleanup remains satisfactory: assigned RTX 3090 at 15 MiB used with no
  compute rows, port 17891 free, no ZAP-IT process, and the mode-0700 shared
  memory root empty.

## Required remediation

1. Preserve `oap/reports/008-a-report.md` byte-for-byte. Do not amend/rebase/
   force-push the existing commits.
2. Use the repository-pinned secret scanner's normal baseline update mechanism
   to add exactly the one reviewed `Secret Keyword` result for
   `oap/reports/008-a-report.md` line 173 to `.secrets.baseline`.
3. Verify structurally that:
   - the prior five `(path, detector, hashed finding)` tuples are unchanged;
   - exactly one tuple is added, on the expected immutable report path and
     detector type;
   - no result is removed;
   - plugin/filter configuration is unchanged;
   - only normal scanner-generated timestamp/line metadata changes in addition
     to the one result.
4. Do not suppress the detector globally, weaken exact tracked-tree equality,
   add broad allowlists, alter release CI, or exclude OAP reports from scanning.
5. Commit the exact 008-b active selector/order plus the reviewed one-entry
   baseline update as the only non-report changes in this round.
6. Avoid reproducing the scanner-triggering command spelling in the 008-b order
   or report. Evidence may call it the "tracked-tree secret-baseline scan" and
   must use the release helper's default baseline path for commands it records.

## Verification and acceptance

- The tracked-tree release helper using its default baseline path must report
  exactly six known findings with no additions/removals.
- Run focused release-candidate/scanner tests, documentation checks, Ruff,
  compile/diff checks, package build, wheel/sdist verification, archive scans
  using the default baseline path, and the canonical CPU suite. No GPU/service
  test is needed because product/runtime bytes must be unchanged this round.
- Push the baseline/transcript implementation commit and require all seven
  current PR-head checks to succeed. A rerun of the failed old head is not a
  substitute for a new commit/check set.
- Publish only `oap/reports/008-b-report.md` in the final SELF child. That report
  must not copy the trigger text. It must state the exact reviewed path/line/type,
  prove the prior-five-plus-one set, list verification statuses, confirm 008-a
  immutability, and record literal implementation SHA plus `SELF`.
- Require all seven report-head checks successful and none pending, failed or
  missing before signaling.
- Prior orders/reports and `CRITICAL.md` remain byte-identical. Critical
  register action is `NONE`.

## Scope and non-goals

Expected round diff before the report is exactly `.secrets.baseline`,
`oap/active`, and the new `oap/orders/008-b-*.md`. No source, tests, workflows,
docs, dependency files, 008-a artifacts or live-evidence content may change.

No GPU allocation, service start, model/cache access, fixture use, listener,
release/tag/upload, merge/auto-merge, new PR, Objective 009, history rewrite or
external mutation is authorized.

## Deferred human adjudication

- Decision: `NONE`

This is a verified local false-positive registration with exact mechanical
scope, not an unresolved security-policy dilemma.

## Publication/report contract

- Amend only PR #64 and its existing branch/title/base.
- Push all non-report bytes first and record the literal implementation SHA.
- Publish one final report-only child changing exactly
  `oap/reports/008-b-report.md`; verify its first parent and remote bytes.
- Explicitly answer the strongest reason not to accept: adding a scanner
  finding to the baseline can conceal a real secret. Answer with the exact
  content assessment, immutable source line, one-entry structural diff,
  unchanged prior findings/configuration and green tracked-tree/archive CI.
- Coding never merges or starts another objective. Send exact FIFO `OK` only
  after the final remote checks and topology are verified.
